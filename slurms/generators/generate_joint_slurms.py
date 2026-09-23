#!/usr/bin/env python3

"""
Generate the full 24-model x 3-dataset Slurm matrix for Phase 3:
ordered nine-class joint credence estimation.

Each generated GPU job runs:

    python -m scripts.beliefs.score_joint

for one model x one dataset with --resume.

Phase 3:
  * generates balanced ordered nine-class joint train/cal examples in memory;
  * trains NEW joint probes at the probe-specific selected layers;
  * saves those fitted probes in outputs/joint/<model>/<dataset>.joblib;
  * streams the same P x x test pairs using the default x_then_P rendering;
  * checkpoints hidden Parquet parts and resumes from completed chunks.

Generated files:
    slurms/generated/joint/<model>__<dataset>.slurm

Logs:
    slurms/generated/joint/logs/

Required upstream input:
    outputs/pairs/<model>/<dataset>.parquet

Canonical outputs:
    outputs/joint/<model>/<dataset>.parquet
    outputs/joint/<model>/<dataset>.joblib
    outputs/joint/<model>/<dataset>.json
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any

import yaml


DATASETS = ("cities_loc", "med_indications", "defs")
PROBES = ("sawmil", "svm", "mean_difference")
EXPECTED_MODEL_COUNT = 24


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def safe_token(value: str) -> str:
    prefix = "i-" if value.startswith("_") else ""
    value = value.lstrip("_")
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    return (prefix + value)[:56] or "model"


def load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _extract_registry_entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict):
        for key in ("name", "config", "model_name", "key"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if len(entry) == 1:
            key = next(iter(entry))
            if isinstance(key, str) and key.strip():
                return key.strip()

    raise ValueError(
        "Could not resolve a model name from registry entry "
        f"{entry!r}."
    )


def load_model_registry(path: Path) -> list[str]:
    raw = load_yaml(path)

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        if "models" in raw:
            models = raw["models"]
            if isinstance(models, list):
                entries = models
            elif isinstance(models, dict):
                entries = list(models.keys())
            else:
                raise ValueError(
                    f"{path}: 'models' must be a list or mapping."
                )
        else:
            entries = list(raw.keys())
    else:
        raise ValueError(
            f"{path}: expected a YAML list or mapping."
        )

    names = [
        _extract_registry_entry_name(entry)
        for entry in entries
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            f"{path}: duplicate model registry entries."
        )

    if len(names) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_MODEL_COUNT} models, "
            f"found {len(names)}."
        )

    return names


def _coerce_layer(
    value: Any,
    *,
    model_name: str,
    probe: str,
    dataset: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid selected layer for {model_name}/{probe}/{dataset}: "
            f"{value!r}"
        )

    try:
        layer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid selected layer for {model_name}/{probe}/{dataset}: "
            f"{value!r}"
        ) from exc

    if layer < 0:
        raise ValueError(
            f"Selected layer must be nonnegative for "
            f"{model_name}/{probe}/{dataset}; got {layer}."
        )

    return layer


def validate_model_configs(
    model_names: list[str],
    *,
    config_dir: Path,
) -> list[tuple[str, dict[str, Any]]]:
    loaded: list[tuple[str, dict[str, Any]]] = []

    for model_name in model_names:
        path = (
            config_dir
            / f"{model_name}.yaml"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Registry model {model_name!r} has no config: {path}"
            )

        cfg = load_yaml(
            path
        ) or {}
        if not isinstance(
            cfg,
            dict,
        ):
            raise ValueError(
                f"{path}: expected a YAML mapping."
            )

        if str(
            cfg.get(
                "name"
            )
        ) != model_name:
            raise ValueError(
                f"{path}: config name={cfg.get('name')!r} does not match "
                f"registry key {model_name!r}."
            )

        if "model" not in cfg:
            raise ValueError(
                f"{path}: missing required field 'model'."
            )

        selected = cfg.get(
            "selected_layers"
        )
        if not isinstance(
            selected,
            dict,
        ):
            raise ValueError(
                f"{path}: missing selected_layers mapping."
            )

        for probe in PROBES:
            probe_layers = selected.get(
                probe
            )
            if not isinstance(
                probe_layers,
                dict,
            ):
                raise ValueError(
                    f"{path}: no selected_layers mapping for {probe!r}."
                )

            for dataset in DATASETS:
                if dataset not in probe_layers:
                    raise ValueError(
                        f"{path}: missing selected layer for "
                        f"{probe}/{dataset}."
                    )

                _coerce_layer(
                    probe_layers[
                        dataset
                    ],
                    model_name=model_name,
                    probe=probe,
                    dataset=dataset,
                )

        loaded.append(
            (
                model_name,
                cfg,
            )
        )

    return loaded


def infer_model_size_billions(
    model_name: str,
    cfg: dict[str, Any],
) -> float:
    candidates = [
        model_name,
        str(
            cfg.get(
                "name",
                "",
            )
        ),
        str(
            cfg.get(
                "model",
                "",
            )
        ),
    ]

    matches: list[float] = []

    for candidate in candidates:
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            candidate,
        ):
            matches.append(
                float(
                    value
                )
            )

    if not matches:
        raise ValueError(
            "Could not infer model size from "
            f"model_name={model_name!r}, model={cfg.get('model')!r}."
        )

    return max(
        matches
    )


def batch_size_for_size(
    size_billions: float,
    override: int | None,
) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError(
                "--batch_size must be positive."
            )
        return override

    if size_billions <= 9:
        return 16
    if size_billions <= 14:
        return 12
    if size_billions <= 32:
        return 8
    return 2


def memory_for_size(
    size_billions: float,
    override: str | None,
) -> str:
    if override is not None:
        return override

    if size_billions <= 9:
        return "64G"
    if size_billions <= 14:
        return "96G"
    if size_billions <= 32:
        return "128G"
    return "192G"


def hours_for_size(
    size_billions: float,
    override: int | None,
) -> int:
    """
    Phase 3 is dominated by streaming millions of joint test pairs.

    The defaults deliberately use a 24-hour checkpoint window for most models.
    Because score_joint uses --resume and persists both trained probes
    and scored parts, a timed-out job can simply be resubmitted.
    """
    if override is not None:
        if override <= 0:
            raise ValueError(
                "--hours must be positive."
            )
        return override

    if size_billions <= 4:
        return 4
    if size_billions <= 9:
        return 8
    if size_billions <= 14:
        return 8
    if size_billions <= 32:
        return 8
    return 8


def render_job(
    *,
    repo_root: Path,
    generated_dir: Path,
    model_name: str,
    cfg: dict[str, Any],
    dataset: str,
    account: str,
    partition: str,
    cpus: int,
    memory: str,
    hours: int,
    batch_size: int,
    score_chunk_size: int,
) -> str:
    size_billions = infer_model_size_billions(
        model_name,
        cfg,
    )

    selected = cfg[
        "selected_layers"
    ]
    selected_for_dataset = {
        probe: _coerce_layer(
            selected[
                probe
            ][
                dataset
            ],
            model_name=model_name,
            probe=probe,
            dataset=dataset,
        )
        for probe in PROBES
    }

    model_token = safe_token(
        model_name
    )
    dataset_token = safe_token(
        dataset
    )
    job_name = (
        f"joint-{model_token}-{dataset_token}"
    )

    log_dir = (
        generated_dir
        / "logs"
    )

    pair_path = (
        repo_root
        / "outputs"
        / "pairs"
        / model_name
        / f"{dataset}.parquet"
    )

    output_dir = (
        repo_root
        / "outputs"
        / "joint"
        / model_name
    )
    output_parquet = (
        output_dir
        / f"{dataset}.parquet"
    )
    output_bundle = (
        output_dir
        / f"{dataset}.joblib"
    )
    output_summary = (
        output_dir
        / f"{dataset}.json"
    )

    selected_summary = ", ".join(
        f"{probe}={selected_for_dataset[probe]}"
        for probe in PROBES
    )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --gpus=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={hours:02d}:00:00",
        f"#SBATCH --output={log_dir}/{job_name}-%j.out",
        f"#SBATCH --error={log_dir}/{job_name}-%j.err",
        "",
        "set -euo pipefail",
        "",
        f"PROJECT_ROOT={q(repo_root)}",
        'cd "${PROJECT_ROOT}"',
        "",
        "module load miniforge3",
        'eval "$(conda shell.bash hook)"',
        "conda activate lockean",
        "",
        "export HF_HOME=/scratch/dies_s_neu/huggingface",
        'export HF_HUB_CACHE="${HF_HOME}/hub"',
        'export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"',
        'export HF_DATASETS_CACHE="${HF_HOME}/datasets"',
        'export TRANSFORMERS_CACHE="${HF_HOME}/transformers"',
        "export TORCH_HOME=/scratch/dies_s_neu/torch",
        "export XDG_CACHE_HOME=/scratch/dies_s_neu/xdg-cache",
        "",
        'mkdir -p "${HF_HOME}"',
        'mkdir -p "${HF_HUB_CACHE}"',
        'mkdir -p "${HF_DATASETS_CACHE}"',
        'mkdir -p "${TRANSFORMERS_CACHE}"',
        'mkdir -p "${TORCH_HOME}"',
        'mkdir -p "${XDG_CACHE_HOME}"',
        f"mkdir -p {q(output_dir)}",
        f"mkdir -p {q(log_dir)}",
        "",
        'if [[ -z "${HF_TOKEN:-}" ]]; then',
        '  echo "WARNING: HF_TOKEN is not set. Gated models may fail to download." >&2',
        "fi",
        "",
        'echo "============================================================"',
        'echo "Graded Stability: Phase 3 ordered joint scoring"',
        f'echo "Model: {model_name}"',
        f'echo "HF model: {cfg.get("model")}"',
        f'echo "Dataset: {dataset}"',
        f'echo "Detected size: {size_billions:g}B"',
        f'echo "Selected layers: {selected_summary}"',
        f'echo "Batch size: {batch_size}"',
        f'echo "Score chunk size: {score_chunk_size}"',
        f'echo "Memory: {memory}"',
        f'echo "Wall time: {hours}h"',
        f'echo "Pair input: {pair_path}"',
        f'echo "Output Parquet: {output_parquet}"',
        f'echo "Probe bundle: {output_bundle}"',
        f'echo "Summary: {output_summary}"',
        'echo "Job ID: ${SLURM_JOB_ID}"',
        'echo "Node: $(hostname)"',
        'echo "Started: $(date)"',
        'echo "============================================================"',
        "python --version",
        "nvidia-smi",
        "",
        f"PAIR_PATH={q(pair_path)}",
        'if [[ ! -s "${PAIR_PATH}" ]]; then',
        '  echo "ERROR: required pair Parquet is missing or empty: ${PAIR_PATH}" >&2',
        "  exit 1",
        "fi",
        "",
        "# --resume preserves fitted joint probes and scored checkpoint parts.",
        "srun python -m scripts.beliefs.score_joint \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        f"  --batch_size {batch_size} \\",
        f"  --score_chunk_size {score_chunk_size} \\",
        "  --device cuda \\",
        "  --resume",
        "",
        'echo "============================================================"',
        'echo "Finished: $(date)"',
        'echo "============================================================"',
        "",
    ]

    return "\n".join(
        lines
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the full 72-job GPU Slurm matrix for Phase 3 "
            "ordered nine-class joint scoring."
        )
    )

    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(
            "/work/neu/p2026_0086_neu/Sam/graded_stability"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--account",
        default="p2026_0086_neu",
    )
    parser.add_argument(
        "--partition",
        default="b200-batch",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--memory",
        default=None,
        help=(
            "Override memory for every job. Default: size-based."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help=(
            "Override wall time for every job. Default: 12h for <=4B, "
            "24h otherwise. Jobs are resumable."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Override batch size for every job. Default: size-based."
        ),
    )
    parser.add_argument(
        "--score_chunk_size",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated Slurm files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cpus <= 0:
        raise ValueError(
            "--cpus must be positive."
        )
    if args.score_chunk_size <= 0:
        raise ValueError(
            "--score_chunk_size must be positive."
        )

    repo_root = args.repo_root.resolve()
    config_dir = (
        repo_root
        / "configs"
        / "model"
    )

    registry_path = (
        args.registry
        if args.registry is not None
        else repo_root / "configs" / "model_list.yaml"
    )

    generated_dir = (
        repo_root
        / "slurms"
        / "generated"
        / "joint"
    )
    log_dir = (
        generated_dir
        / "logs"
    )

    generated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_names = load_model_registry(
        registry_path
    )
    model_configs = validate_model_configs(
        model_names,
        config_dir=config_dir,
    )

    expected_jobs = (
        EXPECTED_MODEL_COUNT
        * len(DATASETS)
    )

    generated: list[Path] = []

    for model_name, cfg in model_configs:
        size_billions = infer_model_size_billions(
            model_name,
            cfg,
        )
        batch_size = batch_size_for_size(
            size_billions,
            args.batch_size,
        )
        memory = memory_for_size(
            size_billions,
            args.memory,
        )
        hours = hours_for_size(
            size_billions,
            args.hours,
        )

        for dataset in DATASETS:
            destination = (
                generated_dir
                / f"{model_name}__{dataset}.slurm"
            )

            if (
                destination.exists()
                and not args.overwrite
            ):
                raise FileExistsError(
                    f"{destination} already exists. "
                    "Pass --overwrite to replace generated Slurm files."
                )

            destination.write_text(
                render_job(
                    repo_root=repo_root,
                    generated_dir=generated_dir,
                    model_name=model_name,
                    cfg=cfg,
                    dataset=dataset,
                    account=args.account,
                    partition=args.partition,
                    cpus=args.cpus,
                    memory=memory,
                    hours=hours,
                    batch_size=batch_size,
                    score_chunk_size=args.score_chunk_size,
                ),
                encoding="utf-8",
            )
            destination.chmod(
                0o750
            )
            generated.append(
                destination
            )

    if len(generated) != expected_jobs:
        raise RuntimeError(
            f"Expected {expected_jobs} jobs but generated "
            f"{len(generated)}."
        )

    print(
        f"Generated {len(generated)} Phase-3 Slurm files "
        f"({EXPECTED_MODEL_COUNT} models x {len(DATASETS)} datasets)."
    )
    print(f"Slurms: {generated_dir}")
    print(f"Logs:   {log_dir}")
    print()
    print("Resource summary:")

    for model_name, cfg in model_configs:
        size_billions = infer_model_size_billions(
            model_name,
            cfg,
        )
        print(
            f"  {model_name:24s} "
            f"{size_billions:>5g}B | "
            f"batch={batch_size_for_size(size_billions, args.batch_size):>2d} | "
            f"mem={memory_for_size(size_billions, args.memory):>4s} | "
            f"time={hours_for_size(size_billions, args.hours):>2d}h"
        )

    print()
    print("Submit all 72 with:")
    print(
        f'  find {q(generated_dir)} -maxdepth 1 -name "*.slurm" '
        '-print0 | sort -z | xargs -0 -n1 sbatch'
    )


if __name__ == "__main__":
    main()
