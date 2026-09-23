"""
Generate the full 24-model x 3-dataset Slurm matrix for layer sweeps.

The model roster is read from configs/model_list.yaml. The generator accepts
several simple registry shapes so the registry can remain lightweight:

    # Option A: plain list
    - llama-3.1-8b
    - _llama-3.1-8b

    # Option B: mapping with a models list
    models:
      - llama-3.1-8b
      - _llama-3.1-8b

    # Option C: mapping keyed by model name
    models:
      llama-3.1-8b: {}
      _llama-3.1-8b: {}

    # Option D: top-level mapping keyed by model name
    llama-3.1-8b: {}
    _llama-3.1-8b: {}

Each registry entry must ultimately resolve to a concrete YAML under
configs/model/<model_name>.yaml.

One generated job corresponds to one model x one dataset and runs all probes
configured in configs/experiments/sweep_layers.yaml. Thus:

    24 models x 3 datasets = 72 Slurm files.

Generated files:
    slurms/generated/layer_sweep/<model>__<dataset>.slurm

Logs:
    slurms/generated/layer_sweep/logs/

Layer-sweep outputs:
    outputs/layer_sweep/<model>/<dataset>.parquet

Every job uses --resume. This is safe for both new and partially completed
sweeps: an absent Parquet file starts normally, while existing completed
(layer, probe) pairs are skipped.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any

import yaml


DATASETS = ("cities_loc", "med_indications", "defs")
EXPECTED_MODEL_COUNT = 24


def q(value: str | Path) -> str:
    """Shell-quote one value."""
    return shlex.quote(str(value))


def safe_token(value: str) -> str:
    """Make a compact Slurm-safe token while preserving base/instruct distinction."""
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
    """Resolve one registry-list entry to a model config key."""
    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict):
        # Support concise registry entries such as:
        #   - name: llama-3.1-8b
        # or
        #   - config: llama-3.1-8b
        # or
        #   - model_name: llama-3.1-8b
        for key in ("name", "config", "model_name", "key"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Also support a one-key mapping:
        #   - llama-3.1-8b: {...}
        if len(entry) == 1:
            key = next(iter(entry))
            if isinstance(key, str) and key.strip():
                return key.strip()

    raise ValueError(
        "Could not resolve a model name from registry entry "
        f"{entry!r}. Expected a string or a small mapping with "
        "name/config/model_name/key."
    )


def load_model_registry(path: Path) -> list[str]:
    """
    Read ordered model names from configs/model_list.yaml.

    Several lightweight YAML shapes are accepted; see module docstring.
    """
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
                    f"{path}: 'models' must be a list or mapping, "
                    f"not {type(models).__name__}."
                )
        else:
            # Treat the top-level mapping itself as the registry.
            entries = list(raw.keys())

    else:
        raise ValueError(
            f"{path}: expected a YAML list or mapping; "
            f"found {type(raw).__name__}."
        )

    model_names = [
        _extract_registry_entry_name(entry)
        for entry in entries
    ]

    # Preserve registry order while rejecting duplicates.
    seen: set[str] = set()
    duplicates: list[str] = []
    ordered: list[str] = []

    for name in model_names:
        if name in seen:
            duplicates.append(name)
            continue
        seen.add(name)
        ordered.append(name)

    if duplicates:
        raise ValueError(
            f"{path}: duplicate model registry entries: "
            f"{sorted(set(duplicates))}"
        )

    if len(ordered) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_MODEL_COUNT} registered models "
            f"for the final sweep, found {len(ordered)}:\n{ordered}"
        )

    return ordered


def validate_model_configs(
    model_names: list[str],
    *,
    config_dir: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Load and validate every concrete model config named by the registry."""
    loaded: list[tuple[str, dict[str, Any]]] = []

    for model_name in model_names:
        path = config_dir / f"{model_name}.yaml"

        if not path.exists():
            raise FileNotFoundError(
                f"Registry model {model_name!r} has no config: {path}"
            )

        cfg = load_yaml(path) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"{path}: expected a YAML mapping.")

        # Concrete configs should contain these after the refactor.
        for required in ("name", "model"):
            if required not in cfg:
                raise ValueError(
                    f"{path}: missing required field {required!r}."
                )

        if str(cfg["name"]) != model_name:
            raise ValueError(
                f"{path}: config name={cfg['name']!r} does not match "
                f"registry key {model_name!r}."
            )

        loaded.append((model_name, cfg))

    return loaded


def infer_model_size_billions(
    model_name: str,
    cfg: dict[str, Any],
) -> float:
    """Infer nominal parameter scale from model/config identifiers."""
    candidates = [
        model_name,
        str(cfg.get("name", "")),
        str(cfg.get("model", "")),
    ]

    matches: list[float] = []

    for candidate in candidates:
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            candidate,
        ):
            matches.append(float(value))

    if not matches:
        raise ValueError(
            "Could not infer model size from "
            f"model_name={model_name!r}, model={cfg.get('model')!r}."
        )

    return max(matches)


def batch_size_for_size(
    size_billions: float,
    override: int | None,
) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("--batch_size must be positive.")
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
    Conservative wall times for the three-probe sweep.

    These are intentionally roomier than the old sAwMIL-only sweep because
    every layer now evaluates sAwMIL + SVM + mean-difference and checkpoints
    Parquet results.
    """
    if override is not None:
        if override <= 0:
            raise ValueError("--hours must be positive.")
        return override

    if size_billions <= 4:
        return 3
    if size_billions <= 9:
        return 5
    if size_billions <= 14:
        return 7
    if size_billions <= 32:
        return 10
    return 20


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
    experiment_config: Path,
) -> str:
    """Render one model x dataset Slurm script."""
    size_billions = infer_model_size_billions(model_name, cfg)

    model_token = safe_token(model_name)
    dataset_token = safe_token(dataset)
    job_name = f"ls-{model_token}-{dataset_token}"

    log_dir = generated_dir / "logs"
    output_parquet = (
        repo_root
        / "outputs"
        / "layer_sweep"
        / model_name
        / f"{dataset}.parquet"
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
        f"mkdir -p {q(output_parquet.parent)}",
        f"mkdir -p {q(log_dir)}",
        "",
        'if [[ -z "${HF_TOKEN:-}" ]]; then',
        '  echo "WARNING: HF_TOKEN is not set. Gated models may fail to download." >&2',
        "fi",
        "",
        'echo "============================================================"',
        'echo "Graded Stability: layer sweep"',
        f'echo "Model: {model_name}"',
        f'echo "HF model: {cfg.get("model")}"',
        f'echo "Dataset: {dataset}"',
        f'echo "Detected size: {size_billions:g}B"',
        f'echo "Batch size: {batch_size}"',
        f'echo "Memory: {memory}"',
        f'echo "Wall time: {hours}h"',
        f'echo "Output Parquet: {output_parquet}"',
        'echo "Job ID: ${SLURM_JOB_ID}"',
        'echo "Node: $(hostname)"',
        'echo "Started: $(date)"',
        'echo "============================================================"',
        "python --version",
        "nvidia-smi",
        "",
        "# --resume is intentionally unconditional:",
        "# - if no Parquet exists, the sweep starts normally;",
        "# - if one exists, completed (layer, probe) pairs are skipped.",
        "srun python -m scripts.probes.sweep_layers \\",
        f"  --config {q(experiment_config)} \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        f"  --batch_size {batch_size} \\",
        "  --device cuda \\",
        "  --resume",
        "",
        'echo "============================================================"',
        'echo "Finished: $(date)"',
        'echo "============================================================"',
        "",
    ]

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the full 24-model x 3-dataset = 72-job Slurm matrix "
            "for the clean three-probe layer sweep."
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
        help=(
            "Model registry YAML. Default: "
            "<repo_root>/configs/model_list.yaml"
        ),
    )
    parser.add_argument(
        "--experiment_config",
        type=Path,
        default=None,
        help=(
            "Layer-sweep experiment config. Default: "
            "<repo_root>/configs/experiments/sweep_layers.yaml"
        ),
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
            "Override memory for every job, e.g. 128G. "
            "Default: size-based."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help=(
            "Override wall time in hours for every job. "
            "Default: size-based."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Override batch size for every job. "
            "Default: size-based."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated Slurm files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    config_dir = repo_root / "configs" / "model"

    registry_path = (
        args.registry
        if args.registry is not None
        else repo_root / "configs" / "model_list.yaml"
    )
    experiment_config = (
        args.experiment_config
        if args.experiment_config is not None
        else repo_root
        / "configs"
        / "experiments"
        / "sweep_layers.yaml"
    )

    if not experiment_config.exists():
        raise FileNotFoundError(
            f"Layer-sweep experiment config not found: "
            f"{experiment_config}"
        )

    generated_dir = (
        repo_root
        / "slurms"
        / "generated"
        / "layer_sweep"
    )
    log_dir = generated_dir / "logs"

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

    print(
        f"Registry: {registry_path}"
    )
    print(
        f"Models:   {len(model_configs)}"
    )
    print(
        f"Datasets: {len(DATASETS)}"
    )
    print(
        f"Expected jobs: {expected_jobs}"
    )
    print()

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
                    experiment_config=experiment_config,
                ),
                encoding="utf-8",
            )
            destination.chmod(0o750)
            generated.append(destination)

    if len(generated) != expected_jobs:
        raise RuntimeError(
            f"Expected {expected_jobs} jobs but generated "
            f"{len(generated)}."
        )

    print(
        f"Generated {len(generated)} layer-sweep Slurm files "
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
