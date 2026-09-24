"""
Generate the full 24-model x 3-dataset Slurm matrix for the behavioral
resilience experiment.

Each generated GPU job runs, sequentially, for one model x one dataset:

    1. python -m scripts.behavior.build_challenge_set --mode both --overwrite
    2. python -m scripts.behavior.run_challenge --resume
    3. python -m scripts.behavior.analyze_challenge --overwrite

The behavioral scorer checkpoints periodically and is resumable. If a job times
out during challenge generation, simply resubmit the same Slurm file: set
construction is deterministic, run_challenge resumes from completed statements,
and analysis runs only after challenge generation is complete.

Generated files:
    slurms/generated/behavior/<model>__<dataset>.slurm

Logs:
    slurms/generated/behavior/logs/

Required upstream inputs:
    outputs/atomic/<model>/<dataset>.parquet
    outputs/gamma/<model>/<dataset>.parquet

Canonical outputs:
    outputs/behavior/sets/<model>/<dataset>/
    outputs/behavior/challenge/<model>/<dataset>.parquet
    outputs/behavior/challenge/<model>/<dataset>.json
    outputs/behavior/analysis/<model>/<dataset>/

Default wall time:
    <= 32B: 30 minutes
    >  32B: 60 minutes

Use --minutes 30 to force every model, including 70/72B models, to use a
30-minute wall time.
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


def validate_model_configs(
    model_names: list[str],
    *,
    config_dir: Path,
) -> list[tuple[str, dict[str, Any]]]:
    loaded: list[tuple[str, dict[str, Any]]] = []

    for model_name in model_names:
        path = config_dir / f"{model_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Registry model {model_name!r} has no config: {path}"
            )

        cfg = load_yaml(path) or {}
        if not isinstance(cfg, dict):
            raise ValueError(
                f"{path}: expected a YAML mapping."
            )

        if str(cfg.get("name")) != model_name:
            raise ValueError(
                f"{path}: config name={cfg.get('name')!r} does not match "
                f"registry key {model_name!r}."
            )

        if "model" not in cfg:
            raise ValueError(
                f"{path}: missing required field 'model'."
            )

        loaded.append((model_name, cfg))

    return loaded


def infer_model_size_billions(
    model_name: str,
    cfg: dict[str, Any],
) -> float:
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

    # These mirror the size-aware settings that have worked elsewhere in
    # this repository. Behavioral scoring is much smaller than joint scoring.
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


def minutes_for_size(
    size_billions: float,
    override: int | None,
) -> int:
    """
    The 27B smoke test completed comfortably within a short interactive run.
    Use 30 minutes through 32B and give 70/72B models a 60-minute cushion.
    Jobs remain resumable if they hit the wall-time limit.
    """
    if override is not None:
        if override <= 0:
            raise ValueError("--minutes must be positive.")
        return override

    if size_billions <= 32:
        return 30
    return 60


def slurm_time(minutes: int) -> str:
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}:00"


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
    minutes: int,
    batch_size: int,
    save_every: int,
    cv_folds: int,
    cv_repeats: int,
    sequence_permutations: int,
) -> str:
    size_billions = infer_model_size_billions(
        model_name,
        cfg,
    )

    model_token = safe_token(model_name)
    dataset_token = safe_token(dataset)
    job_name = f"beh-{model_token}-{dataset_token}"

    log_dir = generated_dir / "logs"

    atomic_path = (
        repo_root
        / "outputs"
        / "atomic"
        / model_name
        / f"{dataset}.parquet"
    )
    gamma_path = (
        repo_root
        / "outputs"
        / "gamma"
        / model_name
        / f"{dataset}.parquet"
    )

    sets_dir = (
        repo_root
        / "outputs"
        / "behavior"
        / "sets"
        / model_name
        / dataset
    )
    behavior_path = (
        repo_root
        / "outputs"
        / "behavior"
        / "challenge"
        / model_name
        / f"{dataset}.parquet"
    )
    analysis_dir = (
        repo_root
        / "outputs"
        / "behavior"
        / "analysis"
        / model_name
        / dataset
    )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --gpus=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={slurm_time(minutes)}",
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
        f"mkdir -p {q(log_dir)}",
        "",
        'if [[ -z "${HF_TOKEN:-}" ]]; then',
        '  echo "WARNING: HF_TOKEN is not set. Gated models may fail to download." >&2',
        "fi",
        "",
        'echo "============================================================"',
        'echo "Graded Stability: behavioral resilience experiment"',
        f'echo "Model: {model_name}"',
        f'echo "HF model: {cfg.get("model")}"',
        f'echo "Dataset: {dataset}"',
        f'echo "Detected size: {size_billions:g}B"',
        f'echo "Batch size: {batch_size}"',
        f'echo "Memory: {memory}"',
        f'echo "Wall time: {minutes} min"',
        f'echo "Atomic input: {atomic_path}"',
        f'echo "Gamma input: {gamma_path}"',
        f'echo "Behavior output: {behavior_path}"',
        f'echo "Analysis output: {analysis_dir}"',
        'echo "Job ID: ${SLURM_JOB_ID}"',
        'echo "Node: $(hostname)"',
        'echo "Started: $(date)"',
        'echo "============================================================"',
        "python --version",
        "nvidia-smi",
        "",
        f"ATOMIC_PATH={q(atomic_path)}",
        f"GAMMA_PATH={q(gamma_path)}",
        'if [[ ! -s "${ATOMIC_PATH}" ]]; then',
        '  echo "ERROR: required atomic Parquet is missing or empty: ${ATOMIC_PATH}" >&2',
        "  exit 1",
        "fi",
        'if [[ ! -s "${GAMMA_PATH}" ]]; then',
        '  echo "ERROR: required gamma Parquet is missing or empty: ${GAMMA_PATH}" >&2',
        "  exit 1",
        "fi",
        "",
        'echo "---------------- BUILD CHALLENGE SET ----------------"',
        "srun python -m scripts.behavior.build_challenge_set \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        "  --mode both \\",
        "  --overwrite",
        "",
        'echo "---------------- RUN CHALLENGE ----------------------"',
        "# --resume makes timed-out/resubmitted jobs continue from checkpoints.",
        "srun python -m scripts.behavior.run_challenge \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        f"  --batch_size {batch_size} \\",
        f"  --save_every {save_every} \\",
        "  --device cuda \\",
        "  --resume",
        "",
        'echo "---------------- ANALYZE CHALLENGE ------------------"',
        "# This runs only after behavioral generation is complete.",
        "srun python -m scripts.behavior.analyze_challenge \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        f"  --cv_folds {cv_folds} \\",
        f"  --cv_repeats {cv_repeats} \\",
        f"  --sequence_permutations {sequence_permutations} \\",
        "  --overwrite",
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
            "Generate the full 72-job GPU Slurm matrix for behavioral "
            "resilience generation and per-unit analysis."
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
        "--minutes",
        type=int,
        default=None,
        help=(
            "Override wall time for every job. Default: 30 min through 32B, "
            "60 min for >32B."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Override challenge batch size for every job. Default: size-based."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=32,
        help="Checkpoint behavioral output every N newly scored statements.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cv_repeats",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sequence_permutations",
        type=int,
        default=1000,
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
        raise ValueError("--cpus must be positive.")
    if args.save_every <= 0:
        raise ValueError("--save_every must be positive.")
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be >= 2.")
    if args.cv_repeats <= 0:
        raise ValueError("--cv_repeats must be positive.")
    if args.sequence_permutations < 0:
        raise ValueError("--sequence_permutations must be >= 0.")

    repo_root = args.repo_root.resolve()
    config_dir = repo_root / "configs" / "model"

    registry_path = (
        args.registry
        if args.registry is not None
        else repo_root / "configs" / "model_list.yaml"
    )

    generated_dir = (
        repo_root
        / "slurms"
        / "generated"
        / "behavior"
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
        minutes = minutes_for_size(
            size_billions,
            args.minutes,
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
                    minutes=minutes,
                    batch_size=batch_size,
                    save_every=args.save_every,
                    cv_folds=args.cv_folds,
                    cv_repeats=args.cv_repeats,
                    sequence_permutations=args.sequence_permutations,
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
        f"Generated {len(generated)} behavioral Slurm files "
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
            f"time={minutes_for_size(size_billions, args.minutes):>3d} min"
        )

    print()
    print("Submit all 72 with:")
    print(
        f'  find {q(generated_dir)} -maxdepth 1 -name "*.slurm" '
        '-print0 | sort -z | xargs -0 -n1 sbatch'
    )


if __name__ == "__main__":
    main()
