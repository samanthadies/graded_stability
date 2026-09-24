"""
Generate the full 24-model x 3-dataset Slurm matrix for the model-agnostic
atomic set-definition + pair-construction bridge.

Each generated CPU job runs, in order:

    python -m scripts.beliefs.define_sets
    python -m scripts.beliefs.build_pairs

for one model x one dataset.

The set-definition step enriches:
    outputs/atomic/<model>/<dataset>.parquet

and writes:
    outputs/atomic/<model>/<dataset>.json

The pair-construction step writes:
    outputs/pairs/<model>/<dataset>.parquet
    outputs/pairs/<model>/<dataset>.json

Because this stage performs no LLM forward passes, jobs use the CPU partition
and request no GPU.

Completed pair outputs are treated as the completion marker. If both pair
outputs already exist and are nonempty, the job exits successfully. Otherwise
set definition and pair construction are recomputed with --overwrite. Both
operations are deterministic and cheap relative to model scoring.
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


def render_job(
    *,
    repo_root: Path,
    generated_dir: Path,
    model_name: str,
    dataset: str,
    account: str,
    partition: str,
    cpus: int,
    memory: str,
    hours: int,
    pair_chunk_size: int,
) -> str:
    model_token = safe_token(model_name)
    dataset_token = safe_token(dataset)
    job_name = f"pairs-{model_token}-{dataset_token}"

    log_dir = generated_dir / "logs"

    atomic_dir = (
        repo_root
        / "outputs"
        / "atomic"
        / model_name
    )
    atomic_parquet = (
        atomic_dir
        / f"{dataset}.parquet"
    )
    atomic_summary = (
        atomic_dir
        / f"{dataset}.json"
    )

    pair_dir = (
        repo_root
        / "outputs"
        / "pairs"
        / model_name
    )
    pair_parquet = (
        pair_dir
        / f"{dataset}.parquet"
    )
    pair_summary = (
        pair_dir
        / f"{dataset}.json"
    )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
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
        f"mkdir -p {q(atomic_dir)}",
        f"mkdir -p {q(pair_dir)}",
        f"mkdir -p {q(log_dir)}",
        "",
        'echo "============================================================"',
        'echo "Graded Stability: define atomic sets + construct pairs"',
        f'echo "Model: {model_name}"',
        f'echo "Dataset: {dataset}"',
        f'echo "Pair chunk size: {pair_chunk_size}"',
        f'echo "Atomic Parquet: {atomic_parquet}"',
        f'echo "Atomic summary: {atomic_summary}"',
        f'echo "Pair Parquet: {pair_parquet}"',
        f'echo "Pair summary: {pair_summary}"',
        'echo "Job ID: ${SLURM_JOB_ID}"',
        'echo "Node: $(hostname)"',
        'echo "Started: $(date)"',
        'echo "============================================================"',
        "python --version",
        "",
        f"ATOMIC_PARQUET={q(atomic_parquet)}",
        f"PAIR_PARQUET={q(pair_parquet)}",
        f"PAIR_SUMMARY={q(pair_summary)}",
        "",
        'if [[ ! -s "${ATOMIC_PARQUET}" ]]; then',
        '  echo "ERROR: required atomic Parquet is missing or empty: ${ATOMIC_PARQUET}" >&2',
        "  exit 1",
        "fi",
        "",
        "# The pair outputs are the completion marker for this bridge stage.",
        'if [[ -s "${PAIR_PARQUET}" && -s "${PAIR_SUMMARY}" ]]; then',
        '  echo "Complete pair outputs already exist; nothing to do."',
        "  exit 0",
        "fi",
        "",
        "# Recompute selection columns deterministically. This is intentionally",
        "# --overwrite because the atomic Parquet may already have been enriched.",
        "srun python -m scripts.beliefs.define_sets \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        "  --overwrite",
        "",
        "# Pair construction is also deterministic and streams chunks to Parquet.",
        "srun python -m scripts.beliefs.build_pairs \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
        f"  --chunk_size {pair_chunk_size} \\",
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
            "Generate 72 CPU Slurm jobs for define_sets + pair construction."
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
        default="cpu",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--memory",
        default="16G",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--pair_chunk_size",
        type=int,
        default=100_000,
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
    if args.hours <= 0:
        raise ValueError("--hours must be positive.")
    if args.pair_chunk_size <= 0:
        raise ValueError("--pair_chunk_size must be positive.")

    repo_root = args.repo_root.resolve()
    registry_path = (
        args.registry
        if args.registry is not None
        else repo_root / "configs" / "model_list.yaml"
    )

    generated_dir = (
        repo_root
        / "slurms"
        / "generated"
        / "pairs"
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

    expected_jobs = (
        EXPECTED_MODEL_COUNT
        * len(DATASETS)
    )

    generated: list[Path] = []

    for model_name in model_names:
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
                    dataset=dataset,
                    account=args.account,
                    partition=args.partition,
                    cpus=args.cpus,
                    memory=args.memory,
                    hours=args.hours,
                    pair_chunk_size=args.pair_chunk_size,
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
        f"Generated {len(generated)} define-sets/pair Slurm files "
        f"({EXPECTED_MODEL_COUNT} models x {len(DATASETS)} datasets)."
    )
    print(f"Slurms: {generated_dir}")
    print(f"Logs:   {log_dir}")
    print(
        f"Resources: partition={args.partition}, cpus={args.cpus}, "
        f"mem={args.memory}, time={args.hours}h"
    )
    print()
    print("Submit all 72 with:")
    print(
        f'  find {q(generated_dir)} -maxdepth 1 -name "*.slurm" '
        '-print0 | sort -z | xargs -0 -n1 sbatch'
    )


if __name__ == "__main__":
    main()
