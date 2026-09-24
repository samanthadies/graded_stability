"""
Generate the full 24-model x 3-dataset Slurm matrix for gamma computation.

Each generated CPU job runs:

    python -m scripts.beliefs.compute_gamma

for one model x one dataset.

Required inputs:
    outputs/atomic/<model>/<dataset>.parquet
    outputs/pairs/<model>/<dataset>.parquet

Conditional/joint score Parquets are normally expected, but may be absent for
valid degenerate cases in which the upstream stage wrote a JSON summary and
there were no P x x pairs to score.

Outputs:
    outputs/gamma/<model>/<dataset>.parquet
    outputs/gamma/<model>/<dataset>.json

Because gamma computation performs no LLM forward passes, jobs use the CPU
partition and request no GPU.

Completed gamma outputs are treated as the completion marker. If both output
files already exist and are nonempty, the job exits successfully. Otherwise
gamma is recomputed with --overwrite.
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
    minutes: int,
) -> str:
    model_token = safe_token(model_name)
    dataset_token = safe_token(dataset)
    job_name = f"gamma-{model_token}-{dataset_token}"

    log_dir = generated_dir / "logs"

    atomic_path = (
        repo_root
        / "outputs"
        / "atomic"
        / model_name
        / f"{dataset}.parquet"
    )
    pairs_path = (
        repo_root
        / "outputs"
        / "pairs"
        / model_name
        / f"{dataset}.parquet"
    )
    conditional_path = (
        repo_root
        / "outputs"
        / "conditional"
        / model_name
        / f"{dataset}.parquet"
    )
    conditional_summary = (
        repo_root
        / "outputs"
        / "conditional"
        / model_name
        / f"{dataset}.json"
    )
    joint_path = (
        repo_root
        / "outputs"
        / "joint"
        / model_name
        / f"{dataset}.parquet"
    )
    joint_summary = (
        repo_root
        / "outputs"
        / "joint"
        / model_name
        / f"{dataset}.json"
    )

    gamma_dir = (
        repo_root
        / "outputs"
        / "gamma"
        / model_name
    )
    gamma_parquet = (
        gamma_dir
        / f"{dataset}.parquet"
    )
    gamma_summary = (
        gamma_dir
        / f"{dataset}.json"
    )

    hours, remaining_minutes = divmod(
        int(minutes),
        60,
    )
    wall_time = (
        f"{hours:02d}:{remaining_minutes:02d}:00"
    )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={wall_time}",
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
        f"mkdir -p {q(gamma_dir)}",
        f"mkdir -p {q(log_dir)}",
        "",
        'echo "============================================================"',
        'echo "Graded Stability: gamma computation"',
        f'echo "Model: {model_name}"',
        f'echo "Dataset: {dataset}"',
        f'echo "Atomic: {atomic_path}"',
        f'echo "Pairs: {pairs_path}"',
        f'echo "Conditional: {conditional_path}"',
        f'echo "Conditional summary: {conditional_summary}"',
        f'echo "Joint: {joint_path}"',
        f'echo "Joint summary: {joint_summary}"',
        f'echo "Gamma Parquet: {gamma_parquet}"',
        f'echo "Gamma summary: {gamma_summary}"',
        f'echo "Wall time: {wall_time}"',
        'echo "Job ID: ${SLURM_JOB_ID}"',
        'echo "Node: $(hostname)"',
        'echo "Started: $(date)"',
        'echo "============================================================"',
        "python --version",
        "",
        f"ATOMIC={q(atomic_path)}",
        f"PAIRS={q(pairs_path)}",
        f"CONDITIONAL={q(conditional_path)}",
        f"CONDITIONAL_SUMMARY={q(conditional_summary)}",
        f"JOINT={q(joint_path)}",
        f"JOINT_SUMMARY={q(joint_summary)}",
        f"GAMMA_PARQUET={q(gamma_parquet)}",
        f"GAMMA_SUMMARY={q(gamma_summary)}",
        "",
        "# Atomic selection must exist and be nonempty.",
        'if [[ ! -s "${ATOMIC}" ]]; then',
        '  echo "ERROR: required atomic input is missing or empty: ${ATOMIC}" >&2',
        "  exit 1",
        "fi",
        "",
        "# The pair Parquet must exist. A valid zero-row Parquet is allowed.",
        'if [[ ! -e "${PAIRS}" ]]; then',
        '  echo "ERROR: required pair input is missing: ${PAIRS}" >&2',
        "  exit 1",
        "fi",
        "",
        "# Conditional/joint scoring may legitimately finish without a Parquet",
        "# when the pair table is degenerate. In that case the upstream JSON",
        "# summary is the completion marker. Require either the Parquet or JSON.",
        'if [[ ! -s "${CONDITIONAL}" && ! -s "${CONDITIONAL_SUMMARY}" ]]; then',
        '  echo "ERROR: conditional stage has neither score Parquet nor summary JSON." >&2',
        '  echo "  Parquet: ${CONDITIONAL}" >&2',
        '  echo "  Summary: ${CONDITIONAL_SUMMARY}" >&2',
        "  exit 1",
        "fi",
        "",
        'if [[ ! -s "${JOINT}" && ! -s "${JOINT_SUMMARY}" ]]; then',
        '  echo "ERROR: joint stage has neither score Parquet nor summary JSON." >&2',
        '  echo "  Parquet: ${JOINT}" >&2',
        '  echo "  Summary: ${JOINT_SUMMARY}" >&2',
        "  exit 1",
        "fi",
        "",
        "# Existing gamma Parquet + summary are the completion marker.",
        'if [[ -s "${GAMMA_PARQUET}" && -s "${GAMMA_SUMMARY}" ]]; then',
        '  echo "Complete gamma outputs already exist; nothing to do."',
        "  exit 0",
        "fi",
        "",
        "# Gamma is deterministic and inexpensive, so recompute incomplete outputs.",
        "srun python -m scripts.beliefs.compute_gamma \\",
        f"  --model_name {q(model_name)} \\",
        f"  --dataset {q(dataset)} \\",
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
            "Generate 72 CPU Slurm jobs for gamma computation."
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
        "--minutes",
        type=int,
        default=30,
        help="Wall time in minutes. Default: 30.",
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
    if args.minutes <= 0:
        raise ValueError(
            "--minutes must be positive."
        )

    repo_root = args.repo_root.resolve()
    registry_path = (
        args.registry
        if args.registry is not None
        else repo_root
        / "configs"
        / "model_list.yaml"
    )

    generated_dir = (
        repo_root
        / "slurms"
        / "generated"
        / "gamma"
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
                    minutes=args.minutes,
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
        f"Generated {len(generated)} gamma Slurm files "
        f"({EXPECTED_MODEL_COUNT} models x {len(DATASETS)} datasets)."
    )
    print(f"Slurms: {generated_dir}")
    print(f"Logs:   {log_dir}")
    print(
        f"Resources: partition={args.partition}, "
        f"cpus={args.cpus}, mem={args.memory}, "
        f"time={args.minutes}m"
    )
    print()
    print("Submit all 72 with:")
    print(
        f'  find {q(generated_dir)} -maxdepth 1 -name "*.slurm" '
        '-print0 | sort -z | xargs -0 -n1 sbatch'
    )


if __name__ == "__main__":
    main()
