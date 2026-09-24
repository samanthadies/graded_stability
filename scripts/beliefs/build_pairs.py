"""
Constructs the probe-specific pairs (P, x) used to evaluate conditional
belief and graded stability, excluding self-pairs by default.

Examples:
    python -m scripts.beliefs.build_pairs --model_name _llama-3.1-8b --dataset cities_loc
    python -m scripts.beliefs.build_pairs --model_name _llama-3.1-8b --dataset cities_loc --probes sawmil --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from stability.beliefs.pairs import (
    write_pair_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact probe-specific atomic P x x pair table."
        )
    )

    parser.add_argument(
        "--model_name",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "cities_loc",
            "med_indications",
            "defs",
        ),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help=(
            "Optional subset of probes. Default: every probe present in "
            "the atomic Parquet."
        ),
    )

    parser.add_argument(
        "--atomic_dir",
        type=Path,
        default=Path(
            "outputs/atomic"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/pairs"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100_000,
        help=(
            "Maximum number of pair rows materialized in memory at once."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional per-probe row limit for smoke tests."
        ),
    )
    parser.add_argument(
        "--include_self",
        action="store_true",
        help=(
            "Include P=x pairs. Default behavior excludes self-pairs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def _write_json_atomic(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk_size must be positive."
        )
    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be positive when provided."
        )

    atomic_path = (
        args.atomic_dir
        / args.model_name
        / f"{args.dataset}.parquet"
    )
    output_model_dir = (
        args.output_dir
        / args.model_name
    )
    output_path = (
        output_model_dir
        / f"{args.dataset}.parquet"
    )
    summary_path = (
        output_model_dir
        / f"{args.dataset}.json"
    )

    if not atomic_path.exists():
        raise FileNotFoundError(
            f"Atomic Parquet does not exist: {atomic_path}"
        )

    for path in (
        output_path,
        summary_path,
    ):
        if (
            path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                f"{path} exists; pass --overwrite to replace pair outputs."
            )

    atomic = pd.read_parquet(
        atomic_path
    )

    if atomic.empty:
        raise ValueError(
            f"Atomic Parquet is empty: {atomic_path}"
        )

    observed_models = set(
        atomic["model_name"]
        .dropna()
        .astype(str)
        .unique()
    )
    observed_datasets = set(
        atomic["dataset"]
        .dropna()
        .astype(str)
        .unique()
    )

    if observed_models != {
        args.model_name
    }:
        raise ValueError(
            f"Atomic model_name values {observed_models} do not match "
            f"--model_name={args.model_name!r}."
        )

    if observed_datasets != {
        args.dataset
    }:
        raise ValueError(
            f"Atomic dataset values {observed_datasets} do not match "
            f"--dataset={args.dataset!r}."
        )

    summaries = write_pair_parquet(
        atomic,
        output_path=output_path,
        probes=args.probes,
        exclude_self=not args.include_self,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )

    total_pairs = sum(
        summary.num_pairs
        for summary in summaries.values()
    )

    all_empty = total_pairs == 0

    payload = {
        "schema_version": 1,
        "status": (
            "degenerate"
            if all_empty
            else "complete"
        ),
        "reason": (
            "no_pairs_for_any_requested_probe"
            if all_empty
            else None
        ),
        "model_name": args.model_name,
        "dataset": args.dataset,
        "atomic_path": str(
            atomic_path
        ),
        "pair_path": str(
            output_path
        ),
        "exclude_self": not args.include_self,
        "limit_per_probe": args.limit,
        "pair_schema": [
            "probe",
            "pair_id",
            "P_id",
            "x_id",
        ],
        "rendering": {
            "stored_text": False,
            "statement_config": "configs/statements.yaml",
            "conditional_default": "given_x_P",
            "joint_default": "x_then_P",
        },
        "num_pairs_total": int(
            total_pairs
        ),
        "probes": {
            probe: summary.to_dict()
            for probe, summary in summaries.items()
        },
    }

    _write_json_atomic(
        payload,
        summary_path,
    )

    print("=" * 72)
    print("PAIR TABLE COMPLETE")
    print("=" * 72)
    print(f"Model:    {args.model_name}")
    print(f"Dataset:  {args.dataset}")
    print(f"Atomic:   {atomic_path}")
    print(f"Pairs:    {output_path}")
    print(f"Summary:  {summary_path}")
    print(
        f"Self pairs: "
        f"{'included' if args.include_self else 'excluded'}"
    )
    print(
        f"Total pairs: {total_pairs:,}"
    )
    if all_empty:
        print(
            "Status:      DEGENERATE "
            "(no requested probe has any P x x pairs)"
        )
    print()

    for probe_name, summary in summaries.items():
        suffix = ""
        if summary.status != "complete":
            suffix = (
                f" | {summary.status.upper()}"
                f" ({summary.reason})"
            )

        print(
            f"{probe_name:18s} "
            f"P={summary.num_P:,} | "
            f"x={summary.num_x:,} | "
            f"pairs={summary.num_pairs:,}"
            f"{suffix}"
        )


if __name__ == "__main__":
    main()
