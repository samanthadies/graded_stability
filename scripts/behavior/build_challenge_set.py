"""
Builds the conversational challenge sets used for behavioral validation, including
the general challenge sample and probability-matched high/low-stability belief pairs.

Examples:
    python -m scripts.behavior.build_challenge_set --model_name _llama-3.1-8b --dataset cities_loc --mode both
    python -m scripts.behavior.build_challenge_set --model_name _llama-3.1-8b --dataset cities_loc --mode matched --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from stability.behavior.config import load_challenge_config
from stability.behavior.sets import build_atomic_matched_pairs, build_general_challenge_set
from stability.data.loading import load_probe_dataset
from stability.utils.io import write_parquet_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build behavioral challenge sets.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=("cities_loc", "med_indications", "defs")
    )
    parser.add_argument("--mode", choices=("general", "matched", "both"), default="general")
    parser.add_argument(
        "--challenge_config", type=Path, default=Path("configs/experiments/challenge.yaml")
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--atomic_dir", type=Path, default=Path("outputs/atomic"))
    parser.add_argument("--gamma_dir", type=Path, default=Path("outputs/gamma"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/behavior/sets"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--matched_estimators",
        nargs="+",
        choices=("conditional", "joint", "consensus"),
        default=["conditional", "joint", "consensus"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _guard(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _empty_matched() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "probe", "estimator", "match_id", "atomic_match_id",
            "challenge_sequence", "statement_a_id", "statement_b_id",
            "atomic_gap", "high_statement_id", "low_statement_id", "gamma_gap",
            "high_atomic_probability", "low_atomic_probability",
            "high_gamma_conditional", "low_gamma_conditional",
            "high_gamma_joint", "low_gamma_joint", "matching_method",
        ]
    )


def main() -> None:
    args = parse_args()
    atomic_path = args.atomic_dir / args.model_name / f"{args.dataset}.parquet"
    if not atomic_path.exists():
        raise FileNotFoundError(atomic_path)
    atomic = pd.read_parquet(atomic_path)

    root = args.output_dir / args.model_name / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    challenge: pd.DataFrame | None = None

    if args.mode in {"general", "both"}:
        config = load_challenge_config(args.challenge_config)
        data = load_probe_dataset(
            args.dataset, construction="atomic", data_dir=args.data_dir
        )
        challenge, membership = build_general_challenge_set(
            atomic=atomic,
            data=data,
            config=config,
            model_name=args.model_name,
            seed=args.seed,
        )

        challenge_path = root / "general.parquet"
        membership_path = root / "general_membership.parquet"
        summary_path = root / "general.json"
        _guard([challenge_path, membership_path, summary_path], overwrite=args.overwrite)
        write_parquet_atomic(challenge, challenge_path)
        write_parquet_atomic(membership, membership_path)

        degenerate = challenge.empty
        _write_json(
            {
                "schema_version": 2,
                "status": "degenerate_no_P" if degenerate else "complete",
                "mode": "general",
                "model_name": args.model_name,
                "dataset": args.dataset,
                "challenge_config": config.name,
                "challenge_config_fingerprint": config.fingerprint,
                "counterbalancing_seed": config.seed if args.seed is None else int(args.seed),
                "num_unique_statements": int(len(challenge)),
                "num_membership_rows": int(len(membership)),
                "probe_P_counts": {
                    str(k): int(v) for k, v in membership.groupby("probe").size().to_dict().items()
                } if not membership.empty else {},
                "challenge_sequence_counts": {
                    str(k): int(v)
                    for k, v in challenge["challenge_sequence"].value_counts().sort_index().to_dict().items()
                } if not challenge.empty else {},
            },
            summary_path,
        )

        print("=" * 72)
        print("GENERAL BEHAVIORAL CHALLENGE SET COMPLETE")
        print("=" * 72)
        print(f"Model:       {args.model_name}")
        print(f"Dataset:     {args.dataset}")
        print(f"Statements:  {len(challenge):,}")
        print(f"Membership:  {len(membership):,}")
        if degenerate:
            print("Status:      degenerate_no_P (empty placeholder outputs written)")

    if args.mode in {"matched", "both"}:
        gamma_path = args.gamma_dir / args.model_name / f"{args.dataset}.parquet"
        if not gamma_path.exists():
            raise FileNotFoundError(gamma_path)

        if challenge is None:
            general_path = root / "general.parquet"
            if not general_path.exists():
                raise FileNotFoundError(
                    f"{general_path} is required for sequence-blocked matching. "
                    "Run --mode general or --mode both first."
                )
            challenge = pd.read_parquet(general_path)

        if challenge.empty:
            matched = _empty_matched()
        else:
            matched = build_atomic_matched_pairs(
                atomic=atomic,
                gamma=pd.read_parquet(gamma_path),
                challenge_set=challenge,
                estimators=args.matched_estimators,
            )
            if matched.empty:
                matched = _empty_matched()

        matched_path = root / "matched_pairs.parquet"
        summary_path = root / "matched_pairs.json"
        _guard([matched_path, summary_path], overwrite=args.overwrite)
        write_parquet_atomic(matched, matched_path)

        counts = (
            matched.groupby(["probe", "estimator"]).size().to_dict()
            if not matched.empty else {}
        )
        atomic_match_counts = (
            matched[["probe", "atomic_match_id"]]
            .drop_duplicates()
            .groupby("probe")
            .size()
            .to_dict()
            if not matched.empty else {}
        )
        sequence_counts = (
            matched[["probe", "atomic_match_id", "challenge_sequence"]]
            .drop_duplicates()
            .groupby("challenge_sequence")
            .size()
            .to_dict()
            if not matched.empty else {}
        )

        _write_json(
            {
                "schema_version": 3,
                "status": "degenerate_no_matches" if matched.empty else "complete",
                "mode": "matched",
                "model_name": args.model_name,
                "dataset": args.dataset,
                "estimators": list(args.matched_estimators),
                "matching_method": (
                    "maximum-cardinality minimum-total-absolute-atomic-distance "
                    "within exact challenge-sequence blocks"
                ),
                "blocked_on_challenge_sequence": True,
                "selection_uses_gamma": False,
                "selection_uses_behavior": False,
                "gamma_used_only_for_orientation": True,
                "hard_atomic_gap_threshold": None,
                "hard_gamma_gap_threshold": None,
                "pair_limit": None,
                "num_output_rows": int(len(matched)),
                "atomic_match_counts": {str(k): int(v) for k, v in atomic_match_counts.items()},
                "challenge_sequence_match_counts": {
                    str(k): int(v) for k, v in sequence_counts.items()
                },
                "estimator_row_counts": {
                    f"{probe}/{estimator}": int(count)
                    for (probe, estimator), count in counts.items()
                },
            },
            summary_path,
        )

        print("\n" + "=" * 72)
        print("ATOMIC-MATCHED PAIR SELECTION COMPLETE")
        print("=" * 72)
        print("Method: sequence-blocked maximum-cardinality / minimum-total Pr(P) distance")
        print("Gamma thresholds: none")
        print("Atomic thresholds: none")
        print(f"Rows: {len(matched):,}")
        if matched.empty:
            print("Status: degenerate_no_matches (empty placeholder output written)")


if __name__ == "__main__":
    main()
