"""
Analyzes one model-dataset behavioral challenge run, including gamma's predictive
value beyond atomic credence and the matched-pair behavioral-resilience comparison.

Examples:
    python -m scripts.behavior.analyze_challenge --model_name _llama-3.1-8b --dataset cities_loc
    python -m scripts.behavior.analyze_challenge --model_name _llama-3.1-8b --dataset cities_loc --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from stability.behavior.analysis import (
    COMPLEMENTARY_CONTINUOUS_OUTCOME,
    PRIMARY_CONTINUOUS_OUTCOME,
    PRIMARY_SAMPLE,
    ROBUSTNESS_SAMPLE,
    SECONDARY_BINARY_OUTCOME,
    analyze_matched_pairs,
    association_summary,
    build_probe_analysis_table,
    challenge_order_summaries,
    cross_validated_predictive_validity,
    sequence_effect_tests,
    summarize_cv,
    summarize_matched_sensitivity,
)
from stability.utils.io import write_parquet_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze behavioral validation.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=("cities_loc", "med_indications", "defs")
    )
    parser.add_argument("--behavior_dir", type=Path, default=Path("outputs/behavior/challenge"))
    parser.add_argument("--sets_dir", type=Path, default=Path("outputs/behavior/sets"))
    parser.add_argument("--atomic_dir", type=Path, default=Path("outputs/atomic"))
    parser.add_argument("--gamma_dir", type=Path, default=Path("outputs/gamma"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/behavior/analysis"))
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_repeats", type=int, default=10)
    parser.add_argument("--spline_knots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence_permutations", type=int, default=1000)
    parser.add_argument(
        "--sensitivity_atomic_gaps",
        nargs="+",
        type=float,
        default=[0.001, 0.0025, 0.005, 0.01, 0.02],
    )
    parser.add_argument(
        "--sensitivity_gamma_gaps",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.20],
    )
    parser.add_argument("--sensitivity_min_pairs", type=int, default=10)
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


def _optional(frame: pd.DataFrame, path: Path) -> None:
    if not frame.empty:
        write_parquet_atomic(frame, path)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2 or args.cv_repeats < 1 or args.spline_knots < 2:
        raise ValueError("Invalid CV/spline settings.")
    if args.sequence_permutations < 0 or args.sensitivity_min_pairs < 1:
        raise ValueError("Invalid permutation/sensitivity settings.")

    behavior_path = args.behavior_dir / args.model_name / f"{args.dataset}.parquet"
    set_root = args.sets_dir / args.model_name / args.dataset
    membership_path = set_root / "general_membership.parquet"
    matched_path = set_root / "matched_pairs.parquet"
    atomic_path = args.atomic_dir / args.model_name / f"{args.dataset}.parquet"
    gamma_path = args.gamma_dir / args.model_name / f"{args.dataset}.parquet"
    for path in (behavior_path, membership_path, atomic_path, gamma_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_root = args.output_dir / args.model_name / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis_rows": output_root / "analysis_rows.parquet",
        "association_summary": output_root / "association_summary.parquet",
        "cv_oof_predictions": output_root / "cv_oof_predictions.parquet",
        "cv_repeat_metrics": output_root / "cv_repeat_metrics.parquet",
        "cv_summary": output_root / "cv_summary.parquet",
        "cv_deltas": output_root / "cv_deltas.parquet",
        "sequence_summary": output_root / "sequence_summary.parquet",
        "template_position_summary": output_root / "template_position_summary.parquet",
        "sequence_effect_tests": output_root / "sequence_effect_tests.parquet",
        "matched_results": output_root / "matched_results.parquet",
        "matched_summary": output_root / "matched_summary.parquet",
        "matched_sensitivity": output_root / "matched_sensitivity.parquet",
        "summary": output_root / "summary.json",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError("Behavior analysis exists. Pass --overwrite to replace it.")

    behavior = pd.read_parquet(behavior_path)
    membership = pd.read_parquet(membership_path)
    atomic = pd.read_parquet(atomic_path)
    gamma = pd.read_parquet(gamma_path)

    if membership.empty:
        _write_json(
            {
                "schema_version": 2,
                "status": "skipped_degenerate_no_P",
                "model_name": args.model_name,
                "dataset": args.dataset,
                "primary_sample": PRIMARY_SAMPLE,
                "primary_continuous_outcome": PRIMARY_CONTINUOUS_OUTCOME,
                "complementary_continuous_outcome": COMPLEMENTARY_CONTINUOUS_OUTCOME,
                "secondary_binary_outcome": SECONDARY_BINARY_OUTCOME,
            },
            paths["summary"],
        )
        print(f"SKIP {args.model_name}/{args.dataset}: no P statements.")
        return

    analysis_rows = build_probe_analysis_table(
        behavior=behavior,
        membership=membership,
        atomic=atomic,
        gamma=gamma,
    )
    associations = association_summary(analysis_rows)
    oof_predictions, repeat_metrics = cross_validated_predictive_validity(
        analysis_rows,
        n_splits=args.cv_folds,
        n_repeats=args.cv_repeats,
        n_knots=args.spline_knots,
        seed=args.seed,
    )
    cv_summary, cv_deltas = summarize_cv(repeat_metrics)
    sequence_summary, template_summary = challenge_order_summaries(analysis_rows)
    sequence_tests = sequence_effect_tests(
        analysis_rows,
        n_permutations=args.sequence_permutations,
        seed=args.seed,
    )

    if matched_path.exists():
        matched_results, matched_summary = analyze_matched_pairs(
            analysis_rows=analysis_rows,
            matched_pairs=pd.read_parquet(matched_path),
        )
        matched_sensitivity = summarize_matched_sensitivity(
            matched_results,
            atomic_gap_thresholds=args.sensitivity_atomic_gaps,
            gamma_gap_thresholds=args.sensitivity_gamma_gaps,
            min_pairs=args.sensitivity_min_pairs,
        )
    else:
        matched_results = pd.DataFrame()
        matched_summary = pd.DataFrame()
        matched_sensitivity = pd.DataFrame()

    write_parquet_atomic(analysis_rows, paths["analysis_rows"])
    _optional(associations, paths["association_summary"])
    _optional(oof_predictions, paths["cv_oof_predictions"])
    _optional(repeat_metrics, paths["cv_repeat_metrics"])
    _optional(cv_summary, paths["cv_summary"])
    _optional(cv_deltas, paths["cv_deltas"])
    _optional(sequence_summary, paths["sequence_summary"])
    _optional(template_summary, paths["template_position_summary"])
    _optional(sequence_tests, paths["sequence_effect_tests"])
    _optional(matched_results, paths["matched_results"])
    _optional(matched_summary, paths["matched_summary"])
    _optional(matched_sensitivity, paths["matched_sensitivity"])

    all_counts = analysis_rows.groupby("probe").size().to_dict()
    primary_counts = (
        analysis_rows.loc[analysis_rows["round0_agrees_with_atomic"]]
        .groupby("probe")
        .size()
        .to_dict()
    )
    _write_json(
        {
            "schema_version": 2,
            "status": "complete",
            "model_name": args.model_name,
            "dataset": args.dataset,
            "primary_sample": PRIMARY_SAMPLE,
            "robustness_sample": ROBUSTNESS_SAMPLE,
            "primary_continuous_outcome": PRIMARY_CONTINUOUS_OUTCOME,
            "complementary_continuous_outcome": COMPLEMENTARY_CONTINUOUS_OUTCOME,
            "secondary_binary_outcome": SECONDARY_BINARY_OUTCOME,
            "cv_metric_scope": "pooled out-of-fold predictions within each repeat",
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "spline_knots": int(args.spline_knots),
            "sequence_test": "permutation test of one-way F statistic",
            "sequence_permutations": int(args.sequence_permutations),
            "seed": int(args.seed),
            "probe_all_counts": {str(k): int(v) for k, v in all_counts.items()},
            "probe_primary_counts": {str(k): int(v) for k, v in primary_counts.items()},
            "matched_analysis_present": bool(not matched_summary.empty),
            "matched_primary_selection": "threshold-free sequence-blocked atomic matching",
            "matched_sensitivity_atomic_gaps": [float(x) for x in args.sensitivity_atomic_gaps],
            "matched_sensitivity_gamma_gaps": [float(x) for x in args.sensitivity_gamma_gaps],
        },
        paths["summary"],
    )

    print("=" * 80)
    print("BEHAVIORAL VALIDATION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Model:    {args.model_name}")
    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {output_root}")
    print("\nRows by probe:")
    print(analysis_rows.groupby("probe").size().to_string())
    print("\nPrimary-sample rows by probe:")
    print(
        analysis_rows.loc[analysis_rows["round0_agrees_with_atomic"]]
        .groupby("probe")
        .size()
        .to_string()
    )
    if not cv_deltas.empty:
        delta_columns = [
            column
            for column in ("delta_r2", "delta_rmse", "delta_log_loss", "delta_roc_auc")
            if column in cv_deltas.columns
        ]
        print("\nMean pooled-OOF improvement from adding gamma:")
        print(
            cv_deltas.groupby(["probe", "sample", "estimator", "outcome"])[delta_columns]
            .mean()
            .to_string()
        )
    if not matched_summary.empty:
        print("\nSequence-blocked atomic-matched discrimination:")
        print(matched_summary.to_string(index=False))


if __name__ == "__main__":
    main()
