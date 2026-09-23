from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stability.beliefs.gamma import (
    DIRECT_PROBABILITY_COLUMNS,
    JOINT_PROBABILITY_COLUMNS,
    attach_pair_metadata,
    compute_gamma,
    direct_conditional_scores,
    joint_conditional_scores,
)
from stability.utils.io import (
    write_parquet_atomic,
)


SUPPORTED_SOURCES = (
    "conditional",
    "joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Lockean gamma from direct-conditional and/or "
            "joint-derived probe probabilities."
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
        "--sources",
        nargs="+",
        choices=SUPPORTED_SOURCES,
        default=list(
            SUPPORTED_SOURCES
        ),
        help=(
            "Probability sources to compute. Default: conditional joint."
        ),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help=(
            "Optional probe subset. Default: every probe present in the "
            "atomic selection table."
        ),
    )
    parser.add_argument(
        "--zero_tolerance",
        type=float,
        default=1e-12,
        help=(
            "Joint conditional denominators <= this value are undefined. "
            "Default: 1e-12."
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
        "--pairs_dir",
        type=Path,
        default=Path(
            "outputs/pairs"
        ),
    )
    parser.add_argument(
        "--conditional_dir",
        type=Path,
        default=Path(
            "outputs/conditional"
        ),
    )
    parser.add_argument(
        "--joint_dir",
        type=Path,
        default=Path(
            "outputs/joint"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/gamma"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def _write_json_atomic(
    payload: dict[str, Any],
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


def _validate_identity(
    atomic: pd.DataFrame,
    *,
    model_name: str,
    dataset: str,
) -> None:
    required = {
        "model_name",
        "dataset",
        "probe",
        "statement_id",
        "threshold",
        "is_P",
    }
    missing = required - set(
        atomic.columns
    )
    if missing:
        raise ValueError(
            f"Atomic Parquet is missing required columns {sorted(missing)}."
        )

    models = set(
        atomic[
            "model_name"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
    )
    datasets = set(
        atomic[
            "dataset"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
    )

    if models != {
        model_name
    }:
        raise ValueError(
            f"Atomic model_name values {sorted(models)} do not match "
            f"{model_name!r}."
        )

    if datasets != {
        dataset
    }:
        raise ValueError(
            f"Atomic dataset values {sorted(datasets)} do not match "
            f"{dataset!r}."
        )


def _normalize_probes(
    atomic: pd.DataFrame,
    requested: list[str] | None,
) -> list[str]:
    available = sorted(
        atomic[
            "probe"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    if requested is None:
        return available

    probes = list(
        dict.fromkeys(
            str(
                probe
            ).strip()
            for probe in requested
        )
    )

    missing = sorted(
        set(
            probes
        )
        - set(
            available
        )
    )
    if missing:
        raise ValueError(
            f"Requested probes are absent from the atomic table: {missing}. "
            f"Available: {available}."
        )

    return probes


def _probe_threshold(
    atomic: pd.DataFrame,
    *,
    probe_name: str,
) -> float:
    subset = atomic[
        atomic[
            "probe"
        ].astype(
            str
        )
        == probe_name
    ]

    values = (
        pd.to_numeric(
            subset[
                "threshold"
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(
        values
    ) != 1:
        raise ValueError(
            f"{probe_name}: expected exactly one finite threshold; "
            f"found {values.tolist()}."
        )

    threshold = float(
        values[
            0
        ]
    )
    if not np.isfinite(
        threshold
    ):
        raise ValueError(
            f"{probe_name}: threshold is non-finite."
        )

    return threshold


def _expected_P_ids(
    atomic: pd.DataFrame,
    *,
    probe_name: str,
) -> set[int]:
    subset = atomic[
        (
            atomic[
                "probe"
            ].astype(
                str
            )
            == probe_name
        )
        & atomic[
            "is_P"
        ].astype(
            bool
        )
    ]

    return set(
        pd.to_numeric(
            subset[
                "statement_id"
            ],
            errors="raise",
        )
        .astype(
            np.int64
        )
        .tolist()
    )


def _load_probe_pairs(
    path: Path,
    *,
    probe_name: str,
) -> pd.DataFrame:
    pairs = pd.read_parquet(
        path,
        columns=[
            "probe",
            "pair_id",
            "P_id",
            "x_id",
        ],
        filters=[
            (
                "probe",
                "==",
                probe_name,
            )
        ],
    )

    if pairs.empty:
        raise ValueError(
            f"{probe_name}: pair table contains no rows."
        )

    return pairs


def _load_source_scores(
    path: Path,
    *,
    probe_name: str,
    source: str,
) -> pd.DataFrame:
    probability_columns = (
        list(
            DIRECT_PROBABILITY_COLUMNS
        )
        if source
        == "conditional"
        else list(
            JOINT_PROBABILITY_COLUMNS
        )
    )

    columns = [
        "probe",
        "pair_id",
        *probability_columns,
    ]

    scores = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            (
                "probe",
                "==",
                probe_name,
            )
        ],
    )

    if scores.empty:
        raise ValueError(
            f"{probe_name}/{source}: score file contains no rows."
        )

    observed = set(
        scores[
            "probe"
        ]
        .astype(
            str
        )
        .unique()
    )
    if observed != {
        probe_name
    }:
        raise ValueError(
            f"{probe_name}/{source}: unexpected probe values "
            f"{sorted(observed)}."
        )

    return scores.drop(
        columns=[
            "probe",
        ]
    )


def main() -> None:
    args = parse_args()

    sources = list(
        dict.fromkeys(
            str(
                source
            ).strip()
            .lower()
            for source in args.sources
        )
    )

    if (
        not np.isfinite(
            args.zero_tolerance
        )
        or args.zero_tolerance < 0
    ):
        raise ValueError(
            "--zero_tolerance must be finite and nonnegative."
        )

    atomic_path = (
        args.atomic_dir
        / args.model_name
        / f"{args.dataset}.parquet"
    )
    pairs_path = (
        args.pairs_dir
        / args.model_name
        / f"{args.dataset}.parquet"
    )
    conditional_path = (
        args.conditional_dir
        / args.model_name
        / f"{args.dataset}.parquet"
    )
    joint_path = (
        args.joint_dir
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

    for path, description in (
        (
            atomic_path,
            "atomic selection Parquet",
        ),
        (
            pairs_path,
            "pair Parquet",
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required {description} does not exist: {path}"
            )

    source_paths = {
        "conditional": conditional_path,
        "joint": joint_path,
    }

    for path in (
        output_path,
        summary_path,
    ):
        if (
            path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                f"{path} exists; pass --overwrite to replace gamma outputs."
            )

    atomic = pd.read_parquet(
        atomic_path
    )
    if atomic.empty:
        raise ValueError(
            f"Atomic Parquet is empty: {atomic_path}"
        )

    _validate_identity(
        atomic,
        model_name=args.model_name,
        dataset=args.dataset,
    )

    probes = _normalize_probes(
        atomic,
        args.probes,
    )

    pair_probe_frame = pd.read_parquet(
        pairs_path,
        columns=["probe"],
    )
    available_pair_probes = set(
        pair_probe_frame["probe"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if pair_probe_frame.empty:
        output_model_dir.mkdir(parents=True, exist_ok=True)
        empty_result = pd.DataFrame(
            {
                "model_name": pd.Series(dtype="object"),
                "dataset": pd.Series(dtype="object"),
                "probe": pd.Series(dtype="object"),
                "P_id": pd.Series(dtype="int64"),
                "source": pd.Series(dtype="object"),
                "gamma": pd.Series(dtype="float64"),
                "numerator": pd.Series(dtype="int64"),
                "denominator": pd.Series(dtype="int64"),
                "num_rows": pd.Series(dtype="int64"),
                "num_undefined": pd.Series(dtype="int64"),
                "threshold": pd.Series(dtype="float64"),
                "min_score": pd.Series(dtype="float64"),
                "max_score": pd.Series(dtype="float64"),
                "mean_score": pd.Series(dtype="float64"),
                "zero_tolerance": pd.Series(dtype="float64"),
            }
        )
        write_parquet_atomic(empty_result, output_path)
        skipped_probes = {
            probe_name: {
                "status": "degenerate",
                "reason": "no_P_statements",
                "num_P": 0,
                "num_pairs": 0,
            }
            for probe_name in probes
        }
        summary = {
            "schema_version": 1,
            "status": "degenerate",
            "reason": "empty_pair_table",
            "model_name": args.model_name,
            "dataset": args.dataset,
            "sources": sources,
            "probes": probes,
            "num_rows": 0,
            "probe_summaries": {},
            "skipped_probes": skipped_probes,
            "paths": {
                "atomic": str(atomic_path),
                "pairs": str(pairs_path),
                "conditional": str(conditional_path) if conditional_path.exists() else None,
                "joint": str(joint_path) if joint_path.exists() else None,
                "output": str(output_path),
            },
        }
        _write_json_atomic(summary, summary_path)
        print("=" * 72)
        print("GAMMA COMPUTATION")
        print("=" * 72)
        print(f"Model:           {args.model_name}")
        print(f"Dataset:         {args.dataset}")
        print("Pair rows:       0")
        print("Status:          DEGENERATE (empty pair table)")
        print(f"Parquet:         {output_path}")
        print(f"Summary:         {summary_path}")
        print("=" * 72)
        return

    print("=" * 72)
    print("GAMMA COMPUTATION")
    print("=" * 72)
    print(f"Model:           {args.model_name}")
    print(f"Dataset:         {args.dataset}")
    print(f"Probes:          {probes}")
    print(f"Sources:         {sources}")
    print(f"Zero tolerance:  {args.zero_tolerance:.3e}")
    print(f"Atomic:          {atomic_path}")
    print(f"Pairs:           {pairs_path}")
    if "conditional" in sources:
        print(f"Conditional:     {conditional_path}")
    if "joint" in sources:
        print(f"Joint:           {joint_path}")
    print(f"Output:          {output_path}")
    print("=" * 72)

    result_frames: list[
        pd.DataFrame
    ] = []
    summary_probes: dict[
        str,
        Any,
    ] = {}

    skipped_probes: dict[str, Any] = {}

    for probe_name in probes:
        expected_P_ids = _expected_P_ids(
            atomic,
            probe_name=probe_name,
        )

        if (
            not expected_P_ids
            or probe_name not in available_pair_probes
        ):
            skipped_probes[probe_name] = {
                "status": "degenerate",
                "reason": (
                    "no_P_statements"
                    if not expected_P_ids
                    else "no_pair_rows"
                ),
                "num_P": int(len(expected_P_ids)),
                "num_pairs": 0,
            }
            print()
            print(
                f"{probe_name}: SKIPPED "
                f"({skipped_probes[probe_name]['reason']})"
            )
            continue

        threshold = _probe_threshold(
            atomic,
            probe_name=probe_name,
        )

        pairs = _load_probe_pairs(
            pairs_path,
            probe_name=probe_name,
        )

        observed_P_ids = set(
            pd.to_numeric(
                pairs[
                    "P_id"
                ],
                errors="raise",
            )
            .astype(
                np.int64
            )
            .tolist()
        )

        if observed_P_ids != expected_P_ids:
            missing_P = (
                expected_P_ids
                - observed_P_ids
            )
            extra_P = (
                observed_P_ids
                - expected_P_ids
            )
            raise ValueError(
                f"{probe_name}: pair-table P IDs do not match atomic is_P. "
                f"Missing={len(missing_P)}, extra={len(extra_P)}."
            )

        probe_summary: dict[
            str,
            Any,
        ] = {
            "threshold": threshold,
            "num_P": len(
                expected_P_ids
            ),
            "num_pairs": int(
                len(
                    pairs
                )
            ),
            "sources": {},
        }

        print()
        print(
            f"{probe_name}: threshold={threshold:.8f} | "
            f"P={len(expected_P_ids):,} | pairs={len(pairs):,}"
        )

        for source in sources:
            source_path = source_paths[source]

            if not source_path.exists():
                probe_summary["sources"][source] = {
                    "status": "skipped",
                    "reason": "score_parquet_missing",
                    "num_rows": 0,
                    "num_P_with_defined_gamma": 0,
                    "num_P_with_undefined_gamma": 0,
                    "total_defined_pairs": 0,
                    "total_undefined_pairs": 0,
                    "mean_gamma": None,
                }
                print(
                    f"  {source:11s} SKIPPED "
                    "(score Parquet missing)"
                )
                continue

            scores = _load_source_scores(
                source_path,
                probe_name=probe_name,
                source=source,
            )

            if source == "conditional":
                prepared = direct_conditional_scores(
                    scores
                )
                zero_tolerance = None
            else:
                prepared = joint_conditional_scores(
                    scores,
                    zero_tolerance=args.zero_tolerance,
                )
                zero_tolerance = float(
                    args.zero_tolerance
                )

            pair_scores = attach_pair_metadata(
                pairs,
                prepared,
                probe_name=probe_name,
            )

            gamma = compute_gamma(
                pair_scores,
                threshold=threshold,
                probe_name=probe_name,
                source=source,
                zero_tolerance=zero_tolerance,
            )

            gamma_P_ids = set(
                gamma[
                    "P_id"
                ]
                .astype(
                    np.int64
                )
                .tolist()
            )
            if gamma_P_ids != expected_P_ids:
                raise RuntimeError(
                    f"{probe_name}/{source}: gamma P IDs do not match "
                    "the atomic P set."
                )

            result_frames.append(
                gamma
            )

            total_undefined = int(
                gamma[
                    "num_undefined"
                ].sum()
            )
            num_P_undefined = int(
                gamma[
                    "gamma"
                ].isna()
                .sum()
            )

            probe_summary[
                "sources"
            ][
                source
            ] = {
                "num_rows": int(
                    len(
                        gamma
                    )
                ),
                "num_P_with_defined_gamma": int(
                    gamma[
                        "gamma"
                    ].notna()
                    .sum()
                ),
                "num_P_with_undefined_gamma": num_P_undefined,
                "total_defined_pairs": int(
                    gamma[
                        "denominator"
                    ].sum()
                ),
                "total_undefined_pairs": total_undefined,
                "mean_gamma": (
                    float(
                        gamma[
                            "gamma"
                        ]
                        .dropna()
                        .mean()
                    )
                    if gamma[
                        "gamma"
                    ].notna().any()
                    else None
                ),
            }

            print(
                f"  {source:11s} "
                f"defined={int(gamma['denominator'].sum()):,} | "
                f"undefined={total_undefined:,} | "
                f"mean gamma="
                f"{probe_summary['sources'][source]['mean_gamma']}"
            )

    if result_frames:
        result = pd.concat(
            result_frames,
            ignore_index=True,
            sort=False,
        ).sort_values(
            [
                "probe",
                "P_id",
                "source",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )
    else:
        result = pd.DataFrame(
            {
                "probe": pd.Series(dtype="object"),
                "P_id": pd.Series(dtype="int64"),
                "source": pd.Series(dtype="object"),
                "gamma": pd.Series(dtype="float64"),
                "numerator": pd.Series(dtype="int64"),
                "denominator": pd.Series(dtype="int64"),
                "num_rows": pd.Series(dtype="int64"),
                "num_undefined": pd.Series(dtype="int64"),
                "threshold": pd.Series(dtype="float64"),
                "min_score": pd.Series(dtype="float64"),
                "max_score": pd.Series(dtype="float64"),
                "mean_score": pd.Series(dtype="float64"),
                "zero_tolerance": pd.Series(dtype="float64"),
            }
        )

    # Add run identity only once at the compact gamma stage.
    result.insert(
        0,
        "model_name",
        args.model_name,
    )
    result.insert(
        1,
        "dataset",
        args.dataset,
    )

    write_parquet_atomic(
        result,
        output_path,
    )

    if len(result) == 0:
        run_status = "degenerate"
        run_reason = "no_gamma_rows"
    elif skipped_probes:
        run_status = "partial"
        run_reason = "one_or_more_probes_skipped"
    else:
        run_status = "complete"
        run_reason = None

    summary = {
        "schema_version": 1,
        "status": run_status,
        "reason": run_reason,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "sources": sources,
        "probes": probes,
        "gamma_definition": (
            "#{defined x : score(P|x) > threshold} / #{defined x}"
        ),
        "threshold_comparison": "strict_greater_than",
        "direct_score_definition": "prob_true",
        "joint_score_definition": (
            "(prob_TT + prob_NT) / "
            "(prob_TT + prob_TF + prob_NT + prob_NF)"
        ),
        "joint_label_order": {
            "first_symbol": "x",
            "second_symbol": "P",
        },
        "zero_tolerance": float(
            args.zero_tolerance
        ),
        "zero_denominator_convention": (
            "joint-derived conditional is undefined and excluded from "
            "gamma denominator when denominator <= zero_tolerance"
        ),
        "paths": {
            "atomic": str(
                atomic_path
            ),
            "pairs": str(
                pairs_path
            ),
            "conditional": (
                str(
                    conditional_path
                )
                if "conditional" in sources
                else None
            ),
            "joint": (
                str(
                    joint_path
                )
                if "joint" in sources
                else None
            ),
            "output": str(
                output_path
            ),
        },
        "num_rows": int(
            len(
                result
            )
        ),
        "probe_summaries": summary_probes,
        "skipped_probes": skipped_probes,
    }

    _write_json_atomic(
        summary,
        summary_path,
    )

    print()
    print("=" * 72)
    print("GAMMA COMPUTATION COMPLETE")
    print("=" * 72)
    print(f"Parquet: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows:    {len(result):,}")


if __name__ == "__main__":
    main()
