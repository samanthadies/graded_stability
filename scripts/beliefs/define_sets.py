=from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from stability.beliefs.selection import (
    SELECTION_COLUMNS,
    define_sets,
)
from stability.utils.io import (
    read_parquet_if_exists,
    write_parquet_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich an atomic score Parquet in-place with probe-specific "
            "belief (P) and admissible-condition (x) sets."
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
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/atomic"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recompute selection columns and summary if they already exist."
        ),
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


def _already_defined(
    frame: pd.DataFrame,
) -> bool:
    return all(
        column in frame.columns
        for column in SELECTION_COLUMNS
    )



_DEGENERATE_NO_TRUE_PATTERN = re.compile(
    r"^(?P<probe>[^:]+): no statements were predicted true, "
    r"so the atomic belief threshold cannot be defined\.$"
)


def _normalize_labels_for_degenerate_rows(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "abstain": "neither",
                "suspended": "neither",
                "suspension": "neither",
            }
        )
    )

    invalid = sorted(
        set(normalized.unique())
        - {"true", "false", "neither"}
    )
    if invalid:
        raise ValueError(
            "Cannot construct degenerate set rows because pred_label contains "
            f"unexpected values: {invalid}."
        )

    return normalized


def _enrich_degenerate_probe_rows(
    probe_rows: pd.DataFrame,
) -> pd.DataFrame:
    result = probe_rows.copy()
    predicted = _normalize_labels_for_degenerate_rows(
        result["pred_label"]
    )

    # Ensure the complete selection schema exists even when define_sets()
    # could not produce it for this probe.
    for column in SELECTION_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    # Current clean-pipeline semantics.
    if "prob_true" not in result.columns:
        raise ValueError(
            "Atomic Parquet is missing 'prob_true'; cannot populate score."
        )

    result["score"] = pd.to_numeric(
        result["prob_true"],
        errors="raise",
    )
    result["threshold"] = float("nan")
    result["is_P"] = False
    result["is_x"] = predicted.isin(
        ["true", "neither"]
    ).to_numpy()
    result["belief_status"] = predicted.to_numpy()

    # Retain this convenience column when downstream/debugging code expects it.
    result["pred_label_normalized"] = predicted.to_numpy()

    return result


def _degenerate_probe_summary(
    probe_name: str,
    probe_rows: pd.DataFrame,
    message: str,
) -> dict:
    predicted = _normalize_labels_for_degenerate_rows(
        probe_rows["pred_label"]
    )
    counts = {
        str(label): int(count)
        for label, count in (
            predicted.value_counts()
            .sort_index()
            .to_dict()
            .items()
        )
    }

    num_x = int(
        predicted.isin(
            ["true", "neither"]
        ).sum()
    )

    return {
        "status": "degenerate",
        "reason": "no_predicted_true",
        "message": message,
        "threshold": None,
        "num_rows": int(len(probe_rows)),
        "num_P": 0,
        "num_x": num_x,
        "num_suspensions": int(
            predicted.eq("neither").sum()
        ),
        "num_disbeliefs": int(
            predicted.eq("false").sum()
        ),
        "predicted_label_counts": counts,
    }


def _define_sets_skipping_degenerate_probes(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if "probe" not in scores.columns:
        return define_sets(scores)

    working = scores.copy()
    degenerate_frames: list[pd.DataFrame] = []
    skipped: dict[str, dict] = {}

    enriched_usable: pd.DataFrame | None = None
    usable_summary: dict | None = None

    while not working.empty:
        try:
            enriched_usable, usable_summary = define_sets(
                working
            )
            break
        except ValueError as exc:
            message = str(exc)
            match = _DEGENERATE_NO_TRUE_PATTERN.fullmatch(
                message
            )

            if match is None:
                raise

            probe_name = match.group(
                "probe"
            ).strip()

            probe_mask = (
                working["probe"]
                .astype(str)
                .eq(probe_name)
            )

            if not probe_mask.any():
                raise RuntimeError(
                    "define_sets reported a degenerate probe that could not "
                    f"be found in the score table: {probe_name!r}."
                ) from exc

            probe_rows = working.loc[
                probe_mask
            ].copy()

            degenerate_frames.append(
                _enrich_degenerate_probe_rows(
                    probe_rows
                )
            )
            skipped[probe_name] = (
                _degenerate_probe_summary(
                    probe_name,
                    probe_rows,
                    message,
                )
            )

            print(
                f"WARNING: preserving degenerate probe {probe_name!r}: "
                "no atomic statements were predicted true; "
                "threshold is undefined and P is empty."
            )

            working = working.loc[
                ~probe_mask
            ].copy()

    if enriched_usable is not None:
        enriched_frames = [
            enriched_usable,
            *degenerate_frames,
        ]
        enriched = pd.concat(
            enriched_frames,
            ignore_index=True,
            sort=False,
        )

        # Restore deterministic long-table ordering.
        sort_columns = [
            column
            for column in (
                "probe",
                "statement_id",
            )
            if column in enriched.columns
        ]
        if sort_columns:
            enriched = (
                enriched.sort_values(
                    sort_columns,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        summary = dict(
            usable_summary
            if usable_summary is not None
            else {}
        )
    else:
        # Every requested probe was degenerate. This is a scientifically valid
        # terminal state: there are no believed P statements and therefore no
        # P-x pairs or gamma values to compute.
        enriched = pd.concat(
            degenerate_frames,
            ignore_index=True,
            sort=False,
        )

        sort_columns = [
            column
            for column in (
                "probe",
                "statement_id",
            )
            if column in enriched.columns
        ]
        if sort_columns:
            enriched = (
                enriched.sort_values(
                    sort_columns,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        summary = {
            "schema_version": 1,
            "status": "degenerate",
            "reason": "all_probes_no_predicted_true",
            "num_rows": int(len(enriched)),
            "probes": {},
        }

    summary.setdefault(
        "probes",
        {},
    )

    # Keep degenerate probes visible in the primary probe summary as well as in
    # the explicit skipped_probes field.
    for probe_name, probe_summary in skipped.items():
        summary["probes"][
            probe_name
        ] = dict(
            probe_summary
        )

    if skipped:
        summary["skipped_probes"] = skipped

    return enriched, summary

def main() -> None:
    args = parse_args()

    model_dir = (
        args.output_dir
        / args.model_name
    )
    parquet_path = (
        model_dir
        / f"{args.dataset}.parquet"
    )
    summary_path = (
        model_dir
        / f"{args.dataset}.json"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Atomic score Parquet does not exist: {parquet_path}"
        )

    scores = read_parquet_if_exists(
        parquet_path
    )

    if scores.empty:
        raise ValueError(
            f"Atomic score Parquet is empty: {parquet_path}"
        )

    if _already_defined(
        scores
    ) and not args.overwrite:
        raise FileExistsError(
            "Atomic selection columns already exist. "
            "Pass --overwrite to recompute them."
        )

    # The CLI arguments should agree with the identity stored in the Parquet.
    if "model_name" not in scores.columns:
        raise ValueError(
            "Atomic Parquet is missing 'model_name'."
        )
    if "dataset" not in scores.columns:
        raise ValueError(
            "Atomic Parquet is missing 'dataset'."
        )

    observed_models = set(
        scores["model_name"]
        .dropna()
        .astype(str)
        .unique()
    )
    observed_datasets = set(
        scores["dataset"]
        .dropna()
        .astype(str)
        .unique()
    )

    if observed_models != {
        args.model_name
    }:
        raise ValueError(
            f"Atomic Parquet model_name values {observed_models} do not "
            f"match --model_name={args.model_name!r}."
        )

    if observed_datasets != {
        args.dataset
    }:
        raise ValueError(
            f"Atomic Parquet dataset values {observed_datasets} do not "
            f"match --dataset={args.dataset!r}."
        )

    enriched, summary = _define_sets_skipping_degenerate_probes(
        scores
    )

    write_parquet_atomic(
        enriched,
        parquet_path,
    )
    _write_json_atomic(
        summary,
        summary_path,
    )

    print("=" * 72)
    print("ATOMIC SET DEFINITION COMPLETE")
    print("=" * 72)
    print(f"Model:    {args.model_name}")
    print(f"Dataset:  {args.dataset}")
    print(f"Parquet:  {parquet_path}")
    print(f"Summary:  {summary_path}")
    print(f"Rows:     {len(enriched):,}")
    print()

    for probe_name, probe_summary in (
        summary["probes"].items()
    ):
        if (
            probe_summary.get("status")
            == "degenerate"
        ):
            print(
                f"{probe_name:18s} "
                "DEGENERATE | threshold=undefined | "
                f"P={probe_summary['num_P']:,} | "
                f"x={probe_summary['num_x']:,} | "
                f"suspensions={probe_summary['num_suspensions']:,} | "
                f"disbeliefs={probe_summary['num_disbeliefs']:,}"
            )
            continue

        print(
            f"{probe_name:18s} "
            f"threshold={probe_summary['threshold']:.8f} | "
            f"P={probe_summary['num_P']:,} | "
            f"x={probe_summary['num_x']:,} | "
            f"suspensions={probe_summary['num_suspensions']:,} | "
            f"disbeliefs={probe_summary['num_disbeliefs']:,}"
        )


if __name__ == "__main__":
    main()
