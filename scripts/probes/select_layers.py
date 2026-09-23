from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stability.utils.io import write_parquet_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best layer per model/dataset/probe."
    )
    parser.add_argument(
        "--input_dir",
        default="outputs/layer_sweep",
    )
    parser.add_argument(
        "--output",
        default="outputs/layer_sweep/selected_layers.parquet",
    )
    parser.add_argument(
        "--metric",
        default="test_log_loss",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize the metric instead of minimizing it.",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help=(
            "Only select layers for model/dataset/probe groups whose terminal "
            "rows (complete or degenerate) cover every model layer."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    files = sorted(
        path
        for path in input_dir.glob("*/*.parquet")
        if path.resolve() != output_path.resolve()
    )

    if not files:
        raise FileNotFoundError(
            f"No model/dataset Parquet files found under {input_dir}."
        )

    frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_parquet(path)

        required = {
            "status",
            "model_name",
            "dataset",
            "probe",
            "layer",
            args.metric,
        }
        if args.require_complete:
            required.add("num_model_layers")

        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns {sorted(missing)}."
            )

        # For completeness checking, both successful and explicitly degenerate
        # layers count as terminal evaluations. A degenerate layer was evaluated;
        # it simply has no mathematically valid probe to score.
        if args.require_complete:
            coverage = frame[
                frame["status"].astype(str).isin(
                    ["complete", "degenerate"]
                )
            ].copy()
        
            if not coverage.empty:
                coverage_frames.append(coverage)

        frame = frame[
            frame["status"].astype(str) == "complete"
        ].copy()
        
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No completed sweep rows were found."
        )

    all_results = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )
    
    coverage_results = None

    if args.require_complete:
        if not coverage_frames:
            raise RuntimeError(
                "No terminal sweep rows were found."
            )
    
        coverage_results = pd.concat(
            coverage_frames,
            ignore_index=True,
            sort=False,
        )

    metric = pd.to_numeric(
        all_results[args.metric],
        errors="coerce",
    )
    all_results = all_results[
        metric.notna()
    ].copy()
    all_results[args.metric] = metric[
        metric.notna()
    ]

    group_columns = [
        "model_name",
        "dataset",
        "probe",
    ]

    if args.require_complete:
        complete_groups: list[tuple[str, str, str]] = []
        skipped_groups: list[tuple[str, str, str, int, int]] = []

        for group_key, group in coverage_results.groupby(
            group_columns,
            sort=True,
        ):
            num_model_layers = pd.to_numeric(
                group["num_model_layers"],
                errors="coerce",
            ).dropna().unique()

            if len(num_model_layers) != 1:
                raise ValueError(
                    "Expected exactly one num_model_layers value for "
                    f"{group_key}; found {num_model_layers.tolist()}."
                )

            expected_layers = int(num_model_layers[0])
            completed_layers = int(group["layer"].nunique())

            if completed_layers == expected_layers:
                complete_groups.append(group_key)
            else:
                skipped_groups.append(
                    (
                        str(group_key[0]),
                        str(group_key[1]),
                        str(group_key[2]),
                        completed_layers,
                        expected_layers,
                    )
                )

        if not complete_groups:
            raise RuntimeError(
                "No fully completed model/dataset/probe sweeps were found."
            )

        complete_index = pd.MultiIndex.from_tuples(
            complete_groups,
            names=group_columns,
        )
        result_index = pd.MultiIndex.from_frame(
            all_results[group_columns]
        )
        all_results = all_results[
            result_index.isin(complete_index)
        ].copy()

        if skipped_groups:
            print(
                f"Skipping {len(skipped_groups)} incomplete "
                "model/dataset/probe sweep(s):"
            )
            for model_name, dataset, probe, completed, expected in skipped_groups:
                print(
                    f"  {model_name} / {dataset} / {probe}: "
                    f"{completed}/{expected} layers complete"
                )
            print()

    if args.maximize:
        best_indices = all_results.groupby(
            group_columns,
            sort=True,
        )[args.metric].idxmax()
    else:
        best_indices = all_results.groupby(
            group_columns,
            sort=True,
        )[args.metric].idxmin()

    selected = (
        all_results.loc[best_indices]
        .sort_values(group_columns)
        .reset_index(drop=True)
    )

    preferred_columns = [
        "model_name",
        "hf_model",
        "dataset",
        "construction",
        "probe",
        "layer",
        args.metric,
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "seed",
        "num_model_layers",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in selected.columns
    ]

    selected = selected[columns].rename(
        columns={"layer": "selected_layer"}
    )

    write_parquet_atomic(
        selected,
        output_path,
    )

    print(f"Saved: {output_path}")
    print(f"Selected rows: {len(selected)}")
    print()
    print(
        selected.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()
