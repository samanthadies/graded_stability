from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "layer_sweep"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "layer_selection"
)


DATASETS = (
    "cities_loc",
    "med_indications",
    "defs",
)

DATASET_LABELS = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}

PROBES = (
    "sawmil",
    "svm",
    "mean_difference",
)

PROBE_LABELS = {
    "sawmil": "sAwMIL",
    "svm": "SVM",
    "mean_difference": "Mass Mean",
}

FAMILY_ORDER = (
    "llama",
    "gemma",
    "mistral",
    "qwen",
)

EXPECTED_MODELS = 24
EXPECTED_SELECTED_ROWS = (
    EXPECTED_MODELS
    * len(DATASETS)
    * len(PROBES)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select best atomic probe layers using "
            "minimum test-set log loss."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing analysis outputs.",
    )

    return parser.parse_args()


def model_family(
    model: str,
) -> str:
    bare = str(model).lstrip("_").lower()

    if bare.startswith("llama"):
        return "llama"

    if bare.startswith("gemma"):
        return "gemma"

    if bare.startswith("mistral"):
        return "mistral"

    if bare.startswith("qwen"):
        return "qwen"

    return bare.split("-")[0]


def model_sort_key(
    model: str,
) -> tuple[Any, ...]:

    model = str(model)
    bare = model.lstrip("_").lower()

    family = model_family(
        model
    )

    try:
        family_rank = FAMILY_ORDER.index(
            family
        )
    except ValueError:
        family_rank = 99

    sizes = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])"
            r"(\d+(?:\.\d+)?)\s*[Bb]"
            r"(?![A-Za-z])",
            bare,
        )
    ]

    size = (
        max(sizes)
        if sizes
        else float("inf")
    )

    instruction_rank = (
        1
        if model.startswith("_")
        else 0
    )

    return (
        family_rank,
        size,
        instruction_rank,
        bare,
    )


def display_model(
    model: str,
) -> str:

    model = str(model)

    if model.startswith("_"):
        return f"{model[1:]} (i)"

    return model


def load_sweep_results(
    input_dir: Path,
) -> pd.DataFrame:
    files = sorted(
        input_dir.glob("*/*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No layer-sweep parquet files "
            f"found under {input_dir}"
        )

    required = {
        "status",
        "model_name",
        "dataset",
        "probe",
        "layer",
        "num_model_layers",
        "test_log_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "cal_log_loss",
    }

    frames: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_parquet(
            path
        )

        missing = (
            required
            - set(frame.columns)
        )

        if missing:
            raise ValueError(
                f"{path}: missing required columns "
                f"{sorted(missing)}"
            )

        frames.append(
            frame
        )

    results = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    results = results.loc[
        results[
            "dataset"
        ].astype(str).isin(
            DATASETS
        )
        & results[
            "probe"
        ].astype(str).isin(
            PROBES
        )
    ].copy()

    numeric_columns = (
        "layer",
        "num_model_layers",
        "test_log_loss",
        "cal_log_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    )

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    return results


def select_best_layers(
    results: pd.DataFrame,
) -> pd.DataFrame:

    complete = results.loc[
        results[
            "status"
        ].astype(str).eq(
            "complete"
        )
        & np.isfinite(
            results[
                "test_log_loss"
            ]
        )
    ].copy()

    if complete.empty:
        raise RuntimeError(
            "No completed rows with finite test log loss."
        )

    selected_rows: list[
        dict[str, Any]
    ] = []

    grouping = (
        "model_name",
        "dataset",
        "probe",
    )

    for _, group in complete.groupby(
        list(grouping),
        sort=False,
    ):
        # Sort first so idxmin resolves exact ties
        # toward the earlier layer.
        group = (
            group
            .sort_values("layer")
            .copy()
        )

        best_index = (
            group[
                "test_log_loss"
            ]
            .idxmin()
        )

        best = group.loc[
            best_index
        ]

        selected_rows.append(
            {
                "model_name": str(
                    best["model_name"]
                ),
                "dataset": str(
                    best["dataset"]
                ),
                "probe": str(
                    best["probe"]
                ),
                "selected_layer": int(
                    best["layer"]
                ),
                "num_model_layers": int(
                    best["num_model_layers"]
                ),
                "test_log_loss": float(
                    best["test_log_loss"]
                ),
                "cal_log_loss": float(
                    best["cal_log_loss"]
                ),
                "test_accuracy": float(
                    best["test_accuracy"]
                ),
                "test_balanced_accuracy": float(
                    best[
                        "test_balanced_accuracy"
                    ]
                ),
                "test_macro_f1": float(
                    best["test_macro_f1"]
                ),
                "test_weighted_f1": float(
                    best[
                        "test_weighted_f1"
                    ]
                ),
            }
        )

    selected = pd.DataFrame(
        selected_rows
    )

    if len(selected) != EXPECTED_SELECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED_ROWS} "
            "model x dataset x probe selections, "
            f"found {len(selected)}."
        )

    models = (
        selected[
            "model_name"
        ]
        .astype(str)
        .unique()
    )

    if len(models) != EXPECTED_MODELS:
        raise RuntimeError(
            f"Expected {EXPECTED_MODELS} models, "
            f"found {len(models)}."
        )

    return selected


def sort_long_results(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    frame = selected.copy()

    dataset_rank = {
        dataset: index
        for index, dataset
        in enumerate(DATASETS)
    }

    probe_rank = {
        probe: index
        for index, probe
        in enumerate(PROBES)
    }

    frame[
        "_model_sort"
    ] = frame[
        "model_name"
    ].map(
        model_sort_key
    )

    frame[
        "_dataset_rank"
    ] = frame[
        "dataset"
    ].map(
        dataset_rank
    )

    frame[
        "_probe_rank"
    ] = frame[
        "probe"
    ].map(
        probe_rank
    )

    frame = (
        frame
        .sort_values(
            [
                "_model_sort",
                "_dataset_rank",
                "_probe_rank",
            ]
        )
        .drop(
            columns=[
                "_model_sort",
                "_dataset_rank",
                "_probe_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return frame


def make_wide_table(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    wide = selected.pivot(
        index="model_name",
        columns=[
            "dataset",
            "probe",
        ],
        values="selected_layer",
    )

    expected_columns = pd.MultiIndex.from_product(
        [
            DATASETS,
            PROBES,
        ],
        names=[
            "dataset",
            "probe",
        ],
    )

    wide = wide.reindex(
        columns=expected_columns
    )

    models = sorted(
        wide.index.astype(str),
        key=model_sort_key,
    )

    wide = wide.reindex(
        models
    )

    # Give the CSV readable flat column names.
    flat = pd.DataFrame(
        index=wide.index
    )

    flat[
        "Model"
    ] = [
        display_model(model)
        for model in wide.index
    ]

    for dataset in DATASETS:
        for probe in PROBES:
            column_name = (
                f"{DATASET_LABELS[dataset]} "
                f"{PROBE_LABELS[probe]}"
            )

            flat[
                column_name
            ] = (
                wide[
                    (
                        dataset,
                        probe,
                    )
                ]
                .astype(int)
                .to_numpy()
            )

    return flat.reset_index(
        drop=True
    )

def latex_escape_model(
    model: str,
) -> str:
    return (
        display_model(model)
        .replace("_", r"\_")
    )


def make_latex_table(
    selected: pd.DataFrame,
) -> str:
    wide = selected.pivot(
        index="model_name",
        columns=[
            "dataset",
            "probe",
        ],
        values="selected_layer",
    )

    expected_columns = pd.MultiIndex.from_product(
        [
            DATASETS,
            PROBES,
        ],
        names=[
            "dataset",
            "probe",
        ],
    )

    wide = wide.reindex(
        columns=expected_columns
    )

    models = sorted(
        wide.index.astype(str),
        key=model_sort_key,
    )

    wide = wide.reindex(
        models
    )

    lines: list[str] = []

    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccccccc}",
            r"\toprule",
            (
                r"\textbf{Model} & "
                r"\multicolumn{9}{c}{"
                r"\textbf{Best Layer $\ell^{\star}$}} \\"
            ),
            r"\cmidrule(lr){2-10}",
            (
                r"& \multicolumn{3}{c}{"
                r"\textbf{City Locations}} "
                r"& \multicolumn{3}{c}{"
                r"\textbf{Medical Indications}} "
                r"& \multicolumn{3}{c}{"
                r"\textbf{Word Definitions}} \\"
            ),
            (
                r"\cmidrule(lr){2-4}"
                r"\cmidrule(lr){5-7}"
                r"\cmidrule(lr){8-10}"
            ),
            (
                r"& \textbf{sAwMIL} "
                r"& \textbf{SVM} "
                r"& \textbf{Mass Mean} "
                r"& \textbf{sAwMIL} "
                r"& \textbf{SVM} "
                r"& \textbf{Mass Mean} "
                r"& \textbf{sAwMIL} "
                r"& \textbf{SVM} "
                r"& \textbf{Mass Mean} \\"
            ),
            r"\midrule",
        ]
    )

    previous_family: str | None = None

    for model in models:
        family = model_family(
            model
        )

        if (
            previous_family is not None
            and family != previous_family
        ):
            lines.append(
                r"\addlinespace[2pt]"
            )

        values: list[str] = []

        for dataset in DATASETS:
            for probe in PROBES:
                value = wide.loc[
                    model,
                    (
                        dataset,
                        probe,
                    ),
                ]

                if pd.isna(
                    value
                ):
                    values.append(
                        "--"
                    )
                else:
                    values.append(
                        str(
                            int(value)
                        )
                    )

        lines.append(
            (
                f"{latex_escape_model(model)} & "
                + " & ".join(values)
                + r" \\"
            )
        )

        previous_family = family

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{",
            (
                r"\textbf{Selected probe layers.} "
                r"Best transformer layer $\ell^{\star}$ "
                r"for each LLM, dataset, and probe, "
                r"selected by minimizing atomic test-set "
                r"multiclass log loss across layers."
            ),
            r"}",
            r"\label{tab:si:selected_layers}",
            r"\end{table*}",
        ]
    )

    return "\n".join(
        lines
    ) + "\n"

def check_output(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if (
        path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"{path} exists; pass --overwrite."
        )


def write_outputs(
    *,
    selected: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_path = (
        output_dir
        / "selected_layers.parquet"
    )

    wide_path = (
        output_dir
        / "selected_layers_wide.csv"
    )

    latex_path = (
        output_dir
        / "selected_layers_table.tex"
    )

    for path in (
        long_path,
        wide_path,
        latex_path,
    ):
        check_output(
            path,
            overwrite=overwrite,
        )

    selected.to_parquet(
        long_path,
        index=False,
    )

    wide = make_wide_table(
        selected
    )

    wide.to_csv(
        wide_path,
        index=False,
    )

    latex = make_latex_table(
        selected
    )

    latex_path.write_text(
        latex,
        encoding="utf-8",
    )

    print(
        f"Wrote: {long_path}"
    )

    print(
        f"Wrote: {wide_path}"
    )

    print(
        f"Wrote: {latex_path}"
    )


def print_summary(
    selected: pd.DataFrame,
) -> None:
    print(
        "\nSelected layers "
        "(minimum test log loss)"
    )

    print(
        "=" * 78
    )

    for dataset in DATASETS:
        print(
            f"\n{DATASET_LABELS[dataset]}"
        )

        subset = selected.loc[
            selected[
                "dataset"
            ].eq(
                dataset
            )
        ]

        for probe in PROBES:
            probe_rows = subset.loc[
                subset[
                    "probe"
                ].eq(
                    probe
                )
            ]

            layers = (
                probe_rows[
                    "selected_layer"
                ]
                .astype(int)
                .to_numpy()
            )

            print(
                f"  {PROBE_LABELS[probe]:10s} "
                f"median={np.median(layers):.1f} | "
                f"min={layers.min():d} | "
                f"max={layers.max():d}"
            )


def main() -> None:
    args = parse_args()

    input_dir = (
        args.input_dir
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    print(
        f"Input:  {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    results = load_sweep_results(
        input_dir
    )

    selected = select_best_layers(
        results
    )

    selected = sort_long_results(
        selected
    )

    write_outputs(
        selected=selected,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    print_summary(
        selected
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()