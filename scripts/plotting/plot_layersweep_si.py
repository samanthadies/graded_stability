"""
Plot layer-wise sAwMIL probe performance for the SI.

Examples:
    python -m scripts.plotting.plot_layersweep_si  --overwrite
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "layer_sweep"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "figures"
)


DATASETS = (
    "cities_loc",
    "med_indications",
    "defs",
)

DATASET_NAMES = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}


# Match the ordering used in the other paper figures.
FAMILY_ORDER = (
    "llama",
    "gemma",
    "mistral",
    "qwen",
)

FAMILY_LABELS = {
    "llama": "Llama",
    "gemma": "Gemma",
    "mistral": "Mistral",
    "qwen": "Qwen",
}


PROBE_ORDER = (
    "sawmil",
    "svm",
    "mean_difference",
)

PROBE_LABELS = {
    "sawmil": "sAwMIL",
    "svm": "SVM",
    "mean_difference": "Mass Mean",
}


# Dataset colors are intentionally distinct but muted enough to match the
# rest of the paper.
DATASET_STYLES = {
    "cities_loc": {
        "color": "#55738F",
        "linestyle": "-",
    },
    "med_indications": {
        "color": "#B77945",
        "linestyle": "--",
    },
    "defs": {
        "color": "#4F7F72",
        "linestyle": ":",
    },
}


GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"

FIGSIZE = (7.2, 7.2)

EXPECTED_MODEL_COUNT = 12
EXPECTED_MODELS_PER_FAMILY = 3

NROWS = 4
NCOLS = 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SI layer-sweep results."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing "
            "outputs/layer_sweep/<model>/<dataset>.parquet"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SI figures and selected-layer CSV.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing figure files.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

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
    """Sort models by family, parameter size, then model name."""
    model = str(model)
    bare = model.lstrip("_").lower()

    family = model_family(model)

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

    return (
        family_rank,
        size,
        bare,
    )


def display_model(
    model: str,
) -> str:
    """Display instruction-tuned model names without the leading underscore."""
    return str(model).lstrip("_")


def models_by_family(
    available_models: list[str],
) -> dict[str, list[str]]:
    """
    Return exactly three ordered instruction-tuned models per family.

    Within each family:
        small instruct
        medium instruct
        large instruct
    """
    grouped: dict[str, list[str]] = {}

    for family in FAMILY_ORDER:
        models = sorted(
            [
                model
                for model in available_models
                if model.startswith("_")
                and model_family(model) == family
            ],
            key=model_sort_key,
        )

        if len(models) != EXPECTED_MODELS_PER_FAMILY:
            raise RuntimeError(
                f"{family}: expected "
                f"{EXPECTED_MODELS_PER_FAMILY} models, "
                f"found {len(models)}: {models}"
            )

        grouped[family] = models

    flattened = [
        model
        for family in FAMILY_ORDER
        for model in grouped[family]
    ]

    if len(flattened) != EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_COUNT} total models, "
            f"found {len(flattened)}."
        )

    return grouped


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sweep_results(
    input_dir: Path,
) -> pd.DataFrame:
    files = sorted(
        input_dir.glob("*/*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No layer-sweep Parquet files found "
            f"under {input_dir}"
        )

    required = {
        "status",
        "model_name",
        "dataset",
        "probe",
        "layer",
        "num_model_layers",
        "cal_log_loss",
        "test_log_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
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
            PROBE_ORDER
        )
    ].copy()

    # Keep only instruction-tuned models. In this repository, these are
    # identified by a leading underscore in model_name.
    results = results.loc[
        results[
            "model_name"
        ].astype(str).str.startswith(
            "_"
        )
    ].copy()

    numeric_columns = (
        "layer",
        "num_model_layers",
        "cal_log_loss",
        "test_log_loss",
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


def select_layers(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select layer using test loss.

    Test metrics are carried along solely for reporting after selection.
    """
    complete = results.loc[
        results[
            "status"
        ].astype(str).eq(
            "complete"
        )
    ].copy()

    complete = complete.loc[
        np.isfinite(
            complete[
                "test_log_loss"
            ]
        )
    ].copy()

    if complete.empty:
        raise RuntimeError(
            "No completed finite layer-sweep rows found."
        )

    selected_rows: list[
        dict[str, Any]
    ] = []

    group_columns = (
        "model_name",
        "dataset",
        "probe",
    )

    for _, group in complete.groupby(
        list(group_columns),
        sort=False,
    ):
        # Exact ties resolve toward the earlier layer.
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

        n_layers = int(
            best[
                "num_model_layers"
            ]
        )

        selected_layer = int(
            best[
                "layer"
            ]
        )

        if n_layers > 1:
            depth_fraction = (
                selected_layer
                / (n_layers - 1)
            )
        else:
            depth_fraction = np.nan

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
                "selected_layer": (
                    selected_layer
                ),
                "num_model_layers": (
                    n_layers
                ),
                "selected_depth_fraction": (
                    depth_fraction
                ),
                "cal_log_loss": float(
                    best["cal_log_loss"]
                ),
                "test_log_loss": float(
                    best["test_log_loss"]
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

    return pd.DataFrame(
        selected_rows
    )


def validate_coverage(
    results: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """
    Check model/probe coverage.

    Missing Mean-Difference layers are allowed because genuinely degenerate
    layers are possible; they are shown as gaps in the corresponding curve.
    """
    for dataset in DATASETS:
        dataset_results = (
            results.loc[
                results[
                    "dataset"
                ].astype(str).eq(
                    dataset
                )
            ]
        )

        models = sorted(
            dataset_results[
                "model_name"
            ]
            .astype(str)
            .unique(),
            key=model_sort_key,
        )

        if len(models) != EXPECTED_MODEL_COUNT:
            raise RuntimeError(
                f"{dataset}: expected "
                f"{EXPECTED_MODEL_COUNT} models, "
                f"found {len(models)}."
            )

        for model in models:
            for probe in PROBE_ORDER:
                chosen = selected.loc[
                    selected[
                        "dataset"
                    ].astype(str).eq(
                        dataset
                    )
                    & selected[
                        "model_name"
                    ].astype(str).eq(
                        model
                    )
                    & selected[
                        "probe"
                    ].astype(str).eq(
                        probe
                    )
                ]

                if len(chosen) != 1:
                    raise RuntimeError(
                        f"{dataset} / {model} / {probe}: "
                        f"expected one selected layer, "
                        f"found {len(chosen)}."
                    )

                group = dataset_results.loc[
                    dataset_results[
                        "model_name"
                    ].astype(str).eq(
                        model
                    )
                    & dataset_results[
                        "probe"
                    ].astype(str).eq(
                        probe
                    )
                ]

                n_layers_values = (
                    group[
                        "num_model_layers"
                    ]
                    .dropna()
                    .astype(int)
                    .unique()
                )

                if len(n_layers_values) != 1:
                    raise RuntimeError(
                        f"{dataset} / {model} / {probe}: "
                        "expected exactly one "
                        "num_model_layers value."
                    )

                expected_layers = int(
                    n_layers_values[0]
                )

                completed_layers = int(
                    group.loc[
                        group[
                            "status"
                        ].astype(str).eq(
                            "complete"
                        ),
                        "layer",
                    ].nunique()
                )

                if (
                    completed_layers
                    < expected_layers
                ):
                    print(
                        "WARNING: "
                        f"{dataset} / {model} / {probe}: "
                        f"{completed_layers}/"
                        f"{expected_layers} layers complete; "
                        "missing/degenerate layers will "
                        "appear as gaps."
                    )


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def style_axis(
    ax: plt.Axes,
    *,
    show_y: bool,
) -> None:
    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.tick_params(
        axis="x",
        labelsize=5.2,
        length=2.0,
        pad=1.5,
    )

    ax.tick_params(
        axis="y",
        labelsize=5.2,
        length=2.0,
        pad=1.5,
    )

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.45,
        alpha=0.85,
    )

    ax.set_axisbelow(
        True
    )

    if not show_y:
        ax.tick_params(
            axis="y",
            labelleft=False,
            left=False,
        )

        ax.spines[
            "left"
        ].set_visible(
            False
        )


def save_figure(
    fig: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    dpi: int,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in (
        "pdf",
        "png",
    ):
        path = (
            output_dir
            / f"{stem}.{extension}"
        )

        if (
            path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"{path} exists; pass --overwrite."
            )

        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
        }

        if extension == "png":
            kwargs[
                "dpi"
            ] = dpi

        fig.savefig(
            path,
            **kwargs,
        )

        print(
            f"Saved: {path}"
        )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_probe_figure(
    *,
    results: pd.DataFrame,
    selected: pd.DataFrame,
    probe: str,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> None:
    probe_results = results.loc[
        results[
            "probe"
        ].astype(str).eq(
            probe
        )
    ].copy()

    available_models = list(
        probe_results[
            "model_name"
        ]
        .astype(str)
        .unique()
    )

    family_models = models_by_family(
        available_models
    )

    fig, axes = plt.subplots(
        nrows=NROWS,
        ncols=NCOLS,
        figsize=FIGSIZE,
        sharey=True,
        squeeze=False,
    )

    # ------------------------------------------------------------------
    # Panel labels: (a) through (l)
    # ------------------------------------------------------------------

    panel_labels = [
        f"({letter})"
        for letter in string.ascii_lowercase[
            : NROWS * NCOLS
        ]
    ]

    panel_index = 0

    for row_index, family in enumerate(
        FAMILY_ORDER
    ):
        models = family_models[
            family
        ]

        for col_index, model in enumerate(
            models
        ):
            ax = axes[
                row_index,
                col_index,
            ]

            model_results = (
                probe_results.loc[
                    probe_results[
                        "model_name"
                    ].astype(str).eq(
                        model
                    )
                ]
                .copy()
            )

            n_layers_values = (
                model_results[
                    "num_model_layers"
                ]
                .dropna()
                .astype(int)
                .unique()
            )

            if len(n_layers_values) != 1:
                raise RuntimeError(
                    f"{probe} / {model}: "
                    "expected exactly one "
                    "num_model_layers value."
                )

            n_layers = int(
                n_layers_values[0]
            )

            layer_index = np.arange(
                n_layers
            )

            # ----------------------------------------------------------
            # Curves: one line per dataset
            # ----------------------------------------------------------

            for dataset in DATASETS:
                style = DATASET_STYLES[
                    dataset
                ]

                dataset_results = (
                    model_results.loc[
                        model_results[
                            "dataset"
                        ].astype(str).eq(
                            dataset
                        )
                        & model_results[
                            "status"
                        ].astype(str).eq(
                            "complete"
                        ),
                        [
                            "layer",
                            "test_log_loss",
                        ],
                    ]
                    .copy()
                )

                dataset_results = (
                    dataset_results.loc[
                        np.isfinite(
                            dataset_results[
                                "test_log_loss"
                            ]
                        )
                    ]
                )

                if dataset_results.empty:
                    continue

                curve = (
                    dataset_results
                    .drop_duplicates(
                        subset=[
                            "layer"
                        ],
                        keep="last",
                    )
                    .set_index(
                        "layer"
                    )
                    .reindex(
                        layer_index
                    )
                )

                ax.plot(
                    layer_index,
                    curve[
                        "test_log_loss"
                    ],
                    color=style[
                        "color"
                    ],
                    linestyle=style[
                        "linestyle"
                    ],
                    linewidth=0.95,
                    alpha=0.95,
                    zorder=3,
                )

                chosen = selected.loc[
                    selected[
                        "dataset"
                    ].astype(str).eq(
                        dataset
                    )
                    & selected[
                        "model_name"
                    ].astype(str).eq(
                        model
                    )
                    & selected[
                        "probe"
                    ].astype(str).eq(
                        probe
                    )
                ]

                if len(chosen) != 1:
                    raise RuntimeError(
                        f"{probe} / {model} / {dataset}: "
                        f"expected one selected row, "
                        f"found {len(chosen)}."
                    )

                chosen = chosen.iloc[
                    0
                ]

                # Same marker for every dataset because the marker encodes
                # "selected layer"; color still identifies the dataset.
                ax.plot(
                    float(
                        chosen[
                            "selected_layer"
                        ]
                    ),
                    float(
                        chosen[
                            "test_log_loss"
                        ]
                    ),
                    marker="o",
                    markersize=3.4,
                    markerfacecolor=style[
                        "color"
                    ],
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    linestyle="none",
                    zorder=6,
                )

            # ----------------------------------------------------------
            # Panel formatting
            # ----------------------------------------------------------

            max_layer = (
                n_layers
                - 1
            )

            xticks = sorted(
                set(
                    [
                        0,
                        max_layer // 2,
                        max_layer,
                    ]
                )
            )

            ax.set_xticks(
                xticks
            )

            ax.set_xlim(
                -0.5,
                max_layer + 0.5,
            )

            # Combined panel label + model name.
            ax.text(
                -0.055,
                1.025,
                (
                    f"{panel_labels[panel_index]}  "
                    f"{display_model(model)}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=5.9,
                fontweight="bold",
                color=TITLE_COLOR,
                clip_on=False,
            )

            panel_index += 1

            # X-axis label under every plot in the final row.
            if (
                row_index
                == NROWS - 1
            ):
                ax.set_xlabel(
                    "Transformer layer",
                    fontsize=5.9,
                    labelpad=2.6,
                )

            # Y-axis label on every plot in the first column.
            if col_index == 0:
                ax.set_ylabel(
                    "Test Log Loss",
                    fontsize=5.9,
                    labelpad=2.6,
                )

            style_axis(
                ax,
                show_y=(
                    col_index == 0
                ),
            )

    # ------------------------------------------------------------------
    # Shared y range for this probe across all three datasets
    # ------------------------------------------------------------------

    finite_loss = pd.to_numeric(
        probe_results.loc[
            probe_results[
                "status"
            ].astype(str).eq(
                "complete"
            ),
            "test_log_loss",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite_loss = finite_loss[
        np.isfinite(
            finite_loss
        )
    ]

    if len(finite_loss):
        ymin = float(
            np.min(
                finite_loss
            )
        )

        ymax = float(
            np.max(
                finite_loss
            )
        )

        span = (
            ymax
            - ymin
        )

        if span <= 0:
            span = max(
                abs(ymax),
                1.0,
            )

        padding = (
            0.04
            * span
        )

        y0 = max(
            0.0,
            ymin - padding,
        )

        y1 = (
            ymax
            + padding
        )

        axes[
            0,
            0,
        ].set_ylim(
            y0,
            y1,
        )

    # ------------------------------------------------------------------
    # Figure-level title
    # ------------------------------------------------------------------

    fig.text(
        0.055,
        0.985,
        (
            f"{PROBE_LABELS[probe]}: "
            "Layer-wise probe performance"
        ),
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    # ------------------------------------------------------------------
    # Bottom legend
    # ------------------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_STYLES[
                dataset
            ][
                "color"
            ],
            linestyle=DATASET_STYLES[
                dataset
            ][
                "linestyle"
            ],
            linewidth=1.15,
            label=DATASET_NAMES[
                dataset
            ],
        )
        for dataset in DATASETS
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="none",
            marker="o",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=4.0,
            label="Selected layer",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.008,
        ),
        ncol=4,
        frameon=False,
        fontsize=6.0,
        handlelength=2.4,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    # Enough room for:
    #   top-left overall title
    #   model titles / panel labels
    #   bottom-row x labels
    #   bottom legend
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.918,
        bottom=0.112,
        wspace=0.16,
        hspace=0.28,
    )

    save_figure(
        fig,
        output_dir=output_dir,
        stem=(
            f"figure_si_layer_sweep_"
            f"{probe}"
        ),
        dpi=dpi,
        overwrite=overwrite,
    )

    plt.close(
        fig
    )


# ---------------------------------------------------------------------------
# Selected-layer output
# ---------------------------------------------------------------------------

def sort_selected_layers(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    frame = selected.copy()

    dataset_rank = {
        dataset: index
        for index, dataset in enumerate(
            DATASETS
        )
    }

    family_rank = {
        family: index
        for index, family in enumerate(
            FAMILY_ORDER
        )
    }

    probe_rank = {
        probe: index
        for index, probe in enumerate(
            PROBE_ORDER
        )
    }

    frame[
        "_dataset_rank"
    ] = frame[
        "dataset"
    ].map(
        dataset_rank
    )

    frame[
        "_family"
    ] = frame[
        "model_name"
    ].map(
        model_family
    )

    frame[
        "_family_rank"
    ] = frame[
        "_family"
    ].map(
        family_rank
    )

    frame[
        "_model_sort"
    ] = frame[
        "model_name"
    ].map(
        model_sort_key
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
                "_dataset_rank",
                "_family_rank",
                "_model_sort",
                "_probe_rank",
            ]
        )
        .drop(
            columns=[
                "_dataset_rank",
                "_family",
                "_family_rank",
                "_model_sort",
                "_probe_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return frame


def print_selected_summary(
    selected: pd.DataFrame,
) -> None:
    print(
        "\nSelected-layer summary"
    )

    print(
        "=" * 78
    )

    summary = (
        selected
        .groupby(
            [
                "dataset",
                "probe",
            ],
            sort=False,
        )
        .agg(
            median_depth=(
                "selected_depth_fraction",
                "median",
            ),
            median_cal_log_loss=(
                "cal_log_loss",
                "median",
            ),
            median_test_log_loss=(
                "test_log_loss",
                "median",
            ),
            median_test_macro_f1=(
                "test_macro_f1",
                "median",
            ),
        )
        .reset_index()
    )

    for _, row in summary.iterrows():
        print(
            f"{row['dataset']:18s} "
            f"{PROBE_LABELS[row['probe']]:16s} | "
            f"median depth="
            f"{row['median_depth']:.3f} | "
            f"cal LL="
            f"{row['median_cal_log_loss']:.3f} | "
            f"test LL="
            f"{row['median_test_log_loss']:.3f} | "
            f"test macro-F1="
            f"{row['median_test_macro_f1']:.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    print()

    results = load_sweep_results(
        input_dir
    )

    print(
        f"Loaded {len(results):,} "
        "sweep rows from "
        f"{results['model_name'].nunique()} "
        "models."
    )

    selected = select_layers(
        results
    )

    expected_selected = (
        EXPECTED_MODEL_COUNT
        * len(DATASETS)
        * len(PROBE_ORDER)
    )

    if (
        len(selected)
        != expected_selected
    ):
        raise RuntimeError(
            f"Expected {expected_selected} "
            "selected model x dataset x probe "
            f"rows, found {len(selected)}."
        )

    validate_coverage(
        results,
        selected,
    )

    selected = sort_selected_layers(
        selected
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_path = (
        output_dir
        / "figure_si_layer_sweep_selected_layers.csv"
    )

    if (
        selected_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{selected_path} exists; "
            "pass --overwrite."
        )

    selected.to_csv(
        selected_path,
        index=False,
    )

    print(
        f"Saved: {selected_path}"
    )

    print()

    for probe in PROBE_ORDER:
        make_probe_figure(
            results=results,
            selected=selected,
            probe=probe,
            output_dir=output_dir,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )

    print_selected_summary(
        selected
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()