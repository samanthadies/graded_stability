from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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

ESTIMATOR = "direct"

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

# Match the palette used in the other manuscript/SI figures.
FAMILY_COLORS = {
    "llama": "#A63D4D",
    "gemma": "#3F6699",
    "mistral": "#C08A1D",
    "qwen": "#3E8064",
}

EXPECTED_MODELS = (
    "_llama-3.2-3b",
    "_llama-3.1-8b",
    "_llama-3.1-70b",
    "_gemma-7b",
    "_gemma-2-9b",
    "_gemma-2-27b",
    "_mistral-7b",
    "_mistral-12b",
    "_mistral-3.1-24b",
    "_qwen-2.5-7b",
    "_qwen-2.5-14b",
    "_qwen-2.5-72b",
)

# Text-decoder layer counts used elsewhere in the plotting code.
MODEL_DECODER_LAYERS = {
    "llama-3.2-3b": 28,
    "llama-3.1-8b": 32,
    "llama-3.1-70b": 80,
    "gemma-7b": 28,
    "gemma-2-9b": 42,
    "gemma-2-27b": 46,
    "mistral-7b": 32,
    "mistral-12b": 40,
    "mistral-3.1-24b": 40,
    "qwen-2.5-7b": 28,
    "qwen-2.5-14b": 48,
    "qwen-2.5-72b": 80,
}

GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"
ERROR_COLOR = "#2f2f2f"

FIGSIZE = (7.2, 3.75)
BAR_WIDTH = 0.72
OUTPUT_STEM = "figure_si_model_scale"


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}; "
            f"available={frame.columns.tolist()}"
        )


def display_model(model: str) -> str:
    return str(model).lstrip("_")


def model_family(model: str) -> str:
    bare = display_model(model).lower()
    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family
    raise ValueError(
        f"Unrecognized model family for {model!r}; "
        f"expected one of {list(FAMILY_ORDER)}."
    )


def nominal_parameter_count_b(model: str) -> float:
    """Parse the nominal billions count from the checkpoint name."""
    bare = display_model(model).lower()
    values = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            bare,
        )
    ]
    if not values:
        raise ValueError(f"Could not parse parameter count from {model!r}.")
    return max(values)


def decoder_layers(model: str) -> int:
    bare = display_model(model)
    if bare not in MODEL_DECODER_LAYERS:
        raise ValueError(f"Missing decoder-layer metadata for {model!r}.")
    return int(MODEL_DECODER_LAYERS[bare])


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=5.5)
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.45,
        alpha=0.85,
    )
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure,
    *,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{OUTPUT_STEM}.pdf"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    log.info("Wrote %s", path)
    return path


def summarize_statement_level(
    statement_level: pd.DataFrame,
    *,
    probe: str,
) -> pd.DataFrame:
    """
    Compute one row per model/domain with mean gamma and SE across propositions.
    """
    require_columns(
        statement_level,
        ("model", "dataset", "probe", "estimator", "gamma"),
        source="statement_level.parquet",
    )

    expected = set(EXPECTED_MODELS)
    frame = statement_level.loc[
        statement_level["probe"].astype(str).eq(probe)
        & statement_level["dataset"].astype(str).isin(DATASETS)
        & statement_level["estimator"].astype(str).eq(ESTIMATOR)
        & statement_level["model"].astype(str).isin(expected)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            f"No instruction-tuned Direct statement-level rows for probe={probe!r}."
        )

    frame["gamma"] = pd.to_numeric(frame["gamma"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["gamma"].to_numpy(dtype=float))].copy()

    def se(values: pd.Series) -> float:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return np.nan
        return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    summary = (
        frame.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_gamma=("gamma", "mean"),
            n=("gamma", "count"),
            se_gamma=("gamma", se),
        )
    )

    summary["family"] = summary["model"].astype(str).map(model_family)
    summary["params_b"] = summary["model"].astype(str).map(nominal_parameter_count_b)
    summary["layers"] = summary["model"].astype(str).map(decoder_layers)

    return summary


def ordered_panel(
    summary: pd.DataFrame,
    *,
    dataset: str,
    scale: str,
) -> pd.DataFrame:
    panel = summary.loc[
        summary["dataset"].astype(str).eq(dataset)
    ].copy()

    # Complete the expected model set so accidental missing rows are visible.
    expected = pd.DataFrame({"model": list(EXPECTED_MODELS)})
    expected["family"] = expected["model"].map(model_family)
    expected["params_b"] = expected["model"].map(nominal_parameter_count_b)
    expected["layers"] = expected["model"].map(decoder_layers)

    panel = expected.merge(
        panel[["model", "mean_gamma", "se_gamma", "n"]],
        on="model",
        how="left",
        validate="one_to_one",
    )

    if scale == "params_b":
        primary = "params_b"
    elif scale == "layers":
        primary = "layers"
    else:
        raise ValueError(f"Unknown scale {scale!r}.")

    # User-requested tie-break: within equal scale, smaller mean gamma first.
    # Any missing mean is pushed to the end of its tied block.
    panel["_mean_sort"] = pd.to_numeric(panel["mean_gamma"], errors="coerce").fillna(np.inf)
    panel["_family_sort"] = panel["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})

    panel = panel.sort_values(
        [primary, "_mean_sort", "_family_sort", "model"],
        kind="stable",
    ).reset_index(drop=True)

    return panel


def format_param_tick(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value), atol=1e-9):
        return str(int(round(value)))
    return f"{value:g}"


def plot_bar_panel(
    ax: plt.Axes,
    *,
    panel: pd.DataFrame,
    scale: str,
    show_y_labels: bool,
) -> None:
    x = np.arange(1, len(panel) + 1, dtype=float)
    means = pd.to_numeric(panel["mean_gamma"], errors="coerce").to_numpy(dtype=float)
    ses = pd.to_numeric(panel["se_gamma"], errors="coerce").to_numpy(dtype=float)
    colors = [FAMILY_COLORS[family] for family in panel["family"].astype(str)]

    ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="none",
        zorder=3,
    )

    valid_err = np.isfinite(means) & np.isfinite(ses)
    if valid_err.any():
        ax.errorbar(
            x[valid_err],
            means[valid_err],
            yerr=ses[valid_err],
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            zorder=5,
        )

    if scale == "params_b":
        tick_labels = [format_param_tick(v) for v in panel["params_b"].to_numpy(dtype=float)]
        xlabel = "Parameters (B)"
    elif scale == "layers":
        tick_labels = [str(int(v)) for v in panel["layers"].to_numpy(dtype=int)]
        xlabel = "Layers"
    else:
        raise ValueError(f"Unknown scale {scale!r}.")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_xlabel(xlabel, fontsize=6.2, labelpad=2.5)
    ax.set_xlim(0.35, len(panel) + 0.65)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    style_axis(ax)

    if not show_y_labels:
        ax.tick_params(axis="y", labelleft=False)
        ax.spines["left"].set_visible(False)


def make_figure(
    *,
    summary: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=FIGSIZE,
        sharey=True,
    )
    axes = np.asarray(axes)

    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.90,
        bottom=0.15,
        wspace=0.15,
        hspace=0.72,
    )

    panel_labels = {
        (0, 0): "(a)",
        (0, 1): "(b)",
        (0, 2): "(c)",
        (1, 0): "(d)",
        (1, 1): "(e)",
        (1, 2): "(f)",
    }

    for row_idx, scale in enumerate(("params_b", "layers")):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            panel = ordered_panel(
                summary,
                dataset=dataset,
                scale=scale,
            )

            plot_bar_panel(
                ax,
                panel=panel,
                scale=scale,
                show_y_labels=(col_idx == 0),
            )

            if row_idx == 0:
                ax.set_title(
                    DATASET_NAMES[dataset],
                    loc="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color=TITLE_COLOR,
                    y=1.11,
                    pad=0,
                )

            ax.text(
                0.0,
                1.025,
                panel_labels[(row_idx, col_idx)],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=TITLE_COLOR,
            )

    legend_handles = [
        Patch(
            facecolor=FAMILY_COLORS[family],
            edgecolor="none",
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.015),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    fig.canvas.draw()
    first_row_pos = axes[0, 0].get_position()
    second_row_pos = axes[1, 0].get_position()

    fig.text(
        0.035,
        first_row_pos.y1 + 0.08,
        "Parameter count",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.text(
        0.035,
        second_row_pos.y1 + 0.08,
        "Layers",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    fig.text(
        0.045,
        (first_row_pos.y0 + first_row_pos.y1) / 2.0,
        r"Mean graded stability $\bar{\gamma}(P)$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )
    fig.text(
        0.045,
        (second_row_pos.y0 + second_row_pos.y1) / 2.0,
        r"Mean graded stability $\bar{\gamma}(P)$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )

    return save_figure(
        fig,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SI model-scale figure using parameter count and decoder layers "
            "for instruction-tuned Direct models."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/analysis/stability_variation"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    parser.add_argument(
        "--probe",
        default="sawmil",
        choices=["sawmil", "svm", "mean_difference"],
        help="Probe shown in the figure. Default: sawmil.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    statement_path = args.input_dir / "statement_level.parquet"
    if not statement_path.exists():
        raise FileNotFoundError(statement_path)

    statement_level = pd.read_parquet(statement_path)
    summary = summarize_statement_level(
        statement_level,
        probe=args.probe,
    )

    print("=" * 100)
    print("SI: MODEL SCALE")
    print("=" * 100)
    print(f"Input:      {statement_path}")
    print(f"Output:     {args.output_dir / (OUTPUT_STEM + '.pdf')}")
    print(f"Probe:      {args.probe}")
    print(f"Estimator:  {ESTIMATOR}")
    print("Models:     instruction-tuned only")
    print("Error bars: ±1 SE across proposition-level gamma(P)")
    print("=" * 100)

    # Print the realized ordering for reproducibility / quick visual checking.
    for dataset in DATASETS:
        print(f"\n{DATASET_NAMES[dataset]}")
        for scale, label in (("params_b", "parameter count"), ("layers", "decoder layers")):
            panel = ordered_panel(summary, dataset=dataset, scale=scale)
            print(f"  {label} order:")
            for _, row in panel.iterrows():
                print(
                    "    "
                    f"{display_model(row['model']):<20} "
                    f"params={row['params_b']:>5g}B  "
                    f"layers={int(row['layers']):>2d}  "
                    f"mean={row['mean_gamma']:.4f}  "
                    f"SE={row['se_gamma']:.4f}"
                )

    print("\n" + "=" * 100)

    path = make_figure(
        summary=summary,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    plt.close("all")
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
