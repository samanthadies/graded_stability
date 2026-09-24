"""
Generates the base-model SI graded-stability distributions, providing the pretrained
counterpart to the instruction-tuned domain comparison in main-text Figure 4.

Example:
    python -m scripts.plotting.plot_fig4_base --probe sawmil --overwrite
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
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
INSTRUCTION_PREFIX = ""

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

FAMILY_COLORS = {
    "llama": "#A63D4D",
    "gemma": "#3F6699",
    "mistral": "#C08A1D",
    "qwen": "#3E8064",
}

# The exact 12 base checkpoints shown in this SI variant.
# This guarantees the same model set and ordering across all dataset panels.
EXPECTED_MODELS = (
    "llama-3.2-3b",
    "llama-3.1-8b",
    "llama-3.1-70b",
    "gemma-7b",
    "gemma-2-9b",
    "gemma-2-27b",
    "mistral-7b",
    "mistral-12b",
    "mistral-3.1-24b",
    "qwen-2.5-7b",
    "qwen-2.5-14b",
    "qwen-2.5-72b",
)

# Text-decoder layer counts (num_hidden_layers / n_layers) for the checkpoints
# above. Keys use display names so the leading instruction underscore is not
# part of the architecture metadata.
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

SCALE_MODE_LABELS = {
    "relative": "Relative model scale",
    "parameters": "Parameters (B)",
    "layers": "Decoder layers",
}

SCALE_OUTPUT_SUFFIX = {
    "relative": "relative_scale",
    "parameters": "parameter_scale",
    "layers": "decoder_layer_scale",
}

MEDIAN_COLOR = "#2f2f2f"
GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"

FIGSIZE = (7.2, 1.75)

# This checkpoint has historically had degenerate / missing sAwMIL outputs in
# City Locations and Word Definitions. Preserve its x-axis slot in Row 1. If
# no distribution is available, show the zero-valued horizontal median mark
# rather than fabricating a violin density.
DEGENERATE_MODEL = "mistral-3.1-24b"
DEGENERATE_DATASETS = {"cities_loc", "defs"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def display_model(model: str) -> str:
    """Display base model names with an explicit '(b)' suffix."""
    return f"{str(model)} (b)"


def model_family(model: str) -> str:
    """Return the canonical broad family encoded by color."""
    bare = str(model).lstrip("_").lower()

    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family

    raise ValueError(
        f"Unrecognized model family for {model!r}; "
        f"expected one of {list(FAMILY_ORDER)}."
    )


def model_sort_key(model: str) -> tuple[Any, ...]:
    """Sort by family, nominal parameter count, then checkpoint name."""
    bare = str(model).lstrip("_").lower()
    family_rank = FAMILY_ORDER.index(model_family(model))

    sizes = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            bare,
        )
    ]
    size = max(sizes) if sizes else float("inf")

    return (
        family_rank,
        size,
        bare,
    )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}; "
            f"available={frame.columns.tolist()}"
        )


def finite_values(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def style_axis(
    ax: plt.Axes,
    *,
    x_grid: bool = False,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=7,
    )
    ax.tick_params(
        axis="x",
        labelsize=5.4,
    )

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.45,
        alpha=0.85,
    )

    if x_grid:
        ax.grid(
            axis="x",
            color=GRID_COLOR,
            linewidth=0.35,
            alpha=0.45,
        )

    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    overwrite: bool,
) -> None:
    """Save the main-text figure as PDF only."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / f"{stem}.pdf"

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists; pass --overwrite."
        )

    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    log.info(
        "Wrote %s",
        path,
    )


# ---------------------------------------------------------------------------
# Row 1: full model-level violins
# ---------------------------------------------------------------------------


def draw_violin(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    position: float,
    color: str,
    width: float,
    alpha: float,
    bw_method: str | float | None,
) -> None:
    """Draw one full violin with a conventional horizontal median line."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    median = float(np.median(values))

    # KDE is undefined for a single unique value. Draw only the horizontal
    # median bar rather than inventing artificial spread.
    if len(values) < 2 or np.unique(values).size < 2:
        ax.hlines(
            median,
            position - width * 0.24,
            position + width * 0.24,
            color=MEDIAN_COLOR,
            linewidth=1.05,
            zorder=6,
        )
        return

    violin = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=bw_method,
        points=100,
    )

    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(alpha)
        body.set_zorder(3)

    violin["cmedians"].set_color(MEDIAN_COLOR)
    violin["cmedians"].set_linewidth(1.05)
    violin["cmedians"].set_zorder(6)


def plot_distribution_panel(
    ax: plt.Axes,
    *,
    statement_level: pd.DataFrame,
    dataset: str,
    violin_width: float,
    violin_alpha: float,
    bw_method: str | float | None,
    show_y_labels: bool,
) -> None:
    panel = statement_level.loc[
        statement_level["dataset"].astype(str).eq(dataset)
    ].copy()

    models = sorted(
        EXPECTED_MODELS,
        key=model_sort_key,
    )
    x = np.arange(1, len(models) + 1, dtype=float)

    for position, model in zip(x, models):
        model_panel = panel.loc[
            panel["model"].astype(str).eq(model)
        ]
        values = finite_values(
            model_panel,
            "gamma",
        )

        family = model_family(model)
        color = FAMILY_COLORS[family]

        if (
            model == DEGENERATE_MODEL
            and dataset in DEGENERATE_DATASETS
            and len(values) == 0
        ):
            ax.hlines(
                0.0,
                float(position) - violin_width * 0.24,
                float(position) + violin_width * 0.24,
                color=MEDIAN_COLOR,
                linewidth=1.05,
                zorder=6,
                clip_on=False,
            )
        else:
            draw_violin(
                ax,
                values=values,
                position=float(position),
                color=color,
                width=violin_width,
                alpha=violin_alpha,
                bw_method=bw_method,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_model(model) for model in models],
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_xlim(0.35, len(models) + 0.65)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    style_axis(ax)

    if not show_y_labels:
        ax.tick_params(
            axis="y",
            labelleft=False,
        )
        ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Row 2: Direct instruction-tuned scale trajectories
# ---------------------------------------------------------------------------


def prepare_family_scale_data(
    panel: pd.DataFrame,
    *,
    family: str,
    scale_mode: str,
) -> pd.DataFrame:
    """Prepare one family's Direct instruction-tuned scale trajectory."""
    family_panel = panel.loc[
        panel["model"].astype(str).map(model_family).eq(family)
    ].copy()

    if family_panel.empty:
        return family_panel

    # Collapse any accidental duplicate summary rows for the same checkpoint.
    family_panel["params_b"] = pd.to_numeric(
        family_panel["params_b"],
        errors="coerce",
    )
    family_panel["mean_gamma"] = pd.to_numeric(
        family_panel["mean_gamma"],
        errors="coerce",
    )

    family_panel = (
        family_panel.groupby("model", as_index=False)
        .agg(
            params_b=("params_b", "mean"),
            mean_gamma=("mean_gamma", "mean"),
        )
    )

    family_panel = family_panel.loc[
        np.isfinite(family_panel["params_b"])
        & np.isfinite(family_panel["mean_gamma"])
    ].copy()

    if family_panel.empty:
        return family_panel

    family_panel = family_panel.sort_values(
        ["params_b", "model"],
        kind="stable",
    ).reset_index(drop=True)

    if scale_mode == "relative":
        n_models = len(family_panel)
        if n_models == 1:
            x_values = np.array([1.0], dtype=float)
        elif n_models == 2:
            x_values = np.array([0.0, 2.0], dtype=float)
        elif n_models == 3:
            x_values = np.array([0.0, 1.0, 2.0], dtype=float)
        else:
            x_values = np.linspace(0.0, 2.0, n_models)

    elif scale_mode == "parameters":
        x_values = family_panel["params_b"].to_numpy(dtype=float)

    elif scale_mode == "layers":
        missing = [
            display_model(model)
            for model in family_panel["model"].astype(str)
            if display_model(model) not in MODEL_DECODER_LAYERS
        ]
        if missing:
            raise ValueError(
                "Missing decoder-layer metadata for models: "
                + ", ".join(sorted(set(missing)))
            )

        x_values = np.array(
            [
                MODEL_DECODER_LAYERS[display_model(model)]
                for model in family_panel["model"].astype(str)
            ],
            dtype=float,
        )

    else:
        raise ValueError(
            f"Unknown scale_mode={scale_mode!r}."
        )

    family_panel["scale_x"] = x_values
    return family_panel


def configure_scale_axis(
    ax: plt.Axes,
    *,
    scale_mode: str,
) -> None:
    if scale_mode == "relative":
        ax.set_xlim(-0.15, 2.15)
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_xticklabels(
            ["Small", "Medium", "Large"],
            fontsize=5.8,
        )

    elif scale_mode == "parameters":
        ax.set_xlim(0.0, 82.0)
        ax.set_xticks([0, 20, 40, 60, 80])

    elif scale_mode == "layers":
        ax.set_xlim(0.0, 84.0)
        ax.set_xticks([0, 20, 40, 60, 80])

    else:
        raise ValueError(
            f"Unknown scale_mode={scale_mode!r}."
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))


def plot_scale_panel(
    ax: plt.Axes,
    *,
    model_summary: pd.DataFrame,
    dataset: str,
    scale_mode: str,
    show_y_labels: bool,
) -> None:
    panel = model_summary.loc[
        model_summary["dataset"].astype(str).eq(dataset)
    ].copy()

    plotted = 0

    for family in FAMILY_ORDER:
        trajectory = prepare_family_scale_data(
            panel,
            family=family,
            scale_mode=scale_mode,
        )

        if trajectory.empty:
            continue

        x = trajectory["scale_x"].to_numpy(dtype=float)
        y = trajectory["mean_gamma"].to_numpy(dtype=float)

        # Sort by nominal parameter count, not by x. This matters for decoder
        # layers because two checkpoints can have the same layer count (e.g.,
        # Mistral 12B and 24B) while still representing distinct checkpoints.
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() == 0:
            continue

        color = FAMILY_COLORS[family]

        ax.plot(
            x[valid],
            y[valid],
            color=color,
            linestyle="-",
            linewidth=1.05,
            marker="o",
            markersize=3.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.65,
            alpha=0.95,
            zorder=4,
        )
        plotted += 1

    configure_scale_axis(
        ax,
        scale_mode=scale_mode,
    )
    style_axis(
        ax,
        x_grid=(scale_mode != "relative"),
    )

    if not show_y_labels:
        ax.tick_params(
            axis="y",
            labelleft=False,
        )
        ax.spines["left"].set_visible(False)

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No valid scale groups",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
            color="#666666",
        )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def make_figure(
    *,
    statement_level: pd.DataFrame,
    probe: str,
    output_dir: Path,
    stem: str,
    overwrite: bool,
    violin_width: float,
    violin_alpha: float,
    bw_method: str | float | None,
) -> None:
    require_columns(
        statement_level,
        ("model", "dataset", "probe", "estimator", "gamma"),
        source="statement_level.parquet",
    )

    expected_set = set(EXPECTED_MODELS)
    statements = statement_level.loc[
        statement_level["probe"].astype(str).eq(probe)
        & statement_level["dataset"].astype(str).isin(DATASETS)
        & statement_level["estimator"].astype(str).eq(ESTIMATOR)
        & statement_level["model"].astype(str).isin(expected_set)
    ].copy()

    if statements.empty:
        raise RuntimeError(f"No base-model Direct statement-level rows for probe={probe!r}.")

    for model in EXPECTED_MODELS:
        model_family(model)

    present = set(statements["model"].astype(str).unique())
    missing = [model for model in EXPECTED_MODELS if model not in present]
    if missing:
        log.warning(
            "No statement-level rows for expected models: %s",
            ", ".join(display_model(m) for m in missing),
        )

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.76, bottom=0.34, wspace=0.15)
    panel_labels = ("(a)", "(b)", "(c)")

    for col_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        plot_distribution_panel(
            ax,
            statement_level=statements,
            dataset=dataset,
            violin_width=violin_width,
            violin_alpha=violin_alpha,
            bw_method=bw_method,
            show_y_labels=(col_index == 0),
        )
        ax.set_title(
            DATASET_NAMES[dataset], loc="center", fontsize=7.2,
            fontweight="bold", color=TITLE_COLOR, y=1.12, pad=0,
        )
        ax.text(
            0.0, 1.025, panel_labels[col_index], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.2, fontweight="bold",
            color=TITLE_COLOR,
        )

    legend_handles = [
        Patch(
            facecolor=FAMILY_COLORS[family], edgecolor="none",
            alpha=violin_alpha, label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.55, -0.15),
        ncol=4, frameon=False, fontsize=6.3, handlelength=1.4,
        handletextpad=0.45, columnspacing=1.15, borderaxespad=0.0,
    )

    fig.canvas.draw()
    first_pos = axes[0].get_position()
    fig.text(
        0.035, first_pos.y1 + 0.10, r"Graded belief stability $\gamma(P)$",
        ha="left", va="bottom", fontsize=7.7, fontweight="bold", color=TITLE_COLOR,
    )
    fig.text(
        0.05, (first_pos.y0 + first_pos.y1) / 2.0,
        r"Graded stability $\gamma(P)$", ha="center", va="center",
        rotation=90, fontsize=7,
    )

    save_figure(fig, output_dir=output_dir, stem=stem, overwrite=overwrite)
    plt.close(fig)
    print(f"Output: {output_dir / (stem + '.pdf')}")


def parse_bw_method(value: str) -> str | float | None:
    text = str(value).strip().lower()
    if text in ("none", "default"):
        return None
    if text in ("scott", "silverman"):
        return text
    try:
        numeric = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--bw_method must be one of {default, scott, silverman} or a positive float."
        ) from exc
    if numeric <= 0:
        raise argparse.ArgumentTypeError("Numeric --bw_method must be positive.")
    return numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot base-model Direct graded-stability distributions."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("outputs/analysis/stability_variation"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument(
        "--probe", default="sawmil", choices=["sawmil", "svm", "mean_difference"],
        help="Probe shown in the figure. Default: sawmil.",
    )
    parser.add_argument(
        "--stem", default="figure_si_4_relative_scale_base",
        help="Output stem. Default preserves the filename already referenced by the SI.",
    )
    parser.add_argument("--violin_width", type=float, default=0.82)
    parser.add_argument("--violin_alpha", type=float, default=0.86)
    parser.add_argument("--bw_method", type=parse_bw_method, default="scott")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.violin_width <= 1.4):
        raise ValueError("--violin_width must be in (0, 1.4].")
    if not (0.0 < args.violin_alpha <= 1.0):
        raise ValueError("--violin_alpha must be in (0, 1].")

    statement_path = args.input_dir / "statement_level.parquet"
    if not statement_path.exists():
        raise FileNotFoundError(statement_path)
    statement_level = pd.read_parquet(statement_path)

    print("=" * 100)
    print("FIGURE SI 4: GRADED STABILITY DISTRIBUTIONS (BASE)")
    print("=" * 100)
    print(f"Input:      {statement_path}")
    print(f"Output:     {args.output_dir / (args.stem + '.pdf')}")
    print(f"Probe:      {args.probe}")
    print(f"Estimator:  {ESTIMATOR}")
    print("Models:     base only")
    print("Layout:     proposition-level distributions only")
    print("=" * 100)

    make_figure(
        statement_level=statement_level,
        probe=args.probe,
        output_dir=args.output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
        violin_width=args.violin_width,
        violin_alpha=args.violin_alpha,
        bw_method=args.bw_method,
    )


if __name__ == "__main__":
    main()
