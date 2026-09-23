from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
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

# Instruction-tuned models are stored with a leading underscore in the current
# analysis outputs. Only these models are shown in this figure.
INSTRUCTION_PREFIX = "_"
ESTIMATOR = "direct"

PROBE_SPECS = (
    ("svm", "SVM"),
    ("mean_difference", "Mass Mean"),
)

FAMILY_COLORS = {
    "llama": "#A63D4D",
    "gemma": "#3F6699",
    "mistral": "#C08A1D",
    "qwen": "#3E8064",
}

FAMILY_LABELS = {
    "llama": "Llama",
    "gemma": "Gemma",
    "mistral": "Mistral",
    "qwen": "Qwen",
}

FAMILY_ORDER = (
    "llama",
    "gemma",
    "mistral",
    "qwen",
)

MEDIAN_COLOR = "#2f2f2f"
GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"

FIGSIZE = (7.2, 4.6)

# This model has historically appeared as a degenerate zero-distance case in
# the saved coherence outputs. Keep its x-axis slot even if the pair-level file
# contains no sampled rows for it.
DEGENERATE_MODEL = "_mistral-3.1-24b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def model_family(model: str) -> str:
    """Return the canonical model-family key used for colors and sorting."""
    bare = str(model).lstrip("_").lower()

    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family

    raise ValueError(
        f"Unrecognized model family for {model!r}. "
        f"Expected one of {list(FAMILY_ORDER)}."
    )


def model_sort_key(model: str) -> tuple[Any, ...]:
    """Sort models by family, then parameter count, then name."""
    bare = str(model).lstrip("_").lower()
    family = model_family(model)
    family_rank = FAMILY_ORDER.index(family)

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


def display_model(model: str) -> str:
    """Display instruction-tuned model names without the leading '_' or '(i)'."""
    return str(model).lstrip("_")


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


def style_axis(ax: plt.Axes) -> None:
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
    )
    log.info(
        "Wrote %s",
        path,
    )


def nice_upper_limit(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.10

    padded = value * 1.03

    if padded <= 0.10:
        step = 0.01
    elif padded <= 0.25:
        step = 0.025
    elif padded <= 0.50:
        step = 0.05
    else:
        step = 0.10

    return math.ceil(
        padded / step
    ) * step


def finite_values(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


# ---------------------------------------------------------------------------
# Violin drawing
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
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return

    median = float(
        np.median(values)
    )

    # violinplot cannot estimate a KDE from a single unique value. For a
    # degenerate sampled distribution, show the value with the same horizontal
    # median convention used for non-degenerate violins.
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

    # Matplotlib's standard median artist is a horizontal line. Restyle it to
    # match the rest of the figure while retaining that familiar convention.
    violin["cmedians"].set_color(MEDIAN_COLOR)
    violin["cmedians"].set_linewidth(1.05)
    violin["cmedians"].set_zorder(6)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------



def make_figure(
    *,
    pair_sample: pd.DataFrame,
    output_dir: Path,
    stem: str,
    overwrite: bool,
    y_max: float | None,
    violin_width: float,
    violin_alpha: float,
    bw_method: str | float | None,
) -> None:
    require_columns(
        pair_sample,
        ("model", "dataset", "probe", "estimator", "rms_adjustment"),
        source="pair_sample.parquet",
    )

    model_text = pair_sample["model"].astype(str)
    probe_names = {probe for probe, _ in PROBE_SPECS}

    frame = pair_sample.loc[
        pair_sample["probe"].astype(str).isin(probe_names)
        & pair_sample["estimator"].astype(str).eq(ESTIMATOR)
        & pair_sample["dataset"].astype(str).isin(DATASETS)
        & model_text.str.startswith(INSTRUCTION_PREFIX)
    ].copy()

    frame["rms_adjustment"] = pd.to_numeric(
        frame["rms_adjustment"],
        errors="coerce",
    )
    frame = frame.loc[np.isfinite(frame["rms_adjustment"])].copy()

    if frame.empty:
        raise RuntimeError(
            "No finite instruction-tuned Direct pair-sample coherence rows "
            "for the requested SVM / Mass Mean probes."
        )

    missing_probes = [
        probe
        for probe, _ in PROBE_SPECS
        if not frame["probe"].astype(str).eq(probe).any()
    ]
    if missing_probes:
        raise RuntimeError(
            "Missing finite pair-sample rows for probes: "
            + ", ".join(missing_probes)
        )

    if (frame["rms_adjustment"] < 0).any():
        raise ValueError("Found negative rms_adjustment values.")

    for model in sorted(frame["model"].astype(str).unique()):
        model_family(model)

    all_values = frame["rms_adjustment"].to_numpy(dtype=float)
    observed_max = float(np.max(all_values))

    if y_max is None:
        plot_y_max = nice_upper_limit(observed_max)
        n_clipped = 0
    else:
        plot_y_max = float(y_max)
        n_clipped = int(np.sum(all_values > plot_y_max))

    all_models = sorted(
        set(frame["model"].astype(str).unique()) | {DEGENERATE_MODEL},
        key=model_sort_key,
    )

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.05],
        wspace=0.15,
        hspace=1.35,
    )

    axes: list[list[plt.Axes]] = [[], []]
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    panel_index = 0
    x = np.arange(1, len(all_models) + 1, dtype=float)

    for row_index, (probe, _) in enumerate(PROBE_SPECS):
        probe_frame = frame.loc[frame["probe"].astype(str).eq(probe)].copy()

        for col_index, dataset in enumerate(DATASETS):
            ax = fig.add_subplot(gs[row_index, col_index])
            axes[row_index].append(ax)

            panel = probe_frame.loc[
                probe_frame["dataset"].astype(str).eq(dataset)
            ].copy()

            for position, model in zip(x, all_models):
                model_panel = panel.loc[
                    panel["model"].astype(str).eq(model)
                ]
                values = finite_values(model_panel, "rms_adjustment")
                family = model_family(model)
                color = FAMILY_COLORS[family]

                if model == DEGENERATE_MODEL and len(values) == 0:
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
                [display_model(model) for model in all_models],
                rotation=55,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set_xlim(0.35, len(all_models) + 0.65)
            ax.set_ylim(0.0, plot_y_max)
            style_axis(ax)

            # Dataset names function as column headers, so show them only
            # once above the first (SVM) row rather than repeating them above
            # the Mass Mean row.
            if row_index == 0:
                ax.set_title(
                    DATASET_NAMES[dataset],
                    loc="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color=TITLE_COLOR,
                    y=1.105,
                    pad=0,
                )

            ax.text(
                0.0,
                1.025,
                panel_labels[panel_index],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=TITLE_COLOR,
            )
            panel_index += 1

            if col_index == 0:
                ax.set_ylabel(r"RMS adjustment $d_{\text{CCK}}$", fontsize=7)
                ax.yaxis.set_label_coords(-0.13, 0.5)
            else:
                ax.tick_params(axis="y", labelleft=False)
                ax.spines["left"].set_visible(False)

    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    legend_handles = [
        Patch(
            facecolor=FAMILY_COLORS[family],
            edgecolor="none",
            alpha=violin_alpha,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=6.5,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    # Finalize panel geometry before placing figure-level headings.
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.89,
        bottom=0.0,
    )

    legend_pos = legend_ax.get_position()
    legend_ax.set_position(
        [
            legend_pos.x0,
            max(0.05, legend_pos.y0 - 0.18),
            legend_pos.width,
            legend_pos.height,
        ]
    )

    fig.canvas.draw()

    # Overall figure heading.
    first_pos = axes[0][0].get_position()

    # Probe labels: horizontal section headings.  The SVM label sits above
    # the column titles; the Mass Mean label sits in the enlarged inter-row
    # gap, clear of the rotated model names from the first row.
    for row_index, (_, probe_label) in enumerate(PROBE_SPECS):
        row_pos = axes[row_index][0].get_position()
        if row_index == 0:
            label_x = row_pos.x0 - 0.05
            label_y = row_pos.y1 + 0.040
        else:
            label_x = row_pos.x0 - 0.05
            label_y = row_pos.y1 + 0.04

        fig.text(
            label_x,
            label_y,
            probe_label,
            ha="left",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        overwrite=overwrite,
    )
    plt.close(fig)

    print(
        f"Observed sampled RMS range across both probes: "
        f"{all_values.min():.6f} to {observed_max:.6f}"
    )
    print(f"Common displayed y-axis: 0 to {plot_y_max:.6f}")
    print(f"Instruction-tuned models shown: {len(all_models)}")

    if n_clipped:
        fraction = n_clipped / len(all_values)
        print(
            f"WARNING: explicit --y_max clips "
            f"{n_clipped:,}/{len(all_values):,} sampled pair-level "
            f"values ({fraction:.3%}) above the visible plotting range."
        )


def parse_bw_method(
    value: str,
) -> str | float | None:
    text = str(value).strip().lower()

    if text in (
        "none",
        "default",
    ):
        return None

    if text in (
        "scott",
        "silverman",
    ):
        return text

    try:
        numeric = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--bw_method must be one of "
            "{default, scott, silverman} or a positive float."
        ) from exc

    if numeric <= 0:
        raise argparse.ArgumentTypeError(
            "Numeric --bw_method must be positive."
        )

    return numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SI Figure 3 with SVM and Mass Mean rows using "
            "instruction-tuned Direct pair-level coherence-distance violins."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            "outputs/analysis/coherence"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/figures"
        ),
    )

    parser.add_argument(
        "--stem",
        default="figure_si_3_svm_mass_mean",
    )

    parser.add_argument(
        "--y_max",
        type=float,
        default=None,
        help=(
            "Optional common upper y-limit. If omitted, the full sampled "
            "range is shown. If supplied, the script reports how many sampled "
            "pair-level values lie above the visible plotting range."
        ),
    )

    parser.add_argument(
        "--violin_width",
        type=float,
        default=0.82,
        help=(
            "Width of each full violin. Default: 0.82."
        ),
    )

    parser.add_argument(
        "--violin_alpha",
        type=float,
        default=0.86,
        help=(
            "Opacity of violin bodies. Default: 0.86."
        ),
    )

    parser.add_argument(
        "--bw_method",
        type=parse_bw_method,
        default="scott",
        help=(
            "Matplotlib violin KDE bandwidth: scott, silverman, default, "
            "or a positive float. Default: scott."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (
        args.y_max is not None
        and args.y_max <= 0
    ):
        raise ValueError(
            "--y_max must be positive."
        )

    if not (
        0.0
        < args.violin_width
        <= 1.4
    ):
        raise ValueError(
            "--violin_width must be in (0, 1.4]."
        )

    if not (
        0.0
        < args.violin_alpha
        <= 1.0
    ):
        raise ValueError(
            "--violin_alpha must be in (0, 1]."
        )

    path = args.input_dir / "pair_sample.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    pair_sample = pd.read_parquet(path)

    primary = pair_sample.loc[
        pair_sample["probe"].astype(str).isin([probe for probe, _ in PROBE_SPECS])
        & pair_sample["estimator"].astype(str).eq(ESTIMATOR)
        & pair_sample["model"].astype(str).str.startswith(INSTRUCTION_PREFIX)
    ].copy()

    print("=" * 100)
    print("FIGURE SI 3: SVM + MASS MEAN DISTANCE TO PROBABILISTIC COHERENCE")
    print("=" * 100)
    print(f"Input:     {path}")
    print(f"Output:    {args.output_dir / (args.stem + '.pdf')}")
    print("Probes:    SVM, Mass Mean")
    print("Models:    instruction-tuned only")
    print("Estimator: Direct only")
    print()
    print("PAIR-SAMPLE SUMMARY")
    print("-" * 100)

    if not primary.empty:
        summary = (
            primary.groupby(["probe", "dataset"], sort=True)
            .agg(
                n_models=("model", "nunique"),
                n_sampled_pairs=("rms_adjustment", "count"),
                median_rms=("rms_adjustment", "median"),
                q25_rms=("rms_adjustment", lambda x: x.quantile(0.25)),
                q75_rms=("rms_adjustment", lambda x: x.quantile(0.75)),
                q95_rms=("rms_adjustment", lambda x: x.quantile(0.95)),
            )
            .reset_index()
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("=" * 100)

    make_figure(
        pair_sample=pair_sample,
        output_dir=args.output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
        y_max=args.y_max,
        violin_width=args.violin_width,
        violin_alpha=args.violin_alpha,
        bw_method=args.bw_method,
    )


if __name__ == "__main__":
    main()
