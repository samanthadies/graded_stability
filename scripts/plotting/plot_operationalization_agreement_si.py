"""
Generates the SI comparison of Direct Conditional and Joint-to-Conditional graded
stability, summarizing their proposition-level agreement across instruction models.

Examples:
    python -m scripts.plotting.plot_operationalization_agreement_si --probe sawmil --overwrite
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
from matplotlib.gridspec import GridSpec
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

GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"
ZERO_LINE_COLOR = "#777777"

FIGSIZE = (7.2, 2.7)
DEGENERATE_MODEL = "_mistral-3.1-24b"


def model_family(model: str) -> str:
    bare = str(model).lstrip("_").lower()
    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family
    raise ValueError(
        f"Unrecognized model family for {model!r}; expected one of {list(FAMILY_ORDER)}."
    )


def model_sort_key(model: str) -> tuple[Any, ...]:
    bare = str(model).lstrip("_").lower()
    sizes = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            bare,
        )
    ]
    size = max(sizes) if sizes else float("inf")
    return (
        FAMILY_ORDER.index(model_family(model)),
        size,
        bare,
    )


def display_model(model: str) -> str:
    return str(model).lstrip("_")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}; available={frame.columns.tolist()}"
        )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=5.4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, *, output_dir: Path, stem: str, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")
    fig.savefig(path, bbox_inches="tight")
    log.info("Wrote %s", path)


def make_figure(
    *,
    model_summary: pd.DataFrame,
    probe: str,
    output_dir: Path,
    stem: str,
    overwrite: bool,
    y_min: float,
    y_max: float,
) -> None:
    require_columns(
        model_summary,
        ("model", "dataset", "probe", "raw_spearman_rho"),
        source="model_summary.parquet",
    )

    expected_set = set(EXPECTED_MODELS)
    frame = model_summary.loc[
        model_summary["probe"].astype(str).eq(probe)
        & model_summary["model"].astype(str).isin(expected_set)
        & model_summary["dataset"].astype(str).isin(DATASETS)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            f"No instruction-tuned model-summary rows for probe={probe!r}."
        )

    values = pd.to_numeric(frame["raw_spearman_rho"], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values):
        observed_min = float(values.min())
        observed_max = float(values.max())
        if observed_min < y_min or observed_max > y_max:
            raise ValueError(
                f"Requested y-limits [{y_min:g}, {y_max:g}] clip observed raw "
                f"Spearman rho values [{observed_min:.3f}, {observed_max:.3f}]."
            )

    all_models = sorted(EXPECTED_MODELS, key=model_sort_key)

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(nrows=1, ncols=3, figure=fig, wspace=0.15)
    panel_labels = ("(a)", "(b)", "(c)")
    axes: list[plt.Axes] = []

    for col_index, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, col_index])
        axes.append(ax)

        panel = frame.loc[frame["dataset"].astype(str).eq(dataset)].copy()
        present_models = set(panel["model"].astype(str))
        models = [model for model in all_models if model in present_models or model == DEGENERATE_MODEL]

        panel = panel.assign(model=panel["model"].astype(str)).set_index("model").reindex(models)
        rho = pd.to_numeric(panel["raw_spearman_rho"], errors="coerce").to_numpy(dtype=float, copy=True)

        for idx, model in enumerate(models):
            if model == DEGENERATE_MODEL and not np.isfinite(rho[idx]):
                rho[idx] = 0.0

        x = np.arange(1, len(models) + 1, dtype=float)
        colors = [FAMILY_COLORS[model_family(model)] for model in models]

        ax.bar(x, rho, width=0.72, color=colors, edgecolor="none", zorder=3)
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.65, linestyle="--", zorder=0)
        ax.set_xlim(0.35, len(models) + 0.65)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x)
        ax.set_xticklabels([display_model(model) for model in models], rotation=55, ha="right", rotation_mode="anchor")
        style_axis(ax)

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
            panel_labels[col_index],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
        )

        if col_index != 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)

    fig.subplots_adjust(left=0.105, right=0.995, top=0.80, bottom=0.33)

    fig.canvas.draw()
    first_pos = axes[0].get_position()
    fig.text(
        0.035,
        first_pos.y1 + 0.10,
        "Agreement between graded stability operationalizations",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.text(
        0.05,
        (first_pos.y0 + first_pos.y1) / 2.0,
        r"Spearman $\rho$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="none", label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.0,
    )

    save_figure(fig, output_dir=output_dir, stem=stem, overwrite=overwrite)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SI operationalization agreement for instruction-tuned models only."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/analysis/operationalization_agreement"),
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
    )
    parser.add_argument(
        "--stem",
        default="figure_si_operationalization_agreement",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.y_min >= args.y_max:
        raise ValueError("--y_min must be less than --y_max.")

    path = args.input_dir / "model_summary.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_parquet(path)
    primary = frame.loc[
        frame["probe"].astype(str).eq(args.probe)
        & frame["model"].astype(str).isin(set(EXPECTED_MODELS))
    ].copy()

    print("=" * 100)
    print("FIGURE SI: DIRECT VS. JOINT STABILITY AGREEMENT")
    print("=" * 100)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir / (args.stem + '.pdf')}")
    print(f"Probe:  {args.probe}")
    print("Models: instruction-tuned only")
    print()

    if not primary.empty:
        summary = (
            primary.groupby("dataset", sort=True)
            .agg(n_models=("model", "nunique"), median_raw_rho=("raw_spearman_rho", "median"))
            .reset_index()
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("=" * 100)

    make_figure(
        model_summary=frame,
        probe=args.probe,
        output_dir=args.output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
