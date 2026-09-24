"""
Generates the SI comparison of matched pretrained and instruction-tuned checkpoints,
measuring the within-proposition change in Direct Conditional graded stability.

Example:
    python -m scripts.plotting.plot_base_v_instruct_si --probe sawmil --overwrite
"""

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

ESTIMATORS = ("direct",)

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

GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"
ZERO_LINE_COLOR = "#777777"
ERRORBAR_COLOR = "#2f2f2f"

FIGSIZE = (7.2, 2.9)
OUTPUT_STEM = "figure_si_base_v_instruct"

DEGENERATE_CHECKPOINT = "mistral-3.1-24b"
DEGENERATE_DATASETS = {"cities_loc", "defs"}

BAR_ALPHA = 0.95


# -------- metadata helpers --------
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
    return (FAMILY_ORDER.index(model_family(model)), size, bare)


def display_checkpoint(checkpoint: str) -> str:
    return str(checkpoint).lstrip("_")


# -------- generic helpers --------
def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}; available={frame.columns.tolist()}"
        )


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=5.4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def symmetric_limit(
    values: Iterable[float],
    *,
    minimum: float = 0.10,
    step: float = 0.05,
    padding: float = 1.10,
) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return minimum
    max_abs = max(float(np.max(np.abs(arr))) * padding, minimum)
    return float(math.ceil(max_abs / step) * step)


def save_figure(fig: plt.Figure, *, output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{OUTPUT_STEM}.pdf"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    log.info("Wrote %s", path)
    return path


# -------- SE computation --------
def summarize_statement_level_se(statement_deltas: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        statement_deltas,
        ("probe", "dataset", "estimator", "canonical_checkpoint"),
        source="instruction_statement_deltas.parquet",
    )

    delta_col = first_existing(
        statement_deltas,
        (
            "delta_gamma_instruct_minus_base",
            "delta_gamma_matched",
            "matched_delta_gamma",
            "delta_gamma",
            "gamma_delta",
        ),
    )
    if delta_col is None:
        raise ValueError(
            "instruction_statement_deltas.parquet: could not identify the statement-level matched "
            "delta-gamma column. Tried aliases: delta_gamma_instruct_minus_base, "
            "delta_gamma_matched, matched_delta_gamma, delta_gamma, gamma_delta"
        )

    frame = statement_deltas.copy()
    frame[delta_col] = pd.to_numeric(frame[delta_col], errors="coerce")
    frame = frame.loc[np.isfinite(frame[delta_col])].copy()
    if frame.empty:
        raise RuntimeError(
            "instruction_statement_deltas.parquet contains no finite statement-level delta-gamma values."
        )

    grouped = (
        frame.groupby(["probe", "dataset", "estimator", "canonical_checkpoint"], as_index=False)
        .agg(
            n_statements=(delta_col, "size"),
            sd_delta_gamma=(delta_col, lambda x: float(np.std(np.asarray(x, dtype=float), ddof=1)) if len(x) > 1 else 0.0),
            mean_delta_gamma_reconstructed=(delta_col, "mean"),
        )
    )
    grouped["se_delta_gamma"] = grouped["sd_delta_gamma"] / np.sqrt(grouped["n_statements"].clip(lower=1))
    return grouped


def merge_summary_with_se(summary: pd.DataFrame, statement_deltas: pd.DataFrame | None) -> pd.DataFrame:
    frame = summary.copy()

    if statement_deltas is None:
        log.warning("No instruction_statement_deltas.parquet found; plotting bars without error bars.")
        frame["se_delta_gamma"] = np.nan
        frame["n_statements"] = np.nan
        return frame

    se_summary = summarize_statement_level_se(statement_deltas)
    frame = frame.merge(
        se_summary[[
            "probe", "dataset", "estimator", "canonical_checkpoint",
            "n_statements", "se_delta_gamma", "mean_delta_gamma_reconstructed"
        ]],
        on=["probe", "dataset", "estimator", "canonical_checkpoint"],
        how="left",
        validate="one_to_one",
    )

    mean_col = "mean_delta_gamma_matched"
    diffs = np.abs(
        pd.to_numeric(frame[mean_col], errors="coerce")
        - pd.to_numeric(frame["mean_delta_gamma_reconstructed"], errors="coerce")
    )
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) and float(np.nanmax(diffs)) > 1e-8:
        log.warning(
            "Statement-level reconstructed means do not exactly match summary means; max absolute difference = %.6g",
            float(np.nanmax(diffs)),
        )

    frame = frame.drop(columns=["mean_delta_gamma_reconstructed"], errors="ignore")
    return frame


# -------- plotting --------
def plot_instruction_panel(
    ax: plt.Axes,
    *,
    instruction_pairs: pd.DataFrame,
    dataset: str,
    instruction_limit: float,
    show_y_labels: bool,
) -> None:
    panel = instruction_pairs.loc[
        instruction_pairs["dataset"].astype(str).eq(dataset)
    ].copy()

    if panel.empty:
        ax.text(
            0.5,
            0.5,
            "No valid checkpoint pairs",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#666666",
        )
        ax.set_ylim(-instruction_limit, instruction_limit)
        style_axis(ax)
        return

    checkpoints = set(panel["canonical_checkpoint"].astype(str).unique())
    if dataset in DEGENERATE_DATASETS:
        checkpoints.add(DEGENERATE_CHECKPOINT)
    checkpoints = sorted(checkpoints, key=model_sort_key)

    x = np.arange(1, len(checkpoints) + 1, dtype=float)

    subset = (
        panel.loc[panel["estimator"].astype(str).eq("direct")]
        .assign(canonical_checkpoint=lambda d: d["canonical_checkpoint"].astype(str))
        .set_index("canonical_checkpoint")
        .reindex(checkpoints)
    )

    values = pd.to_numeric(
        subset["mean_delta_gamma_matched"],
        errors="coerce",
    ).to_numpy(dtype=float, copy=True)

    se_values = pd.to_numeric(
        subset.get("se_delta_gamma", np.nan),
        errors="coerce",
    ).to_numpy(dtype=float, copy=True)

    if dataset in DEGENERATE_DATASETS:
        for idx, checkpoint in enumerate(checkpoints):
            if checkpoint == DEGENERATE_CHECKPOINT and not np.isfinite(values[idx]):
                values[idx] = 0.0
                se_values[idx] = 0.0

    se_values = np.where(np.isfinite(se_values), se_values, 0.0)
    colors = [FAMILY_COLORS[model_family(checkpoint)] for checkpoint in checkpoints]
    ax.bar(
        x,
        values,
        width=0.72,
        color=colors,
        alpha=BAR_ALPHA,
        edgecolor="none",
        zorder=3,
        yerr=se_values,
        ecolor=ERRORBAR_COLOR,
        capsize=2.2,
        error_kw={"elinewidth": 0.85, "capthick": 0.85, "zorder": 4},
    )

    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.65, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_checkpoint(checkpoint) for checkpoint in checkpoints],
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(0.35, len(checkpoints) + 0.65)
    ax.set_ylim(-instruction_limit, instruction_limit)
    style_axis(ax)

    if not show_y_labels:
        ax.tick_params(axis="y", labelleft=False)
        ax.spines["left"].set_visible(False)


def make_figure(
    *,
    instruction_pairs: pd.DataFrame,
    statement_deltas: pd.DataFrame | None,
    probe: str,
    output_dir: Path,
    overwrite: bool,
    instruction_limit: float | None,
) -> Path:
    require_columns(
        instruction_pairs,
        (
            "probe",
            "dataset",
            "estimator",
            "canonical_checkpoint",
            "mean_delta_gamma_matched",
        ),
        source="instruction_pair_summary.parquet",
    )

    summary = merge_summary_with_se(instruction_pairs, statement_deltas)

    instructions = summary.loc[
        summary["probe"].astype(str).eq(probe)
        & summary["dataset"].astype(str).isin(DATASETS)
        & summary["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    if instructions.empty:
        raise RuntimeError(f"No instruction-pair rows for probe={probe!r}.")

    if instruction_limit is None:
        y_extent = np.maximum(
            np.abs(pd.to_numeric(instructions["mean_delta_gamma_matched"], errors="coerce")),
            0.0,
        ) + np.maximum(
            pd.to_numeric(instructions["se_delta_gamma"], errors="coerce").fillna(0.0),
            0.0,
        )
        instruction_limit_value = symmetric_limit(
            y_extent,
            minimum=0.10,
            step=0.05,
            padding=1.10,
        )
    else:
        instruction_limit_value = float(instruction_limit)

    upper = np.abs(pd.to_numeric(instructions["mean_delta_gamma_matched"], errors="coerce")) + pd.to_numeric(instructions["se_delta_gamma"], errors="coerce").fillna(0.0)
    finite_upper = np.asarray(upper, dtype=float)
    finite_upper = finite_upper[np.isfinite(finite_upper)]
    if len(finite_upper) and np.max(finite_upper) > instruction_limit_value:
        raise ValueError(
            f"--instruction_limit={instruction_limit_value:g} clips instruction effects/error bars; "
            f"need at least {np.max(finite_upper):.4f}."
        )

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(nrows=1, ncols=3, figure=fig, wspace=0.15)
    panel_labels = ("(a)", "(b)", "(c)")
    axes: list[plt.Axes] = []

    for col_index, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, col_index])
        axes.append(ax)

        plot_instruction_panel(
            ax,
            instruction_pairs=instructions,
            dataset=dataset,
            instruction_limit=instruction_limit_value,
            show_y_labels=(col_index == 0),
        )

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

    fig.subplots_adjust(left=0.105, right=0.995, top=0.80, bottom=0.34)
    fig.canvas.draw()
    first_pos = axes[0].get_position()

    fig.text(
        0.035,
        first_pos.y1 + 0.10,
        "Effect of instruction tuning on graded stability",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.text(
        0.05,
        (first_pos.y0 + first_pos.y1) / 2.0,
        r"Mean matched $\Delta\gamma(P)$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )

    family_handles = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="none", label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=6.1,
        handlelength=1.35,
        handletextpad=0.4,
        columnspacing=0.95,
        borderaxespad=0.0,
    )

    return save_figure(fig, output_dir=output_dir, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an SI figure for the matched base-vs-instruct effect on graded stability using the Direct Conditional estimator."
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
    )
    parser.add_argument(
        "--instruction_limit",
        type=float,
        default=None,
        help="Optional symmetric y-limit for the instruction-tuning effect (including error bars).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instruction_limit is not None and args.instruction_limit <= 0:
        raise ValueError("--instruction_limit must be positive.")

    summary_path = args.input_dir / "instruction_pair_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary_frame = pd.read_parquet(summary_path)

    detail_path = args.input_dir / "instruction_statement_deltas.parquet"
    detail_frame: pd.DataFrame | None
    if detail_path.exists():
        detail_frame = pd.read_parquet(detail_path)
    else:
        detail_frame = None

    primary = summary_frame.loc[
        summary_frame["probe"].astype(str).eq(args.probe)
        & summary_frame["dataset"].astype(str).isin(DATASETS)
        & summary_frame["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    print("=" * 100)
    print("FIGURE SI: EFFECT OF INSTRUCTION TUNING ON GRADED STABILITY (DIRECT)")
    print("=" * 100)
    print(f"Summary input:   {summary_path}")
    print(f"Detail input:    {detail_path if detail_frame is not None else '[not found]'}")
    print(f"Output:          {args.output_dir / (OUTPUT_STEM + '.pdf')}")
    print(f"Probe:           {args.probe}")
    print("Estimator:       Direct Conditional only")
    print("Error bars:      +/- 1 SE from statement-level matched deltas")
    print()

    if not primary.empty:
        summary = (
            primary.groupby(["dataset"], sort=True)
            .agg(
                n_pairs=("canonical_checkpoint", "nunique"),
                mean_effect=("mean_delta_gamma_matched", "mean"),
                median_effect=("mean_delta_gamma_matched", "median"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("=" * 100)

    make_figure(
        instruction_pairs=summary_frame,
        statement_deltas=detail_frame,
        probe=args.probe,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        instruction_limit=args.instruction_limit,
    )


if __name__ == "__main__":
    main()
