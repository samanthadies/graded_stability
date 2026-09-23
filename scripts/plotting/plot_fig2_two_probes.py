from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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

PROBE_SPECS = (
    ("svm", "SVM"),
    ("mean_difference", "Mass Mean"),
)

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
MEDIAN_COLOR = "#2f2f2f"
ZERO_LINE_COLOR = "#777777"
RESIDUAL_DISTRIBUTION_COLOR = "#765B73"

FIGSIZE = (7.2, 6)


def display_model(model: str) -> str:
    return str(model).lstrip("_")


def model_family(model: str) -> str:
    bare = display_model(model).lower()
    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family
    raise ValueError(
        f"Unrecognized model family for {model!r}; expected one of {list(FAMILY_ORDER)}."
    )


def parameter_count_b(model: str) -> float:
    bare = display_model(model).lower()
    sizes = [
        float(value)
        for value in re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])", bare)
    ]
    if not sizes:
        raise ValueError(f"Could not parse parameter count from {model!r}.")
    return max(sizes)


def model_sort_key(model: str) -> tuple[Any, ...]:
    return (
        FAMILY_ORDER.index(model_family(model)),
        parameter_count_b(model),
        display_model(model).lower(),
    )


def ordered_models() -> list[str]:
    return sorted(EXPECTED_MODELS, key=model_sort_key)


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


def resolve_r2_column(model_summary: pd.DataFrame) -> str:
    column = first_existing(
        model_summary,
        ("cv_r_squared", "spline_cv_r_squared", "spline_r_squared", "full_fit_r_squared"),
    )
    if column is None:
        raise ValueError("Could not identify the stability-vs-credence R^2 column in model_summary.parquet.")
    return column


def resolve_observed_rho_column(cross_model: pd.DataFrame) -> str:
    column = first_existing(
        cross_model,
        ("residual_gamma_spearman_rho", "residual_spearman_rho", "spearman_residual_rho", "residual_rho"),
    )
    if column is None:
        raise ValueError("Could not identify pairwise residual-correlation column in cross_model_agreement.parquet.")
    return column


def resolve_model_pair_columns(cross_model: pd.DataFrame) -> tuple[str, str]:
    exact_pairs = (
        ("model_a", "model_b"),
        ("model_1", "model_2"),
        ("model_i", "model_j"),
        ("model_x", "model_y"),
        ("model_left", "model_right"),
        ("left_model", "right_model"),
        ("model1", "model2"),
        ("checkpoint_a", "checkpoint_b"),
        ("checkpoint_1", "checkpoint_2"),
    )
    for left, right in exact_pairs:
        if left in cross_model.columns and right in cross_model.columns:
            return left, right

    candidates = []
    for column in cross_model.columns:
        name = str(column).lower()
        if "model" not in name and "checkpoint" not in name:
            continue
        if name in {"n_models", "num_models", "model_family", "family_model"}:
            continue
        candidates.append(str(column))

    if len(candidates) == 2:
        return candidates[0], candidates[1]

    raise ValueError(
        "Could not unambiguously identify the two model columns in cross_model_agreement.parquet. "
        f"Available columns: {cross_model.columns.tolist()}"
    )


def style_axis(ax: plt.Axes, *, x_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(axis="x", labelsize=5.2)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.85)
    if x_grid:
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.35, alpha=0.45)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, *, output_dir: Path, stem: str, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    log.info("Wrote %s", path)
    return path


def plot_r2_bars(ax: plt.Axes, *, panel: pd.DataFrame, r2_col: str, r2_max: float) -> None:
    models = ordered_models()
    x = np.arange(1, len(models) + 1, dtype=float)

    indexed = (
        panel.assign(model=lambda d: d["model"].astype(str))
        .drop_duplicates("model")
        .set_index("model")
        .reindex(models)
    )
    values = pd.to_numeric(indexed[r2_col], errors="coerce").to_numpy(dtype=float, copy=True)

    colors = [FAMILY_COLORS[model_family(model)] for model in models]
    ax.bar(x, values, width=0.72, color=colors, edgecolor="none", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_model(model) for model in models],
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(0.35, len(models) + 0.65)
    ax.set_ylim(0.0, r2_max)
    ax.set_yticks(np.arange(0.0, r2_max + 1e-9, 0.2))
    style_axis(ax)


def build_residual_matrix(
    panel: pd.DataFrame,
    *,
    left_model_col: str,
    right_model_col: str,
    rho_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    models = ordered_models()
    model_to_index = {model: idx for idx, model in enumerate(models)}
    n = len(models)
    pair_values: dict[tuple[int, int], list[float]] = {}

    for _, row in panel.iterrows():
        left = str(row[left_model_col])
        right = str(row[right_model_col])
        if left not in model_to_index or right not in model_to_index or left == right:
            continue

        rho = pd.to_numeric(pd.Series([row[rho_col]]), errors="coerce").iloc[0]
        if not np.isfinite(rho):
            continue

        i = model_to_index[left]
        j = model_to_index[right]
        key = (min(i, j), max(i, j))
        pair_values.setdefault(key, []).append(float(rho))

    matrix = np.full((n, n), np.nan, dtype=float)
    for (i, j), values in pair_values.items():
        value = float(np.mean(values))
        matrix[i, j] = value
        matrix[j, i] = value

    upper = matrix[np.triu_indices(n, k=1)]
    upper = upper[np.isfinite(upper)]
    return matrix, upper


def plot_residual_distribution(
    ax: plt.Axes,
    *,
    residual: pd.DataFrame,
    left_model_col: str,
    right_model_col: str,
    rho_col: str,
    y_limits: tuple[float, float],
    jitter_seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    """Plot all three dataset-level residual-agreement distributions in one panel."""
    values_by_dataset: dict[str, np.ndarray] = {}
    positions = np.arange(1, len(DATASETS) + 1, dtype=float)

    for dataset_index, (position, dataset) in enumerate(zip(positions, DATASETS)):
        pair_panel = residual.loc[
            residual["dataset"].astype(str).eq(dataset)
        ].copy()

        _, values = build_residual_matrix(
            pair_panel,
            left_model_col=left_model_col,
            right_model_col=right_model_col,
            rho_col=rho_col,
        )
        values_by_dataset[dataset] = values

        if len(values) == 0:
            continue

        if len(values) >= 2 and np.unique(values).size >= 2:
            violin = ax.violinplot(
                [values],
                positions=[position],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
                points=100,
            )
            for body in violin["bodies"]:
                body.set_facecolor(RESIDUAL_DISTRIBUTION_COLOR)
                body.set_edgecolor("none")
                body.set_alpha(0.32)
                body.set_zorder(2)

        rng = np.random.default_rng(
            1729 + jitter_seed_offset + dataset_index
        )
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=7.0,
            color=RESIDUAL_DISTRIBUTION_COLOR,
            alpha=0.34,
            edgecolors="none",
            zorder=3,
        )

        median = float(np.median(values))
        ax.hlines(
            median,
            position - 0.23,
            position + 0.23,
            color=MEDIAN_COLOR,
            linewidth=1.25,
            zorder=5,
        )

        y_min, y_max = y_limits
        y_span = y_max - y_min
        ax.text(
            position,
            y_max - 0.06 * y_span,
            rf"median $\rho$ = {median:.2f}",
            ha="center",
            va="top",
            fontsize=5.8,
            color=TITLE_COLOR,
        )

    ax.axhline(
        0.0,
        color=ZERO_LINE_COLOR,
        linewidth=0.65,
        linestyle="--",
        zorder=0,
    )
    ax.set_xlim(0.45, len(DATASETS) + 0.55)
    ax.set_ylim(*y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [DATASET_NAMES[dataset] for dataset in DATASETS],
        fontsize=6.4,
    )
    ax.set_ylabel(
        r"Residual $\rho$",
        fontsize=7,
        labelpad=5.0,
    )
    style_axis(ax)

    return values_by_dataset


def make_figure(
    *,
    model_summary: pd.DataFrame,
    cross_model: pd.DataFrame,
    output_dir: Path,
    stem: str,
    overwrite: bool,
    r2_max: float,
    rho_min: float,
    rho_max: float,
) -> Path:
    require_columns(
        model_summary,
        ("model", "dataset", "probe", "estimator"),
        source="model_summary.parquet",
    )
    require_columns(
        cross_model,
        ("dataset", "probe", "estimator"),
        source="cross_model_agreement.parquet",
    )

    r2_col = resolve_r2_column(model_summary)
    rho_col = resolve_observed_rho_column(cross_model)
    left_model_col, right_model_col = resolve_model_pair_columns(cross_model)

    expected_set = set(EXPECTED_MODELS)
    # distribution_ylim = (-rho_abs_max, rho_abs_max)
    distribution_ylim = (rho_min, rho_max)

    observed_r2: list[float] = []
    observed_rho: list[float] = []
    probe_top: dict[str, pd.DataFrame] = {}
    probe_residual: dict[str, pd.DataFrame] = {}

    for probe, _ in PROBE_SPECS:
        top = model_summary.loc[
            model_summary["probe"].astype(str).eq(probe)
            & model_summary["dataset"].astype(str).isin(DATASETS)
            & model_summary["estimator"].astype(str).eq(ESTIMATOR)
            & model_summary["model"].astype(str).isin(expected_set)
        ].copy()

        residual = cross_model.loc[
            cross_model["probe"].astype(str).eq(probe)
            & cross_model["dataset"].astype(str).isin(DATASETS)
            & cross_model["estimator"].astype(str).eq(ESTIMATOR)
            & cross_model[left_model_col].astype(str).isin(expected_set)
            & cross_model[right_model_col].astype(str).isin(expected_set)
        ].copy()

        if top.empty:
            raise RuntimeError(
                f"No instruction-tuned Direct model-summary rows for probe={probe!r}."
            )
        if residual.empty:
            raise RuntimeError(
                f"No instruction-tuned Direct cross-model rows for probe={probe!r}."
            )

        probe_top[probe] = top
        probe_residual[probe] = residual

        vals_r2 = pd.to_numeric(top[r2_col], errors="coerce").to_numpy(dtype=float)
        vals_r2 = vals_r2[np.isfinite(vals_r2)]
        observed_r2.extend(vals_r2.tolist())

        vals_rho = pd.to_numeric(residual[rho_col], errors="coerce").to_numpy(dtype=float)
        vals_rho = vals_rho[np.isfinite(vals_rho)]
        observed_rho.extend(vals_rho.tolist())

    if observed_r2 and float(np.max(observed_r2)) > r2_max:
        raise ValueError(
            f"--r2_max={r2_max:g} clips observed R^2 values "
            f"(max={np.max(observed_r2):.3f}). Increase --r2_max."
        )

    if observed_rho:
        obs_min = float(np.min(observed_rho))
        obs_max = float(np.max(observed_rho))
        if obs_min < distribution_ylim[0] or obs_max > distribution_ylim[1]:
            raise ValueError(
                "Distribution y-limits clip observed residual correlations "
                f"(observed range [{obs_min:.3f}, {obs_max:.3f}], configured range "
                f"[{distribution_ylim[0]:.3f}, {distribution_ylim[1]:.3f}])."
            )

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=[1.0, 0.65, 1.0, 0.65],
        wspace=0.17,
        hspace=0.9,
    )

    top_axes_by_probe: dict[str, list[plt.Axes]] = {}
    bottom_axes_by_probe: dict[str, plt.Axes] = {}

    # Panel labels are deliberately grouped as (a-c), (d), (e-g), (h).
    top_panel_labels = {
        "svm": ("(a)", "(b)", "(c)"),
        "mean_difference": ("(e)", "(f)", "(g)"),
    }
    bottom_panel_labels = {
        "svm": "(d)",
        "mean_difference": "(h)",
    }

    for probe_idx, (probe, probe_label) in enumerate(PROBE_SPECS):
        top_row = 2 * probe_idx
        bottom_row = top_row + 1
        top = probe_top[probe]
        residual = probe_residual[probe]

        top_axes: list[plt.Axes] = []
        for col_idx, dataset in enumerate(DATASETS):
            ax_top = fig.add_subplot(gs[top_row, col_idx])
            top_axes.append(ax_top)

            panel = top.loc[
                top["dataset"].astype(str).eq(dataset)
            ].copy()
            plot_r2_bars(
                ax_top,
                panel=panel,
                r2_col=r2_col,
                r2_max=r2_max,
            )
            ax_top.set_title(
                DATASET_NAMES[dataset],
                loc="center",
                fontsize=7.0,
                fontweight="bold",
                color=TITLE_COLOR,
                y=1.08,
                pad=0,
            )
            ax_top.text(
                0.0,
                1.02,
                top_panel_labels[probe][col_idx],
                transform=ax_top.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                color=TITLE_COLOR,
            )

            if col_idx != 0:
                ax_top.tick_params(axis="y", labelleft=False)
                ax_top.spines["left"].set_visible(False)

        ax_bottom = fig.add_subplot(gs[bottom_row, :])
        plot_residual_distribution(
            ax_bottom,
            residual=residual,
            left_model_col=left_model_col,
            right_model_col=right_model_col,
            rho_col=rho_col,
            y_limits=distribution_ylim,
            jitter_seed_offset=100 * probe_idx,
        )
        ax_bottom.text(
            0.0,
            1.025,
            bottom_panel_labels[probe],
            transform=ax_bottom.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=TITLE_COLOR,
        )

        top_axes_by_probe[probe] = top_axes
        bottom_axes_by_probe[probe] = ax_bottom

    legend_handles = [
        Patch(
            facecolor=FAMILY_COLORS[family],
            edgecolor="none",
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=MEDIAN_COLOR,
            linewidth=1.25,
            label=r"Median Residual $\rho$",
        )
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.94,
        bottom=0.075,
    )
    fig.canvas.draw()

    # Match the single-probe figures: each probe block has the same two row titles.
    row_title_x = 0.035
    y_label_x = 0.05
    for probe, probe_label in PROBE_SPECS:
        top_pos = top_axes_by_probe[probe][0].get_position()
        bottom_pos = bottom_axes_by_probe[probe].get_position()

        fig.text(
            row_title_x,
            top_pos.y1 + 0.05,
            f"{probe_label}",
            ha="left",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=TITLE_COLOR,
        )
        fig.text(
            y_label_x,
            (top_pos.y0 + top_pos.y1) / 2.0,
            r"Held-out $R^2$",
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.012),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=6.1,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    path = save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        overwrite=overwrite,
    )
    plt.close(fig)

    log.info("R^2 column: %s", r2_col)
    log.info("Residual-rho column: %s", rho_col)
    log.info("Cross-model pair columns: %s, %s", left_model_col, right_model_col)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the SI Figure 2 variant comparing SVM and Mass Mean using "
            "instruction-tuned Direct models, with one wide residual panel per probe."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/analysis/stability_vs_credence"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    parser.add_argument(
        "--stem",
        default="figure_si_2_bars_svm_mass_mean",
        help="Output stem (default: figure_si_2_bars_svm_mass_mean).",
    )
    parser.add_argument(
        "--r2_max",
        type=float,
        default=0.8,
        help="Upper y-limit for held-out R^2 panels. Default: 0.8.",
    )
    parser.add_argument(
        "--rho_min",
        type=float,
        default=-0.3,
        help="Lower y-limit for residual-distribution panels. Default: -0.3.",
    )
    parser.add_argument(
        "--rho_max",
        type=float,
        default=0.8,
        help="Upper y-limit for residual-distribution panels. Default: 0.8.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.r2_max <= 0:
        raise ValueError("--r2_max must be positive.")
    if args.rho_min >= args.rho_max:
        raise ValueError("--rho_min must be smaller than --rho_max.")

    paths = {
        "model_summary": args.input_dir / "model_summary.parquet",
        "cross_model": args.input_dir / "cross_model_agreement.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    model_summary = pd.read_parquet(paths["model_summary"])
    cross_model = pd.read_parquet(paths["cross_model"])

    print("=" * 100)
    print("FIGURE SI 2 VARIANT: SVM + MASS MEAN")
    print("=" * 100)
    print(f"Input:       {args.input_dir}")
    print(f"Output:      {args.output_dir / (args.stem + '.pdf')}")
    print(f"Estimator:   {ESTIMATOR}")
    print("Models:      instruction-tuned only")
    print(f"Probes:      {', '.join(label for _, label in PROBE_SPECS)}")
    print(f"r2_max:      {args.r2_max}")
    print(f"rho_ylim:    [{args.rho_min}, {args.rho_max}]")
    print("=" * 100)

    make_figure(
        model_summary=model_summary,
        cross_model=cross_model,
        output_dir=args.output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
        r2_max=args.r2_max,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
    )


if __name__ == "__main__":
    main()
