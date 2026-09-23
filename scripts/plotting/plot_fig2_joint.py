from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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

ESTIMATOR = "joint"

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

# Exact instruction-tuned checkpoint set used throughout the main-text figures.
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

# Text-decoder layer counts (num_hidden_layers / n_layers).
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

TOP_MODE_LABELS = {
    "bars": None,
    "relative": "Relative model scale",
    "parameters": "Parameters (B)",
    "layers": "Decoder layers",
}

TOP_MODE_OUTPUT_SUFFIX = {
    "bars": "bars",
    "relative": "relative_scale",
    "parameters": "parameter_scale",
    "layers": "decoder_layer_scale",
}

DISTRIBUTION_YLIM = (-0.40, 0.90)

BOTTOM_MODES = (
    "distribution",
    "heatmap",
)

GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"
MEDIAN_COLOR = "#2f2f2f"
ZERO_LINE_COLOR = "#777777"
RESIDUAL_DISTRIBUTION_COLOR = "#765B73"

RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residual_agreement",
    ["#3F6699", "#F7F5F0", "#A63D4D"],
    N=256,
)
RESIDUAL_CMAP.set_bad("#F4F4F4")

FIGSIZE = (7.2, 4.85)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def display_model(model: str) -> str:
    """Display instruction-tuned model names without '_' or '(i)'."""
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


def parameter_count_b(model: str) -> float:
    """Parse nominal parameter count in billions from a checkpoint name."""
    bare = display_model(model).lower()
    sizes = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            bare,
        )
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


def first_existing(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def resolve_r2_column(model_summary: pd.DataFrame) -> str:
    """Prefer the final held-out R^2 metric, with legacy fallbacks."""
    column = first_existing(
        model_summary,
        (
            "cv_r_squared",
            "spline_cv_r_squared",
            "spline_r_squared",
            "full_fit_r_squared",
        ),
    )
    if column is None:
        raise ValueError(
            "Could not identify the stability-vs-credence R^2 column in "
            "model_summary.parquet. Expected one of "
            "['cv_r_squared', 'spline_cv_r_squared', "
            "'spline_r_squared', 'full_fit_r_squared']."
        )
    return column


def resolve_observed_rho_column(cross_model: pd.DataFrame) -> str:
    column = first_existing(
        cross_model,
        (
            "residual_gamma_spearman_rho",
            "residual_spearman_rho",
            "spearman_residual_rho",
            "residual_rho",
        ),
    )
    if column is None:
        raise ValueError(
            "Could not identify pairwise residual-correlation column in "
            "cross_model_agreement.parquet."
        )
    return column


def resolve_model_pair_columns(cross_model: pd.DataFrame) -> tuple[str, str]:
    """
    Resolve the two model identifiers used by each pairwise correlation row.

    Exact known schemas are preferred. A conservative heuristic is used only
    when exactly two model/checkpoint-like columns remain.
    """
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
        if name in {
            "n_models",
            "num_models",
            "model_family",
            "family_model",
        }:
            continue
        candidates.append(str(column))

    if len(candidates) == 2:
        return candidates[0], candidates[1]

    raise ValueError(
        "Could not unambiguously identify the two model columns in "
        "cross_model_agreement.parquet. Tried common pairs such as "
        "(model_a, model_b), (model_i, model_j), and (model_1, model_2). "
        f"Available columns: {cross_model.columns.tolist()}"
    )


def style_axis(
    ax: plt.Axes,
    *,
    x_grid: bool = False,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=5.4)
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    log.info("Wrote %s", path)
    return path


def default_output_stem(
    *,
    top_mode: str,
    bottom_mode: str,
) -> str:
    stem = f"figure_si_2_{TOP_MODE_OUTPUT_SUFFIX[top_mode]}_joint"
    if bottom_mode == "heatmap":
        stem += "_heatmap"
    return stem


# ---------------------------------------------------------------------------
# Top row: held-out R^2
# ---------------------------------------------------------------------------


def prepare_family_r2_trajectory(
    panel: pd.DataFrame,
    *,
    family: str,
    r2_col: str,
    top_mode: str,
) -> pd.DataFrame:
    family_panel = panel.loc[
        panel["model"].astype(str).map(model_family).eq(family)
    ].copy()

    if family_panel.empty:
        return family_panel

    family_panel[r2_col] = pd.to_numeric(
        family_panel[r2_col],
        errors="coerce",
    )
    family_panel = (
        family_panel.groupby("model", as_index=False)
        .agg(r2=(r2_col, "mean"))
    )
    family_panel["params_b"] = family_panel["model"].map(parameter_count_b)
    family_panel = family_panel.loc[
        np.isfinite(family_panel["r2"])
        & np.isfinite(family_panel["params_b"])
    ].copy()

    if family_panel.empty:
        return family_panel

    family_panel = family_panel.sort_values(
        ["params_b", "model"],
        kind="stable",
    ).reset_index(drop=True)

    if top_mode == "relative":
        n_models = len(family_panel)
        if n_models == 1:
            x_values = np.array([1.0], dtype=float)
        elif n_models == 2:
            x_values = np.array([0.0, 2.0], dtype=float)
        elif n_models == 3:
            x_values = np.array([0.0, 1.0, 2.0], dtype=float)
        else:
            x_values = np.linspace(0.0, 2.0, n_models)

    elif top_mode == "parameters":
        x_values = family_panel["params_b"].to_numpy(dtype=float)

    elif top_mode == "layers":
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
        raise ValueError(f"Unsupported trajectory top_mode={top_mode!r}.")

    family_panel["scale_x"] = x_values
    return family_panel


def configure_top_scale_axis(
    ax: plt.Axes,
    *,
    top_mode: str,
) -> None:
    if top_mode == "relative":
        ax.set_xlim(-0.15, 2.15)
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_xticklabels(
            ["Small", "Medium", "Large"],
            fontsize=5.8,
        )
    elif top_mode == "parameters":
        ax.set_xlim(0.0, 82.0)
        ax.set_xticks([0, 20, 40, 60, 80])
    elif top_mode == "layers":
        ax.set_xlim(0.0, 84.0)
        ax.set_xticks([0, 20, 40, 60, 80])
    else:
        raise ValueError(f"Unsupported trajectory top_mode={top_mode!r}.")


def plot_r2_bars(
    ax: plt.Axes,
    *,
    panel: pd.DataFrame,
    r2_col: str,
    r2_max: float,
) -> None:
    models = ordered_models()
    x = np.arange(1, len(models) + 1, dtype=float)

    indexed = (
        panel.assign(model=lambda d: d["model"].astype(str))
        .drop_duplicates("model")
        .set_index("model")
        .reindex(models)
    )
    values = pd.to_numeric(
        indexed[r2_col],
        errors="coerce",
    ).to_numpy(dtype=float, copy=True)

    colors = [FAMILY_COLORS[model_family(model)] for model in models]
    ax.bar(
        x,
        values,
        width=0.72,
        color=colors,
        edgecolor="none",
        zorder=3,
    )
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


def plot_r2_scale(
    ax: plt.Axes,
    *,
    panel: pd.DataFrame,
    r2_col: str,
    top_mode: str,
    r2_max: float,
) -> None:
    for family in FAMILY_ORDER:
        trajectory = prepare_family_r2_trajectory(
            panel,
            family=family,
            r2_col=r2_col,
            top_mode=top_mode,
        )
        if trajectory.empty:
            continue

        x = trajectory["scale_x"].to_numpy(dtype=float)
        y = trajectory["r2"].to_numpy(dtype=float)
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

    configure_top_scale_axis(ax, top_mode=top_mode)
    ax.set_ylim(0.0, r2_max)
    ax.set_yticks(np.arange(0.0, r2_max + 1e-9, 0.2))
    style_axis(ax, x_grid=(top_mode != "relative"))
    ax.set_xlabel(
        TOP_MODE_LABELS[top_mode],
        fontsize=6.2,
        labelpad=2.5,
    )


# ---------------------------------------------------------------------------
# Bottom row: pairwise residual-correlation heatmaps
# ---------------------------------------------------------------------------


def build_residual_matrix(
    panel: pd.DataFrame,
    *,
    left_model_col: str,
    right_model_col: str,
    rho_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a symmetric residual-correlation matrix and its finite upper-triangle
    pairwise values, both ordered according to EXPECTED_MODELS.
    """
    models = ordered_models()
    model_to_index = {model: idx for idx, model in enumerate(models)}
    n = len(models)

    pair_values: dict[tuple[int, int], list[float]] = {}

    for _, row in panel.iterrows():
        left = str(row[left_model_col])
        right = str(row[right_model_col])

        if left not in model_to_index or right not in model_to_index:
            continue
        if left == right:
            continue

        rho = pd.to_numeric(
            pd.Series([row[rho_col]]),
            errors="coerce",
        ).iloc[0]
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


def plot_residual_heatmap(
    ax: plt.Axes,
    *,
    matrix: np.ndarray,
    pair_values: np.ndarray,
    rho_abs_max: float,
    show_y_labels: bool,
) -> Any:
    models = ordered_models()
    display = [display_model(model) for model in models]

    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked,
        cmap=RESIDUAL_CMAP,
        vmin=-rho_abs_max,
        vmax=rho_abs_max,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )

    positions = np.arange(len(models))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        display,
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=4.7,
    )
    ax.set_yticks(positions)

    if show_y_labels:
        ax.set_yticklabels(display, fontsize=4.7)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    # Thin separators make the four 3-model family blocks legible without adding
    # another legend or explicit family labels inside the heatmap.
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color="white", linewidth=0.8, alpha=0.95)
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.95)

    ax.tick_params(axis="x", length=2.0, pad=1.0)
    ax.tick_params(axis="y", length=2.0, pad=1.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if len(pair_values):
        median = float(np.median(pair_values))
        ax.text(
            0.99,
            1.015,
            rf"median $\rho$ = {median:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color=TITLE_COLOR,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No valid model pairs",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
            color="#666666",
        )

    return image



def plot_residual_distribution(
    ax: plt.Axes,
    *,
    residual: pd.DataFrame,
    left_model_col: str,
    right_model_col: str,
    rho_col: str,
    y_limits: tuple[float, float],
) -> dict[str, np.ndarray]:
    """
    Plot the three dataset-level distributions of pairwise residual agreement in
    one shared panel.

    This intentionally keeps the pairwise model correlations as the unit of
    observation. The violin summarizes their distribution, lightly jittered
    points expose the underlying model-pair values, and the horizontal black
    segment marks the median.
    """
    values_by_dataset: dict[str, np.ndarray] = {}
    positions = np.arange(1, len(DATASETS) + 1, dtype=float)

    for dataset_index, (position, dataset) in enumerate(
        zip(positions, DATASETS)
    ):
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

        # Reproducible jitter lets readers see the model-pair observations
        # without making the figure change between runs.
        rng = np.random.default_rng(1729 + dataset_index)
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


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def make_figure(
    *,
    model_summary: pd.DataFrame,
    cross_model: pd.DataFrame,
    probe: str,
    top_mode: str,
    bottom_mode: str,
    output_dir: Path,
    stem: str,
    overwrite: bool,
    r2_max: float,
    rho_abs_max: float,
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
            f"No instruction-tuned Joint-to-Conditional model-summary rows for probe={probe!r}."
        )
    if residual.empty:
        raise RuntimeError(
            f"No instruction-tuned Joint-to-Conditional cross-model rows for probe={probe!r}."
        )

    observed_r2 = pd.to_numeric(
        top[r2_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    observed_r2 = observed_r2[np.isfinite(observed_r2)]
    if len(observed_r2) and float(np.max(observed_r2)) > r2_max:
        raise ValueError(
            f"--r2_max={r2_max:g} clips observed R^2 values "
            f"(max={np.max(observed_r2):.3f}). Increase --r2_max."
        )

    observed_rho = pd.to_numeric(
        residual[rho_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    observed_rho = observed_rho[np.isfinite(observed_rho)]
    if len(observed_rho) and float(np.max(np.abs(observed_rho))) > rho_abs_max:
        raise ValueError(
            f"--rho_abs_max={rho_abs_max:g} clips observed residual correlations "
            f"(need at least {np.max(np.abs(observed_rho)):.3f})."
        )

    # Bar-mode model labels need more room between rows than the scale variants.
    row_space = 0.82

    distribution_ymin, distribution_ymax = DISTRIBUTION_YLIM

    if bottom_mode == "heatmap":
        figsize = (7.2, 4.85)
        height_ratios = [0.72, 1.0]
        right_margin = 0.91
        bottom_margin = 0.12
    else:
        observed_distribution_min = float(np.min(observed_rho)) if len(observed_rho) else 0.0
        observed_distribution_max = float(np.max(observed_rho)) if len(observed_rho) else 0.0
        if observed_distribution_min < distribution_ymin or observed_distribution_max > distribution_ymax:
            raise ValueError(
                "Distribution y-limits clip observed residual correlations "
                f"(observed range [{observed_distribution_min:.3f}, {observed_distribution_max:.3f}], "
                f"configured range [{distribution_ymin:.3f}, {distribution_ymax:.3f}])."
            )

        figsize = (7.2, 3.95 if top_mode == "bars" else 3.6)
        height_ratios = [1.0, 0.55]
        right_margin = 0.995
        bottom_margin = 0.14

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        height_ratios=height_ratios,
        wspace=0.17,
        hspace=row_space,
    )

    top_axes: list[plt.Axes] = []

    for col_index, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, col_index])
        top_axes.append(ax)

        panel = top.loc[
            top["dataset"].astype(str).eq(dataset)
        ].copy()

        if top_mode == "bars":
            plot_r2_bars(
                ax,
                panel=panel,
                r2_col=r2_col,
                r2_max=r2_max,
            )
        else:
            plot_r2_scale(
                ax,
                panel=panel,
                r2_col=r2_col,
                top_mode=top_mode,
                r2_max=r2_max,
            )

        ax.set_title(
            DATASET_NAMES[dataset],
            loc="center",
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
            y=1.10,
            pad=0,
        )
        ax.text(
            0.0,
            1.025,
            ("(a)", "(b)", "(c)")[col_index],
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

    heatmap_image = None
    bottom_axes: list[plt.Axes] = []

    if bottom_mode == "distribution":
        ax = fig.add_subplot(gs[1, :])
        bottom_axes.append(ax)
        plot_residual_distribution(
            ax,
            residual=residual,
            left_model_col=left_model_col,
            right_model_col=right_model_col,
            rho_col=rho_col,
            y_limits=DISTRIBUTION_YLIM,
        )
        ax.text(
            0.0,
            1.025,
            "(d)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    elif bottom_mode == "heatmap":
        for col_index, dataset in enumerate(DATASETS):
            ax = fig.add_subplot(gs[1, col_index])
            bottom_axes.append(ax)

            pair_panel = residual.loc[
                residual["dataset"].astype(str).eq(dataset)
            ].copy()
            matrix, pair_values = build_residual_matrix(
                pair_panel,
                left_model_col=left_model_col,
                right_model_col=right_model_col,
                rho_col=rho_col,
            )
            heatmap_image = plot_residual_heatmap(
                ax,
                matrix=matrix,
                pair_values=pair_values,
                rho_abs_max=rho_abs_max,
                show_y_labels=(col_index == 0),
            )
            ax.text(
                0.0,
                1.015,
                ("(d)", "(e)", "(f)")[col_index],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=TITLE_COLOR,
            )
    else:
        raise ValueError(f"Unknown bottom_mode={bottom_mode!r}.")

    fig.subplots_adjust(
        left=0.105 if bottom_mode == "distribution" else 0.115,
        right=right_margin,
        top=0.91,
        bottom=bottom_margin,
    )
    fig.canvas.draw()

    if bottom_mode == "heatmap":
        if heatmap_image is None:
            raise RuntimeError("Heatmap image was not created.")

        bottom_positions = [ax.get_position() for ax in bottom_axes]
        heat_y0 = min(pos.y0 for pos in bottom_positions)
        heat_y1 = max(pos.y1 for pos in bottom_positions)
        cax = fig.add_axes([
            0.925,
            heat_y0,
            0.012,
            heat_y1 - heat_y0,
        ])
        cbar = fig.colorbar(
            heatmap_image,
            cax=cax,
            orientation="vertical",
        )
        cbar.set_label(
            r"Residual Spearman $\rho$",
            fontsize=6.3,
            labelpad=4.0,
        )
        cbar.ax.tick_params(labelsize=5.7, length=2.5)
        cbar.outline.set_linewidth(0.45)

    # Family legend applies to the top row only.
    if top_mode == "bars":
        legend_handles = [
            Patch(
                facecolor=FAMILY_COLORS[family],
                edgecolor="none",
                label=FAMILY_LABELS[family],
            )
            for family in FAMILY_ORDER
        ]
    else:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=FAMILY_COLORS[family],
                marker="o",
                markersize=3.3,
                linewidth=1.05,
                label=FAMILY_LABELS[family],
            )
            for family in FAMILY_ORDER
        ]

    if bottom_mode == "distribution":
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MEDIAN_COLOR,
                linewidth=1.25,
                label=r"Median Residual $\rho$",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.015),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=6.3,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    top_pos = top_axes[0].get_position()
    bottom_pos = bottom_axes[0].get_position()

    # Figure-level top-row y label. The distribution version owns its y label
    # directly because it is one wide panel; the heatmap colorbar carries rho.
    y_label_x = 0.05
    fig.text(
        y_label_x,
        (top_pos.y0 + top_pos.y1) / 2.0,
        r"Held-out $R^2$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )

    path = save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        overwrite=overwrite,
    )
    plt.close(fig)

    log.info("Top-row metric: %s", r2_col)
    log.info("Bottom-row metric: %s", rho_col)
    log.info(
        "Cross-model pair columns: %s, %s",
        left_model_col,
        right_model_col,
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 2 using instruction-tuned Joint-to-Conditional models: held-out "
            "stability-vs-credence R^2 plus cross-model residual agreement."
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
        "--probe",
        default="sawmil",
        choices=["sawmil", "svm", "mean_difference"],
        help="Probe shown in the figure. Main-text default: sawmil.",
    )
    parser.add_argument(
        "--top_mode",
        default="bars",
        choices=["bars", "relative", "parameters", "layers"],
        help=(
            "Top-row x-axis encoding: bars=12 checkpoint bars (default), "
            "relative=Small/Medium/Large family trajectories, "
            "parameters=parameter count in billions, "
            "layers=decoder-layer count."
        ),
    )
    parser.add_argument(
        "--bottom_mode",
        default="distribution",
        choices=list(BOTTOM_MODES),
        help=(
            "Residual-agreement display. distribution (default) shows the "
            "three dataset-level model-pair distributions in one wide panel; "
            "heatmap shows three symmetric model-by-model matrices."
        ),
    )
    parser.add_argument(
        "--stem",
        default=None,
        help=(
            "Optional output stem. If omitted, the filename is generated from "
            "--top_mode; heatmap variants append '_heatmap'."
        ),
    )
    parser.add_argument(
        "--r2_max",
        type=float,
        default=0.8,
        help=(
            "Upper y-limit for held-out R^2 panels. Default: 0.8; "
            "use 1.0 for the full conventional range."
        ),
    )
    parser.add_argument(
        "--rho_abs_max",
        type=float,
        default=0.5,
        help=(
            "Symmetric residual-Spearman limit. Used as the y-limit for the "
            "distribution panel and the color limit for heatmaps. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.r2_max <= 0:
        raise ValueError("--r2_max must be positive.")
    if args.rho_abs_max <= 0 or args.rho_abs_max > 1:
        raise ValueError("--rho_abs_max must be in (0, 1].")

    stem = (
        args.stem
        if args.stem is not None
        else default_output_stem(
            top_mode=args.top_mode,
            bottom_mode=args.bottom_mode,
        )
    )

    paths = {
        "model_summary": args.input_dir / "model_summary.parquet",
        "cross_model": args.input_dir / "cross_model_agreement.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    model_summary = pd.read_parquet(paths["model_summary"])
    cross_model = pd.read_parquet(paths["cross_model"])

    r2_col = resolve_r2_column(model_summary)
    rho_col = resolve_observed_rho_column(cross_model)
    pair_cols = resolve_model_pair_columns(cross_model)

    print("=" * 100)
    print("FIGURE 2: STABILITY VS. ATOMIC CREDENCE")
    print("=" * 100)
    print(f"Input:       {args.input_dir}")
    print(f"Output:      {args.output_dir / (stem + '.pdf')}")
    print(f"Probe:       {args.probe}")
    print(f"Estimator:   {ESTIMATOR}")
    print("Models:      instruction-tuned only")
    print(f"Top mode:    {args.top_mode}")
    print(f"Bottom mode: {args.bottom_mode}")
    print(f"R^2 column:  {r2_col}")
    print(f"Rho column:  {rho_col}")
    print(f"Pair cols:   {pair_cols[0]}, {pair_cols[1]}")
    print("=" * 100)

    make_figure(
        model_summary=model_summary,
        cross_model=cross_model,
        probe=args.probe,
        top_mode=args.top_mode,
        bottom_mode=args.bottom_mode,
        output_dir=args.output_dir,
        stem=stem,
        overwrite=args.overwrite,
        r2_max=args.r2_max,
        rho_abs_max=args.rho_abs_max,
    )


if __name__ == "__main__":
    main()
