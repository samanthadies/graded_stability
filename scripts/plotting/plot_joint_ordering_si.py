"""
Generates the SI sensitivity figure comparing the Joint-to-Conditional estimator under
"x and P" versus "P and x" conjunction order at both pair and proposition levels.

Example:
    python -m scripts.plotting.plot_joint_ordering_si --overwrite
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

PROBE = "sawmil"

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
BAR_ALPHA = 0.95
ERROR_COLOR = "#2f2f2f"
FIGSIZE = (7.2, 3.0)
OUTPUT_STEM = "figure_si_joint_ordering"


# ---------------------------------------------------------------------------
# Helpers
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
        f"Unrecognized model family for {model!r}; expected one of {list(FAMILY_ORDER)}."
    )



def model_sort_key(model: str) -> tuple[Any, ...]:
    bare = display_model(model).lower()
    family_rank = FAMILY_ORDER.index(model_family(model))
    sizes = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            bare,
        )
    ]
    size = max(sizes) if sizes else float("inf")
    return (family_rank, size, bare)



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



def save_figure(fig: plt.Figure, *, output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{OUTPUT_STEM}.pdf"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    log.info("Wrote %s", path)
    return path



def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    working = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    working = working.loc[np.isfinite(working["x"]) & np.isfinite(working["y"])].copy()
    if len(working) < 2:
        return np.nan
    if working["x"].nunique(dropna=True) < 2 or working["y"].nunique(dropna=True) < 2:
        return np.nan
    return float(working["x"].corr(working["y"], method="spearman"))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def bootstrap_median_se(
    frame: pd.DataFrame,
    *,
    value_col: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    cluster_col: str | None = None,
) -> tuple[float, float, int]:
    """Return observed median, bootstrap SE of the median, and resampling-unit count."""
    required = [value_col]
    if cluster_col is not None:
        required.append(cluster_col)
    require_columns(frame, required, source="bootstrap cell")

    work = frame[required].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[np.isfinite(work[value_col].to_numpy(dtype=float))].copy()
    if cluster_col is not None:
        work = work.loc[work[cluster_col].notna()].copy()

    if work.empty:
        return np.nan, np.nan, 0

    observed = float(work[value_col].median())

    if cluster_col is None:
        values = work[value_col].to_numpy(dtype=float)
        n_units = len(values)
        if n_units <= 1:
            return observed, 0.0, n_units
        boot = np.empty(n_bootstrap, dtype=float)
        chunk = 1000
        filled = 0
        while filled < n_bootstrap:
            current = min(chunk, n_bootstrap - filled)
            idx = rng.integers(0, n_units, size=(current, n_units))
            boot[filled:filled+current] = np.median(values[idx], axis=1)
            filled += current
        return observed, float(np.std(boot, ddof=1)), n_units

    clusters = [g[value_col].to_numpy(dtype=float) for _, g in work.groupby(cluster_col, sort=True)]
    n_units = len(clusters)
    if n_units <= 1:
        return observed, 0.0, n_units

    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_units, size=n_units)
        values = np.concatenate([clusters[i] for i in sampled])
        boot[b] = float(np.median(values))
    return observed, float(np.std(boot, ddof=1)), n_units


def prepare_pair_summary(
    model_summary: pd.DataFrame,
    pair_level: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    require_columns(
        model_summary,
        ("model", "dataset", "probe", "median_abs_conditional_order_difference"),
        source="model_summary.parquet",
    )
    require_columns(
        pair_level,
        ("model", "dataset", "probe", "P_id", "abs_conditional_order_difference"),
        source="pair_level.parquet",
    )
    expected = set(EXPECTED_MODELS)
    out = model_summary.loc[
        model_summary["probe"].astype(str).eq(PROBE)
        & model_summary["dataset"].astype(str).isin(DATASETS)
        & model_summary["model"].astype(str).isin(expected)
    ].copy()
    out["median_abs_conditional_order_difference"] = pd.to_numeric(
        out["median_abs_conditional_order_difference"], errors="coerce"
    )
    duplicates = out.duplicated(["model", "dataset", "probe"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate model-summary rows in model_summary.parquet:\n"
            + out.loc[duplicates, ["model", "dataset", "probe"]].to_string(index=False)
        )
    if out.empty:
        raise RuntimeError("No instruction-tuned sAwMIL model-summary rows found.")

    ses = []
    n_units = []
    for row_index, row in out.reset_index(drop=True).iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        cell = pair_level.loc[
            pair_level["probe"].astype(str).eq(PROBE)
            & pair_level["model"].astype(str).eq(model)
            & pair_level["dataset"].astype(str).eq(dataset)
        ].copy()
        rng = np.random.default_rng(seed + row_index)
        observed, se, n = bootstrap_median_se(
            cell,
            value_col="abs_conditional_order_difference",
            cluster_col="P_id",
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        saved = float(row["median_abs_conditional_order_difference"])
        if np.isfinite(saved) and np.isfinite(observed) and not np.isclose(saved, observed, rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"Pair-level median does not match model_summary for {model} / {dataset}: "
                f"summary={saved:.12g}, reconstructed={observed:.12g}."
            )
        ses.append(se)
        n_units.append(n)

    out = out.reset_index(drop=True)
    out["bootstrap_se"] = ses
    out["bootstrap_n_units"] = n_units
    return out


def prepare_gamma_summary(
    model_summary: pd.DataFrame,
    statement_level: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    require_columns(
        model_summary,
        ("model", "dataset", "probe", "median_abs_gamma_order_difference"),
        source="model_summary.parquet",
    )
    require_columns(
        statement_level,
        ("model", "dataset", "probe", "abs_gamma_order_difference"),
        source="statement_level.parquet",
    )
    expected = set(EXPECTED_MODELS)
    out = model_summary.loc[
        model_summary["probe"].astype(str).eq(PROBE)
        & model_summary["dataset"].astype(str).isin(DATASETS)
        & model_summary["model"].astype(str).isin(expected)
    ].copy()
    out["median_abs_gamma_order_difference"] = pd.to_numeric(
        out["median_abs_gamma_order_difference"], errors="coerce"
    )
    duplicates = out.duplicated(["model", "dataset", "probe"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate gamma-summary rows in model_summary.parquet:\n"
            + out.loc[duplicates, ["model", "dataset", "probe"]].to_string(index=False)
        )
    if out.empty:
        raise RuntimeError("No instruction-tuned sAwMIL gamma-summary rows found.")

    ses = []
    n_units = []
    for row_index, row in out.reset_index(drop=True).iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        cell = statement_level.loc[
            statement_level["probe"].astype(str).eq(PROBE)
            & statement_level["model"].astype(str).eq(model)
            & statement_level["dataset"].astype(str).eq(dataset)
        ].copy()
        rng = np.random.default_rng(seed + 10000 + row_index)
        observed, se, n = bootstrap_median_se(
            cell,
            value_col="abs_gamma_order_difference",
            cluster_col=None,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        saved = float(row["median_abs_gamma_order_difference"])
        if np.isfinite(saved) and np.isfinite(observed) and not np.isclose(saved, observed, rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"Gamma-level median does not match model_summary for {model} / {dataset}: "
                f"summary={saved:.12g}, reconstructed={observed:.12g}."
            )
        ses.append(se)
        n_units.append(n)

    out = out.reset_index(drop=True)
    out["bootstrap_se"] = ses
    out["bootstrap_n_units"] = n_units
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_bar_panel(
    ax: plt.Axes,
    *,
    panel: pd.DataFrame,
    value_col: str,
    se_col: str,
    models: list[str],
    y_min: float,
    y_max: float,
    show_y_labels: bool,
    zero_line: bool = False,
) -> None:
    indexed = panel.set_index("model").reindex(models)
    values = pd.to_numeric(indexed[value_col], errors="coerce").to_numpy(dtype=float)
    ses = pd.to_numeric(indexed[se_col], errors="coerce").to_numpy(dtype=float)
    x = np.arange(1, len(models) + 1, dtype=float)
    colors = [FAMILY_COLORS[model_family(model)] for model in models]

    ax.bar(x, values, width=0.72, color=colors, alpha=BAR_ALPHA, edgecolor="none", zorder=3)
    valid = np.isfinite(values) & np.isfinite(ses)
    ax.errorbar(
        x[valid], values[valid], yerr=ses[valid], fmt="none",
        ecolor=ERROR_COLOR, elinewidth=0.75, capsize=1.8, capthick=0.75, zorder=5,
    )
    if zero_line:
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.65, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([display_model(model) for model in models], rotation=55, ha="right", rotation_mode="anchor")
    ax.set_xlim(0.35, len(models) + 0.65)
    ax.set_ylim(y_min, y_max)
    style_axis(ax)
    if not show_y_labels:
        ax.tick_params(axis="y", labelleft=False)
        ax.spines["left"].set_visible(False)



def auto_pair_ymax(pair_summary: pd.DataFrame) -> float:
    values = pd.to_numeric(pair_summary["median_abs_conditional_order_difference"], errors="coerce") + pd.to_numeric(pair_summary["bootstrap_se"], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.35
    vmax = float(values.max())
    return max(0.05, float(np.ceil((vmax * 1.08) / 0.05) * 0.05))


def auto_gamma_ymax(gamma_summary: pd.DataFrame) -> float:
    values = pd.to_numeric(gamma_summary["median_abs_gamma_order_difference"], errors="coerce") + pd.to_numeric(gamma_summary["bootstrap_se"], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.35
    vmax = float(values.max())
    return max(0.05, float(np.ceil((vmax * 1.08) / 0.05) * 0.05))



def make_figure(
    *,
    pair_summary: pd.DataFrame,
    gamma_summary: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
    pair_y_max: float | None,
    gamma_y_max: float | None,
) -> Path:
    models = sorted(EXPECTED_MODELS, key=model_sort_key)
    for model in models:
        model_family(model)

    final_pair_y_max = auto_pair_ymax(pair_summary) if pair_y_max is None else float(pair_y_max)
    final_gamma_y_max = auto_gamma_ymax(gamma_summary) if gamma_y_max is None else float(gamma_y_max)
    if final_pair_y_max <= 0:
        raise ValueError("pair_y_max must be positive.")
    if final_gamma_y_max <= 0:
        raise ValueError("gamma_y_max must be positive.")

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(nrows=2, ncols=3, figure=fig, height_ratios=[1.0, 1.0], wspace=0.15, hspace=1.15)
    axes_by_row: list[list[plt.Axes]] = [[], []]
    panel_labels = {
        (0, 0): "(a)",
        (0, 1): "(b)",
        (0, 2): "(c)",
        (1, 0): "(d)",
        (1, 1): "(e)",
        (1, 2): "(f)",
    }

    for col_index, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, col_index])
        axes_by_row[0].append(ax)
        pair_panel = pair_summary.loc[pair_summary["dataset"].astype(str).eq(dataset)].copy()
        plot_bar_panel(
            ax,
            panel=pair_panel,
            value_col="median_abs_conditional_order_difference",
            se_col="bootstrap_se",
            models=models,
            y_min=0.0,
            y_max=final_pair_y_max,
            show_y_labels=(col_index == 0),
        )
        ax.set_title(
            DATASET_NAMES[dataset],
            loc="center",
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
            y=1.15,
            pad=0,
        )
        ax.text(0.0, 1.025, panel_labels[(0, col_index)], transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=TITLE_COLOR)

        ax = fig.add_subplot(gs[1, col_index])
        axes_by_row[1].append(ax)
        rho_panel = gamma_summary.loc[gamma_summary["dataset"].astype(str).eq(dataset)].copy()
        plot_bar_panel(
            ax,
            panel=rho_panel,
            value_col="median_abs_gamma_order_difference",
            se_col="bootstrap_se",
            models=models,
            y_min=0.0,
            y_max=final_gamma_y_max,
            show_y_labels=(col_index == 0),
        )
        ax.text(0.0, 1.025, panel_labels[(1, col_index)], transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=TITLE_COLOR)

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="none", alpha=BAR_ALPHA, label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, -0.28),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.105, right=0.995, top=0.91, bottom=0.0)
    fig.canvas.draw()
    row_positions = [axes_by_row[row][0].get_position() for row in range(2)]
    row_title_x = 0.035
    y_label_x = 0.05

    fig.text(
        row_title_x,
        row_positions[0].y1 + 0.075,
        r"Order sensitivity of pair-level $\widehat{\Pr}(P\mid x)$",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.text(
        row_title_x,
        row_positions[1].y1 + 0.075,
        r"Order sensitivity of graded stability $\gamma(P)$",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    fig.text(
        y_label_x,
        (row_positions[0].y0 + row_positions[0].y1) / 2.0,
        r"Median $|\Delta\,\widehat{\Pr}(P\mid x)|$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )
    fig.text(
        y_label_x,
        (row_positions[1].y0 + row_positions[1].y1) / 2.0,
        r"Median $|\Delta\gamma(P)|$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )

    path = save_figure(fig, output_dir=output_dir, overwrite=overwrite)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SI joint-ordering sensitivity for instruction-tuned models "
            "using the sAwMIL probe only."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/analysis/joint_ordering"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    parser.add_argument(
        "--pair_y_max",
        type=float,
        default=None,
        help="Optional upper bound for the top-row y-axis.",
    )
    parser.add_argument(
        "--gamma_y_max",
        type=float,
        default=None,
        help="Optional upper bound for the bottom-row y-axis.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples used to estimate the SE of each median. Default: 10000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap random seed. Default: 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n_bootstrap must be at least 100.")

    model_summary_path = args.input_dir / "model_summary.parquet"
    pair_level_path = args.input_dir / "pair_level.parquet"
    statement_level_path = args.input_dir / "statement_level.parquet"
    for path in (model_summary_path, pair_level_path, statement_level_path):
        if not path.exists():
            raise FileNotFoundError(path)

    model_summary = pd.read_parquet(model_summary_path)
    pair_level = pd.read_parquet(pair_level_path)
    statement_level = pd.read_parquet(statement_level_path)
    pair_summary = prepare_pair_summary(
        model_summary, pair_level, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    gamma_summary = prepare_gamma_summary(
        model_summary, statement_level, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    print("=" * 100)
    print("FIGURE SI: JOINT ORDERING SENSITIVITY")
    print("=" * 100)
    print(f"Model summary:    {model_summary_path}")
    print(f"Pair level:       {pair_level_path}")
    print(f"Statement level:  {statement_level_path}")
    print(f"Output:           {args.output_dir / (OUTPUT_STEM + '.pdf')}")
    print(f"Probe:            {PROBE}")
    print("Models:           instruction-tuned only")
    print("Top row:          median |Δ Pr(P|x)|")
    print("Bottom row:       median |Δγ(P)|")
    print(f"Error bars:       +/- 1 bootstrap SE ({args.n_bootstrap:,} resamples)")
    print("=" * 100)

    make_figure(
        pair_summary=pair_summary,
        gamma_summary=gamma_summary,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        pair_y_max=args.pair_y_max,
        gamma_y_max=args.gamma_y_max,
    )

    printable = pair_summary[["model", "dataset", "median_abs_conditional_order_difference", "bootstrap_se"]].rename(
        columns={"median_abs_conditional_order_difference": "pair_median", "bootstrap_se": "pair_bootstrap_se"}
    ).merge(
        gamma_summary[["model", "dataset", "median_abs_gamma_order_difference", "bootstrap_se"]].rename(
            columns={"median_abs_gamma_order_difference": "gamma_median", "bootstrap_se": "gamma_bootstrap_se"}
        ),
        on=["model", "dataset"], how="outer",
    )
    print("\nORDER-SENSITIVITY MEDIANS WITH BOOTSTRAP SE")
    print("-" * 100)
    print(printable.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
