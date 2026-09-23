from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style: matched to plot_fig2.py
# ---------------------------------------------------------------------------

DATASETS = ("cities_loc", "med_indications", "defs")
DATASET_NAMES = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}
ESTIMATORS = ("direct",)

DIRECT_COLOR = "#765B73"
GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"
ZERO_LINE_COLOR = "#777777"
MEDIAN_COLOR = "#2f2f2f"
PRIMARY_BAND_COLOR = "#eeeeee"

ESTIMATOR_COLORS = {
    "direct": DIRECT_COLOR,
}

SPLINE_SPECS = (
    ("linear", "Linear"),
    ("df2", "Spline\n$df=2$"),
    ("df3", "Spline\n$df=3$"),
    ("df4", "Spline\n$df=4$"),
    ("df5", "Spline\n$df=5$"),
)

SPLINE_FIGSIZE = (7.2, 2.55)
PERMUTATION_FIGSIZE = (7.2, 2.25)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = [c for c in columns if c not in frame.columns]
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


def style_axis(
    ax: plt.Axes,
    *,
    grid_axis: str = "y",
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(
        axis=grid_axis,
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
    dpi: int,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite.")

        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi

        fig.savefig(path, **kwargs)
        log.info("Wrote %s", path)


def resolve_r2_column(frame: pd.DataFrame) -> str:
    column = first_existing(
        frame,
        (
            "cv_r_squared",
            "spline_cv_r_squared",
            "spline_r_squared",
            "full_fit_r_squared",
        ),
    )
    if column is None:
        raise ValueError("Could not identify held-out spline R^2 column.")
    return column


def resolve_linear_r2_column(frame: pd.DataFrame) -> str:
    column = first_existing(
        frame,
        (
            "linear_cv_r_squared",
            "cv_linear_r_squared",
            "linear_r_squared",
        ),
    )
    if column is None:
        raise ValueError("Could not identify held-out linear R^2 column.")
    return column


def resolve_null_stat_column(frame: pd.DataFrame) -> str:
    column = first_existing(
        frame,
        (
            "median_residual_spearman_rho",
            "median_pairwise_residual_rho",
            "median_cross_model_residual_rho",
            "median_residual_gamma_spearman_rho",
            "null_statistic",
            "statistic",
        ),
    )
    if column is None:
        raise ValueError(
            "Could not identify the per-permutation median residual "
            "correlation statistic."
        )
    return column


def filter_primary_rows(
    frame: pd.DataFrame,
    *,
    probe: str,
    source: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        ("model", "dataset", "probe", "estimator"),
        source=source,
    )

    out = frame.loc[
        frame["probe"].astype(str).eq(probe)
        & frame["dataset"].astype(str).isin(DATASETS)
        & frame["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    if out.empty:
        raise RuntimeError(f"{source}: no rows for probe={probe!r}.")
    return out


def key_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(
        zip(
            frame["model"].astype(str),
            frame["dataset"].astype(str),
            frame["estimator"].astype(str),
        )
    )


def validate_spline_df(
    frame: pd.DataFrame,
    expected: int,
    *,
    source: str,
) -> None:
    if "spline_df" not in frame.columns:
        return

    values = (
        pd.to_numeric(frame["spline_df"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if len(values) and set(values.tolist()) != {expected}:
        raise ValueError(
            f"{source}: expected spline_df={expected}; "
            f"found {sorted(values.tolist())}."
        )


# ---------------------------------------------------------------------------
# Figure 1: spline-specification robustness
# ---------------------------------------------------------------------------

def load_spline_robustness(
    *,
    canonical_dir: Path,
    robustness_dir: Path,
    probe: str,
) -> pd.DataFrame:
    canonical_path = canonical_dir / "model_summary.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)

    canonical = filter_primary_rows(
        pd.read_parquet(canonical_path),
        probe=probe,
        source=str(canonical_path),
    )
    validate_spline_df(canonical, 3, source=str(canonical_path))

    spline_col = resolve_r2_column(canonical)
    linear_col = resolve_linear_r2_column(canonical)
    canonical_keys = key_set(canonical)

    frames: list[pd.DataFrame] = []

    linear = canonical[
        ["model", "dataset", "probe", "estimator", linear_col]
    ].rename(columns={linear_col: "heldout_r2"})
    linear["spec"] = "linear"
    linear["spec_order"] = 0
    frames.append(linear)

    primary = canonical[
        ["model", "dataset", "probe", "estimator", spline_col]
    ].rename(columns={spline_col: "heldout_r2"})
    primary["spec"] = "df3"
    primary["spec_order"] = 2
    frames.append(primary)

    for spline_df, order in ((2, 1), (4, 3), (5, 4)):
        path = robustness_dir / f"df{spline_df}" / "model_summary.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

        frame = filter_primary_rows(
            pd.read_parquet(path),
            probe=probe,
            source=str(path),
        )
        validate_spline_df(frame, spline_df, source=str(path))

        if key_set(frame) != canonical_keys:
            missing = sorted(canonical_keys - key_set(frame))
            extra = sorted(key_set(frame) - canonical_keys)
            raise ValueError(
                f"{path}: model/dataset/estimator units differ from canonical "
                f"analysis. missing={missing[:6]}, extra={extra[:6]}"
            )

        spline_col_alt = resolve_r2_column(frame)

        # Same seed/folds => same linear baseline. Check rather than assume.
        linear_col_alt = resolve_linear_r2_column(frame)
        left = canonical[
            ["model", "dataset", "estimator", linear_col]
        ].rename(columns={linear_col: "canonical_linear"})
        right = frame[
            ["model", "dataset", "estimator", linear_col_alt]
        ].rename(columns={linear_col_alt: "alt_linear"})
        check = left.merge(
            right,
            on=["model", "dataset", "estimator"],
            validate="one_to_one",
        )
        a = pd.to_numeric(check["canonical_linear"], errors="coerce")
        b = pd.to_numeric(check["alt_linear"], errors="coerce")
        diff = np.abs(a - b)
        diff = diff[np.isfinite(diff)]
        if len(diff) and float(diff.max()) > 1e-10:
            raise ValueError(
                f"{path}: linear baseline differs from canonical run; "
                f"max |delta|={float(diff.max()):.3e}."
            )

        out = frame[
            ["model", "dataset", "probe", "estimator", spline_col_alt]
        ].rename(columns={spline_col_alt: "heldout_r2"})
        out["spec"] = f"df{spline_df}"
        out["spec_order"] = order
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    combined["heldout_r2"] = pd.to_numeric(
        combined["heldout_r2"],
        errors="coerce",
    )
    return combined


def r2_limits(frame: pd.DataFrame) -> tuple[float, float]:
    values = pd.to_numeric(frame["heldout_r2"], errors="coerce")
    values = values[np.isfinite(values)].to_numpy(float)

    if len(values) == 0:
        raise RuntimeError("No finite held-out R^2 values.")

    minimum = float(values.min())
    maximum = float(values.max())

    lower = 0.0
    if minimum < 0:
        lower = math.floor((minimum - 0.025) * 10.0) / 10.0

    upper = max(0.8, math.ceil((maximum + 0.025) * 10.0) / 10.0)
    upper = min(1.0, upper)

    if maximum > upper:
        raise ValueError(
            f"Automatic R^2 upper limit {upper:.2f} clips "
            f"maximum {maximum:.3f}."
        )

    return lower, upper


def draw_violin(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    position: float,
    color: str,
) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    if len(values) >= 2 and np.nanstd(values) > 0:
        violin = ax.violinplot(
            [values],
            positions=[position],
            widths=0.23,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method="scott",
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.32)

    offsets = (
        np.linspace(-0.045, 0.045, len(values))
        if len(values) > 1
        else np.array([0.0])
    )
    ax.scatter(
        position + offsets,
        values,
        s=4.2,
        alpha=0.34,
        color=color,
        edgecolors="none",
        zorder=3,
    )

    median = float(np.median(values))
    ax.plot(
        [position - 0.075, position + 0.075],
        [median, median],
        color=MEDIAN_COLOR,
        linewidth=1.0,
        zorder=5,
    )


def make_spline_figure(
    robustness: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
    dpi: int,
    overwrite: bool,
) -> None:
    low, high = r2_limits(robustness)
    log.info("Spline robustness R^2 limits: [%.2f, %.2f]", low, high)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=SPLINE_FIGSIZE,
        sharey=True,
    )

    x = np.arange(1, len(SPLINE_SPECS) + 1, dtype=float)
    offsets = {"direct": 0.0}
    labels = ("(a)", "(b)", "(c)")

    for col, dataset in enumerate(DATASETS):
        ax = axes[col]
        panel = robustness.loc[robustness["dataset"].eq(dataset)]

        # Primary df=3 specification.
        ax.axvspan(
            x[2] - 0.43,
            x[2] + 0.43,
            color=PRIMARY_BAND_COLOR,
            alpha=0.85,
            zorder=-3,
        )

        for i, (spec, _) in enumerate(SPLINE_SPECS):
            for estimator in ESTIMATORS:
                values = pd.to_numeric(
                    panel.loc[
                        panel["spec"].eq(spec)
                        & panel["estimator"].eq(estimator),
                        "heldout_r2",
                    ],
                    errors="coerce",
                ).dropna().to_numpy(float)

                draw_violin(
                    ax,
                    values,
                    position=x[i] + offsets[estimator],
                    color=ESTIMATOR_COLORS[estimator],
                )

        if low < 0:
            ax.axhline(
                0.0,
                color=ZERO_LINE_COLOR,
                linewidth=0.65,
                linestyle="--",
                zorder=0,
            )

        ax.set_xlim(0.45, len(SPLINE_SPECS) + 0.55)
        ax.set_ylim(low, high)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in SPLINE_SPECS])
        style_axis(ax)

        ax.set_title(
            DATASET_NAMES[dataset],
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
            y=1.08,
            pad=0,
        )
        ax.text(
            0.0,
            1.02,
            labels[col],
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
            ha="left",
            va="bottom",
        )

        if col != 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.84,
        bottom=0.26,
        wspace=0.16,
    )
    fig.text(
        0.035,
        0.54,
        r"Held-out $R^2$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
    )

    legend_handles = [
        Patch(facecolor=DIRECT_COLOR, edgecolor="none", label="Direct"),
        Line2D([0], [0], color=MEDIAN_COLOR, linewidth=1.0, label="Median"),
        Patch(
            facecolor=PRIMARY_BAND_COLOR,
            edgecolor="none",
            label=r"Primary $df=3$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.035),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.0,
    )

    save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        dpi=dpi,
        overwrite=overwrite,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: residual-agreement permutation nulls
# ---------------------------------------------------------------------------

def load_permutation_data(
    *,
    canonical_dir: Path,
    probe: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = canonical_dir / "permutation_summary.parquet"
    null_path = canonical_dir / "permutation_null.parquet"

    for path in (summary_path, null_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = pd.read_parquet(summary_path)
    null = pd.read_parquet(null_path)

    require_columns(
        summary,
        (
            "dataset",
            "probe",
            "estimator",
            "observed_median_residual_spearman_rho",
            "p_greater",
            "n_permutations_valid",
        ),
        source=str(summary_path),
    )
    require_columns(
        null,
        ("dataset", "probe", "estimator"),
        source=str(null_path),
    )

    summary = summary.loc[
        summary["probe"].astype(str).eq(probe)
        & summary["dataset"].astype(str).isin(DATASETS)
        & summary["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    null = null.loc[
        null["probe"].astype(str).eq(probe)
        & null["dataset"].astype(str).isin(DATASETS)
        & null["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    if len(summary) != 3:
        raise ValueError(
            f"Expected three Direct permutation-summary rows for {probe}; "
            f"found {len(summary)}."
        )

    null_col = resolve_null_stat_column(null)
    null["_null_stat"] = pd.to_numeric(null[null_col], errors="coerce")
    null = null.dropna(subset=["_null_stat"])

    counts = (
        null.groupby(["dataset", "estimator"])
        .size()
        .rename("raw_n")
        .reset_index()
    )
    check = summary.merge(
        counts,
        on=["dataset", "estimator"],
        how="left",
        validate="one_to_one",
    )
    expected = pd.to_numeric(check["n_permutations_valid"], errors="coerce")
    observed = pd.to_numeric(check["raw_n"], errors="coerce")

    bad = check.loc[
        expected.ne(observed).fillna(True),
        ["dataset", "estimator", "n_permutations_valid", "raw_n"],
    ]
    if not bad.empty:
        raise ValueError(
            "Raw permutation counts disagree with saved summary:\n"
            + bad.to_string(index=False)
        )

    return summary, null


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return r"$p=\mathrm{NA}$"
    if value < 0.001:
        return r"$p<.001$"
    if value < 0.01:
        return rf"$p={value:.3f}$".replace("0.", ".")
    return rf"$p={value:.2f}$".replace("0.", ".")


def rho_bound(
    summary: pd.DataFrame,
    null: pd.DataFrame,
) -> float:
    observed = pd.to_numeric(
        summary["observed_median_residual_spearman_rho"],
        errors="coerce",
    ).to_numpy(float)
    null_values = pd.to_numeric(
        null["_null_stat"],
        errors="coerce",
    ).to_numpy(float)

    values = np.concatenate(
        [
            observed[np.isfinite(observed)],
            null_values[np.isfinite(null_values)],
        ]
    )
    if len(values) == 0:
        return 0.4

    maximum = float(np.max(np.abs(values)))
    bound = max(0.3, math.ceil((maximum + 0.025) * 20.0) / 20.0)
    return min(1.0, bound)


def make_permutation_figure(
    summary: pd.DataFrame,
    null: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
    dpi: int,
    overwrite: bool,
    bins: int,
) -> None:
    bound = rho_bound(summary, null)
    log.info("Permutation-null rho limits: [%.2f, %.2f]", -bound, bound)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=PERMUTATION_FIGSIZE,
        sharex=True,
        squeeze=False,
    )

    panel_labels = ("(a)", "(b)", "(c)")
    bin_edges = np.linspace(-bound, bound, bins + 1)
    estimator = "direct"

    for col, dataset in enumerate(DATASETS):
        ax = axes[0, col]

        values = pd.to_numeric(
            null.loc[
                null["dataset"].eq(dataset)
                & null["estimator"].eq(estimator),
                "_null_stat",
            ],
            errors="coerce",
        ).dropna().to_numpy(float)

        match = summary.loc[
            summary["dataset"].eq(dataset)
            & summary["estimator"].eq(estimator)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one summary row for {dataset}/{estimator}; "
                f"found {len(match)}."
            )

        record = match.iloc[0]
        observed = float(
            record["observed_median_residual_spearman_rho"]
        )
        p_value = float(record["p_greater"])
        n_perm = int(record["n_permutations_valid"])

        ax.hist(
            values,
            bins=bin_edges,
            density=True,
            color=DIRECT_COLOR,
            alpha=0.72,
            edgecolor="none",
            zorder=2,
        )
        ax.axvline(
            0.0,
            color=ZERO_LINE_COLOR,
            linewidth=0.70,
            linestyle="--",
            zorder=3,
        )
        ax.axvline(
            observed,
            color=MEDIAN_COLOR,
            linewidth=1.25,
            zorder=5,
        )

        ax.set_xlim(-bound, bound)
        style_axis(ax)
        ax.tick_params(axis="y", labelleft=False, length=0)
        ax.spines["left"].set_visible(False)

        ax.text(
            0.0,
            1.025,
            panel_labels[col],
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
            ha="left",
            va="bottom",
        )

        ax.set_title(
            DATASET_NAMES[dataset],
            fontsize=7.2,
            fontweight="bold",
            color=TITLE_COLOR,
            y=1.105,
            pad=0,
        )

        annotation = (
            rf"$\rho_{{obs}}={observed:.2f}$"
            + "\n"
            + format_p(p_value)
            + "\n"
            + rf"$N_{{perm}}={n_perm:,}$"
        )
        ax.text(
            0.97,
            0.93,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.0,
            color=TITLE_COLOR,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "pad": 1.4,
            },
            zorder=6,
        )

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.82,
        bottom=0.29,
        wspace=0.15,
    )

    fig.text(
        0.53,
        0.14,
        r"Median pairwise residual Spearman $\rho$",
        ha="center",
        va="center",
        fontsize=7,
    )

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MEDIAN_COLOR,
                linewidth=1.25,
                label="Observed median",
            ),
            Line2D(
                [0],
                [0],
                color=ZERO_LINE_COLOR,
                linewidth=0.70,
                linestyle="--",
                label="Null center (0)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.02),
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        dpi=dpi,
        overwrite=overwrite,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SI robustness figures for the "
            "stability--belief-probability analysis."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/analysis/stability_vs_credence"),
    )
    parser.add_argument(
        "--robustness_dir",
        type=Path,
        default=Path(
            "outputs/analysis/stability_vs_credence_robustness"
        ),
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
        "--spline_stem",
        default="figure_si_spline_robustness",
    )
    parser.add_argument(
        "--permutation_stem",
        default="figure_si_residual_permutation_nulls",
    )
    parser.add_argument("--bins", type=int, default=28)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.bins < 5:
        parser.error("--bins must be at least 5.")
    return args


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("SI: STABILITY--BELIEF-PROBABILITY ROBUSTNESS")
    print("=" * 100)
    print(f"Canonical input:  {args.input_dir}")
    print(f"Robustness input: {args.robustness_dir}")
    print(f"Output:           {args.output_dir}")
    print(f"Probe:            {args.probe}")
    print()

    robustness = load_spline_robustness(
        canonical_dir=args.input_dir,
        robustness_dir=args.robustness_dir,
        probe=args.probe,
    )
    summary, null = load_permutation_data(
        canonical_dir=args.input_dir,
        probe=args.probe,
    )

    spline_summary = (
        robustness.groupby(
            ["dataset", "estimator", "spec", "spec_order"],
            sort=True,
        )["heldout_r2"]
        .agg(["count", "median"])
        .reset_index()
        .sort_values(
            ["dataset", "estimator", "spec_order"],
            kind="stable",
        )
    )
    print("Spline robustness (count and median held-out R^2):")
    print(
        spline_summary[
            ["dataset", "estimator", "spec", "count", "median"]
        ]
        .round({"median": 4})
        .to_string(index=False)
    )
    print()

    print("Permutation summaries:")
    print(
        summary[
            [
                "dataset",
                "estimator",
                "observed_median_residual_spearman_rho",
                "p_greater",
                "n_permutations_valid",
            ]
        ]
        .sort_values(["dataset", "estimator"], kind="stable")
        .round(
            {
                "observed_median_residual_spearman_rho": 4,
                "p_greater": 4,
            }
        )
        .to_string(index=False)
    )
    print("=" * 100)

    make_spline_figure(
        robustness,
        output_dir=args.output_dir,
        stem=args.spline_stem,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    make_permutation_figure(
        summary,
        null,
        output_dir=args.output_dir,
        stem=args.permutation_stem,
        dpi=args.dpi,
        overwrite=args.overwrite,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
