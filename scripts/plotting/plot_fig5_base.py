"""
Generates the base-model SI behavioral-resilience figure using Direct Conditional
graded stability and the probability-matched conversational challenge pairs.

Example:
    python -m scripts.plotting.plot_fig5_base --probe sawmil --overwrite
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Figure constants
# ---------------------------------------------------------------------------

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

# Repository estimator name for Direct Conditional in the behavioral outputs.
ESTIMATOR = "conditional"

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

GRID_COLOR = "#e3e3e3"
TITLE_COLOR = "#3f3f3f"
ZERO_LINE_COLOR = "#777777"
ERROR_COLOR = "#2f2f2f"

FIGSIZE = (7.2, 2.85)
BAR_WIDTH = 0.72
OUTPUT_STEM = "figure_si_5_base"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


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
    """Display base model names with an explicit '(b)' suffix."""
    return f"{str(model)} (b)"


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


def model_sort_key(model: str) -> tuple[object, ...]:
    return (
        FAMILY_ORDER.index(model_family(model)),
        parameter_count_b(model),
        display_model(model).lower(),
    )


def ordered_models() -> list[str]:
    return sorted(EXPECTED_MODELS, key=model_sort_key)


def style_axis(ax: plt.Axes) -> None:
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
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    log.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_summary(
    frame: pd.DataFrame,
    *,
    probe: str,
    sample: str,
    estimator: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        (
            "model_name",
            "dataset",
            "probe",
            "sample",
            "estimator",
            "mean_movement_difference",
        ),
        source="matched_unit_summary.parquet",
    )

    expected = set(EXPECTED_MODELS)
    out = frame.loc[
        frame["probe"].astype(str).eq(probe)
        & frame["sample"].astype(str).eq(sample)
        & frame["dataset"].astype(str).isin(DATASETS)
        & frame["estimator"].astype(str).eq(estimator)
        & frame["model_name"].astype(str).isin(expected)
    ].copy()

    out["mean_movement_difference"] = pd.to_numeric(
        out["mean_movement_difference"],
        errors="coerce",
    )
    out = out.loc[
        np.isfinite(out["mean_movement_difference"].to_numpy(dtype=float))
    ].copy()

    duplicates = out.duplicated(
        ["model_name", "dataset", "probe", "sample", "estimator"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            "Duplicate model-level rows in matched_unit_summary.parquet:\n"
            + out.loc[
                duplicates,
                ["model_name", "dataset", "probe", "sample", "estimator"],
            ].to_string(index=False)
        )

    if out.empty:
        raise RuntimeError(
            f"No base-model Direct behavioral-resilience rows remain for probe={probe!r}, sample={sample!r}."
        )

    return out


def add_challenge_sequence_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one verified challenge_sequence column per matched pair."""
    out = frame.copy()
    high_col = "high_challenge_sequence"
    low_col = "low_challenge_sequence"

    if high_col in out.columns and low_col in out.columns:
        high = out[high_col]
        low = out[low_col]
        comparable = high.notna() & low.notna()
        mismatch = comparable & ~high.astype(str).eq(low.astype(str))
        if mismatch.any():
            raise ValueError("Matched pairs with different high/low challenge sequences were found.")
        if high.isna().any() or low.isna().any():
            raise ValueError("Missing high/low challenge-sequence values were found.")
        out["challenge_sequence"] = high.astype(str)
    elif "challenge_sequence" in out.columns:
        if out["challenge_sequence"].isna().any():
            raise ValueError("Missing challenge_sequence values were found.")
        out["challenge_sequence"] = out["challenge_sequence"].astype(str)
    else:
        raise ValueError(
            "matched_results_all.parquet needs either `challenge_sequence` or both "
            "`high_challenge_sequence` and `low_challenge_sequence`."
        )
    return out


def prepare_pair_results(
    frame: pd.DataFrame,
    *,
    probe: str,
    sample: str,
    estimator: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        ("model_name", "dataset", "probe", "sample", "estimator", "match_id", "movement_difference"),
        source="matched_results_all.parquet",
    )
    expected = set(EXPECTED_MODELS)
    out = frame.loc[
        frame["probe"].astype(str).eq(probe)
        & frame["sample"].astype(str).eq(sample)
        & frame["dataset"].astype(str).isin(DATASETS)
        & frame["estimator"].astype(str).eq(estimator)
        & frame["model_name"].astype(str).isin(expected)
    ].copy()
    if out.empty:
        raise RuntimeError(
            f"No base-model Direct pair-level behavioral rows remain for probe={probe!r}, sample={sample!r}."
        )
    out["movement_difference"] = pd.to_numeric(out["movement_difference"], errors="coerce")
    out = out.loc[np.isfinite(out["movement_difference"].to_numpy(dtype=float))].copy()
    out = add_challenge_sequence_column(out)
    duplicate_keys = ["model_name", "dataset", "probe", "sample", "estimator", "match_id"]
    duplicates = out.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate matched-pair rows in matched_results_all.parquet:\n"
            + out.loc[duplicates, duplicate_keys].head(20).to_string(index=False)
        )
    return out


def stratified_bootstrap_mean_se(
    frame: pd.DataFrame,
    *,
    value_col: str,
    stratum_col: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Return observed mean, bootstrap SE, and number of observed strata."""
    work = frame[[value_col, stratum_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[
        np.isfinite(work[value_col].to_numpy(dtype=float)) & work[stratum_col].notna()
    ].copy()
    if work.empty:
        return np.nan, np.nan, 0

    observed_mean = float(work[value_col].mean())
    strata = [g[value_col].to_numpy(dtype=float) for _, g in work.groupby(stratum_col, sort=True)]
    n_total = int(sum(len(v) for v in strata))
    if n_total <= 1:
        return observed_mean, 0.0, len(strata)

    boot = np.empty(n_bootstrap, dtype=float)
    chunk_size = 1000
    filled = 0
    while filled < n_bootstrap:
        current = min(chunk_size, n_bootstrap - filled)
        sums = np.zeros(current, dtype=float)
        for values in strata:
            idx = rng.integers(0, len(values), size=(current, len(values)))
            sums += values[idx].sum(axis=1)
        boot[filled:filled+current] = sums / float(n_total)
        filled += current

    return observed_mean, float(np.std(boot, ddof=1)), len(strata)


def build_plot_table(
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Complete the model x dataset grid and compute +/- 1 bootstrap SE."""
    rows: list[dict[str, object]] = []
    indexed = summary.set_index(["model_name", "dataset"])

    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(ordered_models()):
            key = (model, dataset)
            if key not in indexed.index:
                log.warning("Missing model-level summary row for %s / %s", model, dataset)
                rows.append({
                    "model_name": model, "dataset": dataset, "mean": np.nan,
                    "bootstrap_se": np.nan, "n_pairs": 0, "n_sequences": 0,
                })
                continue

            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Non-unique summary key: {key}")
            saved_mean = float(row["mean_movement_difference"])

            cell = pairs.loc[
                pairs["model_name"].astype(str).eq(model)
                & pairs["dataset"].astype(str).eq(dataset)
            ].copy()
            if cell.empty:
                raise RuntimeError(
                    f"No pair-level rows for model-level Direct effect {model} / {dataset}."
                )

            rng = np.random.default_rng(seed + dataset_index * 1000 + model_index)
            pair_mean, bootstrap_se, n_sequences = stratified_bootstrap_mean_se(
                cell,
                value_col="movement_difference",
                stratum_col="challenge_sequence",
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            if not np.isclose(saved_mean, pair_mean, rtol=1e-8, atol=1e-10):
                raise ValueError(
                    f"Pair-level mean does not match matched_unit_summary for {model} / {dataset}: "
                    f"summary={saved_mean:.12g}, pairs={pair_mean:.12g}."
                )
            rows.append({
                "model_name": model,
                "dataset": dataset,
                "mean": saved_mean,
                "bootstrap_se": bootstrap_se,
                "n_pairs": int(len(cell)),
                "n_sequences": int(n_sequences),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Axis-limit helpers
# ---------------------------------------------------------------------------


def nice_step(span: float) -> float:
    if span <= 0.10:
        return 0.02
    if span <= 0.25:
        return 0.05
    if span <= 0.50:
        return 0.10
    return 0.20


def automatic_y_limits(plot_table: pd.DataFrame) -> tuple[float, float, float]:
    means = pd.to_numeric(plot_table["mean"], errors="coerce").to_numpy(dtype=float)
    ses = pd.to_numeric(plot_table["bootstrap_se"], errors="coerce").to_numpy(dtype=float)
    values = np.concatenate([means - ses, means + ses, means, np.array([0.0])])
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -0.05, 0.20, 0.05

    raw_min = min(0.0, float(np.min(values)))
    raw_max = max(0.0, float(np.max(values)))
    raw_span = max(raw_max - raw_min, 0.05)
    pad = 0.10 * raw_span

    target_min = min(0.0, raw_min - pad)
    target_max = max(0.0, raw_max + pad)
    step = nice_step(target_max - target_min)

    y_min = math.floor(target_min / step) * step
    y_max = math.ceil(target_max / step) * step

    # Keep a little space on both sides of zero so the dashed reference line is
    # visually distinct from the frame even if all model effects share one sign.
    if y_min >= 0.0:
        y_min = -step
    if y_max <= 0.0:
        y_max = step

    return float(y_min), float(y_max), float(step)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(
    *,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
    n_bootstrap: int,
    seed: int,
    y_min: float | None,
    y_max: float | None,
) -> Path:
    plot_table = build_plot_table(summary, pairs, n_bootstrap=n_bootstrap, seed=seed)

    auto_min, auto_max, tick_step = automatic_y_limits(plot_table)
    common_y_min = auto_min if y_min is None else float(y_min)
    common_y_max = auto_max if y_max is None else float(y_max)
    if common_y_min >= common_y_max:
        raise ValueError("y_min must be smaller than y_max.")

    means_for_limits = pd.to_numeric(plot_table["mean"], errors="coerce").to_numpy(dtype=float)
    ses_for_limits = pd.to_numeric(plot_table["bootstrap_se"], errors="coerce").to_numpy(dtype=float)
    lows = means_for_limits - ses_for_limits
    highs = means_for_limits + ses_for_limits
    lows = lows[np.isfinite(lows)]
    highs = highs[np.isfinite(highs)]
    if len(lows) and float(np.min(lows)) < common_y_min:
        raise ValueError(
            f"y_min={common_y_min:g} clips a +/- 1 bootstrap SE error bar; need <= {np.min(lows):.4f}."
        )
    if len(highs) and float(np.max(highs)) > common_y_max:
        raise ValueError(
            f"y_max={common_y_max:g} clips a +/- 1 bootstrap SE error bar; need >= {np.max(highs):.4f}."
        )

    models = ordered_models()
    x = np.arange(1, len(models) + 1, dtype=float)
    colors = [FAMILY_COLORS[model_family(model)] for model in models]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGSIZE,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.78,
        bottom=0.31,
        wspace=0.15,
    )

    panel_labels = ("(a)", "(b)", "(c)")

    for col_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        panel = (
            plot_table.loc[plot_table["dataset"].astype(str).eq(dataset)]
            .set_index("model_name")
            .reindex(models)
        )
        means = panel["mean"].to_numpy(dtype=float)
        ses = panel["bootstrap_se"].to_numpy(dtype=float)

        ax.bar(
            x,
            means,
            width=BAR_WIDTH,
            color=colors,
            edgecolor="none",
            zorder=3,
        )

        valid = np.isfinite(means) & np.isfinite(ses)
        ax.errorbar(
            x[valid], means[valid], yerr=ses[valid], fmt="none",
            ecolor=ERROR_COLOR, elinewidth=0.75, capsize=1.8, capthick=0.75, zorder=5,
        )

        ax.axhline(
            0.0,
            color=ZERO_LINE_COLOR,
            linewidth=0.70,
            linestyle="--",
            zorder=1,
        )

        ax.set_xlim(0.35, len(models) + 0.65)
        ax.set_ylim(common_y_min, common_y_max)

        # If the user manually chooses either limit, derive a clean tick step
        # from the displayed range rather than preserving the automatic one.
        if y_min is not None or y_max is not None:
            tick_step = nice_step(common_y_max - common_y_min)
        first_tick = math.ceil(common_y_min / tick_step - 1e-10) * tick_step
        yticks = np.arange(
            first_tick,
            common_y_max + tick_step * 0.5,
            tick_step,
        )
        ax.set_yticks(yticks)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_model(model) for model in models],
            rotation=55,
            ha="right",
            rotation_mode="anchor",
        )

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

    fig.canvas.draw()
    first_pos = axes[0].get_position()

    fig.text(
        0.04,
        (first_pos.y0 + first_pos.y1) / 2.0,
        r"Average $\Delta M$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
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
        bbox_to_anchor=(0.50, 0.0),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.0,
    )

    path = save_figure(
        fig,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    plt.close(fig)

    printable = plot_table.copy()
    printable["model"] = printable["model_name"].map(display_model)
    printable["dataset_name"] = printable["dataset"].map(DATASET_NAMES)
    printable = printable[["dataset_name", "model", "mean", "bootstrap_se", "n_pairs", "n_sequences"]]

    print("\nMODEL-LEVEL BEHAVIORAL RESILIENCE EFFECTS WITH BOOTSTRAP SE")
    print("-" * 100)
    print(
        printable.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nDisplayed y-axis: [{common_y_min:.4f}, {common_y_max:.4f}]")

    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure SI 5: base Direct behavioral resilience effects."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/analysis/behavioral_resilience"),
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
        "--sample",
        default="round0_agreement",
        help="Behavioral sample to plot. Main-text default: round0_agreement.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of sequence-stratified bootstrap resamples per model x dataset. Default: 10000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap random seed. Default: 0.",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=None,
        help="Optional common lower y-axis bound.",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=None,
        help="Optional common upper y-axis bound.",
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

    summary_path = args.input_dir / "matched_unit_summary.parquet"
    pairs_path = args.input_dir / "matched_results_all.parquet"
    for path in (summary_path, pairs_path):
        if not path.exists():
            raise FileNotFoundError(path)

    raw_summary = pd.read_parquet(summary_path)
    raw_pairs = pd.read_parquet(pairs_path)
    estimator = ESTIMATOR
    summary = prepare_summary(
        raw_summary, probe=args.probe, sample=args.sample, estimator=estimator,
    )
    pairs = prepare_pair_results(
        raw_pairs, probe=args.probe, sample=args.sample, estimator=estimator,
    )

    print("=" * 100)
    print("FIGURE SI 5: BEHAVIORAL RESILIENCE (BASE)")
    print("=" * 100)
    print(f"Summary:    {summary_path}")
    print(f"Pairs:      {pairs_path}")
    print(f"Output:     {args.output_dir / (OUTPUT_STEM + '.pdf')}")
    print(f"Probe:      {args.probe}")
    print(f"Sample:     {args.sample}")
    print("Estimator: Direct Conditional only")
    print("Models:     base only")
    print(f"Error bars: +/- 1 bootstrap SE ({args.n_bootstrap:,} sequence-stratified resamples)")
    print("=" * 100)

    make_figure(
        summary=summary,
        pairs=pairs,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
