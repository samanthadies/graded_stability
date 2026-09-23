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
ERROR_COLOR = "#2f2f2f"

FIGSIZE = (7.2, 5.0)
BAR_WIDTH = 0.72
OUTPUT_STEM = "figure_si_5_svm_mass_mean"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}; available={frame.columns.tolist()}")


def display_model(model: str) -> str:
    return str(model).lstrip("_")


def model_family(model: str) -> str:
    bare = str(model).lstrip("_").lower()
    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family
    raise ValueError(f"Unrecognized model family for {model!r}; expected one of {list(FAMILY_ORDER)}.")


def parameter_count_b(model: str) -> float:
    bare = str(model).lstrip("_").lower()
    sizes = [float(value) for value in re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])", bare)]
    if not sizes:
        raise ValueError(f"Could not parse parameter count from {model!r}.")
    return max(sizes)


def model_sort_key(model: str) -> tuple[object, ...]:
    return (FAMILY_ORDER.index(model_family(model)), parameter_count_b(model), str(model).lstrip("_").lower())


def ordered_models() -> list[str]:
    return sorted(EXPECTED_MODELS, key=model_sort_key)


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


def prepare_summary(frame: pd.DataFrame, *, probe: str, sample: str) -> pd.DataFrame:
    require_columns(
        frame,
        ("model_name", "dataset", "probe", "sample", "estimator", "mean_movement_difference"),
        source="matched_unit_summary.parquet",
    )
    expected = set(EXPECTED_MODELS)
    out = frame.loc[
        frame["probe"].astype(str).eq(probe)
        & frame["sample"].astype(str).eq(sample)
        & frame["dataset"].astype(str).isin(DATASETS)
        & frame["estimator"].astype(str).eq(ESTIMATOR)
        & frame["model_name"].astype(str).isin(expected)
    ].copy()
    out["mean_movement_difference"] = pd.to_numeric(out["mean_movement_difference"], errors="coerce")
    out = out.loc[np.isfinite(out["mean_movement_difference"].to_numpy(dtype=float))].copy()
    duplicates = out.duplicated(["model_name", "dataset", "probe", "sample", "estimator"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate model-level rows in matched_unit_summary.parquet:\n"
            + out.loc[duplicates, ["model_name", "dataset", "probe", "sample", "estimator"]].to_string(index=False)
        )
    if out.empty:
        raise RuntimeError(f"No instruction-tuned Direct behavioral-resilience rows remain for probe={probe!r}, sample={sample!r}.")
    return out


def add_challenge_sequence_column(frame: pd.DataFrame) -> pd.DataFrame:
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


def prepare_pair_results(frame: pd.DataFrame, *, probe: str, sample: str) -> pd.DataFrame:
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
        & frame["estimator"].astype(str).eq(ESTIMATOR)
        & frame["model_name"].astype(str).isin(expected)
    ].copy()
    if out.empty:
        raise RuntimeError(
            f"No instruction-tuned Direct pair-level behavioral rows remain for probe={probe!r}, sample={sample!r}."
        )
    out["movement_difference"] = pd.to_numeric(out["movement_difference"], errors="coerce")
    out = out.loc[np.isfinite(out["movement_difference"].to_numpy(dtype=float))].copy()
    out = add_challenge_sequence_column(out)
    keys = ["model_name", "dataset", "probe", "sample", "estimator", "match_id"]
    duplicates = out.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate matched-pair rows in matched_results_all.parquet:\n"
            + out.loc[duplicates, keys].head(20).to_string(index=False)
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
        boot[filled:filled + current] = sums / float(n_total)
        filled += current
    return observed_mean, float(np.std(boot, ddof=1)), len(strata)


def build_plot_table(
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    probe_index: int,
) -> pd.DataFrame:
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
                raise RuntimeError(f"No pair-level rows for {model} / {dataset}.")
            rng = np.random.default_rng(seed + probe_index * 100000 + dataset_index * 1000 + model_index)
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
                "model_name": model, "dataset": dataset, "mean": saved_mean,
                "bootstrap_se": bootstrap_se, "n_pairs": int(len(cell)),
                "n_sequences": int(n_sequences),
            })
    return pd.DataFrame(rows)


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
    if y_min >= 0.0:
        y_min = -step
    if y_max <= 0.0:
        y_max = step
    return float(y_min), float(y_max), float(step)


def make_figure(
    *,
    summary_by_probe: dict[str, pd.DataFrame],
    pairs_by_probe: dict[str, pd.DataFrame],
    output_dir: Path,
    overwrite: bool,
    n_bootstrap: int,
    seed: int,
    y_min: float | None,
    y_max: float | None,
) -> Path:
    plot_by_probe = {
        probe: build_plot_table(
            summary_by_probe[probe], pairs_by_probe[probe],
            n_bootstrap=n_bootstrap, seed=seed, probe_index=probe_idx,
        )
        for probe_idx, (probe, _) in enumerate(PROBE_SPECS)
    }
    combined = pd.concat(list(plot_by_probe.values()), ignore_index=True)
    auto_min, auto_max, tick_step = automatic_y_limits(combined)
    common_y_min = auto_min if y_min is None else float(y_min)
    common_y_max = auto_max if y_max is None else float(y_max)
    if common_y_min >= common_y_max:
        raise ValueError("y_min must be smaller than y_max.")

    means = pd.to_numeric(combined["mean"], errors="coerce").to_numpy(dtype=float)
    ses = pd.to_numeric(combined["bootstrap_se"], errors="coerce").to_numpy(dtype=float)
    lows = (means - ses)[np.isfinite(means - ses)]
    highs = (means + ses)[np.isfinite(means + ses)]
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

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharey=True)
    axes = np.asarray(axes)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.87, bottom=0.24, wspace=0.15, hspace=0.90)

    panel_labels = [f"({chr(ord('a') + i)})" for i in range(6)]
    label_idx = 0
    for probe_idx, (probe, probe_label) in enumerate(PROBE_SPECS):
        plot_table = plot_by_probe[probe]
        for col_index, dataset in enumerate(DATASETS):
            ax = axes[probe_idx, col_index]
            panel = (
                plot_table.loc[plot_table["dataset"].astype(str).eq(dataset)]
                .set_index("model_name").reindex(models)
            )
            means = panel["mean"].to_numpy(dtype=float)
            ses = panel["bootstrap_se"].to_numpy(dtype=float)
            ax.bar(x, means, width=BAR_WIDTH, color=colors, edgecolor="none", zorder=3)
            valid = np.isfinite(means) & np.isfinite(ses)
            ax.errorbar(
                x[valid], means[valid], yerr=ses[valid], fmt="none",
                ecolor=ERROR_COLOR, elinewidth=0.75, capsize=1.8,
                capthick=0.75, zorder=5,
            )
            ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.70, linestyle="--", zorder=1)
            ax.set_xlim(0.35, len(models) + 0.65)
            ax.set_ylim(common_y_min, common_y_max)
            row_tick_step = nice_step(common_y_max - common_y_min) if (y_min is not None or y_max is not None) else tick_step
            first_tick = math.ceil(common_y_min / row_tick_step - 1e-10) * row_tick_step
            ax.set_yticks(np.arange(first_tick, common_y_max + row_tick_step * 0.5, row_tick_step))
            ax.set_xticks(x)
            ax.set_xticklabels(
                [display_model(model) for model in models], rotation=55,
                ha="right", rotation_mode="anchor",
            )
            style_axis(ax)
            if probe_idx == 0:
                ax.set_title(
                    DATASET_NAMES[dataset], loc="center", fontsize=7.2,
                    fontweight="bold", color=TITLE_COLOR, y=1.105, pad=0,
                )
            ax.text(
                0.0, 1.025, panel_labels[label_idx], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=TITLE_COLOR,
            )
            label_idx += 1
            if col_index != 0:
                ax.tick_params(axis="y", labelleft=False)
                ax.spines["left"].set_visible(False)

    fig.canvas.draw()
    first_row_pos = axes[0, 0].get_position()
    second_row_pos = axes[1, 0].get_position()
    fig.text(0.035, first_row_pos.y1 + 0.04, "SVM", ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=TITLE_COLOR)
    fig.text(0.035, second_row_pos.y1 + 0.04, "Mass Mean", ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=TITLE_COLOR)
    fig.text(0.04, (first_row_pos.y0 + first_row_pos.y1) / 2.0, r"Average $\Delta M$", ha="center", va="center", rotation=90, fontsize=7)
    fig.text(0.04, (second_row_pos.y0 + second_row_pos.y1) / 2.0, r"Average $\Delta M$", ha="center", va="center", rotation=90, fontsize=7)

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="none", label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.50, 0.075),
        ncol=4, frameon=False, fontsize=6.3, handlelength=1.45,
        handletextpad=0.45, columnspacing=1.05, borderaxespad=0.0,
    )

    path = save_figure(fig, output_dir=output_dir, overwrite=overwrite)
    plt.close(fig)

    print("\nMODEL-LEVEL BEHAVIORAL RESILIENCE EFFECTS WITH BOOTSTRAP SE")
    print("-" * 100)
    for probe, probe_label in PROBE_SPECS:
        printable = plot_by_probe[probe].copy()
        printable["model"] = printable["model_name"].map(display_model)
        printable["dataset_name"] = printable["dataset"].map(DATASET_NAMES)
        printable = printable[["dataset_name", "model", "mean", "bootstrap_se", "n_pairs", "n_sequences"]]
        print(f"\n{probe_label}")
        print(printable.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nDisplayed y-axis: [{common_y_min:.4f}, {common_y_max:.4f}]")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure SI 5: instruction-tuned Direct behavioral resilience for SVM and Mass Mean probes."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("outputs/analysis/behavioral_resilience"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--sample", default="round0_agreement", help="Behavioral sample to plot.")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Sequence-stratified bootstrap resamples per cell. Default: 10000.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap random seed. Default: 0.")
    parser.add_argument("--y_min", type=float, default=None, help="Optional common lower y-axis bound.")
    parser.add_argument("--y_max", type=float, default=None, help="Optional common upper y-axis bound.")
    parser.add_argument("--overwrite", action="store_true")
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
    summary_by_probe = {
        probe: prepare_summary(raw_summary, probe=probe, sample=args.sample)
        for probe, _ in PROBE_SPECS
    }
    pairs_by_probe = {
        probe: prepare_pair_results(raw_pairs, probe=probe, sample=args.sample)
        for probe, _ in PROBE_SPECS
    }

    print("=" * 100)
    print("FIGURE SI 5: BEHAVIORAL RESILIENCE (SVM + MASS MEAN)")
    print("=" * 100)
    print(f"Summary:    {summary_path}")
    print(f"Pairs:      {pairs_path}")
    print(f"Output:     {args.output_dir / (OUTPUT_STEM + '.pdf')}")
    print(f"Sample:     {args.sample}")
    print("Estimator:  Direct Conditional only")
    print("Models:     instruction-tuned only")
    print("Probes:     SVM and Mass Mean")
    print(f"Error bars: +/- 1 bootstrap SE ({args.n_bootstrap:,} sequence-stratified resamples)")
    print("=" * 100)

    make_figure(
        summary_by_probe=summary_by_probe,
        pairs_by_probe=pairs_by_probe,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
