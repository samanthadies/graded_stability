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

# Exact instruction-tuned checkpoint set used in the other revised main-text
# figures. Repository keys use a leading underscore for instruction models.
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
ERROR_COLOR = "#222222"

FIGSIZE = (7.2, 2.85)
BAR_WIDTH = 0.72
OUTPUT_STEM = "figure_5"


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
    """Display instruction-tuned model names without '_' or '(i)'."""
    return str(model).lstrip("_")


def latex_escape(text: str) -> str:
    """Escape the small set of LaTeX-special characters used in table text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


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
        & frame["estimator"].astype(str).eq(ESTIMATOR)
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
            "No instruction-tuned Direct behavioral-resilience rows remain "
            f"for probe={probe!r}, sample={sample!r}."
        )

    return out


def add_challenge_sequence_column(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with one canonical `challenge_sequence` column per pair.

    Current behavioral outputs may store either:
      * challenge_sequence, or
      * high_challenge_sequence + low_challenge_sequence.

    When high/low columns are present, verify that the two members of every
    matched pair received the same sequence before using that sequence as the
    bootstrap stratum.
    """
    out = frame.copy()

    high_col = "high_challenge_sequence"
    low_col = "low_challenge_sequence"

    if high_col in out.columns and low_col in out.columns:
        high = out[high_col]
        low = out[low_col]

        comparable = high.notna() & low.notna()
        mismatch = comparable & ~high.astype(str).eq(low.astype(str))
        if mismatch.any():
            bad_cols = [
                col
                for col in (
                    "model_name",
                    "dataset",
                    "match_id",
                    high_col,
                    low_col,
                )
                if col in out.columns
            ]
            raise ValueError(
                "Matched pairs with different high/low challenge sequences "
                "were found:\n"
                + out.loc[mismatch, bad_cols].head(20).to_string(index=False)
            )

        if high.isna().any() or low.isna().any():
            raise ValueError(
                "Missing high/low challenge-sequence values were found in "
                "matched_results_all.parquet."
            )

        # Prefer the explicitly verified pair-level sequence even if an older
        # generic challenge_sequence column also happens to be present.
        out["challenge_sequence"] = high.astype(str)

    elif "challenge_sequence" in out.columns:
        if out["challenge_sequence"].isna().any():
            raise ValueError(
                "Missing challenge_sequence values were found in "
                "matched_results_all.parquet."
            )
        out["challenge_sequence"] = out["challenge_sequence"].astype(str)

    else:
        raise ValueError(
            "matched_results_all.parquet does not contain a usable challenge "
            "sequence column. Expected either `challenge_sequence` or both "
            "`high_challenge_sequence` and `low_challenge_sequence`."
        )

    return out


def prepare_pair_results(
    frame: pd.DataFrame,
    *,
    probe: str,
    sample: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        (
            "model_name",
            "dataset",
            "probe",
            "sample",
            "estimator",
            "match_id",
            "movement_difference",
        ),
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
            "No instruction-tuned Direct pair-level behavioral rows remain "
            f"for probe={probe!r}, sample={sample!r}."
        )

    out["movement_difference"] = pd.to_numeric(
        out["movement_difference"],
        errors="coerce",
    )
    out = out.loc[
        np.isfinite(out["movement_difference"].to_numpy(dtype=float))
    ].copy()

    out = add_challenge_sequence_column(out)

    duplicate_keys = [
        "model_name",
        "dataset",
        "probe",
        "sample",
        "estimator",
        "match_id",
    ]
    duplicates = out.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate matched-pair rows in matched_results_all.parquet:\n"
            + out.loc[duplicates, duplicate_keys]
            .sort_values(duplicate_keys)
            .head(20)
            .to_string(index=False)
        )

    # This is diagnostic rather than required for plotting, but it catches a
    # stale pre-sequence-blocked behavioral output immediately.
    if "matching_method" in out.columns:
        methods = sorted(
            out["matching_method"].dropna().astype(str).unique().tolist()
        )
        unexpected = [
            method
            for method in methods
            if "sequence" not in method.lower()
        ]
        if unexpected:
            raise ValueError(
                "Pair-level results contain matching_method values that do not "
                f"look sequence-blocked: {unexpected}"
            )

    return out


# ---------------------------------------------------------------------------
# Sequence-stratified matched-pair bootstrap
# ---------------------------------------------------------------------------


def stratified_bootstrap_mean_ci(
    frame: pd.DataFrame,
    *,
    value_col: str,
    stratum_col: str,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int]:
    """
    Return observed mean, percentile bootstrap CI, bootstrap SE, and stratum count.

    Matched proposition pairs are the resampling units. Resampling is performed
    independently within each observed challenge-sequence stratum while
    preserving the number of pairs in that stratum. The stratum-specific
    resamples are then recombined and the overall mean is computed using the
    original stratum sample-size weights.
    """
    require_columns(
        frame,
        (value_col, stratum_col),
        source="model x dataset pair-level bootstrap cell",
    )

    work = frame[[value_col, stratum_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[
        np.isfinite(work[value_col].to_numpy(dtype=float))
        & work[stratum_col].notna()
    ].copy()

    if work.empty:
        return np.nan, np.nan, np.nan, np.nan, 0

    observed_mean = float(work[value_col].mean())

    strata: list[np.ndarray] = []
    for _, stratum in work.groupby(stratum_col, sort=True):
        values = stratum[value_col].to_numpy(dtype=float)
        if len(values):
            strata.append(values)

    if not strata:
        return observed_mean, observed_mean, observed_mean, 0.0, 0

    n_total = int(sum(len(values) for values in strata))
    if n_total == 1:
        return observed_mean, observed_mean, observed_mean, 0.0, len(strata)

    bootstrap_means = np.empty(n_bootstrap, dtype=float)

    # Chunking prevents very large temporary arrays when a cell has many pairs.
    chunk_size = 1000
    filled = 0

    while filled < n_bootstrap:
        current = min(chunk_size, n_bootstrap - filled)
        replicate_sums = np.zeros(current, dtype=float)

        for values in strata:
            indices = rng.integers(
                0,
                len(values),
                size=(current, len(values)),
            )
            replicate_sums += values[indices].sum(axis=1)

        bootstrap_means[filled : filled + current] = (
            replicate_sums / float(n_total)
        )
        filled += current

    alpha = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [alpha, 1.0 - alpha],
    )

    bootstrap_se = float(np.std(bootstrap_means, ddof=1))

    return (
        observed_mean,
        float(ci_low),
        float(ci_high),
        bootstrap_se,
        len(strata),
    )



def pair_level_dispersion(
    frame: pd.DataFrame,
    *,
    value_col: str,
    stratum_col: str,
) -> tuple[float, float, float]:
    """
    Return descriptive pair-level SD, naive IID SE, and a stratum-aware SE.

    pair_sd:
        Sample SD of pair-level Delta M values. This measures heterogeneity
        across matched proposition pairs; it is not uncertainty in the mean.

    iid_se:
        pair_sd / sqrt(n). Included only as a familiar diagnostic. It ignores
        the challenge-sequence blocking.

    stratified_se:
        Fixed-allocation stratified SE based on within-sequence sample
        variances:
            sqrt(sum_h n_h * s_h^2) / n_total.
        This mirrors the fixed stratum sizes used by the sequence-stratified
        bootstrap and should be close to the bootstrap SE for sufficiently
        large samples.
    """
    work = frame[[value_col, stratum_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[
        np.isfinite(work[value_col].to_numpy(dtype=float))
        & work[stratum_col].notna()
    ].copy()

    n_total = len(work)
    if n_total == 0:
        return np.nan, np.nan, np.nan

    if n_total == 1:
        return 0.0, 0.0, 0.0

    values = work[value_col].to_numpy(dtype=float)
    pair_sd = float(np.std(values, ddof=1))
    iid_se = float(pair_sd / np.sqrt(n_total))

    variance_sum = 0.0
    for _, stratum in work.groupby(stratum_col, sort=True):
        stratum_values = stratum[value_col].to_numpy(dtype=float)
        n_h = len(stratum_values)
        if n_h <= 1:
            # A one-pair stratum has no estimable within-stratum variance.
            # Treat its empirical resampling variance as zero, matching the
            # nonparametric bootstrap behavior for that singleton stratum.
            continue
        s_h_sq = float(np.var(stratum_values, ddof=1))
        variance_sum += n_h * s_h_sq

    stratified_se = float(
        np.sqrt(variance_sum) / n_total
    )

    return pair_sd, iid_se, stratified_se



def build_plot_table(
    *,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    """
    Build the 36 model x dataset plotting cells and their bootstrap intervals.

    The saved model-level mean is retained as the plotted bar height, but it is
    first checked against the mean reconstructed directly from pair-level
    movement_difference values.
    """
    rows: list[dict[str, object]] = []
    models = ordered_models()

    summary_indexed = summary.set_index(["model_name", "dataset"])

    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(models):
            key = (model, dataset)

            if key not in summary_indexed.index:
                log.warning(
                    "Missing model-level summary row for %s / %s",
                    model,
                    dataset,
                )
                rows.append(
                    {
                        "model_name": model,
                        "dataset": dataset,
                        "mean": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_pairs": 0,
                        "n_sequences": 0,
                        "pair_sd": np.nan,
                        "iid_se": np.nan,
                        "stratified_se": np.nan,
                        "bootstrap_se": np.nan,
                    }
                )
                continue

            summary_row = summary_indexed.loc[key]
            if isinstance(summary_row, pd.DataFrame):
                raise ValueError(f"Non-unique summary key: {key}")

            saved_mean = float(summary_row["mean_movement_difference"])

            cell = pairs.loc[
                pairs["model_name"].astype(str).eq(model)
                & pairs["dataset"].astype(str).eq(dataset)
            ].copy()

            if cell.empty:
                raise RuntimeError(
                    "No pair-level movement_difference rows for a model-level "
                    f"effect that exists in matched_unit_summary: {model} / {dataset}."
                )

            # Stable but distinct bootstrap stream for every model x dataset cell.
            cell_seed = seed + dataset_index * 1000 + model_index
            rng = np.random.default_rng(cell_seed)

            pair_mean, ci_low, ci_high, bootstrap_se, n_sequences = (
                stratified_bootstrap_mean_ci(
                    cell,
                    value_col="movement_difference",
                    stratum_col="challenge_sequence",
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    rng=rng,
                )
            )

            pair_sd, iid_se, stratified_se = pair_level_dispersion(
                cell,
                value_col="movement_difference",
                stratum_col="challenge_sequence",
            )

            if not np.isclose(
                saved_mean,
                pair_mean,
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError(
                    "Pair-level movement_difference mean does not match "
                    "matched_unit_summary mean_movement_difference for "
                    f"{model} / {dataset}: summary={saved_mean:.12g}, "
                    f"pairs={pair_mean:.12g}."
                )

            rows.append(
                {
                    "model_name": model,
                    "dataset": dataset,
                    "mean": saved_mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_pairs": int(len(cell)),
                    "n_sequences": int(n_sequences),
                    "pair_sd": pair_sd,
                    "iid_se": iid_se,
                    "stratified_se": stratified_se,
                    "bootstrap_se": bootstrap_se,
                }
            )

    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Supplementary uncertainty table
# ---------------------------------------------------------------------------


def write_uncertainty_table(
    plot_table: pd.DataFrame,
    *,
    output_dir: Path,
    probe: str,
    sample: str,
    n_bootstrap: int,
    ci_level: float,
    overwrite: bool,
) -> Path:
    """
    Write a booktabs-style LaTeX table with model-level behavioral effects.

    The main figure visualizes +/- 1 bootstrap SE. This table reports the
    corresponding mean, bootstrap SE, full percentile bootstrap CI, and number
    of retained matched proposition pairs for every displayed model x dataset
    cell.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = (
        output_dir
        / f"table_si_behavioral_resilience_uncertainty_{probe}.tex"
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")

    models = ordered_models()
    ci_percent = 100.0 * ci_level
    ci_label = (
        f"{ci_percent:.0f}\\% CI"
        if np.isclose(ci_percent, round(ci_percent))
        else f"{ci_percent:.1f}\\% CI"
    )

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"\textbf{Dataset} & \textbf{Model} & \textbf{$n$} & "
            r"\textbf{Mean $\Delta M$} & \textbf{Bootstrap SE} & "
            rf"\textbf{{{ci_label}}} \\"
        ),
        r"\midrule",
    ]

    for dataset_index, dataset in enumerate(DATASETS):
        panel = (
            plot_table.loc[
                plot_table["dataset"].astype(str).eq(dataset)
            ]
            .set_index("model_name")
            .reindex(models)
        )

        for model_index, model in enumerate(models):
            row = panel.loc[model]

            dataset_text = (
                latex_escape(DATASET_NAMES[dataset])
                if model_index == 0
                else ""
            )
            model_text = latex_escape(display_model(model))

            if (
                np.isfinite(row["mean"])
                and np.isfinite(row["bootstrap_se"])
                and np.isfinite(row["ci_low"])
                and np.isfinite(row["ci_high"])
            ):
                n_text = f"${int(row['n_pairs'])}$"
                mean_text = f"${float(row['mean']):.4f}$"
                se_text = f"${float(row['bootstrap_se']):.4f}$"
                ci_text = (
                    f"$[{float(row['ci_low']):.4f},\\,"
                    f"{float(row['ci_high']):.4f}]$"
                )
            else:
                n_text = "$0$"
                mean_text = "--"
                se_text = "--"
                ci_text = "--"

            lines.append(
                f"{dataset_text} & \\texttt{{{model_text}}} & "
                f"{n_text} & {mean_text} & {se_text} & {ci_text} \\\\"
            )

        if dataset_index < len(DATASETS) - 1:
            lines.append(r"\addlinespace[2.5pt]")

    sample_text = latex_escape(sample)
    probe_text = latex_escape(probe)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{\textbf{Uncertainty in behavioral resilience effects.} "
                r"For each instruction-tuned LLM, we report the number of retained "
                r"matched proposition pairs $n$, the mean behavioral movement "
                r"difference $\Delta M$, the bootstrap standard error, and the "
                + ci_label
                + r" for the primary \texttt{"
                + sample_text
                + r"} sample using the \texttt{"
                + probe_text
                + r"} probe and Direct Conditional graded stability. "
                + rf"Uncertainty is estimated from {n_bootstrap:,} "
                r"sequence-stratified matched-pair bootstrap resamples. "
                r"The main-text figure displays $\pm 1$ bootstrap standard error; "
                r"the full percentile bootstrap confidence intervals are reported here.}"
            ),
            r"\label{tab:si:behavioral_resilience_uncertainty}",
            r"\end{table*}",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)
    return path


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


def automatic_y_limits(
    plot_table: pd.DataFrame,
) -> tuple[float, float, float]:
    means = pd.to_numeric(
        plot_table["mean"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ses = pd.to_numeric(
        plot_table["bootstrap_se"],
        errors="coerce",
    ).to_numpy(dtype=float)

    lows = means - ses
    highs = means + ses

    all_values = np.concatenate(
        [
            lows,
            highs,
            means,
            np.array([0.0]),
        ]
    )
    all_values = all_values[np.isfinite(all_values)]

    if len(all_values) == 0:
        return -0.05, 0.20, 0.05

    raw_min = float(np.min(all_values))
    raw_max = float(np.max(all_values))
    raw_span = max(raw_max - raw_min, 0.05)
    pad = 0.08 * raw_span

    target_min = min(0.0, raw_min - pad)
    target_max = max(0.0, raw_max + pad)
    step = nice_step(target_max - target_min)

    y_min = math.floor(target_min / step) * step
    y_max = math.ceil(target_max / step) * step

    # Keep at least one tick below/above zero when needed so the zero reference
    # remains visually distinct from the frame.
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
    table_output_dir: Path,
    probe: str,
    sample: str,
    overwrite: bool,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    y_min: float | None,
    y_max: float | None,
) -> Path:
    plot_table = build_plot_table(
        summary=summary,
        pairs=pairs,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )

    table_path = write_uncertainty_table(
        plot_table,
        output_dir=table_output_dir,
        probe=probe,
        sample=sample,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        overwrite=overwrite,
    )

    auto_min, auto_max, tick_step = automatic_y_limits(plot_table)
    common_y_min = auto_min if y_min is None else float(y_min)
    common_y_max = auto_max if y_max is None else float(y_max)
    if common_y_min >= common_y_max:
        raise ValueError("y_min must be smaller than y_max.")

    means_for_limits = pd.to_numeric(
        plot_table["mean"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ses_for_limits = pd.to_numeric(
        plot_table["bootstrap_se"],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_low = means_for_limits - ses_for_limits
    finite_high = means_for_limits + ses_for_limits
    finite_low = finite_low[np.isfinite(finite_low)]
    finite_high = finite_high[np.isfinite(finite_high)]

    if len(finite_low) and float(np.min(finite_low)) < common_y_min:
        raise ValueError(
            f"y_min={common_y_min:g} clips a +/- 1 bootstrap SE error bar; "
            f"need <= {np.min(finite_low):.4f}."
        )
    if len(finite_high) and float(np.max(finite_high)) > common_y_max:
        raise ValueError(
            f"y_max={common_y_max:g} clips a +/- 1 bootstrap SE error bar; "
            f"need >= {np.max(finite_high):.4f}."
        )

    models = ordered_models()
    x = np.arange(1, len(models) + 1, dtype=float)
    colors = [
        FAMILY_COLORS[model_family(model)]
        for model in models
    ]

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

    for col_index, (ax, dataset) in enumerate(
        zip(axes, DATASETS)
    ):
        panel = (
            plot_table.loc[
                plot_table["dataset"].astype(str).eq(dataset)
            ]
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

        valid = (
            np.isfinite(means)
            & np.isfinite(ses)
        )

        ax.errorbar(
            x[valid],
            means[valid],
            yerr=ses[valid],
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            zorder=5,
        )

        ax.axhline(
            0.0,
            color=ZERO_LINE_COLOR,
            linewidth=0.70,
            linestyle="--",
            zorder=1,
        )

        ax.set_xlim(
            0.35,
            len(models) + 0.65,
        )
        ax.set_ylim(
            common_y_min,
            common_y_max,
        )

        # If the user manually chooses either limit, derive a clean tick step
        # from the displayed range rather than preserving the automatic one.
        if y_min is not None or y_max is not None:
            tick_step = nice_step(
                common_y_max - common_y_min
            )

        first_tick = (
            math.ceil(
                common_y_min / tick_step - 1e-10
            )
            * tick_step
        )
        yticks = np.arange(
            first_tick,
            common_y_max + tick_step * 0.5,
            tick_step,
        )
        ax.set_yticks(yticks)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                display_model(model)
                for model in models
            ],
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
            ax.tick_params(
                axis="y",
                labelleft=False,
            )
            ax.spines["left"].set_visible(False)

    fig.canvas.draw()
    first_pos = axes[0].get_position()

    fig.text(
        0.035,
        first_pos.y1 + 0.105,
        r"Higher $\gamma(P)$ is associated with greater resistance to challenge",
        ha="left",
        va="bottom",
        fontsize=7.7,
        fontweight="bold",
        color=TITLE_COLOR,
    )
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
        bbox_to_anchor=(0.50, 0.025),
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
    printable["model"] = printable["model_name"].map(
        display_model
    )
    printable["dataset_name"] = printable["dataset"].map(
        DATASET_NAMES
    )
    printable = printable[
        [
            "dataset_name",
            "model",
            "mean",
            "pair_sd",
            "iid_se",
            "stratified_se",
            "bootstrap_se",
            "ci_low",
            "ci_high",
            "n_pairs",
            "n_sequences",
        ]
    ]

    print(
        "\nMODEL-LEVEL BEHAVIORAL RESILIENCE EFFECTS "
        "WITH BOOTSTRAP SE AND FULL PERCENTILE CIs"
    )
    print("-" * 120)
    print(
        printable.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(
        f"\nBootstrap: {n_bootstrap:,} matched-pair resamples per "
        "model x dataset, stratified by challenge sequence."
    )
    print(
        "Figure error bars: +/- 1 bootstrap SE. "
        f"Supplementary table CI: {100 * ci_level:.1f}% percentile bootstrap."
    )
    print(
        "Diagnostics: pair_sd = SD of pair-level Delta M; "
        "iid_se = pair_sd/sqrt(n); stratified_se = analytic fixed-stratum SE; "
        "bootstrap_se = SD of the stratified bootstrap mean distribution."
    )
    print(
        f"Displayed y-axis: "
        f"[{common_y_min:.4f}, {common_y_max:.4f}]"
    )
    print(f"Supplementary LaTeX table: {table_path}")

    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 5: instruction-tuned Direct model-level behavioral "
            "resilience effects with +/- 1 sequence-stratified bootstrap "
            "SE error bars and a supplementary percentile-CI table."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            "outputs/analysis/behavioral_resilience"
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
        choices=[
            "sawmil",
            "svm",
            "mean_difference",
        ],
        help=(
            "Probe shown in the figure. "
            "Main-text default: sawmil."
        ),
    )
    parser.add_argument(
        "--sample",
        default="round0_agreement",
        help=(
            "Behavioral sample to plot. "
            "Main-text default: round0_agreement."
        ),
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help=(
            "Number of sequence-stratified matched-pair bootstrap "
            "resamples per model x dataset. Default: 10000."
        ),
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help=(
            "Percentile-bootstrap confidence level for the SI table. "
            "The main figure always shows +/- 1 bootstrap SE. Default: 0.95."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Bootstrap random seed. Default: 0."
        ),
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
        raise ValueError(
            "--n_bootstrap must be at least 100."
        )
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError(
            "--ci_level must be in (0, 1)."
        )

    summary_path = (
        args.input_dir
        / "matched_unit_summary.parquet"
    )
    pairs_path = (
        args.input_dir
        / "matched_results_all.parquet"
    )

    for path in (
        summary_path,
        pairs_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = prepare_summary(
        pd.read_parquet(summary_path),
        probe=args.probe,
        sample=args.sample,
    )
    pairs = prepare_pair_results(
        pd.read_parquet(pairs_path),
        probe=args.probe,
        sample=args.sample,
    )

    print("=" * 100)
    print("FIGURE 5: BEHAVIORAL RESILIENCE")
    print("=" * 100)
    print(f"Summary input: {summary_path}")
    print(f"Pair input:    {pairs_path}")
    print(
        f"Figure output: "
        f"{args.output_dir / (OUTPUT_STEM + '.pdf')}"
    )
    print(
        f"Table output:  "
        f"{args.input_dir / f'table_si_behavioral_resilience_uncertainty_{args.probe}.tex'}"
    )
    print(f"Probe:         {args.probe}")
    print(f"Sample:        {args.sample}")
    print("Estimator:     Direct Conditional only")
    print("Models:        instruction-tuned only")
    print(
        f"Bootstrap:     {args.n_bootstrap:,} resamples; "
        "figure shows +/- 1 SE; "
        f"SI table reports {100 * args.ci_level:.1f}% CI; "
        "stratified by challenge sequence"
    )
    print("=" * 100)

    make_figure(
        summary=summary,
        pairs=pairs,
        output_dir=args.output_dir,
        table_output_dir=args.input_dir,
        probe=args.probe,
        sample=args.sample,
        overwrite=args.overwrite,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        seed=args.seed,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
