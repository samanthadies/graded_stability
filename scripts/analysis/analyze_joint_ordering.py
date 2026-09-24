"""
Tests the Joint-to-Conditional estimator's sensitivity to conjunction order by comparing
results from "x and P" with otherwise matched results from "P and x".

Examples:
    python -m scripts.analysis.analyze_joint_ordering --probes sawmil --overwrite
    python -m scripts.analysis.analyze_joint_ordering --models _llama-3.1-8b --datasets cities_loc --probes sawmil --overwrite

To generate the alternate-order inputs first:
    python -m scripts.beliefs.score_joint --model_name _llama-3.1-8b --dataset cities_loc --template P_then_x --output_dir outputs/joint_ordering/P_then_x
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


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

PROBES = (
    "sawmil",
    "svm",
    "mean_difference",
)

JOINT_LABELS = (
    "TT",
    "TF",
    "TN",
    "FT",
    "FF",
    "FN",
    "NT",
    "NF",
    "NN",
)

JOINT_PROBABILITY_COLUMNS = tuple(
    f"prob_{label}"
    for label in JOINT_LABELS
)

DEFAULT_ZERO_TOLERANCE = 1e-12
PROBABILITY_SUM_ATOL = 1e-5
PROBABILITY_BOUND_ATOL = 1e-8

FAMILY_ORDER = (
    "llama",
    "gemma",
    "mistral",
    "qwen",
)

PAIR_LEVEL_COLUMNS = [
    "model",
    "dataset",
    "probe",
    "pair_id",
    "P_id",
    "x_id",
    "threshold",
    "conditional_x_then_P",
    "conditional_P_then_x",
    "defined_x_then_P",
    "defined_P_then_x",
    "defined_both",
    "abs_conditional_order_difference",
]

STATEMENT_LEVEL_COLUMNS = [
    "model",
    "dataset",
    "probe",
    "P_id",
    "threshold",
    "num_pairs",
    "num_defined_x_then_P",
    "num_defined_P_then_x",
    "num_defined_both",
    "num_above_x_then_P",
    "num_above_P_then_x",
    "gamma_x_then_P",
    "gamma_P_then_x",
    "abs_gamma_order_difference",
]

MODEL_SUMMARY_COLUMNS = [
    "model",
    "dataset",
    "probe",
    "threshold",
    "n_pairs_total",
    "n_pairs_defined_x_then_P",
    "n_pairs_defined_P_then_x",
    "n_pairs_defined_both",
    "pair_defined_fraction_both",
    "median_abs_conditional_order_difference",
    "mean_abs_conditional_order_difference",
    "max_abs_conditional_order_difference",
    "n_P_total",
    "n_P_gamma_defined_x_then_P",
    "n_P_gamma_defined_P_then_x",
    "n_P_gamma_defined_both",
    "median_abs_gamma_order_difference",
    "mean_abs_gamma_order_difference",
    "max_abs_gamma_order_difference",
]



class DegenerateCase(Exception):
    """Expected scientific edge case with no beliefs/pairs to analyze."""


STATUS_COLUMNS = [
    "model",
    "dataset",
    "probe",
    "status",
    "reason",
    "atomic_path",
    "pairs_path",
    "joint_x_then_P_path",
    "joint_P_then_x_path",
    "n_pairs",
    "n_P",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze order sensitivity of the Joint-to-Conditional estimator "
            "for x-and-P versus P-and-x conjunctions."
        )
    )

    parser.add_argument(
        "--atomic_dir",
        type=Path,
        default=Path("outputs/atomic"),
    )
    parser.add_argument(
        "--pairs_dir",
        type=Path,
        default=Path("outputs/pairs"),
    )
    parser.add_argument(
        "--joint_x_then_P_dir",
        type=Path,
        default=Path("outputs/joint"),
        help="Canonical main-analysis Joint outputs for 'x and P'.",
    )
    parser.add_argument(
        "--joint_P_then_x_dir",
        type=Path,
        default=Path("outputs/joint_ordering/P_then_x"),
        help="Ordering-experiment Joint outputs for 'P and x'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/analysis/joint_ordering"),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional model subset. By default, discover model directories "
            "from --atomic_dir."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=PROBES,
        default=list(PROBES),
    )

    parser.add_argument(
        "--zero_tolerance",
        type=float,
        default=DEFAULT_ZERO_TOLERANCE,
        help=(
            "Treat Joint conditional denominators <= this value as undefined. "
            "Default: 1e-12."
        ),
    )
    parser.add_argument(
        "--skip_probability_validation",
        action="store_true",
        help="Skip Joint probability bound/simplex validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


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


def model_family(model: str) -> str:
    bare = str(model).lstrip("_").lower()
    for family in FAMILY_ORDER:
        if bare.startswith(family):
            return family
    return bare.split("-")[0]


def model_size(model: str) -> float:
    bare = str(model).lstrip("_").lower()
    matches = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            bare,
        )
    ]
    return max(matches) if matches else float("inf")


def model_sort_key(model: str) -> tuple[Any, ...]:
    family = model_family(model)
    try:
        family_rank = FAMILY_ORDER.index(family)
    except ValueError:
        family_rank = 99

    instruct_rank = 1 if str(model).startswith("_") else 0
    return (
        family_rank,
        model_size(model),
        instruct_rank,
        str(model).lstrip("_").lower(),
    )


def discover_models(atomic_dir: Path) -> list[str]:
    if not atomic_dir.exists():
        raise FileNotFoundError(
            f"Atomic output directory does not exist: {atomic_dir}"
        )

    models = sorted(
        [
            path.name
            for path in atomic_dir.iterdir()
            if path.is_dir()
        ],
        key=model_sort_key,
    )

    if not models:
        raise RuntimeError(
            f"No model directories found under {atomic_dir}."
        )

    return models


def ensure_output_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists; pass --overwrite to replace it."
        )


def validate_probability_matrix(
    frame: pd.DataFrame,
    *,
    source: str,
) -> None:
    require_columns(
        frame,
        JOINT_PROBABILITY_COLUMNS,
        source=source,
    )

    matrix = frame.loc[:, JOINT_PROBABILITY_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).any(axis=1).sum())
        raise ValueError(
            f"{source}: {bad:,} rows contain non-finite Joint probabilities."
        )

    if (
        np.min(matrix) < -PROBABILITY_BOUND_ATOL
        or np.max(matrix) > 1.0 + PROBABILITY_BOUND_ATOL
    ):
        raise ValueError(
            f"{source}: Joint probabilities fall outside [0, 1] within "
            f"tolerance {PROBABILITY_BOUND_ATOL}."
        )

    sums = matrix.sum(axis=1)
    bad_sum = ~np.isclose(
        sums,
        1.0,
        atol=PROBABILITY_SUM_ATOL,
        rtol=0.0,
    )
    if bad_sum.any():
        raise ValueError(
            f"{source}: {int(bad_sum.sum()):,} rows do not sum to 1 within "
            f"atol={PROBABILITY_SUM_ATOL}; observed sum range "
            f"[{sums.min():.6f}, {sums.max():.6f}]."
        )


def read_probe_rows(
    path: Path,
    *,
    probe: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read one probe from a Parquet file, with a fallback for unpartitioned files."""
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("probe", "==", probe)],
        )
    except (ValueError, TypeError, OSError):
        frame = pd.read_parquet(
            path,
            columns=columns,
        )
        frame = frame.loc[
            frame["probe"].astype(str).eq(probe)
        ].copy()

    return frame


def load_atomic_probe(
    path: Path,
    *,
    probe: str,
) -> pd.DataFrame:
    columns = [
        "probe",
        "statement_id",
        "threshold",
        "is_P",
    ]

    frame = read_probe_rows(
        path,
        probe=probe,
        columns=columns,
    )

    return frame


def probe_threshold(
    atomic: pd.DataFrame,
    *,
    probe: str,
) -> float:
    if atomic.empty:
        raise ValueError(
            f"{probe}: atomic table contains no rows."
        )

    values = (
        pd.to_numeric(
            atomic["threshold"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(values) != 1:
        raise ValueError(
            f"{probe}: expected exactly one finite atomic threshold; "
            f"found {values.tolist()}."
        )

    threshold = float(values[0])
    if not np.isfinite(threshold):
        raise ValueError(
            f"{probe}: atomic threshold is non-finite."
        )

    return threshold


def expected_P_ids(
    atomic: pd.DataFrame,
) -> set[int]:
    subset = atomic.loc[
        atomic["is_P"].astype(bool)
    ]

    return set(
        pd.to_numeric(
            subset["statement_id"],
            errors="raise",
        )
        .astype(np.int64)
        .tolist()
    )


def load_pairs(
    path: Path,
    *,
    probe: str,
) -> pd.DataFrame:
    columns = [
        "probe",
        "pair_id",
        "P_id",
        "x_id",
    ]

    frame = read_probe_rows(
        path,
        probe=probe,
        columns=columns,
    )

    if frame.empty:
        return frame

    frame = frame.copy()
    for column in ("pair_id", "P_id", "x_id"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        ).astype(np.int64)

    if frame["pair_id"].duplicated().any():
        raise ValueError(
            f"{probe}: pair table has duplicate pair_id values."
        )

    return frame


def validate_P_coverage(
    *,
    pairs: pd.DataFrame,
    expected: set[int],
    probe: str,
) -> None:
    observed = set(
        pairs["P_id"].astype(np.int64).tolist()
    )

    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise ValueError(
            f"{probe}: pair-table P IDs do not match atomic is_P. "
            f"Missing={len(missing)}, extra={len(extra)}."
        )


def load_joint_scores(
    path: Path,
    *,
    probe: str,
    validate_probabilities: bool,
) -> pd.DataFrame:
    columns = [
        "probe",
        "pair_id",
        *JOINT_PROBABILITY_COLUMNS,
    ]

    frame = read_probe_rows(
        path,
        probe=probe,
        columns=columns,
    )

    if frame.empty:
        return frame

    frame = frame.copy()
    frame["pair_id"] = pd.to_numeric(
        frame["pair_id"],
        errors="raise",
    ).astype(np.int64)

    if frame["pair_id"].duplicated().any():
        raise ValueError(
            f"{probe}: Joint score table {path} has duplicate pair_id values."
        )

    for column in JOINT_PROBABILITY_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if validate_probabilities:
        validate_probability_matrix(
            frame,
            source=f"{path} [{probe}]",
        )

    return frame


def joint_to_conditional(
    frame: pd.DataFrame,
    *,
    order: str,
    zero_tolerance: float,
) -> pd.DataFrame:
    """
    Convert one linguistic ordering of the nine-class Joint distribution into
    canonical Pr(P | x).

    x_then_P means the Joint labels already represent (x_state, P_state).

    P_then_x means the Joint labels represent (P_state, x_state), so the 3x3
    table is transposed conceptually before applying the same conditional rule.
    """
    if order == "x_then_P":
        numerator = (
            frame["prob_TT"].to_numpy(dtype=float)
            + frame["prob_NT"].to_numpy(dtype=float)
        )
        denominator = (
            frame["prob_TT"].to_numpy(dtype=float)
            + frame["prob_TF"].to_numpy(dtype=float)
            + frame["prob_NT"].to_numpy(dtype=float)
            + frame["prob_NF"].to_numpy(dtype=float)
        )
    elif order == "P_then_x":
        numerator = (
            frame["prob_TT"].to_numpy(dtype=float)
            + frame["prob_TN"].to_numpy(dtype=float)
        )
        denominator = (
            frame["prob_TT"].to_numpy(dtype=float)
            + frame["prob_FT"].to_numpy(dtype=float)
            + frame["prob_TN"].to_numpy(dtype=float)
            + frame["prob_FN"].to_numpy(dtype=float)
        )
    else:
        raise ValueError(
            f"Unknown Joint ordering: {order!r}."
        )

    defined = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > zero_tolerance)
    )

    conditional = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )
    conditional[defined] = (
        numerator[defined]
        / denominator[defined]
    )

    # Allow tiny numerical excursions only within floating-point tolerance.
    bad = (
        defined
        & (
            (conditional < -PROBABILITY_BOUND_ATOL)
            | (conditional > 1.0 + PROBABILITY_BOUND_ATOL)
        )
    )
    if bad.any():
        raise ValueError(
            f"{order}: {int(bad.sum()):,} derived conditionals fall outside "
            "[0, 1]."
        )

    conditional[defined] = np.clip(
        conditional[defined],
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "pair_id": frame["pair_id"].to_numpy(dtype=np.int64),
            f"conditional_{order}": conditional,
            f"defined_{order}": defined,
        }
    )


def attach_exact_pair_coverage(
    *,
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    probe: str,
    order: str,
) -> pd.DataFrame:
    pair_ids = set(
        pairs["pair_id"].astype(np.int64).tolist()
    )
    score_ids = set(
        scores["pair_id"].astype(np.int64).tolist()
    )

    missing = pair_ids - score_ids
    extra = score_ids - pair_ids
    if missing or extra:
        raise ValueError(
            f"{probe}/{order}: score coverage does not match pair table. "
            f"Missing scores={len(missing):,}; extra scores={len(extra):,}."
        )

    merged = pairs[
        ["pair_id", "P_id", "x_id"]
    ].merge(
        scores,
        on="pair_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(pairs):
        raise RuntimeError(
            f"{probe}/{order}: pair/score merge changed row count."
        )

    return merged


def compute_statement_gamma(
    pair_level: pd.DataFrame,
    *,
    model: str,
    dataset: str,
    probe: str,
    threshold: float,
) -> pd.DataFrame:
    working = pair_level.copy()

    working["above_x_then_P"] = (
        working["defined_x_then_P"].astype(bool)
        & (
            working["conditional_x_then_P"]
            > threshold
        )
    )
    working["above_P_then_x"] = (
        working["defined_P_then_x"].astype(bool)
        & (
            working["conditional_P_then_x"]
            > threshold
        )
    )

    records: list[dict[str, Any]] = []

    for P_id, group in working.groupby(
        "P_id",
        sort=False,
        dropna=False,
    ):
        defined_xp = group["defined_x_then_P"].astype(bool)
        defined_px = group["defined_P_then_x"].astype(bool)
        defined_both = defined_xp & defined_px

        n_xp = int(defined_xp.sum())
        n_px = int(defined_px.sum())
        n_both = int(defined_both.sum())

        above_xp = int(group.loc[defined_xp, "above_x_then_P"].sum())
        above_px = int(group.loc[defined_px, "above_P_then_x"].sum())

        gamma_xp = (
            float(above_xp / n_xp)
            if n_xp > 0
            else np.nan
        )
        gamma_px = (
            float(above_px / n_px)
            if n_px > 0
            else np.nan
        )

        abs_diff = (
            abs(gamma_xp - gamma_px)
            if np.isfinite(gamma_xp) and np.isfinite(gamma_px)
            else np.nan
        )

        records.append(
            {
                "model": model,
                "dataset": dataset,
                "probe": probe,
                "P_id": int(P_id),
                "threshold": float(threshold),
                "num_pairs": int(len(group)),
                "num_defined_x_then_P": n_xp,
                "num_defined_P_then_x": n_px,
                "num_defined_both": n_both,
                "num_above_x_then_P": above_xp,
                "num_above_P_then_x": above_px,
                "gamma_x_then_P": gamma_xp,
                "gamma_P_then_x": gamma_px,
                "abs_gamma_order_difference": abs_diff,
            }
        )

    return pd.DataFrame(
        records,
        columns=STATEMENT_LEVEL_COLUMNS,
    )


def finite_summary(
    values: pd.Series,
) -> tuple[float, float, float]:
    arr = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.median(arr)),
        float(np.mean(arr)),
        float(np.max(arr)),
    )


def summarize_model_probe(
    *,
    pair_level: pd.DataFrame,
    statement_level: pd.DataFrame,
    model: str,
    dataset: str,
    probe: str,
    threshold: float,
) -> dict[str, Any]:
    pair_median, pair_mean, pair_max = finite_summary(
        pair_level["abs_conditional_order_difference"]
    )
    gamma_median, gamma_mean, gamma_max = finite_summary(
        statement_level["abs_gamma_order_difference"]
    )

    n_pairs = int(len(pair_level))
    n_pairs_xp = int(pair_level["defined_x_then_P"].astype(bool).sum())
    n_pairs_px = int(pair_level["defined_P_then_x"].astype(bool).sum())
    n_pairs_both = int(pair_level["defined_both"].astype(bool).sum())

    gamma_xp_defined = np.isfinite(
        pd.to_numeric(
            statement_level["gamma_x_then_P"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )
    gamma_px_defined = np.isfinite(
        pd.to_numeric(
            statement_level["gamma_P_then_x"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    return {
        "model": model,
        "dataset": dataset,
        "probe": probe,
        "threshold": float(threshold),
        "n_pairs_total": n_pairs,
        "n_pairs_defined_x_then_P": n_pairs_xp,
        "n_pairs_defined_P_then_x": n_pairs_px,
        "n_pairs_defined_both": n_pairs_both,
        "pair_defined_fraction_both": (
            float(n_pairs_both / n_pairs)
            if n_pairs > 0
            else np.nan
        ),
        "median_abs_conditional_order_difference": pair_median,
        "mean_abs_conditional_order_difference": pair_mean,
        "max_abs_conditional_order_difference": pair_max,
        "n_P_total": int(len(statement_level)),
        "n_P_gamma_defined_x_then_P": int(gamma_xp_defined.sum()),
        "n_P_gamma_defined_P_then_x": int(gamma_px_defined.sum()),
        "n_P_gamma_defined_both": int((gamma_xp_defined & gamma_px_defined).sum()),
        "median_abs_gamma_order_difference": gamma_median,
        "mean_abs_gamma_order_difference": gamma_mean,
        "max_abs_gamma_order_difference": gamma_max,
    }


def analyze_one(
    *,
    model: str,
    dataset: str,
    probe: str,
    atomic_path: Path,
    pairs_path: Path,
    joint_xp_path: Path,
    joint_px_path: Path,
    zero_tolerance: float,
    validate_probabilities: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    atomic = load_atomic_probe(
        atomic_path,
        probe=probe,
    )

    if atomic.empty:
        raise DegenerateCase(
            f"{model}/{dataset}/{probe}: no atomic rows."
        )

    expected_P = expected_P_ids(atomic)
    if not expected_P:
        raise DegenerateCase(
            f"{model}/{dataset}/{probe}: atomic belief set P is empty."
        )

    threshold = probe_threshold(
        atomic,
        probe=probe,
    )

    pairs = load_pairs(
        pairs_path,
        probe=probe,
    )
    if pairs.empty:
        raise DegenerateCase(
            f"{model}/{dataset}/{probe}: pair table contains no rows."
        )

    validate_P_coverage(
        pairs=pairs,
        expected=expected_P,
        probe=probe,
    )

    scores_xp = load_joint_scores(
        joint_xp_path,
        probe=probe,
        validate_probabilities=validate_probabilities,
    )
    scores_px = load_joint_scores(
        joint_px_path,
        probe=probe,
        validate_probabilities=validate_probabilities,
    )

    if scores_xp.empty:
        raise ValueError(
            f"{model}/{dataset}/{probe}: x_then_P Joint score table has no rows."
        )
    if scores_px.empty:
        raise ValueError(
            f"{model}/{dataset}/{probe}: P_then_x Joint score table has no rows."
        )

    conditional_xp = joint_to_conditional(
        scores_xp,
        order="x_then_P",
        zero_tolerance=zero_tolerance,
    )
    conditional_px = joint_to_conditional(
        scores_px,
        order="P_then_x",
        zero_tolerance=zero_tolerance,
    )

    merged_xp = attach_exact_pair_coverage(
        pairs=pairs,
        scores=conditional_xp,
        probe=probe,
        order="x_then_P",
    )
    merged_px = attach_exact_pair_coverage(
        pairs=pairs,
        scores=conditional_px,
        probe=probe,
        order="P_then_x",
    )

    pair_level = merged_xp.merge(
        merged_px[
            [
                "pair_id",
                "conditional_P_then_x",
                "defined_P_then_x",
            ]
        ],
        on="pair_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(pair_level) != len(pairs):
        raise RuntimeError(
            f"{model}/{dataset}/{probe}: joining the two orders changed pair count."
        )

    pair_level["defined_both"] = (
        pair_level["defined_x_then_P"].astype(bool)
        & pair_level["defined_P_then_x"].astype(bool)
    )

    pair_level["abs_conditional_order_difference"] = np.where(
        pair_level["defined_both"].to_numpy(dtype=bool),
        np.abs(
            pair_level["conditional_x_then_P"].to_numpy(dtype=float)
            - pair_level["conditional_P_then_x"].to_numpy(dtype=float)
        ),
        np.nan,
    )

    pair_level.insert(0, "probe", probe)
    pair_level.insert(0, "dataset", dataset)
    pair_level.insert(0, "model", model)
    pair_level.insert(6, "threshold", float(threshold))
    pair_level = pair_level[
        PAIR_LEVEL_COLUMNS
    ].copy()

    statement_level = compute_statement_gamma(
        pair_level,
        model=model,
        dataset=dataset,
        probe=probe,
        threshold=threshold,
    )

    summary = summarize_model_probe(
        pair_level=pair_level,
        statement_level=statement_level,
        model=model,
        dataset=dataset,
        probe=probe,
        threshold=threshold,
    )

    return pair_level, statement_level, summary


def main() -> None:
    args = parse_args()

    if (
        not np.isfinite(args.zero_tolerance)
        or args.zero_tolerance < 0
    ):
        raise ValueError(
            "--zero_tolerance must be finite and nonnegative."
        )

    models = (
        sorted(
            list(dict.fromkeys(args.models)),
            key=model_sort_key,
        )
        if args.models is not None
        else discover_models(args.atomic_dir)
    )
    datasets = list(dict.fromkeys(args.datasets))
    probes = list(dict.fromkeys(args.probes))

    output_paths = {
        "pair_level": args.output_dir / "pair_level.parquet",
        "statement_level": args.output_dir / "statement_level.parquet",
        "model_summary": args.output_dir / "model_summary.parquet",
        "analysis_status": args.output_dir / "analysis_status.parquet",
        "summary_compact": args.output_dir / "summary_compact.csv",
        "analysis_config": args.output_dir / "analysis_config.json",
    }

    for path in output_paths.values():
        ensure_output_available(
            path,
            overwrite=args.overwrite,
        )

    print("=" * 100)
    print("JOINT ORDERING SENSITIVITY ANALYSIS")
    print("=" * 100)
    print(f"Models:             {len(models)}")
    print(f"Datasets:           {datasets}")
    print(f"Probes:             {probes}")
    print(f"x then P Joint:     {args.joint_x_then_P_dir}")
    print(f"P then x Joint:     {args.joint_P_then_x_dir}")
    print(f"Atomic:             {args.atomic_dir}")
    print(f"Pairs:              {args.pairs_dir}")
    print(f"Output:             {args.output_dir}")
    print(f"Zero tolerance:     {args.zero_tolerance:.3e}")
    print(
        "Prob validation:    "
        + (
            "OFF"
            if args.skip_probability_validation
            else "ON"
        )
    )
    print("=" * 100)

    pair_frames: list[pd.DataFrame] = []
    statement_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    status_records: list[dict[str, Any]] = []

    total = len(models) * len(datasets) * len(probes)
    counter = 0

    for model in models:
        for dataset in datasets:
            atomic_path = (
                args.atomic_dir
                / model
                / f"{dataset}.parquet"
            )
            pairs_path = (
                args.pairs_dir
                / model
                / f"{dataset}.parquet"
            )
            joint_xp_path = (
                args.joint_x_then_P_dir
                / model
                / f"{dataset}.parquet"
            )
            joint_px_path = (
                args.joint_P_then_x_dir
                / model
                / f"{dataset}.parquet"
            )

            required_paths = {
                "atomic": atomic_path,
                "pairs": pairs_path,
                "joint_x_then_P": joint_xp_path,
                "joint_P_then_x": joint_px_path,
            }

            for probe in probes:
                counter += 1
                prefix = (
                    f"[{counter:03d}/{total:03d}] "
                    f"{model} / {dataset} / {probe}"
                )

                missing = [
                    name
                    for name, path in required_paths.items()
                    if not path.exists()
                ]

                if missing:
                    reason = (
                        "missing_required_files: "
                        + ", ".join(missing)
                    )
                    log.warning("%s -> %s", prefix, reason)
                    status_records.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "probe": probe,
                            "status": "missing",
                            "reason": reason,
                            "atomic_path": str(atomic_path),
                            "pairs_path": str(pairs_path),
                            "joint_x_then_P_path": str(joint_xp_path),
                            "joint_P_then_x_path": str(joint_px_path),
                            "n_pairs": 0,
                            "n_P": 0,
                        }
                    )
                    continue

                try:
                    pair_level, statement_level, summary = analyze_one(
                        model=model,
                        dataset=dataset,
                        probe=probe,
                        atomic_path=atomic_path,
                        pairs_path=pairs_path,
                        joint_xp_path=joint_xp_path,
                        joint_px_path=joint_px_path,
                        zero_tolerance=args.zero_tolerance,
                        validate_probabilities=(
                            not args.skip_probability_validation
                        ),
                    )
                except DegenerateCase as exc:
                    # Expected for degenerate probe/model combinations with no P
                    # set or no pairs. Record rather than inventing a zero effect.
                    reason = str(exc)
                    log.warning("%s -> skipped: %s", prefix, reason)
                    status_records.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "probe": probe,
                            "status": "skipped",
                            "reason": reason,
                            "atomic_path": str(atomic_path),
                            "pairs_path": str(pairs_path),
                            "joint_x_then_P_path": str(joint_xp_path),
                            "joint_P_then_x_path": str(joint_px_path),
                            "n_pairs": 0,
                            "n_P": 0,
                        }
                    )
                    continue

                pair_frames.append(pair_level)
                statement_frames.append(statement_level)
                summary_records.append(summary)

                status_records.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "probe": probe,
                        "status": "ok",
                        "reason": "",
                        "atomic_path": str(atomic_path),
                        "pairs_path": str(pairs_path),
                        "joint_x_then_P_path": str(joint_xp_path),
                        "joint_P_then_x_path": str(joint_px_path),
                        "n_pairs": int(len(pair_level)),
                        "n_P": int(len(statement_level)),
                    }
                )

                log.info(
                    "%s -> pairs=%s, P=%s, median |Δp|=%.6f, median |Δγ|=%.6f",
                    prefix,
                    f"{len(pair_level):,}",
                    f"{len(statement_level):,}",
                    summary["median_abs_conditional_order_difference"],
                    summary["median_abs_gamma_order_difference"],
                )

    pair_output = (
        pd.concat(
            pair_frames,
            ignore_index=True,
            sort=False,
        )
        if pair_frames
        else pd.DataFrame(columns=PAIR_LEVEL_COLUMNS)
    )
    statement_output = (
        pd.concat(
            statement_frames,
            ignore_index=True,
            sort=False,
        )
        if statement_frames
        else pd.DataFrame(columns=STATEMENT_LEVEL_COLUMNS)
    )
    model_summary = pd.DataFrame(
        summary_records,
        columns=MODEL_SUMMARY_COLUMNS,
    )
    analysis_status = pd.DataFrame(
        status_records,
        columns=STATUS_COLUMNS,
    )

    # Stable ordering for reproducible inspection and plotting.
    if not pair_output.empty:
        pair_output = pair_output.sort_values(
            ["model", "dataset", "probe", "P_id", "x_id"],
            kind="stable",
        ).reset_index(drop=True)

    if not statement_output.empty:
        statement_output = statement_output.sort_values(
            ["model", "dataset", "probe", "P_id"],
            kind="stable",
        ).reset_index(drop=True)

    if not model_summary.empty:
        model_summary = model_summary.sort_values(
            ["model", "dataset", "probe"],
            kind="stable",
        ).reset_index(drop=True)

    if not analysis_status.empty:
        analysis_status = analysis_status.sort_values(
            ["model", "dataset", "probe"],
            kind="stable",
        ).reset_index(drop=True)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_output.to_parquet(
        output_paths["pair_level"],
        index=False,
    )
    statement_output.to_parquet(
        output_paths["statement_level"],
        index=False,
    )
    model_summary.to_parquet(
        output_paths["model_summary"],
        index=False,
    )
    analysis_status.to_parquet(
        output_paths["analysis_status"],
        index=False,
    )
    model_summary.to_csv(
        output_paths["summary_compact"],
        index=False,
    )

    config = {
        "schema_version": 1,
        "models": models,
        "datasets": datasets,
        "probes": probes,
        "zero_tolerance": float(args.zero_tolerance),
        "probability_validation": (
            not args.skip_probability_validation
        ),
        "inputs": {
            "atomic_dir": str(args.atomic_dir),
            "pairs_dir": str(args.pairs_dir),
            "joint_x_then_P_dir": str(args.joint_x_then_P_dir),
            "joint_P_then_x_dir": str(args.joint_P_then_x_dir),
        },
        "outputs": {
            key: str(path)
            for key, path in output_paths.items()
        },
        "definitions": {
            "conditional_x_then_P": (
                "(prob_TT + prob_NT) / "
                "(prob_TT + prob_TF + prob_NT + prob_NF)"
            ),
            "conditional_P_then_x": (
                "(prob_TT + prob_TN) / "
                "(prob_TT + prob_FT + prob_TN + prob_FN)"
            ),
            "pair_order_sensitivity": (
                "abs(conditional_x_then_P - conditional_P_then_x)"
            ),
            "gamma": (
                "count(defined conditional > atomic threshold) / "
                "count(defined conditional)"
            ),
            "gamma_order_sensitivity": (
                "abs(gamma_x_then_P - gamma_P_then_x)"
            ),
        },
    }
    output_paths["analysis_config"].write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    ok = int(
        analysis_status["status"].eq("ok").sum()
    ) if not analysis_status.empty else 0
    skipped = int(
        analysis_status["status"].eq("skipped").sum()
    ) if not analysis_status.empty else 0
    missing = int(
        analysis_status["status"].eq("missing").sum()
    ) if not analysis_status.empty else 0

    print()
    print("=" * 100)
    print("JOINT ORDERING ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Successful model/dataset/probe cells: {ok}")
    print(f"Skipped / degenerate cells:           {skipped}")
    print(f"Missing-input cells:                  {missing}")
    print(f"Pair-level rows:                      {len(pair_output):,}")
    print(f"Statement-level rows:                 {len(statement_output):,}")
    print(f"Model-summary rows:                   {len(model_summary):,}")
    print()
    for name, path in output_paths.items():
        print(f"{name:18s} {path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
