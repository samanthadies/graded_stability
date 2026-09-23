from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DIRECT_PROBABILITY_COLUMNS = (
    "prob_false",
    "prob_true",
    "prob_neither",
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

PROBABILITY_SUM_ATOL = 1e-6
PROBABILITY_BOUND_ATOL = 1e-8


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{context} is missing required columns {missing}. "
            f"Available columns: {frame.columns.tolist()}"
        )


def validate_probability_matrix(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> np.ndarray:
    """
    Validate a normalized probability matrix and return float64 values.
    """
    columns = list(columns)
    require_columns(
        frame,
        columns,
        context=context,
    )

    probabilities = (
        frame[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    if probabilities.shape[0] == 0:
        raise ValueError(
            f"{context} contains no rows."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"{context} contains NaN or infinite values."
        )

    invalid_bounds = (
        (
            probabilities
            < -PROBABILITY_BOUND_ATOL
        )
        | (
            probabilities
            > 1.0
            + PROBABILITY_BOUND_ATOL
        )
    )
    if invalid_bounds.any():
        rows = np.flatnonzero(
            invalid_bounds.any(
                axis=1
            )
        )[:10]
        raise ValueError(
            f"{context} contains probabilities outside [0, 1]. "
            f"Example row indices: {rows.tolist()}."
        )

    row_sums = probabilities.sum(
        axis=1
    )
    invalid_sums = ~np.isclose(
        row_sums,
        1.0,
        atol=PROBABILITY_SUM_ATOL,
        rtol=0.0,
    )
    if invalid_sums.any():
        rows = np.flatnonzero(
            invalid_sums
        )[:10]
        examples = {
            int(row): float(
                row_sums[row]
            )
            for row in rows
        }
        raise ValueError(
            f"{context} probability rows do not sum to one within "
            f"{PROBABILITY_SUM_ATOL}. Examples: {examples}."
        )

    return probabilities


def direct_conditional_scores(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return pair_id plus the Phase-2 scalar score.

    Direct Conditional gamma uses the True probability directly.
    """
    require_columns(
        frame,
        (
            "pair_id",
            *DIRECT_PROBABILITY_COLUMNS,
        ),
        context="Direct conditional scores",
    )

    if frame["pair_id"].duplicated().any():
        raise ValueError(
            "Direct conditional scores contain duplicate pair_id values."
        )

    if frame.empty:
        return pd.DataFrame(
            {
                "pair_id": pd.Series(dtype="int64"),
                "conditional_score": pd.Series(dtype="float64"),
                "score_defined": pd.Series(dtype="bool"),
            }
        )

    validate_probability_matrix(
        frame,
        DIRECT_PROBABILITY_COLUMNS,
        context="Direct conditional scores",
    )

    score = pd.to_numeric(
        frame["prob_true"],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    return pd.DataFrame(
        {
            "pair_id": pd.to_numeric(
                frame["pair_id"],
                errors="raise",
            ).to_numpy(
                dtype=np.int64
            ),
            "conditional_score": score,
            "score_defined": np.ones(
                len(frame),
                dtype=bool,
            ),
        }
    )


def joint_conditional_scores(
    frame: pd.DataFrame,
    *,
    zero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """
    Convert Phase-3 nine-class scores into scalar CCK conditionals.

    Joint labels are already canonical (x, P) in the clean pipeline.

        numerator   = prob_TT + prob_NT
        denominator = prob_TT + prob_TF + prob_NT + prob_NF

    The ratio is undefined when denominator <= zero_tolerance.
    """
    if (
        not np.isfinite(
            zero_tolerance
        )
        or zero_tolerance < 0
    ):
        raise ValueError(
            "zero_tolerance must be finite and nonnegative."
        )

    require_columns(
        frame,
        (
            "pair_id",
            *JOINT_PROBABILITY_COLUMNS,
        ),
        context="Joint scores",
    )

    if frame["pair_id"].duplicated().any():
        raise ValueError(
            "Joint scores contain duplicate pair_id values."
        )

    if frame.empty:
        return pd.DataFrame(
            {
                "pair_id": pd.Series(dtype="int64"),
                "conditional_score": pd.Series(dtype="float64"),
                "score_defined": pd.Series(dtype="bool"),
                "joint_numerator": pd.Series(dtype="float64"),
                "joint_denominator": pd.Series(dtype="float64"),
            }
        )

    validate_probability_matrix(
        frame,
        JOINT_PROBABILITY_COLUMNS,
        context="Joint scores",
    )

    prob_TT = pd.to_numeric(
        frame["prob_TT"],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )
    prob_TF = pd.to_numeric(
        frame["prob_TF"],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )
    prob_NT = pd.to_numeric(
        frame["prob_NT"],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )
    prob_NF = pd.to_numeric(
        frame["prob_NF"],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    numerator = (
        prob_TT
        + prob_NT
    )
    denominator = (
        prob_TT
        + prob_TF
        + prob_NT
        + prob_NF
    )

    defined = (
        np.isfinite(
            denominator
        )
        & (
            denominator
            > float(
                zero_tolerance
            )
        )
    )

    score = np.full(
        len(frame),
        np.nan,
        dtype=np.float64,
    )
    score[
        defined
    ] = (
        numerator[
            defined
        ]
        / denominator[
            defined
        ]
    )

    return pd.DataFrame(
        {
            "pair_id": pd.to_numeric(
                frame["pair_id"],
                errors="raise",
            ).to_numpy(
                dtype=np.int64
            ),
            "conditional_score": score,
            "score_defined": defined,
            "joint_numerator": numerator,
            "joint_denominator": denominator,
        }
    )


def attach_pair_metadata(
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    probe_name: str,
) -> pd.DataFrame:
    """
    Join scalar scores to compact pair identities and require exact coverage.

    Full gamma is meaningful only when every pair built for a probe has exactly
    one Phase-2/Phase-3 score.
    """
    require_columns(
        pairs,
        (
            "probe",
            "pair_id",
            "P_id",
            "x_id",
        ),
        context="Pair table",
    )
    require_columns(
        scores,
        (
            "pair_id",
            "conditional_score",
            "score_defined",
        ),
        context="Prepared conditional scores",
    )

    pair_probe_values = set(
        pairs["probe"]
        .astype(str)
        .unique()
    )
    if pair_probe_values != {
        str(
            probe_name
        )
    }:
        raise ValueError(
            f"Pair rows do not correspond exactly to probe "
            f"{probe_name!r}: {sorted(pair_probe_values)}."
        )

    if pairs["pair_id"].duplicated().any():
        raise ValueError(
            f"{probe_name}: pair table contains duplicate pair_id values."
        )

    if scores["pair_id"].duplicated().any():
        raise ValueError(
            f"{probe_name}: score table contains duplicate pair_id values."
        )

    pair_ids = set(
        pd.to_numeric(
            pairs["pair_id"],
            errors="raise",
        )
        .astype(
            np.int64
        )
        .tolist()
    )
    score_ids = set(
        pd.to_numeric(
            scores["pair_id"],
            errors="raise",
        )
        .astype(
            np.int64
        )
        .tolist()
    )

    missing = pair_ids - score_ids
    extra = score_ids - pair_ids

    if missing or extra:
        raise ValueError(
            f"{probe_name}: score coverage does not match the pair table. "
            f"Missing scores={len(missing):,}; extra scores={len(extra):,}."
        )

    merged = pairs[
        [
            "pair_id",
            "P_id",
            "x_id",
        ]
    ].merge(
        scores,
        on="pair_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(
        merged
    ) != len(
        pairs
    ):
        raise RuntimeError(
            f"{probe_name}: pair/score merge changed row count."
        )

    return merged


def compute_gamma(
    pair_scores: pd.DataFrame,
    *,
    threshold: float,
    probe_name: str,
    source: str,
    zero_tolerance: float | None = None,
) -> pd.DataFrame:
    """
    Aggregate pair-level scalar conditionals into gamma(P).

    Undefined conditional scores are excluded from the gamma denominator.
    num_rows records all expected pair rows, while denominator records only
    defined rows.
    """
    if (
        not np.isfinite(
            threshold
        )
    ):
        raise ValueError(
            f"{probe_name}: threshold must be finite."
        )

    require_columns(
        pair_scores,
        (
            "P_id",
            "x_id",
            "conditional_score",
            "score_defined",
        ),
        context="Pair-level conditional scores",
    )

    working = pair_scores.copy()
    working[
        "conditional_score"
    ] = pd.to_numeric(
        working[
            "conditional_score"
        ],
        errors="coerce",
    )

    defined = (
        working[
            "score_defined"
        ].astype(
            bool
        )
        & np.isfinite(
            working[
                "conditional_score"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    working[
        "score_defined"
    ] = defined
    working[
        "above_threshold"
    ] = (
        defined
        & (
            working[
                "conditional_score"
            ]
            > float(
                threshold
            )
        )
    )

    records: list[
        dict[str, object]
    ] = []

    for P_id, group in working.groupby(
        "P_id",
        sort=False,
        dropna=False,
    ):
        group_defined = group[
            "score_defined"
        ].astype(
            bool
        )
        defined_scores = group.loc[
            group_defined,
            "conditional_score",
        ]

        numerator = int(
            group.loc[
                group_defined,
                "above_threshold",
            ].sum()
        )
        denominator = int(
            group_defined.sum()
        )
        num_rows = int(
            len(
                group
            )
        )
        num_undefined = (
            num_rows
            - denominator
        )

        records.append(
            {
                "probe": str(
                    probe_name
                ),
                "P_id": int(
                    P_id
                ),
                "source": str(
                    source
                ),
                "gamma": (
                    float(
                        numerator
                        / denominator
                    )
                    if denominator > 0
                    else np.nan
                ),
                "numerator": numerator,
                "denominator": denominator,
                "num_rows": num_rows,
                "num_undefined": int(
                    num_undefined
                ),
                "threshold": float(
                    threshold
                ),
                "min_score": (
                    float(
                        defined_scores.min()
                    )
                    if denominator > 0
                    else np.nan
                ),
                "max_score": (
                    float(
                        defined_scores.max()
                    )
                    if denominator > 0
                    else np.nan
                ),
                "mean_score": (
                    float(
                        defined_scores.mean()
                    )
                    if denominator > 0
                    else np.nan
                ),
                "zero_tolerance": (
                    float(
                        zero_tolerance
                    )
                    if zero_tolerance is not None
                    else np.nan
                ),
            }
        )

    if not records:
        return pd.DataFrame(
            {
                "probe": pd.Series(dtype="object"),
                "P_id": pd.Series(dtype="int64"),
                "source": pd.Series(dtype="object"),
                "gamma": pd.Series(dtype="float64"),
                "numerator": pd.Series(dtype="int64"),
                "denominator": pd.Series(dtype="int64"),
                "num_rows": pd.Series(dtype="int64"),
                "num_undefined": pd.Series(dtype="int64"),
                "threshold": pd.Series(dtype="float64"),
                "min_score": pd.Series(dtype="float64"),
                "max_score": pd.Series(dtype="float64"),
                "mean_score": pd.Series(dtype="float64"),
                "zero_tolerance": pd.Series(dtype="float64"),
            }
        )

    return pd.DataFrame.from_records(
        records
    )
