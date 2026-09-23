from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "model_name",
    "dataset",
    "statement_id",
    "probe",
    "pred_label",
    "prob_false",
    "prob_true",
    "prob_neither",
}

SELECTION_COLUMNS = (
    "score",
    "threshold",
    "is_P",
    "is_x",
    "belief_status",
)


_LABEL_ALIASES = {
    "true": "true",
    "false": "false",
    "neither": "neither",
    "abstain": "neither",
    "suspended": "neither",
    "suspension": "neither",
}


def _normalize_predicted_labels(series: pd.Series) -> pd.Series:
    """Normalize predicted labels to true / false / neither."""
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(_LABEL_ALIASES)

    if mapped.isna().any():
        bad = sorted(
            normalized.loc[mapped.isna()]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Unrecognized predicted labels. Expected true, false, neither, "
            f"abstain, suspended, or suspension; found {bad}."
        )

    return mapped.astype(str)


def _validate_probability_columns(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate atomic probability columns and return float64 arrays."""
    p_false = pd.to_numeric(
        frame["prob_false"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    p_true = pd.to_numeric(
        frame["prob_true"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    p_neither = pd.to_numeric(
        frame["prob_neither"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    probabilities = np.column_stack(
        [p_false, p_true, p_neither]
    )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Atomic probability columns contain non-finite values."
        )

    if (probabilities < -1e-8).any():
        raise ValueError(
            "Atomic probability columns contain negative values. "
            f"Minimum={float(probabilities.min()):.8g}."
        )

    if (probabilities > 1.0 + 1e-8).any():
        raise ValueError(
            "Atomic probability columns contain values above one. "
            f"Maximum={float(probabilities.max()):.8g}."
        )

    sums = probabilities.sum(axis=1)
    if not np.allclose(
        sums,
        1.0,
        rtol=1e-6,
        atol=1e-6,
    ):
        max_error = float(
            np.max(
                np.abs(
                    sums - 1.0
                )
            )
        )
        raise ValueError(
            "Atomic probability rows do not sum to one. "
            f"Maximum absolute error={max_error:.3e}."
        )

    return p_false, p_true, p_neither


def _validate_identity(
    frame: pd.DataFrame,
) -> tuple[str, str]:
    """Require exactly one model and dataset in one atomic result file."""
    model_names = (
        frame["model_name"]
        .dropna()
        .astype(str)
        .unique()
    )
    datasets = (
        frame["dataset"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(model_names) != 1:
        raise ValueError(
            "Atomic file must contain exactly one model_name; "
            f"found {model_names.tolist()}."
        )

    if len(datasets) != 1:
        raise ValueError(
            "Atomic file must contain exactly one dataset; "
            f"found {datasets.tolist()}."
        )

    return str(model_names[0]), str(datasets[0])


def define_sets(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Enrich one long-format atomic score table with P/x set definitions.

    The calculation is performed independently for every probe present in the
    table. statement_id therefore only needs to be unique *within* probe.

    Returns
    -------
    enriched:
        Copy of the input table with score, threshold, is_P, is_x, and
        belief_status columns added.
    summary:
        Compact JSON-serializable metadata keyed by probe.
    """
    if scores.empty:
        raise ValueError("Atomic score table is empty.")

    missing = REQUIRED_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(
            f"Atomic score table is missing required columns "
            f"{sorted(missing)}."
        )

    model_name, dataset = _validate_identity(scores)

    if scores["statement_id"].isna().any():
        raise ValueError(
            "'statement_id' contains missing values."
        )

    enriched = scores.copy()

    # Recompute these columns from scratch so --overwrite can safely refresh
    # an existing enriched atomic Parquet.
    for column in SELECTION_COLUMNS:
        if column in enriched.columns:
            enriched = enriched.drop(
                columns=[column]
            )

    summaries: dict[str, Any] = {}

    output_parts: list[pd.DataFrame] = []

    for probe_name, probe_frame in enriched.groupby(
        "probe",
        sort=True,
        dropna=False,
    ):
        probe_name = str(probe_name)
        probe_frame = probe_frame.copy()

        if probe_frame["statement_id"].duplicated().any():
            examples = (
                probe_frame.loc[
                    probe_frame["statement_id"].duplicated(),
                    "statement_id",
                ]
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"{probe_name}: statement_id must be unique within probe. "
                f"Duplicate examples: {examples}"
            )

        predicted = _normalize_predicted_labels(
            probe_frame["pred_label"]
        )

        _, p_true, _ = _validate_probability_columns(
            probe_frame
        )

        score = p_true.copy()
        is_P = predicted.eq(
            "true"
        ).to_numpy(dtype=bool)
        is_suspension = predicted.eq(
            "neither"
        ).to_numpy(dtype=bool)
        is_disbelief = predicted.eq(
            "false"
        ).to_numpy(dtype=bool)

        is_x = is_P | is_suspension

        if not is_P.any():
            raise ValueError(
                f"{probe_name}: no statements were predicted true, so "
                "the atomic belief threshold cannot be defined."
            )

        threshold = float(
            np.min(
                score[is_P]
            )
        )

        probe_frame["score"] = score
        probe_frame["threshold"] = threshold
        probe_frame["is_P"] = is_P
        probe_frame["is_x"] = is_x
        probe_frame["belief_status"] = predicted.to_numpy(
            dtype=object
        )

        summaries[probe_name] = {
            "threshold": threshold,
            "num_rows": int(
                len(probe_frame)
            ),
            "num_P": int(
                is_P.sum()
            ),
            "num_x": int(
                is_x.sum()
            ),
            "num_suspensions": int(
                is_suspension.sum()
            ),
            "num_disbeliefs": int(
                is_disbelief.sum()
            ),
            "predicted_label_counts": {
                label: int(count)
                for label, count in (
                    predicted.value_counts()
                    .sort_index()
                    .to_dict()
                    .items()
                )
            },
        }

        output_parts.append(
            probe_frame
        )

    enriched = pd.concat(
        output_parts,
        ignore_index=True,
        sort=False,
    )

    enriched = enriched.sort_values(
        ["probe", "statement_id"],
        kind="stable",
    ).reset_index(drop=True)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_name": model_name,
        "dataset": dataset,
        "score_definition": "prob_true",
        "threshold_definition": (
            "minimum prob_true among statements predicted true, "
            "computed separately for each probe"
        ),
        "partition_definition": {
            "P": "pred_label == true",
            "x": "pred_label in {true, neither}",
            "suspension": "pred_label == neither",
            "disbelief": "pred_label == false",
        },
        "num_rows": int(
            len(enriched)
        ),
        "probes": summaries,
    }

    return enriched, summary
