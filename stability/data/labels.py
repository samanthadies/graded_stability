"""
Defines the canonical epistemic label schema and maps benchmark annotations or source
types to the integer labels used for atomic probe training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSchema:
    """Ordered semantic labels and their integer IDs."""

    names: tuple[str, ...]

    @property
    def label_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names)}

    @property
    def id_to_label(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.names)}

    @property
    def classes(self) -> np.ndarray:
        return np.arange(len(self.names), dtype=np.int64)


ATOMIC_SCHEMA = LabelSchema(
    names=("false", "true", "neither"),
)

_ATOMIC_ALIASES = {
    "false": "false",
    "true": "true",
    "neither": "neither",
    "abstain": "neither",
    "suspended": "neither",
}


def get_label_schema(construction: str) -> LabelSchema:
    """Return the ordered label schema for a probing construction."""
    construction = construction.strip().lower()

    if construction == "atomic":
        return ATOMIC_SCHEMA

    raise NotImplementedError(
        f"Label schema for construction={construction!r} has not been "
        "implemented in the clean pipeline yet."
    )


def _map_explicit_atomic_labels(
    series: pd.Series,
    *,
    source_name: str,
) -> np.ndarray:
    """Map explicit semantic atomic labels to canonical integer IDs."""
    schema = ATOMIC_SCHEMA
    normalized = series.astype(str).str.strip().str.lower()
    normalized = normalized.map(_ATOMIC_ALIASES)

    if normalized.isna().any():
        raw = series.astype(str).str.strip().str.lower()
        bad = sorted(raw[normalized.isna()].unique().tolist())
        raise ValueError(
            f"{source_name}: unrecognized atomic labels {bad}."
        )

    mapped = normalized.map(schema.label_to_id)
    return mapped.to_numpy(dtype=np.int64)


def build_labels(
    frame: pd.DataFrame,
    *,
    construction: str,
    source_name: str,
    source_kind: str,
) -> np.ndarray:

    construction = construction.strip().lower()
    source_kind = source_kind.strip().lower()

    if construction != "atomic":
        raise NotImplementedError(
            f"Label construction {construction!r} is not implemented yet."
        )

    # Prefer explicit semantic labels when present. This makes the loader
    # future-compatible with regenerated tables that already store canonical
    # labels.
    if "expected_label" in frame.columns:
        return _map_explicit_atomic_labels(
            frame["expected_label"],
            source_name=source_name,
        )

    if "label" in frame.columns:
        return _map_explicit_atomic_labels(
            frame["label"],
            source_name=source_name,
        )

    schema = ATOMIC_SCHEMA

    if source_kind == "synthetic":
        return np.full(
            len(frame),
            schema.label_to_id["neither"],
            dtype=np.int64,
        )

    if source_kind != "true_false":
        raise ValueError(
            f"{source_name}: unsupported atomic source_kind={source_kind!r}."
        )

    if "correct" not in frame.columns:
        raise ValueError(
            f"{source_name}: true/false atomic data requires a 'correct' column."
        )

    correct = pd.to_numeric(frame["correct"], errors="coerce")

    if correct.isna().any():
        bad = frame.index[correct.isna()].tolist()[:10]
        raise ValueError(
            f"{source_name}: non-numeric/missing 'correct' values at rows {bad}."
        )

    valid = correct.isin([0, 1])
    if not valid.all():
        bad_values = sorted(correct[~valid].unique().tolist())
        raise ValueError(
            f"{source_name}: 'correct' must contain only 0/1; "
            f"found {bad_values}."
        )

    false_id = schema.label_to_id["false"]
    true_id = schema.label_to_id["true"]

    return np.where(
        correct.to_numpy(dtype=np.int64) == 1,
        true_id,
        false_id,
    ).astype(np.int64, copy=False)
