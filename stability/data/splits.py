"""
Validates benchmark split annotations and constructs aligned train, calibration, and
test masks across the source files comprising each dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


SPLIT_COLUMNS = ("in_train", "in_cal", "in_test")


def _coerce_split_column(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
) -> np.ndarray:
    """Convert a split column to bool while rejecting ambiguous values."""
    if series.isna().any():
        bad = series.index[series.isna()].tolist()[:10]
        raise ValueError(
            f"{source_name}: split column {column!r} has missing values "
            f"at rows {bad}."
        )

    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        bad_values = sorted(
            series[numeric.isna()].astype(str).unique().tolist()
        )
        raise ValueError(
            f"{source_name}: split column {column!r} must contain "
            f"boolean/0/1 values; found {bad_values}."
        )

    valid = numeric.isin([0, 1])
    if not valid.all():
        bad_values = sorted(numeric[~valid].unique().tolist())
        raise ValueError(
            f"{source_name}: split column {column!r} must contain "
            f"only 0/1; found {bad_values}."
        )

    return numeric.to_numpy(dtype=np.int64).astype(bool)


def split_masks_from_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and validate train/cal/test masks from one dataframe."""
    missing = [
        column
        for column in SPLIT_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{source_name}: missing split columns {missing}."
        )

    train = _coerce_split_column(
        frame["in_train"],
        source_name=source_name,
        column="in_train",
    )
    cal = _coerce_split_column(
        frame["in_cal"],
        source_name=source_name,
        column="in_cal",
    )
    test = _coerce_split_column(
        frame["in_test"],
        source_name=source_name,
        column="in_test",
    )

    membership_count = (
        train.astype(np.int8)
        + cal.astype(np.int8)
        + test.astype(np.int8)
    )

    if np.any(membership_count > 1):
        bad = np.flatnonzero(membership_count > 1)[:10].tolist()
        raise ValueError(
            f"{source_name}: split assignments overlap at rows {bad}."
        )

    if np.any(membership_count == 0):
        bad = np.flatnonzero(membership_count == 0)[:10].tolist()
        raise ValueError(
            f"{source_name}: rows are missing a split assignment at {bad}."
        )

    return train, cal, test


def concatenate_split_masks(
    frames: Sequence[pd.DataFrame],
    *,
    source_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate validated split masks in source-data order."""
    if len(frames) != len(source_names):
        raise ValueError(
            "frames and source_names must have the same length."
        )
    if not frames:
        raise ValueError("No source frames were provided.")

    train_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for frame, source_name in zip(frames, source_names):
        train, cal, test = split_masks_from_frame(
            frame,
            source_name=source_name,
        )
        train_parts.append(train)
        cal_parts.append(cal)
        test_parts.append(test)

    train_mask = np.concatenate(train_parts).astype(bool, copy=False)
    cal_mask = np.concatenate(cal_parts).astype(bool, copy=False)
    test_mask = np.concatenate(test_parts).astype(bool, copy=False)

    if not train_mask.any():
        raise ValueError("Combined training split is empty.")
    if not cal_mask.any():
        raise ValueError("Combined calibration split is empty.")
    if not test_mask.any():
        raise ValueError("Combined test split is empty.")

    return train_mask, cal_mask, test_mask
