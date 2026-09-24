"""
Loads and combines the benchmark sources into a unified probe dataset with aligned
statements, epistemic labels, source metadata, and train/calibration/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stability.data.labels import (
    LabelSchema,
    build_labels,
    get_label_schema,
)
from stability.data.splits import concatenate_split_masks


@dataclass(frozen=True)
class ProbeDataset:
    """Statements, labels, and split masks aligned in one row order."""

    dataset: str
    construction: str
    statements: tuple[str, ...]
    labels: np.ndarray
    train_mask: np.ndarray
    cal_mask: np.ndarray
    test_mask: np.ndarray
    label_schema: LabelSchema
    source_names: tuple[str, ...]
    source_dataset: np.ndarray
    source_row: np.ndarray

    @property
    def classes(self) -> np.ndarray:
        return self.label_schema.classes

    @property
    def label_names(self) -> tuple[str, ...]:
        return self.label_schema.names

    @property
    def n_rows(self) -> int:
        return len(self.statements)

    def indices(self, split: str) -> np.ndarray:
        """Return integer row indices for train, cal, or test."""
        split = split.strip().lower()

        if split == "train":
            mask = self.train_mask
        elif split in {"cal", "calibration"}:
            mask = self.cal_mask
        elif split == "test":
            mask = self.test_mask
        else:
            raise ValueError(
                "split must be one of: train, cal/calibration, test."
            )

        return np.flatnonzero(mask).astype(np.int64)


def _validate_statement_series(
    series: pd.Series,
    *,
    source_name: str,
    statement_col: str,
) -> tuple[str, ...]:
    """Validate and normalize one source's statement column."""
    if series.isna().any():
        bad = series.index[series.isna()].tolist()[:10]
        raise ValueError(
            f"{source_name}: statement column {statement_col!r} "
            f"contains missing values at rows {bad}."
        )

    statements = tuple(series.astype(str).tolist())

    empty = [
        index
        for index, statement in enumerate(statements)
        if not statement.strip()
    ]
    if empty:
        raise ValueError(
            f"{source_name}: empty statements at rows {empty[:10]}."
        )

    return statements


def _source_specs(
    dataset: str,
    *,
    construction: str,
) -> tuple[tuple[str, str], ...]:
    """
    Resolve source file stems and semantic source kinds.

    For the current atomic benchmark, every dataset is formed by concatenating:
        <dataset>_true_false.csv
        <dataset>_synthetic.csv
    in that order.
    """
    construction = construction.strip().lower()

    if construction == "atomic":
        return (
            (f"{dataset}_true_false", "true_false"),
            (f"{dataset}_synthetic", "synthetic"),
        )

    raise NotImplementedError(
        f"Source resolution for construction={construction!r} "
        "is not implemented yet."
    )


def load_probe_dataset(
    dataset: str,
    *,
    construction: str = "atomic",
    data_dir: str | Path = "data",
    statement_col: str = "statement",
) -> ProbeDataset:
    """
    Load one benchmark into the exact row order used by probe training.

    The current atomic convention concatenates true/false rows first and
    synthetic rows second, matching the legacy layer-sweep pipeline.
    """
    dataset = dataset.strip()
    construction = construction.strip().lower()
    data_dir = Path(data_dir)

    if not dataset:
        raise ValueError("dataset must be non-empty.")

    schema = get_label_schema(construction)
    specs = _source_specs(
        dataset,
        construction=construction,
    )

    frames: list[pd.DataFrame] = []
    source_names: list[str] = []
    statement_parts: list[tuple[str, ...]] = []
    label_parts: list[np.ndarray] = []
    source_dataset_parts: list[np.ndarray] = []
    source_row_parts: list[np.ndarray] = []

    for stem, source_kind in specs:
        path = data_dir / f"{stem}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Probe source data does not exist: {path}"
            )

        frame = pd.read_csv(path)

        if len(frame) == 0:
            raise ValueError(f"{path}: dataframe is empty.")
        if statement_col not in frame.columns:
            raise ValueError(
                f"{path}: missing statement column {statement_col!r}. "
                f"Available columns: {frame.columns.tolist()}"
            )

        statements = _validate_statement_series(
            frame[statement_col],
            source_name=stem,
            statement_col=statement_col,
        )
        labels = build_labels(
            frame,
            construction=construction,
            source_name=stem,
            source_kind=source_kind,
        )

        if len(labels) != len(frame):
            raise RuntimeError(
                f"{stem}: produced {len(labels)} labels for "
                f"{len(frame)} rows."
            )

        frames.append(frame)
        source_names.append(stem)
        statement_parts.append(statements)
        label_parts.append(labels)
        source_dataset_parts.append(
            np.full(len(frame), stem, dtype=object)
        )
        source_row_parts.append(
            np.arange(len(frame), dtype=np.int64)
        )

    train_mask, cal_mask, test_mask = concatenate_split_masks(
        frames,
        source_names=source_names,
    )

    statements = tuple(
        statement
        for part in statement_parts
        for statement in part
    )
    labels = np.concatenate(label_parts).astype(np.int64, copy=False)
    source_dataset = np.concatenate(source_dataset_parts)
    source_row = np.concatenate(source_row_parts).astype(
        np.int64,
        copy=False,
    )

    n_rows = len(statements)

    for name, array in (
        ("labels", labels),
        ("train_mask", train_mask),
        ("cal_mask", cal_mask),
        ("test_mask", test_mask),
        ("source_dataset", source_dataset),
        ("source_row", source_row),
    ):
        if len(array) != n_rows:
            raise RuntimeError(
                f"Combined {name} length {len(array)} does not "
                f"match {n_rows} statements."
            )

    observed_labels = np.unique(labels)
    unexpected = np.setdiff1d(observed_labels, schema.classes)
    if len(unexpected):
        raise ValueError(
            f"Observed unexpected class IDs {unexpected.tolist()}."
        )

    # The training and calibration splits need every class because every current
    # probe trains K class scores and fits a K-class calibrator.
    train_classes = np.unique(labels[train_mask])
    cal_classes = np.unique(labels[cal_mask])

    if not np.array_equal(train_classes, schema.classes):
        raise ValueError(
            "Training split does not contain the complete label schema. "
            f"Expected={schema.classes.tolist()}, "
            f"observed={train_classes.tolist()}."
        )

    if not np.array_equal(cal_classes, schema.classes):
        raise ValueError(
            "Calibration split does not contain the complete label schema. "
            f"Expected={schema.classes.tolist()}, "
            f"observed={cal_classes.tolist()}."
        )

    return ProbeDataset(
        dataset=dataset,
        construction=construction,
        statements=statements,
        labels=labels,
        train_mask=train_mask,
        cal_mask=cal_mask,
        test_mask=test_mask,
        label_schema=schema,
        source_names=tuple(source_names),
        source_dataset=source_dataset,
        source_row=source_row,
    )
