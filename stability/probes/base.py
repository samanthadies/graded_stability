from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LayerActivationsLike(Protocol):
    """Structural type expected by probe implementations."""

    values: np.ndarray
    attention_mask: np.ndarray


class Probe(ABC):
    """Common interface for calibrated multiclass probes."""

    classes_: np.ndarray | None = None

    @abstractmethod
    def fit(
        self,
        activations: LayerActivationsLike,
        labels: np.ndarray,
        train_mask: np.ndarray,
        cal_mask: np.ndarray,
    ) -> "Probe":
        """Fit the probe and its probability calibrator."""
        raise NotImplementedError

    @abstractmethod
    def decision_function(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Return one raw score per example and class, shape [N, K]."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Return calibrated class probabilities, shape [N, K]."""
        raise NotImplementedError

    def predict(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Return class labels in the ordering stored in ``classes_``."""
        if self.classes_ is None:
            raise RuntimeError("Probe must be fit before prediction.")

        probabilities = self.predict_proba(activations, indices)
        return self.classes_[np.argmax(probabilities, axis=1)]


def validate_layer_activations(
    activations: LayerActivationsLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return activation values and attention mask."""
    if not hasattr(activations, "values") or not hasattr(
        activations, "attention_mask"
    ):
        raise TypeError(
            "activations must expose .values and .attention_mask attributes."
        )

    values = np.asarray(activations.values)
    mask = np.asarray(activations.attention_mask)

    if values.ndim != 3:
        raise ValueError(
            f"Activation values must have shape [N, L, D]; got {values.shape}."
        )
    if mask.ndim != 2:
        raise ValueError(
            f"Attention mask must have shape [N, L]; got {mask.shape}."
        )
    if values.shape[:2] != mask.shape:
        raise ValueError(
            "Activation values and attention mask disagree on N/L dimensions: "
            f"{values.shape[:2]} vs {mask.shape}."
        )
    if len(values) == 0:
        raise ValueError("Activation batch is empty.")
    if not np.isfinite(values).all():
        raise ValueError("Activation values contain NaN or infinite values.")

    mask = mask.astype(bool, copy=False)
    if np.any(mask.sum(axis=1) == 0):
        bad = np.flatnonzero(mask.sum(axis=1) == 0)[:10].tolist()
        raise ValueError(
            f"Found sequences with no valid tokens at rows {bad}."
        )

    return values, mask


def validate_labels_and_masks(
    *,
    n_rows: int,
    labels: np.ndarray,
    train_mask: np.ndarray,
    cal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate labels and train/calibration masks.

    Returns
    -------
    labels, train_mask, cal_mask, classes
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    train_mask = np.asarray(train_mask, dtype=bool)
    cal_mask = np.asarray(cal_mask, dtype=bool)

    for name, array in (
        ("labels", labels),
        ("train_mask", train_mask),
        ("cal_mask", cal_mask),
    ):
        if len(array) != n_rows:
            raise ValueError(
                f"{name} has length {len(array)}; expected {n_rows}."
            )

    if np.any(train_mask & cal_mask):
        bad = np.flatnonzero(train_mask & cal_mask)[:10].tolist()
        raise ValueError(f"Training and calibration masks overlap at rows {bad}.")
    if not train_mask.any():
        raise ValueError("Training split is empty.")
    if not cal_mask.any():
        raise ValueError("Calibration split is empty.")

    train_classes = np.unique(labels[train_mask])
    cal_classes = np.unique(labels[cal_mask])

    if len(train_classes) < 2:
        raise ValueError(
            f"Training split must contain at least two classes; "
            f"found {train_classes.tolist()}."
        )

    if not np.array_equal(train_classes, cal_classes):
        raise ValueError(
            "Training and calibration splits must contain exactly the same "
            f"classes. Train={train_classes.tolist()}, "
            f"calibration={cal_classes.tolist()}."
        )

    return labels, train_mask, cal_mask, train_classes


def normalize_indices(
    indices: np.ndarray,
    *,
    n_rows: int,
) -> np.ndarray:
    """Validate and normalize a one-dimensional integer index array."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    if np.any(indices < 0) or np.any(indices >= n_rows):
        bad = indices[(indices < 0) | (indices >= n_rows)][:10].tolist()
        raise IndexError(
            f"Probe indices are out of bounds for {n_rows} rows: {bad}."
        )

    return indices
