from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from stability.probes.base import LayerActivationsLike, Probe


def evaluate_probe(
    *,
    probe: Probe,
    activations: LayerActivationsLike,
    labels: np.ndarray,
    split_mask: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate one fitted probe on one split."""
    labels = np.asarray(labels)
    split_mask = np.asarray(split_mask, dtype=bool)

    if len(labels) != len(split_mask):
        raise ValueError(
            f"labels and split_mask differ in length: "
            f"{len(labels)} vs {len(split_mask)}."
        )

    indices = np.flatnonzero(split_mask).astype(np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot evaluate an empty split.")

    if probe.classes_ is None:
        raise RuntimeError("Probe must be fit before evaluation.")

    classes = np.asarray(probe.classes_)
    y_true = labels[indices]
    probabilities = probe.predict_proba(activations, indices)
    predictions = probe.predict(activations, indices)

    expected_shape = (len(indices), len(classes))
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected probability shape {probabilities.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Probe probabilities contain NaN or infinite values.")
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("Probe probability rows do not sum to one.")

    return {
        "n": int(len(indices)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=classes,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=classes,
            )
        ),
    }


def evaluate_all_splits(
    *,
    probe: Probe,
    activations: LayerActivationsLike,
    labels: np.ndarray,
    train_mask: np.ndarray,
    cal_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate train/cal/test and flatten metrics into one row dictionary."""
    output: dict[str, float | int] = {}

    for split_name, split_mask in (
        ("train", train_mask),
        ("cal", cal_mask),
        ("test", test_mask),
    ):
        metrics = evaluate_probe(
            probe=probe,
            activations=activations,
            labels=labels,
            split_mask=split_mask,
        )
        for metric_name, value in metrics.items():
            output[f"{split_name}_{metric_name}"] = value

    return output
