"""
Generic multiclass one-vs-all mean-difference probe.

@inproceedings{marks2024geometry,
  title={The Geometry of Truth: {E}mergent Linear Structure in Large Language Model Representations of {T}rue/{F}alse Datasets},
  year={2024},
  author={Marks, Samuel and Tegmark, Max},
  booktitle={Proceedings of the 1st Conference on Language Modeling (COLM 2024)},
  note={\url{https://openreview.net/forum?id=aajyHYjjsk}}
}
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from stability.activations.pooling import last_valid_token
from stability.probes.base import (
    LayerActivationsLike,
    Probe,
    normalize_indices,
    validate_labels_and_masks,
    validate_layer_activations,
)
from stability.probes.calibration import MulticlassCalibrator


class MeanDifferenceProbe(Probe):
    """K-class one-vs-all mean-difference probe with shared calibration."""

    def __init__(
        self,
        *,
        normalize_direction: bool = True,
        standardize: bool = False,
        calibration_C: float = 1.0,
        calibration_max_iter: int = 5_000,
        calibration_tol: float = 1e-6,
    ) -> None:
        self.normalize_direction = bool(normalize_direction)
        self.standardize = bool(standardize)

        self.classes_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.directions_: np.ndarray | None = None
        self.midpoints_: np.ndarray | None = None
        self.calibrator_ = MulticlassCalibrator(
            C=calibration_C,
            max_iter=calibration_max_iter,
            tol=calibration_tol,
        )

    def fit(
        self,
        activations: LayerActivationsLike,
        labels: np.ndarray,
        train_mask: np.ndarray,
        cal_mask: np.ndarray,
    ) -> "MeanDifferenceProbe":
        values, _ = validate_layer_activations(activations)
        labels, train_mask, cal_mask, classes = validate_labels_and_masks(
            n_rows=len(values),
            labels=labels,
            train_mask=train_mask,
            cal_mask=cal_mask,
        )
        self.classes_ = classes

        features = last_valid_token(activations)
        X_train = features[train_mask]
        y_train = labels[train_mask]

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_train_fit = self.scaler_.fit_transform(X_train)
        else:
            self.scaler_ = None
            X_train_fit = X_train

        directions: list[np.ndarray] = []
        midpoints: list[np.ndarray] = []

        for cls in self.classes_:
            positive = y_train == cls
            negative = ~positive

            if not positive.any():
                raise ValueError(
                    f"Class {cls!r} has no positive training examples."
                )
            if not negative.any():
                raise ValueError(
                    f"Class {cls!r} has no negative training examples."
                )

            mu_pos = X_train_fit[positive].mean(axis=0, dtype=np.float64)
            mu_neg = X_train_fit[negative].mean(axis=0, dtype=np.float64)
            direction = mu_pos - mu_neg

            if self.normalize_direction:
                norm = float(np.linalg.norm(direction))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError(
                        f"Mean-difference direction for class {cls!r} "
                        f"has invalid norm {norm}."
                    )
                direction = direction / norm

            midpoint = 0.5 * (mu_pos + mu_neg)
            directions.append(direction)
            midpoints.append(midpoint)

        self.directions_ = np.stack(directions, axis=0)
        self.midpoints_ = np.stack(midpoints, axis=0)

        cal_indices = np.flatnonzero(cal_mask).astype(np.int64)
        cal_scores = self.decision_function(activations, cal_indices)
        self.calibrator_.fit(
            cal_scores,
            labels[cal_indices],
            classes=self.classes_,
        )

        return self

    def decision_function(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        if (
            self.classes_ is None
            or self.directions_ is None
            or self.midpoints_ is None
        ):
            raise RuntimeError(
                "MeanDifferenceProbe must be fit before prediction."
            )

        values, _ = validate_layer_activations(activations)
        indices = normalize_indices(indices, n_rows=len(values))

        features = last_valid_token(activations)[indices]
        if self.scaler_ is not None:
            features = self.scaler_.transform(features)
        features = np.asarray(features, dtype=np.float64, order="C")

        offsets = np.einsum(
            "kd,kd->k",
            self.midpoints_,
            self.directions_,
        )
        scores = features @ self.directions_.T - offsets[None, :]

        expected = (len(indices), len(self.classes_))
        if scores.shape != expected:
            raise RuntimeError(
                f"Unexpected mean-difference score shape {scores.shape}; "
                f"expected {expected}."
            )
        if not np.isfinite(scores).all():
            raise RuntimeError(
                "Mean-difference scores contain NaN or infinite values."
            )

        return scores

    def predict_proba(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        scores = self.decision_function(activations, indices)
        return self.calibrator_.predict_proba(scores)
