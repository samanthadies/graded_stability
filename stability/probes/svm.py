"""
Implements the multiclass linear SVM probe using one-vs-rest classifiers over
last-token representations followed by shared probability calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from stability.activations.pooling import last_valid_token
from stability.probes.base import (
    LayerActivationsLike,
    Probe,
    normalize_indices,
    validate_labels_and_masks,
    validate_layer_activations,
)
from stability.probes.calibration import MulticlassCalibrator


class SVMProbe(Probe):
    """K-class one-vs-all linear SVM with shared probability calibration."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        standardize: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-4,
        class_weight: str | dict | None = None,
        seed: int = 0,
        calibration_C: float = 1.0,
        calibration_max_iter: int = 5_000,
        calibration_tol: float = 1e-6,
    ) -> None:
        self.C = float(C)
        self.standardize = bool(standardize)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.seed = int(seed)

        self.classes_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.models_: list[LinearSVC] = []
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
    ) -> "SVMProbe":
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

        self.models_ = []
        for cls in self.classes_:
            y_binary = (y_train == cls).astype(np.int64)

            model = LinearSVC(
                C=self.C,
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
                random_state=self.seed,
            )
            model.fit(X_train_fit, y_binary)
            self.models_.append(model)

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
        if self.classes_ is None or not self.models_:
            raise RuntimeError("SVMProbe must be fit before prediction.")

        values, _ = validate_layer_activations(activations)
        indices = normalize_indices(indices, n_rows=len(values))

        features = last_valid_token(activations)[indices]
        if self.scaler_ is not None:
            features = self.scaler_.transform(features)

        scores = np.column_stack(
            [model.decision_function(features) for model in self.models_]
        ).astype(np.float64, copy=False)

        expected = (len(indices), len(self.classes_))
        if scores.shape != expected:
            raise RuntimeError(
                f"Unexpected SVM score shape {scores.shape}; "
                f"expected {expected}."
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("SVM scores contain NaN or infinite values.")

        return scores

    def predict_proba(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        scores = self.decision_function(activations, indices)
        return self.calibrator_.predict_proba(scores)
