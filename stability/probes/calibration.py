from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning


class MulticlassCalibrator:
    """Map K raw class scores to calibrated K-class probabilities."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 5_000,
        tol: float = 1e-6,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.classes_: np.ndarray | None = None
        self.model_: LogisticRegression | None = None

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        classes: np.ndarray,
    ) -> "MulticlassCalibrator":
        scores = self._validate_scores(scores)
        labels = np.asarray(labels)
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        classes = np.asarray(classes)
        if classes.ndim != 1 or len(classes) < 2:
            raise ValueError(
                f"classes must contain at least two labels; got {classes}."
            )

        if len(scores) != len(labels):
            raise ValueError(
                f"Calibration scores/labels have different lengths: "
                f"{len(scores)} vs {len(labels)}."
            )
        if scores.shape[1] != len(classes):
            raise ValueError(
                f"Calibration scores have {scores.shape[1]} columns; "
                f"expected {len(classes)}."
            )

        observed = np.unique(labels)
        if not np.array_equal(observed, classes):
            raise ValueError(
                "Calibration labels must contain exactly the expected classes. "
                f"Expected={classes.tolist()}, observed={observed.tolist()}."
            )

        self.classes_ = classes.copy()
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            tol=self.tol,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
                module=r"sklearn\.linear_model\._logistic",
            )
            self.model_.fit(scores, labels)

        if set(self.model_.classes_.tolist()) != set(self.classes_.tolist()):
            raise RuntimeError(
                "Calibrator class set does not match the expected class set. "
                f"Expected={self.classes_.tolist()}, "
                f"calibrator={self.model_.classes_.tolist()}."
            )

        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.model_ is None:
            raise RuntimeError("Calibrator must be fit before prediction.")

        scores = self._validate_scores(scores)
        if scores.shape[1] != len(self.classes_):
            raise ValueError(
                f"Scores have {scores.shape[1]} columns; "
                f"expected {len(self.classes_)}."
            )

        raw = np.asarray(self.model_.predict_proba(scores), dtype=np.float64)

        # Align sklearn columns explicitly to self.classes_.
        probabilities = np.zeros(
            (len(raw), len(self.classes_)),
            dtype=np.float64,
        )
        class_to_col = {
            cls: col for col, cls in enumerate(self.classes_.tolist())
        }

        for raw_col, cls in enumerate(self.model_.classes_.tolist()):
            if cls not in class_to_col:
                raise RuntimeError(
                    f"Calibrator returned unexpected class {cls!r}."
                )
            probabilities[:, class_to_col[cls]] = raw[:, raw_col]

        # Numerical guard for downstream cross-entropy / log-loss calculations.
        probabilities = np.clip(probabilities, 1e-15, 1.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)

        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
            raise RuntimeError("Calibrated probabilities have invalid row sums.")

        probabilities = probabilities / row_sums

        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Calibrated probability rows do not sum to one."
            )

        return probabilities

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)

        if scores.ndim != 2:
            raise ValueError(
                f"scores must have shape [N, K]; got {scores.shape}."
            )
        if len(scores) == 0:
            raise ValueError("scores is empty.")
        if not np.isfinite(scores).all():
            raise ValueError("scores contains NaN or infinite values.")

        return scores
