"""
Implements the sAwMIL probe, which learns one-vs-rest epistemic-state classifiers
from token-level activations using multiple-instance learning and probability calibration.

@inproceedings{savcisens2025trilemma,
  title={Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi-Rad, Tina},
  booktitle={Mechanistic Interpretability Workshop at Neur{IPS} 2025},
  year={2025},
  note={\url{https://openreview.net/forum?id=z7dLG2ycRf}},
}
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from stability.probes.base import (
    LayerActivationsLike,
    Probe,
    normalize_indices,
    validate_labels_and_masks,
    validate_layer_activations,
)
from stability.probes.calibration import MulticlassCalibrator


@dataclass(frozen=True)
class _InstanceDataset:
    """Token-level training instances for one binary class-vs-rest MIL head."""

    X: np.ndarray
    y: np.ndarray
    bag_id: np.ndarray
    token_position: np.ndarray


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _valid_token_indices(
    mask_row: np.ndarray,
    *,
    max_tokens_per_bag: int | None,
) -> np.ndarray:
    """Return valid token indices, optionally keeping only the final M tokens."""
    valid = np.flatnonzero(mask_row).astype(np.int64)

    if (
        max_tokens_per_bag is not None
        and valid.size > int(max_tokens_per_bag)
    ):
        valid = valid[-int(max_tokens_per_bag):]

    return valid


def _initial_positive_labels(
    n_tokens: int,
    *,
    mode: str,
    k_init: int,
) -> np.ndarray:
    """Initialize token labels for one positive bag."""
    n_tokens = int(n_tokens)

    if n_tokens <= 0:
        return np.zeros(0, dtype=np.int32)

    mode = str(mode).strip().lower()

    if mode == "last_k":
        labels = np.zeros(n_tokens, dtype=np.int32)
        k = min(max(int(k_init), 0), n_tokens)
        if k > 0:
            labels[n_tokens - k:] = 1
        return labels

    if mode == "all_pos":
        return np.ones(n_tokens, dtype=np.int32)

    raise ValueError(
        f"Unknown sAwMIL init_mode={mode!r}; expected 'last_k' or 'all_pos'."
    )


def _select_top_k_indices(
    scores: np.ndarray,
    *,
    k_top: int,
) -> np.ndarray:
    """Return indices of the top-k scores in descending score order."""
    scores = np.asarray(scores).reshape(-1)
    n = len(scores)

    if n == 0:
        return np.zeros(0, dtype=np.int64)

    k = min(max(int(k_top), 0), n)
    if k == 0:
        return np.zeros(0, dtype=np.int64)

    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top.astype(np.int64)


def _fit_token_scaler(
    *,
    values: np.ndarray,
    attention_mask: np.ndarray,
    train_mask: np.ndarray,
    standardize: bool,
    max_tokens_per_bag: int | None,
) -> StandardScaler | None:
    """
    Fit the token-level StandardScaler on all valid training tokens.

    This intentionally mirrors the legacy implementation: the scaler is fit
    across the pooled token instances from *all* training bags, independent of
    class.
    """
    if not standardize:
        return None

    pooled: list[np.ndarray] = []

    for bag_index in np.flatnonzero(train_mask).astype(np.int64):
        valid = _valid_token_indices(
            attention_mask[int(bag_index)],
            max_tokens_per_bag=max_tokens_per_bag,
        )
        if valid.size == 0:
            continue

        pooled.append(
            np.asarray(
                values[int(bag_index), valid],
                dtype=np.float32,
                order="C",
            )
        )

    if not pooled:
        raise RuntimeError(
            "No valid training tokens were available to fit the sAwMIL scaler."
        )

    X_pool = np.concatenate(pooled, axis=0).astype(
        np.float32,
        copy=False,
    )

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )
    scaler.fit(X_pool)
    return scaler


def _transform(
    X: np.ndarray,
    scaler: StandardScaler | None,
) -> np.ndarray:
    """Apply the fitted feature standardization, preserving float32 output."""
    X = np.asarray(X, dtype=np.float32, order="C")

    if scaler is not None:
        X = scaler.transform(X).astype(
            np.float32,
            copy=False,
        )

    return X


def _build_binary_instance_dataset(
    *,
    values: np.ndarray,
    attention_mask: np.ndarray,
    binary_bag_labels: np.ndarray,
    train_mask: np.ndarray,
    init_mode: str,
    k_init: int,
    max_tokens_per_bag: int | None,
) -> _InstanceDataset:
    """
    Build initial token-level labels for one binary class-vs-rest MIL problem.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    bag_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []

    for bag_index in np.flatnonzero(train_mask).astype(np.int64):
        bag_label = int(binary_bag_labels[int(bag_index)])

        if bag_label not in (0, 1):
            raise ValueError(
                "Binary sAwMIL training expects bag labels in {0,1}; "
                f"got {bag_label} for bag {bag_index}."
            )

        valid = _valid_token_indices(
            attention_mask[int(bag_index)],
            max_tokens_per_bag=max_tokens_per_bag,
        )
        if valid.size == 0:
            continue

        X_tokens = np.asarray(
            values[int(bag_index), valid],
            dtype=np.float32,
            order="C",
        )
        n_tokens = len(X_tokens)

        if bag_label == 1:
            y_tokens = _initial_positive_labels(
                n_tokens,
                mode=init_mode,
                k_init=k_init,
            )
        else:
            y_tokens = np.zeros(
                n_tokens,
                dtype=np.int32,
            )

        X_parts.append(X_tokens)
        y_parts.append(y_tokens)
        bag_parts.append(
            np.full(
                n_tokens,
                int(bag_index),
                dtype=np.int64,
            )
        )
        position_parts.append(
            np.arange(
                n_tokens,
                dtype=np.int32,
            )
        )

    if not X_parts:
        raise RuntimeError(
            "No token instances were available for sAwMIL training."
        )

    return _InstanceDataset(
        X=np.concatenate(X_parts, axis=0).astype(
            np.float32,
            copy=False,
        ),
        y=np.concatenate(y_parts, axis=0).astype(
            np.int32,
            copy=False,
        ),
        bag_id=np.concatenate(bag_parts, axis=0).astype(
            np.int64,
            copy=False,
        ),
        token_position=np.concatenate(
            position_parts,
            axis=0,
        ).astype(
            np.int32,
            copy=False,
        ),
    )


def _relabel_positive_instances(
    *,
    classifier: SGDClassifier,
    X: np.ndarray,
    bag_id: np.ndarray,
    token_position: np.ndarray,
    binary_bag_labels: np.ndarray,
    k_top: int,
    tail_k: int,
) -> np.ndarray:
    """
    Relabel token instances inside positive bags.

    The implementation deliberately preserves the legacy procedure:
    top-k instances are selected by score first, then any selected instances
    outside the optional final ``tail_k`` positions are discarded.
    """
    scores = np.asarray(
        classifier.decision_function(X),
        dtype=np.float32,
    ).reshape(-1)

    relabeled = np.zeros(
        len(X),
        dtype=np.int32,
    )

    order = np.argsort(bag_id)
    bag_sorted = bag_id[order]
    score_sorted = scores[order]
    position_sorted = token_position[order]

    start = 0

    while start < len(order):
        bag = int(bag_sorted[start])

        end = start + 1
        while (
            end < len(order)
            and int(bag_sorted[end]) == bag
        ):
            end += 1

        if int(binary_bag_labels[bag]) == 1:
            bag_scores = score_sorted[start:end]
            bag_positions = position_sorted[start:end]
            n_tokens = end - start

            selected = _select_top_k_indices(
                bag_scores,
                k_top=k_top,
            )

            if int(tail_k) > 0:
                tail_start = max(
                    0,
                    n_tokens - int(tail_k),
                )
                selected = np.asarray(
                    [
                        relative_index
                        for relative_index in selected
                        if int(
                            bag_positions[relative_index]
                        ) >= tail_start
                    ],
                    dtype=np.int64,
                )

            if selected.size:
                global_indices = order[start:end][selected]
                relabeled[global_indices] = 1

        start = end

    return relabeled


class SAWMILProbe(Probe):
    """
    Generic K-class sAwMIL probe.

    The fitted class ordering is stored in ``classes_`` and follows the sorted
    unique labels present in the training/calibration splits.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        epochs: int = 5,
        standardize: bool = True,
        init_mode: str = "last_k",
        k_init: int = 2,
        k_top: int = 2,
        tail_k: int = 2,
        max_tokens_per_bag: int | None = 256,
        seed: int = 0,
        calibration_C: float = 1.0,
        calibration_max_iter: int = 2_000,
        calibration_tol: float = 1e-6,
    ) -> None:
        if max_tokens_per_bag is not None and max_tokens_per_bag <= 0:
            raise ValueError(
                "max_tokens_per_bag must be positive or None."
            )

        self.C = float(C)
        self.epochs = int(epochs)
        self.standardize = bool(standardize)

        self.init_mode = str(init_mode)
        self.k_init = int(k_init)
        self.k_top = int(k_top)
        self.tail_k = int(tail_k)
        self.max_tokens_per_bag = (
            None
            if max_tokens_per_bag is None
            else int(max_tokens_per_bag)
        )
        self.seed = int(seed)

        self.classes_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.models_: list[SGDClassifier] = []
        self.calibrator_ = MulticlassCalibrator(
            C=calibration_C,
            max_iter=calibration_max_iter,
            tol=calibration_tol,
        )

    def _make_classifier(
        self,
        *,
        n_instances: int,
    ) -> SGDClassifier:
        """
        Construct the same averaged hinge-loss SGD classifier as the legacy
        fast-sAwMIL implementation.
        """
        alpha = 1.0 / (
            max(self.C, 1e-12)
            * max(int(n_instances), 1)
        )

        return SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            fit_intercept=True,
            max_iter=self.epochs,
            tol=None,
            shuffle=True,
            random_state=self.seed,
            average=True,
            learning_rate="optimal",
            n_jobs=-1,
        )

    def _fit_binary_head(
        self,
        *,
        values: np.ndarray,
        attention_mask: np.ndarray,
        binary_bag_labels: np.ndarray,
        train_mask: np.ndarray,
    ) -> SGDClassifier:
        """Fit one class-vs-rest sAwMIL head."""
        instances = _build_binary_instance_dataset(
            values=values,
            attention_mask=attention_mask,
            binary_bag_labels=binary_bag_labels,
            train_mask=train_mask,
            init_mode=self.init_mode,
            k_init=self.k_init,
            max_tokens_per_bag=self.max_tokens_per_bag,
        )

        if len(np.unique(instances.y)) < 2:
            raise RuntimeError(
                "Initial sAwMIL instance labels contain only one class."
            )

        X = _transform(
            instances.X,
            self.scaler_,
        )

        initial = self._make_classifier(
            n_instances=len(X),
        )
        initial.fit(
            X,
            instances.y,
        )

        relabeled = _relabel_positive_instances(
            classifier=initial,
            X=X,
            bag_id=instances.bag_id,
            token_position=instances.token_position,
            binary_bag_labels=binary_bag_labels,
            k_top=self.k_top,
            tail_k=self.tail_k,
        )

        if len(np.unique(relabeled)) < 2:
            raise RuntimeError(
                "Relabeled sAwMIL instances contain only one class."
            )

        final = self._make_classifier(
            n_instances=len(X),
        )
        final.fit(
            X,
            relabeled,
        )

        return final

    def fit(
        self,
        activations: LayerActivationsLike,
        labels: np.ndarray,
        train_mask: np.ndarray,
        cal_mask: np.ndarray,
    ) -> "SAWMILProbe":
        _set_seed(self.seed)

        values, attention_mask = validate_layer_activations(
            activations
        )

        labels, train_mask, cal_mask, classes = (
            validate_labels_and_masks(
                n_rows=len(values),
                labels=labels,
                train_mask=train_mask,
                cal_mask=cal_mask,
            )
        )

        self.classes_ = classes

        self.scaler_ = _fit_token_scaler(
            values=values,
            attention_mask=attention_mask,
            train_mask=train_mask,
            standardize=self.standardize,
            max_tokens_per_bag=self.max_tokens_per_bag,
        )

        self.models_ = []

        for cls in self.classes_:
            binary_bag_labels = (
                labels == cls
            ).astype(np.int32)

            self.models_.append(
                self._fit_binary_head(
                    values=values,
                    attention_mask=attention_mask,
                    binary_bag_labels=binary_bag_labels,
                    train_mask=train_mask,
                )
            )

        cal_indices = np.flatnonzero(
            cal_mask
        ).astype(np.int64)

        cal_scores = self.decision_function(
            activations,
            cal_indices,
        )

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
            raise RuntimeError(
                "SAWMILProbe must be fit before prediction."
            )

        values, attention_mask = validate_layer_activations(
            activations
        )
        indices = normalize_indices(
            indices,
            n_rows=len(values),
        )

        margins = np.empty(
            (len(indices), len(self.classes_)),
            dtype=np.float64,
        )

        for output_row, bag_index in enumerate(indices):
            valid = _valid_token_indices(
                attention_mask[int(bag_index)],
                max_tokens_per_bag=self.max_tokens_per_bag,
            )

            # validate_layer_activations already rejects all-padding rows.
            if valid.size == 0:
                raise RuntimeError(
                    f"No valid tokens for bag {bag_index}."
                )

            X_tokens = _transform(
                values[int(bag_index), valid],
                self.scaler_,
            )

            for class_index, classifier in enumerate(
                self.models_
            ):
                token_scores = np.asarray(
                    classifier.decision_function(
                        X_tokens
                    ),
                    dtype=np.float64,
                ).reshape(-1)

                if token_scores.size == 0:
                    raise RuntimeError(
                        f"No token scores for bag {bag_index}, "
                        f"class column {class_index}."
                    )

                margins[
                    output_row,
                    class_index,
                ] = float(token_scores.max())

        if not np.isfinite(margins).all():
            raise RuntimeError(
                "sAwMIL margins contain NaN or infinite values."
            )

        return margins

    def predict_proba(
        self,
        activations: LayerActivationsLike,
        indices: np.ndarray,
    ) -> np.ndarray:
        margins = self.decision_function(
            activations,
            indices,
        )
        return self.calibrator_.predict_proba(
            margins
        )
