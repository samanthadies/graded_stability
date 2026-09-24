"""
Constructs balanced training and calibration data for the ordered nine-class joint probe,
representing every combination of trivalent states for x and P.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stability.beliefs.rendering import render_pair_statement
from stability.data.loading import ProbeDataset


ATOMIC_LABELS = ("false", "true", "neither")
JOINT_LABELS = (
    "TT",
    "TF",
    "TN",
    "FT",
    "FF",
    "FN",
    "NT",
    "NF",
    "NN",
)
JOINT_LABEL_TO_ID = {
    label: index
    for index, label in enumerate(JOINT_LABELS)
}
JOINT_ID_TO_LABEL = {
    index: label
    for label, index in JOINT_LABEL_TO_ID.items()
}

_ATOMIC_TO_SYMBOL = {
    "true": "T",
    "false": "F",
    "neither": "N",
}
_SYMBOL_TO_ATOMIC = {
    value: key
    for key, value in _ATOMIC_TO_SYMBOL.items()
}

DEFAULT_JOINT_TOTALS = {
    "cities_loc": {
        "train": 4000,
        "cal": 1400,
    },
    "med_indications": {
        "train": 3850,
        "cal": 1325,
    },
    "defs": {
        "train": 4725,
        "cal": 1625,
    },
}


@dataclass(frozen=True)
class JointTrainingData:
    statements: tuple[str, ...]
    labels: np.ndarray
    train_mask: np.ndarray
    cal_mask: np.ndarray
    joint_types: tuple[str, ...]
    x_ids: np.ndarray
    P_ids: np.ndarray
    template: str

    @property
    def n_rows(self) -> int:
        return len(self.statements)

    def summary(self) -> dict[str, Any]:
        train_labels = self.labels[self.train_mask]
        cal_labels = self.labels[self.cal_mask]

        return {
            "num_rows": int(self.n_rows),
            "train_rows": int(self.train_mask.sum()),
            "cal_rows": int(self.cal_mask.sum()),
            "template": self.template,
            "ordered": True,
            "first_operand": "x",
            "second_operand": "P",
            "joint_classes": list(JOINT_LABELS),
            "train_label_counts": {
                JOINT_ID_TO_LABEL[int(class_id)]: int(
                    (train_labels == class_id).sum()
                )
                for class_id in np.unique(train_labels)
            },
            "cal_label_counts": {
                JOINT_ID_TO_LABEL[int(class_id)]: int(
                    (cal_labels == class_id).sum()
                )
                for class_id in np.unique(cal_labels)
            },
            "joint_type_counts": {
                label: int(
                    sum(
                        value == label
                        for value in self.joint_types
                    )
                )
                for label in JOINT_LABELS
            },
        }


def label_joint(
    x_label: str,
    P_label: str,
) -> str:
    """Return the ordered joint label with x first and P second."""
    x_label = str(x_label).strip().lower()
    P_label = str(P_label).strip().lower()

    if x_label not in _ATOMIC_TO_SYMBOL:
        raise ValueError(
            f"Invalid x label {x_label!r}; expected {ATOMIC_LABELS}."
        )
    if P_label not in _ATOMIC_TO_SYMBOL:
        raise ValueError(
            f"Invalid P label {P_label!r}; expected {ATOMIC_LABELS}."
        )

    return (
        _ATOMIC_TO_SYMBOL[x_label]
        + _ATOMIC_TO_SYMBOL[P_label]
    )


def _distribute_total(
    total: int,
) -> dict[str, int]:
    """Distribute a split total across the nine joint classes deterministically."""
    if total <= 0:
        raise ValueError(
            "Joint split totals must be positive."
        )

    base, remainder = divmod(
        int(total),
        len(JOINT_LABELS),
    )

    return {
        label: (
            base
            + int(index < remainder)
        )
        for index, label in enumerate(
            JOINT_LABELS
        )
    }


def _sample_directed_pairs(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    exclude_self: bool = True,
) -> list[tuple[int, int]]:
    """
    Sample unique directed pairs without constructing the Cartesian product.

    The pair is ordered: (x_id, P_id).
    """
    left_ids = np.asarray(
        left_ids,
        dtype=np.int64,
    )
    right_ids = np.asarray(
        right_ids,
        dtype=np.int64,
    )

    if n < 0:
        raise ValueError(
            "n must be non-negative."
        )
    if n == 0:
        return []
    if len(left_ids) == 0 or len(right_ids) == 0:
        raise ValueError(
            "Cannot sample joint pairs from an empty label pool."
        )

    max_possible = (
        len(left_ids)
        * len(right_ids)
    )

    if exclude_self:
        shared = np.intersect1d(
            left_ids,
            right_ids,
            assume_unique=False,
        )
        max_possible -= len(shared)

    if n > max_possible:
        raise ValueError(
            f"Requested {n} unique directed pairs but only "
            f"{max_possible} are possible."
        )

    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = max(
        10_000,
        n * 500,
    )

    for _ in range(max_attempts):
        left_id = int(
            left_ids[
                int(
                    rng.integers(
                        len(left_ids)
                    )
                )
            ]
        )
        right_id = int(
            right_ids[
                int(
                    rng.integers(
                        len(right_ids)
                    )
                )
            ]
        )

        if (
            exclude_self
            and left_id == right_id
        ):
            continue

        key = (
            left_id,
            right_id,
        )
        if key in seen:
            continue

        seen.add(key)
        selected.append(key)

        if len(selected) == n:
            return selected

    # Deterministic fallback for dense requests.
    for left_id in left_ids:
        for right_id in right_ids:
            left_id = int(left_id)
            right_id = int(right_id)

            if (
                exclude_self
                and left_id == right_id
            ):
                continue

            key = (
                left_id,
                right_id,
            )
            if key in seen:
                continue

            seen.add(key)
            selected.append(key)

            if len(selected) == n:
                return selected

    raise RuntimeError(
        f"Only sampled {len(selected)} of {n} requested joint pairs."
    )


def build_joint_training_data(
    data: ProbeDataset,
    *,
    train_total: int | None = None,
    cal_total: int | None = None,
    seed: int = 0,
    template: str = "x_then_P",
    statement_config_path: str = "configs/statements.yaml",
) -> JointTrainingData:
    """
    Build balanced nine-class joint train/cal data entirely in memory.

    The rendered order is always semantic (x, P); the selected template decides
    the surface form. With the default x_then_P this is "x and P."
    """
    defaults = DEFAULT_JOINT_TOTALS.get(
        data.dataset
    )

    if train_total is None:
        if defaults is None:
            raise ValueError(
                "Pass train_total for datasets without a configured default."
            )
        train_total = int(
            defaults["train"]
        )

    if cal_total is None:
        if defaults is None:
            raise ValueError(
                "Pass cal_total for datasets without a configured default."
            )
        cal_total = int(
            defaults["cal"]
        )

    split_counts = {
        "train": _distribute_total(
            int(train_total)
        ),
        "cal": _distribute_total(
            int(cal_total)
        ),
    }

    rng = np.random.default_rng(
        seed
    )

    statements: list[str] = []
    labels: list[int] = []
    split_names: list[str] = []
    joint_types: list[str] = []
    x_ids: list[int] = []
    P_ids: list[int] = []

    id_to_label = (
        data.label_schema.id_to_label
    )

    for split_name, split_mask in (
        (
            "train",
            data.train_mask,
        ),
        (
            "cal",
            data.cal_mask,
        ),
    ):
        split_ids = np.flatnonzero(
            split_mask
        ).astype(
            np.int64
        )
        split_label_names = np.asarray(
            [
                id_to_label[
                    int(
                        data.labels[
                            row_id
                        ]
                    )
                ]
                for row_id in split_ids
            ],
            dtype=object,
        )

        for joint_label in JOINT_LABELS:
            x_label = _SYMBOL_TO_ATOMIC[
                joint_label[0]
            ]
            P_label = _SYMBOL_TO_ATOMIC[
                joint_label[1]
            ]

            x_pool = split_ids[
                split_label_names
                == x_label
            ]
            P_pool = split_ids[
                split_label_names
                == P_label
            ]

            pairs = _sample_directed_pairs(
                x_pool,
                P_pool,
                n=split_counts[
                    split_name
                ][
                    joint_label
                ],
                rng=rng,
                exclude_self=True,
            )

            for x_id, P_id in pairs:
                expected = label_joint(
                    x_label,
                    P_label,
                )
                if expected != joint_label:
                    raise RuntimeError(
                        "Joint-label construction mismatch: "
                        f"expected {joint_label}, got {expected}."
                    )

                statement = render_pair_statement(
                    kind="joint",
                    x_statement=str(
                        data.statements[
                            x_id
                        ]
                    ),
                    P_statement=str(
                        data.statements[
                            P_id
                        ]
                    ),
                    variant=template,
                    config_path=statement_config_path,
                )

                statements.append(
                    statement
                )
                labels.append(
                    JOINT_LABEL_TO_ID[
                        joint_label
                    ]
                )
                split_names.append(
                    split_name
                )
                joint_types.append(
                    joint_label
                )
                x_ids.append(
                    x_id
                )
                P_ids.append(
                    P_id
                )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )
    split_array = np.asarray(
        split_names,
        dtype=object,
    )

    train_mask = (
        split_array
        == "train"
    )
    cal_mask = (
        split_array
        == "cal"
    )

    expected_classes = np.arange(
        len(JOINT_LABELS),
        dtype=np.int64,
    )

    train_classes = np.unique(
        labels_array[
            train_mask
        ]
    )
    cal_classes = np.unique(
        labels_array[
            cal_mask
        ]
    )

    if not np.array_equal(
        train_classes,
        expected_classes,
    ):
        raise RuntimeError(
            f"Joint train split has classes {train_classes.tolist()}, "
            f"expected {expected_classes.tolist()}."
        )

    if not np.array_equal(
        cal_classes,
        expected_classes,
    ):
        raise RuntimeError(
            f"Joint calibration split has classes {cal_classes.tolist()}, "
            f"expected {expected_classes.tolist()}."
        )

    return JointTrainingData(
        statements=tuple(
            statements
        ),
        labels=labels_array,
        train_mask=train_mask,
        cal_mask=cal_mask,
        joint_types=tuple(
            joint_types
        ),
        x_ids=np.asarray(
            x_ids,
            dtype=np.int64,
        ),
        P_ids=np.asarray(
            P_ids,
            dtype=np.int64,
        ),
        template=template,
    )
