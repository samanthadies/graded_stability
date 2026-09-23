from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stability.beliefs.rendering import render_pair_statement
from stability.data.loading import ProbeDataset


LABELS = ("false", "true", "neither")
LABEL_TO_ID = {
    "false": 0,
    "true": 1,
    "neither": 2,
}

COOPER_CANTWELL = {
    ("true", "true"): "true",
    ("true", "false"): "false",
    ("true", "neither"): "neither",
    ("neither", "true"): "true",
    ("neither", "false"): "false",
    ("neither", "neither"): "neither",
    ("false", "true"): "neither",
    ("false", "false"): "neither",
    ("false", "neither"): "neither",
}


@dataclass(frozen=True)
class ConditionalTrainingData:
    statements: tuple[str, ...]
    labels: np.ndarray
    train_mask: np.ndarray
    cal_mask: np.ndarray
    conditional_types: tuple[str, ...]
    x_ids: np.ndarray
    P_ids: np.ndarray
    template: str
    semantics: str

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
            "semantics": self.semantics,
            "train_label_counts": {
                LABELS[int(label)]: int(
                    (train_labels == label).sum()
                )
                for label in np.unique(train_labels)
            },
            "cal_label_counts": {
                LABELS[int(label)]: int(
                    (cal_labels == label).sum()
                )
                for label in np.unique(cal_labels)
            },
            "conditional_type_counts": {
                value: int(
                    sum(
                        item == value
                        for item in self.conditional_types
                    )
                )
                for value in sorted(
                    set(self.conditional_types)
                )
            },
        }


def label_conditional(
    x_label: str,
    P_label: str,
    *,
    semantics: str = "cooper_cantwell",
) -> str:
    if semantics != "cooper_cantwell":
        raise ValueError(
            f"Unsupported conditional semantics {semantics!r}; "
            "the clean Phase 2 pipeline currently implements only "
            "'cooper_cantwell'."
        )

    key = (
        str(x_label).strip().lower(),
        str(P_label).strip().lower(),
    )
    if key not in COOPER_CANTWELL:
        raise ValueError(
            f"Invalid atomic label pair {key}; expected labels {LABELS}."
        )

    return COOPER_CANTWELL[key]


def _sample_directed_pairs(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    exclude_self: bool = True,
) -> list[tuple[int, int]]:
    """
    Sample unique directed pairs without materializing a Cartesian product.

    This preserves the legacy sampling semantics: (a,b) and (b,a) are distinct,
    while an exact duplicate directed pair is not repeated.
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
        raise ValueError("n must be non-negative.")
    if n == 0:
        return []
    if len(left_ids) == 0 or len(right_ids) == 0:
        raise ValueError(
            "Cannot sample conditional pairs from an empty label pool."
        )

    max_possible = len(left_ids) * len(right_ids)
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
        f"Only sampled {len(selected)} of {n} requested pairs."
    )


def build_conditional_training_data(
    data: ProbeDataset,
    *,
    train_per_pair_type: int = 750,
    cal_per_pair_type: int = 250,
    seed: int = 0,
    semantics: str = "cooper_cantwell",
    template: str = "given_x_P",
    statement_config_path: str = "configs/statements.yaml",
) -> ConditionalTrainingData:
    """
    Build balanced direct-conditional train/cal examples entirely in memory.
    """
    if train_per_pair_type <= 0:
        raise ValueError(
            "train_per_pair_type must be positive."
        )
    if cal_per_pair_type <= 0:
        raise ValueError(
            "cal_per_pair_type must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    statements: list[str] = []
    labels: list[int] = []
    split_names: list[str] = []
    conditional_types: list[str] = []
    x_ids: list[int] = []
    P_ids: list[int] = []

    id_to_label = (
        data.label_schema.id_to_label
    )

    for split_name, count, split_mask in (
        (
            "train",
            int(train_per_pair_type),
            data.train_mask,
        ),
        (
            "cal",
            int(cal_per_pair_type),
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
                    int(data.labels[row_id])
                ]
                for row_id in split_ids
            ],
            dtype=object,
        )

        for x_label in LABELS:
            x_pool = split_ids[
                split_label_names
                == x_label
            ]

            for P_label in LABELS:
                P_pool = split_ids[
                    split_label_names
                    == P_label
                ]

                sampled = _sample_directed_pairs(
                    x_pool,
                    P_pool,
                    n=count,
                    rng=rng,
                    exclude_self=True,
                )

                expected = label_conditional(
                    x_label,
                    P_label,
                    semantics=semantics,
                )
                expected_id = LABEL_TO_ID[
                    expected
                ]
                conditional_type = (
                    f"{x_label}_given_to_{P_label}"
                )

                for x_id, P_id in sampled:
                    statement = render_pair_statement(
                        kind="conditional",
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
                        expected_id
                    )
                    split_names.append(
                        split_name
                    )
                    conditional_types.append(
                        conditional_type
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

    train_mask = split_array == "train"
    cal_mask = split_array == "cal"

    if not train_mask.any():
        raise RuntimeError(
            "Generated conditional training data contains no train rows."
        )
    if not cal_mask.any():
        raise RuntimeError(
            "Generated conditional training data contains no calibration rows."
        )
    if np.any(
        train_mask
        & cal_mask
    ):
        raise RuntimeError(
            "Conditional train/cal masks overlap."
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
    expected_classes = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    if not np.array_equal(
        train_classes,
        expected_classes,
    ):
        raise RuntimeError(
            f"Conditional training split has classes "
            f"{train_classes.tolist()}, expected [0,1,2]."
        )
    if not np.array_equal(
        cal_classes,
        expected_classes,
    ):
        raise RuntimeError(
            f"Conditional calibration split has classes "
            f"{cal_classes.tolist()}, expected [0,1,2]."
        )

    return ConditionalTrainingData(
        statements=tuple(
            statements
        ),
        labels=labels_array,
        train_mask=train_mask,
        cal_mask=cal_mask,
        conditional_types=tuple(
            conditional_types
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
        semantics=semantics,
    )
