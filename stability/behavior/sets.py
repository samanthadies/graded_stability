from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from stability.behavior.config import ChallengeConfig
from stability.data.loading import ProbeDataset


def _require(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns {missing}.")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    converted = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if converted.isna().any():
        raise ValueError(
            f"Could not interpret boolean values: {series[converted.isna()].head(10).tolist()}."
        )
    return converted.astype(bool)


def _empty_general_frames(*, model_name: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    challenge = pd.DataFrame(
        {
            "model_name": pd.Series(dtype="object"),
            "dataset": pd.Series(dtype="object"),
            "statement_id": pd.Series(dtype="int64"),
            "statement": pd.Series(dtype="object"),
            "source_dataset": pd.Series(dtype="object"),
            "source_row": pd.Series(dtype="int64"),
            "num_selecting_probes": pd.Series(dtype="int64"),
            "challenge_1_id": pd.Series(dtype="object"),
            "challenge_2_id": pd.Series(dtype="object"),
            "challenge_3_id": pd.Series(dtype="object"),
            "challenge_sequence": pd.Series(dtype="object"),
            "challenge_config": pd.Series(dtype="object"),
            "challenge_config_fingerprint": pd.Series(dtype="object"),
        }
    )
    membership = pd.DataFrame(
        {
            "model_name": pd.Series(dtype="object"),
            "dataset": pd.Series(dtype="object"),
            "probe": pd.Series(dtype="object"),
            "statement_id": pd.Series(dtype="int64"),
            "source_dataset": pd.Series(dtype="object"),
            "source_row": pd.Series(dtype="int64"),
            "pred_label": pd.Series(dtype="object"),
            "prob_true": pd.Series(dtype="float64"),
        }
    )
    return challenge, membership


def build_general_challenge_set(
    *,
    atomic: pd.DataFrame,
    data: ProbeDataset,
    config: ChallengeConfig,
    model_name: str,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate probe-specific P sets while preserving long membership.

    Degenerate model/dataset combinations are represented by schema-valid empty
    tables rather than raising, allowing a full batch sweep to continue.
    """
    _require(
        atomic,
        (
            "model_name", "dataset", "probe", "statement_id", "source_dataset",
            "source_row", "pred_label", "prob_true", "is_P",
        ),
        context="Atomic table",
    )
    if set(atomic["model_name"].dropna().astype(str).unique()) != {model_name}:
        raise ValueError("Atomic model identity does not match requested model.")
    if set(atomic["dataset"].dropna().astype(str).unique()) != {data.dataset}:
        raise ValueError("Atomic dataset identity does not match requested dataset.")

    atomic = atomic.copy()
    atomic["statement_id"] = pd.to_numeric(
        atomic["statement_id"], errors="raise"
    ).astype(np.int64)
    atomic["is_P"] = _as_bool(atomic["is_P"])

    membership_columns = [
        column
        for column in (
            "model_name", "dataset", "probe", "statement_id", "source_dataset",
            "source_row", "layer", "pred_label", "prob_false", "prob_true",
            "prob_neither", "threshold",
        )
        if column in atomic.columns
    ]
    membership = atomic.loc[atomic["is_P"], membership_columns].copy()
    if membership.empty:
        return _empty_general_frames(model_name=model_name, dataset=data.dataset)
    if membership.duplicated(["probe", "statement_id"]).any():
        raise ValueError("Duplicate (probe, statement_id) P membership rows.")

    test_ids = set(data.indices("test").tolist())
    selected_ids = set(membership["statement_id"].tolist())
    invalid = selected_ids - test_ids
    if invalid:
        raise ValueError(
            f"Selected P IDs outside the test split: {sorted(invalid)[:10]}."
        )

    ordered_ids = np.asarray(sorted(selected_ids), dtype=np.int64)
    challenge = pd.DataFrame(
        {
            "model_name": model_name,
            "dataset": data.dataset,
            "statement_id": ordered_ids,
            "statement": [data.statements[int(i)] for i in ordered_ids],
            "source_dataset": [str(data.source_dataset[int(i)]) for i in ordered_ids],
            "source_row": [int(data.source_row[int(i)]) for i in ordered_ids],
        }
    )
    counts = membership.groupby("statement_id")["probe"].nunique().rename(
        "num_selecting_probes"
    )
    challenge = challenge.merge(
        counts, left_on="statement_id", right_index=True, how="left", validate="one_to_one"
    )

    sequences = list(config.sequences)
    if len(sequences) != 6:
        raise RuntimeError("Expected six challenge permutations.")
    rng = np.random.default_rng(config.seed if seed is None else int(seed))
    order = np.arange(len(challenge))
    rng.shuffle(order)
    assigned: list[tuple[str, ...] | None] = [None] * len(challenge)
    for rank, row_index in enumerate(order):
        assigned[int(row_index)] = sequences[rank % len(sequences)]
    if any(sequence is None for sequence in assigned):
        raise RuntimeError("Challenge assignment failed.")

    assigned_final = [sequence for sequence in assigned if sequence is not None]
    for position in range(3):
        challenge[f"challenge_{position + 1}_id"] = [
            sequence[position] for sequence in assigned_final
        ]
    challenge["challenge_sequence"] = ["|".join(sequence) for sequence in assigned_final]
    challenge["challenge_config"] = config.name
    challenge["challenge_config_fingerprint"] = config.fingerprint

    return (
        challenge.sort_values("statement_id", kind="stable").reset_index(drop=True),
        membership.sort_values(["probe", "statement_id"], kind="stable").reset_index(drop=True),
    )


def _gamma_wide(gamma: pd.DataFrame) -> pd.DataFrame:
    _require(gamma, ("probe", "P_id", "source", "gamma"), context="Gamma table")
    work = gamma[["probe", "P_id", "source", "gamma"]].copy()
    work["P_id"] = pd.to_numeric(work["P_id"], errors="raise").astype(np.int64)
    work["gamma"] = pd.to_numeric(work["gamma"], errors="coerce")
    if work.duplicated(["probe", "P_id", "source"]).any():
        raise ValueError("Duplicate gamma (probe, P_id, source) rows.")
    return (
        work.pivot(index=["probe", "P_id"], columns="source", values="gamma")
        .reset_index()
        .rename(
            columns={
                "conditional": "gamma_conditional",
                "joint": "gamma_joint",
            }
        )
    )


def _minimum_distance_atomic_pairs(
    frame: pd.DataFrame,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-total-distance matching on atomic Pr(P)."""
    if frame.empty:
        return []

    ordered = (
        frame.reset_index(drop=True)
        .sort_values(["atomic_probability", "statement_id"], kind="stable")
        .reset_index(drop=True)
    )
    probabilities = ordered["atomic_probability"].to_numpy(dtype=np.float64)
    statement_ids = ordered["statement_id"].to_numpy(dtype=np.int64)
    n = len(ordered)
    if n < 2:
        return []

    if n % 2 == 0:
        unmatched_position = None
    else:
        candidates: list[tuple[float, int, int]] = []
        for unmatched in range(0, n, 2):
            cost = 0.0
            for start, stop in ((0, unmatched), (unmatched + 1, n)):
                for position in range(start, stop, 2):
                    cost += abs(
                        float(probabilities[position])
                        - float(probabilities[position + 1])
                    )
            candidates.append((float(cost), int(statement_ids[unmatched]), int(unmatched)))
        candidates.sort()
        unmatched_position = candidates[0][2]

    segment_bounds = (
        ((0, n),)
        if unmatched_position is None
        else ((0, unmatched_position), (unmatched_position + 1, n))
    )
    pairs: list[tuple[int, int, float]] = []
    for start, stop in segment_bounds:
        for position in range(start, stop, 2):
            left = ordered.iloc[position]
            right = ordered.iloc[position + 1]
            pairs.append(
                (
                    int(left["statement_id"]),
                    int(right["statement_id"]),
                    float(
                        abs(
                            float(left["atomic_probability"])
                            - float(right["atomic_probability"])
                        )
                    ),
                )
            )
    return pairs


def _orient_pair(
    left: pd.Series,
    right: pd.Series,
    *,
    estimator: str,
) -> dict[str, Any] | None:
    """Orient an already-selected atomic match by gamma."""
    if estimator == "conditional":
        left_gamma = float(left["gamma_conditional"])
        right_gamma = float(right["gamma_conditional"])
        difference = left_gamma - right_gamma
        if difference == 0.0:
            return None
        high, low = (left, right) if difference > 0 else (right, left)
        gamma_gap = abs(difference)
    elif estimator == "joint":
        left_gamma = float(left["gamma_joint"])
        right_gamma = float(right["gamma_joint"])
        difference = left_gamma - right_gamma
        if difference == 0.0:
            return None
        high, low = (left, right) if difference > 0 else (right, left)
        gamma_gap = abs(difference)
    elif estimator == "consensus":
        direct_difference = float(left["gamma_conditional"]) - float(right["gamma_conditional"])
        joint_difference = float(left["gamma_joint"]) - float(right["gamma_joint"])
        if (
            direct_difference == 0.0
            or joint_difference == 0.0
            or np.sign(direct_difference) != np.sign(joint_difference)
        ):
            return None
        high, low = (left, right) if direct_difference > 0 else (right, left)
        gamma_gap = min(abs(direct_difference), abs(joint_difference))
    else:
        raise ValueError(f"Unknown estimator {estimator!r}.")

    return {
        "high_statement_id": int(high["statement_id"]),
        "low_statement_id": int(low["statement_id"]),
        "gamma_gap": float(gamma_gap),
        "high_atomic_probability": float(high["atomic_probability"]),
        "low_atomic_probability": float(low["atomic_probability"]),
        "high_gamma_conditional": float(high["gamma_conditional"]),
        "low_gamma_conditional": float(low["gamma_conditional"]),
        "high_gamma_joint": float(high["gamma_joint"]),
        "low_gamma_joint": float(low["gamma_joint"]),
    }


def build_atomic_matched_pairs(
    *,
    atomic: pd.DataFrame,
    gamma: pd.DataFrame,
    challenge_set: pd.DataFrame,
    estimators: Sequence[str] = ("conditional", "joint", "consensus"),
) -> pd.DataFrame:
    """Build sequence-blocked, atomic-credence-matched pairs.

    Matching uses only atomic credence *within exact challenge-sequence blocks*.
    Gamma is used only after matching to orient each pair. Therefore the high-
    and low-gamma members of every pair receive the same ordered sequence of
    conversational challenges.
    """
    estimators = tuple(dict.fromkeys(str(e).strip().lower() for e in estimators))
    invalid = set(estimators) - {"conditional", "joint", "consensus"}
    if invalid:
        raise ValueError(f"Unknown estimators: {sorted(invalid)}.")

    _require(atomic, ("probe", "statement_id", "prob_true", "is_P"), context="Atomic table")
    _require(challenge_set, ("statement_id", "challenge_sequence"), context="Challenge set")

    if challenge_set.empty:
        return pd.DataFrame()

    challenge_lookup = challenge_set[["statement_id", "challenge_sequence"]].copy()
    challenge_lookup["statement_id"] = pd.to_numeric(
        challenge_lookup["statement_id"], errors="raise"
    ).astype(np.int64)
    if challenge_lookup["statement_id"].duplicated().any():
        raise ValueError("Challenge set must contain unique statement IDs.")

    work = atomic.copy()
    work["statement_id"] = pd.to_numeric(work["statement_id"], errors="raise").astype(np.int64)
    work["is_P"] = _as_bool(work["is_P"])
    work["atomic_probability"] = pd.to_numeric(work["prob_true"], errors="coerce")
    work = work.loc[
        work["is_P"], ["probe", "statement_id", "atomic_probability"]
    ].copy()
    if work.empty:
        return pd.DataFrame()

    work = work.merge(
        challenge_lookup,
        on="statement_id",
        how="inner",
        validate="many_to_one",
    )
    work = work.merge(
        _gamma_wide(gamma),
        left_on=["probe", "statement_id"],
        right_on=["probe", "P_id"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["P_id"])

    rows: list[dict[str, Any]] = []

    for probe_name, probe_frame in work.groupby("probe", sort=True):
        eligible = (
            probe_frame.dropna(
                subset=["atomic_probability", "gamma_conditional", "gamma_joint", "challenge_sequence"]
            )
            .copy()
            .reset_index(drop=True)
        )
        if len(eligible) < 2:
            continue

        block_counter = 0
        for challenge_sequence, sequence_frame in eligible.groupby("challenge_sequence", sort=True):
            sequence_frame = sequence_frame.sort_values(
                ["atomic_probability", "statement_id"], kind="stable"
            ).reset_index(drop=True)
            if len(sequence_frame) < 2:
                continue

            lookup = sequence_frame.set_index("statement_id", drop=False)
            atomic_pairs = _minimum_distance_atomic_pairs(sequence_frame)
            for local_pair_index, (statement_a_id, statement_b_id, atomic_gap) in enumerate(atomic_pairs):
                left = lookup.loc[statement_a_id]
                right = lookup.loc[statement_b_id]
                atomic_match_id = (
                    f"{probe_name}:sequence:{block_counter:02d}:atomic:{local_pair_index:04d}"
                )
                for estimator in estimators:
                    oriented = _orient_pair(left, right, estimator=estimator)
                    if oriented is None:
                        continue
                    rows.append(
                        {
                            "probe": str(probe_name),
                            "estimator": estimator,
                            "match_id": f"{atomic_match_id}:{estimator}",
                            "atomic_match_id": atomic_match_id,
                            "challenge_sequence": str(challenge_sequence),
                            "statement_a_id": int(statement_a_id),
                            "statement_b_id": int(statement_b_id),
                            "atomic_gap": float(atomic_gap),
                            **oriented,
                            "matching_method": (
                                "sequence_blocked_minimum_total_absolute_atomic_distance"
                            ),
                        }
                    )
            block_counter += 1

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if result["match_id"].duplicated().any():
        raise RuntimeError("Matched-pair IDs are not unique.")
    if not (result["challenge_sequence"].astype(str).str.len() > 0).all():
        raise RuntimeError("Matched pairs are missing challenge-sequence blocks.")

    return (
        result.sort_values(
            ["probe", "estimator", "challenge_sequence", "atomic_match_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
