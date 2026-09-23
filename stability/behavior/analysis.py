from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, f_oneway, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


PRIMARY_SAMPLE = "round0_agreement"
ROBUSTNESS_SAMPLE = "all"
PRIMARY_CONTINUOUS_OUTCOME = "mean_support_loss"
COMPLEMENTARY_CONTINUOUS_OUTCOME = "mean_absolute_movement"
SECONDARY_BINARY_OUTCOME = "ever_changed"


def _pipeline(*, binary: bool, n_knots: int) -> Pipeline:
    model: Any
    if binary:
        model = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0)
    else:
        model = Ridge(alpha=1.0)
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def add_behavior_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    initial_label = frame["round_0_pred_label"].astype(str).str.strip().str.lower()

    for round_number in range(4):
        true_p = pd.to_numeric(frame[f"round_{round_number}_prob_true"], errors="coerce")
        false_p = pd.to_numeric(frame[f"round_{round_number}_prob_false"], errors="coerce")
        frame[f"initial_answer_support_round_{round_number}"] = np.where(
            initial_label.eq("true"),
            true_p,
            np.where(initial_label.eq("false"), false_p, np.nan),
        )

    for challenge_round in (1, 2, 3):
        frame[f"support_loss_round_{challenge_round}"] = (
            frame["initial_answer_support_round_0"]
            - frame[f"initial_answer_support_round_{challenge_round}"]
        )
        frame[f"incremental_support_loss_round_{challenge_round}"] = (
            frame[f"initial_answer_support_round_{challenge_round - 1}"]
            - frame[f"initial_answer_support_round_{challenge_round}"]
        )
        frame[f"absolute_movement_round_{challenge_round}"] = (
            frame[f"initial_answer_support_round_{challenge_round}"]
            - frame[f"initial_answer_support_round_{challenge_round - 1}"]
        ).abs()

    frame["mean_support_loss"] = frame[
        ["support_loss_round_1", "support_loss_round_2", "support_loss_round_3"]
    ].mean(axis=1)
    frame["final_support_loss"] = frame["support_loss_round_3"]
    frame["max_support_loss"] = frame[
        ["support_loss_round_1", "support_loss_round_2", "support_loss_round_3"]
    ].max(axis=1)
    frame["mean_absolute_movement"] = frame[
        [
            "absolute_movement_round_1",
            "absolute_movement_round_2",
            "absolute_movement_round_3",
        ]
    ].mean(axis=1)
    frame["total_absolute_movement"] = frame[
        [
            "absolute_movement_round_1",
            "absolute_movement_round_2",
            "absolute_movement_round_3",
        ]
    ].sum(axis=1)

    changed_columns = []
    for challenge_round in (1, 2, 3):
        column = f"round_{challenge_round}_changed_from_previous"
        frame[column] = (
            frame[f"round_{challenge_round}_pred_label"].astype(str).str.lower()
            != frame[f"round_{challenge_round - 1}_pred_label"].astype(str).str.lower()
        )
        changed_columns.append(column)
    frame["ever_changed"] = frame[changed_columns].any(axis=1)
    frame["num_answer_changes"] = frame[changed_columns].sum(axis=1).astype(int)
    return frame


def build_probe_analysis_table(
    *,
    behavior: pd.DataFrame,
    membership: pd.DataFrame,
    atomic: pd.DataFrame,
    gamma: pd.DataFrame,
) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame()
    if behavior.empty:
        raise ValueError("Behavior output is empty but P membership is non-empty.")

    behavior = behavior.copy()
    behavior["statement_id"] = pd.to_numeric(
        behavior["statement_id"], errors="raise"
    ).astype(np.int64)
    if behavior["statement_id"].duplicated().any():
        raise ValueError("Behavior output must have one row per statement_id.")

    membership = membership.copy()
    membership["statement_id"] = pd.to_numeric(
        membership["statement_id"], errors="raise"
    ).astype(np.int64)
    if membership.duplicated(["probe", "statement_id"]).any():
        raise ValueError("Duplicate membership rows.")

    atomic_work = atomic[
        [
            column
            for column in (
                "probe", "statement_id", "pred_label", "prob_true", "threshold", "layer"
            )
            if column in atomic.columns
        ]
    ].copy()
    atomic_work["statement_id"] = pd.to_numeric(
        atomic_work["statement_id"], errors="raise"
    ).astype(np.int64)
    atomic_work = atomic_work.rename(
        columns={"pred_label": "atomic_pred_label", "prob_true": "atomic_probability"}
    )

    gamma_work = gamma[["probe", "P_id", "source", "gamma"]].copy()
    gamma_work["P_id"] = pd.to_numeric(gamma_work["P_id"], errors="raise").astype(np.int64)
    gamma_work["gamma"] = pd.to_numeric(gamma_work["gamma"], errors="coerce")
    if gamma_work.duplicated(["probe", "P_id", "source"]).any():
        raise ValueError("Duplicate gamma rows.")
    gamma_wide = (
        gamma_work.pivot(index=["probe", "P_id"], columns="source", values="gamma")
        .reset_index()
        .rename(
            columns={
                "P_id": "statement_id",
                "conditional": "gamma_conditional",
                "joint": "gamma_joint",
            }
        )
    )

    merged = membership[["probe", "statement_id"]].merge(
        behavior, on="statement_id", how="left", validate="many_to_one"
    )
    if "round_0_pred_label" not in merged or merged["round_0_pred_label"].isna().any():
        raise ValueError("Behavior results are missing selected P statements.")
    merged = merged.merge(
        atomic_work,
        on=["probe", "statement_id"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        gamma_wide,
        on=["probe", "statement_id"],
        how="left",
        validate="one_to_one",
    )
    merged = add_behavior_outcomes(merged)
    merged["round0_agrees_with_atomic"] = (
        merged["round_0_pred_label"].astype(str).str.strip().str.lower()
        == merged["atomic_pred_label"].astype(str).str.strip().str.lower()
    )
    return merged.sort_values(["probe", "statement_id"], kind="stable").reset_index(drop=True)


def _sample(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == ROBUSTNESS_SAMPLE:
        return frame.copy()
    if sample == PRIMARY_SAMPLE:
        return frame.loc[frame["round0_agrees_with_atomic"]].copy()
    raise ValueError(f"Unknown sample {sample!r}.")


def association_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Simple rank-association diagnostics for atomic credence and gamma."""
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for probe_name, probe_frame in frame.groupby("probe", sort=True):
        for sample_name in (PRIMARY_SAMPLE, ROBUSTNESS_SAMPLE):
            sample_frame = _sample(probe_frame, sample_name)
            for estimator, gamma_column in (
                ("conditional", "gamma_conditional"),
                ("joint", "gamma_joint"),
            ):
                for outcome in (
                    PRIMARY_CONTINUOUS_OUTCOME,
                    COMPLEMENTARY_CONTINUOUS_OUTCOME,
                    SECONDARY_BINARY_OUTCOME,
                ):
                    for predictor_name, predictor_column in (
                        ("atomic", "atomic_probability"),
                        ("gamma", gamma_column),
                    ):
                        complete = sample_frame[[predictor_column, outcome]].apply(
                            pd.to_numeric, errors="coerce"
                        ).dropna()
                        if len(complete) < 3 or complete[predictor_column].nunique() < 2:
                            continue
                        rho, p_value = spearmanr(
                            complete[predictor_column].to_numpy(dtype=float),
                            complete[outcome].to_numpy(dtype=float),
                        )
                        rows.append(
                            {
                                "probe": str(probe_name),
                                "sample": sample_name,
                                "estimator": estimator,
                                "outcome": outcome,
                                "predictor": predictor_name,
                                "n": int(len(complete)),
                                "spearman_rho": float(rho),
                                "spearman_p_two_sided": float(p_value),
                            }
                        )
    return pd.DataFrame(rows)


def _metric_row(
    *,
    y: np.ndarray,
    prediction: np.ndarray,
    binary: bool,
) -> dict[str, float]:
    if binary:
        return {
            "log_loss": float(log_loss(y.astype(int), prediction, labels=[0, 1])),
            "roc_auc": (
                float(roc_auc_score(y.astype(int), prediction))
                if len(np.unique(y.astype(int))) == 2
                else np.nan
            ),
        }
    return {
        "r2": float(r2_score(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
    }


def cross_validated_predictive_validity(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_knots: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate atomic, gamma, and atomic+gamma using pooled OOF predictions.

    For each repeat every observation receives exactly one held-out prediction.
    Metrics are then calculated once over the complete out-of-fold prediction
    vector for that repeat. This avoids averaging fold-level R^2 values.
    """
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    prediction_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []

    for probe_name, probe_frame in frame.groupby("probe", sort=True):
        for sample_name in (PRIMARY_SAMPLE, ROBUSTNESS_SAMPLE):
            sample_frame = _sample(probe_frame, sample_name)
            for estimator, gamma_column in (
                ("conditional", "gamma_conditional"),
                ("joint", "gamma_joint"),
            ):
                for outcome, binary in (
                    (PRIMARY_CONTINUOUS_OUTCOME, False),
                    (COMPLEMENTARY_CONTINUOUS_OUTCOME, False),
                    (SECONDARY_BINARY_OUTCOME, True),
                ):
                    columns = ["statement_id", "atomic_probability", gamma_column, outcome]
                    complete = sample_frame[columns].copy()
                    for column in ("atomic_probability", gamma_column, outcome):
                        complete[column] = pd.to_numeric(complete[column], errors="coerce")
                    complete = complete.dropna().reset_index(drop=True)
                    if len(complete) < max(n_splits, 10):
                        continue

                    y = complete[outcome].to_numpy(dtype=np.float64)
                    if binary:
                        y = y.astype(int)
                        counts = np.bincount(y, minlength=2)
                        effective_splits = min(int(n_splits), int(counts.min()))
                        if effective_splits < 2:
                            continue
                        splitter = RepeatedStratifiedKFold(
                            n_splits=effective_splits,
                            n_repeats=n_repeats,
                            random_state=seed,
                        )
                        splits = list(splitter.split(np.zeros(len(y)), y))
                    else:
                        effective_splits = min(int(n_splits), len(complete))
                        if effective_splits < 2:
                            continue
                        splitter = RepeatedKFold(
                            n_splits=effective_splits,
                            n_repeats=n_repeats,
                            random_state=seed,
                        )
                        splits = list(splitter.split(np.arange(len(y))))

                    specs = {
                        "atomic": ["atomic_probability"],
                        "gamma": [gamma_column],
                        "atomic_plus_gamma": ["atomic_probability", gamma_column],
                    }
                    X_by_model = {
                        model_name: complete[predictors].to_numpy(dtype=np.float64)
                        for model_name, predictors in specs.items()
                    }
                    oof = {
                        model_name: np.full((n_repeats, len(complete)), np.nan, dtype=np.float64)
                        for model_name in specs
                    }
                    fold_assignment = np.full((n_repeats, len(complete)), -1, dtype=np.int64)

                    for split_index, (train_index, test_index) in enumerate(splits):
                        repeat = split_index // effective_splits
                        fold = split_index % effective_splits
                        fold_assignment[repeat, test_index] = fold
                        y_train = y[train_index]
                        for model_name, X in X_by_model.items():
                            model = _pipeline(binary=binary, n_knots=n_knots)
                            model.fit(X[train_index], y_train)
                            if binary:
                                prediction = model.predict_proba(X[test_index])[:, 1]
                            else:
                                prediction = model.predict(X[test_index])
                            oof[model_name][repeat, test_index] = prediction

                    for repeat in range(n_repeats):
                        if (fold_assignment[repeat] < 0).any():
                            raise RuntimeError("OOF fold assignment is incomplete.")
                        for model_name, matrix in oof.items():
                            prediction = matrix[repeat]
                            if not np.isfinite(prediction).all():
                                raise RuntimeError("OOF predictions are incomplete.")
                            metrics = _metric_row(y=y, prediction=prediction, binary=binary)
                            metric_records.append(
                                {
                                    "probe": str(probe_name),
                                    "sample": sample_name,
                                    "estimator": estimator,
                                    "outcome": outcome,
                                    "model": model_name,
                                    "repeat": int(repeat),
                                    "n_total": int(len(complete)),
                                    "n_splits": int(effective_splits),
                                    **metrics,
                                }
                            )
                            for row_index, value in enumerate(prediction):
                                prediction_records.append(
                                    {
                                        "probe": str(probe_name),
                                        "sample": sample_name,
                                        "estimator": estimator,
                                        "outcome": outcome,
                                        "model": model_name,
                                        "repeat": int(repeat),
                                        "fold": int(fold_assignment[repeat, row_index]),
                                        "statement_id": int(complete.iloc[row_index]["statement_id"]),
                                        "y_true": float(y[row_index]),
                                        "y_pred": float(value),
                                    }
                                )

    return pd.DataFrame(prediction_records), pd.DataFrame(metric_records)


def summarize_cv(repeat_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if repeat_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = ["probe", "sample", "estimator", "outcome", "model"]
    metrics = [
        column for column in ("r2", "rmse", "log_loss", "roc_auc")
        if column in repeat_metrics.columns
    ]
    summary = repeat_metrics.groupby(groups, dropna=False, sort=True)[metrics].agg(
        ["mean", "std", "median"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    keys = ["probe", "sample", "estimator", "outcome", "repeat"]
    atomic = repeat_metrics.loc[repeat_metrics["model"].eq("atomic")]
    full = repeat_metrics.loc[repeat_metrics["model"].eq("atomic_plus_gamma")]
    deltas = atomic[keys + metrics].merge(
        full[keys + metrics],
        on=keys,
        how="inner",
        suffixes=("_atomic", "_full"),
        validate="one_to_one",
    )
    if "r2_atomic" in deltas:
        deltas["delta_r2"] = deltas["r2_full"] - deltas["r2_atomic"]
    if "rmse_atomic" in deltas:
        deltas["delta_rmse"] = deltas["rmse_atomic"] - deltas["rmse_full"]
    if "log_loss_atomic" in deltas:
        deltas["delta_log_loss"] = deltas["log_loss_atomic"] - deltas["log_loss_full"]
    if "roc_auc_atomic" in deltas:
        deltas["delta_roc_auc"] = deltas["roc_auc_full"] - deltas["roc_auc_atomic"]
    return summary, deltas


def challenge_order_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    sequence_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    for probe_name, probe_frame in frame.groupby("probe", sort=True):
        for sample_name in (PRIMARY_SAMPLE, ROBUSTNESS_SAMPLE):
            sample_frame = _sample(probe_frame, sample_name)
            for sequence, group in sample_frame.groupby("challenge_sequence", sort=True):
                values = pd.to_numeric(group[PRIMARY_CONTINUOUS_OUTCOME], errors="coerce").dropna()
                sequence_rows.append(
                    {
                        "probe": str(probe_name),
                        "sample": sample_name,
                        "challenge_sequence": str(sequence),
                        "n": int(len(values)),
                        "mean_support_loss": float(values.mean()) if len(values) else np.nan,
                        "sem_support_loss": (
                            float(values.std(ddof=1) / np.sqrt(len(values)))
                            if len(values) > 1 else np.nan
                        ),
                    }
                )
            for position in (1, 2, 3):
                for challenge_id, group in sample_frame.groupby(
                    f"challenge_{position}_id", sort=True
                ):
                    values = pd.to_numeric(
                        group[f"incremental_support_loss_round_{position}"], errors="coerce"
                    ).dropna()
                    template_rows.append(
                        {
                            "probe": str(probe_name),
                            "sample": sample_name,
                            "position": position,
                            "challenge_id": str(challenge_id),
                            "n": int(len(values)),
                            "mean_incremental_support_loss": (
                                float(values.mean()) if len(values) else np.nan
                            ),
                            "sem_incremental_support_loss": (
                                float(values.std(ddof=1) / np.sqrt(len(values)))
                                if len(values) > 1 else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(sequence_rows), pd.DataFrame(template_rows)


def sequence_effect_tests(
    frame: pd.DataFrame,
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation diagnostic for challenge-sequence effects on support loss."""
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for probe_name, probe_frame in frame.groupby("probe", sort=True):
        for sample_name in (PRIMARY_SAMPLE, ROBUSTNESS_SAMPLE):
            sample_frame = _sample(probe_frame, sample_name)[
                ["challenge_sequence", PRIMARY_CONTINUOUS_OUTCOME]
            ].copy()
            sample_frame[PRIMARY_CONTINUOUS_OUTCOME] = pd.to_numeric(
                sample_frame[PRIMARY_CONTINUOUS_OUTCOME], errors="coerce"
            )
            sample_frame = sample_frame.dropna()
            if sample_frame["challenge_sequence"].nunique() < 2:
                continue
            groups = [
                group[PRIMARY_CONTINUOUS_OUTCOME].to_numpy(dtype=float)
                for _, group in sample_frame.groupby("challenge_sequence", sort=True)
            ]
            groups = [values for values in groups if len(values) >= 2]
            if len(groups) < 2:
                continue
            observed, _ = f_oneway(*groups)
            if not np.isfinite(observed):
                continue

            labels = sample_frame["challenge_sequence"].astype(str).to_numpy()
            values = sample_frame[PRIMARY_CONTINUOUS_OUTCOME].to_numpy(dtype=float)
            rng = np.random.default_rng(seed + sum(ord(c) for c in str(probe_name) + sample_name))
            null = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                shuffled = rng.permutation(labels)
                perm_groups = [values[shuffled == label] for label in np.unique(shuffled)]
                stat, _ = f_oneway(*perm_groups)
                null[i] = float(stat)
            p_value = (1 + int(np.sum(null >= observed))) / (n_permutations + 1)
            rows.append(
                {
                    "probe": str(probe_name),
                    "sample": sample_name,
                    "n_sequences": int(len(groups)),
                    "n_rows": int(len(sample_frame)),
                    "observed_f_statistic": float(observed),
                    "permutation_p_greater": float(p_value),
                    "n_permutations": int(n_permutations),
                }
            )
    return pd.DataFrame(rows)


def analyze_matched_pairs(
    *, analysis_rows: pd.DataFrame, matched_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if analysis_rows.empty or matched_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    lookup = analysis_rows[
        [
            "probe", "statement_id", PRIMARY_CONTINUOUS_OUTCOME,
            COMPLEMENTARY_CONTINUOUS_OUTCOME, SECONDARY_BINARY_OUTCOME,
            "round0_agrees_with_atomic", "challenge_sequence",
        ]
    ]
    records: list[dict[str, Any]] = []
    for _, match in matched_pairs.iterrows():
        probe = str(match["probe"])
        probe_rows = lookup.loc[lookup["probe"].astype(str).eq(probe)]
        high = probe_rows.loc[probe_rows["statement_id"].eq(int(match["high_statement_id"]))]
        low = probe_rows.loc[probe_rows["statement_id"].eq(int(match["low_statement_id"]))]
        if len(high) != 1 or len(low) != 1:
            raise ValueError(f"Could not uniquely recover behavior for {match['match_id']}.")
        high_row, low_row = high.iloc[0], low.iloc[0]

        high_sequence = str(high_row["challenge_sequence"])
        low_sequence = str(low_row["challenge_sequence"])
        expected_sequence = str(match.get("challenge_sequence", high_sequence))
        if high_sequence != low_sequence or high_sequence != expected_sequence:
            raise ValueError(
                f"Matched pair {match['match_id']} is not sequence-blocked: "
                f"high={high_sequence}, low={low_sequence}, expected={expected_sequence}."
            )

        for sample_name in (PRIMARY_SAMPLE, ROBUSTNESS_SAMPLE):
            if sample_name == PRIMARY_SAMPLE and not (
                bool(high_row["round0_agrees_with_atomic"])
                and bool(low_row["round0_agrees_with_atomic"])
            ):
                continue

            high_loss = float(high_row[PRIMARY_CONTINUOUS_OUTCOME])
            low_loss = float(low_row[PRIMARY_CONTINUOUS_OUTCOME])
            resilience_difference = low_loss - high_loss

            high_movement = float(high_row[COMPLEMENTARY_CONTINUOUS_OUTCOME])
            low_movement = float(low_row[COMPLEMENTARY_CONTINUOUS_OUTCOME])
            movement_difference = low_movement - high_movement

            high_flip = int(bool(high_row[SECONDARY_BINARY_OUTCOME]))
            low_flip = int(bool(low_row[SECONDARY_BINARY_OUTCOME]))
            flip_difference = low_flip - high_flip

            records.append(
                {
                    **match.to_dict(),
                    "sample": sample_name,
                    "high_mean_support_loss": high_loss,
                    "low_mean_support_loss": low_loss,
                    "resilience_difference": resilience_difference,
                    "resilience_success": bool(resilience_difference > 0),
                    "resilience_tie": bool(resilience_difference == 0),
                    "high_mean_absolute_movement": high_movement,
                    "low_mean_absolute_movement": low_movement,
                    "movement_difference": movement_difference,
                    "movement_success": bool(movement_difference > 0),
                    "movement_tie": bool(movement_difference == 0),
                    "high_ever_changed": high_flip,
                    "low_ever_changed": low_flip,
                    "flip_difference": flip_difference,
                    "flip_success": bool(flip_difference > 0),
                    "flip_tie": bool(flip_difference == 0),
                    "high_challenge_sequence": high_sequence,
                    "low_challenge_sequence": low_sequence,
                }
            )

    pair_results = pd.DataFrame(records)
    if pair_results.empty:
        return pair_results, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for (probe, estimator, sample_name), group in pair_results.groupby(
        ["probe", "estimator", "sample"], sort=True
    ):
        non_tie = group.loc[~group["resilience_tie"]]
        successes = int(non_tie["resilience_success"].sum())
        trials = int(len(non_tie))
        movement_non_tie = group.loc[~group["movement_tie"]]
        movement_successes = int(movement_non_tie["movement_success"].sum())
        movement_trials = int(len(movement_non_tie))
        flip_non_tie = group.loc[~group["flip_tie"]]
        flip_successes = int(flip_non_tie["flip_success"].sum())
        flip_trials = int(len(flip_non_tie))

        summary_rows.append(
            {
                "probe": str(probe),
                "estimator": str(estimator),
                "sample": str(sample_name),
                "n_pairs": int(len(group)),
                "median_atomic_gap": float(group["atomic_gap"].median()),
                "median_gamma_gap": float(group["gamma_gap"].median()),
                "n_non_tie": trials,
                "successes": successes,
                "success_rate": successes / trials if trials else np.nan,
                "mean_resilience_difference": float(group["resilience_difference"].mean()),
                "median_resilience_difference": float(group["resilience_difference"].median()),
                "sign_test_p_greater_than_half": (
                    float(binomtest(successes, trials, p=0.5, alternative="greater").pvalue)
                    if trials else np.nan
                ),
                "movement_non_ties": movement_trials,
                "movement_successes": movement_successes,
                "movement_success_rate": (
                    movement_successes / movement_trials if movement_trials else np.nan
                ),
                "mean_movement_difference": float(group["movement_difference"].mean()),
                "median_movement_difference": float(group["movement_difference"].median()),
                "movement_sign_test_p_greater_than_half": (
                    float(
                        binomtest(
                            movement_successes, movement_trials, p=0.5, alternative="greater"
                        ).pvalue
                    ) if movement_trials else np.nan
                ),
                "flip_non_ties": flip_trials,
                "flip_successes": flip_successes,
                "flip_success_rate": flip_successes / flip_trials if flip_trials else np.nan,
                "flip_sign_test_p_greater_than_half": (
                    float(binomtest(flip_successes, flip_trials, p=0.5, alternative="greater").pvalue)
                    if flip_trials else np.nan
                ),
            }
        )
    return pair_results, pd.DataFrame(summary_rows)


def summarize_matched_sensitivity(
    pair_results: pd.DataFrame,
    *,
    atomic_gap_thresholds: Iterable[float],
    gamma_gap_thresholds: Iterable[float],
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Threshold-grid sensitivity analysis; never used for primary selection."""
    if pair_results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    atomic_thresholds = sorted(set(float(x) for x in atomic_gap_thresholds))
    gamma_thresholds = sorted(set(float(x) for x in gamma_gap_thresholds))
    for (probe, estimator, sample_name), group in pair_results.groupby(
        ["probe", "estimator", "sample"], sort=True
    ):
        for max_atomic_gap in atomic_thresholds:
            for min_gamma_gap in gamma_thresholds:
                subset = group.loc[
                    (pd.to_numeric(group["atomic_gap"], errors="coerce") <= max_atomic_gap)
                    & (pd.to_numeric(group["gamma_gap"], errors="coerce") >= min_gamma_gap)
                ].copy()
                if len(subset) < min_pairs:
                    continue
                resistance = subset.loc[~subset["resilience_tie"]]
                movement = subset.loc[~subset["movement_tie"]]
                flips = subset.loc[~subset["flip_tie"]]
                rows.append(
                    {
                        "probe": str(probe),
                        "estimator": str(estimator),
                        "sample": str(sample_name),
                        "max_atomic_gap": float(max_atomic_gap),
                        "min_gamma_gap": float(min_gamma_gap),
                        "n_pairs": int(len(subset)),
                        "resilience_success_rate": (
                            float(resistance["resilience_success"].mean())
                            if len(resistance) else np.nan
                        ),
                        "mean_resilience_difference": float(subset["resilience_difference"].mean()),
                        "movement_success_rate": (
                            float(movement["movement_success"].mean()) if len(movement) else np.nan
                        ),
                        "mean_movement_difference": float(subset["movement_difference"].mean()),
                        "flip_success_rate": (
                            float(flips["flip_success"].mean()) if len(flips) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)
