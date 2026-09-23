from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from patsy import build_design_matrices, dmatrix

from stability.utils.io import write_parquet_atomic

DATASETS = ("cities_loc", "med_indications", "defs")
PROBES = ("sawmil", "svm", "mean_difference")
ESTIMATORS = {"direct": "conditional", "joint": "joint"}
GAMMA_EXTRA = (
    "numerator", "denominator", "num_rows", "num_undefined",
    "mean_score", "min_score", "max_score", "zero_tolerance",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-fitted stability-vs-atomic-credence analysis."
    )
    p.add_argument("--repo_root", type=Path, default=Path("."))
    p.add_argument(
        "--model_list", type=Path, default=Path("configs/model_list.yaml")
    )
    p.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--probe", nargs="+", choices=PROBES, default=["sawmil"])
    p.add_argument(
        "--estimator", nargs="+", choices=tuple(ESTIMATORS),
        default=list(ESTIMATORS),
    )
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--spline_df", type=int, default=3)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--min_fit_statements", type=int, default=50)
    p.add_argument("--min_overlap", type=int, default=50)
    p.add_argument("--n_permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curve_points", type=int, default=200)
    p.add_argument("--threshold_tolerance", type=float, default=1e-8)
    p.add_argument(
        "--output_dir", type=Path,
        default=Path("outputs/analysis/stability_vs_credence"),
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.spline_df < 2:
        p.error("--spline_df must be >= 2")
    if a.cv_folds < 2:
        p.error("--cv_folds must be >= 2")
    if a.min_fit_statements < max(a.cv_folds, a.spline_df + 2):
        p.error("--min_fit_statements is too small for the requested CV/spline")
    if a.min_overlap < 3:
        p.error("--min_overlap must be >= 3")
    if a.n_permutations < 0:
        p.error("--n_permutations must be >= 0")
    return a


def under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_model_list(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]
    if isinstance(raw, dict):
        models = [str(x) for x in raw]
    elif isinstance(raw, list):
        models = []
        for x in raw:
            if isinstance(x, str):
                models.append(x)
            elif isinstance(x, dict):
                value = next((x[k] for k in ("name", "config", "model_name", "key") if k in x), None)
                if value is None and len(x) == 1:
                    value = next(iter(x))
                if value is None:
                    raise ValueError(f"Cannot infer model key from {x!r}")
                models.append(str(value))
            else:
                raise ValueError(f"Unsupported model-list entry: {x!r}")
    else:
        raise ValueError(f"Unsupported model-list format in {path}")
    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def coerce_bool(series: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    s = series.astype(str).str.strip().str.lower()
    good = {"true", "false", "1", "0", "yes", "no"}
    bad = sorted(set(s.unique()) - good)
    if bad:
        raise ValueError(f"{context}: invalid boolean values {bad[:10]}")
    return s.isin({"true", "1", "yes"})


def one_finite_value(series: pd.Series, context: str, tol: float = 1e-12) -> float | None:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    if np.max(np.abs(x - x[0])) > tol:
        raise ValueError(f"{context}: expected one value; got range [{x.min()}, {x.max()}]")
    return float(x[0])


def load_atomic(path: Path, probe: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = {"probe", "statement_id", "prob_true", "is_P"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing atomic columns {sorted(missing)}")
    df = df.loc[df["probe"].astype(str).eq(probe)].copy()
    if df.empty:
        return pd.DataFrame(columns=["P_id", "atomic_probability"]), {
            "num_atomic_rows": 0, "num_atomic_P": 0,
            "atomic_threshold": None, "reason": "probe_absent_from_atomic",
        }
    df["is_P"] = coerce_bool(df["is_P"], f"{path}/{probe}/is_P")
    threshold = one_finite_value(df["threshold"], f"{path}/{probe}/threshold") if "threshold" in df else None
    believed = df.loc[df["is_P"]].copy()
    if believed.empty:
        return pd.DataFrame(columns=["P_id", "atomic_probability"]), {
            "num_atomic_rows": int(len(df)), "num_atomic_P": 0,
            "atomic_threshold": threshold, "reason": "no_P_statements",
        }
    believed["atomic_probability"] = pd.to_numeric(believed["prob_true"], errors="coerce")
    bad = ~np.isfinite(believed["atomic_probability"]) | ~believed["atomic_probability"].between(0, 1)
    if bad.any():
        raise ValueError(f"{path}/{probe}: {int(bad.sum())} invalid prob_true values")
    believed["P_id"] = believed["statement_id"].astype(str)
    if believed["P_id"].duplicated().any():
        raise ValueError(f"{path}/{probe}: duplicate believed statement_id values")
    keep = ["P_id", "atomic_probability"]
    for col in ("pred_label", "belief_status", "score", "threshold", "statement", "text"):
        if col in believed and col not in keep:
            keep.append(col)
    return believed[keep].copy(), {
        "num_atomic_rows": int(len(df)), "num_atomic_P": int(len(believed)),
        "atomic_threshold": threshold, "reason": None,
    }


def load_gamma(path: Path, probe: str, source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["P_id", "gamma"]), {
            "num_gamma_rows_total": 0, "num_gamma_valid": 0,
            "gamma_threshold": None, "reason": "empty_gamma_table",
        }
    required = {"probe", "P_id", "source", "gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing gamma columns {sorted(missing)}")
    df = df.loc[df["probe"].astype(str).eq(probe) & df["source"].astype(str).eq(source)].copy()
    if df.empty:
        return pd.DataFrame(columns=["P_id", "gamma"]), {
            "num_gamma_rows_total": 0, "num_gamma_valid": 0,
            "gamma_threshold": None, "reason": "probe_source_absent_from_gamma",
        }
    df["P_id"] = df["P_id"].astype(str)
    if df["P_id"].duplicated().any():
        raise ValueError(f"{path}/{probe}/{source}: duplicate P_id values")
    threshold = one_finite_value(df["threshold"], f"{path}/{probe}/{source}/threshold") if "threshold" in df else None
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    valid = np.isfinite(df["gamma"]) & df["gamma"].between(0, 1)
    out = df.loc[valid].copy()
    keep = ["P_id", "gamma"] + [c for c in GAMMA_EXTRA if c in out]
    return out[keep].copy(), {
        "num_gamma_rows_total": int(len(df)), "num_gamma_valid": int(len(out)),
        "gamma_threshold": threshold,
        "reason": None if len(out) else "no_finite_gamma_values",
    }


def stable_seed(seed: int, *parts: str) -> int:
    b = "|".join([str(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little") % (2**32 - 1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = pd.Series(np.asarray(x, dtype=float)); b = pd.Series(np.asarray(y, dtype=float))
    keep = np.isfinite(a) & np.isfinite(b); a = a[keep]; b = b[keep]
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank(method="average").corr(b.rank(method="average")))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y); x = x[keep]; y = y[keep]
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    tss = float(np.sum((y - y.mean()) ** 2))
    return np.nan if tss <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / tss


def folds(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); order = rng.permutation(n); out = np.empty(n, int)
    for fold, ids in enumerate(np.array_split(order, k)):
        out[ids] = fold
    return out


def spline_formula(df: int) -> str:
    # Fixed probability boundaries ensure the same basis definition in each CV fold.
    return f"cr(x, df={df}, constraints='center', lower_bound=0.0, upper_bound=1.0)"


def fit_spline(x_train, y_train, x_test, df: int) -> np.ndarray:
    X = dmatrix(spline_formula(df), {"x": np.asarray(x_train, float)}, return_type="dataframe")
    if np.linalg.matrix_rank(X.to_numpy(float)) < X.shape[1]:
        raise np.linalg.LinAlgError("Natural cubic spline design is rank deficient")
    beta, *_ = np.linalg.lstsq(X.to_numpy(float), np.asarray(y_train, float), rcond=None)
    Xt = build_design_matrices([X.design_info], {"x": np.asarray(x_test, float)})[0]
    return np.asarray(Xt, float) @ beta


def fit_linear(x_train, y_train, x_test) -> np.ndarray:
    X = np.column_stack([np.ones(len(x_train)), np.asarray(x_train, float)])
    Xt = np.column_stack([np.ones(len(x_test)), np.asarray(x_test, float)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y_train, float), rcond=None)
    return Xt @ beta


def cross_fit(df: pd.DataFrame, spline_df: int, cv_folds: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.reset_index(drop=True).copy()
    x = work["atomic_probability"].to_numpy(float); y = work["gamma"].to_numpy(float)
    if len(np.unique(x)) < spline_df + 1:
        raise ValueError("Insufficient unique atomic probabilities for spline fit")
    if len(np.unique(y)) < 2:
        raise ValueError("Gamma is constant")
    f = folds(len(work), cv_folds, seed)
    ps = np.full(len(work), np.nan); pl = np.full(len(work), np.nan)
    for j in range(cv_folds):
        test = f == j; train = ~test
        if len(np.unique(x[train])) < spline_df + 1:
            raise ValueError(f"Fold {j}: insufficient unique training probabilities")
        ps[test] = fit_spline(x[train], y[train], x[test], spline_df)
        pl[test] = fit_linear(x[train], y[train], x[test])
    residual = y - ps; abs_res = np.abs(residual)
    metrics = {
        "cv_r_squared": r2(y, ps),
        "cv_mae": float(abs_res.mean()),
        "cv_rmse": float(np.sqrt(np.mean(residual**2))),
        "linear_cv_r_squared": r2(y, pl),
        "linear_cv_mae": float(np.mean(np.abs(y - pl))),
        "linear_cv_rmse": float(np.sqrt(np.mean((y - pl)**2))),
        "delta_cv_r_squared_spline_vs_linear": r2(y, ps) - r2(y, pl),
        "spearman_atomic_gamma": spearman(x, y),
        "pearson_atomic_gamma": pearson(x, y),
        "mean_absolute_residual_gamma": float(abs_res.mean()),
        "median_absolute_residual_gamma": float(np.median(abs_res)),
        "p90_absolute_residual_gamma": float(np.quantile(abs_res, .90)),
        "p95_absolute_residual_gamma": float(np.quantile(abs_res, .95)),
        "sd_residual_gamma": float(np.std(residual, ddof=1)),
        "sd_gamma": float(np.std(y, ddof=1)),
        "fraction_cv_predictions_outside_0_1": float(np.mean((ps < 0) | (ps > 1))),
    }
    work["cv_fold"] = f; work["cv_predicted_gamma"] = ps
    work["cv_predicted_gamma_linear"] = pl; work["residual_gamma"] = residual
    work["absolute_residual_gamma"] = abs_res
    return work, metrics


def full_curve(df: pd.DataFrame, spline_df: int, n: int) -> tuple[pd.DataFrame, dict[str, float]]:
    x = df["atomic_probability"].to_numpy(float); y = df["gamma"].to_numpy(float)
    grid = np.linspace(float(x.min()), float(x.max()), n)
    fitted = fit_spline(x, y, x, spline_df); pred = fit_spline(x, y, grid, spline_df)
    return pd.DataFrame({"atomic_probability": grid, "fitted_gamma_full": pred}), {
        "full_fit_r_squared": r2(y, fitted),
        "full_fit_mae": float(np.mean(np.abs(y - fitted))),
        "full_fit_rmse": float(np.sqrt(np.mean((y - fitted)**2))),
    }


def analyze_unit(root: Path, model: str, dataset: str, probe: str, estimator: str, args):
    source = ESTIMATORS[estimator]
    atomic_path = root / "outputs" / "atomic" / model / f"{dataset}.parquet"
    gamma_path = root / "outputs" / "gamma" / model / f"{dataset}.parquet"
    atomic, am = load_atomic(atomic_path, probe); gamma, gm = load_gamma(gamma_path, probe, source)
    base = {"model": model, "dataset": dataset, "probe": probe, "estimator": estimator,
            "source": source, "atomic_path": str(atomic_path), "gamma_path": str(gamma_path), **am, **gm}
    if atomic.empty:
        return None, None, None, {**base, "status": "skipped", "reason": am["reason"], "n_common": 0}
    if gamma.empty:
        return None, None, None, {**base, "status": "skipped", "reason": gm["reason"], "n_common": 0}
    if am["atomic_threshold"] is not None and gm["gamma_threshold"] is not None:
        if abs(am["atomic_threshold"] - gm["gamma_threshold"]) > args.threshold_tolerance:
            raise ValueError("Atomic and gamma thresholds disagree")
    extra_gamma = sorted(set(gamma["P_id"]) - set(atomic["P_id"]))
    if extra_gamma:
        raise ValueError(f"Gamma contains P IDs outside atomic belief set: {extra_gamma[:10]}")
    merged = atomic.merge(gamma, on="P_id", how="inner", validate="one_to_one")
    if len(merged) < args.min_fit_statements:
        return None, None, None, {**base, "status": "skipped", "reason": "insufficient_common_P", "n_common": int(len(merged))}
    unique_atomic_probabilities = int(
        merged["atomic_probability"].nunique()
    )
    if unique_atomic_probabilities < args.spline_df + 1:
        return None, None, None, {
            **base,
            "status": "skipped",
            "reason": "insufficient_unique_atomic_probabilities",
            "n_common": int(len(merged)),
            "num_unique_atomic_probabilities": unique_atomic_probabilities,
        }

    unique_gamma_values = int(
        merged["gamma"].nunique()
    )
    if unique_gamma_values < 2:
        return None, None, None, {
            **base,
            "status": "skipped",
            "reason": "constant_gamma",
            "n_common": int(len(merged)),
            "num_unique_atomic_probabilities": unique_atomic_probabilities,
            "num_unique_gamma_values": unique_gamma_values,
        }

    useed = stable_seed(args.seed, model, dataset, probe, estimator)

    try:
        statements, metrics = cross_fit(
            merged,
            args.spline_df,
            args.cv_folds,
            useed,
        )
    except ValueError as exc:
        message = str(exc)
        if (
            "insufficient unique training probabilities" in message.lower()
            or "insufficient unique atomic probabilities" in message.lower()
        ):
            return None, None, None, {
                **base,
                "status": "skipped",
                "reason": "insufficient_unique_atomic_probabilities_for_cv",
                "message": message,
                "n_common": int(len(merged)),
                "num_unique_atomic_probabilities": unique_atomic_probabilities,
            }
        if "gamma is constant" in message.lower():
            return None, None, None, {
                **base,
                "status": "skipped",
                "reason": "constant_gamma",
                "message": message,
                "n_common": int(len(merged)),
                "num_unique_atomic_probabilities": unique_atomic_probabilities,
                "num_unique_gamma_values": unique_gamma_values,
            }
        raise

    curve, fit_metrics = full_curve(
        merged,
        args.spline_df,
        args.curve_points,
    )
    summary = {
        **{k: base[k] for k in ("model", "dataset", "probe", "estimator", "source")},
        "n_atomic_rows": am["num_atomic_rows"], "n_atomic_P": am["num_atomic_P"],
        "n_gamma_rows_total": gm["num_gamma_rows_total"], "n_gamma_valid": gm["num_gamma_valid"],
        "n_common": int(len(merged)),
        "num_unique_atomic_probabilities": unique_atomic_probabilities,
        "num_unique_gamma_values": unique_gamma_values,
        "fraction_atomic_P_with_valid_gamma": len(merged) / am["num_atomic_P"],
        "atomic_threshold": am["atomic_threshold"], "gamma_threshold": gm["gamma_threshold"],
        "min_atomic_probability": float(merged["atomic_probability"].min()),
        "max_atomic_probability": float(merged["atomic_probability"].max()),
        "mean_atomic_probability": float(merged["atomic_probability"].mean()),
        "mean_gamma": float(merged["gamma"].mean()), "median_gamma": float(merged["gamma"].median()),
        "unexplained_variance_fraction_cv": 1.0 - metrics["cv_r_squared"],
        "residual_sd_fraction_of_gamma_sd": metrics["sd_residual_gamma"] / metrics["sd_gamma"] if metrics["sd_gamma"] > 0 else np.nan,
        "spline_df": args.spline_df, "cv_folds": args.cv_folds, "cv_seed": useed,
        "atomic_path": str(atomic_path), "gamma_path": str(gamma_path), **metrics, **fit_metrics,
    }
    for i, (name, value) in enumerate((("model", model), ("dataset", dataset), ("probe", probe), ("estimator", estimator), ("source", source))):
        statements.insert(i, name, value); curve.insert(i, name, value)
    return summary, statements, curve, {**base, "status": "complete", "reason": None, "n_common": int(len(merged))}


def cross_model_agreement(statement_level: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "probe", "estimator", "source"]
    for key, group in statement_level.groupby(keys, sort=True):
        dataset, probe, estimator, source = key
        models = sorted(group["model"].unique())
        per = {m: group.loc[group["model"].eq(m), ["P_id", "atomic_probability", "gamma", "residual_gamma"]].copy() for m in models}
        for i, a_name in enumerate(models):
            a = per[a_name]; n_a = len(a)
            for b_name in models[i+1:]:
                b = per[b_name]; n_b = len(b)
                m = a.merge(b, on="P_id", suffixes=("_a", "_b"), validate="one_to_one")
                n = len(m)
                if n < min_overlap:
                    continue
                rows.append({
                    "dataset": dataset, "probe": probe, "estimator": estimator, "source": source,
                    "model_a": a_name, "model_b": b_name, "n_a": n_a, "n_b": n_b, "n_common": n,
                    "overlap_fraction_a": n/n_a, "overlap_fraction_b": n/n_b,
                    "overlap_jaccard": n/(n_a+n_b-n),
                    "atomic_probability_spearman_rho": spearman(m["atomic_probability_a"], m["atomic_probability_b"]),
                    "raw_gamma_spearman_rho": spearman(m["gamma_a"], m["gamma_b"]),
                    "residual_gamma_spearman_rho": spearman(m["residual_gamma_a"], m["residual_gamma_b"]),
                })
    return pd.DataFrame(rows)


def permutation_tests(statement_level: pd.DataFrame, pair_table: pd.DataFrame, min_overlap: int, n_perm: int, seed: int):
    if pair_table.empty or n_perm == 0:
        return pd.DataFrame(), pd.DataFrame()
    summaries, null_rows = [], []
    keys = ["dataset", "probe", "estimator", "source"]
    for key, pairs_obs in pair_table.groupby(keys, sort=True):
        dataset, probe, estimator, source = key
        group = statement_level.loc[
            statement_level["dataset"].eq(dataset) & statement_level["probe"].eq(probe)
            & statement_level["estimator"].eq(estimator) & statement_level["source"].eq(source)
        ]
        obs = pairs_obs["residual_gamma_spearman_rho"].dropna().to_numpy(float)
        if len(obs) == 0:
            continue
        observed = float(np.median(obs))
        model_data = {}
        for model, d in group.groupby("model", sort=True):
            d = d[["P_id", "residual_gamma"]].reset_index(drop=True)
            model_data[model] = {"ids": d["P_id"].astype(str).tolist(), "r": d["residual_gamma"].to_numpy(float)}
            model_data[model]["idx"] = {pid: i for i, pid in enumerate(model_data[model]["ids"])}
        pair_index = []
        names = sorted(model_data)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                common = sorted(set(model_data[a]["ids"]) & set(model_data[b]["ids"]))
                if len(common) < min_overlap:
                    continue
                ia = np.array([model_data[a]["idx"][p] for p in common], int)
                ib = np.array([model_data[b]["idx"][p] for p in common], int)
                pair_index.append((a, b, ia, ib))
        if len(pair_index) != len(pairs_obs):
            raise RuntimeError(
                "Permutation pair construction does not match observed "
                f"pair table for {key}: observed={len(pairs_obs)}, "
                f"permutation_pairs={len(pair_index)}."
            )

        gseed = stable_seed(
            seed,
            "permutation",
            *map(str, key),
        )
        rng = np.random.default_rng(gseed)
        null = np.full(n_perm, np.nan)
        for p in range(n_perm):
            perm = {m: rng.permutation(d["r"]) for m, d in model_data.items()}
            rhos = [spearman(perm[a][ia], perm[b][ib]) for a, b, ia, ib in pair_index]
            rhos = np.asarray([x for x in rhos if np.isfinite(x)], float)
            if len(rhos):
                null[p] = float(np.median(rhos))
        valid = null[np.isfinite(null)]
        if len(valid):
            q025, q50, q975 = np.quantile(
                valid,
                [.025, .5, .975],
            )
            exceed_greater = int(
                np.sum(valid >= observed)
            )
            exceed_two_sided = int(
                np.sum(np.abs(valid) >= abs(observed))
            )

            # Standard +1 correction for Monte Carlo permutation p-values.
            p_greater = (
                1 + exceed_greater
            ) / (
                len(valid) + 1
            )
            p_two = (
                1 + exceed_two_sided
            ) / (
                len(valid) + 1
            )

            null_mean = float(np.mean(valid))
            null_sd = (
                float(np.std(valid, ddof=1))
                if len(valid) > 1
                else np.nan
            )
            observed_null_z = (
                (observed - null_mean) / null_sd
                if np.isfinite(null_sd) and null_sd > 0
                else np.nan
            )
        else:
            q025 = q50 = q975 = np.nan
            p_greater = p_two = np.nan
            exceed_greater = 0
            exceed_two_sided = 0
            null_mean = np.nan
            null_sd = np.nan
            observed_null_z = np.nan
        summaries.append({
            "dataset": dataset, "probe": probe, "estimator": estimator, "source": source,
            "n_models": int(group["model"].nunique()), "n_model_pairs": int(len(pairs_obs)),
            "min_overlap": min_overlap, "observed_median_residual_spearman_rho": observed,
            "observed_mean_residual_spearman_rho": float(np.mean(obs)),
            "observed_fraction_positive_pairwise_rho": float(np.mean(obs > 0)),
            "n_permutations_requested": n_perm,
            "n_permutations_valid": int(len(valid)),
            "permutation_seed": gseed,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "null_q025": q025,
            "null_median": q50,
            "null_q975": q975,
            "observed_minus_null_median": (
                observed - q50
                if np.isfinite(q50)
                else np.nan
            ),
            "observed_null_z": observed_null_z,
            "n_null_ge_observed": exceed_greater,
            "n_null_abs_ge_observed_abs": exceed_two_sided,
            "p_greater": p_greater,
            "p_two_sided_abs": p_two,
        })
        null_rows.extend({
            "dataset": dataset, "probe": probe, "estimator": estimator, "source": source,
            "permutation": i, "median_residual_spearman_rho": float(v) if np.isfinite(v) else np.nan,
        } for i, v in enumerate(null))
    return pd.DataFrame(summaries), pd.DataFrame(null_rows)


def proposition_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["dataset", "probe", "estimator", "source", "P_id"], sort=True).agg(
        n_models=("model", "nunique"),
        mean_atomic_probability=("atomic_probability", "mean"),
        median_atomic_probability=("atomic_probability", "median"),
        mean_gamma=("gamma", "mean"), median_gamma=("gamma", "median"),
        mean_residual_gamma=("residual_gamma", "mean"),
        median_residual_gamma=("residual_gamma", "median"),
        sd_residual_gamma=("residual_gamma", "std"),
        mean_absolute_residual_gamma=("absolute_residual_gamma", "mean"),
        fraction_positive_residual=("residual_gamma", lambda x: float(np.mean(np.asarray(x,float)>0))),
        fraction_negative_residual=("residual_gamma", lambda x: float(np.mean(np.asarray(x,float)<0))),
    ).reset_index()


def compact_summary(model_summary: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (probe, estimator, source), base in model_summary.groupby(["probe", "estimator", "source"], sort=True):
        for dataset in [*DATASETS, "overall"]:
            d = base if dataset == "overall" else base.loc[base["dataset"].eq(dataset)]
            if d.empty:
                continue
            x = d["cv_r_squared"].dropna().to_numpy(float)
            q1, med, q3 = np.quantile(x, [.25,.5,.75]) if len(x) else (np.nan,)*3
            row = {
                "dataset": dataset, "probe": probe, "estimator": estimator, "source": source,
                "n_model_dataset_units": len(d), "mean_cv_r_squared": float(np.mean(x)) if len(x) else np.nan,
                "median_cv_r_squared": med, "q1_cv_r_squared": q1, "q3_cv_r_squared": q3,
                "fraction_cv_r_squared_positive": float(np.mean(x>0)) if len(x) else np.nan,
                "median_spearman_atomic_gamma": float(d["spearman_atomic_gamma"].median()),
                "median_mean_absolute_residual_gamma": float(d["mean_absolute_residual_gamma"].median()),
                "median_fraction_atomic_P_with_valid_gamma": float(d["fraction_atomic_P_with_valid_gamma"].median()),
            }
            if dataset != "overall" and not perm.empty:
                p = perm.loc[perm["dataset"].eq(dataset)&perm["probe"].eq(probe)&perm["estimator"].eq(estimator)&perm["source"].eq(source)]
                if len(p)==1:
                    r=p.iloc[0]; row.update({
                        "n_cross_model_pairs": int(r["n_model_pairs"]),
                        "median_cross_model_residual_rho": float(r["observed_median_residual_spearman_rho"]),
                        "cross_model_permutation_p_greater": float(r["p_greater"]),
                        "cross_model_null_q025": float(r["null_q025"]), "cross_model_null_q975": float(r["null_q975"]),
                    })
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(); root = args.repo_root.resolve()
    model_list = under_root(root, args.model_list); out = under_root(root, args.output_dir)
    models = load_model_list(model_list)
    if args.model:
        unknown = sorted(set(args.model)-set(models))
        if unknown: raise ValueError(f"Unknown requested models: {unknown}")
        models = list(dict.fromkeys(args.model))
    probes=list(dict.fromkeys(args.probe)); datasets=list(dict.fromkeys(args.dataset)); estimators=list(dict.fromkeys(args.estimator))
    paths = {
        "model": out/"model_summary.parquet", "statements": out/"statement_level.parquet",
        "curves": out/"fit_curves.parquet", "cross": out/"cross_model_agreement.parquet",
        "perm": out/"permutation_summary.parquet", "null": out/"permutation_null.parquet",
        "props": out/"proposition_summary.parquet", "status": out/"analysis_status.parquet",
        "compact": out/"summary_compact.csv", "config": out/"analysis_config.json",
    }
    existing=[p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n  "+"\n  ".join(map(str,existing)))
    out.mkdir(parents=True, exist_ok=True)
    print("="*100); print("STABILITY VS. ATOMIC CREDENCE")
    print(f"Models={len(models)} datasets={datasets} probes={probes} estimators={estimators}")
    print(f"spline_df={args.spline_df} cv={args.cv_folds} min_fit={args.min_fit_statements} min_overlap={args.min_overlap} permutations={args.n_permutations}")
    print("="*100)
    summaries=[]; statements=[]; curves=[]; statuses=[]
    for model in models:
        for dataset in datasets:
            for probe in probes:
                for estimator in estimators:
                    try:
                        s, st, c, status = analyze_unit(root, model, dataset, probe, estimator, args)
                    except Exception as exc:
                        status={"model":model,"dataset":dataset,"probe":probe,"estimator":estimator,"source":ESTIMATORS[estimator],"status":"error","reason":type(exc).__name__,"message":str(exc)}
                        statuses.append(status); print(f"ERROR {model} {dataset} {probe} {estimator}: {exc}"); continue
                    statuses.append(status)
                    if status["status"]!="complete":
                        detail = (
                            f" | {status['message']}"
                            if status.get("message")
                            else ""
                        )
                        print(
                            f"SKIP  {model} {dataset} {probe} {estimator}: "
                            f"{status['reason']}{detail}"
                        )
                        continue
                    summaries.append(s); statements.append(st); curves.append(c)
                    print(f"OK    {model:24s} {dataset:18s} {probe:16s} {estimator:6s} n={s['n_common']:4d} CV_R2={s['cv_r_squared']:+.4f}")
    status_df=pd.DataFrame(statuses)
    if not summaries:
        write_parquet_atomic(status_df, paths["status"]); raise RuntimeError("No analysis units completed")
    model_df=pd.DataFrame(summaries); statement_df=pd.concat(statements,ignore_index=True,sort=False); curve_df=pd.concat(curves,ignore_index=True,sort=False)
    cross_df=cross_model_agreement(statement_df,args.min_overlap)
    perm_df,null_df=permutation_tests(statement_df,cross_df,args.min_overlap,args.n_permutations,args.seed)
    prop_df=proposition_summary(statement_df); compact=compact_summary(model_df,perm_df)
    # deterministic ordering
    for df, cols in ((model_df,["probe","dataset","estimator","model"]),(statement_df,["probe","dataset","estimator","model","P_id"]),(curve_df,["probe","dataset","estimator","model","atomic_probability"]),(cross_df,["probe","dataset","estimator","model_a","model_b"]),(perm_df,["probe","dataset","estimator"]),(null_df,["probe","dataset","estimator","permutation"]),(prop_df,["probe","dataset","estimator","P_id"])):
        if not df.empty: df.sort_values(cols,kind="stable",inplace=True,ignore_index=True)
    for df,p in ((model_df,paths["model"]),(statement_df,paths["statements"]),(curve_df,paths["curves"]),(cross_df,paths["cross"]),(perm_df,paths["perm"]),(null_df,paths["null"]),(prop_df,paths["props"]),(status_df,paths["status"])):
        write_parquet_atomic(df,p)
    write_csv_atomic(compact,paths["compact"])
    write_json_atomic({
        "schema_version":1,"analysis":"stability_vs_atomic_credence",
        "research_question":"Among a model's believed propositions, how much variation in graded stability is explained by variation in atomic credence?",
        "models":models,"datasets":datasets,"probes":probes,"estimators":estimators,
        "spline":{"type":"natural_cubic","effect_df":args.spline_df,"constraints":"center","lower_bound":0.0,"upper_bound":1.0},
        "cross_validation":{"folds":args.cv_folds,"seed":args.seed,"minimum_fit_statements":args.min_fit_statements,"predictions_clipped":False},
        "cross_model":{"minimum_overlap":args.min_overlap,"statistic":"spearman","primary_summary":"median_pairwise_residual_rho"},
        "permutation":{
            "n":args.n_permutations,
            "shuffle":"residuals across P_id independently within model",
            "overlap_structure_preserved":True,
            "pair_set_fixed_to_observed_eligible_pairs":True,
            "summary_statistic":"median pairwise residual Spearman rho",
            "primary_alternative":"greater",
            "primary_p":"p_greater",
            "monte_carlo_plus_one_correction":True,
        },
    },paths["config"])
    complete=int((status_df["status"]=="complete").sum()); skipped=int((status_df["status"]=="skipped").sum()); errors=int((status_df["status"]=="error").sum())
    print("\n"+"="*100); print(f"Complete={complete} skipped={skipped} errors={errors} statement_rows={len(statement_df)} cross_model_pairs={len(cross_df)}")
    if not compact.empty:
        d=compact.copy(); nums=d.select_dtypes(include=[np.number]).columns; d[nums]=d[nums].round(4); print(d.to_string(index=False))
    print(f"Saved to {out}"); print("="*100)
    if errors:
        raise RuntimeError(f"{errors} analysis units failed; inspect {paths['status']}")


if __name__ == "__main__":
    main()
