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

from stability.utils.io import write_parquet_atomic

DATASETS = ("cities_loc", "med_indications", "defs")
PROBES = ("sawmil", "svm", "mean_difference")
DIRECT_SOURCE = "conditional"
JOINT_SOURCE = "joint"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct-vs-Joint graded-stability agreement.")
    p.add_argument("--repo_root", type=Path, default=Path("."))
    p.add_argument("--model_list", type=Path, default=Path("configs/model_list.yaml"))
    p.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--probe", nargs="+", choices=PROBES, default=["sawmil"])
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--min_paired_statements", type=int, default=50)
    p.add_argument("--min_models", type=int, default=3)
    p.add_argument("--n_permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--analysis1_dir", type=Path,
        default=Path("outputs/analysis/stability_vs_credence"),
    )
    p.add_argument(
        "--output_dir", type=Path,
        default=Path("outputs/analysis/operationalization_agreement"),
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.min_paired_statements < 3:
        p.error("--min_paired_statements must be >= 3")
    if a.min_models < 2:
        p.error("--min_models must be >= 2")
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


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


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


def rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, float)).rank(method="average").to_numpy(float)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y); x = x[keep]; y = y[keep]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y); x = x[keep]; y = y[keep]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return pearson(rank_average(x), rank_average(y))


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient using population moments."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y); x = x[keep]; y = y[keep]
    if len(x) < 2:
        return np.nan
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    return np.nan if denom <= 0 else 2.0 * cov / denom


def agreement_metrics(direct: np.ndarray, joint: np.ndarray) -> dict[str, float]:
    direct = np.asarray(direct, float); joint = np.asarray(joint, float)
    keep = np.isfinite(direct) & np.isfinite(joint); direct = direct[keep]; joint = joint[keep]
    if len(direct) == 0:
        return {k: np.nan for k in (
            "spearman_rho", "pearson_r", "ccc", "mae", "median_absolute_difference",
            "rmse", "mean_signed_difference", "median_signed_difference"
        )} | {"n": 0}
    diff = direct - joint; adiff = np.abs(diff)
    return {
        "n": int(len(direct)),
        "spearman_rho": spearman(direct, joint),
        "pearson_r": pearson(direct, joint),
        "ccc": ccc(direct, joint),
        "mae": float(adiff.mean()),
        "median_absolute_difference": float(np.median(adiff)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_signed_difference": float(diff.mean()),
        "median_signed_difference": float(np.median(diff)),
    }


def load_gamma_pair(path: Path, probe: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = {"probe", "P_id", "source", "gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing gamma columns {sorted(missing)}")
    df = df.loc[df["probe"].astype(str).eq(probe)].copy()
    if df.empty:
        return pd.DataFrame(), {"reason": "probe_absent_from_gamma", "n_direct": 0, "n_joint": 0}
    direct = df.loc[df["source"].astype(str).eq(DIRECT_SOURCE)].copy()
    joint = df.loc[df["source"].astype(str).eq(JOINT_SOURCE)].copy()
    if direct.empty:
        return pd.DataFrame(), {"reason": "direct_absent_from_gamma", "n_direct": 0, "n_joint": int(len(joint))}
    if joint.empty:
        return pd.DataFrame(), {"reason": "joint_absent_from_gamma", "n_direct": int(len(direct)), "n_joint": 0}
    for name, sub in (("direct", direct), ("joint", joint)):
        sub["P_id"] = sub["P_id"].astype(str)
        if sub["P_id"].duplicated().any():
            raise ValueError(f"{path}/{probe}/{name}: duplicate P_id values")
        sub["gamma"] = pd.to_numeric(sub["gamma"], errors="coerce")
        bad = ~np.isfinite(sub["gamma"]) | ~sub["gamma"].between(0, 1)
        if bad.any():
            raise ValueError(f"{path}/{probe}/{name}: {int(bad.sum())} invalid gamma values")
    direct_ids, joint_ids = set(direct["P_id"]), set(joint["P_id"])
    if direct_ids != joint_ids:
        raise ValueError(
            f"{path}/{probe}: Direct and Joint P sets differ; "
            f"only_direct={sorted(direct_ids-joint_ids)[:10]}, "
            f"only_joint={sorted(joint_ids-direct_ids)[:10]}"
        )
    extras = ("num_rows", "denominator", "num_undefined", "threshold", "numerator")
    dk = ["P_id", "gamma"] + [c for c in extras if c in direct]
    jk = ["P_id", "gamma"] + [c for c in extras if c in joint]
    direct = direct[dk].rename(columns={c: f"{c}_direct" for c in dk if c != "P_id"})
    joint = joint[jk].rename(columns={c: f"{c}_joint" for c in jk if c != "P_id"})
    paired = direct.merge(joint, on="P_id", how="inner", validate="one_to_one")
    if {"num_rows_direct", "num_rows_joint"}.issubset(paired.columns):
        mismatch = pd.to_numeric(paired["num_rows_direct"], errors="coerce") != pd.to_numeric(paired["num_rows_joint"], errors="coerce")
        if mismatch.any():
            raise ValueError(f"{path}/{probe}: underlying candidate-x set sizes differ for {int(mismatch.sum())} P statements")
    paired["gamma_difference"] = paired["gamma_direct"] - paired["gamma_joint"]
    paired["absolute_gamma_difference"] = paired["gamma_difference"].abs()
    for suffix in ("direct", "joint"):
        denom, total = f"denominator_{suffix}", f"num_rows_{suffix}"
        if {denom, total}.issubset(paired.columns):
            paired[f"defined_fraction_{suffix}"] = pd.to_numeric(paired[denom], errors="coerce") / pd.to_numeric(paired[total], errors="coerce")
    return paired, {"reason": None, "n_direct": int(len(direct)), "n_joint": int(len(joint)), "n_paired": int(len(paired))}


def load_residual_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Analysis-1 statement-level output not found: {path}")
    df = pd.read_parquet(path)
    required = {"model", "dataset", "probe", "estimator", "P_id", "residual_gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[["model", "dataset", "probe", "estimator", "P_id", "residual_gamma"]].copy()
    df["P_id"] = df["P_id"].astype(str)
    df["residual_gamma"] = pd.to_numeric(df["residual_gamma"], errors="coerce")
    return df


def residual_pair_for_unit(residuals: pd.DataFrame, model: str, dataset: str, probe: str, expected: set[str]) -> tuple[pd.DataFrame, str | None]:
    d = residuals.loc[
        residuals["model"].astype(str).eq(model)
        & residuals["dataset"].astype(str).eq(dataset)
        & residuals["probe"].astype(str).eq(probe)
    ].copy()
    if d.empty:
        return pd.DataFrame(), "analysis1_unit_absent"
    a = d.loc[d["estimator"].astype(str).eq("direct"), ["P_id", "residual_gamma"]].copy()
    b = d.loc[d["estimator"].astype(str).eq("joint"), ["P_id", "residual_gamma"]].copy()
    if a.empty or b.empty:
        return pd.DataFrame(), "analysis1_estimator_absent"
    if a["P_id"].duplicated().any() or b["P_id"].duplicated().any():
        raise ValueError(f"{model}/{dataset}/{probe}: duplicate residual P_id")
    if set(a["P_id"]) != set(b["P_id"]):
        return pd.DataFrame(), "analysis1_direct_joint_P_mismatch"
    if set(a["P_id"]) != expected:
        return pd.DataFrame(), "analysis1_raw_P_mismatch"
    out = a.rename(columns={"residual_gamma": "residual_gamma_direct"}).merge(
        b.rename(columns={"residual_gamma": "residual_gamma_joint"}),
        on="P_id", how="inner", validate="one_to_one"
    )
    out["residual_difference"] = out["residual_gamma_direct"] - out["residual_gamma_joint"]
    out["absolute_residual_difference"] = out["residual_difference"].abs()
    return out, None


def analyze_unit(root: Path, residuals: pd.DataFrame, model: str, dataset: str, probe: str, min_paired: int):
    path = root / "outputs" / "gamma" / model / f"{dataset}.parquet"
    paired, meta = load_gamma_pair(path, probe)
    base = {"model": model, "dataset": dataset, "probe": probe, "gamma_path": str(path), **meta}
    if paired.empty:
        return None, None, {**base, "status": "skipped", "reason": meta["reason"] or "empty_pair"}
    if len(paired) < min_paired:
        return None, None, {**base, "status": "skipped", "reason": "insufficient_paired_P"}
    raw = agreement_metrics(paired["gamma_direct"], paired["gamma_joint"])
    rp, reason = residual_pair_for_unit(residuals, model, dataset, probe, set(paired["P_id"].astype(str)))
    if reason is None:
        residual = agreement_metrics(rp["residual_gamma_direct"], rp["residual_gamma_joint"])
        paired = paired.merge(rp, on="P_id", how="left", validate="one_to_one")
        residual_available = True
    else:
        residual = {k: np.nan for k in (
            "spearman_rho", "pearson_r", "ccc", "mae", "median_absolute_difference",
            "rmse", "mean_signed_difference", "median_signed_difference"
        )} | {"n": 0}
        residual_available = False
    summary = {
        "model": model, "dataset": dataset, "probe": probe, "n_paired": int(len(paired)),
        **{f"raw_{k}": v for k, v in raw.items() if k != "n"},
        "residual_available": residual_available, "residual_reason": reason,
        "n_residual_paired": int(residual["n"]),
        **{f"residual_{k}": v for k, v in residual.items() if k != "n"},
        "gamma_path": str(path),
    }
    for col in ("defined_fraction_direct", "defined_fraction_joint"):
        if col in paired:
            values = pd.to_numeric(paired[col], errors="coerce")
            summary[f"mean_{col}"] = float(values.mean())
            summary[f"min_{col}"] = float(values.min())
    paired.insert(0, "model", model); paired.insert(1, "dataset", dataset); paired.insert(2, "probe", probe)
    return summary, paired, {
        **base, "status": "complete", "reason": None,
        "residual_status": "complete" if residual_available else "skipped",
        "residual_reason": reason,
    }


def prepare_units(group: pd.DataFrame, value_type: str) -> dict[str, dict[str, np.ndarray]]:
    units = {}
    dc, jc = (("gamma_direct", "gamma_joint") if value_type == "raw" else ("residual_gamma_direct", "residual_gamma_joint"))
    for model, d in group.groupby("model", sort=True):
        if dc not in d or jc not in d:
            continue
        x = pd.to_numeric(d[dc], errors="coerce").to_numpy(float)
        y = pd.to_numeric(d[jc], errors="coerce").to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y); x = x[keep]; y = y[keep]
        if len(x) < 3:
            continue
        units[str(model)] = {"direct": x, "joint": y, "direct_rank": rank_average(x), "joint_rank": rank_average(y)}
    return units


def permutation_tests(statement_df: pd.DataFrame, probes: list[str], datasets: list[str], min_models: int, n_perm: int, seed: int):
    if n_perm == 0:
        return pd.DataFrame(), pd.DataFrame()
    summaries, null_rows = [], []
    for dataset in datasets:
        for probe in probes:
            g = statement_df.loc[statement_df["dataset"].eq(dataset) & statement_df["probe"].eq(probe)].copy()
            if g.empty:
                continue
            for value_type in ("raw", "residual"):
                units = prepare_units(g, value_type)
                if len(units) < min_models:
                    continue
                obs_rho = np.asarray([pearson(d["direct_rank"], d["joint_rank"]) for d in units.values()], float)
                obs_mae = np.asarray([np.mean(np.abs(d["direct"] - d["joint"])) for d in units.values()], float)
                obs_rho = obs_rho[np.isfinite(obs_rho)]; obs_mae = obs_mae[np.isfinite(obs_mae)]
                if len(obs_rho) < min_models or len(obs_mae) < min_models:
                    continue
                med_rho, med_mae = float(np.median(obs_rho)), float(np.median(obs_mae))
                gseed = stable_seed(seed, "operationalization_permutation", dataset, probe, value_type)
                rng = np.random.default_rng(gseed)
                null_rho = np.full(n_perm, np.nan); null_mae = np.full(n_perm, np.nan)
                for p in range(n_perm):
                    pr, pm = [], []
                    for d in units.values():
                        idx = rng.permutation(len(d["joint"]))
                        pr.append(pearson(d["direct_rank"], d["joint_rank"][idx]))
                        pm.append(float(np.mean(np.abs(d["direct"] - d["joint"][idx]))))
                    pr = np.asarray([x for x in pr if np.isfinite(x)], float)
                    pm = np.asarray([x for x in pm if np.isfinite(x)], float)
                    if len(pr): null_rho[p] = float(np.median(pr))
                    if len(pm): null_mae[p] = float(np.median(pm))
                vr, vm = null_rho[np.isfinite(null_rho)], null_mae[np.isfinite(null_mae)]
                rq = np.quantile(vr, [.025, .5, .975]) if len(vr) else [np.nan]*3
                mq = np.quantile(vm, [.025, .5, .975]) if len(vm) else [np.nan]*3
                nr = int(np.sum(vr >= med_rho)); nm = int(np.sum(vm <= med_mae))
                summaries.append({
                    "dataset": dataset, "probe": probe, "value_type": value_type,
                    "n_models": int(len(units)),
                    "observed_median_spearman_rho": med_rho,
                    "observed_mean_spearman_rho": float(np.mean(obs_rho)),
                    "observed_fraction_positive_rho": float(np.mean(obs_rho > 0)),
                    "observed_median_mae": med_mae,
                    "observed_mean_mae": float(np.mean(obs_mae)),
                    "n_permutations_requested": n_perm, "permutation_seed": gseed,
                    "null_rho_q025": float(rq[0]), "null_rho_median": float(rq[1]), "null_rho_q975": float(rq[2]),
                    "n_null_rho_ge_observed": nr, "p_rho_greater": (1+nr)/(len(vr)+1) if len(vr) else np.nan,
                    "null_mae_q025": float(mq[0]), "null_mae_median": float(mq[1]), "null_mae_q975": float(mq[2]),
                    "n_null_mae_le_observed": nm, "p_mae_less": (1+nm)/(len(vm)+1) if len(vm) else np.nan,
                })
                for p in range(n_perm):
                    null_rows.append({
                        "dataset": dataset, "probe": probe, "value_type": value_type, "permutation": p,
                        "median_spearman_rho": float(null_rho[p]) if np.isfinite(null_rho[p]) else np.nan,
                        "median_mae": float(null_mae[p]) if np.isfinite(null_mae[p]) else np.nan,
                    })
    return pd.DataFrame(summaries), pd.DataFrame(null_rows)


def compact_summary(model_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (probe, dataset), g in model_df.groupby(["probe", "dataset"], sort=True):
        row = {
            "probe": probe, "dataset": dataset, "n_models": int(len(g)),
            "median_raw_spearman_rho": float(g["raw_spearman_rho"].median()),
            "q1_raw_spearman_rho": float(g["raw_spearman_rho"].quantile(.25)),
            "q3_raw_spearman_rho": float(g["raw_spearman_rho"].quantile(.75)),
            "median_raw_mae": float(g["raw_mae"].median()),
            "median_raw_ccc": float(g["raw_ccc"].median()),
            "median_raw_signed_difference": float(g["raw_mean_signed_difference"].median()),
            "n_models_with_residuals": int(g["residual_available"].sum()),
            "median_residual_spearman_rho": float(g["residual_spearman_rho"].median()),
            "median_residual_mae": float(g["residual_mae"].median()),
        }
        for vt in ("raw", "residual"):
            p = perm_df.loc[perm_df["dataset"].eq(dataset)&perm_df["probe"].eq(probe)&perm_df["value_type"].eq(vt)] if not perm_df.empty else pd.DataFrame()
            if len(p) == 1:
                r = p.iloc[0]
                row.update({
                    f"{vt}_permutation_p_rho_greater": float(r["p_rho_greater"]),
                    f"{vt}_permutation_p_mae_less": float(r["p_mae_less"]),
                    f"{vt}_null_rho_median": float(r["null_rho_median"]),
                    f"{vt}_null_mae_median": float(r["null_mae_median"]),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(); root = args.repo_root.resolve()
    model_list = under_root(root, args.model_list); out = under_root(root, args.output_dir)
    analysis1_dir = under_root(root, args.analysis1_dir)
    models = load_model_list(model_list)
    if args.model:
        unknown = sorted(set(args.model)-set(models))
        if unknown: raise ValueError(f"Unknown requested models: {unknown}")
        models = list(dict.fromkeys(args.model))
    datasets = list(dict.fromkeys(args.dataset)); probes = list(dict.fromkeys(args.probe))
    residual_path = analysis1_dir / "statement_level.parquet"
    residuals = load_residual_table(residual_path)
    paths = {
        "model": out/"model_summary.parquet", "statements": out/"statement_level.parquet",
        "perm": out/"permutation_summary.parquet", "null": out/"permutation_null.parquet",
        "status": out/"analysis_status.parquet", "compact": out/"summary_compact.csv",
        "config": out/"analysis_config.json",
    }
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n  "+"\n  ".join(map(str, existing)))
    out.mkdir(parents=True, exist_ok=True)
    print("="*100); print("DIRECT VS. JOINT OPERATIONALIZATION AGREEMENT")
    print(f"Models={len(models)} datasets={datasets} probes={probes}")
    print(f"min_paired={args.min_paired_statements} min_models={args.min_models} permutations={args.n_permutations}")
    print(f"Analysis-1 residuals: {residual_path}"); print("="*100)
    summaries, statements, statuses = [], [], []
    for model in models:
        for dataset in datasets:
            for probe in probes:
                try:
                    s, st, status = analyze_unit(root, residuals, model, dataset, probe, args.min_paired_statements)
                except Exception as exc:
                    status = {"model": model, "dataset": dataset, "probe": probe, "status": "error", "reason": type(exc).__name__, "message": str(exc)}
                    statuses.append(status); print(f"ERROR {model} {dataset} {probe}: {exc}"); continue
                statuses.append(status)
                if status["status"] != "complete":
                    print(f"SKIP  {model} {dataset} {probe}: {status['reason']}"); continue
                summaries.append(s); statements.append(st)
                rt = f" residual_rho={s['residual_spearman_rho']:+.3f}" if s["residual_available"] else " residual=NA"
                print(f"OK    {model:24s} {dataset:18s} {probe:16s} n={s['n_paired']:4d} | rho={s['raw_spearman_rho']:+.3f} | MAE={s['raw_mae']:.3f} |{rt}")
    status_df = pd.DataFrame(statuses)
    if not summaries:
        write_parquet_atomic(status_df, paths["status"]); raise RuntimeError("No analysis units completed")
    model_df = pd.DataFrame(summaries); statement_df = pd.concat(statements, ignore_index=True, sort=False)
    perm_df, null_df = permutation_tests(statement_df, probes, datasets, args.min_models, args.n_permutations, args.seed)
    compact = compact_summary(model_df, perm_df)
    for df, cols in ((model_df,["probe","dataset","model"]),(statement_df,["probe","dataset","model","P_id"]),(status_df,["probe","dataset","model"]),(perm_df,["probe","dataset","value_type"]),(null_df,["probe","dataset","value_type","permutation"]),(compact,["probe","dataset"])):
        if not df.empty: df.sort_values(cols, kind="stable", inplace=True, ignore_index=True)
    for df, path in ((model_df,paths["model"]),(statement_df,paths["statements"]),(perm_df,paths["perm"]),(null_df,paths["null"]),(status_df,paths["status"])):
        write_parquet_atomic(df, path)
    write_csv_atomic(compact, paths["compact"])
    write_json_atomic({
        "schema_version": 1, "analysis": "operationalization_agreement",
        "research_question": "Do Direct and Joint recover the same underlying structure in graded belief stability?",
        "models": models, "datasets": datasets, "probes": probes,
        "pairing": {"unit": "model x dataset x probe", "require_identical_P_sets": True, "require_identical_candidate_x_set_size": True, "minimum_paired_P": args.min_paired_statements},
        "raw_agreement": {"primary_rank_metric": "Spearman rho", "primary_absolute_metric": "MAE", "secondary_metrics": ["Pearson r", "Lin CCC", "RMSE", "median absolute difference", "mean signed difference (Direct - Joint)", "median signed difference (Direct - Joint)"]},
        "residual_agreement": {"source": str(residual_path), "residuals": "cross-fitted residual gamma from stability-vs-atomic-credence analysis", "primary_rank_metric": "Spearman rho", "primary_absolute_metric": "MAE"},
        "permutation": {"n": args.n_permutations, "seed": args.seed, "minimum_models": args.min_models, "shuffle": "Joint values across P_id independently within each model; Direct fixed", "marginal_distributions_preserved": True, "P_sets_preserved": True, "aggregate_rank": "median per-model Spearman rho", "aggregate_absolute": "median per-model MAE", "rank_alternative": "greater", "absolute_alternative": "less", "plus_one_correction": True},
    }, paths["config"])
    complete = int((status_df["status"]=="complete").sum()); skipped = int((status_df["status"]=="skipped").sum()); errors = int((status_df["status"]=="error").sum())
    print("\n"+"="*100); print(f"Complete={complete} skipped={skipped} errors={errors} statement_rows={len(statement_df)}")
    if not compact.empty:
        d = compact.copy(); nums = d.select_dtypes(include=[np.number]).columns; d[nums] = d[nums].round(4); print(d.to_string(index=False))
    print(f"Saved to {out}"); print("="*100)
    if errors:
        raise RuntimeError(f"{errors} analysis units failed; inspect {paths['status']}")


if __name__ == "__main__":
    main()
