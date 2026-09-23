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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate behavioral-resilience analyses.")
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--model_list", type=Path, default=Path("configs/model_list.yaml"))
    parser.add_argument("--analysis_dir", type=Path, default=Path("outputs/behavior/analysis"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/analysis/behavioral_resilience")
    )
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--probe", nargs="+", action="extend", choices=PROBES, default=None
    )
    parser.add_argument("--n_permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.probe is None:
        args.probe = ["sawmil"]
    args.probe = list(dict.fromkeys(args.probe))
    args.dataset = list(dict.fromkeys(args.dataset))
    return args


def under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_model_list(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]
    if isinstance(raw, dict):
        models = list(map(str, raw.keys()))
    elif isinstance(raw, list):
        models = []
        for item in raw:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                value = next(
                    (item[k] for k in ("name", "config", "model_name", "key") if k in item),
                    None,
                )
                if value is None and len(item) == 1:
                    value = next(iter(item))
                if value is None:
                    raise ValueError(f"Cannot infer model key from {item!r}.")
                models.append(str(value))
            else:
                raise ValueError(f"Unsupported model-list entry {item!r}.")
    else:
        raise ValueError(f"Unsupported model-list format: {path}")
    return list(dict.fromkeys(models))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join([str(seed), *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def sign_flip_test(
    values: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    max_exact_patterns: int = 65536,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p": np.nan, "mode": "not_run", "n": 0, "observed": np.nan}
    observed = float(np.median(values))
    n = len(values)
    n_possible = 2 ** n
    if n_possible <= max_exact_patterns:
        ids = np.arange(n_possible, dtype=np.uint64)[:, None]
        bits = (ids >> np.arange(n, dtype=np.uint64)[None, :]) & 1
        signs = np.where(bits == 1, 1.0, -1.0)
        null = np.median(signs * values[None, :], axis=1)
        p = float(np.mean(np.abs(null) >= abs(observed)))
        mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n)
            null[i] = np.median(values * signs)
        p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1))
        mode = "monte_carlo"
    return {"p": p, "mode": mode, "n": int(n), "observed": observed}


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def collect_units(
    *,
    analysis_dir: Path,
    models: list[str],
    datasets: list[str],
    probes: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    buckets: dict[str, list[pd.DataFrame]] = {
        "cv_deltas": [],
        "cv_summary": [],
        "matched_summary": [],
        "matched_results": [],
        "matched_sensitivity": [],
        "association_summary": [],
        "sequence_effect_tests": [],
    }
    status_rows: list[dict[str, Any]] = []

    for model in models:
        for dataset in datasets:
            root = analysis_dir / model / dataset
            summary_path = root / "summary.json"
            if not summary_path.exists():
                status_rows.append(
                    {"model": model, "dataset": dataset, "status": "missing", "reason": "no_summary"}
                )
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            status = str(payload.get("status", "unknown"))
            status_rows.append(
                {"model": model, "dataset": dataset, "status": status, "reason": None}
            )
            if status != "complete":
                continue

            for name in buckets:
                frame = _read_optional(root / f"{name}.parquet")
                if frame.empty:
                    continue
                if "probe" in frame.columns:
                    frame = frame.loc[frame["probe"].astype(str).isin(probes)].copy()
                if frame.empty:
                    continue
                frame.insert(0, "dataset", dataset)
                frame.insert(0, "model_name", model)
                buckets[name].append(frame)

    combined = {
        name: (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame())
        for name, frames in buckets.items()
    }
    return combined, pd.DataFrame(status_rows)


def summarize_cv_units(
    cv_deltas: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cv_deltas.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    delta_columns = [
        c for c in ("delta_r2", "delta_rmse", "delta_log_loss", "delta_roc_auc")
        if c in cv_deltas.columns
    ]
    keys = ["model_name", "dataset", "probe", "sample", "estimator", "outcome"]
    unit = cv_deltas.groupby(keys, as_index=False, sort=True)[delta_columns].agg(
        ["mean", "median", "std"]
    )
    unit.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in unit.columns
    ]
    # Pandas multi-index aggregation also tuple-izes group keys; normalize names.
    unit = unit.rename(columns={f"{k}_": k for k in keys})

    metric_means = [c for c in unit.columns if c.startswith("delta_") and c.endswith("_mean")]
    model_keys = ["model_name", "probe", "sample", "estimator", "outcome"]
    model = unit.groupby(model_keys, as_index=False, sort=True)[metric_means].mean()
    model = model.rename(columns={c: c.replace("_mean", "_model_domain_mean") for c in metric_means})

    rows: list[dict[str, Any]] = []
    for (probe, sample, estimator, outcome), group in model.groupby(
        ["probe", "sample", "estimator", "outcome"], sort=True
    ):
        for column in [c for c in model.columns if c.endswith("_model_domain_mean")]:
            values = group[column].to_numpy(dtype=float)
            test = sign_flip_test(
                values,
                n_permutations=n_permutations,
                seed=stable_seed(seed, "cv", probe, sample, estimator, outcome, column),
            )
            
            values = np.asarray(values, dtype=float)
            valid_values = values[np.isfinite(values)]
            
            if len(valid_values) == 0:
                median_effect = np.nan
                mean_effect = np.nan
            else:
                median_effect = float(np.median(valid_values))
                mean_effect = float(np.mean(valid_values))
                
            rows.append(
                {
                    "probe": probe,
                    "sample": sample,
                    "estimator": estimator,
                    "outcome": outcome,
                    "metric": column.replace("_model_domain_mean", ""),
                    "n_models": int(np.isfinite(values).sum()),
                    "median_model_effect": median_effect,
                    "mean_model_effect": mean_effect,
                    "fraction_models_positive": float(np.nanmean(values > 0)),
                    "sign_flip_p_two_sided": test["p"],
                    "sign_flip_mode": test["mode"],
                }
            )
    return unit, model, pd.DataFrame(rows)


def summarize_matched_units(
    matched_summary: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matched_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    effect_cols = [
        c for c in (
            "mean_resilience_difference",
            "mean_movement_difference",
            "success_rate",
            "movement_success_rate",
            "flip_success_rate",
        ) if c in matched_summary.columns
    ]
    model_keys = ["model_name", "probe", "sample", "estimator"]
    model = matched_summary.groupby(model_keys, as_index=False, sort=True)[effect_cols].mean()

    rows: list[dict[str, Any]] = []
    for (probe, sample, estimator), group in model.groupby(
        ["probe", "sample", "estimator"], sort=True
    ):
        for column in ("mean_resilience_difference", "mean_movement_difference"):
            if column not in group:
                continue
            values = group[column].to_numpy(dtype=float)
            test = sign_flip_test(
                values,
                n_permutations=n_permutations,
                seed=stable_seed(seed, "matched", probe, sample, estimator, column),
            )
            rows.append(
                {
                    "probe": probe,
                    "sample": sample,
                    "estimator": estimator,
                    "metric": column,
                    "n_models": int(np.isfinite(values).sum()),
                    "median_model_effect": float(np.nanmedian(values)),
                    "mean_model_effect": float(np.nanmean(values)),
                    "fraction_models_positive": float(np.nanmean(values > 0)),
                    "sign_flip_p_two_sided": test["p"],
                    "sign_flip_mode": test["mode"],
                }
            )
    return model, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    model_list_path = under_root(root, args.model_list)
    analysis_dir = under_root(root, args.analysis_dir)
    output_dir = under_root(root, args.output_dir)
    models = load_model_list(model_list_path)
    if args.model:
        requested = list(dict.fromkeys(args.model))
        unknown = sorted(set(requested) - set(models))
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")
        models = requested

    outputs = {
        "status": output_dir / "analysis_status.parquet",
        "cv_deltas_all": output_dir / "cv_deltas_all.parquet",
        "cv_unit_summary": output_dir / "cv_unit_summary.parquet",
        "cv_model_summary": output_dir / "cv_model_summary.parquet",
        "cv_global_summary": output_dir / "cv_global_summary.parquet",
        "matched_unit_summary": output_dir / "matched_unit_summary.parquet",
        "matched_model_summary": output_dir / "matched_model_summary.parquet",
        "matched_global_summary": output_dir / "matched_global_summary.parquet",
        "matched_results_all": output_dir / "matched_results_all.parquet",
        "matched_sensitivity_all": output_dir / "matched_sensitivity_all.parquet",
        "association_all": output_dir / "association_all.parquet",
        "sequence_tests_all": output_dir / "sequence_tests_all.parquet",
        "compact": output_dir / "summary_compact.csv",
        "config": output_dir / "analysis_config.json",
    }
    existing = [p for p in outputs.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Aggregate outputs exist; pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    combined, status = collect_units(
        analysis_dir=analysis_dir,
        models=models,
        datasets=args.dataset,
        probes=args.probe,
    )
    cv_unit, cv_model, cv_global = summarize_cv_units(
        combined["cv_deltas"],
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    matched_model, matched_global = summarize_matched_units(
        combined["matched_summary"],
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    write_parquet_atomic(status, outputs["status"])
    for name, frame in (
        ("cv_deltas_all", combined["cv_deltas"]),
        ("cv_unit_summary", cv_unit),
        ("cv_model_summary", cv_model),
        ("cv_global_summary", cv_global),
        ("matched_unit_summary", combined["matched_summary"]),
        ("matched_model_summary", matched_model),
        ("matched_global_summary", matched_global),
        ("matched_results_all", combined["matched_results"]),
        ("matched_sensitivity_all", combined["matched_sensitivity"]),
        ("association_all", combined["association_summary"]),
        ("sequence_tests_all", combined["sequence_effect_tests"]),
    ):
        if not frame.empty:
            write_parquet_atomic(frame, outputs[name])

    compact_frames = []
    if not cv_global.empty:
        tmp = cv_global.copy()
        tmp.insert(0, "summary_type", "cv_incremental")
        compact_frames.append(tmp)
    if not matched_global.empty:
        tmp = matched_global.copy()
        tmp.insert(0, "summary_type", "matched_discrimination")
        compact_frames.append(tmp)
    compact = pd.concat(compact_frames, ignore_index=True, sort=False) if compact_frames else pd.DataFrame()
    _write_csv(compact, outputs["compact"])

    _write_json(
        {
            "schema_version": 1,
            "analysis": "behavioral_resilience",
            "models": models,
            "datasets": args.dataset,
            "probes": args.probe,
            "primary_sample": "round0_agreement",
            "robustness_sample": "all",
            "primary_continuous_outcome": "mean_support_loss",
            "complementary_continuous_outcome": "mean_absolute_movement",
            "secondary_binary_outcome": "ever_changed",
            "cross_domain_inference": (
                "effects are averaged within model across domains before cross-model sign-flip tests"
            ),
            "n_permutations": int(args.n_permutations),
            "seed": int(args.seed),
        },
        outputs["config"],
    )

    print("=" * 100)
    print("BEHAVIORAL RESILIENCE AGGREGATION COMPLETE")
    print("=" * 100)
    print(status["status"].value_counts(dropna=False).to_string())
    if not cv_global.empty:
        print("\nIncremental predictive validity (model as inferential unit):")
        print(cv_global.round(4).to_string(index=False))
    if not matched_global.empty:
        print("\nMatched-pair behavioral discrimination (model as inferential unit):")
        print(matched_global.round(4).to_string(index=False))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
