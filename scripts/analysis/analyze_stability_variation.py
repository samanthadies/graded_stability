"""
Characterizes variation in graded stability across propositions, domains, models,
instruction tuning, and model scale for both conditional estimators.

Examples:
    python -m scripts.analysis.analyze_stability_variation --probe sawmil --overwrite
    python -m scripts.analysis.analyze_stability_variation --model _llama-3.1-8b --probe sawmil --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from stability.utils.io import write_parquet_atomic


DATASETS = ("cities_loc", "med_indications", "defs")
PROBES = ("sawmil", "svm", "mean_difference")
SOURCE_TO_ESTIMATOR = {
    "conditional": "direct",
    "joint": "joint",
}
ESTIMATORS = ("direct", "joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze graded-stability distributions, domain differences, "
            "instruction-tuning effects, and model scale."
        )
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--model_list",
        type=Path,
        default=Path("configs/model_list.yaml"),
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=Path("configs/model"),
    )
    parser.add_argument(
        "--gamma_dir",
        type=Path,
        default=Path("outputs/gamma"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/analysis/stability_variation"),
    )

    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Optional model key. May be supplied multiple times.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--probe",
        nargs="+",
        action="extend",
        choices=PROBES,
        default=None,
        help=(
            "Probe(s) to analyze. May be supplied once with multiple values "
            "or repeated. Defaults to sawmil."
        ),
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=10000,
        help=("Monte Carlo sign-flip draws when exact enumeration is infeasible. "
              "Small sign spaces are enumerated exactly."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_domain_models",
        type=int,
        default=5,
        help="Minimum paired models for a domain comparison.",
    )
    parser.add_argument(
        "--min_instruction_pairs",
        type=int,
        default=3,
        help="Minimum exact base/instruct checkpoint pairs for aggregate test.",
    )
    parser.add_argument(
        "--min_statement_overlap",
        type=int,
        default=20,
        help=(
            "Minimum overlapping believed propositions for matched-P "
            "instruction robustness summary."
        ),
    )
    parser.add_argument(
        "--min_scale_sizes",
        type=int,
        default=3,
        help=(
            "Minimum distinct parameter sizes for a formal scale trend. "
            "Applied to both broad-family primary trends and same-series "
            "robustness trends."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.probe is None:
        args.probe = ["sawmil"]
    args.probe = list(dict.fromkeys(args.probe))
    args.dataset = list(dict.fromkeys(args.dataset))

    if args.n_permutations < 0:
        parser.error("--n_permutations must be >= 0.")
    if args.min_domain_models < 2:
        parser.error("--min_domain_models must be >= 2.")
    if args.min_instruction_pairs < 2:
        parser.error("--min_instruction_pairs must be >= 2.")
    if args.min_statement_overlap < 1:
        parser.error("--min_statement_overlap must be >= 1.")
    if args.min_scale_sizes < 2:
        parser.error("--min_scale_sizes must be >= 2.")

    return args


def under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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


def guard_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Analysis outputs already exist; pass --overwrite:\n  "
            + "\n  ".join(str(path) for path in existing)
        )


def load_model_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]

    if isinstance(raw, dict):
        models = [str(key) for key in raw]
    elif isinstance(raw, list):
        models = []
        for item in raw:
            if isinstance(item, str):
                models.append(item)
                continue

            if not isinstance(item, dict):
                raise ValueError(
                    f"Unsupported model-list entry: {item!r}"
                )

            value = next(
                (
                    item[key]
                    for key in ("name", "config", "model_name", "key")
                    if key in item
                ),
                None,
            )
            if value is None and len(item) == 1:
                value = next(iter(item))
            if value is None:
                raise ValueError(
                    f"Cannot infer model key from {item!r}"
                )
            models.append(str(value))
    else:
        raise ValueError(
            f"Unsupported model-list format in {path}"
        )

    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError(f"No models found in {path}")

    return models


def parse_parameter_billions(*values: str) -> float:
    matches: list[float] = []
    for value in values:
        for match in re.findall(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])",
            str(value),
        ):
            matches.append(float(match))

    if not matches:
        raise ValueError(
            f"Could not infer parameter count from {values!r}"
        )

    return max(matches)


def infer_family_and_series(model_key: str) -> tuple[str, str]:
    """
    Broad family vs. genuinely scale-comparable series.

    Examples:
        qwen-2.5-14b   -> qwen, qwen-2.5
        gemma-2-9b     -> gemma, gemma-2
        gemma-7b       -> gemma, gemma-1
        llama-3.1-8b   -> llama, llama-3.1
        mistral-3.1-24b -> mistral, mistral-3.1

    We intentionally preserve version identity so version changes are not
    interpreted as pure parameter scaling.
    """
    bare = model_key.lstrip("_").lower()

    if bare.startswith("qwen-2.5-"):
        return "qwen", "qwen-2.5"

    if bare.startswith("gemma-2-"):
        return "gemma", "gemma-2"
    if bare.startswith("gemma-"):
        return "gemma", "gemma-1"

    llama_match = re.match(r"(llama-\d+(?:\.\d+)?)", bare)
    if llama_match:
        return "llama", llama_match.group(1)

    mistral_match = re.match(r"(mistral-\d+(?:\.\d+)?)", bare)
    if mistral_match:
        return "mistral", mistral_match.group(1)
    if bare.startswith("mistral-"):
        return "mistral", "mistral"

    return bare.split("-")[0], bare


def canonical_checkpoint_key(model_key: str) -> str:
    """Exact base/instruct identity under repository naming convention."""
    return model_key.lstrip("_")


def infer_instruct(
    *,
    model_key: str,
    config: dict[str, Any],
) -> bool:
    if "instruct" in config:
        return bool(config["instruct"])

    model_name = str(config.get("model", "")).lower()
    if "instruct" in model_name or "instruction" in model_name:
        return True

    return model_key.startswith("_")


def discover_model_metadata(
    *,
    config_dir: Path,
    models: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model in models:
        path = config_dir / f"{model}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model config for {model}: {path}"
            )

        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        family, series = infer_family_and_series(model)

        params_b = parse_parameter_billions(
            str(config.get("name", "")),
            str(config.get("model", "")),
            model,
        )

        instruct = infer_instruct(
            model_key=model,
            config=config,
        )

        rows.append(
            {
                "model": model,
                "canonical_checkpoint": canonical_checkpoint_key(model),
                "family": family,
                "series": series,
                "params_b": float(params_b),
                "log10_params_b": math.log10(float(params_b)),
                "instruct": bool(instruct),
                "model_hf": str(config.get("model", "")),
                "config_path": str(path),
            }
        )

    metadata = pd.DataFrame(rows)

    if metadata["model"].duplicated().any():
        raise ValueError("Duplicate model metadata rows.")

    return metadata


def load_gamma_for_unit(
    path: Path,
    *,
    probe: str,
    estimator: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_parquet(path)

    required = {"probe", "source", "P_id", "gamma"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"{path}: missing required columns {sorted(missing)}"
        )

    source = (
        "conditional"
        if estimator == "direct"
        else "joint"
    )

    subset = frame.loc[
        frame["probe"].astype(str).eq(probe)
        & frame["source"].astype(str).eq(source),
        ["P_id", "gamma"],
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=["P_id", "gamma"])

    subset["P_id"] = subset["P_id"].astype(str)
    subset["gamma"] = pd.to_numeric(
        subset["gamma"],
        errors="coerce",
    )

    subset = subset.loc[
        np.isfinite(subset["gamma"])
        & subset["gamma"].between(0.0, 1.0, inclusive="both")
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=["P_id", "gamma"])

    if subset["P_id"].duplicated().any():
        duplicates = (
            subset.loc[
                subset["P_id"].duplicated(keep=False),
                "P_id",
            ]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{path}/{probe}/{estimator}: duplicate P_id values "
            f"{duplicates}"
        )

    return subset.reset_index(drop=True)


def build_statement_level(
    *,
    root: Path,
    gamma_dir: Path,
    metadata: pd.DataFrame,
    datasets: list[str],
    probes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    statement_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    meta_lookup = metadata.set_index("model", drop=False)

    for model in metadata["model"].tolist():
        meta = meta_lookup.loc[model].to_dict()

        for dataset in datasets:
            gamma_path = gamma_dir / model / f"{dataset}.parquet"

            for probe in probes:
                for estimator in ESTIMATORS:
                    base_status = {
                        "model": model,
                        "dataset": dataset,
                        "probe": probe,
                        "estimator": estimator,
                        "gamma_path": str(gamma_path),
                    }

                    try:
                        frame = load_gamma_for_unit(
                            gamma_path,
                            probe=probe,
                            estimator=estimator,
                        )
                    except FileNotFoundError as exc:
                        status_rows.append(
                            {
                                **base_status,
                                "status": "skipped",
                                "reason": "missing_gamma_file",
                                "message": str(exc),
                                "n_P": 0,
                            }
                        )
                        continue
                    except (KeyError, ValueError) as exc:
                        status_rows.append(
                            {
                                **base_status,
                                "status": "error",
                                "reason": type(exc).__name__,
                                "message": str(exc),
                                "n_P": 0,
                            }
                        )
                        continue

                    if frame.empty:
                        status_rows.append(
                            {
                                **base_status,
                                "status": "skipped",
                                "reason": "no_valid_gamma_rows",
                                "message": None,
                                "n_P": 0,
                            }
                        )
                        continue

                    frame.insert(0, "model", model)
                    frame.insert(1, "dataset", dataset)
                    frame.insert(2, "probe", probe)
                    frame.insert(3, "estimator", estimator)

                    for column in (
                        "canonical_checkpoint",
                        "family",
                        "series",
                        "params_b",
                        "log10_params_b",
                        "instruct",
                        "model_hf",
                    ):
                        frame[column] = meta[column]

                    statement_frames.append(frame)

                    status_rows.append(
                        {
                            **base_status,
                            "status": "complete",
                            "reason": None,
                            "message": None,
                            "n_P": int(len(frame)),
                        }
                    )

    statement_level = (
        pd.concat(statement_frames, ignore_index=True, sort=False)
        if statement_frames
        else pd.DataFrame()
    )
    status = pd.DataFrame(status_rows)

    return statement_level, status


def summarize_model_gamma(statement_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "model",
        "canonical_checkpoint",
        "family",
        "series",
        "params_b",
        "log10_params_b",
        "instruct",
        "dataset",
        "probe",
        "estimator",
    ]

    for key, group in statement_level.groupby(
        group_cols,
        sort=True,
        dropna=False,
    ):
        (
            model,
            canonical_checkpoint,
            family,
            series,
            params_b,
            log10_params_b,
            instruct,
            dataset,
            probe,
            estimator,
        ) = key

        gamma = group["gamma"].to_numpy(dtype=float)

        rows.append(
            {
                "model": model,
                "canonical_checkpoint": canonical_checkpoint,
                "family": family,
                "series": series,
                "params_b": float(params_b),
                "log10_params_b": float(log10_params_b),
                "instruct": bool(instruct),
                "dataset": dataset,
                "probe": probe,
                "estimator": estimator,
                "n_P": int(len(gamma)),
                "mean_gamma": float(np.mean(gamma)),
                "median_gamma": float(np.median(gamma)),
                "sd_gamma": (
                    float(np.std(gamma, ddof=1))
                    if len(gamma) > 1
                    else np.nan
                ),
                "q05_gamma": float(np.quantile(gamma, 0.05)),
                "q10_gamma": float(np.quantile(gamma, 0.10)),
                "q25_gamma": float(np.quantile(gamma, 0.25)),
                "q75_gamma": float(np.quantile(gamma, 0.75)),
                "q90_gamma": float(np.quantile(gamma, 0.90)),
                "q95_gamma": float(np.quantile(gamma, 0.95)),
                "min_gamma": float(np.min(gamma)),
                "max_gamma": float(np.max(gamma)),
                "fraction_gamma_eq_0": float(
                    np.mean(np.isclose(gamma, 0.0, atol=1e-12))
                ),
                "fraction_gamma_eq_1": float(
                    np.mean(np.isclose(gamma, 1.0, atol=1e-12))
                ),
            }
        )

    return pd.DataFrame(rows)


def sign_flip_test(
    values: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    statistic: str = "median",
    max_exact_patterns: int = 65536,
) -> dict[str, Any]:

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    empty = {
        "observed": np.nan,
        "null_q025": np.nan,
        "null_median": np.nan,
        "null_q975": np.nan,
        "p_two_sided": np.nan,
        "n_permutations": 0,
        "test_mode": "not_run",
        "n_possible_sign_patterns": 0,
    }

    if len(values) == 0:
        return empty

    if statistic == "median":
        stat_fn = np.median
    elif statistic == "mean":
        stat_fn = np.mean
    else:
        raise ValueError(
            f"Unsupported sign-flip statistic: {statistic}"
        )

    observed = float(stat_fn(values))
    n = len(values)
    n_possible = 2 ** n

    if n_possible <= max_exact_patterns:
        # Enumerate all sign assignments. Bit j of pattern i determines the
        # sign applied to values[j].
        pattern_ids = np.arange(n_possible, dtype=np.uint64)[:, None]
        bit_ids = np.arange(n, dtype=np.uint64)[None, :]
        bits = (pattern_ids >> bit_ids) & 1
        signs = np.where(bits == 1, 1.0, -1.0)

        signed = signs * values[None, :]
        if statistic == "median":
            null = np.median(signed, axis=1)
        else:
            null = np.mean(signed, axis=1)

        extreme = int(
            np.sum(np.abs(null) >= abs(observed))
        )
        p_two_sided = extreme / n_possible
        test_mode = "exact"
        n_used = n_possible

    else:
        if n_permutations == 0:
            return {
                **empty,
                "observed": observed,
                "test_mode": "monte_carlo_disabled",
                "n_possible_sign_patterns": int(n_possible),
            }

        rng = np.random.default_rng(seed)
        null = np.empty(n_permutations, dtype=float)

        for i in range(n_permutations):
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=n,
                replace=True,
            )
            null[i] = float(stat_fn(values * signs))

        extreme = int(
            np.sum(np.abs(null) >= abs(observed))
        )
        p_two_sided = (
            1 + extreme
        ) / (
            n_permutations + 1
        )
        test_mode = "monte_carlo"
        n_used = n_permutations

    q025, q50, q975 = np.quantile(
        null,
        [0.025, 0.50, 0.975],
    )

    return {
        "observed": observed,
        "null_q025": float(q025),
        "null_median": float(q50),
        "null_q975": float(q975),
        "p_two_sided": float(p_two_sided),
        "n_permutations": int(n_used),
        "test_mode": test_mode,
        "n_possible_sign_patterns": int(n_possible),
    }


def analyze_domain_differences(
    model_summary: pd.DataFrame,
    *,
    datasets: list[str],
    probes: list[str],
    n_permutations: int,
    min_models: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for probe in probes:
        for estimator in ESTIMATORS:
            subset = model_summary.loc[
                model_summary["probe"].eq(probe)
                & model_summary["estimator"].eq(estimator)
            ].copy()

            if subset.empty:
                continue

            for domain_a, domain_b in combinations(datasets, 2):
                a = subset.loc[
                    subset["dataset"].eq(domain_a),
                    ["model", "mean_gamma"],
                ].rename(
                    columns={
                        "mean_gamma": "mean_gamma_a",
                    }
                )
                b = subset.loc[
                    subset["dataset"].eq(domain_b),
                    ["model", "mean_gamma"],
                ].rename(
                    columns={
                        "mean_gamma": "mean_gamma_b",
                    }
                )

                paired = a.merge(
                    b,
                    on="model",
                    how="inner",
                    validate="one_to_one",
                )

                if paired.empty:
                    continue

                paired["delta_gamma_b_minus_a"] = (
                    paired["mean_gamma_b"]
                    - paired["mean_gamma_a"]
                )

                for _, row in paired.iterrows():
                    delta_rows.append(
                        {
                            "model": row["model"],
                            "probe": probe,
                            "estimator": estimator,
                            "domain_a": domain_a,
                            "domain_b": domain_b,
                            "mean_gamma_a": float(
                                row["mean_gamma_a"]
                            ),
                            "mean_gamma_b": float(
                                row["mean_gamma_b"]
                            ),
                            "delta_gamma_b_minus_a": float(
                                row["delta_gamma_b_minus_a"]
                            ),
                        }
                    )

                delta = paired[
                    "delta_gamma_b_minus_a"
                ].to_numpy(dtype=float)

                if len(delta) >= min_models:
                    test = sign_flip_test(
                        delta,
                        n_permutations=n_permutations,
                        seed=stable_seed(
                            seed,
                            "domain",
                            probe,
                            estimator,
                            domain_a,
                            domain_b,
                        ),
                        statistic="median",
                    )
                else:
                    test = {
                        "observed": np.nan,
                        "null_q025": np.nan,
                        "null_median": np.nan,
                        "null_q975": np.nan,
                        "p_two_sided": np.nan,
                        "n_permutations": 0,
                        "test_mode": "not_run",
                        "n_possible_sign_patterns": 0,
                    }

                comparison_rows.append(
                    {
                        "probe": probe,
                        "estimator": estimator,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "n_models": int(len(delta)),
                        "mean_delta_gamma_b_minus_a": float(
                            np.mean(delta)
                        ),
                        "median_delta_gamma_b_minus_a": float(
                            np.median(delta)
                        ),
                        "q25_delta_gamma_b_minus_a": float(
                            np.quantile(delta, 0.25)
                        ),
                        "q75_delta_gamma_b_minus_a": float(
                            np.quantile(delta, 0.75)
                        ),
                        "fraction_delta_positive": float(
                            np.mean(delta > 0)
                        ),
                        "fraction_delta_negative": float(
                            np.mean(delta < 0)
                        ),
                        "sign_flip_statistic": "median",
                        "sign_flip_observed_median": (
                            test["observed"]
                        ),
                        "sign_flip_null_q025": (
                            test["null_q025"]
                        ),
                        "sign_flip_null_median": (
                            test["null_median"]
                        ),
                        "sign_flip_null_q975": (
                            test["null_q975"]
                        ),
                        "sign_flip_p_two_sided": (
                            test["p_two_sided"]
                        ),
                        "sign_flip_n_permutations": (
                            test["n_permutations"]
                        ),
                        "sign_flip_test_mode": test.get(
                            "test_mode",
                            "not_run",
                        ),
                        "sign_flip_n_possible_patterns": test.get(
                            "n_possible_sign_patterns",
                            0,
                        ),
                    }
                )

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(delta_rows),
    )


def find_exact_instruction_pairs(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for canonical_checkpoint, group in metadata.groupby(
        "canonical_checkpoint",
        sort=True,
    ):
        base = group.loc[~group["instruct"]]
        instruct = group.loc[group["instruct"]]

        if len(base) != 1 or len(instruct) != 1:
            continue

        base_row = base.iloc[0]
        instruct_row = instruct.iloc[0]

        if not np.isclose(
            float(base_row["params_b"]),
            float(instruct_row["params_b"]),
        ):
            continue
        if str(base_row["series"]) != str(instruct_row["series"]):
            continue

        rows.append(
            {
                "canonical_checkpoint": canonical_checkpoint,
                "base_model": str(base_row["model"]),
                "instruct_model": str(instruct_row["model"]),
                "family": str(base_row["family"]),
                "series": str(base_row["series"]),
                "params_b": float(base_row["params_b"]),
                "log10_params_b": float(
                    base_row["log10_params_b"]
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_instruction_tuning(
    *,
    statement_level: pd.DataFrame,
    model_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    datasets: list[str],
    probes: list[str],
    n_permutations: int,
    min_pairs: int,
    min_statement_overlap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact_pairs = find_exact_instruction_pairs(metadata)

    pair_rows: list[dict[str, Any]] = []
    statement_rows: list[dict[str, Any]] = []

    if exact_pairs.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    model_summary_lookup = model_summary.set_index(
        ["model", "dataset", "probe", "estimator"],
        drop=False,
    )

    for _, pair in exact_pairs.iterrows():
        base_model = str(pair["base_model"])
        instruct_model = str(pair["instruct_model"])

        for dataset in datasets:
            for probe in probes:
                for estimator in ESTIMATORS:
                    base_key = (
                        base_model,
                        dataset,
                        probe,
                        estimator,
                    )
                    instruct_key = (
                        instruct_model,
                        dataset,
                        probe,
                        estimator,
                    )

                    if (
                        base_key not in model_summary_lookup.index
                        or instruct_key not in model_summary_lookup.index
                    ):
                        continue

                    base_summary = model_summary_lookup.loc[base_key]
                    instruct_summary = model_summary_lookup.loc[
                        instruct_key
                    ]

                    base_frame = statement_level.loc[
                        statement_level["model"].eq(base_model)
                        & statement_level["dataset"].eq(dataset)
                        & statement_level["probe"].eq(probe)
                        & statement_level["estimator"].eq(estimator),
                        ["P_id", "gamma"],
                    ].rename(
                        columns={
                            "gamma": "gamma_base",
                        }
                    )

                    instruct_frame = statement_level.loc[
                        statement_level["model"].eq(instruct_model)
                        & statement_level["dataset"].eq(dataset)
                        & statement_level["probe"].eq(probe)
                        & statement_level["estimator"].eq(estimator),
                        ["P_id", "gamma"],
                    ].rename(
                        columns={
                            "gamma": "gamma_instruct",
                        }
                    )

                    overlap = base_frame.merge(
                        instruct_frame,
                        on="P_id",
                        how="inner",
                        validate="one_to_one",
                    )

                    overlap["delta_gamma_instruct_minus_base"] = (
                        overlap["gamma_instruct"]
                        - overlap["gamma_base"]
                    )

                    n_base = int(len(base_frame))
                    n_instruct = int(len(instruct_frame))
                    n_overlap = int(len(overlap))

                    for _, row in overlap.iterrows():
                        statement_rows.append(
                            {
                                "canonical_checkpoint": pair[
                                    "canonical_checkpoint"
                                ],
                                "base_model": base_model,
                                "instruct_model": instruct_model,
                                "family": pair["family"],
                                "series": pair["series"],
                                "params_b": float(
                                    pair["params_b"]
                                ),
                                "dataset": dataset,
                                "probe": probe,
                                "estimator": estimator,
                                "P_id": str(row["P_id"]),
                                "gamma_base": float(
                                    row["gamma_base"]
                                ),
                                "gamma_instruct": float(
                                    row["gamma_instruct"]
                                ),
                                "delta_gamma_instruct_minus_base": float(
                                    row[
                                        "delta_gamma_instruct_minus_base"
                                    ]
                                ),
                            }
                        )

                    if n_overlap >= min_statement_overlap:
                        matched_delta_mean = float(
                            overlap[
                                "delta_gamma_instruct_minus_base"
                            ].mean()
                        )
                        matched_delta_median = float(
                            overlap[
                                "delta_gamma_instruct_minus_base"
                            ].median()
                        )
                        matched_fraction_positive = float(
                            np.mean(
                                overlap[
                                    "delta_gamma_instruct_minus_base"
                                ] > 0
                            )
                        )
                    else:
                        matched_delta_mean = np.nan
                        matched_delta_median = np.nan
                        matched_fraction_positive = np.nan

                    pair_rows.append(
                        {
                            "canonical_checkpoint": pair[
                                "canonical_checkpoint"
                            ],
                            "base_model": base_model,
                            "instruct_model": instruct_model,
                            "family": pair["family"],
                            "series": pair["series"],
                            "params_b": float(pair["params_b"]),
                            "log10_params_b": float(
                                pair["log10_params_b"]
                            ),
                            "dataset": dataset,
                            "probe": probe,
                            "estimator": estimator,
                            "n_P_base": n_base,
                            "n_P_instruct": n_instruct,
                            "mean_gamma_base": float(
                                base_summary["mean_gamma"]
                            ),
                            "mean_gamma_instruct": float(
                                instruct_summary["mean_gamma"]
                            ),
                            "delta_mean_gamma_all": float(
                                instruct_summary["mean_gamma"]
                                - base_summary["mean_gamma"]
                            ),
                            "n_overlap": n_overlap,
                            "overlap_fraction_base": (
                                n_overlap / n_base
                                if n_base
                                else np.nan
                            ),
                            "overlap_fraction_instruct": (
                                n_overlap / n_instruct
                                if n_instruct
                                else np.nan
                            ),
                            "mean_delta_gamma_matched": (
                                matched_delta_mean
                            ),
                            "median_delta_gamma_matched": (
                                matched_delta_median
                            ),
                            "fraction_matched_delta_positive": (
                                matched_fraction_positive
                            ),
                            "matched_analysis_valid": bool(
                                n_overlap >= min_statement_overlap
                            ),
                        }
                    )

    pair_summary = pd.DataFrame(pair_rows)
    statement_deltas = pd.DataFrame(statement_rows)

    aggregate_rows: list[dict[str, Any]] = []

    if not pair_summary.empty:
        for probe in probes:
            for estimator in ESTIMATORS:
                subset_pe = pair_summary.loc[
                    pair_summary["probe"].eq(probe)
                    & pair_summary["estimator"].eq(estimator)
                ].copy()

                if subset_pe.empty:
                    continue

                for dataset in [*datasets, "overall"]:
                    if dataset == "overall":

                        subset = (
                            subset_pe.groupby(
                                [
                                    "canonical_checkpoint",
                                    "base_model",
                                    "instruct_model",
                                ],
                                as_index=False,
                                sort=True,
                            )
                            .agg(
                                delta_mean_gamma_all=(
                                    "delta_mean_gamma_all",
                                    "mean",
                                ),
                                mean_delta_gamma_matched=(
                                    "mean_delta_gamma_matched",
                                    "mean",
                                ),
                                matched_analysis_valid=(
                                    "matched_analysis_valid",
                                    "all",
                                ),
                                overlap_fraction_base=(
                                    "overlap_fraction_base",
                                    "mean",
                                ),
                                overlap_fraction_instruct=(
                                    "overlap_fraction_instruct",
                                    "mean",
                                ),
                                n_domains=(
                                    "dataset",
                                    "nunique",
                                ),
                            )
                        )
                    else:
                        subset = subset_pe.loc[
                            subset_pe["dataset"].eq(dataset)
                        ].copy()
                        subset["n_domains"] = 1

                    if subset.empty:
                        continue

                    full_effects = subset[
                        "delta_mean_gamma_all"
                    ].dropna().to_numpy(dtype=float)

                    if len(full_effects) >= min_pairs:
                        full_test = sign_flip_test(
                            full_effects,
                            n_permutations=n_permutations,
                            seed=stable_seed(
                                seed,
                                "instruction_full",
                                probe,
                                estimator,
                                dataset,
                            ),
                            statistic="median",
                        )
                    else:
                        full_test = {
                            "observed": np.nan,
                            "null_q025": np.nan,
                            "null_median": np.nan,
                            "null_q975": np.nan,
                            "p_two_sided": np.nan,
                            "n_permutations": 0,
                            "test_mode": "not_run",
                            "n_possible_sign_patterns": 0,
                        }

                    valid_matched = subset.loc[
                        subset["matched_analysis_valid"]
                        & np.isfinite(
                            subset["mean_delta_gamma_matched"]
                        )
                    ].copy()

                    matched_effects = valid_matched[
                        "mean_delta_gamma_matched"
                    ].to_numpy(dtype=float)

                    if len(matched_effects) >= min_pairs:
                        matched_test = sign_flip_test(
                            matched_effects,
                            n_permutations=n_permutations,
                            seed=stable_seed(
                                seed,
                                "instruction_matched",
                                probe,
                                estimator,
                                dataset,
                            ),
                            statistic="median",
                        )
                    else:
                        matched_test = {
                            "observed": np.nan,
                            "null_q025": np.nan,
                            "null_median": np.nan,
                            "null_q975": np.nan,
                            "p_two_sided": np.nan,
                            "n_permutations": 0,
                            "test_mode": "not_run",
                            "n_possible_sign_patterns": 0,
                        }

                    aggregate_rows.append(
                        {
                            "probe": probe,
                            "estimator": estimator,
                            "dataset": dataset,
                            "aggregation_unit": "checkpoint_pair",
                            "n_checkpoint_pairs": int(
                                len(full_effects)
                            ),
                            "mean_domains_per_checkpoint_pair": float(
                                subset["n_domains"].mean()
                            ),
                            "mean_delta_gamma_all": (
                                float(np.mean(full_effects))
                                if len(full_effects)
                                else np.nan
                            ),
                            "median_delta_gamma_all": (
                                float(np.median(full_effects))
                                if len(full_effects)
                                else np.nan
                            ),
                            "q25_delta_gamma_all": (
                                float(np.quantile(full_effects, 0.25))
                                if len(full_effects)
                                else np.nan
                            ),
                            "q75_delta_gamma_all": (
                                float(np.quantile(full_effects, 0.75))
                                if len(full_effects)
                                else np.nan
                            ),
                            "fraction_delta_gamma_all_positive": (
                                float(np.mean(full_effects > 0))
                                if len(full_effects)
                                else np.nan
                            ),
                            "full_sign_flip_observed_median": (
                                full_test["observed"]
                            ),
                            "full_sign_flip_null_q025": (
                                full_test["null_q025"]
                            ),
                            "full_sign_flip_null_median": (
                                full_test["null_median"]
                            ),
                            "full_sign_flip_null_q975": (
                                full_test["null_q975"]
                            ),
                            "full_sign_flip_p_two_sided": (
                                full_test["p_two_sided"]
                            ),
                            "full_sign_flip_n_permutations": (
                                full_test["n_permutations"]
                            ),
                            "full_sign_flip_test_mode": (
                                full_test["test_mode"]
                            ),
                            "full_sign_flip_n_possible_patterns": (
                                full_test["n_possible_sign_patterns"]
                            ),
                            "n_matched_valid_pairs": int(
                                len(matched_effects)
                            ),
                            "mean_matched_delta_gamma": (
                                float(np.mean(matched_effects))
                                if len(matched_effects)
                                else np.nan
                            ),
                            "median_matched_delta_gamma": (
                                float(np.median(matched_effects))
                                if len(matched_effects)
                                else np.nan
                            ),
                            "fraction_matched_delta_positive": (
                                float(np.mean(matched_effects > 0))
                                if len(matched_effects)
                                else np.nan
                            ),
                            "matched_sign_flip_observed_median": (
                                matched_test["observed"]
                            ),
                            "matched_sign_flip_null_q025": (
                                matched_test["null_q025"]
                            ),
                            "matched_sign_flip_null_median": (
                                matched_test["null_median"]
                            ),
                            "matched_sign_flip_null_q975": (
                                matched_test["null_q975"]
                            ),
                            "matched_sign_flip_p_two_sided": (
                                matched_test["p_two_sided"]
                            ),
                            "matched_sign_flip_n_permutations": (
                                matched_test["n_permutations"]
                            ),
                            "matched_sign_flip_test_mode": (
                                matched_test["test_mode"]
                            ),
                            "matched_sign_flip_n_possible_patterns": (
                                matched_test["n_possible_sign_patterns"]
                            ),
                            "mean_overlap_fraction_base": (
                                float(
                                    valid_matched[
                                        "overlap_fraction_base"
                                    ].mean()
                                )
                                if len(valid_matched)
                                else np.nan
                            ),
                            "mean_overlap_fraction_instruct": (
                                float(
                                    valid_matched[
                                        "overlap_fraction_instruct"
                                    ].mean()
                                )
                                if len(valid_matched)
                                else np.nan
                            ),
                        }
                    )

    return (
        pair_summary,
        statement_deltas,
        pd.DataFrame(aggregate_rows),
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if (
        len(x) < 2
        or np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return np.nan

    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()

    return float(np.corrcoef(xr, yr)[0, 1])


def analyze_scale(
    model_summary: pd.DataFrame,
    *,
    min_scale_sizes: int,
    scope: str = "family",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if scope not in {"family", "series"}:
        raise ValueError(
            f"Unsupported scale-analysis scope: {scope!r}"
        )

    trend_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    if scope == "family":
        group_cols = [
            "family",
            "instruct",
            "dataset",
            "probe",
            "estimator",
        ]
    else:
        group_cols = [
            "family",
            "series",
            "instruct",
            "dataset",
            "probe",
            "estimator",
        ]

    for key, group in model_summary.groupby(
        group_cols,
        sort=True,
        dropna=False,
    ):
        if scope == "family":
            (
                family,
                instruct,
                dataset,
                probe,
                estimator,
            ) = key
            series = None
        else:
            (
                family,
                series,
                instruct,
                dataset,
                probe,
                estimator,
            ) = key

        by_size = (
            group.groupby(
                ["params_b", "log10_params_b"],
                as_index=False,
            )
            .agg(
                mean_gamma=("mean_gamma", "mean"),
                n_models=("model", "nunique"),
                models=(
                    "model",
                    lambda s: "|".join(
                        sorted(set(map(str, s)))
                    ),
                ),
                series_at_size=(
                    "series",
                    lambda s: "|".join(
                        sorted(set(map(str, s)))
                    ),
                ),
            )
            .sort_values("params_b")
            .reset_index(drop=True)
        )

        n_sizes = int(
            by_size["params_b"].nunique()
        )

        base_record = {
            "scope": scope,
            "family": family,
            "series": series,
            "instruct": bool(instruct),
            "dataset": dataset,
            "probe": probe,
            "estimator": estimator,
            "n_sizes": n_sizes,
        }

        if n_sizes >= 2:
            smallest = by_size.iloc[0]
            largest = by_size.iloc[-1]

            pair_rows.append(
                {
                    **base_record,
                    "min_params_b": float(
                        smallest["params_b"]
                    ),
                    "max_params_b": float(
                        largest["params_b"]
                    ),
                    "gamma_smallest": float(
                        smallest["mean_gamma"]
                    ),
                    "gamma_largest": float(
                        largest["mean_gamma"]
                    ),
                    "delta_largest_minus_smallest": float(
                        largest["mean_gamma"]
                        - smallest["mean_gamma"]
                    ),
                    "smallest_models": smallest["models"],
                    "largest_models": largest["models"],
                    "smallest_series": smallest["series_at_size"],
                    "largest_series": largest["series_at_size"],
                }
            )

        if n_sizes < min_scale_sizes:
            continue

        x = by_size[
            "log10_params_b"
        ].to_numpy(dtype=float)
        y = by_size[
            "mean_gamma"
        ].to_numpy(dtype=float)

        slope, intercept = np.polyfit(
            x,
            y,
            deg=1,
        )

        trend_rows.append(
            {
                **base_record,
                "min_params_b": float(
                    by_size["params_b"].min()
                ),
                "max_params_b": float(
                    by_size["params_b"].max()
                ),
                "slope_gamma_per_log10_b": float(
                    slope
                ),
                "intercept": float(
                    intercept
                ),
                "spearman_rho_scale_gamma": (
                    spearman(
                        by_size["params_b"].to_numpy(),
                        y,
                    )
                ),
                "gamma_smallest": float(
                    by_size.iloc[0]["mean_gamma"]
                ),
                "gamma_largest": float(
                    by_size.iloc[-1]["mean_gamma"]
                ),
                "delta_largest_minus_smallest": float(
                    by_size.iloc[-1]["mean_gamma"]
                    - by_size.iloc[0]["mean_gamma"]
                ),
                "models_by_size": ";".join(
                    f"{row.params_b:g}B:{row.models}"
                    for row in by_size.itertuples()
                ),
                "series_by_size": ";".join(
                    f"{row.params_b:g}B:{row.series_at_size}"
                    for row in by_size.itertuples()
                ),
            }
        )

    return (
        pd.DataFrame(trend_rows),
        pd.DataFrame(pair_rows),
    )


def make_compact_summary(
    *,
    model_summary: pd.DataFrame,
    domain_comparisons: pd.DataFrame,
    instruction_aggregate: pd.DataFrame,
    scale_trends: pd.DataFrame,
    scale_pair_deltas: pd.DataFrame,
    series_scale_trends: pd.DataFrame,
    series_scale_pair_deltas: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Distribution summaries.
    for (probe, estimator, dataset), group in model_summary.groupby(
        ["probe", "estimator", "dataset"],
        sort=True,
    ):
        rows.append(
            {
                "summary_type": "distribution",
                "probe": probe,
                "estimator": estimator,
                "dataset": dataset,
                "n_models": int(group["model"].nunique()),
                "median_model_mean_gamma": float(
                    group["mean_gamma"].median()
                ),
                "q25_model_mean_gamma": float(
                    group["mean_gamma"].quantile(0.25)
                ),
                "q75_model_mean_gamma": float(
                    group["mean_gamma"].quantile(0.75)
                ),
                "min_model_mean_gamma": float(
                    group["mean_gamma"].min()
                ),
                "max_model_mean_gamma": float(
                    group["mean_gamma"].max()
                ),
                "median_model_median_gamma": float(
                    group["median_gamma"].median()
                ),
            }
        )

    if not domain_comparisons.empty:
        for _, row in domain_comparisons.iterrows():
            rows.append(
                {
                    "summary_type": "domain",
                    "probe": row["probe"],
                    "estimator": row["estimator"],
                    "dataset": (
                        f"{row['domain_b']}_minus_"
                        f"{row['domain_a']}"
                    ),
                    "n_models": int(row["n_models"]),
                    "median_delta": float(
                        row["median_delta_gamma_b_minus_a"]
                    ),
                    "mean_delta": float(
                        row["mean_delta_gamma_b_minus_a"]
                    ),
                    "fraction_positive": float(
                        row["fraction_delta_positive"]
                    ),
                    "p_two_sided": float(
                        row["sign_flip_p_two_sided"]
                    )
                    if np.isfinite(
                        row["sign_flip_p_two_sided"]
                    )
                    else np.nan,
                }
            )

    if not instruction_aggregate.empty:
        for _, row in instruction_aggregate.iterrows():
            rows.append(
                {
                    "summary_type": "instruction",
                    "probe": row["probe"],
                    "estimator": row["estimator"],
                    "dataset": row["dataset"],
                    "n_pairs": int(
                        row["n_checkpoint_pairs"]
                    ),
                    "median_delta": row[
                        "median_delta_gamma_all"
                    ],
                    "mean_delta": row[
                        "mean_delta_gamma_all"
                    ],
                    "fraction_positive": row[
                        "fraction_delta_gamma_all_positive"
                    ],
                    "p_two_sided": row[
                        "full_sign_flip_p_two_sided"
                    ],
                    "n_matched_valid_pairs": int(
                        row["n_matched_valid_pairs"]
                    ),
                    "median_matched_delta": row[
                        "median_matched_delta_gamma"
                    ],
                    "matched_p_two_sided": row[
                        "matched_sign_flip_p_two_sided"
                    ],
                }
            )

    if not scale_trends.empty:
        for _, row in scale_trends.iterrows():
            rows.append(
                {
                    "summary_type": "scale_family_primary",
                    "probe": row["probe"],
                    "estimator": row["estimator"],
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "instruct": row["instruct"],
                    "n_sizes": int(row["n_sizes"]),
                    "slope_gamma_per_log10_b": row[
                        "slope_gamma_per_log10_b"
                    ],
                    "spearman_rho": row[
                        "spearman_rho_scale_gamma"
                    ],
                    "delta_largest_minus_smallest": row[
                        "delta_largest_minus_smallest"
                    ],
                }
            )

    if not scale_pair_deltas.empty:
        for (probe, estimator, dataset), group in scale_pair_deltas.groupby(
            ["probe", "estimator", "dataset"],
            sort=True,
        ):
            values = group[
                "delta_largest_minus_smallest"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "summary_type": "scale_family_direction",
                    "probe": probe,
                    "estimator": estimator,
                    "dataset": dataset,
                    "n_families": int(len(values)),
                    "median_delta_largest_minus_smallest": (
                        float(np.median(values))
                    ),
                    "fraction_positive_scale_delta": float(
                        np.mean(values > 0)
                    ),
                    "fraction_negative_scale_delta": float(
                        np.mean(values < 0)
                    ),
                }
            )

    if not series_scale_trends.empty:
        for _, row in series_scale_trends.iterrows():
            rows.append(
                {
                    "summary_type": "scale_series_robustness",
                    "probe": row["probe"],
                    "estimator": row["estimator"],
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "series": row["series"],
                    "instruct": row["instruct"],
                    "n_sizes": int(row["n_sizes"]),
                    "slope_gamma_per_log10_b": row[
                        "slope_gamma_per_log10_b"
                    ],
                    "spearman_rho": row[
                        "spearman_rho_scale_gamma"
                    ],
                    "delta_largest_minus_smallest": row[
                        "delta_largest_minus_smallest"
                    ],
                }
            )

    if not series_scale_pair_deltas.empty:
        for (probe, estimator, dataset), group in series_scale_pair_deltas.groupby(
            ["probe", "estimator", "dataset"],
            sort=True,
        ):
            values = group[
                "delta_largest_minus_smallest"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "summary_type": "scale_series_direction_robustness",
                    "probe": probe,
                    "estimator": estimator,
                    "dataset": dataset,
                    "n_series": int(len(values)),
                    "median_delta_largest_minus_smallest": (
                        float(np.median(values))
                    ),
                    "fraction_positive_scale_delta": float(
                        np.mean(values > 0)
                    ),
                    "fraction_negative_scale_delta": float(
                        np.mean(values < 0)
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    root = args.repo_root.resolve()
    model_list_path = under_root(
        root,
        args.model_list,
    )
    config_dir = under_root(
        root,
        args.config_dir,
    )
    gamma_dir = under_root(
        root,
        args.gamma_dir,
    )
    output_dir = under_root(
        root,
        args.output_dir,
    )

    models = load_model_list(
        model_list_path
    )

    if args.model:
        requested = list(
            dict.fromkeys(
                args.model
            )
        )
        unknown = sorted(
            set(requested)
            - set(models)
        )
        if unknown:
            raise ValueError(
                f"Unknown requested models: {unknown}"
            )
        models = requested

    metadata = discover_model_metadata(
        config_dir=config_dir,
        models=models,
    )

    output_paths = {
        "statement_level": output_dir / "statement_level.parquet",
        "model_summary": output_dir / "model_summary.parquet",
        "domain_comparisons": output_dir / "domain_comparisons.parquet",
        "domain_model_deltas": output_dir / "domain_model_deltas.parquet",
        "instruction_pair_summary": (
            output_dir / "instruction_pair_summary.parquet"
        ),
        "instruction_statement_deltas": (
            output_dir / "instruction_statement_deltas.parquet"
        ),
        "instruction_aggregate": (
            output_dir / "instruction_aggregate.parquet"
        ),
        # PRIMARY broad-family scale analysis.
        "scale_trends": output_dir / "scale_trends.parquet",
        "scale_pair_deltas": output_dir / "scale_pair_deltas.parquet",
        # STRICT same-series robustness analysis.
        "series_scale_trends": (
            output_dir / "series_scale_trends.parquet"
        ),
        "series_scale_pair_deltas": (
            output_dir / "series_scale_pair_deltas.parquet"
        ),
        "status": output_dir / "analysis_status.parquet",
        "compact": output_dir / "summary_compact.csv",
        "config": output_dir / "analysis_config.json",
    }

    guard_outputs(
        output_paths.values(),
        overwrite=args.overwrite,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("GRADED STABILITY VARIATION")
    print("=" * 100)
    print(f"Models:              {len(models)}")
    print(f"Datasets:            {args.dataset}")
    print(f"Probes:              {args.probe}")
    print(f"Estimators:          {list(ESTIMATORS)}")
    print(f"Permutations:        {args.n_permutations}")
    print(f"Min domain models:   {args.min_domain_models}")
    print(f"Min instruction pairs:{args.min_instruction_pairs}")
    print(f"Min statement overlap:{args.min_statement_overlap}")
    print(f"Min scale sizes:     {args.min_scale_sizes}")
    print(f"Output:              {output_dir}")
    print("=" * 100)

    statement_level, status = build_statement_level(
        root=root,
        gamma_dir=gamma_dir,
        metadata=metadata,
        datasets=args.dataset,
        probes=args.probe,
    )

    if statement_level.empty:
        write_parquet_atomic(
            status,
            output_paths["status"],
        )
        raise RuntimeError(
            "No valid gamma observations were loaded."
        )

    model_summary = summarize_model_gamma(
        statement_level
    )

    (
        domain_comparisons,
        domain_model_deltas,
    ) = analyze_domain_differences(
        model_summary,
        datasets=args.dataset,
        probes=args.probe,
        n_permutations=args.n_permutations,
        min_models=args.min_domain_models,
        seed=args.seed,
    )

    (
        instruction_pair_summary,
        instruction_statement_deltas,
        instruction_aggregate,
    ) = analyze_instruction_tuning(
        statement_level=statement_level,
        model_summary=model_summary,
        metadata=metadata,
        datasets=args.dataset,
        probes=args.probe,
        n_permutations=args.n_permutations,
        min_pairs=args.min_instruction_pairs,
        min_statement_overlap=args.min_statement_overlap,
        seed=args.seed,
    )

    (
        scale_trends,
        scale_pair_deltas,
    ) = analyze_scale(
        model_summary,
        min_scale_sizes=args.min_scale_sizes,
        scope="family",
    )

    (
        series_scale_trends,
        series_scale_pair_deltas,
    ) = analyze_scale(
        model_summary,
        min_scale_sizes=args.min_scale_sizes,
        scope="series",
    )

    compact = make_compact_summary(
        model_summary=model_summary,
        domain_comparisons=domain_comparisons,
        instruction_aggregate=instruction_aggregate,
        scale_trends=scale_trends,
        scale_pair_deltas=scale_pair_deltas,
        series_scale_trends=series_scale_trends,
        series_scale_pair_deltas=series_scale_pair_deltas,
    )

    statement_level = statement_level.sort_values(
        ["probe", "estimator", "dataset", "model", "P_id"],
        kind="stable",
    ).reset_index(drop=True)

    model_summary = model_summary.sort_values(
        ["probe", "estimator", "dataset", "model"],
        kind="stable",
    ).reset_index(drop=True)

    status = status.sort_values(
        ["probe", "estimator", "dataset", "model"],
        kind="stable",
    ).reset_index(drop=True)

    for name, frame, sort_columns in (
        (
            "domain_comparisons",
            domain_comparisons,
            ["probe", "estimator", "domain_a", "domain_b"],
        ),
        (
            "domain_model_deltas",
            domain_model_deltas,
            ["probe", "estimator", "domain_a", "domain_b", "model"],
        ),
        (
            "instruction_pair_summary",
            instruction_pair_summary,
            ["probe", "estimator", "dataset", "canonical_checkpoint"],
        ),
        (
            "instruction_statement_deltas",
            instruction_statement_deltas,
            [
                "probe",
                "estimator",
                "dataset",
                "canonical_checkpoint",
                "P_id",
            ],
        ),
        (
            "instruction_aggregate",
            instruction_aggregate,
            ["probe", "estimator", "dataset"],
        ),
        (
            "scale_trends",
            scale_trends,
            ["probe", "estimator", "dataset", "family", "instruct"],
        ),
        (
            "scale_pair_deltas",
            scale_pair_deltas,
            ["probe", "estimator", "dataset", "family", "instruct"],
        ),
        (
            "series_scale_trends",
            series_scale_trends,
            ["probe", "estimator", "dataset", "family", "series", "instruct"],
        ),
        (
            "series_scale_pair_deltas",
            series_scale_pair_deltas,
            ["probe", "estimator", "dataset", "family", "series", "instruct"],
        ),
    ):
        if not frame.empty:
            frame.sort_values(
                sort_columns,
                kind="stable",
                inplace=True,
            )
            frame.reset_index(
                drop=True,
                inplace=True,
            )

    if not compact.empty:
        compact = compact.reset_index(drop=True)

    write_parquet_atomic(
        statement_level,
        output_paths["statement_level"],
    )
    write_parquet_atomic(
        model_summary,
        output_paths["model_summary"],
    )
    write_parquet_atomic(
        domain_comparisons,
        output_paths["domain_comparisons"],
    )
    write_parquet_atomic(
        domain_model_deltas,
        output_paths["domain_model_deltas"],
    )
    write_parquet_atomic(
        instruction_pair_summary,
        output_paths["instruction_pair_summary"],
    )
    write_parquet_atomic(
        instruction_statement_deltas,
        output_paths["instruction_statement_deltas"],
    )
    write_parquet_atomic(
        instruction_aggregate,
        output_paths["instruction_aggregate"],
    )
    write_parquet_atomic(
        scale_trends,
        output_paths["scale_trends"],
    )
    write_parquet_atomic(
        scale_pair_deltas,
        output_paths["scale_pair_deltas"],
    )
    write_parquet_atomic(
        series_scale_trends,
        output_paths["series_scale_trends"],
    )
    write_parquet_atomic(
        series_scale_pair_deltas,
        output_paths["series_scale_pair_deltas"],
    )
    write_parquet_atomic(
        status,
        output_paths["status"],
    )
    write_csv_atomic(
        compact,
        output_paths["compact"],
    )

    config_payload = {
        "schema_version": 2,
        "analysis": "stability_variation",
        "research_questions": {
            "RQ4a": (
                "How is graded belief stability distributed across "
                "LLMs and domains?"
            ),
            "RQ4b": (
                "Does instruction tuning systematically alter graded "
                "belief stability?"
            ),
            "RQ4c": (
                "Is graded belief stability systematically related "
                "to model scale?"
            ),
        },
        "models": models,
        "datasets": args.dataset,
        "probes": args.probe,
        "estimators": list(ESTIMATORS),
        "model_metadata": {
            "config_dir": str(config_dir),
            "family_vs_series": (
                "family is broad architecture family; series preserves "
                "version identity for scale comparisons"
            ),
            "instruct_rule": (
                "explicit config flag/model name if available, otherwise "
                "leading underscore convention"
            ),
            "exact_pairing_key": "model key with leading underscore removed",
        },
        "domain_analysis": {
            "unit": "model-level mean gamma",
            "paired_within_model": True,
            "effect": "mean_gamma(domain_b) - mean_gamma(domain_a)",
            "aggregate_statistic": "median paired difference",
            "test": "two-sided sign-flip; exact when feasible, Monte Carlo otherwise",
            "n_permutations": int(args.n_permutations),
            "minimum_models": int(args.min_domain_models),
            "plus_one_correction": (
                "Monte Carlo only; exact enumeration uses exact randomization p"
            ),
            "exact_sign_flip_max_patterns": 65536,
        },
        "instruction_analysis": {
            "pairing": "exact base/instruction checkpoints only",
            "primary_effect": (
                "mean_gamma_instruct(full P set) - "
                "mean_gamma_base(full P set)"
            ),
            "robustness_effect": (
                "mean proposition-level gamma difference on overlapping P"
            ),
            "minimum_statement_overlap": int(
                args.min_statement_overlap
            ),
            "overall_aggregation": (
                "for the overall row, first average effects across domains "
                "within each checkpoint pair; checkpoint pair remains the "
                "inferential unit"
            ),
            "aggregate_statistic": "median checkpoint-pair difference",
            "test": "two-sided sign-flip; exact when feasible, Monte Carlo otherwise",
            "n_permutations": int(args.n_permutations),
            "minimum_checkpoint_pairs": int(
                args.min_instruction_pairs
            ),
            "plus_one_correction": (
                "Monte Carlo only; exact enumeration uses exact randomization p"
            ),
            "exact_sign_flip_max_patterns": 65536,
        },
        "scale_analysis": {
            "primary_scope": (
                "family x instruct x dataset x probe x estimator"
            ),
            "primary_output_files": [
                "scale_trends.parquet",
                "scale_pair_deltas.parquet",
            ],
            "same_series_robustness_output_files": [
                "series_scale_trends.parquet",
                "series_scale_pair_deltas.parquet",
            ],
            "predictor": "log10(parameter count in billions)",
            "outcome": "model-level mean gamma",
            "minimum_sizes_for_formal_trend": int(
                args.min_scale_sizes
            ),
            "same_size_weighting": (
                "within a broad family, average model-level mean gamma across "
                "checkpoints sharing the same nominal parameter count before "
                "fitting; each nominal size therefore receives one observation"
            ),
            "primary_metrics": [
                "linear slope",
                "Spearman rho",
                "largest-minus-smallest mean gamma",
            ],
            "two_size_groups": (
                "retain largest-minus-smallest difference descriptively; "
                "do not treat a two-point slope as a formal trend"
            ),
            "interpretation": (
                "Broad-family trends characterize associations across "
                "available checkpoint sizes within a model family and tuning "
                "status. Because model version may vary with parameter count, "
                "they are not interpreted as controlled or causal scaling "
                "effects."
            ),
            "strict_robustness": (
                "repeat the identical analysis within version-preserving model "
                "series wherever enough distinct sizes are available"
            ),
        },
        "seed": int(args.seed),
        "model_list": str(model_list_path),
        "gamma_dir": str(gamma_dir),
    }

    write_json_atomic(
        config_payload,
        output_paths["config"],
    )

    complete = int(
        (
            status["status"]
            == "complete"
        ).sum()
    )
    skipped = int(
        (
            status["status"]
            == "skipped"
        ).sum()
    )
    errors = int(
        (
            status["status"]
            == "error"
        ).sum()
    )

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(
        f"Complete units={complete} skipped={skipped} errors={errors} "
        f"statement_rows={len(statement_level)} "
        f"model_rows={len(model_summary)}"
    )

    print()
    print("MODEL-LEVEL GAMMA DISTRIBUTIONS")
    print("-" * 100)
    dist = (
        model_summary.groupby(
            ["probe", "estimator", "dataset"],
            as_index=False,
        )
        .agg(
            n_models=("model", "nunique"),
            median_model_mean_gamma=("mean_gamma", "median"),
            q25_model_mean_gamma=("mean_gamma", lambda x: x.quantile(0.25)),
            q75_model_mean_gamma=("mean_gamma", lambda x: x.quantile(0.75)),
            min_model_mean_gamma=("mean_gamma", "min"),
            max_model_mean_gamma=("mean_gamma", "max"),
        )
    )
    numeric = dist.select_dtypes(include=[np.number]).columns
    dist[numeric] = dist[numeric].round(4)
    print(dist.to_string(index=False))

    if not domain_comparisons.empty:
        print()
        print("DOMAIN COMPARISONS")
        print("-" * 100)
        display = domain_comparisons.copy()
        numeric = display.select_dtypes(include=[np.number]).columns
        display[numeric] = display[numeric].round(4)
        print(display.to_string(index=False))

    if not instruction_aggregate.empty:
        print()
        print("INSTRUCTION TUNING")
        print("-" * 100)
        display = instruction_aggregate.copy()
        numeric = display.select_dtypes(include=[np.number]).columns
        display[numeric] = display[numeric].round(4)
        print(display.to_string(index=False))

    print()
    print("BROAD-FAMILY SCALE TRENDS (PRIMARY)")
    print("-" * 100)
    if scale_trends.empty:
        print("None retained.")
    else:
        display = scale_trends.copy()
        numeric = display.select_dtypes(include=[np.number]).columns
        display[numeric] = display[numeric].round(4)
        print(display.to_string(index=False))

    print()
    print("SAME-SERIES SCALE TRENDS (STRICT ROBUSTNESS)")
    print("-" * 100)
    if series_scale_trends.empty:
        print("None retained.")
    else:
        display = series_scale_trends.copy()
        numeric = display.select_dtypes(include=[np.number]).columns
        display[numeric] = display[numeric].round(4)
        print(display.to_string(index=False))

    print()
    print(f"Saved to {output_dir}")
    print("=" * 100)

    if errors:
        raise RuntimeError(
            f"{errors} analysis unit(s) failed unexpectedly; "
            f"inspect {output_paths['status']}."
        )


if __name__ == "__main__":
    main()
