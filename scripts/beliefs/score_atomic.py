"""
Fits the atomic belief probes at their preselected layers and records calibrated
True/False/Neither probabilities for the held-out statements in each dataset.

Examples:
    python -m scripts.beliefs.score_atomic --model_name _llama-3.1-8b --dataset cities_loc --resume
    python -m scripts.beliefs.score_atomic --model_name _llama-3.1-8b --dataset cities_loc --probes sawmil --overwrite
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from stability.data.loading import ProbeDataset, load_probe_dataset
from stability.models.activations import collect_layer_activations
from stability.models.loading import load_model_by_name, load_model_config
from stability.probes.base import Probe
from stability.probes.factory import (
    SUPPORTED_PROBES,
    build_probe,
    load_probe_config,
)
from stability.utils.io import (
    read_parquet_if_exists,
    write_parquet_atomic,
)


DEFAULT_OUTPUT_DIR = Path("outputs/atomic")
DEFAULT_MODEL_CONFIG_DIR = Path("configs/model")
DEFAULT_PROBE_CONFIG_DIR = Path("configs/probe")
DEFAULT_DATA_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit atomic probes at their selected layers and save calibrated "
            "test-set credences without saving activations."
        )
    )

    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cities_loc", "med_indications", "defs"),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=SUPPORTED_PROBES,
        default=None,
        help=(
            "Probe types to run. Default: sawmil svm mean_difference. "
            "Useful for debugging or adding a new probe later."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--model_config_dir",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_DIR,
    )
    parser.add_argument(
        "--probe_config_dir",
        type=Path,
        default=DEFAULT_PROBE_CONFIG_DIR,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep complete probe results already present in both the Parquet "
            "and fitted-probe bundle, and run only missing probes."
        ),
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing atomic outputs for this model/dataset.",
    )

    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_probe_names(
    probes: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if probes is None:
        probes = list(SUPPORTED_PROBES)

    normalized = list(
        dict.fromkeys(
            str(probe).strip().lower()
            for probe in probes
        )
    )

    invalid = [
        probe
        for probe in normalized
        if probe not in SUPPORTED_PROBES
    ]
    if invalid:
        raise ValueError(
            f"Unsupported probes {invalid}; expected {SUPPORTED_PROBES}."
        )

    if not normalized:
        raise ValueError("At least one probe must be requested.")

    return normalized


def _selected_layer(
    model_config: dict[str, Any],
    *,
    probe_name: str,
    dataset: str,
) -> int:
    selected = model_config.get("selected_layers")

    if not isinstance(selected, dict):
        raise ValueError(
            "Model config is missing a 'selected_layers' mapping."
        )

    probe_layers = selected.get(probe_name)
    if not isinstance(probe_layers, dict):
        raise ValueError(
            f"Model config has no selected_layers entry for probe "
            f"{probe_name!r}."
        )

    if dataset not in probe_layers:
        raise ValueError(
            f"Model config has no selected layer for "
            f"{probe_name}/{dataset}."
        )

    value = probe_layers[dataset]

    if isinstance(value, bool):
        raise ValueError(
            f"Invalid selected layer for {probe_name}/{dataset}: {value!r}."
        )

    try:
        layer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid selected layer for {probe_name}/{dataset}: {value!r}."
        ) from exc

    if layer < 0:
        raise ValueError(
            f"Selected layer must be nonnegative; got {layer} for "
            f"{probe_name}/{dataset}."
        )

    return layer


def _group_probes_by_layer(
    *,
    probes: list[str],
    selected_layers: dict[str, int],
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}

    for probe_name in probes:
        layer = int(selected_layers[probe_name])
        grouped.setdefault(layer, []).append(probe_name)

    return dict(sorted(grouped.items()))


def _semantic_prediction_labels(
    *,
    predicted_ids: np.ndarray,
    data: ProbeDataset,
) -> np.ndarray:
    mapping = data.label_schema.id_to_label

    try:
        labels = [
            mapping[int(class_id)]
            for class_id in predicted_ids
        ]
    except KeyError as exc:
        raise RuntimeError(
            f"Probe predicted class ID not present in label schema: {exc}."
        ) from exc

    return np.asarray(labels, dtype=object)


def _probability_column(
    probabilities: np.ndarray,
    *,
    probe_classes: np.ndarray,
    data: ProbeDataset,
    label_name: str,
) -> np.ndarray:
    label_to_id = data.label_schema.label_to_id
    if label_name not in label_to_id:
        raise ValueError(
            f"Atomic label schema does not contain {label_name!r}."
        )

    class_id = int(label_to_id[label_name])
    matches = np.flatnonzero(
        np.asarray(probe_classes) == class_id
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one probability column for class ID {class_id} "
            f"({label_name}); found {len(matches)}."
        )

    return probabilities[:, int(matches[0])]



def _score_probe(
    *,
    probe: Probe,
    probe_name: str,
    layer: int,
    activations: Any,
    data: ProbeDataset,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if probe.classes_ is None:
        raise RuntimeError("Probe must be fit before scoring.")

    test_indices = data.indices("test")
    y_true = data.labels[test_indices]

    probabilities = np.asarray(
        probe.predict_proba(
            activations,
            test_indices,
        ),
        dtype=np.float64,
    )

    expected_shape = (
        len(test_indices),
        len(probe.classes_),
    )
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"{probe_name}: probability shape {probabilities.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"{probe_name}: probabilities contain NaN or infinite values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"{probe_name}: probability rows do not sum to one."
        )

    predicted_ids = np.asarray(probe.classes_)[
        np.argmax(probabilities, axis=1)
    ]
    predicted_labels = _semantic_prediction_labels(
        predicted_ids=predicted_ids,
        data=data,
    )

    prob_false = _probability_column(
        probabilities,
        probe_classes=np.asarray(probe.classes_),
        data=data,
        label_name="false",
    )
    prob_true = _probability_column(
        probabilities,
        probe_classes=np.asarray(probe.classes_),
        data=data,
        label_name="true",
    )
    prob_neither = _probability_column(
        probabilities,
        probe_classes=np.asarray(probe.classes_),
        data=data,
        label_name="neither",
    )

    rows = pd.DataFrame(
        {
            "model_name": model_name,
            "dataset": data.dataset,
            "statement_id": test_indices.astype(np.int64),
            "source_dataset": data.source_dataset[test_indices],
            "source_row": data.source_row[test_indices].astype(np.int64),
            "split": "test",
            "label": y_true.astype(np.int64),
            "probe": probe_name,
            "layer": int(layer),
            "pred_label": predicted_labels,
            "prob_false": prob_false,
            "prob_true": prob_true,
            "prob_neither": prob_neither,
        }
    )

    if rows["statement_id"].duplicated().any():
        raise RuntimeError(
            f"{probe_name}: statement_id is not unique within probe output."
        )

    metrics: dict[str, float | int] = {
        "n": int(len(test_indices)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predicted_ids,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predicted_ids,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predicted_ids,
                labels=data.classes,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predicted_ids,
                labels=data.classes,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=data.classes,
            )
        ),
    }

    return rows, metrics


def _empty_bundle(
    *,
    model_name: str,
    hf_model: str,
    dataset: str,
    seed: int,
    max_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "construction": "atomic",
        "model_name": model_name,
        "hf_model": hf_model,
        "dataset": dataset,
        "seed": int(seed),
        "max_length": int(max_length),
        "probes": {},
        "layers": {},
        "metrics": {},
    }


def _load_bundle_if_exists(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(
            f"Expected dictionary probe bundle in {path}; "
            f"found {type(bundle).__name__}."
        )

    for key in ("probes", "layers"):
        if key not in bundle or not isinstance(bundle[key], dict):
            raise ValueError(
                f"Invalid probe bundle {path}: missing dictionary field {key!r}."
            )

    return bundle


def _write_joblib_atomic(
    bundle: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        joblib.dump(
            bundle,
            temp_path,
            compress=3,
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _probe_output_is_complete(
    *,
    frame: pd.DataFrame,
    probe_name: str,
    expected_statement_ids: np.ndarray,
    expected_layer: int,
) -> bool:
    if frame.empty:
        return False

    required = {
        "statement_id",
        "probe",
        "layer",
        "pred_label",
        "prob_false",
        "prob_true",
        "prob_neither",
    }
    if not required.issubset(frame.columns):
        return False

    subset = frame[
        frame["probe"].astype(str) == probe_name
    ].copy()

    if len(subset) != len(expected_statement_ids):
        return False

    if subset["statement_id"].duplicated().any():
        return False

    if set(
        pd.to_numeric(
            subset["layer"],
            errors="coerce",
        ).dropna().astype(int).unique().tolist()
    ) != {int(expected_layer)}:
        return False

    observed_ids = np.sort(
        pd.to_numeric(
            subset["statement_id"],
            errors="coerce",
        ).dropna().astype(np.int64).to_numpy()
    )
    expected_ids = np.sort(
        np.asarray(
            expected_statement_ids,
            dtype=np.int64,
        )
    )

    return np.array_equal(
        observed_ids,
        expected_ids,
    )


def _upsert_probe_rows(
    *,
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    probe_name: str,
) -> pd.DataFrame:
    if existing.empty:
        combined = new_rows.copy()
    else:
        if "probe" not in existing.columns:
            raise ValueError(
                "Existing atomic Parquet does not contain a 'probe' column."
            )

        combined = pd.concat(
            [
                existing[
                    existing["probe"].astype(str) != probe_name
                ],
                new_rows,
            ],
            ignore_index=True,
            sort=False,
        )

    combined = combined.sort_values(
        ["probe", "statement_id"],
        kind="stable",
    ).reset_index(drop=True)

    return combined


def _validate_existing_identity(
    *,
    frame: pd.DataFrame,
    bundle: dict[str, Any] | None,
    model_name: str,
    dataset: str,
) -> None:
    if not frame.empty:
        if "model_name" in frame.columns:
            observed = set(
                frame["model_name"].dropna().astype(str).unique()
            )
            if observed and observed != {model_name}:
                raise ValueError(
                    f"Existing atomic Parquet belongs to models {observed}, "
                    f"not {model_name!r}."
                )

        if "dataset" in frame.columns:
            observed = set(
                frame["dataset"].dropna().astype(str).unique()
            )
            if observed and observed != {dataset}:
                raise ValueError(
                    f"Existing atomic Parquet belongs to datasets {observed}, "
                    f"not {dataset!r}."
                )

    if bundle is not None:
        if bundle.get("model_name") not in (None, model_name):
            raise ValueError(
                f"Existing probe bundle belongs to model "
                f"{bundle.get('model_name')!r}, not {model_name!r}."
            )
        if bundle.get("dataset") not in (None, dataset):
            raise ValueError(
                f"Existing probe bundle belongs to dataset "
                f"{bundle.get('dataset')!r}, not {dataset!r}."
            )


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive.")

    probes = _normalize_probe_names(
        args.probes
    )

    model_config = load_model_config(
        args.model_name,
        config_dir=args.model_config_dir,
    )

    selected_layers = {
        probe_name: _selected_layer(
            model_config,
            probe_name=probe_name,
            dataset=args.dataset,
        )
        for probe_name in probes
    }

    data = load_probe_dataset(
        args.dataset,
        construction="atomic",
        data_dir=args.data_dir,
    )
    test_statement_ids = data.indices("test")

    model_output_dir = (
        args.output_dir
        / args.model_name
    )
    parquet_path = (
        model_output_dir
        / f"{args.dataset}.parquet"
    )
    bundle_path = (
        model_output_dir
        / f"{args.dataset}.joblib"
    )

    if args.overwrite:
        for path in (
            parquet_path,
            bundle_path,
        ):
            if path.exists():
                path.unlink()

    if (
        not args.resume
        and not args.overwrite
        and (
            parquet_path.exists()
            or bundle_path.exists()
        )
    ):
        raise FileExistsError(
            "Atomic output already exists. Use --resume to continue it or "
            f"--overwrite to replace it.\nParquet: {parquet_path}\n"
            f"Bundle:  {bundle_path}"
        )

    existing = (
        read_parquet_if_exists(parquet_path)
        if args.resume
        else pd.DataFrame()
    )
    bundle = (
        _load_bundle_if_exists(bundle_path)
        if args.resume
        else None
    )

    _validate_existing_identity(
        frame=existing,
        bundle=bundle,
        model_name=args.model_name,
        dataset=args.dataset,
    )

    if bundle is None:
        bundle = _empty_bundle(
            model_name=args.model_name,
            hf_model=str(model_config["model"]),
            dataset=args.dataset,
            seed=args.seed,
            max_length=args.max_length,
        )

    complete_probes: set[str] = set()

    if args.resume:
        for probe_name in probes:
            parquet_complete = _probe_output_is_complete(
                frame=existing,
                probe_name=probe_name,
                expected_statement_ids=test_statement_ids,
                expected_layer=selected_layers[probe_name],
            )
            bundle_complete = (
                probe_name in bundle["probes"]
                and int(
                    bundle["layers"].get(
                        probe_name,
                        -1,
                    )
                )
                == int(
                    selected_layers[probe_name]
                )
            )

            if parquet_complete and bundle_complete:
                complete_probes.add(
                    probe_name
                )

    remaining_probes = [
        probe_name
        for probe_name in probes
        if probe_name not in complete_probes
    ]

    print("=" * 72)
    print(f"Model:           {args.model_name}")
    print(f"HF model:        {model_config['model']}")
    print(f"Dataset:         {args.dataset}")
    print(f"Rows:            {data.n_rows:,}")
    print(
        "Train / cal / test: "
        f"{data.train_mask.sum():,} / "
        f"{data.cal_mask.sum():,} / "
        f"{data.test_mask.sum():,}"
    )
    print(f"Probes:          {probes}")
    print(f"Selected layers: {selected_layers}")
    print(f"Parquet:         {parquet_path}")
    print(f"Probe bundle:    {bundle_path}")
    print(f"Resume:          {args.resume}")
    if complete_probes:
        print(
            "Already complete: "
            f"{sorted(complete_probes)}"
        )
    print("=" * 72)

    if not remaining_probes:
        print("All requested probes are already complete; nothing to do.")
        return

    grouped = _group_probes_by_layer(
        probes=remaining_probes,
        selected_layers=selected_layers,
    )

    probe_configs = {
        probe_name: load_probe_config(
            probe_name,
            config_dir=args.probe_config_dir,
        )
        for probe_name in remaining_probes
    }

    _set_seed(args.seed)

    model_bundle = None

    try:
        model_bundle = load_model_by_name(
            args.model_name,
            config_dir=args.model_config_dir,
            device=args.device,
        )

        for layer, layer_probes in grouped.items():
            print("\n" + "=" * 72)
            print(f"LAYER {layer}")
            print(f"Probes: {layer_probes}")
            print("=" * 72)

            _set_seed(args.seed)

            activation_start = time.perf_counter()
            activations = collect_layer_activations(
                model_bundle,
                data.statements,
                layer=layer,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            activation_seconds = (
                time.perf_counter()
                - activation_start
            )

            print(
                f"Activations: shape={activations.values.shape}, "
                f"dtype={activations.values.dtype}, "
                f"RAM={activations.ram_gib:.2f} GiB, "
                f"time={activation_seconds:.1f}s"
            )

            try:
                for probe_name in layer_probes:
                    print(
                        f"\n[{probe_name}] fitting..."
                    )
                    probe_start = (
                        time.perf_counter()
                    )

                    _set_seed(args.seed)

                    probe = build_probe(
                        probe_name,
                        config=probe_configs[probe_name],
                        seed=args.seed,
                    )

                    probe.fit(
                        activations,
                        data.labels,
                        data.train_mask,
                        data.cal_mask,
                    )

                    print(
                        f"[{probe_name}] scoring test split..."
                    )

                    probe_rows, metrics = _score_probe(
                        probe=probe,
                        probe_name=probe_name,
                        layer=layer,
                        activations=activations,
                        data=data,
                        model_name=args.model_name,
                    )

                    probe_seconds = (
                        time.perf_counter()
                        - probe_start
                    )

                    # Update the compact statement-level result table.
                    existing = _upsert_probe_rows(
                        existing=existing,
                        new_rows=probe_rows,
                        probe_name=probe_name,
                    )
                    write_parquet_atomic(
                        existing,
                        parquet_path,
                    )

                    # Keep the fitted probes together in one compact bundle.
                    bundle["probes"][
                        probe_name
                    ] = probe
                    bundle["layers"][
                        probe_name
                    ] = int(layer)
                    bundle["metrics"][
                        probe_name
                    ] = {
                        **metrics,
                        "activation_seconds": float(
                            activation_seconds
                        ),
                        "probe_seconds": float(
                            probe_seconds
                        ),
                    }

                    _write_joblib_atomic(
                        bundle,
                        bundle_path,
                    )

                    print(
                        "Test: "
                        f"accuracy={metrics['accuracy']:.4f} | "
                        f"balanced={metrics['balanced_accuracy']:.4f} | "
                        f"macro-F1={metrics['macro_f1']:.4f} | "
                        f"log-loss={metrics['log_loss']:.4f}"
                    )
                    print(
                        f"Saved {len(probe_rows):,} rows -> "
                        f"{parquet_path}"
                    )
                    print(
                        f"Updated fitted-probe bundle -> "
                        f"{bundle_path}"
                    )

                    del probe

            finally:
                del activations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\n" + "=" * 72)
        print("ATOMIC SCORING COMPLETE")
        print("=" * 72)
        print(f"Parquet:      {parquet_path}")
        print(f"Probe bundle: {bundle_path}")
        print(f"Rows:         {len(existing):,}")
        print()

        for probe_name in probes:
            subset = existing[
                existing["probe"].astype(str)
                == probe_name
            ]
            if subset.empty:
                continue

            metric_info = bundle.get(
                "metrics",
                {},
            ).get(
                probe_name,
                {},
            )

            print(
                f"{probe_name:18s} "
                f"layer={selected_layers[probe_name]:>2d} | "
                f"rows={len(subset):,} | "
                f"log-loss={float(metric_info.get('log_loss', float('nan'))):.6f}"
            )

    finally:
        if model_bundle is not None:
            del model_bundle
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
