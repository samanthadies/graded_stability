from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from stability.data.loading import load_probe_dataset
from stability.evaluation.metrics import evaluate_all_splits
from stability.models.activations import (
    collect_layer_activations,
    get_transformer_layers,
)
from stability.models.loading import (
    load_model_by_name,
    load_model_config,
)
from stability.probes.factory import (
    SUPPORTED_PROBES,
    build_probe,
    load_probe_config,
)
from stability.utils.io import (
    completed_pairs,
    read_parquet_if_exists,
    upsert_result_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep probe performance across transformer layers."
    )

    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cities_loc", "med_indications", "defs"),
    )

    parser.add_argument(
        "--config",
        default="configs/experiments/sweep_layers.yaml",
        help="Experiment-level sweep config.",
    )

    parser.add_argument(
        "--probes",
        nargs="+",
        choices=SUPPORTED_PROBES,
        default=None,
        help="Override probes from the experiment config.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Explicit zero-based layer subset. Default: all model layers.",
    )

    parser.add_argument("--construction", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_config_dir", default=None)
    parser.add_argument("--probe_config_dir", default=None)
    parser.add_argument("--output_dir", default=None)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed (layer, probe) pairs from an existing file.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output and start from scratch.",
    )

    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}.")

    return config


def _coalesce(
    cli_value: Any,
    config: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _requested_layers(
    *,
    explicit_layers: list[int] | None,
    n_layers: int,
) -> list[int]:
    if explicit_layers is None or len(explicit_layers) == 0:
        layers = list(range(n_layers))
    else:
        # Preserve requested order while removing duplicates.
        layers = list(
            dict.fromkeys(
                int(layer)
                for layer in explicit_layers
            )
        )

    invalid = [
        layer
        for layer in layers
        if not 0 <= layer < n_layers
    ]
    if invalid:
        raise ValueError(
            f"Invalid layers {invalid}; model has layers 0..{n_layers - 1}."
        )

    return layers


def _base_result_row(
    *,
    model_name: str,
    hf_model: str,
    dataset: str,
    construction: str,
    probe_name: str,
    layer: int,
    n_layers: int,
    seed: int,
    max_length: int,
    batch_size: int,
    activation_seconds: float,
    activation_ram_gib: float,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "model_name": model_name,
        "hf_model": hf_model,
        "dataset": dataset,
        "construction": construction,
        "probe": probe_name,
        "layer": int(layer),
        "num_model_layers": int(n_layers),
        "seed": int(seed),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "activation_seconds": float(activation_seconds),
        "activation_ram_gib": float(activation_ram_gib),
    }


def main() -> None:
    args = parse_args()
    experiment = _load_yaml(args.config)

    construction = str(
        _coalesce(
            args.construction,
            experiment,
            "construction",
            "atomic",
        )
    )
    batch_size = int(
        _coalesce(
            args.batch_size,
            experiment,
            "batch_size",
            16,
        )
    )
    max_length = int(
        _coalesce(
            args.max_length,
            experiment,
            "max_length",
            64,
        )
    )
    seed = int(
        _coalesce(
            args.seed,
            experiment,
            "seed",
            0,
        )
    )
    device = str(
        _coalesce(
            args.device,
            experiment,
            "device",
            "cuda",
        )
    )

    probes = (
        list(args.probes)
        if args.probes is not None
        else list(
            experiment.get(
                "probes",
                SUPPORTED_PROBES,
            )
        )
    )
    probes = list(
        dict.fromkeys(
            str(name).strip().lower()
            for name in probes
        )
    )

    invalid_probes = [
        name
        for name in probes
        if name not in SUPPORTED_PROBES
    ]
    if invalid_probes:
        raise ValueError(
            f"Unsupported probes {invalid_probes}; "
            f"expected {SUPPORTED_PROBES}."
        )

    paths = experiment.get("paths", {}) or {}

    data_dir = Path(
        args.data_dir
        or paths.get("data_dir", "data")
    )
    model_config_dir = Path(
        args.model_config_dir
        or paths.get(
            "model_config_dir",
            "configs/model",
        )
    )
    probe_config_dir = Path(
        args.probe_config_dir
        or paths.get(
            "probe_config_dir",
            "configs/probe",
        )
    )
    output_dir = Path(
        args.output_dir
        or paths.get(
            "output_dir",
            "outputs/layer_sweep",
        )
    )

    output_path = (
        output_dir
        / args.model_name
        / f"{args.dataset}.parquet"
    )

    if args.overwrite and output_path.exists():
        output_path.unlink()

    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --resume to continue it or --overwrite to replace it."
        )

    _set_seed(seed)

    data = load_probe_dataset(
        args.dataset,
        construction=construction,
        data_dir=data_dir,
    )

    model_config = load_model_config(
        args.model_name,
        config_dir=model_config_dir,
    )

    probe_configs = {
        probe_name: load_probe_config(
            probe_name,
            config_dir=probe_config_dir,
        )
        for probe_name in probes
    }

    existing = (
        read_parquet_if_exists(output_path)
        if args.resume
        else pd.DataFrame()
    )
    done = completed_pairs(existing)
    
    # Treat explicitly recorded degenerate probe/layer combinations as terminal.
    # They have no valid probe direction, so rerunning them will not help.
    if not existing.empty and "status" in existing.columns:
        degenerate = existing[
            existing["status"].astype(str) == "degenerate"
        ]
    
        done.update(
            (
                int(row["layer"]),
                str(row["probe"]),
            )
            for _, row in degenerate.iterrows()
        )

    print("=" * 72)
    print(f"Model:           {args.model_name}")
    print(f"HF model:        {model_config['model']}")
    print(f"Dataset:         {args.dataset}")
    print(f"Construction:    {construction}")
    print(f"Rows:            {data.n_rows:,}")
    print(
        "Train / cal / test: "
        f"{data.train_mask.sum():,} / "
        f"{data.cal_mask.sum():,} / "
        f"{data.test_mask.sum():,}"
    )
    print(f"Probes:          {probes}")
    print(f"Output:          {output_path}")
    print(f"Resume:          {args.resume}")
    print("=" * 72)

    bundle = None

    try:
        bundle = load_model_by_name(
            args.model_name,
            config_dir=model_config_dir,
            device=device,
        )

        transformer_layers = get_transformer_layers(
            bundle.model
        )
        n_layers = len(transformer_layers)

        layers = _requested_layers(
            explicit_layers=args.layers,
            n_layers=n_layers,
        )

        print(f"Layers:          {layers}")
        if done:
            print(
                "Completed pairs: "
                f"{len(done)}"
            )

        for layer in layers:
            missing_probes = [
                probe_name
                for probe_name in probes
                if (int(layer), probe_name) not in done
            ]

            if not missing_probes:
                print(
                    f"Layer {layer}: all requested probes already complete; skipping."
                )
                continue

            print("\n" + "=" * 72)
            print(f"LAYER {layer}/{n_layers - 1}")
            print(f"Probes to run: {missing_probes}")
            print("=" * 72)

            _set_seed(seed)

            layer_start = time.perf_counter()
            activations = collect_layer_activations(
                bundle,
                data.statements,
                layer=layer,
                batch_size=batch_size,
                max_length=max_length,
            )
            activation_seconds = (
                time.perf_counter() - layer_start
            )

            print(
                f"Activations: shape={activations.values.shape}, "
                f"dtype={activations.values.dtype}, "
                f"RAM={activations.ram_gib:.2f} GiB, "
                f"time={activation_seconds:.1f}s"
            )

            try:
                for probe_name in missing_probes:
                    print(
                        f"\n[{probe_name}] fitting/evaluating..."
                    )
                    probe_start = time.perf_counter()

                    try:
                        _set_seed(seed)

                        probe = build_probe(
                            probe_name,
                            config=probe_configs[probe_name],
                            seed=seed,
                        )

                        probe.fit(
                            activations,
                            data.labels,
                            data.train_mask,
                            data.cal_mask,
                        )

                        metrics = evaluate_all_splits(
                            probe=probe,
                            activations=activations,
                            labels=data.labels,
                            train_mask=data.train_mask,
                            cal_mask=data.cal_mask,
                            test_mask=data.test_mask,
                        )

                        probe_seconds = (
                            time.perf_counter()
                            - probe_start
                        )

                        row = _base_result_row(
                            model_name=args.model_name,
                            hf_model=str(model_config["model"]),
                            dataset=args.dataset,
                            construction=construction,
                            probe_name=probe_name,
                            layer=layer,
                            n_layers=n_layers,
                            seed=seed,
                            max_length=max_length,
                            batch_size=batch_size,
                            activation_seconds=activation_seconds,
                            activation_ram_gib=activations.ram_gib,
                        )
                        row.update(metrics)
                        row["probe_seconds"] = float(
                            probe_seconds
                        )
                        row["elapsed_seconds"] = float(
                            activation_seconds
                            + probe_seconds
                        )
                        row["error"] = None

                        upsert_result_rows(
                            output_path,
                            [row],
                        )
                        done.add(
                            (int(layer), probe_name)
                        )

                        print(
                            "Test: "
                            f"accuracy={metrics['test_accuracy']:.4f} | "
                            f"balanced={metrics['test_balanced_accuracy']:.4f} | "
                            f"macro-F1={metrics['test_macro_f1']:.4f} | "
                            f"log-loss={metrics['test_log_loss']:.4f}"
                        )
                        print(
                            f"Saved checkpoint: {output_path}"
                        )

                        del probe

                    except Exception as exc:
                        probe_seconds = (
                            time.perf_counter()
                            - probe_start
                        )

                        error_row = _base_result_row(
                            model_name=args.model_name,
                            hf_model=str(model_config["model"]),
                            dataset=args.dataset,
                            construction=construction,
                            probe_name=probe_name,
                            layer=layer,
                            n_layers=n_layers,
                            seed=seed,
                            max_length=max_length,
                            batch_size=batch_size,
                            activation_seconds=activation_seconds,
                            activation_ram_gib=activations.ram_gib,
                        )
                        error_message = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        
                        # Mean-difference has no mathematically defined direction when the
                        # class mean equals the complementary-class mean exactly. This is a
                        # legitimate degenerate layer, not a transient execution failure.
                        is_mean_difference_degeneracy = (
                            probe_name == "mean_difference"
                            and isinstance(exc, ValueError)
                            and "Mean-difference direction for class" in str(exc)
                            and "invalid norm 0.0" in str(exc)
                        )
                        
                        if is_mean_difference_degeneracy:
                            error_row["status"] = "degenerate"
                        else:
                            error_row["status"] = "error"
                        
                        error_row["probe_seconds"] = float(
                            probe_seconds
                        )
                        
                        error_row["elapsed_seconds"] = float(
                            activation_seconds
                            + probe_seconds
                        )
                        
                        error_row["error"] = error_message
                        
                        upsert_result_rows(
                            output_path,
                            [error_row],
                        )
                        
                        if is_mean_difference_degeneracy:
                            done.add(
                                (int(layer), probe_name)
                            )
                        
                            print(
                                f"[{probe_name}] DEGENERATE: {error_message}"
                            )
                            print(
                                f"Saved degenerate checkpoint: {output_path}"
                            )
                            print(
                                "Continuing to the next probe/layer."
                            )
                        
                            continue
                        
                        print(
                            f"[{probe_name}] ERROR: {error_message}"
                        )
                        print(
                            f"Saved error checkpoint: {output_path}"
                        )
                        
                        # Unexpected errors should still kill the job so that genuine problems
                        # are not silently hidden.
                        raise

            finally:
                del activations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final = read_parquet_if_exists(
            output_path
        )

        print("\n" + "=" * 72)
        print("SWEEP COMPLETE")
        print("=" * 72)
        print(f"Output: {output_path}")
        print(f"Rows:   {len(final)}")

        complete = final[
            final["status"].astype(str) == "complete"
        ].copy()

        if not complete.empty:
            for probe_name in probes:
                subset = complete[
                    complete["probe"] == probe_name
                ]
                if subset.empty:
                    continue

                best_index = subset[
                    "test_log_loss"
                ].astype(float).idxmin()

                best = subset.loc[best_index]

                print(
                    f"{probe_name:18s} "
                    f"best layer={int(best['layer'])} | "
                    f"test log-loss={float(best['test_log_loss']):.6f} | "
                    f"macro-F1={float(best['test_macro_f1']):.6f}"
                )

    finally:
        if bundle is not None:
            del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
