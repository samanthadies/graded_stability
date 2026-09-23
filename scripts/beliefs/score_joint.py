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
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from stability.beliefs.joint import (
    JOINT_ID_TO_LABEL,
    JOINT_LABELS,
    build_joint_training_data,
)
from stability.beliefs.rendering import (
    get_default_variant,
    load_statement_config,
    render_pair_statement,
)
from stability.data.loading import (
    ProbeDataset,
    load_probe_dataset,
)
from stability.models.activations import (
    collect_layer_activations,
)
from stability.models.loading import (
    load_model_by_name,
    load_model_config,
)
from stability.probes.base import Probe
from stability.probes.factory import (
    SUPPORTED_PROBES,
    build_probe,
    load_probe_config,
)
from stability.utils.parquet_parts import (
    completed_rows_from_parts,
    consolidate_parts,
    next_part_index,
    remove_part_tree,
    write_json_atomic,
    write_part_atomic,
)


OUTPUT_COLUMNS = [
    "probe",
    "pair_id",
    "layer",
    "pred_label",
    *[
        f"prob_{label}"
        for label in JOINT_LABELS
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train nine-class joint probes and stream-score compact pair "
            "tables without saving activations or rendered test statements."
        )
    )

    parser.add_argument(
        "--model_name",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "cities_loc",
            "med_indications",
            "defs",
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/joint.yaml"
        ),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=SUPPORTED_PROBES,
        default=None,
    )

    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Joint rendering variant from configs/statements.yaml. "
            "Default comes from the experiment config, then statements.yaml."
        ),
    )
    parser.add_argument(
        "--train_total",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cal_total",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--score_chunk_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-probe test-pair limit for smoke tests.",
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--pairs_dir",
        type=Path,
        default=Path("outputs/pairs"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/joint"),
    )
    parser.add_argument(
        "--model_config_dir",
        type=Path,
        default=Path("configs/model"),
    )
    parser.add_argument(
        "--probe_config_dir",
        type=Path,
        default=Path("configs/probe"),
    )
    parser.add_argument(
        "--statement_config",
        type=Path,
        default=Path("configs/statements.yaml"),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"Expected mapping in {path}."
        )

    return config


def _setting(
    cli_value: Any,
    config: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(
        key,
        default,
    )


def _set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_probes(
    probes: list[str] | None,
    config: dict[str, Any],
) -> list[str]:
    if probes is None:
        probes = list(
            config.get(
                "probes",
                SUPPORTED_PROBES,
            )
        )

    result = list(
        dict.fromkeys(
            str(probe)
            .strip()
            .lower()
            for probe in probes
        )
    )

    invalid = [
        probe
        for probe in result
        if probe not in SUPPORTED_PROBES
    ]
    if invalid:
        raise ValueError(
            f"Unsupported probes {invalid}; expected {SUPPORTED_PROBES}."
        )

    if not result:
        raise ValueError(
            "At least one probe must be requested."
        )

    return result


def _selected_layer(
    model_config: dict[str, Any],
    *,
    probe_name: str,
    dataset: str,
) -> int:
    selected = model_config.get(
        "selected_layers"
    )
    if not isinstance(
        selected,
        dict,
    ):
        raise ValueError(
            "Model config is missing selected_layers."
        )

    probe_layers = selected.get(
        probe_name
    )
    if not isinstance(
        probe_layers,
        dict,
    ):
        raise ValueError(
            f"Model config has no selected layer mapping for {probe_name!r}."
        )

    if dataset not in probe_layers:
        raise ValueError(
            f"Model config has no selected layer for "
            f"{probe_name}/{dataset}."
        )

    layer = int(
        probe_layers[
            dataset
        ]
    )
    if layer < 0:
        raise ValueError(
            "Selected layers must be nonnegative."
        )

    return layer


def _group_by_layer(
    probes: list[str],
    layers: dict[str, int],
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}

    for probe in probes:
        grouped.setdefault(
            int(
                layers[
                    probe
                ]
            ),
            [],
        ).append(
            probe
        )

    return dict(
        sorted(
            grouped.items()
        )
    )


def _evaluate_split(
    *,
    probe: Probe,
    activations: Any,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    indices = np.flatnonzero(
        mask
    ).astype(
        np.int64
    )
    y_true = labels[
        indices
    ]

    probabilities = np.asarray(
        probe.predict_proba(
            activations,
            indices,
        ),
        dtype=np.float64,
    )
    predictions = np.asarray(
        probe.predict(
            activations,
            indices,
        )
    )

    expected_classes = np.arange(
        len(JOINT_LABELS),
        dtype=np.int64,
    )

    return {
        "n": int(
            len(indices)
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=expected_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=expected_classes,
            )
        ),
    }


def _new_bundle(
    *,
    model_name: str,
    hf_model: str,
    dataset: str,
    template: str,
    seed: int,
    max_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "construction": "joint",
        "model_name": model_name,
        "hf_model": hf_model,
        "dataset": dataset,
        "template": template,
        "ordered": True,
        "first_operand": "x",
        "second_operand": "P",
        "joint_classes": list(
            JOINT_LABELS
        ),
        "seed": int(seed),
        "max_length": int(max_length),
        "probes": {},
        "layers": {},
        "metrics": {},
        "skipped_probes": {},
    }


def _load_bundle(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    value = joblib.load(
        path
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Expected dictionary bundle in {path}."
        )

    return value


def _write_joblib_atomic(
    payload: dict[str, Any],
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
            payload,
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


def _probability_column(
    probabilities: np.ndarray,
    *,
    probe_classes: np.ndarray,
    class_id: int,
) -> np.ndarray:
    matches = np.flatnonzero(
        probe_classes
        == int(
            class_id
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one probability column for class {class_id}; "
            f"found {len(matches)}."
        )

    return probabilities[
        :,
        int(
            matches[0]
        ),
    ]


def _score_chunk(
    *,
    probe: Probe,
    probe_name: str,
    layer: int,
    pair_chunk: pd.DataFrame,
    statements: list[str],
    model_bundle: Any,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    activations = collect_layer_activations(
        model_bundle,
        statements,
        layer=layer,
        batch_size=batch_size,
        max_length=max_length,
    )

    try:
        indices = np.arange(
            len(statements),
            dtype=np.int64,
        )

        probabilities = np.asarray(
            probe.predict_proba(
                activations,
                indices,
            ),
            dtype=np.float64,
        )

        expected_shape = (
            len(statements),
            len(JOINT_LABELS),
        )
        if probabilities.shape != expected_shape:
            raise RuntimeError(
                f"{probe_name}: probability shape {probabilities.shape}; "
                f"expected {expected_shape}."
            )

        classes = np.asarray(
            probe.classes_,
            dtype=np.int64,
        )

        if set(
            classes.tolist()
        ) != set(
            range(
                len(JOINT_LABELS)
            )
        ):
            raise RuntimeError(
                f"{probe_name}: joint probe classes are "
                f"{classes.tolist()}, expected 0..8."
            )

        predictions = classes[
            np.argmax(
                probabilities,
                axis=1,
            )
        ]

        result_dict: dict[str, Any] = {
            "probe": probe_name,
            "pair_id": pair_chunk[
                "pair_id"
            ].to_numpy(
                dtype=np.int64
            ),
            "layer": int(layer),
            "pred_label": [
                JOINT_ID_TO_LABEL[
                    int(value)
                ]
                for value in predictions
            ],
        }

        for class_id, label in enumerate(
            JOINT_LABELS
        ):
            result_dict[
                f"prob_{label}"
            ] = _probability_column(
                probabilities,
                probe_classes=classes,
                class_id=class_id,
            )

        result = pd.DataFrame(
            result_dict
        )

        probability_columns = [
            f"prob_{label}"
            for label in JOINT_LABELS
        ]

        if not np.allclose(
            result[
                probability_columns
            ].sum(
                axis=1
            ),
            1.0,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                f"{probe_name}: joint probabilities do not sum to one."
            )

        return result[
            OUTPUT_COLUMNS
        ]

    finally:
        del activations
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_probe_pairs(
    pair_path: Path,
    *,
    probe_name: str,
    limit: int | None,
) -> pd.DataFrame:
    pairs = pd.read_parquet(
        pair_path,
        columns=[
            "probe",
            "pair_id",
            "P_id",
            "x_id",
        ],
        filters=[
            (
                "probe",
                "==",
                probe_name,
            )
        ],
    )

    if pairs.empty:
        raise ValueError(
            f"No pair rows found for probe {probe_name!r} in {pair_path}."
        )

    pairs = pairs.sort_values(
        "pair_id",
        kind="stable",
    ).reset_index(
        drop=True
    )

    if pairs[
        "pair_id"
    ].duplicated().any():
        raise ValueError(
            f"{probe_name}: pair_id is not unique."
        )

    if limit is not None:
        pairs = pairs.head(
            int(limit)
        ).copy()

    return pairs


def _render_test_chunk(
    pair_chunk: pd.DataFrame,
    *,
    data: ProbeDataset,
    template: str,
    statement_config_path: Path,
) -> list[str]:
    P_ids = pair_chunk[
        "P_id"
    ].to_numpy(
        dtype=np.int64
    )
    x_ids = pair_chunk[
        "x_id"
    ].to_numpy(
        dtype=np.int64
    )

    if (
        P_ids.min() < 0
        or x_ids.min() < 0
        or P_ids.max() >= data.n_rows
        or x_ids.max() >= data.n_rows
    ):
        raise ValueError(
            "Pair table references atomic statement IDs outside the dataset."
        )

    return [
        render_pair_statement(
            kind="joint",
            x_statement=str(
                data.statements[
                    int(
                        x_id
                    )
                ]
            ),
            P_statement=str(
                data.statements[
                    int(
                        P_id
                    )
                ]
            ),
            variant=template,
            config_path=statement_config_path,
        )
        for P_id, x_id in zip(
            P_ids,
            x_ids,
        )
    ]



def _is_degenerate_mean_difference_error(
    probe_name: str,
    exc: ValueError,
) -> bool:
    message = str(exc)
    return (
        probe_name == "mean_difference"
        and "Mean-difference direction for class" in message
        and "has invalid norm" in message
    )


def _available_pair_probes(
    pair_path: Path,
) -> set[str]:
    frame = pd.read_parquet(
        pair_path,
        columns=["probe"],
    )
    return set(
        frame["probe"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

def main() -> None:
    args = parse_args()
    experiment = _load_yaml(
        args.config
    )

    probes = _normalize_probes(
        args.probes,
        experiment,
    )

    score_chunk_size = int(
        _setting(
            args.score_chunk_size,
            experiment,
            "score_chunk_size",
            5000,
        )
    )
    batch_size = int(
        _setting(
            args.batch_size,
            experiment,
            "batch_size",
            8,
        )
    )
    max_length = int(
        _setting(
            args.max_length,
            experiment,
            "max_length",
            64,
        )
    )
    seed = int(
        _setting(
            args.seed,
            experiment,
            "seed",
            0,
        )
    )
    device = str(
        _setting(
            args.device,
            experiment,
            "device",
            "cuda",
        )
    )

    train_total = (
        args.train_total
        if args.train_total is not None
        else experiment.get(
            "train_total"
        )
    )
    cal_total = (
        args.cal_total
        if args.cal_total is not None
        else experiment.get(
            "cal_total"
        )
    )

    if train_total is not None:
        train_total = int(
            train_total
        )
    if cal_total is not None:
        cal_total = int(
            cal_total
        )

    statement_config = load_statement_config(
        args.statement_config
    )
    template = (
        args.template
        or experiment.get(
            "template"
        )
        or get_default_variant(
            "joint",
            config=statement_config,
        )
    )

    if score_chunk_size <= 0:
        raise ValueError(
            "score_chunk_size must be positive."
        )
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )
    if max_length <= 0:
        raise ValueError(
            "max_length must be positive."
        )
    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be positive when provided."
        )

    data = load_probe_dataset(
        args.dataset,
        construction="atomic",
        data_dir=args.data_dir,
    )
    model_config = load_model_config(
        args.model_name,
        config_dir=args.model_config_dir,
    )

    selected_layers = {
        probe: _selected_layer(
            model_config,
            probe_name=probe,
            dataset=args.dataset,
        )
        for probe in probes
    }

    pair_path = (
        args.pairs_dir
        / args.model_name
        / f"{args.dataset}.parquet"
    )
    if not pair_path.exists():
        raise FileNotFoundError(
            f"Pair Parquet does not exist: {pair_path}"
        )

    pair_probe_frame = pd.read_parquet(
        pair_path,
        columns=["probe"],
    )

    if pair_probe_frame.empty:
        output_model_dir = (
            args.output_dir
            / args.model_name
        )
        output_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_model_dir
            / f"{args.dataset}.parquet"
        )
        bundle_path = (
            output_model_dir
            / f"{args.dataset}.joblib"
        )
        summary_path = (
            output_model_dir
            / f"{args.dataset}.json"
        )
        parts_dir = (
            output_model_dir
            / f".{args.dataset}.parts"
        )

        if args.overwrite:
            for path in (
                output_path,
                bundle_path,
                summary_path,
            ):
                if path.exists():
                    path.unlink()
            remove_part_tree(
                parts_dir
            )

        if (
            not args.resume
            and not args.overwrite
            and (
                output_path.exists()
                or bundle_path.exists()
                or summary_path.exists()
                or parts_dir.exists()
            )
        ):
            raise FileExistsError(
                "Joint outputs already exist. "
                "Use --resume or --overwrite."
            )

        if args.resume and summary_path.exists():
            existing_summary = _load_yaml(
                summary_path
            )
            if (
                existing_summary.get("status")
                == "degenerate"
                and existing_summary.get("reason")
                == "empty_pair_table"
            ):
                print(
                    f"Joint output already complete "
                    f"(degenerate empty-pair case): {summary_path}"
                )
                return

        skipped_probes = {
            probe_name: {
                "status": "degenerate",
                "reason": "no_pair_rows",
                "message": (
                    "No P x x pair rows exist for this probe, so joint "
                    "training/scoring is not applicable."
                ),
                "layer": int(selected_layers[probe_name]),
            }
            for probe_name in probes
        }

        summary_payload = {
            "schema_version": 1,
            "status": "degenerate",
            "reason": "empty_pair_table",
            "construction": "joint",
            "model_name": args.model_name,
            "dataset": args.dataset,
            "pair_path": str(
                pair_path
            ),
            "output_path": None,
            "bundle_path": None,
            "num_pairs": 0,
            "requested_probes": list(
                probes
            ),
            "active_probes": [],
            "skipped_probes": skipped_probes,
            "selected_layers": selected_layers,
        }

        write_json_atomic(
            summary_payload,
            summary_path,
        )

        print("=" * 72)
        print("JOINT PHASE")
        print("=" * 72)
        print(f"Model:             {args.model_name}")
        print(f"Dataset:           {args.dataset}")
        print(f"Probes requested:  {probes}")
        print("Probes with pairs: []")
        print(f"Pair table:        {pair_path}")
        print("Pair rows:         0")
        print("Status:            DEGENERATE (empty pair table)")
        print(f"Summary:           {summary_path}")
        print("=" * 72)
        return

    available_pair_probes = set(
        pair_probe_frame["probe"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    skipped_probes: dict[str, dict[str, Any]] = {}

    for probe_name in probes:
        if probe_name not in available_pair_probes:
            skipped_probes[probe_name] = {
                "status": "degenerate",
                "reason": "no_pair_rows",
                "message": (
                    "No P x x pair rows exist for this probe, so joint "
                    "training/scoring is not applicable."
                ),
                "layer": int(selected_layers[probe_name]),
            }

    run_probes = [
        probe_name
        for probe_name in probes
        if probe_name in available_pair_probes
    ]

    if not run_probes:
        raise RuntimeError(
            "None of the requested probes have rows in the pair table."
        )

    output_model_dir = (
        args.output_dir
        / args.model_name
    )
    output_path = (
        output_model_dir
        / f"{args.dataset}.parquet"
    )
    bundle_path = (
        output_model_dir
        / f"{args.dataset}.joblib"
    )
    summary_path = (
        output_model_dir
        / f"{args.dataset}.json"
    )
    parts_dir = (
        output_model_dir
        / f".{args.dataset}.parts"
    )

    if args.overwrite:
        for path in (
            output_path,
            bundle_path,
            summary_path,
        ):
            if path.exists():
                path.unlink()
        remove_part_tree(
            parts_dir
        )

    if (
        not args.resume
        and not args.overwrite
        and (
            output_path.exists()
            or bundle_path.exists()
            or summary_path.exists()
            or parts_dir.exists()
        )
    ):
        raise FileExistsError(
            "Joint outputs already exist. Use --resume or --overwrite."
        )

    joint_train = build_joint_training_data(
        data,
        train_total=train_total,
        cal_total=cal_total,
        seed=seed,
        template=str(
            template
        ),
        statement_config_path=str(
            args.statement_config
        ),
    )

    bundle = (
        _load_bundle(
            bundle_path
        )
        if args.resume
        else None
    )

    if bundle is None:
        bundle = _new_bundle(
            model_name=args.model_name,
            hf_model=str(
                model_config[
                    "model"
                ]
            ),
            dataset=args.dataset,
            template=str(
                template
            ),
            seed=seed,
            max_length=max_length,
        )
    else:
        expected_identity = {
            "construction": "joint",
            "model_name": args.model_name,
            "dataset": args.dataset,
            "template": str(
                template
            ),
            "seed": seed,
            "max_length": max_length,
        }
        mismatches = {
            key: (
                bundle.get(
                    key
                ),
                value,
            )
            for key, value in expected_identity.items()
            if bundle.get(
                key
            )
            != value
        }
        if mismatches:
            raise ValueError(
                f"Existing joint probe bundle does not match this run: "
                f"{mismatches}"
            )

    bundle.setdefault(
        "skipped_probes",
        {},
    )
    bundle[
        "skipped_probes"
    ].update(
        skipped_probes
    )

    probe_configs = {
        probe: load_probe_config(
            probe,
            config_dir=args.probe_config_dir,
        )
        for probe in probes
    }

    print("=" * 72)
    print("JOINT PHASE")
    print("=" * 72)
    print(f"Model:             {args.model_name}")
    print(f"HF model:          {model_config['model']}")
    print(f"Dataset:           {args.dataset}")
    print(f"Probes requested:  {probes}")
    print(f"Probes with pairs: {run_probes}")
    print(f"Selected layers:   {selected_layers}")
    print(f"Template:          {template}")
    print(
        "Joint train/cal:   "
        f"{joint_train.train_mask.sum():,} / "
        f"{joint_train.cal_mask.sum():,}"
    )
    print(f"Joint classes:     {list(JOINT_LABELS)}")
    print(f"Pair table:        {pair_path}")
    print(f"Output:            {output_path}")
    print(f"Probe bundle:      {bundle_path}")
    print(f"Score chunk size:  {score_chunk_size:,}")
    print(f"Resume:            {args.resume}")
    print("=" * 72)

    _set_seed(
        seed
    )
    model_bundle = None

    try:
        model_bundle = load_model_by_name(
            args.model_name,
            config_dir=args.model_config_dir,
            device=device,
        )

        missing_probe_models = [
            probe
            for probe in run_probes
            if (
                probe
                not in bundle.get(
                    "skipped_probes",
                    {},
                )
                and (
                probe
                not in bundle.get(
                    "probes",
                    {},
                )
                or int(
                    bundle.get(
                        "layers",
                        {},
                    ).get(
                        probe,
                        -1,
                    )
                )
                != int(
                    selected_layers[
                        probe
                    ]
                )
                )
            )
        ]

        if missing_probe_models:
            grouped = _group_by_layer(
                missing_probe_models,
                selected_layers,
            )

            print(
                "\nTraining new nine-class joint probes..."
            )

            for layer, layer_probes in grouped.items():
                print(
                    f"\nJoint training activations: "
                    f"layer {layer}, probes={layer_probes}"
                )

                _set_seed(
                    seed
                )
                activation_start = time.perf_counter()

                activations = collect_layer_activations(
                    model_bundle,
                    list(
                        joint_train.statements
                    ),
                    layer=layer,
                    batch_size=batch_size,
                    max_length=max_length,
                )

                activation_seconds = (
                    time.perf_counter()
                    - activation_start
                )

                print(
                    f"Collected joint training activations: "
                    f"shape={activations.values.shape}, "
                    f"RAM={activations.ram_gib:.2f} GiB, "
                    f"time={activation_seconds:.1f}s"
                )

                try:
                    for probe_name in layer_probes:
                        print(
                            f"[{probe_name}] fitting separate 9-class joint probe..."
                        )

                        _set_seed(
                            seed
                        )
                        probe_start = time.perf_counter()

                        probe = build_probe(
                            probe_name,
                            config=probe_configs[
                                probe_name
                            ],
                            seed=seed,
                        )

                        try:
                            probe.fit(
                                activations,
                                joint_train.labels,
                                joint_train.train_mask,
                                joint_train.cal_mask,
                            )
                        except ValueError as exc:
                            if not _is_degenerate_mean_difference_error(
                                probe_name,
                                exc,
                            ):
                                raise

                            message = str(exc)
                            print(
                                f"[{probe_name}] SKIPPED: degenerate "
                                f"mean-difference probe ({message})"
                            )
                            bundle.setdefault(
                                "skipped_probes",
                                {},
                            )[
                                probe_name
                            ] = {
                                "status": "degenerate",
                                "reason": "invalid_mean_difference_direction",
                                "message": message,
                                "layer": int(layer),
                            }
                            _write_joblib_atomic(
                                bundle,
                                bundle_path,
                            )
                            del probe
                            continue

                        if set(
                            np.asarray(
                                probe.classes_,
                                dtype=int,
                            ).tolist()
                        ) != set(
                            range(
                                len(JOINT_LABELS)
                            )
                        ):
                            raise RuntimeError(
                                f"{probe_name}: fitted joint probe classes "
                                f"{np.asarray(probe.classes_).tolist()} "
                                "do not contain all nine expected classes."
                            )

                        train_metrics = _evaluate_split(
                            probe=probe,
                            activations=activations,
                            labels=joint_train.labels,
                            mask=joint_train.train_mask,
                        )
                        cal_metrics = _evaluate_split(
                            probe=probe,
                            activations=activations,
                            labels=joint_train.labels,
                            mask=joint_train.cal_mask,
                        )

                        bundle.setdefault(
                            "probes",
                            {},
                        )[
                            probe_name
                        ] = probe
                        bundle.setdefault(
                            "layers",
                            {},
                        )[
                            probe_name
                        ] = int(
                            layer
                        )
                        bundle.setdefault(
                            "metrics",
                            {},
                        )[
                            probe_name
                        ] = {
                            "train": train_metrics,
                            "cal": cal_metrics,
                            "training_activation_seconds": float(
                                activation_seconds
                            ),
                            "probe_fit_seconds": float(
                                time.perf_counter()
                                - probe_start
                            ),
                        }
                        bundle[
                            "training_data"
                        ] = joint_train.summary()

                        _write_joblib_atomic(
                            bundle,
                            bundle_path,
                        )

                        print(
                            f"[{probe_name}] cal: "
                            f"accuracy={cal_metrics['accuracy']:.4f} | "
                            f"macro-F1={cal_metrics['macro_f1']:.4f} | "
                            f"log-loss={cal_metrics['log_loss']:.4f}"
                        )

                        del probe

                finally:
                    del activations
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        else:
            print(
                "\nAll requested joint probe models already exist in the "
                "bundle; skipping training."
            )

        score_summary: dict[str, Any] = {}

        active_probes = [
            probe_name
            for probe_name in run_probes
            if probe_name in bundle.get(
                "probes",
                {},
            )
        ]

        if not active_probes:
            raise RuntimeError(
                "No requested probe could be trained for joint scoring."
            )

        for probe_name in active_probes:
            probe = bundle[
                "probes"
            ][
                probe_name
            ]
            layer = int(
                selected_layers[
                    probe_name
                ]
            )

            pairs = _load_probe_pairs(
                pair_path,
                probe_name=probe_name,
                limit=args.limit,
            )

            completed = completed_rows_from_parts(
                parts_dir,
                probe_name=probe_name,
            )

            if completed > len(
                pairs
            ):
                raise RuntimeError(
                    f"{probe_name}: checkpoint parts contain {completed:,} "
                    f"rows but this run expects only {len(pairs):,}."
                )

            print(
                f"\n[{probe_name}] "
                + (
                    f"resuming at {completed:,}/{len(pairs):,} pairs."
                    if completed
                    else (
                        f"scoring {len(pairs):,} joint pairs "
                        f"at layer {layer}."
                    )
                )
            )

            part_index = next_part_index(
                parts_dir,
                probe_name=probe_name,
            )
            start = completed
            probe_start = time.perf_counter()

            while start < len(
                pairs
            ):
                end = min(
                    start
                    + score_chunk_size,
                    len(
                        pairs
                    ),
                )

                pair_chunk = pairs.iloc[
                    start:end
                ].copy()

                statements = _render_test_chunk(
                    pair_chunk,
                    data=data,
                    template=str(
                        template
                    ),
                    statement_config_path=args.statement_config,
                )

                score_frame = _score_chunk(
                    probe=probe,
                    probe_name=probe_name,
                    layer=layer,
                    pair_chunk=pair_chunk,
                    statements=statements,
                    model_bundle=model_bundle,
                    batch_size=batch_size,
                    max_length=max_length,
                )

                write_part_atomic(
                    score_frame,
                    base_dir=parts_dir,
                    probe_name=probe_name,
                    part_index=part_index,
                )

                start = end
                part_index += 1

                write_json_atomic(
                    {
                        "schema_version": 1,
                        "construction": "joint",
                        "model_name": args.model_name,
                        "dataset": args.dataset,
                        "template": str(
                            template
                        ),
                        "probe": probe_name,
                        "layer": layer,
                        "rows_scored": int(
                            start
                        ),
                        "rows_expected": int(
                            len(
                                pairs
                            )
                        ),
                        "complete": bool(
                            start
                            == len(
                                pairs
                            )
                        ),
                    },
                    output_model_dir
                    / f".{args.dataset}.{probe_name}.progress.json",
                )

                print(
                    f"[{probe_name}] {start:,}/{len(pairs):,} rows scored"
                )

            score_summary[
                probe_name
            ] = {
                "layer": layer,
                "rows": int(
                    len(
                        pairs
                    )
                ),
                "seconds": float(
                    time.perf_counter()
                    - probe_start
                ),
            }

        total_rows = consolidate_parts(
            base_dir=parts_dir,
            probe_order=active_probes,
            output_path=output_path,
            expected_columns=OUTPUT_COLUMNS,
        )

        expected_total = sum(
            item[
                "rows"
            ]
            for item in score_summary.values()
        )

        if total_rows != expected_total:
            raise RuntimeError(
                f"Consolidated {total_rows:,} joint score rows; "
                f"expected {expected_total:,}."
            )

        summary = {
            "schema_version": 1,
            "construction": "joint",
            "model_name": args.model_name,
            "hf_model": str(
                model_config[
                    "model"
                ]
            ),
            "dataset": args.dataset,
            "template": str(
                template
            ),
            "statement_config": str(
                args.statement_config
            ),
            "ordered": True,
            "first_operand": "x",
            "second_operand": "P",
            "joint_classes": list(
                JOINT_LABELS
            ),
            "selected_layers": selected_layers,
            "requested_probes": list(probes),
            "active_probes": list(active_probes),
            "skipped_probes": dict(
                bundle.get(
                    "skipped_probes",
                    {},
                )
            ),
            "training_data": joint_train.summary(),
            "score_chunk_size": score_chunk_size,
            "batch_size": batch_size,
            "max_length": max_length,
            "limit_per_probe": args.limit,
            "num_rows": int(
                total_rows
            ),
            "probes": score_summary,
        }

        write_json_atomic(
            summary,
            summary_path,
        )

        remove_part_tree(
            parts_dir
        )

        for probe_name in probes:
            progress_path = (
                output_model_dir
                / f".{args.dataset}.{probe_name}.progress.json"
            )
            if progress_path.exists():
                progress_path.unlink()

        print("\n" + "=" * 72)
        print("JOINT SCORING COMPLETE")
        print("=" * 72)
        print(f"Parquet:      {output_path}")
        print(f"Probe bundle: {bundle_path}")
        print(f"Summary:      {summary_path}")
        print(f"Rows:         {total_rows:,}")
        print()

        for probe_name in active_probes:
            info = score_summary[
                probe_name
            ]
            metrics = bundle[
                "metrics"
            ][
                probe_name
            ][
                "cal"
            ]
            print(
                f"{probe_name:18s} "
                f"layer={info['layer']:>2d} | "
                f"pairs={info['rows']:,} | "
                f"joint-cal log-loss={metrics['log_loss']:.6f}"
            )

        for probe_name, skip_info in bundle.get(
            "skipped_probes",
            {},
        ).items():
            if probe_name not in probes:
                continue
            print(
                f"{probe_name:18s} "
                f"SKIPPED ({skip_info['reason']})"
            )

    finally:
        if model_bundle is not None:
            del model_bundle

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
