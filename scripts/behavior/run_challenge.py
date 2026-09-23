from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stability.behavior.config import ChallengeConfig, load_challenge_config
from stability.behavior.rendering import render_behavior_prompt
from stability.behavior.scoring import score_binary_prompts
from stability.models.loading import load_model_by_name
from stability.utils.io import read_parquet_if_exists, write_parquet_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavioral challenge validation.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=("cities_loc", "med_indications", "defs")
    )
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument(
        "--challenge_config", type=Path, default=Path("configs/experiments/challenge.yaml")
    )
    parser.add_argument(
        "--model_config_dir", type=Path, default=Path("configs/model")
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/behavior/challenge")
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _validate_input(
    frame: pd.DataFrame,
    *,
    model_name: str,
    dataset: str,
    config: ChallengeConfig,
) -> pd.DataFrame:
    required = {
        "model_name", "dataset", "statement_id", "statement",
        "challenge_1_id", "challenge_2_id", "challenge_3_id",
        "challenge_sequence", "challenge_config_fingerprint",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Challenge set missing columns {sorted(missing)}.")
    if set(frame["model_name"].astype(str).unique()) != {model_name}:
        raise ValueError("Challenge-set model identity mismatch.")
    if set(frame["dataset"].astype(str).unique()) != {dataset}:
        raise ValueError("Challenge-set dataset identity mismatch.")
    if set(frame["challenge_config_fingerprint"].astype(str).unique()) != {config.fingerprint}:
        raise ValueError("Challenge config changed; rebuild the challenge set.")
    frame = frame.copy()
    frame["statement_id"] = pd.to_numeric(frame["statement_id"], errors="raise").astype(np.int64)
    if frame["statement_id"].duplicated().any():
        raise ValueError("Challenge-set statement IDs are not unique.")
    return frame.sort_values("statement_id", kind="stable").reset_index(drop=True)


def _validate_existing(existing: pd.DataFrame, challenge_set: pd.DataFrame) -> set[int]:
    if existing.empty:
        return set()
    required = {"statement_id", "round_3_pred_label", "challenge_sequence"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(f"Existing output cannot resume; missing {sorted(missing)}.")
    existing = existing.copy()
    existing["statement_id"] = pd.to_numeric(existing["statement_id"], errors="raise").astype(np.int64)
    if existing["statement_id"].duplicated().any():
        raise ValueError("Existing behavioral output has duplicate statement IDs.")
    expected = challenge_set.set_index("statement_id")["challenge_sequence"].astype(str)
    for _, row in existing.iterrows():
        statement_id = int(row["statement_id"])
        if statement_id not in expected.index:
            raise ValueError(f"Existing statement_id={statement_id} is absent from input.")
        if str(row["challenge_sequence"]) != str(expected.loc[statement_id]):
            raise ValueError(f"Challenge sequence changed for statement_id={statement_id}.")
    return set(
        existing.loc[existing["round_3_pred_label"].notna(), "statement_id"].astype(np.int64)
    )


def _upsert(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = new_rows.copy()
    else:
        new_ids = set(new_rows["statement_id"].astype(np.int64))
        combined = pd.concat(
            [existing.loc[~existing["statement_id"].astype(np.int64).isin(new_ids)], new_rows],
            ignore_index=True,
            sort=False,
        )
    return combined.sort_values("statement_id", kind="stable").reset_index(drop=True)


def _batch_output(batch: pd.DataFrame, round_scores: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "model_name", "dataset", "statement_id", "source_dataset", "source_row",
        "challenge_1_id", "challenge_2_id", "challenge_3_id", "challenge_sequence",
        "challenge_config", "challenge_config_fingerprint",
    ]
    out = batch[[column for column in columns if column in batch.columns]].reset_index(drop=True).copy()
    for round_number, scores in enumerate(round_scores):
        out = pd.concat(
            [
                out,
                scores.rename(
                    columns={column: f"round_{round_number}_{column}" for column in scores.columns}
                ).reset_index(drop=True),
            ],
            axis=1,
        )

    initial = out["round_0_pred_label"].astype(str).str.lower()
    for round_number in range(4):
        out[f"initial_answer_support_round_{round_number}"] = np.where(
            initial.eq("true"),
            out[f"round_{round_number}_prob_true"],
            out[f"round_{round_number}_prob_false"],
        )
    changed = []
    for challenge_round in (1, 2, 3):
        column = f"round_{challenge_round}_changed_from_previous"
        out[column] = (
            out[f"round_{challenge_round}_pred_label"].astype(str).str.lower()
            != out[f"round_{challenge_round - 1}_pred_label"].astype(str).str.lower()
        )
        changed.append(column)
        out[f"support_loss_round_{challenge_round}"] = (
            out["initial_answer_support_round_0"]
            - out[f"initial_answer_support_round_{challenge_round}"]
        )
        out[f"incremental_support_loss_round_{challenge_round}"] = (
            out[f"initial_answer_support_round_{challenge_round - 1}"]
            - out[f"initial_answer_support_round_{challenge_round}"]
        )
        out[f"absolute_movement_round_{challenge_round}"] = (
            out[f"initial_answer_support_round_{challenge_round}"]
            - out[f"initial_answer_support_round_{challenge_round - 1}"]
        ).abs()
    out["ever_changed"] = out[changed].any(axis=1)
    out["num_answer_changes"] = out[changed].sum(axis=1).astype(int)
    out["mean_support_loss"] = out[
        ["support_loss_round_1", "support_loss_round_2", "support_loss_round_3"]
    ].mean(axis=1)
    out["final_support_loss"] = out["support_loss_round_3"]
    out["max_support_loss"] = out[
        ["support_loss_round_1", "support_loss_round_2", "support_loss_round_3"]
    ].max(axis=1)
    movement_columns = [
        "absolute_movement_round_1",
        "absolute_movement_round_2",
        "absolute_movement_round_3",
    ]
    out["mean_absolute_movement"] = out[movement_columns].mean(axis=1)
    out["total_absolute_movement"] = out[movement_columns].sum(axis=1)
    return out


def _empty_behavior_output(challenge_set: pd.DataFrame) -> pd.DataFrame:
    """Return a schema-valid empty behavioral output for degenerate P sets."""
    base_columns = [
        "model_name", "dataset", "statement_id", "source_dataset", "source_row",
        "challenge_1_id", "challenge_2_id", "challenge_3_id", "challenge_sequence",
        "challenge_config", "challenge_config_fingerprint",
    ]
    columns = [column for column in base_columns if column in challenge_set.columns]
    out = challenge_set[columns].head(0).copy()
    for round_number in range(4):
        for suffix, dtype in (
            ("pred_label", "object"),
            ("prob_true", "float64"),
            ("prob_false", "float64"),
            ("log_score_true", "float64"),
            ("log_score_false", "float64"),
        ):
            out[f"round_{round_number}_{suffix}"] = pd.Series(dtype=dtype)
        out[f"initial_answer_support_round_{round_number}"] = pd.Series(dtype="float64")
    for challenge_round in (1, 2, 3):
        out[f"round_{challenge_round}_changed_from_previous"] = pd.Series(dtype="bool")
        out[f"support_loss_round_{challenge_round}"] = pd.Series(dtype="float64")
        out[f"incremental_support_loss_round_{challenge_round}"] = pd.Series(dtype="float64")
        out[f"absolute_movement_round_{challenge_round}"] = pd.Series(dtype="float64")
    out["ever_changed"] = pd.Series(dtype="bool")
    out["num_answer_changes"] = pd.Series(dtype="int64")
    out["mean_support_loss"] = pd.Series(dtype="float64")
    out["final_support_loss"] = pd.Series(dtype="float64")
    out["max_support_loss"] = pd.Series(dtype="float64")
    out["mean_absolute_movement"] = pd.Series(dtype="float64")
    out["total_absolute_movement"] = pd.Series(dtype="float64")
    return out


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.save_every <= 0 or args.max_length <= 1:
        raise ValueError("Invalid batch/save/max_length setting.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    config = load_challenge_config(args.challenge_config)
    input_path = args.input_path or (
        Path("outputs/behavior/sets") / args.model_name / args.dataset / "general.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    challenge_set = _validate_input(
        pd.read_parquet(input_path),
        model_name=args.model_name,
        dataset=args.dataset,
        config=config,
    )
    if args.limit is not None:
        challenge_set = challenge_set.head(args.limit).copy()

    model_dir = args.output_dir / args.model_name
    output_path = model_dir / f"{args.dataset}.parquet"
    summary_path = model_dir / f"{args.dataset}.json"
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if not args.resume and not args.overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError("Behavior output exists. Use --resume or --overwrite.")

    if challenge_set.empty:
        empty = _empty_behavior_output(challenge_set)
        write_parquet_atomic(empty, output_path)
        _write_json(
            {
                "schema_version": 2,
                "status": "skipped_degenerate_no_P",
                "model_name": args.model_name,
                "dataset": args.dataset,
                "challenge_config": config.name,
                "challenge_config_fingerprint": config.fingerprint,
                "input_path": str(input_path),
                "rows_expected": 0,
                "rows_complete": 0,
            },
            summary_path,
        )
        print("No P statements are available; wrote schema-valid empty behavior output.")
        return

    existing = read_parquet_if_exists(output_path) if args.resume else pd.DataFrame()
    completed = _validate_existing(existing, challenge_set)
    pending = challenge_set.loc[~challenge_set["statement_id"].isin(completed)].copy()

    print("=" * 72)
    print("BEHAVIORAL CHALLENGE EXPERIMENT")
    print("=" * 72)
    print(f"Model:       {args.model_name}")
    print(f"Dataset:     {args.dataset}")
    print(f"Config:      {config.name}")
    print(f"Statements:  {len(challenge_set):,}")
    print(f"Completed:   {len(completed):,}")
    print(f"Remaining:   {len(pending):,}")
    print(f"Output:      {output_path}")
    print(f"Resume:      {args.resume}")
    print("=" * 72)
    if pending.empty:
        print("All requested statements are already complete.")
        return

    bundle = None
    try:
        bundle = load_model_by_name(
            args.model_name, config_dir=args.model_config_dir, device=args.device
        )
        buffered: list[pd.DataFrame] = []
        since_save = 0
        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start : start + args.batch_size].copy()
            statements = batch["statement"].astype(str).tolist()
            sequences = [
                [str(row["challenge_1_id"]), str(row["challenge_2_id"]), str(row["challenge_3_id"])]
                for _, row in batch.iterrows()
            ]
            histories: list[list[str]] = [[] for _ in statements]
            round_scores: list[pd.DataFrame] = []
            for _round in range(4):
                prompts = [
                    render_behavior_prompt(
                        statement=statement,
                        prior_answers=history,
                        challenge_sequence=sequence,
                        tokenizer=bundle.tokenizer,
                        instruct=bundle.instruct,
                        config=config,
                    )
                    for statement, history, sequence in zip(statements, histories, sequences)
                ]
                scores = score_binary_prompts(
                    prompts=prompts,
                    bundle=bundle,
                    answer_options=config.answer_options,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                round_scores.append(scores)
                for row_index, answer in enumerate(scores["pred_label"].astype(str)):
                    histories[row_index].append(answer)

            batch_output = _batch_output(batch, round_scores)
            buffered.append(batch_output)
            since_save += len(batch_output)
            processed = min(start + len(batch), len(pending))
            print(f"Scored {processed:,}/{len(pending):,} remaining statements.")

            if since_save >= args.save_every:
                existing = _upsert(existing, pd.concat(buffered, ignore_index=True, sort=False))
                write_parquet_atomic(existing, output_path)
                buffered = []
                since_save = 0
                _write_json(
                    {
                        "schema_version": 2,
                        "status": "in_progress",
                        "model_name": args.model_name,
                        "dataset": args.dataset,
                        "challenge_config": config.name,
                        "challenge_config_fingerprint": config.fingerprint,
                        "input_path": str(input_path),
                        "rows_expected": int(len(challenge_set)),
                        "rows_complete": int(len(existing)),
                    },
                    summary_path,
                )
                print(f"Checkpointed {len(existing):,} rows -> {output_path}")

        if buffered:
            existing = _upsert(existing, pd.concat(buffered, ignore_index=True, sort=False))
            write_parquet_atomic(existing, output_path)
        if len(existing) != len(challenge_set):
            raise RuntimeError(
                f"Final behavior output has {len(existing):,} rows; expected {len(challenge_set):,}."
            )
        _write_json(
            {
                "schema_version": 2,
                "status": "complete",
                "model_name": args.model_name,
                "hf_model": bundle.hf_name,
                "instruct": bundle.instruct,
                "dataset": args.dataset,
                "challenge_config": config.name,
                "challenge_config_fingerprint": config.fingerprint,
                "input_path": str(input_path),
                "rows_expected": int(len(challenge_set)),
                "rows_complete": int(len(existing)),
                "batch_size": int(args.batch_size),
                "max_length": int(args.max_length),
            },
            summary_path,
        )
        print("\n" + "=" * 72)
        print("BEHAVIORAL CHALLENGE COMPLETE")
        print("=" * 72)
        print(f"Parquet:  {output_path}")
        print(f"Summary:  {summary_path}")
        print(f"Rows:     {len(existing):,}")
        print("\nInitial labels:")
        print(existing["round_0_pred_label"].value_counts().to_string())
        print("\nEver changed:")
        print(existing["ever_changed"].value_counts().to_string())
    finally:
        if bundle is not None:
            del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
