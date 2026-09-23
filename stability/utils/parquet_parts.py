from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_json_atomic(
    payload: dict,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def probe_part_dir(
    base_dir: str | Path,
    *,
    probe_name: str,
) -> Path:
    return (
        Path(base_dir)
        / probe_name
    )


def list_probe_parts(
    base_dir: str | Path,
    *,
    probe_name: str,
) -> list[Path]:
    directory = probe_part_dir(
        base_dir,
        probe_name=probe_name,
    )
    if not directory.exists():
        return []

    return sorted(
        directory.glob(
            "part-*.parquet"
        )
    )


def completed_rows_from_parts(
    base_dir: str | Path,
    *,
    probe_name: str,
) -> int:
    """Count valid rows already checkpointed for one probe."""
    total = 0

    for path in list_probe_parts(
        base_dir,
        probe_name=probe_name,
    ):
        parquet = pq.ParquetFile(
            path
        )
        total += int(
            parquet.metadata.num_rows
        )

    return total


def next_part_index(
    base_dir: str | Path,
    *,
    probe_name: str,
) -> int:
    parts = list_probe_parts(
        base_dir,
        probe_name=probe_name,
    )
    if not parts:
        return 0

    stem = parts[-1].stem
    return int(
        stem.split("-")[-1]
    ) + 1


def write_part_atomic(
    frame: pd.DataFrame,
    *,
    base_dir: str | Path,
    probe_name: str,
    part_index: int,
) -> Path:
    """Write one crash-safe temporary score part."""
    directory = probe_part_dir(
        base_dir,
        probe_name=probe_name,
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / f"part-{part_index:06d}.parquet"
    )
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        frame.to_parquet(
            temp_path,
            index=False,
            compression="zstd",
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return path


def consolidate_parts(
    *,
    base_dir: str | Path,
    probe_order: Iterable[str],
    output_path: str | Path,
    expected_columns: list[str],
) -> int:
    """
    Consolidate all probe parts into one Parquet, preserving probe order.
    """
    base_dir = Path(
        base_dir
    )
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    try:
        for probe_name in probe_order:
            parts = list_probe_parts(
                base_dir,
                probe_name=probe_name,
            )
            for part_path in parts:
                table = pq.read_table(
                    part_path
                )

                if table.column_names != expected_columns:
                    raise ValueError(
                        f"{part_path}: columns {table.column_names} do not "
                        f"match expected {expected_columns}."
                    )

                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path,
                        table.schema,
                        compression="zstd",
                        use_dictionary=[
                            "probe",
                            "pred_label",
                        ],
                        write_statistics=True,
                    )

                writer.write_table(
                    table
                )
                total_rows += table.num_rows

        if writer is None:
            raise RuntimeError(
                "No temporary score parts were available to consolidate."
            )

        writer.close()
        writer = None

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()

    return int(
        total_rows
    )


def remove_part_tree(
    base_dir: str | Path,
) -> None:
    base_dir = Path(
        base_dir
    )
    if base_dir.exists():
        shutil.rmtree(
            base_dir
        )
