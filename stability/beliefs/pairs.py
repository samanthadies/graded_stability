"""
Validates model-specific belief/conditioning sets and constructs the probe-specific
(P, x) pair tables used by the conditional and joint estimation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PAIR_SCHEMA = pa.schema(
    [
        pa.field("probe", pa.string(), nullable=False),
        pa.field("pair_id", pa.int64(), nullable=False),
        pa.field("P_id", pa.int64(), nullable=False),
        pa.field("x_id", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ProbePairSummary:
    probe: str
    num_atomic_rows: int
    num_P: int
    num_x: int
    num_pairs: int
    exclude_self: bool
    status: str = "complete"
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe,
            "num_atomic_rows": self.num_atomic_rows,
            "num_P": self.num_P,
            "num_x": self.num_x,
            "num_pairs": self.num_pairs,
            "exclude_self": self.exclude_self,
            "status": self.status,
            "reason": self.reason,
        }


def _coerce_bool(series: pd.Series, *, name: str) -> np.ndarray:
    """Convert a membership column to a strict boolean ndarray."""
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name!r} contains missing values.")
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(
            f"{name!r} contains non-boolean values: {unknown}."
        )

    return normalized.map(mapping).to_numpy(dtype=bool)


def validate_atomic_selection(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the enriched atomic long table produced by define_sets.py.

    statement_id is required to be unique only within probe because the same
    atomic statement appears once for every probe.
    """
    required = {
        "model_name",
        "dataset",
        "probe",
        "statement_id",
        "is_P",
        "is_x",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Atomic table is missing required columns {sorted(missing)}."
        )

    if frame.empty:
        raise ValueError("Atomic table is empty.")

    if frame["statement_id"].isna().any():
        raise ValueError("'statement_id' contains missing values.")

    validated_parts: list[pd.DataFrame] = []

    for probe_name, probe_frame in frame.groupby(
        "probe",
        sort=True,
        dropna=False,
    ):
        probe_name = str(probe_name)
        probe_frame = probe_frame.copy()

        if probe_frame["statement_id"].duplicated().any():
            examples = (
                probe_frame.loc[
                    probe_frame["statement_id"].duplicated(),
                    "statement_id",
                ]
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"{probe_name}: statement_id is not unique within probe; "
                f"examples={examples}."
            )

        probe_frame["is_P"] = _coerce_bool(
            probe_frame["is_P"],
            name=f"{probe_name}.is_P",
        )
        probe_frame["is_x"] = _coerce_bool(
            probe_frame["is_x"],
            name=f"{probe_name}.is_x",
        )

        if (
            ~probe_frame.loc[
                probe_frame["is_P"],
                "is_x",
            ]
        ).any():
            raise ValueError(
                f"{probe_name}: every P statement must also belong to x."
            )

        validated_parts.append(probe_frame)

    reference_ids: np.ndarray | None = None
    reference_probe: str | None = None

    for probe_frame in validated_parts:
        probe_name = str(probe_frame["probe"].iloc[0])
        ids = np.sort(
            pd.to_numeric(
                probe_frame["statement_id"],
                errors="raise",
            ).to_numpy(dtype=np.int64)
        )

        if reference_ids is None:
            reference_ids = ids
            reference_probe = probe_name
        elif not np.array_equal(reference_ids, ids):
            raise ValueError(
                "Probes do not contain the same atomic statement IDs: "
                f"{reference_probe!r} differs from {probe_name!r}."
            )

    return pd.concat(
        validated_parts,
        ignore_index=True,
        sort=False,
    )


def iter_pair_chunks(
    probe_frame: pd.DataFrame,
    *,
    probe_name: str,
    exclude_self: bool = True,
    chunk_size: int = 100_000,
    limit: int | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield compact pair chunks for one probe without materializing P x x.

    Pair order matches the legacy deterministic order:
      P statements sorted by statement_id, then x statements sorted by
      statement_id. pair_id starts at zero within each probe.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    P_ids = np.sort(
        pd.to_numeric(
            probe_frame.loc[
                probe_frame["is_P"],
                "statement_id",
            ],
            errors="raise",
        ).to_numpy(dtype=np.int64)
    )
    x_ids = np.sort(
        pd.to_numeric(
            probe_frame.loc[
                probe_frame["is_x"],
                "statement_id",
            ],
            errors="raise",
        ).to_numpy(dtype=np.int64)
    )

    pair_id = 0

    buffer_probe: list[np.ndarray] = []
    buffer_pair_id: list[np.ndarray] = []
    buffer_P_id: list[np.ndarray] = []
    buffer_x_id: list[np.ndarray] = []
    buffered_rows = 0

    def flush() -> pd.DataFrame | None:
        nonlocal buffer_probe
        nonlocal buffer_pair_id
        nonlocal buffer_P_id
        nonlocal buffer_x_id
        nonlocal buffered_rows

        if buffered_rows == 0:
            return None

        chunk = pd.DataFrame(
            {
                "probe": np.concatenate(buffer_probe),
                "pair_id": np.concatenate(buffer_pair_id),
                "P_id": np.concatenate(buffer_P_id),
                "x_id": np.concatenate(buffer_x_id),
            }
        )

        buffer_probe = []
        buffer_pair_id = []
        buffer_P_id = []
        buffer_x_id = []
        buffered_rows = 0

        return chunk

    for P_id in P_ids:
        current_x = x_ids

        if exclude_self:
            current_x = current_x[
                current_x != P_id
            ]

        if limit is not None:
            remaining = int(limit) - pair_id
            if remaining <= 0:
                break
            if len(current_x) > remaining:
                current_x = current_x[:remaining]

        start = 0

        while start < len(current_x):
            available = chunk_size - buffered_rows
            take = min(
                available,
                len(current_x) - start,
            )

            x_slice = current_x[
                start : start + take
            ]
            n = len(x_slice)

            buffer_probe.append(
                np.full(
                    n,
                    probe_name,
                    dtype=object,
                )
            )
            buffer_pair_id.append(
                np.arange(
                    pair_id,
                    pair_id + n,
                    dtype=np.int64,
                )
            )
            buffer_P_id.append(
                np.full(
                    n,
                    int(P_id),
                    dtype=np.int64,
                )
            )
            buffer_x_id.append(
                x_slice.astype(
                    np.int64,
                    copy=False,
                )
            )

            pair_id += n
            buffered_rows += n
            start += n

            if buffered_rows >= chunk_size:
                chunk = flush()
                if chunk is not None:
                    yield chunk

        if limit is not None and pair_id >= int(limit):
            break

    chunk = flush()
    if chunk is not None:
        yield chunk


def expected_pair_count(
    probe_frame: pd.DataFrame,
    *,
    exclude_self: bool,
    limit: int | None,
) -> int:
    """Compute expected pair count without constructing the Cartesian product."""
    num_P = int(
        probe_frame["is_P"].sum()
    )
    num_x = int(
        probe_frame["is_x"].sum()
    )

    total = num_P * num_x

    if exclude_self:
        total -= num_P

    if limit is not None:
        total = min(
            total,
            int(limit),
        )

    return int(total)


def write_pair_parquet(
    atomic: pd.DataFrame,
    *,
    output_path: str | Path,
    probes: list[str] | tuple[str, ...] | None = None,
    exclude_self: bool = True,
    chunk_size: int = 100_000,
    limit: int | None = None,
    compression: str = "zstd",
) -> dict[str, ProbePairSummary]:
    """
    Stream all requested probe pair tables into one compact Parquet file.

    The file is ordered by probe and pair_id. Because chunks are written one
    probe at a time, predicate reads on `probe` can skip unrelated row groups.
    """
    atomic = validate_atomic_selection(
        atomic
    )

    available_probes = sorted(
        atomic["probe"]
        .astype(str)
        .unique()
        .tolist()
    )

    if probes is None:
        requested = available_probes
    else:
        requested = list(
            dict.fromkeys(
                str(probe)
                for probe in probes
            )
        )
        missing = [
            probe
            for probe in requested
            if probe not in available_probes
        ]
        if missing:
            raise ValueError(
                f"Requested probes not present in atomic table: {missing}. "
                f"Available={available_probes}."
            )

    if not requested:
        raise ValueError("No probes were requested.")

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    summaries: dict[str, ProbePairSummary] = {}
    writer: pq.ParquetWriter | None = None

    try:
        writer = pq.ParquetWriter(
            temp_path,
            PAIR_SCHEMA,
            compression=compression,
            use_dictionary=["probe"],
            write_statistics=True,
        )

        for probe_name in requested:
            probe_frame = atomic[
                atomic["probe"].astype(str)
                == probe_name
            ].copy()

            expected = expected_pair_count(
                probe_frame,
                exclude_self=exclude_self,
                limit=limit,
            )

            written = 0

            for chunk in iter_pair_chunks(
                probe_frame,
                probe_name=probe_name,
                exclude_self=exclude_self,
                chunk_size=chunk_size,
                limit=limit,
            ):
                table = pa.Table.from_pandas(
                    chunk,
                    schema=PAIR_SCHEMA,
                    preserve_index=False,
                )
                writer.write_table(
                    table,
                    row_group_size=len(chunk),
                )
                written += len(chunk)

            if written != expected:
                raise RuntimeError(
                    f"{probe_name}: wrote {written:,} pairs but expected "
                    f"{expected:,}."
                )

            num_P = int(
                probe_frame["is_P"].sum()
            )
            num_x = int(
                probe_frame["is_x"].sum()
            )

            if num_P == 0:
                status = "degenerate"
                reason = "no_P_statements"
            elif num_x == 0:
                status = "degenerate"
                reason = "no_x_statements"
            elif written == 0:
                status = "empty"
                reason = "no_pairs_after_exclusions"
            else:
                status = "complete"
                reason = None

            summaries[probe_name] = ProbePairSummary(
                probe=probe_name,
                num_atomic_rows=int(
                    len(probe_frame)
                ),
                num_P=num_P,
                num_x=num_x,
                num_pairs=written,
                exclude_self=exclude_self,
                status=status,
                reason=reason,
            )

        writer.close()
        writer = None

        temp_path.replace(
            output_path
        )

    finally:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()

    return summaries
