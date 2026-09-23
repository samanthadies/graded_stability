from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


RESULT_KEY = ("layer", "probe")


def _require_parquet_engine() -> None:
    """Raise a clear message if neither supported pandas parquet engine exists."""
    try:
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import fastparquet  # noqa: F401
        return
    except ImportError as exc:
        raise ImportError(
            "Parquet output requires pyarrow or fastparquet. "
            "Install one in the environment, e.g. `pip install pyarrow`."
        ) from exc


def read_parquet_if_exists(path: str | Path) -> pd.DataFrame:
    """Read a result table or return an empty dataframe."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    _require_parquet_engine()
    return pd.read_parquet(path)


def write_parquet_atomic(
    frame: pd.DataFrame,
    path: str | Path,
) -> None:
    """Atomically replace one Parquet file."""
    _require_parquet_engine()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        frame.to_parquet(
            temp_path,
            index=False,
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def upsert_result_rows(
    path: str | Path,
    rows: Iterable[dict],
) -> pd.DataFrame:
    """
    Insert/replace result rows keyed by (layer, probe), then checkpoint.

    A later successful retry therefore replaces an earlier error row for the
    same layer/probe pair.
    """
    rows = list(rows)
    if not rows:
        return read_parquet_if_exists(path)

    new = pd.DataFrame(rows)
    missing = [
        key for key in RESULT_KEY if key not in new.columns
    ]
    if missing:
        raise ValueError(
            f"Result rows are missing key columns {missing}."
        )

    existing = read_parquet_if_exists(path)

    if existing.empty:
        combined = new
    else:
        for key in RESULT_KEY:
            if key not in existing.columns:
                raise ValueError(
                    f"Existing result file lacks key column {key!r}."
                )

        incoming_keys = set(
            zip(
                new["layer"].astype(int),
                new["probe"].astype(str),
            )
        )
        keep = [
            (int(layer), str(probe)) not in incoming_keys
            for layer, probe in zip(
                existing["layer"],
                existing["probe"],
            )
        ]

        combined = pd.concat(
            [
                existing.loc[keep],
                new,
            ],
            ignore_index=True,
            sort=False,
        )

    combined = combined.sort_values(
        ["layer", "probe"],
        kind="stable",
    ).reset_index(drop=True)

    write_parquet_atomic(
        combined,
        path,
    )
    return combined


def completed_pairs(
    frame: pd.DataFrame,
) -> set[tuple[int, str]]:
    """Return (layer, probe) pairs whose status is complete."""
    if frame.empty:
        return set()

    required = {"layer", "probe", "status"}
    if not required.issubset(frame.columns):
        return set()

    complete = frame[
        frame["status"].astype(str) == "complete"
    ]

    return {
        (int(layer), str(probe))
        for layer, probe in zip(
            complete["layer"],
            complete["probe"],
        )
    }
