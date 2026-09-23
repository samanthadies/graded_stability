from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from stability.utils.io import write_parquet_atomic

DATASETS = ("cities_loc", "med_indications", "defs")
PROBES = ("sawmil", "svm", "mean_difference")
MEASUREMENT_COLUMNS = [
    "P_T", "P_N", "P_F",
    "x_T", "x_N", "x_F",
    "C_T", "C_N", "C_F",
]
PROJECTED_COLUMNS = [f"{c}_coherent" for c in MEASUREMENT_COLUMNS]
ADJUSTMENT_COLUMNS = [f"delta_{c}" for c in MEASUREMENT_COLUMNS]
JOINT_SCORE_COLUMNS = [
    # The first symbol is x, the second is P.
    "prob_TT", "prob_TF", "prob_TN",
    "prob_FT", "prob_FF", "prob_FN",
    "prob_NT", "prob_NF", "prob_NN",
]
CANONICAL_JOINT_COLUMNS = [
    "j_PT_xT", "j_PT_xN", "j_PT_xF",
    "j_PN_xT", "j_PN_xN", "j_PN_xF",
    "j_PF_xT", "j_PF_xN", "j_PF_xF",
]

HALFSPACE_NORMALS = np.zeros((4, 9), dtype=float)
HALFSPACE_NORMALS[0, [0, 6]] = [-1.0, 1.0]       # C_T <= P_T
HALFSPACE_NORMALS[1, [2, 8]] = [-1.0, 1.0]       # C_F <= P_F
HALFSPACE_NORMALS[2, [5, 7]] = [1.0, -1.0]       # x_F <= C_N
HALFSPACE_NORMALS[3, [1, 5, 7]] = [-1.0, -1.0, 1.0]  # C_N <= P_N+x_F
HALFSPACE_NORM_SQUARED = np.sum(HALFSPACE_NORMALS ** 2, axis=1)
VIOLATION_NAMES = (
    "C_T_le_P_T",
    "C_F_le_P_F",
    "x_F_le_C_N",
    "C_N_le_P_N_plus_x_F",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze trivalent CCK coherence.")
    p.add_argument("--repo_root", type=Path, default=Path("."))
    p.add_argument("--model_list", type=Path, default=Path("configs/model_list.yaml"))
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument(
        "--probe",
        nargs="+",
        action="extend",
        choices=PROBES,
        default=None,
        help=(
            "Probe(s) to analyze. May be supplied once with multiple values "
            "or repeated; e.g. --probe sawmil svm mean_difference."
        ),
    )
    p.add_argument("--atomic_dir", type=Path, default=Path("outputs/atomic"))
    p.add_argument("--pairs_dir", type=Path, default=Path("outputs/pairs"))
    p.add_argument("--conditional_dir", type=Path, default=Path("outputs/conditional"))
    p.add_argument("--joint_dir", type=Path, default=Path("outputs/joint"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/analysis/coherence"))
    p.add_argument("--min_pairs", type=int, default=50)
    p.add_argument("--limit_pairs", type=int, default=None)
    p.add_argument("--exact_tolerance", type=float, default=1e-8)
    p.add_argument("--projection_batch_size", type=int, default=50000)
    p.add_argument("--projection_max_iterations", type=int, default=250)
    p.add_argument("--projection_convergence_tolerance", type=float, default=1e-10)
    p.add_argument("--projection_feasibility_tolerance", type=float, default=1e-7)
    p.add_argument("--paired_tie_tolerance", type=float, default=1e-8)
    p.add_argument("--sample_size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_pair_level", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.min_pairs < 1:
        p.error("--min_pairs must be positive")
    if args.limit_pairs is not None and args.limit_pairs < 1:
        p.error("--limit_pairs must be positive")
    if args.projection_batch_size < 1 or args.projection_max_iterations < 1:
        p.error("projection batch size / iterations must be positive")
    if args.sample_size < 0:
        p.error("--sample_size must be nonnegative")
    if args.probe is None:
        args.probe = ["sawmil"]
    return args


def under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_model_list(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]
    if isinstance(raw, dict):
        models = [str(k) for k in raw]
    elif isinstance(raw, list):
        models = []
        for item in raw:
            if isinstance(item, str):
                models.append(item)
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Unsupported model-list entry: {item!r}")
            value = next((item[k] for k in ("name", "config", "model_name", "key") if k in item), None)
            if value is None and len(item) == 1:
                value = next(iter(item))
            if value is None:
                raise ValueError(f"Cannot infer model key from {item!r}")
            models.append(str(value))
    else:
        raise ValueError(f"Unsupported model-list format in {path}")
    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode()
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


def guard_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n  " + "\n  ".join(map(str, existing)))


def normalize_rows(values: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{context}: expected 2D array, got {arr.shape}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{context}: non-finite probabilities")
    if np.any(arr < -1e-8):
        raise ValueError(f"{context}: negative probability {arr.min():.6g}")
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError(f"{context}: zero-mass probability row")
    return arr / totals[:, None]


def find_triplet(frame: pd.DataFrame, context: str) -> tuple[str, str, str]:
    candidates = (
        ("prob_true", "prob_neither", "prob_false"),
        ("prob_T", "prob_N", "prob_F"),
        ("p_true", "p_neither", "p_false"),
        ("p_T", "p_N", "p_F"),
    )
    for triplet in candidates:
        if set(triplet).issubset(frame.columns):
            return triplet
    raise KeyError(f"{context}: no T/N/F probability triplet; columns={frame.columns.tolist()}")


def find_id(frame: pd.DataFrame, candidates: tuple[str, ...], context: str) -> str:
    for c in candidates:
        if c in frame.columns:
            return c
    raise KeyError(f"{context}: no identifier among {candidates}; columns={frame.columns.tolist()}")


def probe_rows(frame: pd.DataFrame, probe: str, context: str) -> pd.DataFrame:
    if "probe" not in frame.columns:
        raise KeyError(f"{context}: missing probe column")
    return frame.loc[frame["probe"].astype(str).eq(probe)].copy()


def load_atomic_lookup(path: Path, probe: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = probe_rows(pd.read_parquet(path), probe, str(path))
    if frame.empty:
        return pd.DataFrame(columns=["atomic_T", "atomic_N", "atomic_F"], index=pd.Index([], name="statement_id"))
    id_col = find_id(frame, ("statement_id", "P_id", "id"), str(path))
    T, N, F = find_triplet(frame, f"{path}/{probe}")
    out = frame[[id_col, T, N, F]].copy()
    out["statement_id"] = out[id_col].astype(str)
    out[["atomic_T", "atomic_N", "atomic_F"]] = normalize_rows(out[[T, N, F]].to_numpy(), f"atomic {path}/{probe}")
    if out["statement_id"].duplicated().any():
        raise ValueError(f"{path}/{probe}: duplicate atomic statement IDs")
    return out[["statement_id", "atomic_T", "atomic_N", "atomic_F"]].set_index("statement_id")


def load_pair_lookup(
    path: Path,
    probe: str,
    limit: int | None,
) -> pd.DataFrame:
    """
    Load the compact identity table that maps pair_id -> (P_id, x_id).
    """
    if not path.exists():
        raise FileNotFoundError(path)

    pairs = pd.read_parquet(
        path,
        columns=["probe", "pair_id", "P_id", "x_id"],
        filters=[("probe", "==", probe)],
    )

    if pairs.empty:
        return pd.DataFrame(
            columns=["probe", "pair_id", "P_id", "x_id"]
        )

    pairs = pairs.sort_values(
        "pair_id",
        kind="stable",
    ).reset_index(drop=True)

    if pairs["pair_id"].duplicated().any():
        raise ValueError(
            f"{path}/{probe}: pair_id is not unique within probe."
        )

    if limit is not None:
        pairs = pairs.head(int(limit)).copy()

    pairs["pair_id"] = pd.to_numeric(
        pairs["pair_id"],
        errors="raise",
    ).astype("int64")
    pairs["P_id"] = pairs["P_id"].astype(str)
    pairs["x_id"] = pairs["x_id"].astype(str)

    return pairs


def _attach_pair_ids(
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    if "pair_id" not in scores.columns:
        raise KeyError(
            f"{context}: score table has no pair_id column; "
            f"columns={scores.columns.tolist()}"
        )

    scores = scores.copy()
    scores["pair_id"] = pd.to_numeric(
        scores["pair_id"],
        errors="raise",
    ).astype("int64")

    if scores["pair_id"].duplicated().any():
        raise ValueError(
            f"{context}: score table contains duplicate pair_id values."
        )

    requested = set(pairs["pair_id"].tolist())
    available = set(scores["pair_id"].tolist())

    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"{context}: score table is missing {len(missing)} pair IDs "
            f"requested by the pair table. Examples: {missing[:10]}"
        )

    # Keep exactly the selected pair-table population
    # while preserving deterministic pair_id order.
    merged = pairs[
        ["pair_id", "P_id", "x_id"]
    ].merge(
        scores,
        on="pair_id",
        how="left",
        validate="one_to_one",
    )

    return merged.sort_values(
        "pair_id",
        kind="stable",
    ).reset_index(drop=True)


def load_direct(
    path: Path,
    pair_path: Path,
    probe: str,
    limit: int | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    pairs = load_pair_lookup(
        pair_path,
        probe,
        limit,
    )
    if pairs.empty:
        return pd.DataFrame(
            columns=["P_id", "x_id", "C_T", "C_N", "C_F"]
        )

    frame = probe_rows(
        pd.read_parquet(path),
        probe,
        str(path),
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["P_id", "x_id", "C_T", "C_N", "C_F"]
        )

    frame = _attach_pair_ids(
        frame,
        pairs,
        context=f"Direct {path}/{probe}",
    )

    T, N, F = find_triplet(
        frame,
        f"Direct {path}/{probe}",
    )

    probs = normalize_rows(
        frame[[T, N, F]].to_numpy(),
        f"Direct {path}/{probe}",
    )

    out = frame[["P_id", "x_id"]].copy()
    out[["C_T", "C_N", "C_F"]] = probs

    if out.duplicated(["P_id", "x_id"]).any():
        raise ValueError(
            f"Direct {path}/{probe}: duplicate P_id/x_id pairs after "
            "attaching pair metadata."
        )

    return out.reset_index(drop=True)


def canonical_joint_rows(
    frame: pd.DataFrame,
    context: str,
) -> np.ndarray:
    """
    Convert current joint probabilities to canonical (P, x) row masses.

    Labels are ordered states of (x, P), with the first symbol x and
    the second symbol P:

        TT, TF, TN, FT, FF, FN, NT, NF, NN.

    The canonical coherence representation instead uses axes (P, x):

        PT,xT  PT,xN  PT,xF
        PN,xT  PN,xN  PN,xF
        PF,xT  PF,xN  PF,xF

    Therefore:
        PT,xT = prob_TT
        PT,xN = prob_NT
        PT,xF = prob_FT

        PN,xT = prob_TN
        PN,xN = prob_NN
        PN,xF = prob_FN

        PF,xT = prob_TF
        PF,xN = prob_NF
        PF,xF = prob_FF
    """
    missing = set(JOINT_SCORE_COLUMNS) - set(frame.columns)
    if missing:
        raise KeyError(
            f"{context}: missing joint columns {sorted(missing)}"
        )

    raw = normalize_rows(
        frame[JOINT_SCORE_COLUMNS].to_numpy(),
        context,
    )

    # Indices in JOINT_SCORE_COLUMNS:
    # TT=0, TF=1, TN=2, FT=3, FF=4, FN=5, NT=6, NF=7, NN=8
    canonical = raw[
        :,
        [
            0,  # PT,xT = TT
            6,  # PT,xN = NT
            3,  # PT,xF = FT
            2,  # PN,xT = TN
            8,  # PN,xN = NN
            5,  # PN,xF = FN
            1,  # PF,xT = TF
            7,  # PF,xN = NF
            4,  # PF,xF = FF
        ],
    ]

    return canonical


def load_joint(
    path: Path,
    pair_path: Path,
    probe: str,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)

    pairs = load_pair_lookup(
        pair_path,
        probe,
        limit,
    )
    if pairs.empty:
        return (
            pd.DataFrame(
                columns=["P_id", "x_id", "C_T", "C_N", "C_F"]
            ),
            pd.DataFrame(
                columns=["P_id", "x_id", *CANONICAL_JOINT_COLUMNS]
            ),
        )

    frame = probe_rows(
        pd.read_parquet(path),
        probe,
        str(path),
    )
    if frame.empty:
        return (
            pd.DataFrame(
                columns=["P_id", "x_id", "C_T", "C_N", "C_F"]
            ),
            pd.DataFrame(
                columns=["P_id", "x_id", *CANONICAL_JOINT_COLUMNS]
            ),
        )

    frame = _attach_pair_ids(
        frame,
        pairs,
        context=f"Joint {path}/{probe}",
    )

    canonical = canonical_joint_rows(
        frame,
        f"Joint {path}/{probe}",
    )

    joint = pd.DataFrame(
        canonical,
        columns=CANONICAL_JOINT_COLUMNS,
    )
    joint.insert(
        0,
        "x_id",
        frame["x_id"].astype(str).to_numpy(),
    )
    joint.insert(
        0,
        "P_id",
        frame["P_id"].astype(str).to_numpy(),
    )

    if joint.duplicated(["P_id", "x_id"]).any():
        raise ValueError(
            f"Joint {path}/{probe}: duplicate P_id/x_id pairs after "
            "attaching pair metadata."
        )

    J = joint[CANONICAL_JOINT_COLUMNS].to_numpy(dtype=float)

    # Cooper conditional x -> P in the full trivalent state space.
    #
    # x non-false + P true  -> T
    # x non-false + P false -> F
    # x false OR P neither  -> N
    C_T = J[:, 0] + J[:, 1]
    C_F = J[:, 6] + J[:, 7]
    C_N = J[:, 2] + J[:, 3] + J[:, 4] + J[:, 5] + J[:, 8]

    cond = joint[["P_id", "x_id"]].copy()
    cond[["C_T", "C_N", "C_F"]] = normalize_rows(
        np.column_stack([C_T, C_N, C_F]),
        f"Joint Cooper {path}/{probe}",
    )

    return (
        cond.reset_index(drop=True),
        joint.reset_index(drop=True),
    )

def attach_atomic(cond: pd.DataFrame, atomic: pd.DataFrame, context: str) -> pd.DataFrame:
    out = cond.copy()
    out["P_id"] = out["P_id"].astype(str)
    out["x_id"] = out["x_id"].astype(str)
    missing_P = sorted(set(out["P_id"]) - set(atomic.index))
    missing_x = sorted(set(out["x_id"]) - set(atomic.index))
    if missing_P or missing_x:
        raise ValueError(f"{context}: pair IDs missing from atomic table; P={missing_P[:10]} x={missing_x[:10]}")
    out[["P_T", "P_N", "P_F"]] = atomic.reindex(out["P_id"])[["atomic_T", "atomic_N", "atomic_F"]].to_numpy(dtype=float)
    out[["x_T", "x_N", "x_F"]] = atomic.reindex(out["x_id"])[["atomic_T", "atomic_N", "atomic_F"]].to_numpy(dtype=float)
    return out[["P_id", "x_id", *MEASUREMENT_COLUMNS]].copy()


def measurement_matrix(frame: pd.DataFrame) -> np.ndarray:
    Y = frame[MEASUREMENT_COLUMNS].to_numpy(
        dtype=float,
        copy=True,
    )
    Y[:, 0:3] = normalize_rows(Y[:, 0:3], "P block")
    Y[:, 3:6] = normalize_rows(Y[:, 3:6], "x block")
    Y[:, 6:9] = normalize_rows(Y[:, 6:9], "C block")
    return Y


def coherence_violations(Y: np.ndarray) -> np.ndarray:
    return np.asarray(Y, dtype=float) @ HALFSPACE_NORMALS.T


def exact_coherence_mask(Y: np.ndarray, tolerance: float) -> np.ndarray:
    return np.all(coherence_violations(Y) <= tolerance, axis=1)


def project_simplex_rows(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    idx = np.arange(1, V.shape[1] + 1, dtype=float)
    cond = U - cssv / idx[None, :] > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(len(V)), rho] / (rho.astype(float) + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def project_product_simplex(Y: np.ndarray) -> np.ndarray:
    Z = Y.copy()
    Z[:, 0:3] = project_simplex_rows(Z[:, 0:3])
    Z[:, 3:6] = project_simplex_rows(Z[:, 3:6])
    Z[:, 6:9] = project_simplex_rows(Z[:, 6:9])
    return Z


def project_halfspace(Y: np.ndarray, normal: np.ndarray, norm2: float) -> np.ndarray:
    v = Y @ normal
    bad = v > 0.0
    if not np.any(bad):
        return Y.copy()
    Z = Y.copy()
    Z[bad] -= (v[bad] / norm2)[:, None] * normal[None, :]
    return Z


def dykstra_project_batch(Y: np.ndarray, max_iterations: int, convergence_tolerance: float) -> tuple[np.ndarray, int]:
    X = np.asarray(Y, dtype=float).copy()
    corrections = [np.zeros_like(X) for _ in range(1 + len(HALFSPACE_NORMALS))]
    for iteration in range(1, max_iterations + 1):
        previous = X.copy()
        Z = X + corrections[0]
        X_new = project_product_simplex(Z)
        corrections[0] = Z - X_new
        X = X_new
        for i, (normal, norm2) in enumerate(zip(HALFSPACE_NORMALS, HALFSPACE_NORM_SQUARED), start=1):
            Z = X + corrections[i]
            X_new = project_halfspace(Z, normal, float(norm2))
            corrections[i] = Z - X_new
            X = X_new
        if float(np.max(np.abs(X - previous))) <= convergence_tolerance:
            return X, iteration
    return X, max_iterations


def projection_feasibility(Z: np.ndarray) -> tuple[float, float, float]:
    halfspace = float(np.max(np.maximum(Z @ HALFSPACE_NORMALS.T, 0.0)))
    simplex = float(max(
        np.max(np.abs(Z[:, 0:3].sum(axis=1) - 1.0)),
        np.max(np.abs(Z[:, 3:6].sum(axis=1) - 1.0)),
        np.max(np.abs(Z[:, 6:9].sum(axis=1) - 1.0)),
    ))
    nonnegative = float(max(0.0, -float(np.min(Z))))
    return halfspace, simplex, nonnegative


def compute_projection(Y: np.ndarray, exact: np.ndarray, batch_size: int, max_iterations: int, convergence_tolerance: float, feasibility_tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    projected_all = Y.copy()
    distance_squared = np.zeros(len(Y), dtype=float)
    incoherent = np.flatnonzero(~exact)
    iterations_used: list[int] = []
    max_halfspace = max_simplex = max_nonneg = 0.0
    for start in range(0, len(incoherent), batch_size):
        idx = incoherent[start:start + batch_size]
        batch = Y[idx]
        projected, iterations = dykstra_project_batch(batch, max_iterations, convergence_tolerance)
        iterations_used.append(iterations)
        halfspace, simplex, nonneg = projection_feasibility(projected)
        max_halfspace = max(max_halfspace, halfspace)
        max_simplex = max(max_simplex, simplex)
        max_nonneg = max(max_nonneg, nonneg)
        if max(halfspace, simplex, nonneg) > feasibility_tolerance:
            raise RuntimeError(
                "CCK projection failed feasibility check: "
                f"halfspace={halfspace:.3g} simplex={simplex:.3g} nonnegative={nonneg:.3g} iterations={iterations}"
            )
        projected_all[idx] = projected
        distance_squared[idx] = np.sum((batch - projected) ** 2, axis=1)
    rms = np.sqrt(distance_squared / 9.0)
    diag = {
        "n_projected_incoherent": int(len(incoherent)),
        "max_projection_iterations": int(max(iterations_used)) if iterations_used else 0,
        "fraction_batches_hitting_iteration_limit": float(np.mean(np.asarray(iterations_used) >= max_iterations)) if iterations_used else 0.0,
        "max_projected_halfspace_violation": max_halfspace,
        "max_projected_simplex_error": max_simplex,
        "max_projected_nonnegativity_violation": max_nonneg,
    }
    return projected_all, distance_squared, rms, diag


def construct_cck_witness(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[None, :]
    if not np.all(exact_coherence_mask(Z, 1e-7)):
        raise ValueError("Witness construction requires coherent measurements")
    pT, pN, pF = Z[:, 0], Z[:, 1], Z[:, 2]
    xT, xF = Z[:, 3], Z[:, 5]
    cT, cN, cF = Z[:, 6], Z[:, 7], Z[:, 8]
    M = np.zeros((len(Z), 3, 3), dtype=float)
    M[:, 0, 2] = pT - cT
    M[:, 2, 2] = pF - cF
    M[:, 1, 2] = xF + pN - cN
    remaining_rows = np.column_stack([cT, cN - xF, cF])
    remaining_xT = xT.copy()
    for row in range(3):
        take = np.minimum(remaining_rows[:, row], remaining_xT)
        M[:, row, 0] = take
        M[:, row, 1] = remaining_rows[:, row] - take
        remaining_xT -= take
    M[np.abs(M) < 1e-12] = 0.0
    if np.min(M) < -1e-7:
        raise RuntimeError("Constructed CCK witness has negative mass")
    return M


def measurement_from_witness(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim == 2:
        M = M[None, :, :]
    P = M.sum(axis=2)
    x = M.sum(axis=1)
    C_T = M[:, 0, 0] + M[:, 0, 1]
    C_F = M[:, 2, 0] + M[:, 2, 1]
    C_N = M[:, :, 2].sum(axis=1) + M[:, 1, 0] + M[:, 1, 1]
    return np.column_stack([P, x, C_T, C_N, C_F])


def self_check_cck_characterization() -> None:
    rng = np.random.default_rng(12345)
    raw = rng.gamma(1.0, 1.0, size=(256, 9))
    raw /= raw.sum(axis=1, keepdims=True)
    latent = raw.reshape(-1, 3, 3)
    Y = measurement_from_witness(latent)
    if not np.all(exact_coherence_mask(Y, 1e-10)):
        raise RuntimeError("Internal CCK self-check failed: coherent latent rows violate inequalities")
    Y2 = measurement_from_witness(construct_cck_witness(Y))
    if not np.allclose(Y, Y2, atol=1e-9, rtol=0.0):
        raise RuntimeError(f"Internal CCK self-check failed: witness roundtrip max error={np.max(np.abs(Y-Y2)):.3g}")


def joint_marginal_diagnostics(joint: pd.DataFrame, atomic: pd.DataFrame) -> pd.DataFrame:
    J = joint[CANONICAL_JOINT_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
    joint_P = J.sum(axis=2)
    joint_x = J.sum(axis=1)
    atomic_P = atomic.reindex(joint["P_id"].astype(str))[["atomic_T", "atomic_N", "atomic_F"]].to_numpy(dtype=float)
    atomic_x = atomic.reindex(joint["x_id"].astype(str))[["atomic_T", "atomic_N", "atomic_F"]].to_numpy(dtype=float)
    out = joint[["P_id", "x_id"]].copy()
    out["joint_P_marginal_rms"] = np.sqrt(np.mean((joint_P - atomic_P) ** 2, axis=1))
    out["joint_x_marginal_rms"] = np.sqrt(np.mean((joint_x - atomic_x) ** 2, axis=1))
    out["joint_atomic_marginal_rms"] = np.sqrt(np.mean(np.column_stack([joint_P-atomic_P, joint_x-atomic_x]) ** 2, axis=1))
    return out


def analyze_measurements(frame: pd.DataFrame, *, model: str, dataset: str, probe: str, estimator: str, args: argparse.Namespace, joint_diag: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    Y = measurement_matrix(frame)
    violations = coherence_violations(Y)
    exact = exact_coherence_mask(Y, args.exact_tolerance)
    projected, distance_squared, rms, projection_diag = compute_projection(
        Y, exact,
        args.projection_batch_size,
        args.projection_max_iterations,
        args.projection_convergence_tolerance,
        args.projection_feasibility_tolerance,
    )
    result = frame[["P_id", "x_id"]].copy()
    result[MEASUREMENT_COLUMNS] = Y
    result[PROJECTED_COLUMNS] = projected
    result[ADJUSTMENT_COLUMNS] = projected - Y
    result["is_exactly_coherent"] = exact
    result["distance_squared"] = distance_squared
    result["rms_adjustment"] = rms
    result["max_absolute_component_adjustment"] = np.max(np.abs(projected - Y), axis=1)
    pos = np.maximum(violations, 0.0)
    result["max_raw_constraint_violation"] = pos.max(axis=1)
    for i, name in enumerate(VIOLATION_NAMES):
        result[f"violation_{name}"] = pos[:, i]
    adjustments = projected - Y
    for block, sl in (("P", slice(0,3)), ("x", slice(3,6)), ("C", slice(6,9))):
        result[f"{block}_block_rms_adjustment"] = np.sqrt(np.mean(adjustments[:, sl] ** 2, axis=1))
    if joint_diag is not None:
        result = result.merge(joint_diag, on=["P_id", "x_id"], how="left", validate="one_to_one")
    result.insert(0, "model", model)
    result.insert(1, "dataset", dataset)
    result.insert(2, "probe", probe)
    result.insert(3, "estimator", estimator)
    incoh = rms[~exact]
    summary: dict[str, Any] = {
        "model": model, "dataset": dataset, "probe": probe, "estimator": estimator,
        "n_pairs": int(len(result)),
        "n_exactly_coherent": int(exact.sum()),
        "fraction_exactly_coherent": float(exact.mean()),
        "mean_distance_squared": float(distance_squared.mean()),
        "median_distance_squared": float(np.median(distance_squared)),
        "mean_rms_adjustment": float(rms.mean()),
        "median_rms_adjustment": float(np.median(rms)),
        "q25_rms_adjustment": float(np.quantile(rms, .25)),
        "q75_rms_adjustment": float(np.quantile(rms, .75)),
        "q90_rms_adjustment": float(np.quantile(rms, .90)),
        "q95_rms_adjustment": float(np.quantile(rms, .95)),
        "mean_rms_adjustment_incoherent_only": float(incoh.mean()) if len(incoh) else 0.0,
        "median_rms_adjustment_incoherent_only": float(np.median(incoh)) if len(incoh) else 0.0,
        "mean_max_absolute_component_adjustment": float(result["max_absolute_component_adjustment"].mean()),
        "mean_max_raw_constraint_violation": float(result["max_raw_constraint_violation"].mean()),
        **projection_diag,
    }
    for i, name in enumerate(VIOLATION_NAMES):
        violation_mask = violations[:, i] > args.exact_tolerance
        violating_values = violations[violation_mask, i]

        summary[f"n_violating_{name}"] = int(violation_mask.sum())
        summary[f"fraction_violating_{name}"] = float(np.mean(violation_mask))
        summary[f"mean_violation_{name}_given_violation"] = (
            float(violating_values.mean())
            if len(violating_values)
            else 0.0
        )
        summary[f"median_violation_{name}_given_violation"] = (
            float(np.median(violating_values))
            if len(violating_values)
            else 0.0
        )
    for block in ("P", "x", "C"):
        vals = result[f"{block}_block_rms_adjustment"]
        summary[f"mean_{block}_block_rms_adjustment"] = float(vals.mean())
        summary[f"median_{block}_block_rms_adjustment"] = float(vals.median())
    if estimator == "joint":
        for c in ("joint_P_marginal_rms", "joint_x_marginal_rms", "joint_atomic_marginal_rms"):
            if c in result.columns:
                summary[f"mean_{c}"] = float(result[c].mean())
                summary[f"median_{c}"] = float(result[c].median())
    return result, summary


def paired_summary(direct: pd.DataFrame, joint: pd.DataFrame, *, model: str, dataset: str, probe: str, tolerance: float) -> dict[str, Any]:
    dkeys = set(zip(direct["P_id"], direct["x_id"]))
    jkeys = set(zip(joint["P_id"], joint["x_id"]))
    if dkeys != jkeys:
        raise ValueError(f"{model}/{dataset}/{probe}: Direct and Joint P,x sets differ")
    merged = direct[["P_id", "x_id", "is_exactly_coherent", "rms_adjustment"]].rename(columns={"is_exactly_coherent":"direct_exact", "rms_adjustment":"direct_rms"}).merge(
        joint[["P_id", "x_id", "is_exactly_coherent", "rms_adjustment"]].rename(columns={"is_exactly_coherent":"joint_exact", "rms_adjustment":"joint_rms"}),
        on=["P_id", "x_id"], validate="one_to_one")
    delta = merged["direct_rms"] - merged["joint_rms"]
    return {
        "model": model, "dataset": dataset, "probe": probe,
        "n_common_pairs": int(len(merged)),
        "fraction_both_exact": float(np.mean(merged["direct_exact"] & merged["joint_exact"])),
        "fraction_direct_only_exact": float(np.mean(merged["direct_exact"] & ~merged["joint_exact"])),
        "fraction_joint_only_exact": float(np.mean(~merged["direct_exact"] & merged["joint_exact"])),
        "fraction_neither_exact": float(np.mean(~merged["direct_exact"] & ~merged["joint_exact"])),
        "mean_delta_rms_direct_minus_joint": float(delta.mean()),
        "median_delta_rms_direct_minus_joint": float(delta.median()),
        "q25_delta_rms_direct_minus_joint": float(delta.quantile(.25)),
        "q75_delta_rms_direct_minus_joint": float(delta.quantile(.75)),
        "fraction_direct_closer": float(np.mean(delta < -tolerance)),
        "fraction_joint_closer": float(np.mean(delta > tolerance)),
        "fraction_tied": float(np.mean(np.abs(delta) <= tolerance)),
    }


def sample_pairs(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n == 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed, replace=False).copy()


def compact_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (probe, dataset, estimator), g in model_summary.groupby(["probe", "dataset", "estimator"], sort=True):
        row = {
            "probe": probe, "dataset": dataset, "estimator": estimator,
            "n_models": int(g["model"].nunique()),
            "median_fraction_exactly_coherent": float(g["fraction_exactly_coherent"].median()),
            "q25_fraction_exactly_coherent": float(g["fraction_exactly_coherent"].quantile(.25)),
            "q75_fraction_exactly_coherent": float(g["fraction_exactly_coherent"].quantile(.75)),
            "median_mean_rms_adjustment": float(g["mean_rms_adjustment"].median()),
            "median_pair_median_rms_adjustment": float(g["median_rms_adjustment"].median()),
            "median_q90_rms_adjustment": float(g["q90_rms_adjustment"].median()),
            "median_P_block_rms_adjustment": float(g["mean_P_block_rms_adjustment"].median()),
            "median_x_block_rms_adjustment": float(g["mean_x_block_rms_adjustment"].median()),
            "median_C_block_rms_adjustment": float(g["mean_C_block_rms_adjustment"].median()),
        }

        for name in VIOLATION_NAMES:
            row[f"median_fraction_violating_{name}"] = float(
                g[f"fraction_violating_{name}"].median()
            )
            row[f"median_mean_violation_{name}_given_violation"] = float(
                g[f"mean_violation_{name}_given_violation"].median()
            )

        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    self_check_cck_characterization()
    root = args.repo_root.resolve()
    model_list_path = under_root(root, args.model_list)
    atomic_dir = under_root(root, args.atomic_dir)
    pairs_dir = under_root(root, args.pairs_dir)
    conditional_dir = under_root(root, args.conditional_dir)
    joint_dir = under_root(root, args.joint_dir)
    output_dir = under_root(root, args.output_dir)
    models = load_model_list(model_list_path)
    if args.model:
        requested = list(dict.fromkeys(args.model))
        unknown = sorted(set(requested) - set(models))
        if unknown:
            raise ValueError(f"Requested models not in {model_list_path}: {unknown}")
        models = requested
    datasets = list(dict.fromkeys(args.dataset))
    probes = list(dict.fromkeys(args.probe))

    paths = {
        "model_summary": output_dir / "model_summary.parquet",
        "paired": output_dir / "direct_joint_summary.parquet",
        "sample": output_dir / "pair_sample.parquet",
        "status": output_dir / "analysis_status.parquet",
        "compact": output_dir / "summary_compact.csv",
        "config": output_dir / "analysis_config.json",
    }
    guard_outputs(paths.values(), args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("TRIVALENT CCK COHERENCE")
    print(f"Models={len(models)} datasets={datasets} probes={probes} estimators=['direct','joint']")
    print(f"min_pairs={args.min_pairs} limit_pairs={args.limit_pairs} exact_tol={args.exact_tolerance:.1e}")
    print(f"projection_batch={args.projection_batch_size} max_iter={args.projection_max_iterations} sample_size={args.sample_size}")
    print("="*100)

    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []

    for model in models:
        for dataset in datasets:
            atomic_path = atomic_dir / model / f"{dataset}.parquet"
            pair_path = pairs_dir / model / f"{dataset}.parquet"
            conditional_path = conditional_dir / model / f"{dataset}.parquet"
            joint_path = joint_dir / model / f"{dataset}.parquet"
            for probe in probes:
                base = {
                    "model": model,
                    "dataset": dataset,
                    "probe": probe,
                    "atomic_path": str(atomic_path),
                    "pair_path": str(pair_path),
                    "conditional_path": str(conditional_path),
                    "joint_path": str(joint_path),
                }
                try:
                    atomic = load_atomic_lookup(atomic_path, probe)
                    if atomic.empty:
                        status_rows.append({**base, "status":"skipped", "reason":"probe_absent_from_atomic"})
                        print(f"SKIP  {model:24s} {dataset:18s} {probe:16s} | probe_absent_from_atomic")
                        continue
                    direct_cond = load_direct(
                        conditional_path,
                        pair_path,
                        probe,
                        args.limit_pairs,
                    )
                    joint_cond, joint_rows = load_joint(
                        joint_path,
                        pair_path,
                        probe,
                        args.limit_pairs,
                    )
                    if direct_cond.empty and joint_cond.empty:
                        status_rows.append({**base, "status":"skipped", "reason":"both_estimators_absent"})
                        print(f"SKIP  {model:24s} {dataset:18s} {probe:16s} | both_estimators_absent")
                        continue
                    dkeys = set(zip(direct_cond["P_id"], direct_cond["x_id"]))
                    jkeys = set(zip(joint_cond["P_id"], joint_cond["x_id"]))
                    if dkeys != jkeys:
                        raise ValueError(f"Direct/Joint P,x sets differ: direct={len(dkeys):,}, joint={len(jkeys):,}")
                    if len(direct_cond) < args.min_pairs:
                        status_rows.append({**base, "status":"skipped", "reason":"insufficient_pairs", "n_pairs":int(len(direct_cond))})
                        print(f"SKIP  {model:24s} {dataset:18s} {probe:16s} | insufficient_pairs n={len(direct_cond)}")
                        continue
                    direct_measure = attach_atomic(direct_cond, atomic, f"{model}/{dataset}/{probe}/direct")
                    joint_measure = attach_atomic(joint_cond, atomic, f"{model}/{dataset}/{probe}/joint")
                    jdiag = joint_marginal_diagnostics(joint_rows, atomic)
                    direct_result, direct_summary = analyze_measurements(direct_measure, model=model, dataset=dataset, probe=probe, estimator="direct", args=args)
                    joint_result, joint_summary = analyze_measurements(joint_measure, model=model, dataset=dataset, probe=probe, estimator="joint", args=args, joint_diag=jdiag)
                    paired_rows.append(paired_summary(direct_result, joint_result, model=model, dataset=dataset, probe=probe, tolerance=args.paired_tie_tolerance))
                    summaries.extend([direct_summary, joint_summary])
                    samples.extend([
                        sample_pairs(direct_result, args.sample_size, stable_seed(args.seed, model, dataset, probe, "direct")),
                        sample_pairs(joint_result, args.sample_size, stable_seed(args.seed, model, dataset, probe, "joint")),
                    ])
                    if args.save_pair_level:
                        pair_dir = output_dir / "pairs" / model / dataset / probe
                        pair_dir.mkdir(parents=True, exist_ok=True)
                        dpath, jpath = pair_dir / "direct.parquet", pair_dir / "joint.parquet"
                        if not args.overwrite and (dpath.exists() or jpath.exists()):
                            raise FileExistsError(f"Pair outputs exist under {pair_dir}")
                        write_parquet_atomic(direct_result, dpath)
                        write_parquet_atomic(joint_result, jpath)
                    status_rows.append({**base, "status":"complete", "reason":None, "n_pairs":int(len(direct_result))})
                    print(
                        f"OK    {model:24s} {dataset:18s} {probe:16s} n={len(direct_result):,} | "
                        f"Direct exact={direct_summary['fraction_exactly_coherent']:.3f} RMS={direct_summary['mean_rms_adjustment']:.4f} | "
                        f"Joint exact={joint_summary['fraction_exactly_coherent']:.3f} RMS={joint_summary['mean_rms_adjustment']:.4f}"
                    )
                except FileNotFoundError as exc:
                    status_rows.append({**base, "status":"skipped", "reason":"missing_input", "message":str(exc)})
                    print(f"SKIP  {model:24s} {dataset:18s} {probe:16s} | missing_input: {exc}")
                except (KeyError, ValueError, RuntimeError) as exc:
                    status_rows.append({**base, "status":"error", "reason":type(exc).__name__, "message":str(exc)})
                    print(f"ERROR {model:24s} {dataset:18s} {probe:16s} | {exc}")

    status = pd.DataFrame(status_rows)
    if not summaries:
        write_parquet_atomic(status, paths["status"])
        raise RuntimeError("No coherence analyses completed")
    model_summary = pd.DataFrame(summaries).sort_values(["probe","dataset","estimator","model"], kind="stable").reset_index(drop=True)
    paired = pd.DataFrame(paired_rows).sort_values(["probe","dataset","model"], kind="stable").reset_index(drop=True)
    sample = pd.concat(samples, ignore_index=True, sort=False) if samples else pd.DataFrame()
    if not sample.empty:
        sample = sample.sort_values(["probe","dataset","estimator","model","P_id","x_id"], kind="stable").reset_index(drop=True)
    status = status.sort_values(["probe","dataset","model"], kind="stable").reset_index(drop=True)
    compact = compact_summary(model_summary).sort_values(["probe","dataset","estimator"], kind="stable").reset_index(drop=True)

    write_parquet_atomic(model_summary, paths["model_summary"])
    write_parquet_atomic(paired, paths["paired"])
    write_parquet_atomic(sample, paths["sample"])
    write_parquet_atomic(status, paths["status"])
    write_csv_atomic(compact, paths["compact"])
    write_json_atomic({
        "schema_version":1,
        "analysis":"trivalent_cck_coherence",
        "research_question":"How closely do LLM-derived atomic and conditional credences approximate a jointly coherent CCK probability model?",
        "models":models,
        "datasets":datasets,
        "probes":probes,
        "estimators":["direct","joint"],
        "measurement_vector":MEASUREMENT_COLUMNS,
        "coherent_set":{
            "latent_model":"3x3 nonnegative row masses over P,x trivalent values summing to one",
            "necessary_and_sufficient_inequalities":["C_T <= P_T","C_F <= P_F","x_F <= C_N","C_N <= P_N + x_F"],
            "formal_self_check":True,
            "violation_frequency_definition":"fraction of pairs whose signed constraint excess is greater than exact_tolerance",
            "violation_magnitude_definition":"mean/median signed constraint excess among pairs violating that specific constraint",
        },
        "approximate_coherence":{
            "distance":"squared Euclidean distance to nearest coherent measurement vector",
            "motivation":"Brier-associated divergence / distance-to-nearest-coherent-system",
            "rms_adjustment":"sqrt(distance_squared/9)",
            "projection_algorithm":"Dykstra",
            "exact_tolerance":args.exact_tolerance,
            "projection_batch_size":args.projection_batch_size,
            "projection_max_iterations":args.projection_max_iterations,
            "projection_convergence_tolerance":args.projection_convergence_tolerance,
            "projection_feasibility_tolerance":args.projection_feasibility_tolerance,
        },
        "pairing":{"require_identical_direct_joint_P_x_sets":True,"minimum_pairs":args.min_pairs,"limit_pairs":args.limit_pairs},
        "joint_diagnostic":"RMS discrepancy between joint-implied and separately measured atomic marginals",
        "null_model":None,
        "null_model_rationale":"Coherence provides an absolute theoretically defined target; zero distance is exact coherence.",
        "sample_size_per_unit_estimator":args.sample_size,
        "save_pair_level":args.save_pair_level,
        "seed":args.seed,
        "model_list":str(model_list_path),
    }, paths["config"])

    complete = int((status["status"] == "complete").sum())
    skipped = int((status["status"] == "skipped").sum())
    errors = int((status["status"] == "error").sum())
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print(f"Complete units={complete} skipped={skipped} errors={errors} model-estimator rows={len(model_summary)} paired rows={len(paired)} sample rows={len(sample)}")
    print(compact.round(4).to_string(index=False))
    print(f"Saved to {output_dir}")
    print("="*100)
    if errors:
        raise RuntimeError(f"{errors} coherence unit(s) failed; inspect {paths['status']}")


if __name__ == "__main__":
    main()
