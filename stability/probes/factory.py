from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from stability.probes.base import Probe
from stability.probes.mean_difference import MeanDifferenceProbe
from stability.probes.sawmil import SAWMILProbe
from stability.probes.svm import SVMProbe


SUPPORTED_PROBES = ("sawmil", "svm", "mean_difference")


def load_probe_config(
    probe_name: str,
    *,
    config_dir: str | Path = "configs/probe",
) -> dict[str, Any]:
    """Load one probe YAML."""
    probe_name = probe_name.strip().lower()
    if probe_name not in SUPPORTED_PROBES:
        raise ValueError(
            f"Unknown probe {probe_name!r}; expected one of {SUPPORTED_PROBES}."
        )

    path = Path(config_dir) / f"{probe_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Probe config does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}.")

    return config


def build_probe(
    probe_name: str,
    *,
    config: dict[str, Any] | None = None,
    config_dir: str | Path = "configs/probe",
    seed: int = 0,
) -> Probe:
    """Instantiate one configured probe."""
    probe_name = probe_name.strip().lower()

    if config is None:
        config = load_probe_config(
            probe_name,
            config_dir=config_dir,
        )

    training = config.get("training", {}) or {}
    calibration = config.get("calibration", {}) or {}

    common_calibration = {
        "calibration_C": float(calibration.get("C", 1.0)),
        "calibration_max_iter": int(calibration.get("max_iter", 5000)),
        "calibration_tol": float(calibration.get("tol", 1e-6)),
    }

    if probe_name == "svm":
        return SVMProbe(
            C=float(training.get("C", 1.0)),
            standardize=bool(training.get("standardize", True)),
            max_iter=int(training.get("max_iter", 10000)),
            tol=float(training.get("tol", 1e-4)),
            class_weight=training.get("class_weight"),
            seed=int(seed),
            **common_calibration,
        )

    if probe_name == "mean_difference":
        return MeanDifferenceProbe(
            normalize_direction=bool(
                training.get("normalize_direction", True)
            ),
            standardize=bool(training.get("standardize", False)),
            **common_calibration,
        )

    if probe_name == "sawmil":
        mil = config.get("mil", {}) or {}

        return SAWMILProbe(
            C=float(training.get("C", 1.0)),
            epochs=int(training.get("epochs", 5)),
            standardize=bool(training.get("standardize", True)),
            init_mode=str(mil.get("init_mode", "last_k")),
            k_init=int(mil.get("k_init", 2)),
            k_top=int(mil.get("k_top", 2)),
            tail_k=int(mil.get("tail_k", 2)),
            max_tokens_per_bag=(
                None
                if mil.get("max_tokens_per_bag", 256) is None
                else int(mil.get("max_tokens_per_bag", 256))
            ),
            seed=int(seed),
            **common_calibration,
        )

    raise ValueError(
        f"Unknown probe {probe_name!r}; expected one of {SUPPORTED_PROBES}."
    )
