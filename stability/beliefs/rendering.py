"""
Loads statement-rendering conventions and formats atomic statement pairs as either
explicit conditionals or conjunctions for the Direct Conditional and Joint probes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml


StatementKind = Literal["conditional", "joint"]
VALID_KINDS = {"conditional", "joint"}


def strip_final_period(text: object) -> str:
    value = str(text).strip()
    return (
        value[:-1].rstrip()
        if value.endswith(".")
        else value
    )


def lowercase_initial_the(text: object) -> str:
    value = str(text).strip()
    if value.startswith("The "):
        return "the " + value[4:]
    return value


def load_statement_config(
    path: str | Path = "configs/statements.yaml",
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Statement construction config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected mapping in {path}."
        )

    return config


def get_default_variant(
    kind: StatementKind,
    *,
    config: dict[str, Any],
) -> str:
    if kind not in VALID_KINDS:
        raise ValueError(
            f"Unknown statement kind {kind!r}; expected {sorted(VALID_KINDS)}."
        )

    section = config.get(kind)
    if not isinstance(section, dict):
        raise ValueError(
            f"Statement config has no {kind!r} section."
        )

    variant = section.get("default")
    if not isinstance(variant, str) or not variant:
        raise ValueError(
            f"Statement config {kind!r} section has no valid default."
        )

    return variant


def render_pair_statement(
    *,
    kind: StatementKind,
    P_statement: str,
    x_statement: str,
    variant: str | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/statements.yaml",
) -> str:
    """Render one direct-conditional or joint statement."""
    if config is None:
        config = load_statement_config(
            config_path
        )

    if kind not in VALID_KINDS:
        raise ValueError(
            f"Unknown statement kind {kind!r}; expected {sorted(VALID_KINDS)}."
        )

    section = config.get(kind)
    if not isinstance(section, dict):
        raise ValueError(
            f"Statement config has no {kind!r} section."
        )

    if variant is None:
        variant = get_default_variant(
            kind,
            config=config,
        )

    templates = section.get("templates")
    if not isinstance(templates, dict):
        raise ValueError(
            f"Statement config {kind!r} section has no templates mapping."
        )

    spec = templates.get(variant)
    if not isinstance(spec, dict):
        raise ValueError(
            f"Unknown {kind} statement variant {variant!r}. "
            f"Available={sorted(templates)}."
        )

    template = spec.get("template")
    if not isinstance(template, str):
        raise ValueError(
            f"Statement variant {kind}/{variant} has no string template."
        )

    P_clean = str(P_statement).strip()
    x_clean = str(x_statement).strip()

    if bool(
        spec.get(
            "strip_final_period",
            True,
        )
    ):
        P_clean = strip_final_period(
            P_clean
        )
        x_clean = strip_final_period(
            x_clean
        )

    lowercase = spec.get(
        "lowercase_initial_the",
        [],
    )
    if lowercase is None:
        lowercase = []
    if not isinstance(
        lowercase,
        list,
    ):
        raise ValueError(
            f"{kind}/{variant}: lowercase_initial_the must be a list."
        )

    if "P" in lowercase:
        P_clean = lowercase_initial_the(
            P_clean
        )
    if "x" in lowercase:
        x_clean = lowercase_initial_the(
            x_clean
        )

    return template.format(
        P=P_clean,
        x=x_clean,
    )
