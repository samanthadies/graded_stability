from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ChallengeConfig:
    schema_version: int
    name: str
    initial_prompt: str
    challenge_templates: dict[str, str]
    answer_options: dict[str, str]
    seed: int

    @property
    def challenge_ids(self) -> tuple[str, ...]:
        return tuple(self.challenge_templates.keys())

    @property
    def sequences(self) -> tuple[tuple[str, ...], ...]:
        return tuple(permutations(self.challenge_ids))

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "initial_prompt": self.initial_prompt,
            "challenge_templates": self.challenge_templates,
            "answer_options": self.answer_options,
            "seed": self.seed,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _nonempty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def load_challenge_config(
    path: str | Path = "configs/experiments/challenge.yaml",
) -> ChallengeConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")

    templates_raw = raw.get("challenge_templates")
    if not isinstance(templates_raw, dict):
        raise ValueError("challenge_templates must be a mapping.")
    templates = {
        _nonempty(key, field="challenge template ID"): _nonempty(
            value, field=f"challenge_templates.{key}"
        )
        for key, value in templates_raw.items()
    }
    if len(templates) != 3:
        raise ValueError(
            "This experiment expects exactly three challenge templates for 3! "
            f"counterbalancing; found {len(templates)}."
        )

    answers_raw = raw.get("answer_options")
    if not isinstance(answers_raw, dict) or set(answers_raw) != {"true", "false"}:
        raise ValueError("answer_options must contain exactly true and false.")
    answers = {
        label: _nonempty(answers_raw[label], field=f"answer_options.{label}")
        for label in ("true", "false")
    }
    if answers["true"] == answers["false"]:
        raise ValueError("True and false answer texts must differ.")

    counter = raw.get("counterbalancing", {})
    if not isinstance(counter, dict):
        raise ValueError("counterbalancing must be a mapping.")
    if str(counter.get("strategy", "all_permutations")) != "all_permutations":
        raise ValueError("Only all_permutations counterbalancing is supported.")

    return ChallengeConfig(
        schema_version=int(raw.get("schema_version", 1)),
        name=_nonempty(raw.get("name"), field="name"),
        initial_prompt=_nonempty(raw.get("initial_prompt"), field="initial_prompt"),
        challenge_templates=templates,
        answer_options=answers,
        seed=int(counter.get("seed", 0)),
    )
