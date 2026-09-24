"""
Renders the prompts used at each round of the behavioral challenge experiment
from the configured templates and target statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from stability.behavior.config import ChallengeConfig


def render_behavior_prompt(
    *,
    statement: str,
    prior_answers: Sequence[str],
    challenge_sequence: Sequence[str],
    tokenizer: Any,
    instruct: bool,
    config: ChallengeConfig,
) -> str:
    if len(prior_answers) > len(challenge_sequence):
        raise ValueError("More prior answers than challenge turns.")

    for answer in prior_answers:
        if answer not in config.answer_options:
            raise ValueError(f"Unknown prior answer label {answer!r}.")
    for challenge_id in challenge_sequence:
        if challenge_id not in config.challenge_templates:
            raise ValueError(f"Unknown challenge template ID {challenge_id!r}.")

    initial_user = config.initial_prompt.format(statement=str(statement))

    if instruct:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Instruct tokenizer has no apply_chat_template().")
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("Instruct tokenizer.chat_template is not set.")

        messages: list[dict[str, str]] = [{"role": "user", "content": initial_user}]
        for round_index, answer_label in enumerate(prior_answers):
            messages.append(
                {"role": "assistant", "content": config.answer_options[answer_label]}
            )
            challenge_id = challenge_sequence[round_index]
            messages.append(
                {"role": "user", "content": config.challenge_templates[challenge_id]}
            )

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise RuntimeError("Expected chat-template rendering to return one string.")
        return rendered

    transcript = initial_user
    for round_index, answer_label in enumerate(prior_answers):
        transcript += config.answer_options[answer_label]
        challenge_id = challenge_sequence[round_index]
        transcript += "\n\n" + config.challenge_templates[challenge_id]
    return transcript
