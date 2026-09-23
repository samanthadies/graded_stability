from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def render_statements(
    statements: Sequence[str],
    tokenizer: Any,
    *,
    instruct: bool,
    chat_system_prompt: str | None = None,
) -> list[str]:
    """
    Render raw statements into the text actually passed to the tokenizer.

    Non-instruct models are returned unchanged.

    Instruct models use the model-provided chat template and no assistant
    generation prompt, matching the legacy probing pipeline. By default
    (``chat_system_prompt is None``), the conversation contains only the user
    message, exactly as before.

    A model may explicitly configure ``chat_system_prompt``. In that case a
    system message is supplied before the user message. In particular, an
    explicit empty string is distinct from ``None``: it tells templates such
    as Mistral Small 3.1's not to substitute their long default system prompt.
    """
    texts = [str(statement) for statement in statements]

    if not instruct:
        return texts

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Model is marked instruct=True, but its tokenizer does not expose "
            "apply_chat_template()."
        )

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Model is marked instruct=True, but tokenizer.chat_template is not set."
        )

    if chat_system_prompt is None:
        # Preserve the existing behavior for every model that does not
        # explicitly opt into a system prompt through its model config.
        conversations = [
            [{"role": "user", "content": text}]
            for text in texts
        ]
    else:
        # An explicit empty string is intentional. Some chat templates inject
        # a long default system prompt only when the system message is absent.
        conversations = [
            [
                {"role": "system", "content": str(chat_system_prompt)},
                {"role": "user", "content": text},
            ]
            for text in texts
        ]

    rendered = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
    )

    if isinstance(rendered, str):
        # Some tokenizers may return a scalar for a single conversation.
        rendered = [rendered]

    rendered = list(rendered)

    if len(rendered) != len(texts):
        raise RuntimeError(
            "Chat-template rendering changed batch cardinality: "
            f"{len(texts)} inputs -> {len(rendered)} outputs."
        )

    return rendered


def tokenize_statements(
    statements: Sequence[str],
    tokenizer: Any,
    *,
    instruct: bool,
    max_length: int,
    chat_system_prompt: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Render and tokenize a batch of statements to a fixed sequence length.

    ``chat_system_prompt=None`` preserves the previous tokenization behavior.
    Models that explicitly configure a system prompt can pass it through here.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    rendered = render_statements(
        statements,
        tokenizer,
        instruct=instruct,
        chat_system_prompt=chat_system_prompt,
    )

    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    if "input_ids" not in encoded:
        raise ValueError("Tokenizer output does not contain input_ids.")
    if "attention_mask" not in encoded:
        raise ValueError("Tokenizer output does not contain attention_mask.")

    expected = (len(rendered), max_length)

    if tuple(encoded["input_ids"].shape) != expected:
        raise RuntimeError(
            f"Unexpected input_ids shape {tuple(encoded['input_ids'].shape)}; "
            f"expected {expected}."
        )
    if tuple(encoded["attention_mask"].shape) != expected:
        raise RuntimeError(
            f"Unexpected attention_mask shape "
            f"{tuple(encoded['attention_mask'].shape)}; expected {expected}."
        )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
