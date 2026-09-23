from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from stability.models.loading import ModelBundle


LABEL_ORDER = ("true", "false")


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(str(text), add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise RuntimeError("Unexpected batched tokenization for one string.")
        ids = ids[0]
    ids = [int(token_id) for token_id in ids]
    if not ids:
        raise ValueError(f"Text tokenized to zero tokens: {text!r}")
    return ids


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def score_binary_prompts(
    *,
    prompts: Sequence[str],
    bundle: ModelBundle,
    answer_options: Mapping[str, str],
    batch_size: int = 8,
    max_length: int = 1024,
) -> pd.DataFrame:
    """Return normalized sequence probabilities for True and False."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be at least 2.")
    if set(answer_options) != set(LABEL_ORDER):
        raise ValueError("answer_options must contain exactly true and false.")
    if not prompts:
        return pd.DataFrame(
            columns=[
                "pred_label", "prob_true", "prob_false",
                "log_score_true", "log_score_false",
            ]
        )

    tokenizer = bundle.tokenizer
    candidate_ids = {
        label: _encode(tokenizer, answer_options[label]) for label in LABEL_ORDER
    }
    if candidate_ids["true"] == candidate_ids["false"]:
        raise ValueError("True and false answer strings tokenize identically.")

    flat_sequences: list[list[int]] = []
    metadata: list[tuple[int, int]] = []

    for prompt in prompts:
        prompt_ids = _encode(tokenizer, str(prompt))
        for label_index, label in enumerate(LABEL_ORDER):
            answer_ids = candidate_ids[label]
            max_prompt = max_length - len(answer_ids)
            if max_prompt < 1:
                raise ValueError(f"Answer {label!r} is too long for max_length.")
            kept_prompt = prompt_ids[-max_prompt:]
            flat_sequences.append(kept_prompt + answer_ids)
            metadata.append((label_index, len(kept_prompt)))

    sequence_scores = np.empty(len(flat_sequences), dtype=np.float64)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id.")

    for start in range(0, len(flat_sequences), batch_size):
        end = min(start + batch_size, len(flat_sequences))
        sequences = flat_sequences[start:end]
        batch_meta = metadata[start:end]
        width = max(len(sequence) for sequence in sequences)

        input_ids = torch.full(
            (len(sequences), width), int(pad_id), dtype=torch.long, device=bundle.device
        )
        attention_mask = torch.zeros(
            (len(sequences), width), dtype=torch.long, device=bundle.device
        )
        for row_index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row_index, :length] = torch.tensor(
                sequence, dtype=torch.long, device=bundle.device
            )
            attention_mask[row_index, :length] = 1

        with torch.inference_mode():
            outputs = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

        if logits.ndim != 3:
            raise RuntimeError(f"Expected [B,L,V] logits; got {tuple(logits.shape)}.")
        log_probs = F.log_softmax(logits.float(), dim=-1)

        for local_index, (label_index, prompt_length) in enumerate(batch_meta):
            label = LABEL_ORDER[label_index]
            total = 0.0
            for answer_offset, token_id in enumerate(candidate_ids[label]):
                prediction_position = prompt_length + answer_offset - 1
                if prediction_position < 0:
                    raise RuntimeError("Cannot score without at least one prompt token.")
                total += float(
                    log_probs[local_index, prediction_position, int(token_id)].item()
                )
            sequence_scores[start + local_index] = total

        del outputs, logits, log_probs, input_ids, attention_mask

    matrix = sequence_scores.reshape(len(prompts), len(LABEL_ORDER))
    probabilities = _softmax(matrix)
    predictions = np.asarray(LABEL_ORDER, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]

    result = pd.DataFrame(
        {
            "pred_label": predictions,
            "prob_true": probabilities[:, 0],
            "prob_false": probabilities[:, 1],
            "log_score_true": matrix[:, 0],
            "log_score_false": matrix[:, 1],
        }
    )
    if not np.allclose(
        result[["prob_true", "prob_false"]].sum(axis=1),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("Behavioral probability rows do not sum to one.")
    return result
