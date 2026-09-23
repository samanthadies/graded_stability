from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm

from stability.models.loading import ModelBundle
from stability.models.tokenization import tokenize_statements


@dataclass(frozen=True)
class LayerActivations:
    """Token-level hidden states and masks for one transformer layer."""

    values: np.ndarray
    attention_mask: np.ndarray
    layer: int

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.values.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.values.shape[2])

    @property
    def ram_gib(self) -> float:
        return float(
            (self.values.nbytes + self.attention_mask.nbytes)
            / (1024 ** 3)
        )


class _LayerHook:
    """Capture the tensor output of one transformer decoder block."""

    def __init__(self) -> None:
        self.output: torch.Tensor | None = None

    def __call__(
        self,
        module: Any,
        module_inputs: Any,
        module_outputs: Any,
    ) -> None:
        del module, module_inputs

        if isinstance(module_outputs, (tuple, list)):
            module_outputs = module_outputs[0]

        if not isinstance(module_outputs, torch.Tensor):
            raise TypeError(
                "Expected hooked layer output to be a tensor or a tuple/list "
                f"whose first element is a tensor; got {type(module_outputs)!r}."
            )

        self.output = module_outputs


def get_transformer_layers(model: Any):
    """
    Return the decoder-layer sequence for supported Hugging Face architectures.

    Candidate paths intentionally include ordinary causal-LM wrappers as well
    as multimodal text backbones such as Mistral 3.
    """
    candidate_paths = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("model", "model", "language_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )

    for path in candidate_paths:
        module = model

        try:
            for attribute in path:
                module = getattr(module, attribute)
        except AttributeError:
            continue

        try:
            n_layers = len(module)
        except TypeError:
            continue

        if n_layers > 0:
            return module

    available = [name for name, _ in model.named_children()]
    raise ValueError(
        "Could not locate transformer decoder layers. "
        f"Top-level model children: {available}"
    )


def _validate_statements(statements: Sequence[str]) -> list[str]:
    statements = [str(statement) for statement in statements]

    if not statements:
        raise ValueError("No statements were provided.")

    empty = [
        index
        for index, statement in enumerate(statements)
        if not statement.strip()
    ]
    if empty:
        raise ValueError(
            f"Empty statements found at rows {empty[:10]}."
        )

    return statements


def collect_layer_activations(
    bundle: ModelBundle,
    statements: Sequence[str],
    *,
    layer: int,
    batch_size: int = 16,
    max_length: int = 64,
    show_progress: bool = True,
) -> LayerActivations:
    """
    Collect one transformer's token-level activations in CPU float32 RAM.

    Parameters
    ----------
    bundle:
        Loaded model/tokenizer/device bundle.
    statements:
        Text inputs in dataset row order.
    layer:
        Zero-based decoder-layer index.
    batch_size:
        Number of statements per forward pass.
    max_length:
        Fixed padded/truncated sequence length.
    show_progress:
        Whether to display a tqdm progress bar.

    Returns
    -------
    LayerActivations
        values: float32 [N, L, D]
        attention_mask: bool [N, L]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    statements = _validate_statements(statements)
    layers = get_transformer_layers(bundle.model)

    if not 0 <= int(layer) < len(layers):
        raise ValueError(
            f"Layer {layer} is out of range; model has layers "
            f"0..{len(layers) - 1}."
        )

    n_rows = len(statements)
    values: np.ndarray | None = None
    masks = np.empty((n_rows, max_length), dtype=bool)

    hook = _LayerHook()
    handle = layers[int(layer)].register_forward_hook(hook)

    progress = tqdm(
        total=n_rows,
        unit="rows",
        desc=f"Layer {layer}: activations",
        leave=False,
        disable=not show_progress,
    )

    try:
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)

            encoded = tokenize_statements(
                statements[start:stop],
                bundle.tokenizer,
                instruct=bundle.instruct,
                max_length=max_length,
                chat_system_prompt=bundle.config.get("chat_system_prompt"),
            )

            input_ids = encoded["input_ids"].to(bundle.device)
            attention_mask = encoded["attention_mask"].to(bundle.device)

            hook.output = None

            with torch.inference_mode():
                bundle.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            if hook.output is None:
                raise RuntimeError(
                    f"Layer hook produced no output at layer {layer} "
                    f"for rows {start}:{stop}."
                )

            # Move directly to float32 on CPU. In particular, do not cast BF16
            # activations through FP16: large finite BF16 values can overflow.
            hidden_t = hook.output.detach().to(
                device="cpu",
                dtype=torch.float32,
            )

            if not torch.isfinite(hidden_t).all():
                raise RuntimeError(
                    f"Non-finite model activations at layer {layer}, "
                    f"rows {start}:{stop}."
                )

            hidden = hidden_t.numpy()
            batch_mask = (
                attention_mask.detach()
                .cpu()
                .numpy()
                .astype(bool, copy=False)
            )

            expected_leading = (stop - start, max_length)
            if hidden.shape[:2] != expected_leading:
                raise RuntimeError(
                    f"Unexpected hidden shape {hidden.shape} at layer {layer}; "
                    f"expected leading dimensions {expected_leading}."
                )
            if batch_mask.shape != expected_leading:
                raise RuntimeError(
                    f"Unexpected attention-mask shape {batch_mask.shape}; "
                    f"expected {expected_leading}."
                )

            if values is None:
                hidden_size = int(hidden.shape[-1])
                values = np.empty(
                    (n_rows, max_length, hidden_size),
                    dtype=np.float32,
                )
            elif hidden.shape[-1] != values.shape[-1]:
                raise RuntimeError(
                    "Hidden size changed between batches: "
                    f"{values.shape[-1]} -> {hidden.shape[-1]}."
                )

            values[start:stop] = hidden
            masks[start:stop] = batch_mask

            hook.output = None
            del hidden_t, hidden, batch_mask, input_ids, attention_mask
            progress.update(stop - start)

    finally:
        progress.close()
        handle.remove()

    if values is None:
        raise RuntimeError("No activations were collected.")

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Collected activation array for layer {layer} contains "
            "NaN or infinite values."
        )

    return LayerActivations(
        values=values,
        attention_mask=masks,
        layer=int(layer),
    )
