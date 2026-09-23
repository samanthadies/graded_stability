from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


@dataclass
class ModelBundle:
    """Loaded model plus the metadata needed by activation extraction."""

    model: Any
    tokenizer: Any
    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return str(self.config["name"])

    @property
    def hf_name(self) -> str:
        return str(self.config["model"])

    @property
    def instruct(self) -> bool:
        return bool(self.config.get("instruct", False))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}; found {type(data).__name__}."
        )
    return data


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge dictionaries, with override taking precedence."""
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def load_model_config(
    model_name: str,
    *,
    config_dir: str | Path = "configs/model",
) -> dict[str, Any]:
    """
    Load one model YAML and resolve simple sibling defaults.

    Supported model defaults look like:

        defaults:
          - base_llama
          - _self_

    Each string default other than ``_self_`` is resolved relative to
    ``config_dir`` as ``<default>.yaml``.
    """
    config_dir = Path(config_dir)
    path = config_dir / f"{model_name}.yaml"
    raw = _load_yaml(path)

    defaults = raw.get("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ValueError(f"{path}: defaults must be a list.")

    merged: dict[str, Any] = {}

    for entry in defaults:
        if entry == "_self_":
            continue

        if not isinstance(entry, str):
            raise ValueError(
                f"{path}: unsupported model defaults entry {entry!r}. "
                "Only sibling YAML names and _self_ are supported."
            )

        base_path = config_dir / f"{entry}.yaml"
        base_cfg = _load_yaml(base_path)
        base_cfg.pop("defaults", None)
        merged = _deep_merge(merged, base_cfg)

    raw_self = copy.deepcopy(raw)
    raw_self.pop("defaults", None)
    merged = _deep_merge(merged, raw_self)

    required = ("name", "model")
    missing = [
        key
        for key in required
        if key not in merged or merged[key] in (None, "???")
    ]
    if missing:
        raise ValueError(
            f"{path}: unresolved required model fields {missing}."
        )

    return merged


def _resolve_compute_dtype(device: torch.device) -> torch.dtype:
    """
    Prefer BF16 on CUDA hardware that supports it.

    BF16 is important for models such as Gemma 2 because it preserves a much
    wider numerical range than FP16.
    """
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    # CPU inference is primarily for smoke tests. Float32 is the safest default.
    return torch.float32


def _load_tokenizer(
    *,
    model_name: str,
    model_type: str,
    instruct: bool,
):
    """
    Load a tokenizer and apply architecture-specific compatibility fixes.
    """
    if model_type == "mistral3":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True,
        )

        # Mistral Small 3.1 stores the instruct chat template separately in
        # chat_template.json. Loading it directly avoids requiring the
        # PixtralProcessor / torchvision for text-only probing.
        if instruct:
            template_path = hf_hub_download(
                repo_id=model_name,
                filename="chat_template.json",
            )
            with open(template_path, "r", encoding="utf-8") as handle:
                template_data = json.load(handle)

            chat_template = template_data.get("chat_template")
            if not chat_template:
                raise RuntimeError(
                    f"No chat template found for instruct model {model_name}."
                )
            tokenizer.chat_template = chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name} has neither a pad token "
                "nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    # The legacy layer-sweep pipeline used right padding. Keeping that default
    # makes equivalence testing straightforward; downstream pooling remains
    # padding-side agnostic.
    tokenizer.padding_side = "right"

    return tokenizer


def load_model(
    model_config: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> ModelBundle:
    """
    Load a supported Hugging Face model and tokenizer.

    Architecture-specific model classes and tokenizer workarounds are hidden
    here so the remainder of the codebase can operate on ModelBundle.
    """
    device = torch.device(device)
    dtype = _resolve_compute_dtype(device)

    model_name = str(model_config["model"])
    instruct = bool(model_config.get("instruct", False))

    hf_config = AutoConfig.from_pretrained(model_name)

    if hf_config.model_type == "mistral3":
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="eager",
        )

    tokenizer = _load_tokenizer(
        model_name=model_name,
        model_type=str(hf_config.model_type),
        instruct=instruct,
    )

    model.eval()
    model = model.to(device=device, dtype=dtype)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=dict(model_config),
        device=device,
        dtype=dtype,
    )


def load_model_by_name(
    model_name: str,
    *,
    config_dir: str | Path = "configs/model",
    device: str | torch.device = "cuda",
) -> ModelBundle:
    """Convenience wrapper around load_model_config() + load_model()."""
    config = load_model_config(
        model_name,
        config_dir=config_dir,
    )
    return load_model(
        config,
        device=device,
    )
