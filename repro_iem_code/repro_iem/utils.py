from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_torch_dtype(dtype_name: str | None):
    if dtype_name in {None, "auto"}:
        return dtype_name
    return getattr(torch, dtype_name)


def default_lora_targets(config_name: str) -> list[str]:
    if config_name in {"LlamaConfig", "MistralConfig", "GemmaConfig", "Qwen2Config"}:
        return list(DEFAULT_LORA_TARGETS)
    raise ValueError(f"LoRA target modules are unknown for config {config_name}.")


def attach_lora_adapter(
    model,
    rank: int | None,
    dropout: float,
    alpha: int | None = None,
    target_modules: list[str] | None = None,
):
    if rank is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    if target_modules is None:
        target_modules = default_lora_targets(model.config.__class__.__name__)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else 2 * rank,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=None,
    )
    adapted = get_peft_model(model, config)
    adapted.print_trainable_parameters()
    return adapted


def copy_tokenizer_assets(source_model_dir: str | os.PathLike[str], destination_dir: str | os.PathLike[str]) -> list[str]:
    source = Path(source_model_dir)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    copied = []
    for path in source.iterdir():
        if "token" not in path.name:
            continue
        target = destination / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
        copied.append(path.name)
    return copied

