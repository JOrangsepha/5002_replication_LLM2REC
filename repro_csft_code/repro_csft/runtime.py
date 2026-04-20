from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None

from repro_csft.csft_dataset import CSFTDataset, CSFTDatasetConfig, DEFAULT_PROMPT_TEMPLATE


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2-0.5B")
    trust_remote_code: bool = field(default=True)
    torch_dtype: Optional[str] = field(default="bfloat16")


@dataclass
class DataArguments:
    train_file: str = field(default="data/AmazonMix-6/5-core/train/AmazonMix-6.csv")
    eval_file: Optional[str] = field(default="data/AmazonMix-6/5-core/valid/AmazonMix-6.csv")
    train_sample_size: int = field(default=-1)
    eval_sample_size: int = field(default=2000)
    max_length: int = field(default=1024)
    max_history: int = field(default=-1)
    history_column: str = field(default="history_item_title")
    target_column: str = field(default="item_title")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    train_on_prompt: bool = field(default=False)
    drop_target_equals_last_history: bool = field(default=False)


@dataclass
class RuntimeArguments:
    train_from_scratch: bool = field(default=False)
    set_nccl_compat: bool = field(default=False)
    tokenizer_padding_side: str = field(default="left")
    pre_eval: bool = field(default=True)
    early_stopping_patience: Optional[int] = field(default=5)
    wandb_project: str = field(default="")
    enable_single_process_model_parallel: bool = field(default=True)
    global_batch_size: int = field(default=0)
    micro_batch_size: int = field(default=0)


@dataclass
class LoraArguments:
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(default=None)


def resolve_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None
    dtype_name = dtype_name.lower()
    return {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]


def apply_runtime_env(runtime_args: RuntimeArguments):
    if runtime_args.set_nccl_compat:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"
    if runtime_args.wandb_project:
        os.environ["WANDB_PROJECT"] = runtime_args.wandb_project


def maybe_apply_lora(model, lora_args: LoraArguments):
    if not lora_args.use_lora:
        return model
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("LoRA requires `peft` package. Install it first.")

    if lora_args.lora_target_modules:
        target_modules = [x.strip() for x in lora_args.lora_target_modules.split(",") if x.strip()]
    else:
        target_modules = [name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
        target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError("No LoRA target modules found. Provide --lora_target_modules explicitly.")

    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


def build_tokenizer(model_args: ModelArguments, runtime_args: RuntimeArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = runtime_args.tokenizer_padding_side
    return tokenizer


def load_model(model_args: ModelArguments, runtime_args: RuntimeArguments):
    if runtime_args.train_from_scratch:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=resolve_dtype(model_args.torch_dtype),
        )
    if runtime_args.enable_single_process_model_parallel and torch.cuda.device_count() > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        model.is_parallelizable = True
        model.model_parallel = True
    return model


def build_dataset(tokenizer, data_args: DataArguments, split: str):
    base_kwargs = dict(
        max_length=data_args.max_length,
        max_history=data_args.max_history,
        history_column=data_args.history_column,
        target_column=data_args.target_column,
        train_on_prompt=data_args.train_on_prompt,
        prompt_template=data_args.prompt_template,
        drop_target_equals_last_history=data_args.drop_target_equals_last_history,
    )
    if split == "train":
        return CSFTDataset(tokenizer=tokenizer, config=CSFTDatasetConfig(csv_path=data_args.train_file, sample_size=data_args.train_sample_size, **base_kwargs))
    if split == "eval":
        if not data_args.eval_file:
            return None
        return CSFTDataset(tokenizer=tokenizer, config=CSFTDatasetConfig(csv_path=data_args.eval_file, sample_size=data_args.eval_sample_size, **base_kwargs))
    raise ValueError(f"Unknown split: {split}")


def maybe_align_gradient_accumulation(training_args: TrainingArguments, runtime_args: RuntimeArguments):
    if runtime_args.global_batch_size <= 0 or runtime_args.micro_batch_size <= 0:
        return
    if runtime_args.global_batch_size % runtime_args.micro_batch_size != 0:
        raise ValueError("global_batch_size must be divisible by micro_batch_size.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    training_args.gradient_accumulation_steps = max(1, (runtime_args.global_batch_size // runtime_args.micro_batch_size) // max(1, world_size))
