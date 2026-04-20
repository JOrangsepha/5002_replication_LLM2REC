from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from repro_iem.dataset_registry import load_dataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    peft_model_name_or_path: Optional[str] = field(default=None)
    bidirectional: bool = field(default=False)
    max_seq_length: Optional[int] = field(default=None)
    torch_dtype: Optional[str] = field(default=None)
    attn_implementation: str = field(default="sdpa")
    pooling_mode: str = field(default="mean")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_file_path: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)


@dataclass
class CustomArguments:
    simcse_dropout: float = field(default=0.1)
    lora_dropout: float = field(default=0.05)
    lora_r: Optional[int] = field(default=8)
    stop_after_n_steps: int = field(default=10000)
    experiment_id: Optional[str] = field(default=None)
    loss_class: str = field(default="HardNegativeNLLLoss")
    loss_scale: float = field(default=50.0)


@dataclass
class CheckpointArguments:
    source_model_dir: str = field(default="")
    destination_dir: str = field(default="")


def load_train_dataset(data_args: DataArguments):
    return load_dataset(data_args.dataset_name, split="train", file_path=data_args.dataset_file_path)
