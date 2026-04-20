from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CONFIG_MAPPING,
    HfArgumentParser,
    MODEL_FOR_MASKED_LM_MAPPING,
    TrainingArguments,
)
from transformers.utils.versions import require_version

from llm2vec.models import GemmaBiForMNTP, LlamaBiForMNTP, MistralBiForMNTP, Qwen2BiForMNTP
from repro_iem.utils import resolve_torch_dtype


require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

MODEL_TYPES = tuple(config.model_type for config in MODEL_FOR_MASKED_LM_MAPPING.keys())


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    use_auth_token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default="sdpa")
    low_cpu_mem_usage: bool = field(default=False)

    def __post_init__(self):
        if self.config_overrides and (self.config_name or self.model_name_or_path):
            raise ValueError("--config_overrides cannot be combined with config_name/model_name_or_path")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=True)
    validation_split_percentage: int = field(default=5)
    max_seq_length: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    mlm_probability: float = field(default=0.15)
    line_by_line: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "Streaming mode requires datasets>=2.0.0")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either dataset_name or train/validation file input.")


@dataclass
class CustomArguments:
    lora_dropout: float = field(default=0.05)
    lora_r: Optional[int] = field(default=8)
    mask_token_type: str = field(default="blank")
    stop_after_n_steps: int = field(default=10000)
    data_collator_type: str = field(default="default")


@dataclass
class MntpDataState:
    raw_datasets: any
    tokenized_datasets: any
    train_dataset: any
    eval_dataset: any


def pick_model_class(config):
    name = config.__class__.__name__
    mapping = {
        "MistralConfig": MistralBiForMNTP,
        "LlamaConfig": LlamaBiForMNTP,
        "GemmaConfig": GemmaBiForMNTP,
        "Qwen2Config": Qwen2BiForMNTP,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported config class: {name}")
    return mapping[name]


def build_parser():
    return HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))


def load_arguments():
    parser = build_parser()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def build_config(model_args: ModelArguments):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        return AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    if model_args.model_name_or_path:
        return AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config = CONFIG_MAPPING[model_args.model_type]()
    if model_args.config_overrides:
        config.update_from_string(model_args.config_overrides)
    return config


def build_tokenizer(model_args: ModelArguments):
    kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        return AutoTokenizer.from_pretrained(model_args.tokenizer_name, **kwargs)
    if model_args.model_name_or_path:
        return AutoTokenizer.from_pretrained(model_args.model_name_or_path, **kwargs)
    raise ValueError("tokenizer_name or model_name_or_path is required to build a tokenizer.")


def configure_special_tokens(tokenizer, custom_args: CustomArguments):
    if tokenizer.mask_token is None:
        if custom_args.mask_token_type == "blank":
            tokenizer.mask_token = "_"
        elif custom_args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif custom_args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(f"Unsupported mask_token_type: {custom_args.mask_token_type}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_model(model_args: ModelArguments, config):
    model_class = pick_model_class(config)
    return model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(model_args.model_name_or_path and ".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=resolve_torch_dtype(model_args.torch_dtype),
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )


def load_text_datasets(data_args: DataArguments):
    raw = load_dataset("text", data_files=data_args.dataset_name)
    split = raw["train"].train_test_split(test_size=0.1)
    raw["train"] = split["train"].shuffle(seed=42)
    raw["validation"] = split["test"]
    return raw


def tokenize_datasets(raw_datasets, tokenizer, data_args: DataArguments, training_args: TrainingArguments):
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = min(tokenizer.model_max_length, 1024)
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            lines = [line for line in examples[text_column_name] if line and not line.isspace()]
            return tokenizer(
                lines,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            return raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache if not data_args.streaming else None,
                desc="Running tokenizer on dataset line_by_line",
            )

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache if not data_args.streaming else None,
            desc="Running tokenizer on every text in dataset",
        )

    def group_texts(examples):
        concatenated = {key: list(chain(*examples[key])) for key in examples}
        total_length = len(concatenated[next(iter(examples.keys()))])
        total_length = (total_length // max_seq_length) * max_seq_length
        return {
            key: [values[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for key, values in concatenated.items()
        }

    with training_args.main_process_first(desc="grouping texts together"):
        return tokenized.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache if not data_args.streaming else None,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
