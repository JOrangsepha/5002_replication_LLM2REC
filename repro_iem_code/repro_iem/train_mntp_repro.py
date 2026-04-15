from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Optional, Tuple

import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CONFIG_MAPPING,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    MODEL_FOR_MASKED_LM_MAPPING,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from llm2vec.models import GemmaBiForMNTP, LlamaBiForMNTP, MistralBiForMNTP, Qwen2BiForMNTP
from repro_iem.utils import attach_lora_adapter, resolve_torch_dtype


require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
MODEL_TYPES = tuple(config.model_type for config in MODEL_FOR_MASKED_LM_MAPPING.keys())


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


class FullMaskingCollator(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
                for row in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked = torch.bernoulli(probability_matrix).bool()
        labels[~masked] = -100
        inputs[masked] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels


class StopAfterStepsCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class MntpTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        return dataset

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", save_dir)
        self.model.save_peft_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.args, os.path.join(save_dir, "training_args.bin"))


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
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **kwargs)
    else:
        raise ValueError("tokenizer_name or model_name_or_path is required to build a tokenizer.")

    if tokenizer.mask_token is None:
        if model_args.model_name_or_path and "Qwen" in model_args.model_name_or_path:
            pass
    return tokenizer


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


def main():
    model_args, data_args, training_args, custom_args = load_arguments()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if model_args.use_auth_token is not None:
        warnings.warn("use_auth_token is deprecated; use token instead.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("Specify only one of token and use_auth_token.")
        model_args.token = model_args.use_auth_token

    send_example_telemetry("run_mlm", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty.")

    set_seed(training_args.seed)

    raw_datasets = load_text_datasets(data_args)
    config = build_config(model_args)
    tokenizer = build_tokenizer(model_args)
    configure_special_tokens(tokenizer, custom_args)

    model = build_model(model_args, config)
    model.model = attach_lora_adapter(
        model.model,
        rank=custom_args.lora_r,
        dropout=custom_args.lora_dropout,
    )

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, data_args, training_args)

    train_dataset = tokenized_datasets["train"] if training_args.do_train else None
    eval_dataset = tokenized_datasets["validation"] if training_args.do_eval else None

    if data_args.max_train_samples is not None and train_dataset is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None and eval_dataset is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir) if training_args.do_eval else None

    def preprocess_logits_for_metrics(logits, labels):
        return logits[0].argmax(dim=-1) if isinstance(logits, tuple) else logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = preds[:, :-1].reshape(-1)
        labels = labels[:, 1:].reshape(-1)
        mask = labels != -100
        return metric.compute(predictions=preds[mask], references=labels[mask])

    collator_cls = FullMaskingCollator if custom_args.data_collator_type == "all_mask" else DataCollatorForLanguageModeling
    if custom_args.data_collator_type not in {"all_mask", "default"}:
        raise ValueError(f"Unsupported data_collator_type: {custom_args.data_collator_type}")

    pad_to_multiple_of = 8 if data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length else None
    data_collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    trainer = MntpTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None
        ),
    )

    trainer.add_callback(StopAfterStepsCallback(custom_args.stop_after_n_steps))

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

