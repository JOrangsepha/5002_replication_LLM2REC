from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

try:
    from repro_csft.csft_dataset import (
        CSFTDataset,
        CSFTDatasetConfig,
        DEFAULT_PROMPT_TEMPLATE,
    )
except ImportError:
    # Allow direct execution: python repro_csft/train_csft.py
    from csft_dataset import CSFTDataset, CSFTDatasetConfig, DEFAULT_PROMPT_TEMPLATE


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2-0.5B",
        metadata={"help": "Base model for CSFT training."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code from model repo."},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "One of: auto, bfloat16, float16, float32."},
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="data/AmazonMix-6/5-core/train/AmazonMix-6.csv",
        metadata={"help": "Path to CSFT training CSV."},
    )
    eval_file: Optional[str] = field(
        default="data/AmazonMix-6/5-core/valid/AmazonMix-6.csv",
        metadata={"help": "Path to validation CSV. Set null to disable eval."},
    )
    train_sample_size: int = field(
        default=-1,
        metadata={"help": "If > 0, randomly sample this many train rows."},
    )
    eval_sample_size: int = field(
        default=2000,
        metadata={"help": "If > 0, randomly sample this many eval rows."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Max token length for each training sample."},
    )
    max_history: int = field(
        default=-1,
        metadata={"help": "Max number of history items kept in prompt; <=0 means no truncation."},
    )
    history_column: str = field(
        default="history_item_title",
        metadata={"help": "CSV column containing history title list."},
    )
    target_column: str = field(
        default="item_title",
        metadata={"help": "CSV column containing next-item target title."},
    )
    prompt_template: str = field(
        default=DEFAULT_PROMPT_TEMPLATE,
        metadata={"help": "Prompt template. Must contain {history}."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "If true, compute loss on prompt+target; else only target."},
    )
    drop_target_equals_last_history: bool = field(
        default=False,
        metadata={"help": "Whether to drop rows where target equals last history item."},
    )


@dataclass
class RuntimeArguments:
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "If true, initialize model from config instead of pretrained weights."},
    )
    set_nccl_compat: bool = field(
        default=False,
        metadata={"help": "Set NCCL compatibility envs (disable NVLink/IB/GDR)."},
    )
    tokenizer_padding_side: str = field(
        default="left",
        metadata={"help": "Tokenizer padding side: left or right."},
    )
    pre_eval: bool = field(
        default=True,
        metadata={"help": "Run one evaluation before training when eval dataset is available."},
    )
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={"help": "Early stopping patience; set null to disable."},
    )
    wandb_project: str = field(
        default="",
        metadata={"help": "If set, export WANDB_PROJECT to this value."},
    )
    enable_single_process_model_parallel: bool = field(
        default=True,
        metadata={"help": "Enable model_parallel in single-process multi-GPU setup."},
    )
    global_batch_size: int = field(
        default=0,
        metadata={"help": "If >0 with micro_batch_size >0, auto-compute gradient_accumulation_steps like original script."},
    )
    micro_batch_size: int = field(
        default=0,
        metadata={"help": "Micro batch size for auto gradient accumulation calculation."},
    )


@dataclass
class LoraArguments:
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated modules, e.g. q_proj,k_proj,v_proj,o_proj."},
    )


def resolve_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None
    dtype_name = dtype_name.lower()
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def infer_lora_target_modules(model) -> list[str]:
    common_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = name.split(".")[-1]
            if last in common_targets:
                found.add(last)
    return sorted(found)


def maybe_apply_lora(model, lora_args: LoraArguments):
    if not lora_args.use_lora:
        return model

    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "LoRA requires `peft` package. Install it first, e.g. `pip install peft`."
        )

    if lora_args.lora_target_modules:
        target_modules = [
            x.strip() for x in lora_args.lora_target_modules.split(",") if x.strip()
        ]
    else:
        target_modules = infer_lora_target_modules(model)

    if not target_modules:
        raise ValueError(
            "No LoRA target modules found. Please provide --lora_target_modules explicitly."
        )

    logger.info("Applying LoRA to modules: %s", target_modules)
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def apply_runtime_env(runtime_args: RuntimeArguments):
    if runtime_args.set_nccl_compat:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"

    if runtime_args.wandb_project:
        os.environ["WANDB_PROJECT"] = runtime_args.wandb_project


def build_dataset(tokenizer, data_args: DataArguments, split: str) -> Optional[CSFTDataset]:
    if split == "train":
        cfg = CSFTDatasetConfig(
            csv_path=data_args.train_file,
            max_length=data_args.max_length,
            max_history=data_args.max_history,
            history_column=data_args.history_column,
            target_column=data_args.target_column,
            sample_size=data_args.train_sample_size,
            train_on_prompt=data_args.train_on_prompt,
            prompt_template=data_args.prompt_template,
            drop_target_equals_last_history=data_args.drop_target_equals_last_history,
        )
        return CSFTDataset(tokenizer=tokenizer, config=cfg)

    if split == "eval":
        if not data_args.eval_file:
            return None
        cfg = CSFTDatasetConfig(
            csv_path=data_args.eval_file,
            max_length=data_args.max_length,
            max_history=data_args.max_history,
            history_column=data_args.history_column,
            target_column=data_args.target_column,
            sample_size=data_args.eval_sample_size,
            train_on_prompt=data_args.train_on_prompt,
            prompt_template=data_args.prompt_template,
            drop_target_equals_last_history=data_args.drop_target_equals_last_history,
        )
        return CSFTDataset(tokenizer=tokenizer, config=cfg)

    raise ValueError(f"Unknown split: {split}")


def load_model(model_args: ModelArguments, runtime_args: RuntimeArguments):
    if runtime_args.train_from_scratch:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        logger.info("Initializing model from scratch with config: %s", model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=resolve_dtype(model_args.torch_dtype),
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size != 1
    if (
        runtime_args.enable_single_process_model_parallel
        and (not use_ddp)
        and torch.cuda.device_count() > 1
    ):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


def disable_eval_for_training_args(training_args: TrainingArguments):
    if hasattr(training_args, "evaluation_strategy"):
        training_args.evaluation_strategy = "no"
    if hasattr(training_args, "eval_strategy"):
        training_args.eval_strategy = "no"
    training_args.load_best_model_at_end = False


def maybe_align_gradient_accumulation(
    training_args: TrainingArguments, runtime_args: RuntimeArguments
):
    if runtime_args.global_batch_size <= 0 or runtime_args.micro_batch_size <= 0:
        return

    if runtime_args.global_batch_size % runtime_args.micro_batch_size != 0:
        raise ValueError(
            "global_batch_size must be divisible by micro_batch_size."
        )

    grad_steps = runtime_args.global_batch_size // runtime_args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        grad_steps = max(1, grad_steps // world_size)

    training_args.gradient_accumulation_steps = grad_steps
    logger.info(
        "Auto-set gradient_accumulation_steps=%d from global_batch_size=%d, micro_batch_size=%d, world_size=%d",
        grad_steps,
        runtime_args.global_batch_size,
        runtime_args.micro_batch_size,
        world_size,
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, RuntimeArguments, LoraArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            model_args,
            data_args,
            training_args,
            runtime_args,
            lora_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
            runtime_args,
            lora_args,
        ) = parser.parse_args_into_dataclasses()

    if "{history}" not in data_args.prompt_template:
        raise ValueError("prompt_template must include '{history}'.")

    if runtime_args.tokenizer_padding_side not in {"left", "right"}:
        raise ValueError("tokenizer_padding_side must be 'left' or 'right'.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    apply_runtime_env(runtime_args)
    set_seed(training_args.seed)
    maybe_align_gradient_accumulation(training_args, runtime_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = runtime_args.tokenizer_padding_side

    model = load_model(model_args, runtime_args)
    model = maybe_apply_lora(model, lora_args)
    model.config.use_cache = False

    train_dataset = build_dataset(tokenizer, data_args, split="train")
    eval_dataset = build_dataset(tokenizer, data_args, split="eval")

    logger.info("Train dataset size: %d", len(train_dataset))
    logger.info("Eval dataset size: %d", len(eval_dataset) if eval_dataset is not None else 0)

    if eval_dataset is None:
        disable_eval_for_training_args(training_args)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    callbacks = None
    if eval_dataset is not None and runtime_args.early_stopping_patience is not None:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=runtime_args.early_stopping_patience
            )
        ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if runtime_args.pre_eval and eval_dataset is not None:
        trainer.evaluate()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    logger.info("Training finished. Last log entry: %s", metrics)


if __name__ == "__main__":
    main()
