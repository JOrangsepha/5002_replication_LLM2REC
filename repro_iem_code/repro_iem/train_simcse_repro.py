from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from llm2vec import LLM2Vec
from llm2vec.loss.utils import load_loss
from peft import get_peft_model
from torch import nn
from tqdm import tqdm
from transformers import HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed

from repro_iem.dataset_registry import load_dataset
from repro_iem.utils import attach_lora_adapter, resolve_torch_dtype


transformers.logging.set_verbosity_error()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")


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
class ContrastiveCollator:
    model: LLM2Vec

    def __call__(self, features: List[Any]) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        num_text_groups = len(features[0].texts)
        grouped_texts = [[] for _ in range(num_text_groups)]
        labels = []

        for sample in features:
            for index, text in enumerate(sample.texts):
                grouped_texts[index].append(text)
            labels.append(sample.label)

        tokenized_groups = [self.model.tokenize(texts) for texts in grouped_texts]
        return tokenized_groups, torch.tensor(labels)


class StopAfterStepsCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class SimCSETrainer(Trainer):
    def __init__(self, *args, loss_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        query_reps = self.model(features[0])
        positive_reps = self.model(features[1])
        negative_reps = self.model(features[2]) if len(features) > 2 else None
        loss = self.loss_function(query_reps, positive_reps, negative_reps)

        if not return_outputs:
            return loss

        output = torch.cat([model(group)["sentence_embedding"][:, None] for group in features], dim=1)
        return loss, output

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", save_dir)
        self.model.save(save_dir)
        torch.save(self.args, os.path.join(save_dir, "training_args.bin"))


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def main():
    model_args, data_args, training_args, custom_args = parse_args()

    kwargs = []
    if training_args.ddp_find_unused_parameters:
        kwargs.append(
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        )
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
    )
    if data_args.max_train_samples is not None:
        limit = min(len(dataset), data_args.max_train_samples)
        train_examples = [dataset[index] for index in range(limit)]
    else:
        train_examples = [
            dataset[index]
            for index in tqdm(
                range(len(dataset)),
                desc="Loading train examples...",
                disable=not accelerator.is_main_process,
            )
        ]

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=resolve_torch_dtype(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
    )

    model.model = attach_lora_adapter(
        model.model,
        rank=custom_args.lora_r,
        dropout=custom_args.lora_dropout,
    )

    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    trainer = SimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=ContrastiveCollator(model),
        tokenizer=model.tokenizer,
        loss_function=train_loss,
    )
    trainer.add_callback(StopAfterStepsCallback(custom_args.stop_after_n_steps))
    trainer.train()


if __name__ == "__main__":
    main()

