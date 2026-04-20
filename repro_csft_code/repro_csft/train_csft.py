from __future__ import annotations

import logging
import os
import sys

from transformers import EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments, set_seed

from repro_csft.runtime import (
    DataArguments,
    LoraArguments,
    ModelArguments,
    RuntimeArguments,
    apply_runtime_env,
    build_dataset,
    build_tokenizer,
    load_model,
    maybe_align_gradient_accumulation,
    maybe_apply_lora,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RuntimeArguments, LoraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def main():
    model_args, data_args, training_args, runtime_args, lora_args = parse_args()

    if "{history}" not in data_args.prompt_template:
        raise ValueError("prompt_template must include '{history}'.")
    if runtime_args.tokenizer_padding_side not in {"left", "right"}:
        raise ValueError("tokenizer_padding_side must be 'left' or 'right'.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    apply_runtime_env(runtime_args)
    set_seed(training_args.seed)
    maybe_align_gradient_accumulation(training_args, runtime_args)

    tokenizer = build_tokenizer(model_args, runtime_args)
    model = load_model(model_args, runtime_args)
    model = maybe_apply_lora(model, lora_args)
    model.config.use_cache = False

    train_dataset = build_dataset(tokenizer, data_args, split="train")
    eval_dataset = build_dataset(tokenizer, data_args, split="eval")
    if eval_dataset is None:
        training_args.evaluation_strategy = "no"
        if hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False

    callbacks = []
    if eval_dataset is not None and runtime_args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=runtime_args.early_stopping_patience))

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, callbacks=callbacks or None)

    if runtime_args.pre_eval and eval_dataset is not None:
        trainer.evaluate()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
