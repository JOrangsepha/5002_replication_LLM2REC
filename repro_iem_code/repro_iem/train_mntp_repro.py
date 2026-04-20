from __future__ import annotations

import logging
import math
import os
import warnings

import datasets
import evaluate
import torch
import transformers
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, is_torch_tpu_available, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from repro_iem.mntp_runtime import (
    CustomArguments,
    DataArguments,
    build_config,
    build_model,
    build_tokenizer,
    configure_special_tokens,
    load_arguments,
    load_text_datasets,
    tokenize_datasets,
)
from repro_iem.mntp_runtime import ModelArguments
from repro_iem.utils import attach_lora_adapter


logger = logging.getLogger(__name__)


class FullMaskingCollator(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True) for row in labels.tolist()]
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

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def _save(self, output_dir=None, state_dict=None):
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", save_dir)
        self.model.save_peft_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.args, os.path.join(save_dir, "training_args.bin"))


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

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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
    model.model = attach_lora_adapter(model.model, rank=custom_args.lora_r, dropout=custom_args.lora_dropout)

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

    data_collator = collator_cls(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability, pad_to_multiple_of=8)
    trainer = MntpTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None),
        preprocess_logits_for_metrics=(preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None),
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
