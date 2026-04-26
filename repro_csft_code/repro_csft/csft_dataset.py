from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_PROMPT_TEMPLATE = (
    "Below is a recommendation task.\n\n"
    "User interaction history:\n{history}\n\n"
    "Predict the next item title the user is most likely to interact with.\n"
    "Answer:"
)


@dataclass
class CSFTDatasetConfig:
    csv_path: str
    max_length: int = 1024
    max_history: int = 50
    history_column: str = "history_item_title"
    target_column: str = "item_title"
    sample_size: int = -1
    seed: int = 42
    train_on_prompt: bool = False
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    drop_target_equals_last_history: bool = False


class CSFTDataset(Dataset):
    """CSFT dataset for next-item title generation from interaction histories."""

    def __init__(self, tokenizer, config: CSFTDatasetConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.examples = self._load_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]

        prompt_ids = self.tokenizer(
            ex["prompt"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            ex["target"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids = target_ids + [eos_id]

        input_ids = prompt_ids + target_ids
        if self.config.train_on_prompt:
            labels = input_ids.copy()
        else:
            labels = ([-100] * len(prompt_ids)) + target_ids

        # Keep the right-most tokens when sequence exceeds max length.
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[-self.config.max_length :]
            labels = labels[-self.config.max_length :]

        # Safety fallback: ensure labels contain at least one supervised token.
        if all(x == -100 for x in labels):
            keep = min(len(target_ids), self.config.max_length)
            input_ids = target_ids[-keep:]
            labels = target_ids[-keep:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _load_examples(self) -> List[dict]:
        usecols = [self.config.history_column, self.config.target_column]
        df = pd.read_csv(self.config.csv_path, usecols=usecols)

        if self.config.sample_size > 0 and self.config.sample_size < len(df):
            df = df.sample(n=self.config.sample_size, random_state=self.config.seed)

        records = df.to_dict("records")
        examples: List[dict] = []

        for row in records:
            history_items = self._parse_history(row.get(self.config.history_column))
            if not history_items:
                continue

            if self.config.max_history > 0:
                history_items = history_items[-self.config.max_history :]

            target = str(row.get(self.config.target_column, "")).strip().replace("\n", " ")
            if not target:
                continue

            if (
                self.config.drop_target_equals_last_history
                and history_items
                and target == history_items[-1]
            ):
                continue

            history_text = ", ".join(history_items)
            prompt = self.config.prompt_template.format(history=history_text)
            examples.append({"prompt": prompt, "target": target})

        return examples

    @staticmethod
    def _parse_history(raw_value) -> List[str]:
        if raw_value is None:
            return []

        if isinstance(raw_value, list):
            values = raw_value
        elif isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if not raw_value:
                return []
            try:
                parsed = ast.literal_eval(raw_value)
                values = parsed if isinstance(parsed, list) else [raw_value]
            except (ValueError, SyntaxError):
                values = [raw_value]
        else:
            values = [str(raw_value)]

        cleaned: List[str] = []
        for v in values:
            text = str(v).strip().replace("\n", " ")
            if text:
                cleaned.append(text)
        return cleaned
