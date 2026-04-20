from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    def __init__(self, tokenizer, config: CSFTDatasetConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.examples = self._load_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]
        prompt_ids = self._tokenize(ex["prompt"])
        target_ids = self._tokenize(ex["target"])
        target_ids = self._append_eos(target_ids)

        input_ids = prompt_ids + target_ids
        labels = input_ids.copy() if self.config.train_on_prompt else ([-100] * len(prompt_ids) + target_ids)
        input_ids, labels = self._truncate_to_max_length(input_ids, labels)
        input_ids, labels = self._ensure_supervised_tokens(input_ids, labels, target_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

    def _append_eos(self, token_ids: List[int]) -> List[int]:
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not token_ids or token_ids[-1] != eos_id):
            return token_ids + [eos_id]
        return token_ids

    def _truncate_to_max_length(self, input_ids: List[int], labels: List[int]) -> tuple[List[int], List[int]]:
        if len(input_ids) <= self.config.max_length:
            return input_ids, labels
        return input_ids[-self.config.max_length :], labels[-self.config.max_length :]

    def _ensure_supervised_tokens(self, input_ids: List[int], labels: List[int], target_ids: List[int]) -> tuple[List[int], List[int]]:
        if all(x == -100 for x in labels):
            keep = min(len(target_ids), self.config.max_length)
            return target_ids[-keep:], target_ids[-keep:]
        return input_ids, labels

    def _load_examples(self) -> List[dict]:
        usecols = [self.config.history_column, self.config.target_column]
        df = pd.read_csv(self.config.csv_path, usecols=usecols)
        if self.config.sample_size > 0 and self.config.sample_size < len(df):
            df = df.sample(n=self.config.sample_size, random_state=self.config.seed)

        examples: List[dict] = []
        for row in df.to_dict("records"):
            history_items = self._parse_history(row.get(self.config.history_column))
            if not history_items:
                continue
            if self.config.max_history > 0:
                history_items = history_items[-self.config.max_history :]

            target = str(row.get(self.config.target_column, "")).strip().replace("\n", " ")
            if not target:
                continue
            if self.config.drop_target_equals_last_history and history_items and target == history_items[-1]:
                continue

            examples.append({"prompt": self.config.prompt_template.format(history=", ".join(history_items)), "target": target})
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
