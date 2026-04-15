from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ContrastiveSample:
    id_: int
    query: str
    positive: str
    negative: str | None = None
    task_name: str | None = None
    aug_query: str | None = None


class TrainSample:
    def __init__(self, guid: str = "", texts: List[str] | None = None, label: int | float = 0):
        self.guid = guid
        self.texts = texts or []
        self.label = label

    def __str__(self) -> str:
        return f"<TrainSample> label: {self.label}, texts: {'; '.join(self.texts)}"


class ContrastiveDataset(torch.utils.data.Dataset):
    def load_data(self, file_path: str | None = None):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

