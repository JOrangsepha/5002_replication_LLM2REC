import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .recdata import NormalRecData


@dataclass
class SequenceSplit:
    train: Any
    valid: Any
    test: Any
    select_pool: list[int]
    item_num: int


class SequenceRuntime:
    def __init__(self, config: dict):
        self.config = config
        self.splits: Optional[SequenceSplit] = None

    def load_splits(self) -> SequenceSplit:
        train, valid, test, select_pool, item_num = NormalRecData(self.config).load_data()
        self.splits = SequenceSplit(train, valid, test, select_pool, item_num)
        self.config['select_pool'] = select_pool
        self.config['item_num'] = item_num
        self.config['eos_token'] = item_num + 1
        return self.splits

    def load_pretrained_embeddings(self):
        if not self.config.get('embedding'):
            return None

        pretrained_item_embeddings = torch.tensor(
            np.load(self.config['embedding']),
            dtype=torch.float32,
        ).to(self.config['device'])

        if self.config.get('seq_embedding'):
            base_seq_embedding_path = self.config['seq_embedding']
            seq_embeddings = []
            for split_name in ('train', 'val', 'test'):
                emb_path = base_seq_embedding_path.format(split_name)
                seq_embeddings.append(
                    torch.tensor(np.load(emb_path), dtype=torch.float32).to(self.config['device'])
                )
            return [pretrained_item_embeddings, *seq_embeddings]

        return pretrained_item_embeddings

    def make_dataloader(self, split_name: str, batch_size: Optional[int] = None, shuffle: bool = False):
        if self.splits is None:
            self.load_splits()

        dataset = getattr(self.splits, split_name)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config['eval_batch_size'],
            shuffle=shuffle,
        )
