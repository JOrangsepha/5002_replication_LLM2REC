from __future__ import annotations

import json
import os
import random

from accelerate.logging import get_logger

from repro_iem.recdata.base import ContrastiveDataset, ContrastiveSample, TrainSample
from repro_iem.recdata.item_titles import _batched_shuffle


logger = get_logger(__name__, log_level="INFO")

PAIR_FILE_MAP = {
    "Arts": "Arts_Crafts_and_Sewing/training_item_pairs_gap24.jsonl",
    "Electronics": "Electronics/training_item_pairs_gap24.jsonl",
    "Home": "Home_and_Kitchen/training_item_pairs_gap24.jsonl",
    "Movies": "Movies_and_TV/training_item_pairs_gap24.jsonl",
    "Tools": "Tools_and_Home_Improvement/training_item_pairs_gap24.jsonl",
    "Games": "Video_Games/training_item_pairs_gap24.jsonl",
}
MAX_SAMPLES_PER_DATASET = 100000


class ItemPairDataset(ContrastiveDataset):
    def __init__(
        self,
        dataset_name: str = "ItemRec",
        split: str = "train",
        file_path: str = "dataset/llm2vec",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.file_path = file_path
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.data: list[ContrastiveSample] = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str | None = None):
        logger.info("Loading ItemRec data from %s", file_path)
        dataset_to_indices: dict[str, list[int]] = {}
        all_samples: list[ContrastiveSample] = []
        sample_id = 0

        for dataset_key, relative_path in PAIR_FILE_MAP.items():
            absolute_path = os.path.join(file_path, relative_path)
            with open(absolute_path, "r", encoding="utf-8") as handle:
                pairs = json.loads(handle.read().strip())

            if len(pairs) > MAX_SAMPLES_PER_DATASET:
                pairs = random.sample(pairs, MAX_SAMPLES_PER_DATASET)

            dataset_to_indices.setdefault(dataset_key, [])
            for anchor, positive in pairs:
                dataset_to_indices[dataset_key].append(sample_id)
                all_samples.append(
                    ContrastiveSample(
                        id_=sample_id,
                        query=self.separator + anchor,
                        positive=self.separator + positive,
                        task_name=dataset_key,
                    )
                )
                sample_id += 1

        self.data = [all_samples[idx] for idx in _batched_shuffle(dataset_to_indices, self.effective_batch_size, self.shuffle_individual_datasets)]
        logger.info("Loaded %d item pairs", len(self.data))

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split != "train":
            raise AssertionError("ItemPairDataset only supports the train split.")
        return TrainSample(texts=[sample.query, sample.positive], label=1.0)

