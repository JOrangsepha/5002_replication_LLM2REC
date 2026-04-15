from __future__ import annotations

import ast
import json
import os
import random

import pandas as pd
from accelerate.logging import get_logger

from repro_iem.recdata.base import ContrastiveDataset, ContrastiveSample, TrainSample
from repro_iem.recdata.item_titles import _batched_shuffle


logger = get_logger(__name__, log_level="INFO")

TRAIN_CSV_MAP = {
    "Arts": "Arts_Crafts_and_Sewing/5-core/train/Arts_Crafts_and_Sewing_5_2014-9-2023-10.csv",
    "Electronics": "Electronics/5-core/train/Electronics_5_2016-9-2023-10.csv",
    "Home": "Home_and_Kitchen/5-core/train/Home_and_Kitchen_5_2016-9-2023-10.csv",
    "Movies": "Movies_and_TV/5-core/train/Movies_and_TV_5_2019-9-2023-10.csv",
    "Tools": "Tools_and_Home_Improvement/5-core/train/Tools_and_Home_Improvement_5_2016-9-2023-10.csv",
    "Games": "Video_Games/5-core/train/Video_Games_5_1996-9-2023-10.csv",
}

ITEM_TITLE_JSON_MAP = {
    "Arts": "Arts_Crafts_and_Sewing/5-core/downstream/item_titles.json",
    "Electronics": "Electronics/5-core/downstream/item_titles.json",
    "Home": "Home_and_Kitchen/5-core/downstream/item_titles.json",
    "Movies": "Movies_and_TV/5-core/downstream/item_titles.json",
    "Tools": "Tools_and_Home_Improvement/5-core/downstream/item_titles.json",
    "Games": "Video_Games/5-core/downstream/item_titles.json",
}

MAX_SAMPLES_PER_DATASET = 100000


class SeqRecDataset(ContrastiveDataset):
    def __init__(
        self,
        dataset_name: str = "SeqRec",
        split: str = "train",
        file_path: str = "./data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        data_augmentation: bool = False,
        augmentation_rate: float = 0.2,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.file_path = file_path
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.data_augmentation = data_augmentation
        self.augmentation_rate = augmentation_rate
        self.negative_item_pool: list[str] = []
        self.data: list[ContrastiveSample] = []
        self.load_data(file_path)
        self.negative_item_pool = [item for item in self.negative_item_pool if item is not None]

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str | None = None):
        logger.info("Loading SeqRec data from %s", file_path)
        dataset_to_indices: dict[str, list[int]] = {}
        all_samples: list[ContrastiveSample] = []
        sample_id = 0

        for dataset_key in TRAIN_CSV_MAP:
            dataset_to_indices.setdefault(dataset_key, [])
            rows = self._make_samples(file_path, dataset_key)
            for row in rows:
                dataset_to_indices[dataset_key].append(sample_id)
                sample = ContrastiveSample(
                    id_=sample_id,
                    query=self.separator + row["query"],
                    positive=self.separator + row["positive"],
                    negative=self.separator + row["negative"],
                    task_name=dataset_key,
                    aug_query=(self.separator + row["aug_query"]) if self.data_augmentation else None,
                )
                all_samples.append(sample)
                self.negative_item_pool.append(sample.negative)
                sample_id += 1

        self.data = [all_samples[idx] for idx in _batched_shuffle(dataset_to_indices, self.effective_batch_size, self.shuffle_individual_datasets)]
        logger.info("Loaded %d sequential contrastive samples", len(self.data))

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split != "train":
            raise AssertionError("SeqRecDataset only supports the train split.")

        texts = [sample.query, sample.positive, sample.negative]
        if self.data_augmentation:
            texts.append(sample.aug_query)
        return TrainSample(texts=texts, label=1.0)

    def _make_samples(self, file_path: str, dataset_key: str) -> list[dict]:
        with open(os.path.join(file_path, ITEM_TITLE_JSON_MAP[dataset_key]), "r", encoding="utf-8") as handle:
            raw_titles = json.load(handle)

        title_by_zero_based_id = {int(key) - 1: value for key, value in raw_titles.items()}
        candidate_ids = list(title_by_zero_based_id.keys())

        table = pd.read_csv(os.path.join(file_path, TRAIN_CSV_MAP[dataset_key]))
        table = table.sample(n=MAX_SAMPLES_PER_DATASET, random_state=42)

        samples = []
        for _, row in table.iterrows():
            history_ids = ast.literal_eval(row["history_item_id"])
            positive_id = row["item_id"]

            negative_id = random.choice(candidate_ids)
            while negative_id == positive_id or negative_id in history_ids:
                negative_id = random.choice(candidate_ids)

            history_titles = ast.literal_eval(row["history_item_title"])
            positive_title = row["item_title"]
            negative_title = title_by_zero_based_id[negative_id]

            augmented_history_text = None
            if self.data_augmentation:
                if len(history_ids) <= 2:
                    augmented_ids = history_ids
                else:
                    num_drop = int(len(history_ids) * self.augmentation_rate)
                    num_drop = max(1, num_drop)
                    kept_ids = set(random.sample(history_ids, len(history_ids) - num_drop))
                    augmented_ids = [item_id for item_id in history_ids if item_id in kept_ids]
                augmented_history_text = ", ".join(title_by_zero_based_id[item_id] for item_id in augmented_ids)

            samples.append(
                {
                    "query": ", ".join(history_titles),
                    "positive": positive_title,
                    "negative": negative_title,
                    "aug_query": augmented_history_text,
                }
            )
        return samples

