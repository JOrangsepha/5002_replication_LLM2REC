from __future__ import annotations

import os
import random

from accelerate.logging import get_logger

from repro_iem.recdata.base import ContrastiveDataset, ContrastiveSample, TrainSample


logger = get_logger(__name__, log_level="INFO")

TITLE_FILE_MAP = {
    "Mix6": "AmazonMix-6/5-core/info/item_titles.txt",
}


class ItemTitleDataset(ContrastiveDataset):
    def __init__(
        self,
        dataset_name: str = "ItemTitles",
        split: str = "train",
        file_path: str = "data",
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
        logger.info("Loading ItemTitles data from %s", file_path)
        dataset_to_indices: dict[str, list[int]] = {}
        all_samples: list[ContrastiveSample] = []
        sample_id = 0

        for dataset_key, relative_path in TITLE_FILE_MAP.items():
            dataset_to_indices.setdefault(dataset_key, [])
            absolute_path = os.path.join(file_path, relative_path)
            with open(absolute_path, "r", encoding="utf-8") as handle:
                titles = [line.strip() for line in handle]

            for title in titles:
                prefixed = self.separator + title
                dataset_to_indices[dataset_key].append(sample_id)
                all_samples.append(
                    ContrastiveSample(
                        id_=sample_id,
                        query=prefixed,
                        positive=prefixed,
                        task_name=dataset_key,
                    )
                )
                sample_id += 1

        self.data = [all_samples[idx] for idx in _batched_shuffle(dataset_to_indices, self.effective_batch_size, self.shuffle_individual_datasets)]
        logger.info("Loaded %d title pairs", len(self.data))

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split != "train":
            raise AssertionError("ItemTitleDataset only supports the train split.")
        return TrainSample(texts=[sample.query, sample.positive], label=1.0)


def _batched_shuffle(dataset_to_indices: dict[str, list[int]], effective_batch_size: int, shuffle_each_dataset: bool) -> list[int]:
    if shuffle_each_dataset:
        for indices in dataset_to_indices.values():
            random.shuffle(indices)

    batches: list[list[int]] = []
    for dataset_name, indices in dataset_to_indices.items():
        for start in range(0, len(indices), effective_batch_size):
            batch = indices[start : start + effective_batch_size]
            if len(batch) == effective_batch_size:
                batches.append(batch)
            else:
                logger.info("Skip 1 batch for dataset %s.", dataset_name)

    random.shuffle(batches)
    return [idx for batch in batches for idx in batch]

