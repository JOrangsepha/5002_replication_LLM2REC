from __future__ import annotations

from repro_iem.recdata.item_pairs import ItemPairDataset
from repro_iem.recdata.item_titles import ItemTitleDataset
from repro_iem.recdata.seqrec import SeqRecDataset


DATASET_REGISTRY = {
    "ItemRec": ItemPairDataset,
    "SeqRec": SeqRecDataset,
    "ItemTitles": ItemTitleDataset,
}


def normalize_dataset_name(name: str) -> tuple[str, bool]:
    enable_seq_aug = name.endswith("_SeqAug")
    dataset_name = name[: -len("_SeqAug")] if enable_seq_aug else name
    return dataset_name, enable_seq_aug


def load_dataset(name: str, split: str = "train", file_path: str | None = None, **kwargs):
    if split not in {"train", "validation", "test"}:
        raise NotImplementedError(f"Unsupported split: {split}")

    dataset_name, enable_seq_aug = normalize_dataset_name(name)
    if dataset_name not in DATASET_REGISTRY:
        raise NotImplementedError(f"Unsupported dataset: {name}")

    if enable_seq_aug:
        kwargs["data_augmentation"] = True

    dataset_cls = DATASET_REGISTRY[dataset_name]
    return dataset_cls(split=split, file_path=file_path, **kwargs)
