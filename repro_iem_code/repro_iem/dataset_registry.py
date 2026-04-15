from __future__ import annotations

from repro_iem.recdata.item_pairs import ItemPairDataset
from repro_iem.recdata.item_titles import ItemTitleDataset
from repro_iem.recdata.seqrec import SeqRecDataset


DATASET_REGISTRY = {
    "ItemRec": ItemPairDataset,
    "SeqRec": SeqRecDataset,
    "ItemTitles": ItemTitleDataset,
}


def load_dataset(name: str, split: str = "train", file_path: str | None = None, **kwargs):
    if split not in {"train", "validation", "test"}:
        raise NotImplementedError(f"Unsupported split: {split}")

    dataset_name = name
    enable_seq_aug = False
    if dataset_name.endswith("_SeqAug"):
        dataset_name = dataset_name[: -len("_SeqAug")]
        enable_seq_aug = True

    if dataset_name not in DATASET_REGISTRY:
        raise NotImplementedError(f"Unsupported dataset: {name}")

    dataset_cls = DATASET_REGISTRY[dataset_name]
    if enable_seq_aug:
        kwargs["data_augmentation"] = True

    return dataset_cls(split=split, file_path=file_path, **kwargs)

