"""
data/dataset.py
---------------
Handles loading and preprocessing of Amazon/Goodreads datasets.

Paper reference (Section 4.1.1):
- 5-core filtering: keep users/items with >= 5 interactions
- Max sequence length: 10
- Leave-one-out split: last item = test, second-to-last = val, rest = train
- Items represented by their titles (text)

Expected data format (from LLM2Rec repo ./data/):
Each dataset is a .pkl or .json file with fields:
  - user_id, item_id, item_title, timestamp
"""

import os
import json
import pickle
from collections import defaultdict
from torch.utils.data import Dataset


# ── 1. Load raw data ─────────────────────────────────────────────────────────

def load_dataset(data_path: str) -> list[dict]:
    """
    Load a dataset file. Supports .json, .jsonl, and .pkl formats.
    Returns a list of interaction dicts with keys:
      user_id, item_id, item_title, timestamp
    """
    ext = os.path.splitext(data_path)[-1]
    if ext == ".json":
        with open(data_path) as f:
            return json.load(f)
    elif ext == ".jsonl":
        with open(data_path) as f:
            return [json.loads(line) for line in f]
    elif ext == ".pkl":
        with open(data_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ── 2. Build user interaction sequences ──────────────────────────────────────

def build_sequences(interactions: list[dict]) -> dict[str, list]:
    """
    Group interactions by user, sort by timestamp, and return:
      { user_id: [(item_id, item_title), ...] }  sorted oldest → newest
    """
    user_items = defaultdict(list)
    for row in interactions:
        user_items[row["user_id"]].append(
            (row["timestamp"], row["item_id"], row["item_title"])
        )
    # Sort each user's history by timestamp
    sequences = {}
    for user, records in user_items.items():
        records.sort(key=lambda x: x[0])
        sequences[user] = [(item_id, title) for _, item_id, title in records]
    return sequences


# ── 3. 5-core filtering (Section 4.1.1) ──────────────────────────────────────

def five_core_filter(sequences: dict, min_count: int = 5) -> dict:
    """
    Iteratively remove users and items with fewer than min_count interactions
    until convergence. Paper uses 5-core.
    """
    while True:
        # Count item frequencies
        item_counts = defaultdict(int)
        for seq in sequences.values():
            for item_id, _ in seq:
                item_counts[item_id] += 1

        # Filter items
        valid_items = {iid for iid, cnt in item_counts.items() if cnt >= min_count}

        # Rebuild sequences keeping only valid items
        new_sequences = {}
        for user, seq in sequences.items():
            filtered = [(iid, title) for iid, title in seq if iid in valid_items]
            if len(filtered) >= min_count:
                new_sequences[user] = filtered

        if len(new_sequences) == len(sequences):
            break  # Converged
        sequences = new_sequences

    return sequences


# ── 4. Leave-one-out split (Section 4.1.1) ───────────────────────────────────

def leave_one_out_split(sequences: dict, max_len: int = 10):
    """
    Split each user's sequence:
      - test  = last item
      - val   = second-to-last item
      - train = everything else (capped at max_len most recent)

    Returns three dicts: train_seqs, val_seqs, test_seqs
    Each entry: { user_id: {"history": [...titles...], "target_id": str, "target_title": str} }
    """
    train_seqs, val_seqs, test_seqs = {}, {}, {}

    for user, seq in sequences.items():
        if len(seq) < 3:
            continue  # Need at least 3 items for a valid split

        # Trim to max_len + 2 (we'll pop test and val)
        if len(seq) > max_len + 2:
            seq = seq[-(max_len + 2):]

        test_item  = seq[-1]
        val_item   = seq[-2]
        train_hist = seq[:-2]

        # History is represented as titles (Section 3.1)
        history_titles = [title for _, title in train_hist]

        train_seqs[user] = {
            "history": history_titles,
            "target_id": test_item[0],
            "target_title": test_item[1],
        }
        val_seqs[user] = {
            "history": history_titles,
            "target_id": val_item[0],
            "target_title": val_item[1],
        }
        test_seqs[user] = {
            "history": history_titles,
            "target_id": test_item[0],
            "target_title": test_item[1],
        }

    return train_seqs, val_seqs, test_seqs


# ── 5. Build item title lookup ────────────────────────────────────────────────

def build_item_lookup(sequences: dict) -> dict[str, str]:
    """Returns { item_id: item_title } for all items in the dataset."""
    item_lookup = {}
    for seq in sequences.values():
        for item_id, title in seq:
            item_lookup[item_id] = title
    return item_lookup


# ── 6. PyTorch Dataset classes ────────────────────────────────────────────────

class CSFTDataset(Dataset):
    """
    Dataset for Stage 1: Collaborative Supervised Fine-Tuning (Section 3.2).

    Each sample is (history_titles, target_title).
    The instruction format follows Figure 3 of the paper:
      Input:  "item1, item2, item3, ..."
      Output: "target_item"
    """

    def __init__(self, sequences: dict):
        self.samples = []
        for user_data in sequences.values():
            history = user_data["history"]
            target  = user_data["target_title"]
            if history:  # Skip empty histories
                self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        # Format: comma-separated item titles (Section 3.2, Figure 3)
        input_text  = ", ".join(history)
        target_text = target
        return {"input_text": input_text, "target_text": target_text}


class IEMDataset(Dataset):
    """
    Dataset for Stage 2: Item-level Embedding Modeling (Section 3.3).

    Each sample is a single item title.
    Used for both MNTP and contrastive learning.
    """

    def __init__(self, item_lookup: dict[str, str]):
        self.items = list(item_lookup.values())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {"item_title": self.items[idx]}


class RecommenderDataset(Dataset):
    """
    Dataset for downstream recommenders (GRU4Rec, SASRec).
    Provides item-id sequences for training the recommender.
    """

    def __init__(self, sequences: dict, item_lookup: dict, max_len: int = 10):
        self.samples   = []
        self.item2id   = {iid: i for i, iid in enumerate(item_lookup.keys())}
        self.max_len   = max_len

        for user_data in sequences.values():
            history_ids = [
                self.item2id[iid]
                for iid in [
                    # Map titles back to ids via reverse lookup
                ]
            ]
            target_id = self.item2id.get(user_data["target_id"], None)
            if target_id is not None:
                self.samples.append((history_ids, target_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        # Pad or truncate history to max_len
        if len(history) < self.max_len:
            history = [0] * (self.max_len - len(history)) + history
        else:
            history = history[-self.max_len:]
        return {"history": history, "target": target}


# ── 7. Full preprocessing pipeline ───────────────────────────────────────────

def preprocess_dataset(data_path: str, max_len: int = 10, min_core: int = 5):
    """
    Full pipeline: load → build sequences → 5-core filter → split.
    Returns: train_seqs, val_seqs, test_seqs, item_lookup
    """
    print(f"Loading data from {data_path}...")
    interactions = load_dataset(data_path)

    print("Building sequences...")
    sequences = build_sequences(interactions)
    print(f"  Users before filtering: {len(sequences)}")

    print(f"Applying {min_core}-core filtering...")
    sequences = five_core_filter(sequences, min_count=min_core)
    print(f"  Users after filtering:  {len(sequences)}")

    item_lookup = build_item_lookup(sequences)
    print(f"  Total items: {len(item_lookup)}")

    print("Splitting (leave-one-out)...")
    train_seqs, val_seqs, test_seqs = leave_one_out_split(sequences, max_len=max_len)
    print(f"  Train samples: {len(train_seqs)}")

    return train_seqs, val_seqs, test_seqs, item_lookup
