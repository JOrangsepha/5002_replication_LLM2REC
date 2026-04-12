import os, ast, json, torch
import pandas as pd
from torch.utils.data import Dataset

# ── Format A: CSV (Amazon train/valid/test splits) ────────────────────────────
def load_csv_split(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    samples = []
    for _, row in df.iterrows():
        history_titles = ast.literal_eval(row["history_item_title"])
        target_title   = str(row["item_title"])
        target_id      = int(row["item_id"])
        if history_titles:
            samples.append({"history": history_titles,
                            "target_title": target_title,
                            "target_id": target_id})
    return samples

# ── Format B+C: TXT integer sequences + item_titles.json ─────────────────────
def load_item_titles_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def load_txt_split(txt_path, item_titles):
    samples = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            ids = line.strip().split()
            if len(ids) < 2:
                continue
            history_titles = [item_titles.get(str(i), f"item_{i}") for i in ids[:-1]]
            target_title   = item_titles.get(str(ids[-1]), f"item_{ids[-1]}")
            samples.append({"history": history_titles,
                            "target_title": target_title,
                            "target_id": int(ids[-1])})
    return samples

# ── AmazonMix-6 item_titles.txt ("title\tindex" per line) ────────────────────
def load_item_titles_txt(txt_path):
    titles = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                title, idx = line.rsplit("\t", 1)
                titles[idx.strip()] = title.strip()
    return titles

# ── High-level loaders ────────────────────────────────────────────────────────
def load_amazon_dataset(dataset_dir):
    core_dir       = os.path.join(dataset_dir, "5-core")
    downstream_dir = os.path.join(core_dir, "downstream")

    def find_csv(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        return os.path.join(folder, files[0])

    csft_samples  = load_csv_split(find_csv(os.path.join(core_dir, "train")))
    item_titles   = load_item_titles_json(os.path.join(downstream_dir, "item_titles.json"))
    train_samples = load_txt_split(os.path.join(downstream_dir, "train_data.txt"), item_titles)
    val_samples   = load_txt_split(os.path.join(downstream_dir, "val_data.txt"),   item_titles)
    test_samples  = load_txt_split(os.path.join(downstream_dir, "test_data.txt"),  item_titles)
    return csft_samples, train_samples, val_samples, test_samples, item_titles

def load_goodreads_dataset(dataset_dir):
    clean_dir   = os.path.join(dataset_dir, "clean")
    item_titles = load_item_titles_json(os.path.join(clean_dir, "item_titles.json"))
    train_samples = load_txt_split(os.path.join(clean_dir, "train_data.txt"), item_titles)
    val_samples   = load_txt_split(os.path.join(clean_dir, "val_data.txt"),   item_titles)
    test_samples  = load_txt_split(os.path.join(clean_dir, "test_data.txt"),  item_titles)
    # Goodreads has no separate CSV — train_samples used for both CSFT and rec
    return train_samples, train_samples, val_samples, test_samples, item_titles

def load_amazonmix6_dataset(dataset_dir):
    core_dir     = os.path.join(dataset_dir, "5-core")
    csft_samples = load_csv_split(os.path.join(core_dir, "train", "AmazonMix-6.csv"))
    item_titles  = load_item_titles_txt(os.path.join(core_dir, "info", "item_titles.txt"))
    return csft_samples, item_titles

# ── PyTorch Dataset classes (unchanged interface, works with new loaders) ─────
class CSFTDataset(Dataset):
    def __init__(self, samples):
        self.samples = [(s["history"], s["target_title"]) for s in samples if s["history"]]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        history, target = self.samples[idx]
        return {"input_text": ", ".join(history), "target_text": target}

class IEMDataset(Dataset):
    def __init__(self, item_titles): self.items = item_titles
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return {"item_title": self.items[idx]}

class RecDataset(Dataset):
    def __init__(self, samples, item2idx, max_len=10):
        self.max_len = max_len
        self.samples = []
        for s in samples:
            history_idx = [item2idx[t] for t in s["history"] if t in item2idx]
            target_idx  = item2idx.get(s["target_title"])
            if target_idx is not None and history_idx:
                self.samples.append((history_idx, target_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        history, target = self.samples[idx]
        if len(history) > self.max_len:
            history = history[-self.max_len:]
        else:
            history = [0] * (self.max_len - len(history)) + history
        return {"history": torch.tensor(history, dtype=torch.long),
                "target":  torch.tensor(target,  dtype=torch.long)}

# ── Top-level convenience ─────────────────────────────────────────────────────
AMAZON_DATASETS = {"Video_Games","Arts_Crafts_and_Sewing",
                   "Movies_and_TV","Baby_Products","Sports_and_Outdoors"}

def load_dataset_for_experiment(data_root, dataset_name):
    dataset_dir = os.path.join(data_root, dataset_name)
    if dataset_name in AMAZON_DATASETS:
        csft_s, train_s, val_s, test_s, item_titles = load_amazon_dataset(dataset_dir)
    elif dataset_name == "Goodreads":
        csft_s, train_s, val_s, test_s, item_titles = load_goodreads_dataset(dataset_dir)
    elif dataset_name == "AmazonMix-6":
        csft_s, item_titles = load_amazonmix6_dataset(dataset_dir)
        train_s = val_s = test_s = []
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    all_titles = list(set(item_titles.values()))
    item2idx   = {title: i for i, title in enumerate(all_titles)}
    return (CSFTDataset(csft_s), IEMDataset(all_titles),
            RecDataset(train_s, item2idx), RecDataset(val_s, item2idx),
            RecDataset(test_s, item2idx), item_titles, item2idx)
