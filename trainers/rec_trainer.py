"""
trainers/rec_trainer.py
-----------------------
Trains GRU4Rec / SASRec on top of LLM2Rec embeddings. (Section 3.4 / 4.1.3)

Paper hyperparameters:
  - Loss:         Cross-entropy
  - Optimizer:    AdamW
  - LR:           1e-3 (SASRec), 1e-4 (GRU4Rec)
  - Weight decay: 1e-4
  - Dropout:      0.3
  - Projected dim: 128 (linear adapter)
  - Max epochs:   500
  - Early stop:   20 epochs no improvement on val NDCG@10
  - Evaluation:   Full ranking (all items as candidates)
  - Metrics:      Recall@10, Recall@20, NDCG@10, NDCG@20
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


# ── Dataset for recommender training ─────────────────────────────────────────

class RecDataset(Dataset):
    """
    Each sample: (history_item_indices, target_item_index)
    history_item_indices: list of item ids (ints) representing the history
    """

    def __init__(self, sequences: dict, item2idx: dict[str, int], max_len: int = 10):
        self.samples = []
        self.max_len = max_len

        for user_data in sequences.values():
            # Map item titles to indices
            history_idx = [
                item2idx[t] for t in user_data["history"] if t in item2idx
            ]
            target_idx  = item2idx.get(user_data["target_title"])
            if target_idx is None or len(history_idx) == 0:
                continue
            self.samples.append((history_idx, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        # Truncate or pad history to max_len
        if len(history) > self.max_len:
            history = history[-self.max_len:]
        else:
            history = [0] * (self.max_len - len(history)) + history  # Left-pad with 0
        return {
            "history": torch.tensor(history, dtype=torch.long),
            "target":  torch.tensor(target,  dtype=torch.long),
        }


# ── Evaluation metrics (Section 4.1.1) ───────────────────────────────────────

def recall_at_k(ranks: list[int], k: int) -> float:
    """Recall@K: fraction of users where target item is in top-K."""
    return float(np.mean([1.0 if r < k else 0.0 for r in ranks]))


def ndcg_at_k(ranks: list[int], k: int) -> float:
    """NDCG@K: normalised discounted cumulative gain."""
    scores = []
    for r in ranks:
        if r < k:
            scores.append(1.0 / math.log2(r + 2))   # rank is 0-indexed
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ── Full-ranking evaluation (Section 4.1.1) ───────────────────────────────────

@torch.no_grad()
def full_ranking_evaluate(
    model,
    dataset: RecDataset,
    item_emb_table: torch.Tensor,    # (N, llm_dim) — all LLM2Rec embeddings
    device: torch.device,
    batch_size: int = 256,
    ks: list[int] = [10, 20],
) -> dict:
    """
    Full ranking: score ALL items for each user, rank the target item.
    Paper: "perform full ranking with all items in the dataset as candidates"
    """
    import math
    model.eval()
    item_emb_table = item_emb_table.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ranks = []

    for batch in dataloader:
        history = batch["history"].to(device)         # (B, L)
        targets = batch["target"].to(device)          # (B,)

        # Get item embeddings for each history position
        # history shape: (B, L) — indices into item_emb_table
        history_embs = item_emb_table[history]        # (B, L, llm_dim)

        # Score all items
        logits = model(history_embs, item_emb_table)  # (B, N)

        # Find rank of target item for each user
        for i, target_idx in enumerate(targets):
            score_row    = logits[i]                  # (N,)
            target_score = score_row[target_idx].item()
            # Rank = number of items with higher score (0-indexed)
            rank = (score_row > target_score).sum().item()
            all_ranks.append(rank)

    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(all_ranks, k)
        results[f"NDCG@{k}"]   = ndcg_at_k(all_ranks, k)
    return results


# ── Recommender Trainer ───────────────────────────────────────────────────────

class RecTrainer:
    """
    Trains a downstream recommender (GRU4Rec or SASRec) using
    LLM2Rec item embeddings, with early stopping on val NDCG@10.
    """

    def __init__(
        self,
        model,
        item_emb_table: torch.Tensor,   # (N, llm_dim)
        model_name: str = "SASRec",
        device: str = "cuda",
        max_epochs: int = 500,
        early_stop_patience: int = 20,
        batch_size: int = 256,
    ):
        self.model          = model.to(torch.device(device if torch.cuda.is_available() else "cpu"))
        self.item_emb_table = item_emb_table
        self.model_name     = model_name
        self.device         = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_epochs     = max_epochs
        self.patience       = early_stop_patience
        self.batch_size     = batch_size

        # Paper hyperparameters (Section 4.1.3)
        lr = 1e-3 if model_name == "SASRec" else 1e-4
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, train_dataset: RecDataset, val_dataset: RecDataset):
        import math

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        item_emb = self.item_emb_table.to(self.device)

        best_ndcg   = -1.0
        no_improve  = 0
        best_state  = None

        for epoch in range(1, self.max_epochs + 1):
            # ── Training ──
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                history = batch["history"].to(self.device)  # (B, L)
                targets = batch["target"].to(self.device)   # (B,)

                history_embs = item_emb[history]             # (B, L, llm_dim)
                logits       = self.model(history_embs, item_emb)  # (B, N)

                loss = self.loss_fn(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

            # ── Validation ──
            val_metrics = full_ranking_evaluate(
                self.model, val_dataset, self.item_emb_table, self.device,
                batch_size=self.batch_size,
            )
            val_ndcg = val_metrics["NDCG@10"]

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} "
                    f"| Val NDCG@10: {val_ndcg:.4f} | Val R@10: {val_metrics['Recall@10']:.4f}"
                )

            # ── Early stopping ──
            if val_ndcg > best_ndcg:
                best_ndcg  = val_ndcg
                no_improve = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs).")
                    break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"Best Val NDCG@10: {best_ndcg:.4f}")

    def test(self, test_dataset: RecDataset) -> dict:
        metrics = full_ranking_evaluate(
            self.model, test_dataset, self.item_emb_table, self.device,
            batch_size=self.batch_size,
        )
        print("\n=== Test Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return metrics


import math  # needed by ndcg_at_k inside the module
