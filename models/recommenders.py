"""
models/recommenders.py
----------------------
Downstream sequential recommenders (Section 3.4, Appendix A.2)

Both models accept pre-computed LLM2Rec embeddings via a linear adapter:
  z'_i = W * z_i + b    (Section 3.4)

The adapter projects LLM hidden dim (e.g. 896 for Qwen2-0.5B) → 128 dims.

GRU4Rec (ref [12] in paper):
  - GRU-based sequential recommender
  - Input: sequence of adapted item embeddings
  - Output: score for each candidate item

SASRec (ref [15] in paper):
  - Transformer self-attention based recommender
  - Input: sequence of adapted item embeddings
  - Output: score for each candidate item
"""

import torch
import torch.nn as nn
import math


# ── Linear adapter (Section 3.4) ─────────────────────────────────────────────

class EmbeddingAdapter(nn.Module):
    """
    z'_i = W * z_i + b
    Projects LLM2Rec embedding dimension to recommender hidden dimension.
    Parameters are optimised by the downstream recommender objective.
    """

    def __init__(self, llm_dim: int, rec_dim: int):
        super().__init__()
        self.linear = nn.Linear(llm_dim, rec_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


# ── GRU4Rec (Appendix A.2, ref [12]) ─────────────────────────────────────────

class GRU4Rec(nn.Module):
    """
    GRU-based sequential recommender.
    Uses GRU to model the sequential dynamics of user interactions.
    Training objective: cross-entropy loss (Appendix A.2).

    Input:  item_embs  (B, L, llm_dim) — LLM2Rec embeddings for history
            item_table (N, llm_dim)    — all item embeddings for scoring
    Output: logits     (B, N)          — score over all items
    """

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.adapter = EmbeddingAdapter(llm_dim, hidden_dim)
        self.gru     = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        history_embs: torch.Tensor,   # (B, L, llm_dim)
        item_table: torch.Tensor,     # (N, llm_dim)
    ) -> torch.Tensor:
        # Adapt embeddings to recommender dimension
        h = self.adapter(history_embs)     # (B, L, D)
        h = self.dropout(h)

        # GRU over history
        out, _ = self.gru(h)              # (B, L, D)
        user_repr = out[:, -1, :]         # (B, D) — last hidden state

        # Adapt item table
        item_repr = self.adapter(item_table)   # (N, D)

        # Dot-product scoring
        logits = torch.mm(user_repr, item_repr.T)  # (B, N)
        return logits

    def get_user_repr(self, history_embs: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.adapter(history_embs))
        out, _ = self.gru(h)
        return out[:, -1, :]


# ── SASRec (Appendix A.2, ref [15]) ──────────────────────────────────────────

class SASRecBlock(nn.Module):
    """Single transformer block for SASRec."""

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.3):
        super().__init__()
        self.attn    = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation.
    Uses causal (left-to-right) self-attention over item sequence.
    Training objective: cross-entropy loss (Appendix A.2).

    Input:  history_embs (B, L, llm_dim) — LLM2Rec embeddings for history
            item_table   (N, llm_dim)    — all item embeddings
    Output: logits       (B, N)          — score over all items
    """

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int = 128,
        max_len: int = 10,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.adapter  = EmbeddingAdapter(llm_dim, hidden_dim)
        self.pos_emb  = nn.Embedding(max_len + 1, hidden_dim)  # Learnable position embeddings
        self.blocks   = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(
        self,
        history_embs: torch.Tensor,   # (B, L, llm_dim)
        item_table: torch.Tensor,     # (N, llm_dim)
    ) -> torch.Tensor:
        B, L, _ = history_embs.shape

        # Adapt embeddings
        h = self.adapter(history_embs)   # (B, L, D)

        # Add positional embeddings
        positions = torch.arange(1, L + 1, device=h.device).unsqueeze(0)  # (1, L)
        h = self.dropout(h + self.pos_emb(positions))

        # Causal mask for self-attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, attn_mask=causal_mask)

        h = self.norm(h)
        user_repr = h[:, -1, :]             # (B, D) — last position

        # Adapt item table
        item_repr = self.adapter(item_table)  # (N, D)

        logits = torch.mm(user_repr, item_repr.T)  # (B, N)
        return logits

    def get_user_repr(self, history_embs: torch.Tensor) -> torch.Tensor:
        B, L, _ = history_embs.shape
        h = self.adapter(history_embs)
        positions = torch.arange(1, L + 1, device=h.device).unsqueeze(0)
        h = self.dropout(h + self.pos_emb(positions))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device)
        for block in self.blocks:
            h = block(h, attn_mask=causal_mask)
        return self.norm(h)[:, -1, :]
