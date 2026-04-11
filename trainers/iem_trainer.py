"""
trainers/iem_trainer.py
-----------------------
Stage 2: Item-level Embedding Modeling (Section 3.3)

Two sub-stages:
  2a. MNTP — Masked Next Token Prediction with BIDIRECTIONAL attention (Eq. 3)
      - Switch causal mask → full (bidirectional) attention
      - Randomly mask 20% of tokens, predict them
      - 1,000 steps, batch=32
      - L_MNTP = -sum_{i in I} sum_{s=0}^{l_i} p(t_{i,s} | t_{i,<s})

  2b. Item-level Contrastive Learning (Eq. 4)
      - Pass each item title through model TWICE with different random dropout
      - Average-pool token hidden states → item embedding
      - InfoNCE loss with in-batch negatives, temperature τ=0.2
      - 1,000 steps, batch=256
      - L_IC = -sum_{i} log[ E(t1_i)·E(t2_i)/τ / sum_j E(t1_j)·E(t2_j)/τ ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW


# ── Helper: switch model to bidirectional attention ───────────────────────────

def enable_bidirectional_attention(model):
    """
    Converts the model from causal (unidirectional) to bidirectional attention.
    This is done by removing the causal attention mask so every token
    can attend to every other token in both directions. (Section 3.3.1)

    Works for Qwen2 / LLaMA-style models by patching the forward pass.
    """
    # Qwen2 / LLaMA store causal mask logic in their attention modules.
    # We disable it by monkey-patching the model config.
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    # For each attention layer, disable causal masking
    for layer in model.model.layers:
        attn = layer.self_attn
        # Store original forward
        original_forward = attn.forward

        def make_bidirectional_forward(orig_forward):
            def bidirectional_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                **kwargs,
            ):
                # Pass a full (non-causal) attention mask — all ones
                # This allows every token to attend to every other token
                bsz, seq_len, _ = hidden_states.shape
                full_mask = torch.zeros(
                    bsz, 1, seq_len, seq_len,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )  # 0 = attend (no masking needed in additive mask form)
                return orig_forward(
                    hidden_states,
                    attention_mask=full_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )
            return bidirectional_forward

        attn.forward = make_bidirectional_forward(original_forward)

    print("Bidirectional attention enabled.")
    return model


# ── Helper: average pool token embeddings → item embedding ───────────────────

def average_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average-pool token-level hidden states into a single item embedding.
    Masks out padding tokens. (Section 3.3.2, Eq. before Eq. 4)

    z_i = (1/l_i) * sum_j z^j_i
    """
    mask = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
    summed = (hidden_states * mask).sum(dim=1)            # (B, H)
    count  = mask.sum(dim=1).clamp(min=1e-9)              # (B, 1)
    return summed / count                                  # (B, H)


# ── 2a. MNTP Collator ────────────────────────────────────────────────────────

class MNTPCollator:
    """
    Tokenizes item titles and randomly masks 20% of tokens.
    Masked positions become the prediction targets (labels).
    Unmasked positions get label=-100 (ignored in loss).
    """

    def __init__(self, tokenizer, max_len: int = 64, mask_prob: float = 0.2):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.mask_prob = mask_prob
        # Use [MASK] token if available, else use a random token
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None) or tokenizer.unk_token_id

    def __call__(self, batch: list[dict]) -> dict:
        texts = [x["item_title"] for x in batch]
        enc   = self.tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].clone()
        labels         = torch.full_like(input_ids, -100)
        attention_mask = enc["attention_mask"]

        # Randomly select 20% of non-padding tokens to mask
        for i in range(input_ids.size(0)):
            non_pad = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            n_mask  = max(1, int(len(non_pad) * self.mask_prob))
            perm    = torch.randperm(len(non_pad))[:n_mask]
            mask_positions = non_pad[perm]

            # Save original ids as labels, replace input with mask token
            labels[i, mask_positions]     = input_ids[i, mask_positions]
            input_ids[i, mask_positions]  = self.mask_token_id

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── 2b. Contrastive Collator ─────────────────────────────────────────────────

class ContrastiveCollator:
    """
    Tokenizes item titles. Returns TWO tokenized views of each item
    (the dropout randomness during forward pass creates the augmentation).
    (Section 3.3.2)
    """

    def __init__(self, tokenizer, max_len: int = 64):
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __call__(self, batch: list[dict]) -> dict:
        texts = [x["item_title"] for x in batch]
        # Tokenize once — the two views come from two separate forward passes
        # with model dropout enabled
        enc = self.tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


# ── InfoNCE contrastive loss (Equation 4) ────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Symmetric InfoNCE loss from Eq. 4:
      L_IC = -sum_i log [ E(t1_i)·E(t2_i)/τ / sum_j E(t1_j)·E(t2_j)/τ ]

    z1, z2: (B, H) normalized item embeddings from two dropout views.
    In-batch negatives: all other items in the batch are negatives.
    """
    # L2 normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Similarity matrix: (B, B)
    sim_matrix = torch.mm(z1, z2.T) / temperature

    # Labels: diagonal (each item is its own positive)
    labels = torch.arange(z1.size(0), device=z1.device)

    # Symmetric loss (both directions)
    loss = (
        F.cross_entropy(sim_matrix, labels) +
        F.cross_entropy(sim_matrix.T, labels)
    ) / 2
    return loss


# ── IEM Trainer ───────────────────────────────────────────────────────────────

class IEMTrainer:
    """
    Stage 2 trainer. Loads the CSFT checkpoint, enables bidirectional attention,
    then runs MNTP followed by contrastive learning.
    """

    def __init__(
        self,
        csft_model_path: str,
        device: str = "cuda",
        mntp_lr: float = 3e-4,
        mntp_steps: int = 1_000,
        mntp_batch_size: int = 32,
        cl_lr: float = 2e-4,
        cl_steps: int = 1_000,
        cl_batch_size: int = 256,
        cl_temperature: float = 0.2,
        cl_dropout: float = 0.2,
        mask_prob: float = 0.2,
        save_path: str = "./checkpoints/iem",
    ):
        self.device         = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mntp_steps     = mntp_steps
        self.cl_steps       = cl_steps
        self.cl_batch_size  = cl_batch_size
        self.cl_temperature = cl_temperature
        self.save_path      = save_path

        print(f"Loading CSFT checkpoint from {csft_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(csft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            csft_model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(self.device)

        # Set dropout for contrastive augmentation (Section 3.3.2)
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = cl_dropout

        # Enable bidirectional attention (Section 3.3.1)
        self.model = enable_bidirectional_attention(self.model)

        self.mntp_lr    = mntp_lr
        self.cl_lr      = cl_lr
        self.mask_prob  = mask_prob
        self.mntp_batch = mntp_steps

    def run_mntp(self, item_dataset):
        """Step 2a: Masked Next Token Prediction (Equation 3)."""
        print("\n--- Stage 2a: MNTP ---")
        collator   = MNTPCollator(self.tokenizer, mask_prob=self.mask_prob)
        dataloader = DataLoader(
            item_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
        )
        optimizer = AdamW(self.model.parameters(), lr=self.mntp_lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=self.mntp_steps,
        )

        self.model.train()
        step = 0
        while step < self.mntp_steps:
            for batch in dataloader:
                if step >= self.mntp_steps:
                    break
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % 100 == 0:
                    print(f"  MNTP step {step}/{self.mntp_steps} | Loss: {loss.item():.4f}")
                step += 1

        print("MNTP complete.")

    def run_contrastive(self, item_dataset):
        """Step 2b: Item-level Contrastive Learning (Equation 4)."""
        print("\n--- Stage 2b: Contrastive Learning ---")
        collator   = ContrastiveCollator(self.tokenizer)
        dataloader = DataLoader(
            item_dataset,
            batch_size=self.cl_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        optimizer = AdamW(self.model.parameters(), lr=self.cl_lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=self.cl_steps,
        )

        self.model.train()
        step = 0
        while step < self.cl_steps:
            for batch in dataloader:
                if step >= self.cl_steps:
                    break
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Two forward passes with dropout enabled = two different views
                # (Section 3.3.2: "passed through LLM twice with random masking")
                out1 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                out2 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Average pool last hidden state → item embedding
                h1 = out1.hidden_states[-1]   # (B, L, H)
                h2 = out2.hidden_states[-1]

                z1 = average_pool(h1, attention_mask)  # (B, H)
                z2 = average_pool(h2, attention_mask)

                loss = info_nce_loss(z1, z2, temperature=self.cl_temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % 100 == 0:
                    print(f"  CL step {step}/{self.cl_steps} | Loss: {loss.item():.4f}")
                step += 1

        print("Contrastive learning complete.")

    def train(self, item_dataset):
        """Run both MNTP and contrastive learning sequentially."""
        self.run_mntp(item_dataset)
        self.run_contrastive(item_dataset)
        self._save_model()

    def get_item_embeddings(self, item_titles: list[str], batch_size: int = 64) -> torch.Tensor:
        """
        Generate embeddings for a list of item titles.
        Used downstream by recommenders (Section 3.4).
        E(i) = avg_pool(LLM(t_i))
        """
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(item_titles), batch_size):
                batch_titles = item_titles[i: i + batch_size]
                enc = self.tokenizer(
                    batch_titles,
                    max_length=64,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(self.device)

                out = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True,
                )
                h = out.hidden_states[-1]
                z = average_pool(h, enc["attention_mask"])
                all_embeddings.append(z.cpu().float())

        return torch.cat(all_embeddings, dim=0)   # (N_items, H)

    def _save_model(self):
        import os
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"IEM model saved to {self.save_path}")
