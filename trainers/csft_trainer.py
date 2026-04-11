"""
trainers/csft_trainer.py
------------------------
Stage 1: Collaborative Supervised Fine-Tuning (Section 3.2)

Key paper details:
- Backbone: Qwen2-0.5B (decoder-only LLM, causal attention kept)
- Input: comma-separated item title history
- Output: next item title (autoregressive)
- Loss: cross-entropy on TARGET TOKENS ONLY (not input tokens)
- Optimizer: AdamW, lr=3e-4
- Steps: 10,000, effective batch size=128
- Loss formula (Eq. 2):
    L_CSFT = -sum_{(u,i) in X} sum_{s=0}^{l_i} p(t_{i,s} | t_u, t_{i,<s})
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW


# ── 1. Collator: formats input/output and masks loss on input tokens ──────────

class CSFTCollator:
    """
    Tokenizes (input_text, target_text) pairs.
    Critically: sets labels=-100 for all INPUT tokens so the loss
    is computed ONLY on target (next item) tokens. (Section 3.2)
    """

    def __init__(self, tokenizer, max_input_len: int = 512, max_target_len: int = 64):
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __call__(self, batch: list[dict]) -> dict:
        input_texts  = [x["input_text"]  for x in batch]
        target_texts = [x["target_text"] for x in batch]

        # Tokenize inputs and targets separately to know boundary
        input_enc = self.tokenizer(
            input_texts,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )
        target_enc = self.tokenizer(
            target_texts,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # No BOS for target, just EOS
        )

        all_input_ids = []
        all_labels    = []
        all_attn_masks = []

        for i in range(len(batch)):
            inp_ids = input_enc["input_ids"][i]
            tgt_ids = target_enc["input_ids"][i] + [self.tokenizer.eos_token_id]

            # Concatenate: [input tokens] + [target tokens]
            full_ids = inp_ids + tgt_ids

            # Labels: -100 for input portion (masked from loss), real ids for target
            labels = [-100] * len(inp_ids) + tgt_ids

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attn_masks.append([1] * len(full_ids))

        # Pad to max length in batch (right-padding)
        max_len = max(len(x) for x in all_input_ids)
        pad_id  = self.tokenizer.pad_token_id or 0

        padded_ids   = []
        padded_labels = []
        padded_masks  = []
        for ids, labs, masks in zip(all_input_ids, all_labels, all_attn_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_masks.append(masks + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(padded_ids,    dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks,  dtype=torch.long),
            "labels":         torch.tensor(padded_labels, dtype=torch.long),
        }


# ── 2. CSFT Trainer ───────────────────────────────────────────────────────────

class CSFTTrainer:
    """
    Fine-tunes a decoder-only LLM for next-item prediction.
    Implements Equation 2 from the paper.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        device: str = "cuda",
        lr: float = 3e-4,
        total_steps: int = 10_000,
        effective_batch_size: int = 128,
        micro_batch_size: int = 16,      # Adjust based on GPU memory
        warmup_steps: int = 500,
        save_path: str = "./checkpoints/csft",
    ):
        self.device              = torch.device(device if torch.cuda.is_available() else "cpu")
        self.total_steps         = total_steps
        self.effective_batch_size = effective_batch_size
        self.micro_batch_size    = micro_batch_size
        self.grad_accum_steps    = effective_batch_size // micro_batch_size
        self.save_path           = save_path

        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train(self, train_dataset, val_dataset=None):
        """
        Main training loop implementing Equation 2.
        Uses gradient accumulation to simulate large batch sizes.
        """
        collator   = CSFTCollator(self.tokenizer)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

        self.model.train()
        global_step = 0
        accum_loss  = 0.0
        self.optimizer.zero_grad()

        print(f"Starting CSFT training for {self.total_steps} steps...")
        print(f"  Micro batch: {self.micro_batch_size}, Grad accum: {self.grad_accum_steps}")

        while global_step < self.total_steps:
            for batch in dataloader:
                if global_step >= self.total_steps:
                    break

                # Move batch to device
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                # Forward pass — HuggingFace computes cross-entropy on labels
                # where label=-100 positions are automatically ignored
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

                # Gradient accumulation step
                if (global_step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    effective_step = (global_step + 1) // self.grad_accum_steps
                    if effective_step % 100 == 0:
                        print(f"  Step {effective_step}/{self.total_steps} | Loss: {accum_loss:.4f}")
                        accum_loss = 0.0

                global_step += 1

        self._save_model()
        print("CSFT training complete.")

    def _save_model(self):
        import os
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"Model saved to {self.save_path}")
