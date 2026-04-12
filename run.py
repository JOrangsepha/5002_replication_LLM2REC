"""
run.py
------
Main entry point for the full LLM2Rec pipeline.

Usage:
  python run.py --data_dir ./data --dataset AmazonMix-6 --stage all

Stages:
  1. csft   — Collaborative Supervised Fine-Tuning
  2. iem    — Item-level Embedding Modeling (MNTP + contrastive)
  3. embed  — Generate item embeddings from trained model
  4. rec    — Train and evaluate downstream recommender
  all       — Run all stages in order
"""

import os
import argparse
import torch
import pickle

from data.dataset import load_dataset_for_experiment
from trainers.csft_trainer import CSFTTrainer
from trainers.iem_trainer import IEMTrainer
from trainers.rec_trainer import RecTrainer
from models.recommenders import GRU4Rec, SASRec


def parse_args():
    parser = argparse.ArgumentParser(description="LLM2Rec Pipeline")
    parser.add_argument("--data_dir",     type=str,   default="./data")
    parser.add_argument("--dataset",      type=str,   default="Games",
                        help="Dataset name matching file in data_dir")
    parser.add_argument("--stage",        type=str,   default="all",
                        choices=["csft", "iem", "embed", "rec", "all"])
    parser.add_argument("--rec_model",    type=str,   default="SASRec",
                        choices=["SASRec", "GRU4Rec"])
    parser.add_argument("--backbone",     type=str,   default="Qwen/Qwen2-0.5B")
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--output_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--max_seq_len",  type=int,   default=10)
    # CSFT
    parser.add_argument("--csft_steps",   type=int,   default=10_000)
    parser.add_argument("--csft_batch",   type=int,   default=128)
    # IEM
    parser.add_argument("--mntp_steps",   type=int,   default=1_000)
    parser.add_argument("--cl_steps",     type=int,   default=1_000)
    parser.add_argument("--cl_batch",     type=int,   default=256)
    # Rec
    parser.add_argument("--max_epochs",   type=int,   default=500)
    parser.add_argument("--early_stop",   type=int,   default=20)
    parser.add_argument("--seeds",        type=int,   nargs="+", default=[42, 123, 2024],
                        help="Random seeds for averaging results (paper uses 3 seeds)")
    return parser.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_path     = os.path.join(args.data_dir, f"{args.dataset}.json")
    csft_ckpt     = os.path.join(args.output_dir, "csft")
    iem_ckpt      = os.path.join(args.output_dir, "iem")
    embed_path    = os.path.join(args.output_dir, f"{args.dataset}_embeddings.pt")
    meta_path     = os.path.join(args.output_dir, f"{args.dataset}_meta.pkl")

    # ── Step 0: Preprocessing ──────────────────────────────────────────────
    print("=" * 60)
    print(f"Loading dataset: {args.dataset}")
    (csft_dataset, iem_dataset,
    train_rec_dataset, val_rec_dataset, test_rec_dataset,
    item_titles, item2idx) = load_dataset_for_experiment(args.data_dir, args.dataset)

     # Save metadata for later stages
    with open(meta_path, "wb") as f:
        pickle.dump({"item_titles": item_titles, "item2idx": item2idx}, f)

    # ── Stage 1: CSFT ─────────────────────────────────────────────────────
    if args.stage in ("csft", "all"):
        print("\n" + "=" * 60)
        print("Stage 1: Collaborative Supervised Fine-Tuning (CSFT)")

        trainer = CSFTTrainer(
            model_name=args.backbone,
            device=args.device,
            lr=3e-4,
            total_steps=args.csft_steps,
            effective_batch_size=args.csft_batch,
            micro_batch_size=16,
            save_path=csft_ckpt,
        )
        trainer.train(csft_dataset)

    # ── Stage 2: IEM ──────────────────────────────────────────────────────
    if args.stage in ("iem", "all"):
        print("\n" + "=" * 60)
        print("Stage 2: Item-level Embedding Modeling (IEM)")

        iem_trainer = IEMTrainer(
            csft_model_path=csft_ckpt,
            device=args.device,
            mntp_steps=args.mntp_steps,
            cl_steps=args.cl_steps,
            cl_batch_size=args.cl_batch,
            save_path=iem_ckpt,
        )
        iem_trainer.train(iem_dataset)

    # ── Stage 3: Generate item embeddings ─────────────────────────────────
    if args.stage in ("embed", "all"):
        print("\n" + "=" * 60)
        print("Stage 3: Generating item embeddings")

        iem_trainer = IEMTrainer(
            csft_model_path=iem_ckpt,
            device=args.device,
        )
        item_emb_table = iem_trainer.get_item_embeddings(item_titles, batch_size=64)
        torch.save(item_emb_table, embed_path)
        print(f"Saved embeddings: {item_emb_table.shape} → {embed_path}")

    # ── Stage 4: Train & evaluate recommender ─────────────────────────────
    if args.stage in ("rec", "all"):
        print("\n" + "=" * 60)
        print(f"Stage 4: Training {args.rec_model}")

        item_emb_table = torch.load(embed_path)
        llm_dim        = item_emb_table.shape[1]

        train_dataset = train_rec_dataset
        val_dataset   = val_rec_dataset
        test_dataset  = test_rec_dataset

        # Average results across multiple seeds (Section 4.1.1)
        all_results = []
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)

            if args.rec_model == "SASRec":
                rec_model = SASRec(llm_dim=llm_dim, hidden_dim=128,
                                   max_len=args.max_seq_len, dropout=0.3)
            else:
                rec_model = GRU4Rec(llm_dim=llm_dim, hidden_dim=128, dropout=0.3)

            rec_trainer = RecTrainer(
                model=rec_model,
                item_emb_table=item_emb_table,
                model_name=args.rec_model,
                device=args.device,
                max_epochs=args.max_epochs,
                early_stop_patience=args.early_stop,
            )
            rec_trainer.train(train_dataset, val_dataset)
            results = rec_trainer.test(test_dataset)
            all_results.append(results)

        # Average across seeds
        print("\n=== Final averaged results across seeds ===")
        avg_results = {}
        for metric in all_results[0]:
            avg = sum(r[metric] for r in all_results) / len(all_results)
            avg_results[metric] = avg
            print(f"  {metric}: {avg:.4f}")


if __name__ == "__main__":
    main()
