# LLM2Rec — Implementation from Scratch
# Based on: "LLM2Rec: Large Language Models Are Powerful Embedding Models
#             for Sequential Recommendation" (KDD '25)

## Project structure
```
llm2rec/
├── data/
│   └── dataset.py          # Data loading, 5-core filter, leave-one-out split
├── models/
│   └── recommenders.py     # GRU4Rec, SASRec, EmbeddingAdapter
├── trainers/
│   ├── csft_trainer.py     # Stage 1: Collaborative Supervised Fine-Tuning
│   ├── iem_trainer.py      # Stage 2: MNTP + Contrastive Learning
│   └── rec_trainer.py      # Downstream recommender training + evaluation
├── run.py                  # Main pipeline entry point
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Data
Download pre-processed datasets from the authors' repo:
  https://github.com/HappyPointer/LLM2Rec  (./data/ folder)

Place in ./data/  e.g.  ./data/Games.json

## Running

### Full pipeline (all stages)
```bash
python run.py --dataset Games --stage all --rec_model SASRec --device cuda
```

### Individual stages
```bash
# Stage 1 only
python run.py --dataset Games --stage csft

# Stage 2 only (requires CSFT checkpoint)
python run.py --dataset Games --stage iem

# Generate embeddings (requires IEM checkpoint)
python run.py --dataset Games --stage embed

# Train recommender (requires embeddings)
python run.py --dataset Games --stage rec --rec_model GRU4Rec
```

### Out-of-domain evaluation
```bash
# Train on Games (in-domain), test on Sports (out-of-domain)
# First generate embeddings for Sports using the model trained on Games:
python run.py --dataset Sports --stage embed
python run.py --dataset Sports --stage rec --rec_model SASRec
```

## Key hyperparameters (matching the paper)

| Stage       | Parameter            | Value          |
|-------------|----------------------|----------------|
| CSFT        | Backbone             | Qwen2-0.5B     |
| CSFT        | Learning rate        | 3e-4           |
| CSFT        | Steps                | 10,000         |
| CSFT        | Effective batch      | 128            |
| IEM (MNTP)  | Mask probability     | 20%            |
| IEM (MNTP)  | Steps                | 1,000          |
| IEM (MNTP)  | Batch size           | 32             |
| IEM (CL)    | Temperature τ        | 0.2            |
| IEM (CL)    | Dropout              | 0.2            |
| IEM (CL)    | Steps                | 1,000          |
| IEM (CL)    | Batch size           | 256            |
| Rec         | Hidden dim           | 128            |
| Rec (SASRec)| Learning rate        | 1e-3           |
| Rec (GRU4Rec)| Learning rate       | 1e-4           |
| Rec         | Weight decay         | 1e-4           |
| Rec         | Dropout              | 0.3            |
| Rec         | Max epochs           | 500            |
| Rec         | Early stop patience  | 20 epochs      |
| Rec         | Seeds                | 42, 123, 2024  |

## GPU requirements
- Stage 1 (CSFT): GPU required. ~8-16GB VRAM with micro_batch_size=16
  and gradient accumulation. Use A100/A40 or Colab Pro+.
- Stage 2 (IEM): GPU recommended. <2 hours on a single A40.
- Stage 4 (Rec): Light — can run on CPU, but GPU is faster.
