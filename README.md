# "LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation" (KDD '25)

## Project structure
```
llm2rec/
├── data/
│   └── dataset.py          # Data loading for CSV, TXT, and JSON formats
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
Download pre-processed datasets from the authors' official repo:
  https://github.com/HappyPointer/LLM2Rec  (./data/ folder)

Unzip and place the data folder at the project root. The expected structure is:
```
data/
├── AmazonMix-6/            ← used for CSFT pre-training (all 6 Amazon datasets mixed)
├── Video_Games/            ← in-domain evaluation
├── Arts_Crafts_and_Sewing/ ← in-domain evaluation
├── Movies_and_TV/          ← in-domain evaluation
├── Sports_and_Outdoors/    ← out-of-domain evaluation
├── Baby_Products/          ← out-of-domain evaluation
└── Goodreads/              ← out-of-domain evaluation (cross-platform)
```

Each dataset folder contains files in one of three formats:
- CSV  — train/valid/test splits for Amazon datasets (5-core/train/, valid/, test/)
- TXT  — integer ID sequences for downstream recommender (5-core/downstream/)
- JSON — item ID to title mapping (item_titles.json)

## Running

### Recommended order (matches the paper's pipeline)

**Step 1 — CSFT pre-training on AmazonMix-6**
```bash
python run.py --data_dir ./data --dataset AmazonMix-6 --stage csft --device cuda
```

**Step 2 — IEM (MNTP + contrastive learning)**
```bash
python run.py --data_dir ./data --dataset AmazonMix-6 --stage iem --device cuda
```

**Step 3 — Generate item embeddings for each evaluation dataset**
```bash
python run.py --data_dir ./data --dataset Video_Games --stage embed --device cuda
python run.py --data_dir ./data --dataset Arts_Crafts_and_Sewing --stage embed --device cuda
python run.py --data_dir ./data --dataset Movies_and_TV --stage embed --device cuda
python run.py --data_dir ./data --dataset Sports_and_Outdoors --stage embed --device cuda
python run.py --data_dir ./data --dataset Baby_Products --stage embed --device cuda
python run.py --data_dir ./data --dataset Goodreads --stage embed --device cuda
```

**Step 4 — Train and evaluate downstream recommenders**
```bash
# In-domain datasets
python run.py --data_dir ./data --dataset Video_Games --stage rec --rec_model SASRec --device cuda
python run.py --data_dir ./data --dataset Arts_Crafts_and_Sewing --stage rec --rec_model SASRec --device cuda
python run.py --data_dir ./data --dataset Movies_and_TV --stage rec --rec_model SASRec --device cuda

# Out-of-domain datasets
python run.py --data_dir ./data --dataset Sports_and_Outdoors --stage rec --rec_model SASRec --device cuda
python run.py --data_dir ./data --dataset Baby_Products --stage rec --rec_model SASRec --device cuda
python run.py --data_dir ./data --dataset Goodreads --stage rec --rec_model SASRec --device cuda

# Repeat above with --rec_model GRU4Rec for full Table 3 results
```

### Minimal run (for testing or limited compute)
If you only want to verify the pipeline works end-to-end on one dataset:
```bash
python run.py --data_dir ./data --dataset Video_Games --stage all --rec_model SASRec --device cuda
```

### CPU testing (Windows)
When testing on CPU, use `num_workers=0` and `pin_memory=False` in
`trainers/csft_trainer.py` to avoid Windows multiprocessing issues:
```bash
python run.py --data_dir ./data --dataset Video_Games --stage csft --device cpu
```

## Available dataset names
Use exactly these names for --dataset:

| Name                  | Split type  | Used for              |
|-----------------------|-------------|-----------------------|
| AmazonMix-6           | In-domain   | CSFT pre-training     |
| Video_Games           | In-domain   | Evaluation            |
| Arts_Crafts_and_Sewing| In-domain   | Evaluation            |
| Movies_and_TV         | In-domain   | Evaluation            |
| Sports_and_Outdoors   | Out-of-domain | Evaluation          |
| Baby_Products         | Out-of-domain | Evaluation          |
| Goodreads             | Out-of-domain | Evaluation          |

## Key hyperparameters (from the paper, Section 4.1.3)

| Stage          | Parameter            | Value         |
|----------------|----------------------|---------------|
| CSFT           | Backbone             | Qwen2-0.5B    |
| CSFT           | Learning rate        | 3e-4          |
| CSFT           | Steps                | 10,000        |
| CSFT           | Effective batch size | 128           |
| IEM (MNTP)     | Mask probability     | 20%           |
| IEM (MNTP)     | Steps                | 1,000         |
| IEM (MNTP)     | Batch size           | 32            |
| IEM (CL)       | Temperature τ        | 0.2           |
| IEM (CL)       | Dropout              | 0.2           |
| IEM (CL)       | Steps                | 1,000         |
| IEM (CL)       | Batch size           | 256           |
| Rec            | Projected hidden dim | 128           |
| Rec (SASRec)   | Learning rate        | 1e-3          |
| Rec (GRU4Rec)  | Learning rate        | 1e-4          |
| Rec            | Weight decay         | 1e-4          |
| Rec            | Dropout              | 0.3           |
| Rec            | Max epochs           | 500           |
| Rec            | Early stop patience  | 20 epochs     |
| Rec            | Random seeds         | 42, 123, 2024 |
