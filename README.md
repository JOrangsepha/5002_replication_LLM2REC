# "LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation" (KDD '25)

## Overview

This workspace contains two standalone reproduction packages:

- `repro_csft_code/repro_csft/`: Stage 1 CSFT reproduction
- `repro_iem_code/repro_iem/`: Stage 2 IEM reproduction, split into MNTP and SimCSE

Each package has its own README and runnable shell script. The top-level README is kept as a single entry point for setup, data, and the recommended reproduction flow.

## Setup

Install the shared dependencies first:

```bash
pip install -r requirements.txt
```

If you want to use the original CSFT or IEM scripts exactly as shipped in the repro folders, run them from their package directories as described below.

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

## Reproduction Scripts

### CSFT reproduction

The CSFT package lives in `repro_csft_code/repro_csft/`.

From `repro_csft_code`:

```bash
bash repro_csft/run_csft_repro.sh
```

Available modes:

```bash
bash repro_csft/run_csft_repro.sh original
bash repro_csft/run_csft_repro.sh instruction
bash repro_csft/run_csft_repro.sh strict
bash repro_csft/run_csft_repro.sh strict_small
```

`strict` is the closest 1:1 reproduction mode, while `strict_small` first builds a smaller dataset for quicker local testing.

### IEM reproduction

The IEM package lives in `repro_iem_code/repro_iem/`.

From `repro_iem_code`:

```bash
bash repro_iem/run_iem_repro.sh
```

Available modes:

```bash
bash repro_iem/run_iem_repro.sh strict
bash repro_iem/run_iem_repro.sh mntp
bash repro_iem/run_iem_repro.sh simcse
```

`strict` runs MNTP first, prepares the checkpoint, then runs SimCSE. `mntp` and `simcse` let you run each stage separately.

## Suggested Order

If you want the full reproduction flow, run:

1. CSFT strict reproduction
2. IEM strict reproduction

If you only need one stage, run the corresponding package directly from its folder.

## Key Hyperparameters

The following settings are taken from the paper's Section 4.1.3 and are the most useful reference values when reproducing the experiments:

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

## Package Notes

- `repro_csft_code/repro_csft/README.md` explains the CSFT templates, strict mode, and small-dataset workflow in more detail.
- `repro_iem_code/repro_iem/README.md` explains the MNTP + SimCSE split and the `run_iem_repro.sh` modes.
