# LLM2Rec Implementation Project (Group 2)

## Project Overview

This repository contains our reproduction work for **"LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation"**. The codebase is organized into three main parts:

- `repro_csft_code/repro_csft/`: Stage 1 reproduction for CSFT training.
- `repro_iem_code/repro_iem/`: Stage 2 reproduction for IEM training, including MNTP and SimCSE.
- `step3/`: Embedding extraction and downstream sequential recommendation evaluation.

This top-level README is the main entry point for running the whole project. Each source-code subdirectory also has its own README with more detailed instructions.

Project repository: [link](https://github.com/JOrangsepha/5002_replication_LLM2REC.git)

## Directory Structure

```text
5002_replication_LLM2REC/
├─ README.md
├─ requirements.txt
├─ run_all.sh
├─ repro_csft_code/
│  └─ repro_csft/
├─ repro_iem_code/
│  └─ repro_iem/
├─ step3/
└─ data/                       # need to add separately from the link below
```

## How to Compile
To prepare the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Execute

### Option 1: Run the full pipeline

From the project root:

```bash
bash run_all.sh
```

### Option 2: Run each stage separately

CSFT stage:

```bash
cd repro_csft_code
bash repro_csft/run_csft_repro.sh strict
```

IEM stage:

```bash
cd repro_iem_code
bash repro_iem/run_iem_repro.sh strict
```

Step 3 evaluation is executed by `run_all.sh`, or can be run manually with the Python scripts in `step3/`.

## Data Preparation

Download the processed datasets from the official project source and place the `data/` folder at the repository root. Data can be downloaded here: [link](https://drive.google.com/file/d/1GIXWaaaNuUkUtuFy5JTN0OwAQiLGb2z4/view).

Expected examples include:

- `data/AmazonMix-6/`
- `data/Video_Games/`
- `data/Arts_Crafts_and_Sewing/`
- `data/Movies_and_TV/`
- `data/Sports_and_Outdoors/`
- `data/Baby_Products/`
- `data/Goodreads/`

The training and evaluation scripts assume this folder structure already exists.

## Description of Source Files

### Top-level files

- `requirements.txt`: Python package dependencies required by the whole project.
- `run_all.sh`: End-to-end pipeline script that runs CSFT, IEM, and Step 3 in sequence.
- `LLM2RecLargeLanguageModelsArePowerfulEmbeddingModelsForSequentialRecommendation.pdf`: Reference paper used for the reproduction.
- `README.md`: Main project documentation.

### `repro_csft_code/repro_csft/`

- `train_csft.py`: Main training entry point for the CSFT reproduction.
- `runtime.py`: Runtime helpers used during CSFT training.
- `csft_dataset.py`: Dataset construction logic for CSFT input-output pairs.
- `create_small_csft_dataset.py`: Utility script to create a smaller dataset for quick testing.
- `run_csft_repro.sh`: Shell script that launches CSFT in different modes.
- `csft_config.example.json`: Example CSFT configuration.
- `csft_config.instruction_prompt.json`: CSFT configuration using an instruction-style prompt template.
- `csft_config.strict_1to1.json`: Strict reproduction configuration aligned with the original setting.
- `csft_config.strict_small.json`: Small-scale strict configuration for quick local tests.
- `__init__.py`: Package marker file.
- `README.md`: CSFT-specific documentation.

### `repro_iem_code/repro_iem/`

- `train_mntp_repro.py`: Main training script for the MNTP stage.
- `train_simcse_repro.py`: Main training script for the SimCSE stage.
- `runtime.py`: Shared runtime helpers for the IEM pipeline.
- `mntp_runtime.py`: Runtime logic specific to MNTP training.
- `dataset_registry.py`: Registers available datasets for IEM training.
- `prepare_simcse_checkpoint.py`: Prepares the MNTP checkpoint before SimCSE training.
- `utils.py`: Shared utility functions for the IEM code.
- `run_iem_repro.sh`: Shell script to run MNTP, SimCSE, or the strict full IEM pipeline.
- `iem_config.mntp.strict.json`: Strict MNTP configuration file.
- `iem_config.simcse.strict.json`: Strict SimCSE configuration file.
- `recdata/base.py`: Base dataset classes for recommendation data.
- `recdata/item_pairs.py`: Dataset logic for item-pair contrastive training.
- `recdata/item_titles.py`: Dataset loader for item titles.
- `recdata/seqrec.py`: Dataset loader for sequential recommendation data.
- `recdata/__init__.py`: Package marker for the `recdata` module.
- `__init__.py`: Package marker file.
- `README.md`: IEM-specific documentation.

### `step3/`

- `get_embedding.py`: Extracts item embeddings from a trained checkpoint.
- `evaluation.py`: Runs downstream recommendation evaluation using extracted embeddings.
- `seqrec/base.py`: Base classes for the recommendation framework.
- `seqrec/runtime.py`: Runtime helpers for the evaluation stage.
- `seqrec/runner.py`: Main driver utilities for model training and evaluation in Step 3.
- `seqrec/trainer.py`: Training loop for downstream recommender models.
- `seqrec/evaluator.py`: Evaluation metrics and scoring logic.
- `seqrec/utils.py`: Utility functions for Step 3.
- `seqrec/recdata.py`: Recommendation dataset processing utilities.
- `seqrec/modules.py`: Shared neural network modules.
- `seqrec/default.yaml`: Default configuration file for Step 3 models.
- `seqrec/models/Embedding2.py`: Embedding-based model component.
- `seqrec/models/SASRec/_model.py`: SASRec model implementation.
- `seqrec/models/SASRec/config.yaml`: SASRec configuration.
- `seqrec/models/GRU4Rec/_model.py`: GRU4Rec model implementation.
- `seqrec/models/GRU4Rec/config.yaml`: GRU4Rec configuration.
- `seqrec/models/__init__.py`: Model package marker file.

## Example: How to Run the Program

### Full pipeline example

```bash
pip install -r requirements.txt
bash run_all.sh
```

### Quick CSFT test example

```bash
cd repro_csft_code
bash repro_csft/run_csft_repro.sh strict_small
```

### IEM strict example

```bash
cd repro_iem_code
bash repro_iem/run_iem_repro.sh strict
```

## Operating System Tested

- Windows 11 with PowerShell for repository setup and documentation checks.
- The training scripts are written as Bash scripts, so the recommended execution environment is Linux, macOS, or Windows with WSL/Git Bash.

## Additional Notes

- A GPU environment is strongly recommended for the full training pipeline.
- Before running the scripts, you may need to update model paths in the JSON configuration files to match your local machine.
- The `data/` directory is required but is not included in the repository because the file is too big.
- The subdirectory READMEs provide more detailed instructions for the CSFT and IEM stages.
