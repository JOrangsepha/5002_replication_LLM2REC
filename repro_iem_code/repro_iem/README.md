# IEM Reproduction README

## Overview

This folder contains the source code for the **IEM** stage of our reproduction. In this project, IEM is split into two parts:

1. `MNTP`: Further training based on the CSFT checkpoint.
2. `SimCSE`: Contrastive learning for representation refinement.

This directory can be run independently after the required data and checkpoint paths are prepared.

## How to Compile

This module is written in Python, so there is **no compilation step**.

Install dependencies first:

```bash
pip install -r ../../requirements.txt
```

## How to Execute

Run the script from `repro_iem_code/`:

```bash
cd repro_iem_code
bash repro_iem/run_iem_repro.sh strict
```

Supported modes:

- `strict`: Run MNTP, prepare the checkpoint, then run SimCSE.
- `mntp`: Run only the MNTP stage.
- `simcse`: Prepare the checkpoint and run only the SimCSE stage.

Examples:

```bash
bash repro_iem/run_iem_repro.sh strict
bash repro_iem/run_iem_repro.sh mntp
bash repro_iem/run_iem_repro.sh simcse
```

You can also execute the Python entry files directly:

```bash
python repro_iem/train_mntp_repro.py repro_iem/iem_config.mntp.strict.json
python repro_iem/train_simcse_repro.py repro_iem/iem_config.simcse.strict.json
```

## Description of Each Source File

- `train_mntp_repro.py`: Main entry point for the MNTP training stage.
- `train_simcse_repro.py`: Main entry point for the SimCSE training stage.
- `runtime.py`: Shared runtime helper code used across the IEM scripts.
- `mntp_runtime.py`: Runtime components specific to MNTP training.
- `dataset_registry.py`: Maps dataset names to their corresponding dataset classes.
- `prepare_simcse_checkpoint.py`: Copies or prepares files needed before running SimCSE on top of the MNTP output.
- `utils.py`: Shared helper functions used by the IEM scripts.
- `run_iem_repro.sh`: Bash launcher for the strict, MNTP-only, or SimCSE-only pipeline.
- `iem_config.mntp.strict.json`: Configuration for the MNTP stage.
- `iem_config.simcse.strict.json`: Configuration for the SimCSE stage.
- `recdata/base.py`: Base dataset definitions for recommendation-related data loading.
- `recdata/item_pairs.py`: Dataset implementation for item-pair based contrastive learning.
- `recdata/item_titles.py`: Dataset loader for item-title text data.
- `recdata/seqrec.py`: Dataset loader for sequential recommendation data.
- `recdata/__init__.py`: Package initialization file for the dataset submodule.
- `__init__.py`: Package initialization file.
- `README.md`: Documentation for this IEM folder.

## Example to Show How to Run the Program

### Full IEM pipeline

```bash
cd repro_iem_code
bash repro_iem/run_iem_repro.sh strict
```

### MNTP only

```bash
cd repro_iem_code
bash repro_iem/run_iem_repro.sh mntp
```

### SimCSE only

```bash
cd repro_iem_code
bash repro_iem/run_iem_repro.sh simcse
```

## Input Requirement

The default configuration expects:

- A CSFT checkpoint path in `iem_config.mntp.strict.json`
- Data files under the repository `data/` directory
- An MNTP output checkpoint before running SimCSE

Before execution, update the JSON configuration paths if your local folder structure is different.

## Operating System Tested

- Windows 11 for repository management and verification.
- Recommended execution environment: Linux or Windows with WSL/Git Bash because the launcher and training flow depend on Bash and `torchrun`.

## Additional Notes

- The `strict` mode is the recommended mode for complete reproduction.
- If `torchrun` is unavailable, install a correct PyTorch environment first.
- Make sure the CSFT output checkpoint exists before running the full IEM pipeline.
