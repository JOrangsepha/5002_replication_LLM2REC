# CSFT Reproduction README

## Overview

This folder contains the source code for the **CSFT (Collaborative Supervised Fine-Tuning)** stage of our LLM2Rec reproduction. It can be run independently from the other stages.

## How to Compile

This part of the project is implemented in Python, so there is **no compilation step**.

Install the required packages first:

```bash
pip install -r ../../requirements.txt
```

If you only want the minimum core packages for this stage, the main dependencies are PyTorch, Transformers, Pandas, and optionally PEFT for LoRA.

## How to Execute

Run the script from `repro_csft_code/`:

```bash
cd repro_csft_code
bash repro_csft/run_csft_repro.sh strict
```

Available modes:

- `original`
- `instruction`
- `strict`
- `strict_small`

Examples:

```bash
bash repro_csft/run_csft_repro.sh
bash repro_csft/run_csft_repro.sh original
bash repro_csft/run_csft_repro.sh instruction
bash repro_csft/run_csft_repro.sh strict
bash repro_csft/run_csft_repro.sh strict_small
```

You can also run the training file directly:

```bash
python repro_csft/train_csft.py repro_csft/csft_config.strict_1to1.json
```

## Description of Each Source File

- `train_csft.py`: Main CSFT training script using the Hugging Face training pipeline.
- `runtime.py`: Helper functions for loading configuration, model setup, and training runtime behavior.
- `csft_dataset.py`: Builds the CSFT training dataset from the input data files.
- `create_small_csft_dataset.py`: Creates a smaller dataset for debugging or fast local testing.
- `run_csft_repro.sh`: Shell script that selects a configuration mode and launches training.
- `csft_config.example.json`: Default example configuration file.
- `csft_config.instruction_prompt.json`: Configuration with an instruction-style prompt format.
- `csft_config.strict_1to1.json`: Strict reproduction configuration intended to stay close to the original implementation.
- `csft_config.strict_small.json`: Strict configuration for a reduced-size quick test.
- `__init__.py`: Package initialization file.
- `README.md`: Documentation for this CSFT folder.

## Example to Show How to Run the Program

### Standard strict run

```bash
cd repro_csft_code
bash repro_csft/run_csft_repro.sh strict
```

### Quick small-scale run

```bash
cd repro_csft_code
bash repro_csft/run_csft_repro.sh strict_small
```

In `strict_small` mode, the script automatically creates a mini dataset first if it does not already exist.

## Input Requirement

The default training data should be placed under:

```text
data/AmazonMix-6/5-core/train/AmazonMix-6.csv
data/AmazonMix-6/5-core/valid/AmazonMix-6.csv
```

For quick testing, the mini dataset is expected under:

```text
data/AmazonMix-6-mini/5-core/train/AmazonMix-6-mini.csv
data/AmazonMix-6-mini/5-core/valid/AmazonMix-6-mini.csv
```

## Operating System Tested

- Windows 11 for file checking and project setup.
- Recommended execution environment: Linux or Windows with WSL/Git Bash, because the launcher is a Bash script.

## Additional Notes

- You may need to change `model_name_or_path` in the JSON configuration files to the local path of your pretrained model.
- GPU training is recommended.
- The `strict` mode is the best choice when you want the closest reproduction setting.
