#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_csft_repro.sh
#   bash run_csft_repro.sh original
#   bash rrun_csft_repro.sh instruction
#   bash run_csft_repro.sh strict
#   bash run_csft_repro.sh strict_small
# or
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 repro_csft/train_csft.py <config_path>

MODE="${1:-original}"

case "$MODE" in
  original)
    CONFIG="csft_config.example.json"
    ;;
  instruction)
    CONFIG="csft_config.instruction_prompt.json"
    ;;
  strict)
    CONFIG="csft_config.strict_1to1.json"
    ;;
  strict_small)
    CONFIG="csft_config.strict_small.json"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: original | instruction | strict | strict_small"
    exit 1
    ;;
esac

if [[ "$MODE" == "strict_small" ]]; then
  TRAIN_SMALL="data/AmazonMix-6-mini/5-core/train/AmazonMix-6-mini.csv"
  VALID_SMALL="data/AmazonMix-6-mini/5-core/valid/AmazonMix-6-mini.csv"
  if [[ ! -f "$TRAIN_SMALL" || ! -f "$VALID_SMALL" ]]; then
    echo "[CSFT] mini dataset not found, generating..."
    python create_small_csft_dataset.py
  fi
fi

echo "[CSFT] mode=$MODE config=$CONFIG"
python train_csft.py "$CONFIG"
