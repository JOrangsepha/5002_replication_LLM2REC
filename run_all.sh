#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# One-command end-to-end reproduction pipeline
# ------------------------------------------------------------
# Default flow:
#   1) Install dependencies (optional)
#   2) Run CSFT stage
#   3) Run IEM stage (MNTP -> prepare checkpoint -> SimCSE)
#   4) Run step3 embedding extraction + downstream evaluation
#
# Quick start:
#   bash run_all.sh
#
# Common overrides:
#   INSTALL_DEPS=1 bash run_all.sh
#   CSFT_MODE=strict_small bash run_all.sh
#   IEM_MODE=strict bash run_all.sh
#   STEP3_DATASETS="Games_5core 0;Movies_5core 1" bash run_all.sh
#   STEP3_MODELS="IEM_1000 ./repro_iem_code/repro_iem/output/.../checkpoint-100" bash run_all.sh
# ------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

CSFT_DIR="${PROJECT_ROOT}/repro_csft_code/repro_csft"
IEM_DIR="${PROJECT_ROOT}/repro_iem_code/repro_iem"
STEP3_DIR="${PROJECT_ROOT}/step3"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"

CSFT_MODE="${CSFT_MODE:-strict}"
IEM_MODE="${IEM_MODE:-strict}"

# step3 evaluation inputs
STEP3_MODELS="${STEP3_MODELS:-IEM_500 ./step3/checkpoints/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-500;IEM_1000 ./step3/checkpoints/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-1000}"
STEP3_DATASETS="${STEP3_DATASETS:-Games_5core 0;Arts_5core 1;Movies_5core 0;Sports_5core 1;Baby_5core 0;Goodreads 1}"
REC_MODEL="${REC_MODEL:-SASRec}"
LR="${LR:-1.0e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
DROPOUT="${DROPOUT:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SKIP_STEP3="${SKIP_STEP3:-0}"

log() { printf '[run_all] %s\n' "$*"; }
fail() { printf '[run_all][ERROR] %s\n' "$*" >&2; exit 1; }

require_file() { [[ -f "$1" ]] || fail "Required file not found: $1"; }
require_dir() { [[ -d "$1" ]] || fail "Required directory not found: $1"; }

parse_entries() {
  local raw="$1"
  local -n out_arr="$2"
  IFS=';' read -r -a out_arr <<< "$raw"
}

check_prerequisites() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"
  require_file "$REQ_FILE"
  require_dir "$PROJECT_ROOT/data"
  require_file "$CSFT_DIR/train_csft.py"
  require_file "$CSFT_DIR/run_csft_repro.sh"
  require_file "$IEM_DIR/train_mntp_repro.py"
  require_file "$IEM_DIR/train_simcse_repro.py"
  require_file "$IEM_DIR/prepare_simcse_checkpoint.py"
  require_file "$IEM_DIR/run_iem_repro.sh"
  require_file "$STEP3_DIR/get_embedding.py"
  require_file "$STEP3_DIR/evaluation.py"
}

maybe_install_deps() {
  if [[ "$INSTALL_DEPS" == "1" ]]; then
    log "Installing dependencies from requirements.txt"
    "$PYTHON_BIN" -m pip install -r "$REQ_FILE"
  else
    log "Skip dependency installation (INSTALL_DEPS=$INSTALL_DEPS)"
  fi
}

run_csft_stage() {
  log "================ CSFT stage ================"
  (cd "$CSFT_DIR" && bash ./run_csft_repro.sh "$CSFT_MODE")
}

run_iem_stage() {
  log "================ IEM stage ================"
  (cd "$IEM_DIR" && bash ./run_iem_repro.sh "$IEM_MODE")
}

run_step3_stage() {
  if [[ "$SKIP_STEP3" == "1" ]]; then
    log "Skipping step3 stage (SKIP_STEP3=$SKIP_STEP3)"
    return
  fi

  log "================ step3 stage ================"
  local model_entries=()
  local dataset_entries=()
  parse_entries "$STEP3_MODELS" model_entries
  parse_entries "$STEP3_DATASETS" dataset_entries

  local model_setting dataset_setting dataset cuda_device save_info model_path embs
  for dataset_setting in "${dataset_entries[@]}"; do
    dataset="$(awk '{print $1}' <<< "$dataset_setting")"
    cuda_device="$(awk '{print $2}' <<< "$dataset_setting")"
    [[ -n "$dataset" && -n "$cuda_device" ]] || fail "Invalid STEP3_DATASETS entry: $dataset_setting"

    for model_setting in "${model_entries[@]}"; do
      save_info="$(awk '{print $1}' <<< "$model_setting")"
      model_path="$(awk '{print $2}' <<< "$model_setting")"
      [[ -n "$save_info" && -n "$model_path" ]] || fail "Invalid STEP3_MODELS entry: $model_setting"
      [[ -d "$PROJECT_ROOT/${model_path#./}" || -d "$model_path" ]] || fail "Model path not found: $model_path"

      log "Step3 run | dataset=$dataset | checkpoint=$save_info | gpu=$cuda_device"
      mkdir -p "$STEP3_DIR/item_info/$dataset"
      CUDA_VISIBLE_DEVICES="$cuda_device" "$PYTHON_BIN" "$STEP3_DIR/get_embedding.py" \
        --dataset="$dataset" \
        --model_path="$model_path" \
        --save_info="$save_info"

      embs="$STEP3_DIR/item_info/$dataset/${save_info}_title_item_embs.npy"
      [[ -f "$embs" ]] || fail "Embedding file not found: $embs"

      WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$cuda_device" "$PYTHON_BIN" "$STEP3_DIR/evaluation.py" \
        --model="$REC_MODEL" \
        --dataset="$dataset" \
        --lr="$LR" \
        --weight_decay="$WEIGHT_DECAY" \
        --embedding="$embs" \
        --dropout="$DROPOUT" \
        --batch_size="$BATCH_SIZE"
    done
  done
}

main() {
  check_prerequisites
  maybe_install_deps
  run_csft_stage
  run_iem_stage
  run_step3_stage
  log "All stages finished successfully."
}

main "$@"
