#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-strict}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

# Keep package imports stable when torchrun launches scripts under repro_iem/.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${PROJECT_ROOT}"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v py >/dev/null 2>&1; then
  PYTHON_BIN="py -3"
else
  echo "Python interpreter not found."
  exit 1
fi

MNTP_CONFIG="${ROOT_DIR}/iem_config.mntp.strict.json"
SIMCSE_CONFIG="${ROOT_DIR}/iem_config.simcse.strict.json"

if [[ "$MODE" != "strict" && "$MODE" != "mntp" && "$MODE" != "simcse" ]]; then
  echo "Unknown mode: $MODE"
  echo "Use: strict | mntp | simcse"
  exit 1
fi

run_mntp() {
  echo "[IEM] Stage 2A: MNTP"
  torchrun --nproc_per_node=1 --master_port=29501 "${ROOT_DIR}/train_mntp_repro.py" "$MNTP_CONFIG"
}

prepare_simcse_input() {
  ${PYTHON_BIN} "${ROOT_DIR}/prepare_simcse_checkpoint.py" \
    --source_model_dir "$(${PYTHON_BIN} -c "import json; print(json.load(open(r'${MNTP_CONFIG}'))['model_name_or_path'])")" \
    --destination_dir "$(${PYTHON_BIN} -c "import json; import os; cfg=json.load(open(r'${MNTP_CONFIG}')); print(os.path.join(cfg['output_dir'], 'checkpoint-100'))")"
}

run_simcse() {
  echo "[IEM] Stage 2B: SimCSE"
  torchrun --nproc_per_node=1 --master_port=29502 "${ROOT_DIR}/train_simcse_repro.py" "$SIMCSE_CONFIG"
}

case "$MODE" in
  strict)
    run_mntp
    prepare_simcse_input
    run_simcse
    ;;
  mntp)
    run_mntp
    ;;
  simcse)
    prepare_simcse_input
    run_simcse
    ;;
esac
