#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-/workspace}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
export MODEL_LENGTH="${MODEL_LENGTH:-768}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export MAX_DURATION="${MAX_DURATION:-3ep}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500ba}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
export PROGRESS_EVAL_SAMPLES="${PROGRESS_EVAL_SAMPLES:-8}"
export PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
export PROGRESS_EVAL_STEPS="${PROGRESS_EVAL_STEPS:-1}"
export CURRICULUM_START="${CURRICULUM_START:-100}"
export CURRICULUM_END="${CURRICULUM_END:-1000}"
export DATASET_MODE=full
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_STORAGE_ROOT}/outputs/embedding-flow-map}"
export RUN_NAME="${RUN_NAME:-gsm8k-embedding-flow-map-${TIMESTAMP}}"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-lora-flow-map}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [ -z "${NUM_DEVICES:-}" ]; then
  NUM_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
fi
export NUM_DEVICES
if [ "${NUM_DEVICES}" -lt 1 ]; then
  echo "No CUDA devices detected." >&2
  exit 2
fi
export GRAD_ACCUM="${GRAD_ACCUM:-$(( (GLOBAL_BATCH_SIZE + NUM_DEVICES - 1) / NUM_DEVICES ))}"

if [ "${RUN_BOOTSTRAP:-true}" = "true" ]; then
  bash "${SCRIPT_DIR}/runpod_bootstrap.sh"
fi

CREDENTIALS_FILE="${E2D_CREDENTIALS_FILE:-${E2D_STORAGE_ROOT}/credentials/e2d.env}"
if [ -f "${CREDENTIALS_FILE}" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${CREDENTIALS_FILE}"
  set +a
fi

echo "Starting joint GSM8K AR + embedding flow-map training: ${RUN_NAME}"
echo "Outputs: ${OUTPUT_ROOT}/${RUN_NAME}"
exec bash "${SCRIPT_DIR}/run_train_embedding_flow_map_gsm8k.sh"
