#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-/workspace}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
export MODEL_LENGTH="${MODEL_LENGTH:-768}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export MAX_DURATION="${MAX_DURATION:-3ep}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500ba}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
export PROGRESS_EVAL_SAMPLES="${PROGRESS_EVAL_SAMPLES:-16}"
export PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_STORAGE_ROOT}/outputs/ar-shared-sft-lora-baseline}"
export RUN_NAME="${RUN_NAME:-gsm8k-ar-shared-sft-lora-${TIMESTAMP}}"
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

echo "Starting matched GSM8K shared + SFT LoRA baseline: ${RUN_NAME}"
echo "Outputs: ${OUTPUT_ROOT}/${RUN_NAME}"
exec bash "${SCRIPT_DIR}/run_train_ar_shared_sft_lora_gsm8k.sh"
