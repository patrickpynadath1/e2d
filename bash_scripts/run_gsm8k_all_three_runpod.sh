#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${PIPELINE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-/workspace}"
PIPELINE_OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_STORAGE_ROOT}/outputs/gsm8k-all-three-${TIMESTAMP}}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
export MODEL_LENGTH="${MODEL_LENGTH:-768}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500ba}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
export PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-lora-flow-map}"
export WANDB_MODE="${WANDB_MODE:-online}"

STANDARD_MAX_DURATION="${STANDARD_MAX_DURATION:-3ep}"
MATCHED_MAX_DURATION="${MATCHED_MAX_DURATION:-3ep}"
FLOW_MAX_DURATION="${FLOW_MAX_DURATION:-3ep}"
STANDARD_GRAD_ACCUM="${STANDARD_GRAD_ACCUM:-${GRAD_ACCUM:-1}}"
MATCHED_GRAD_ACCUM="${MATCHED_GRAD_ACCUM:-${GRAD_ACCUM:-1}}"
FLOW_GRAD_ACCUM="${FLOW_GRAD_ACCUM:-${GRAD_ACCUM:-2}}"
STANDARD_PROGRESS_EVAL_SAMPLES="${STANDARD_PROGRESS_EVAL_SAMPLES:-16}"
MATCHED_PROGRESS_EVAL_SAMPLES="${MATCHED_PROGRESS_EVAL_SAMPLES:-16}"
FLOW_PROGRESS_EVAL_SAMPLES="${FLOW_PROGRESS_EVAL_SAMPLES:-8}"

if [ -z "${NUM_DEVICES:-}" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; set NUM_DEVICES explicitly." >&2
    exit 2
  fi
  NUM_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
fi
export NUM_DEVICES
if [ "${NUM_DEVICES}" -lt 1 ]; then
  echo "No CUDA devices detected." >&2
  exit 2
fi

mkdir -p \
  "${PIPELINE_OUTPUT_ROOT}/standard" \
  "${PIPELINE_OUTPUT_ROOT}/matched" \
  "${PIPELINE_OUTPUT_ROOT}/flow"

echo "GSM8K three-stage comparison"
echo "  standard PEFT LoRA: ${STANDARD_MAX_DURATION}, accum ${STANDARD_GRAD_ACCUM}"
echo "  matched dual LoRA:  ${MATCHED_MAX_DURATION}, accum ${MATCHED_GRAD_ACCUM}"
echo "  embedding flow map: ${FLOW_MAX_DURATION}, accum ${FLOW_GRAD_ACCUM}"
echo "  outputs:            ${PIPELINE_OUTPUT_ROOT}"

if [ "${RUN_BOOTSTRAP:-true}" = "true" ]; then
  bash "${SCRIPT_DIR}/runpod_bootstrap.sh"
else
  # shellcheck source=../setup_env.sh
  source "${REPO_ROOT}/setup_env.sh"
fi

echo "[1/3] Starting standard PEFT LoRA SFT baseline."
(
  export RUN_BOOTSTRAP=false
  export GRAD_ACCUM="${STANDARD_GRAD_ACCUM}"
  export MAX_DURATION="${STANDARD_MAX_DURATION}"
  export PROGRESS_EVAL_SAMPLES="${STANDARD_PROGRESS_EVAL_SAMPLES}"
  export OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT}/standard"
  export RUN_NAME="${STANDARD_RUN_NAME:-gsm8k-ar-lora-${TIMESTAMP}}"
  bash "${SCRIPT_DIR}/run_train_ar_lora_gsm8k_runpod.sh"
)

echo "[2/3] Starting matched shared + SFT LoRA baseline."
(
  export RUN_BOOTSTRAP=false
  export GRAD_ACCUM="${MATCHED_GRAD_ACCUM}"
  export MAX_DURATION="${MATCHED_MAX_DURATION}"
  export PROGRESS_EVAL_SAMPLES="${MATCHED_PROGRESS_EVAL_SAMPLES}"
  export OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT}/matched"
  export RUN_NAME="${MATCHED_RUN_NAME:-gsm8k-ar-shared-sft-lora-${TIMESTAMP}}"
  bash "${SCRIPT_DIR}/run_train_ar_shared_sft_lora_gsm8k_runpod.sh"
)

echo "[3/3] Starting embedding flow-map training."
(
  export RUN_BOOTSTRAP=false
  export GRAD_ACCUM="${FLOW_GRAD_ACCUM}"
  export MAX_DURATION="${FLOW_MAX_DURATION}"
  export PROGRESS_EVAL_SAMPLES="${FLOW_PROGRESS_EVAL_SAMPLES}"
  export OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT}/flow"
  export RUN_NAME="${FLOW_RUN_NAME:-gsm8k-embedding-flow-map-${TIMESTAMP}}"
  bash "${SCRIPT_DIR}/run_train_embedding_flow_map_gsm8k_runpod.sh"
)

echo "All three runs completed. Results are under ${PIPELINE_OUTPUT_ROOT}."
