#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${PIPELINE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-/workspace}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_STORAGE_ROOT}/outputs/gsm8k-lora-flow-${TIMESTAMP}}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
export MODEL_LENGTH="${MODEL_LENGTH:-768}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
REQUESTED_GRAD_ACCUM="${GRAD_ACCUM:-}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500ba}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
export DATASET_MODE=full
export PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-lora-flow-map}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-true}"
RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_FLOW="${RUN_FLOW:-true}"
BASELINE_VARIANT="${BASELINE_VARIANT:-standard}"
BASELINE_MAX_DURATION="${BASELINE_MAX_DURATION:-3ep}"
FLOW_MAX_DURATION="${FLOW_MAX_DURATION:-3ep}"
BASELINE_PROGRESS_EVAL_SAMPLES="${BASELINE_PROGRESS_EVAL_SAMPLES:-16}"
FLOW_PROGRESS_EVAL_SAMPLES="${FLOW_PROGRESS_EVAL_SAMPLES:-8}"
if [ "${BASELINE_VARIANT}" = "matched_dual" ]; then
  BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-gsm8k-ar-shared-sft-lora-${TIMESTAMP}}"
else
  BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-gsm8k-ar-lora-${TIMESTAMP}}"
fi
FLOW_RUN_NAME="${FLOW_RUN_NAME:-gsm8k-embedding-flow-map-${TIMESTAMP}}"

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
if [ -n "${REQUESTED_GRAD_ACCUM}" ]; then
  export GRAD_ACCUM="${REQUESTED_GRAD_ACCUM}"
else
  # Default to one training example per device per microbatch, which is much
  # safer for the doubled flow sequence than a per-device batch of eight.
  export GRAD_ACCUM="$(( (GLOBAL_BATCH_SIZE + NUM_DEVICES - 1) / NUM_DEVICES ))"
fi

mkdir -p "${OUTPUT_ROOT}/baseline" "${OUTPUT_ROOT}/flow"

echo "GSM8K LoRA/flow pipeline"
echo "  model:          ${MODEL_NAME}"
echo "  devices:        ${NUM_DEVICES}"
echo "  global batch:   ${GLOBAL_BATCH_SIZE}"
echo "  grad accum:     ${GRAD_ACCUM}"
echo "  sequence length:${MODEL_LENGTH}"
echo "  baseline:       ${BASELINE_RUN_NAME} (${BASELINE_MAX_DURATION})"
echo "  baseline type:  ${BASELINE_VARIANT}"
echo "  flow:           ${FLOW_RUN_NAME} (${FLOW_MAX_DURATION})"
echo "  outputs:        ${OUTPUT_ROOT}"

if [ "${RUN_BOOTSTRAP}" = "true" ]; then
  echo "[setup] Synchronizing and validating the Runpod environment."
  bash "${SCRIPT_DIR}/runpod_bootstrap.sh"
else
  # shellcheck source=../setup_env.sh
  source "${REPO_ROOT}/setup_env.sh"
fi

# runpod_bootstrap loads credentials in its subprocess; load them again for the
# two training subprocesses launched below.
CREDENTIALS_FILE="${E2D_CREDENTIALS_FILE:-${E2D_STORAGE_ROOT}/credentials/e2d.env}"
if [ -f "${CREDENTIALS_FILE}" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${CREDENTIALS_FILE}"
  set +a
fi

if [ "${RUN_BASELINE}" = "true" ]; then
  if [ "${BASELINE_VARIANT}" = "standard" ]; then
    BASELINE_SCRIPT="${SCRIPT_DIR}/run_train_ar_lora_gsm8k.sh"
    echo "[1/2] Starting standard causal SFT + PEFT LoRA baseline."
  elif [ "${BASELINE_VARIANT}" = "matched_dual" ]; then
    BASELINE_SCRIPT="${SCRIPT_DIR}/run_train_ar_shared_sft_lora_gsm8k.sh"
    echo "[1/2] Starting matched shared + SFT LoRA causal baseline."
  else
    echo "BASELINE_VARIANT must be 'standard' or 'matched_dual'." >&2
    exit 2
  fi
  (
    export MAX_DURATION="${BASELINE_MAX_DURATION}"
    export PROGRESS_EVAL_SAMPLES="${BASELINE_PROGRESS_EVAL_SAMPLES}"
    export RUN_NAME="${BASELINE_RUN_NAME}"
    export OUTPUT_ROOT="${OUTPUT_ROOT}/baseline"
    bash "${BASELINE_SCRIPT}"
  )
  echo "[1/2] Baseline completed: ${OUTPUT_ROOT}/baseline/${BASELINE_RUN_NAME}"
else
  echo "[1/2] Baseline skipped (RUN_BASELINE=${RUN_BASELINE})."
fi

if [ "${RUN_FLOW}" = "true" ]; then
  echo "[2/2] Starting joint AR + embedding flow-map training."
  (
    export MAX_DURATION="${FLOW_MAX_DURATION}"
    export PROGRESS_EVAL_SAMPLES="${FLOW_PROGRESS_EVAL_SAMPLES}"
    export RUN_NAME="${FLOW_RUN_NAME}"
    export OUTPUT_ROOT="${OUTPUT_ROOT}/flow"
    bash "${SCRIPT_DIR}/run_train_embedding_flow_map_gsm8k.sh"
  )
  echo "[2/2] Flow-map run completed: ${OUTPUT_ROOT}/flow/${FLOW_RUN_NAME}"
else
  echo "[2/2] Flow-map run skipped (RUN_FLOW=${RUN_FLOW})."
fi

echo "Pipeline complete. Results are under ${OUTPUT_ROOT}."
