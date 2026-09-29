#!/usr/bin/env bash
# Generate with vLLM: one model replica per visible GPU (1-8).
# Example:
#   CUDA_VISIBLE_DEVICES=0,2,4,6 PER_DEVICE_BATCH_SIZE=32 \
#     bash bash_scripts/generate_tulu3_selfdistill_advanced.sh
# Rerun with the same settings to resume. Worker logs: ${DISTILL_DATA_ROOT}.shards/.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
set +u
source setup_env.sh
set -u

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-${BATCH_GEN_SIZE:-32}}
if [[ ! "${PER_DEVICE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_DEVICE_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi
DISTILL_FILE_STEM=${DISTILL_FILE_STEM:-tulu3_qwen3_1p7b_thinking}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/tulu3_qwen3_1p7b_selfdistill_advanced_bsz${PER_DEVICE_BATCH_SIZE}_maxlen${MAX_SEQ_LEN}}
read -r -a SOURCE_SPLITS <<< "${DISTILL_SOURCE_SPLITS:-train}"

exec "${PYTHON:-python}" scripts/generate_tulu3_selfdistill.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --data-root "${DISTILL_DATA_ROOT}" \
  --jsonl-stem "${DISTILL_FILE_STEM}" \
  --max-length "${MAX_SEQ_LEN}" \
  --gen-max-length "${GEN_MAX_SEQ_LEN:-4096}" \
  --prompt-max-length "${PROMPT_MAX_SEQ_LEN:-1024}" \
  --device cuda \
  --dtype "${DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --source-dataset "${DISTILL_SOURCE_NAME:-allenai/tulu-3-sft-mixture}" \
  --source-splits "${SOURCE_SPLITS[@]}" \
  --max-samples "${DISTILL_MAX_SAMPLES:-0}" \
  --eval-ratio "${EVAL_RATIO:-0.01}" \
  --seed "${SPLIT_SEED:-42}" \
  --progress-every-batches "${PROGRESS_EVERY_BATCHES:-1}" \
  --empty-cache-every-batches "${EMPTY_CACHE_EVERY_BATCHES:-20}" \
  --batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --num-shards 0 \
  "$@"
