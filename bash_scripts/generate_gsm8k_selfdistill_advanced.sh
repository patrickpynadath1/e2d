#!/usr/bin/env bash
# One model replica per visible GPU (1-8); configurable batches within each GPU.
# Example:
#   export CUDA_VISIBLE_DEVICES=0,2,4,6
#   PER_DEVICE_BATCH_SIZE=4 bash bash_scripts/generate_gsm8k_selfdistill_advanced.sh
# Rerun with the same settings to resume. Multi-GPU shards/logs are kept in
# ${DISTILL_DATA_ROOT}.shards/. DISTILL_MAX_SAMPLES is a global usable-row limit.
# Batched outputs require VERIFY=false with the current batch-one verifier.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
set +u
source setup_env.sh
set -u

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
if [[ ! "${PER_DEVICE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_DEVICE_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/gsm8k_qwen3_1p7b_selfdistill_advanced_bsz${PER_DEVICE_BATCH_SIZE}_maxlen${MAX_SEQ_LEN}}

exec "${PYTHON:-python}" scripts/generate_gsm8k_selfdistill.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --data-root "${DISTILL_DATA_ROOT}" \
  --max-length "${MAX_SEQ_LEN}" \
  --device cuda \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --source-dataset "${GSM8K_DATASET_PATH:-openai/gsm8k}" \
  --source-config "${GSM8K_CONFIG_NAME:-main}" \
  --source-split "${GSM8K_SPLIT:-train}" \
  --max-samples "${DISTILL_MAX_SAMPLES:-0}" \
  --eval-ratio "${EVAL_RATIO:-0.02}" \
  --seed "${SPLIT_SEED:-42}" \
  --progress-every "${PROGRESS_EVERY:-25}" \
  --batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --num-shards 0 \
  "$@"
