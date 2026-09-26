#!/usr/bin/env bash
# One model replica per visible GPU (1-8), with configurable per-device batches.
# Example:
#   export CUDA_VISIBLE_DEVICES=0,2,4,6
#   PER_DEVICE_BATCH_SIZE=4 bash bash_scripts/generate_ultrachat_selfdistill_advanced.sh
# Resume with the same settings. Shards/logs: ${DISTILL_DATA_ROOT}.shards/.
# DISTILL_MAX_SAMPLES is global. Batched outputs require VERIFY=false.
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
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/ultrachat_qwen3_1p7b_selfdistill_advanced_bsz${PER_DEVICE_BATCH_SIZE}_maxlen${MAX_SEQ_LEN}}
read -r -a SOURCE_SPLITS <<< "${DISTILL_SOURCE_SPLITS:-train_sft train_gen}"

exec "${PYTHON:-python}" scripts/generate_ultrachat_selfdistill.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --data-root "${DISTILL_DATA_ROOT}" \
  --max-length "${MAX_SEQ_LEN}" \
  --device cuda \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --source-dataset "${DISTILL_SOURCE_NAME:-HuggingFaceH4/ultrachat_200k}" \
  --source-splits "${SOURCE_SPLITS[@]}" \
  --max-samples "${DISTILL_MAX_SAMPLES:-0}" \
  --eval-ratio "${EVAL_RATIO:-0.01}" \
  --seed "${SPLIT_SEED:-42}" \
  --progress-every "${PROGRESS_EVERY:-25}" \
  --batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --num-shards 0 \
  "$@"
