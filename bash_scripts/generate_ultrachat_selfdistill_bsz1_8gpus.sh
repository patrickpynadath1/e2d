#!/usr/bin/env bash
# One model replica per visible GPU, always generating one unpadded prompt.
# Usage (from bash_scripts/):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash generate_ultrachat_selfdistill_bsz1_8gpus.sh
# Final files and defaults match generate_ultrachat_selfdistill_bsz1.sh.
# Resumable shards and worker logs live in ${DISTILL_DATA_ROOT}.shards/.
# Rerun the same command to resume. When switching from an unfinished
# single-GPU run, set DISTILL_DATA_ROOT to a new directory.
# DISTILL_MAX_SAMPLES is a global limit across all eight workers.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
set +u
source setup_env.sh
set -u

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/ultrachat_qwen3_1p7b_selfdistill_bsz1_maxlen${MAX_SEQ_LEN}}
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
  "$@" \
  --num-shards 8
