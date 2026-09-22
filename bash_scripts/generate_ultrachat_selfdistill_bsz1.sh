#!/usr/bin/env bash
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

python scripts/generate_ultrachat_selfdistill.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --data-root "${DISTILL_DATA_ROOT}" \
  --max-length "${MAX_SEQ_LEN}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --source-dataset "${DISTILL_SOURCE_NAME:-HuggingFaceH4/ultrachat_200k}" \
  --source-splits "${SOURCE_SPLITS[@]}" \
  --max-samples "${DISTILL_MAX_SAMPLES:-0}" \
  --eval-ratio "${EVAL_RATIO:-0.01}" \
  --seed "${SPLIT_SEED:-42}" \
  --progress-every "${PROGRESS_EVERY:-25}" \
  "$@"
