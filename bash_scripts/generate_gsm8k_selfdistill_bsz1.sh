#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
# setup_env.sh expects unset conda variables and uses bash activation hooks.
set +u
source setup_env.sh
set -u

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/gsm8k_qwen3_1p7b_selfdistill_bsz1_maxlen${MAX_SEQ_LEN}}

python scripts/generate_gsm8k_selfdistill.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --data-root "${DISTILL_DATA_ROOT}" \
  --max-length "${MAX_SEQ_LEN}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --source-dataset "${GSM8K_DATASET_PATH:-openai/gsm8k}" \
  --source-config "${GSM8K_CONFIG_NAME:-main}" \
  --source-split "${GSM8K_SPLIT:-train}" \
  --max-samples "${DISTILL_MAX_SAMPLES:-0}" \
  --eval-ratio "${EVAL_RATIO:-0.02}" \
  --seed "${SPLIT_SEED:-42}" \
  --progress-every "${PROGRESS_EVERY:-25}" \
  "$@"
