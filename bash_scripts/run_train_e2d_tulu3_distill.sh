#!/bin/bash

set -euo pipefail

# Tulu 3 has 939,343 chat conversations. The shared driver discards each
# reference final response and generates a replacement with Qwen3-1.7B.
export DISTILL_SOURCE_NAME="${DISTILL_SOURCE_NAME:-allenai/tulu-3-sft-mixture}"
export DISTILL_SOURCE_SPLITS="${DISTILL_SOURCE_SPLITS:-train}"
export DISTILL_SOURCE_FORMAT="${DISTILL_SOURCE_FORMAT:-messages}"
export DISTILL_SOURCE_LABEL="${DISTILL_SOURCE_LABEL:-Tulu 3}"
export DISTILL_SOURCE_SLUG="${DISTILL_SOURCE_SLUG:-tulu3}"
export DISTILL_RUN_PREFIX="${DISTILL_RUN_PREFIX:-tulu3_distill}"
export DISTILL_RUN_TAG="${DISTILL_RUN_TAG:-e2d_tulu3_distill}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_train_e2d_ultrachat.sh" "$@"
