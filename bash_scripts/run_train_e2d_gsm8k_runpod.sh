#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export DATASET_MODE=full
export MODEL_LENGTH="${MODEL_LENGTH:-768}"
export BLOCK_SIZE="${BLOCK_SIZE:-10}"
export EVAL_BLOCK_SIZE="${EVAL_BLOCK_SIZE:-10}"
export MAX_DURATION="${MAX_DURATION:-30000ba}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000ba}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2500ba}"
export CHECKPOINTS_TO_KEEP="${CHECKPOINTS_TO_KEEP:-1}"
export ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-true}"
export ENABLE_EMA="${ENABLE_EMA:-true}"
export AUTORESUME="${AUTORESUME:-true}"
export WARMUP="${WARMUP:-100ba}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT:-/workspace/outputs}/training}"
export RUN_NAME="${RUN_NAME:-e2d-gsm8k-full-${TIMESTAMP}}"

exec bash "${SCRIPT_DIR}/run_train_e2d_gsm8k_small.sh"
