#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export DATASET_MODE=full
export NUM_DEVICES="${NUM_DEVICES:-2}"
export MODEL_LENGTH="${MODEL_LENGTH:-768}"
export BLOCK_SIZE="${BLOCK_SIZE:-10}"
export EVAL_BLOCK_SIZE="${EVAL_BLOCK_SIZE:-10}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export GRAD_ACCUM="${GRAD_ACCUM:-64}"
export MAX_DURATION="${MAX_DURATION:-500ba}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100ba}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500ba}"
export ENABLE_SPEC_EVAL="${ENABLE_SPEC_EVAL:-true}"
export SPEC_EVAL_MAX_NEW_TOKENS="${SPEC_EVAL_MAX_NEW_TOKENS:-64}"
export SPEC_EVAL_NUM_PROMPTS="${SPEC_EVAL_NUM_PROMPTS:-8}"
export CHECKPOINTS_TO_KEEP="${CHECKPOINTS_TO_KEEP:-1}"
export ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-true}"
export ENABLE_EMA="${ENABLE_EMA:-true}"
export AUTORESUME="${AUTORESUME:-true}"
export WARMUP="${WARMUP:-100ba}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT:-/workspace/outputs}/training}"
export RUN_NAME="${RUN_NAME:-e2d-gsm8k-full-${TIMESTAMP}}"

exec bash "${SCRIPT_DIR}/run_train_e2d_gsm8k_small.sh"
