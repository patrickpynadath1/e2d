#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export FLOW_MODEL_CONFIG=riemannian_latent_flow_e2d
export DATASET_MODE=small
export TRAIN_SAMPLES="${TRAIN_SAMPLES:-50}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
export NUM_DEVICES="${NUM_DEVICES:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MODEL_LENGTH="${MODEL_LENGTH:-256}"
export BLOCK_SIZE="${BLOCK_SIZE:-4}"
export EVAL_BLOCK_SIZE="${EVAL_BLOCK_SIZE:-4}"
export INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
export MAX_DURATION="${MAX_DURATION:-200ba}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-25ba}"
export EVAL_BATCHES="${EVAL_BATCHES:-4}"
export SPEC_EVAL_MAX_NEW_TOKENS="${SPEC_EVAL_MAX_NEW_TOKENS:-32}"
export SPEC_EVAL_NUM_PROMPTS="${SPEC_EVAL_NUM_PROMPTS:-2}"
export LR="${LR:-1e-4}"
export WARMUP="${WARMUP:-10ba}"
export ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-false}"
export AUTORESUME="${AUTORESUME:-false}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT:-/workspace/outputs}/overfit}"
export RUN_NAME="${RUN_NAME:-riemannian-flow-gsm8k-overfit50-${TIMESTAMP}}"

exec bash "${SCRIPT_DIR}/run_train_latent_flow_gsm8k_small.sh"
