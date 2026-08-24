#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=../setup_env.sh
source setup_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
MODEL_LENGTH="${MODEL_LENGTH:-768}"
MAX_DURATION="${MAX_DURATION:-3ep}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
EVAL_SUBSET_NUM_BATCHES="${EVAL_SUBSET_NUM_BATCHES:--1}"
AUTORESUME="${AUTORESUME:-false}"
LR="${LR:-2e-5}"
WARMUP="${WARMUP:-25ba}"
PROGRESS_EVAL_SAMPLES="${PROGRESS_EVAL_SAMPLES:-16}"
PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-qwen05b-full-sft}"
WANDB_MODE="${WANDB_MODE:-offline}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-gsm8k-qwen25-05b-full-sft-${TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT}/ar-full-sft-qwen05b}"
export WANDB_PROJECT WANDB_MODE

if (( GLOBAL_BATCH_SIZE % (NUM_DEVICES * MICRO_BATCH_SIZE) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NUM_DEVICES * MICRO_BATCH_SIZE." >&2
  exit 2
fi
GRAD_ACCUM="${GRAD_ACCUM:-$((GLOBAL_BATCH_SIZE / NUM_DEVICES / MICRO_BATCH_SIZE))}"

echo "Starting full-parameter Qwen 0.5B GSM8K SFT"
echo "  model:        ${MODEL_NAME}"
echo "  GPUs:         ${CUDA_VISIBLE_DEVICES}"
echo "  global batch: ${GLOBAL_BATCH_SIZE}"
echo "  microbatch:   ${MICRO_BATCH_SIZE}"
echo "  grad accum:   ${GRAD_ACCUM}"
echo "  duration:     ${MAX_DURATION}"
echo "  outputs:      ${OUTPUT_ROOT}/${RUN_NAME}"

uv run composer -n "${NUM_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME}" \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.length="${MODEL_LENGTH}" \
  training.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  training.grad_accum="${GRAD_ACCUM}" \
  training.autoresume="${AUTORESUME}" \
  composer.optimizer.lr="${LR}" \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup="${WARMUP}" \
  composer.lr_scheduler.alpha_f=0.1 \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.eval_subset_num_batches="${EVAL_SUBSET_NUM_BATCHES}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval=1ba \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.callbacks.hf_compatible_checkpointing.save_to_hub=false \
  composer.callbacks.hf_compatible_checkpointing.hub_repo_id=null \
  +composer/callbacks@composer.callbacks=gsm8k_progress_evaluator \
  composer.callbacks.gsm8k_progress_evaluator.num_samples="${PROGRESS_EVAL_SAMPLES}" \
  composer.callbacks.gsm8k_progress_evaluator.max_new_tokens="${PROGRESS_EVAL_TOKENS}" \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  ~composer.algorithms.ema
