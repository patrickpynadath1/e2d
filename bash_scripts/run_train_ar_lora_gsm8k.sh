#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=../setup_env.sh
source setup_env.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MODEL_LENGTH="${MODEL_LENGTH:-768}"
MAX_DURATION="${MAX_DURATION:-3ep}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-100ba}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
PROGRESS_EVAL_SAMPLES="${PROGRESS_EVAL_SAMPLES:-16}"
PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
RUN_NAME="${RUN_NAME:-gsm8k-ar-lora-r${LORA_R}-${MAX_DURATION}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT}/ar-lora-baseline}"
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-embedding-flow-map}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_MODE

uv run composer -n "${NUM_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME}" \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm_lora \
  model.config.length="${MODEL_LENGTH}" \
  model.config.backbone_config.lora_r="${LORA_R}" \
  model.config.backbone_config.lora_alpha="${LORA_ALPHA}" \
  model.config.backbone_config.lora_dropout="${LORA_DROPOUT}" \
  +composer/callbacks@composer.callbacks=gsm8k_progress_evaluator \
  composer.callbacks.gsm8k_progress_evaluator.num_samples="${PROGRESS_EVAL_SAMPLES}" \
  composer.callbacks.gsm8k_progress_evaluator.max_new_tokens="${PROGRESS_EVAL_TOKENS}" \
  training.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  training.grad_accum="${GRAD_ACCUM}" \
  composer.optimizer.lr="${LR}" \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup="${WARMUP}" \
  composer.lr_scheduler.alpha_f=0.1 \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval=1ba \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  ~composer.algorithms.ema
