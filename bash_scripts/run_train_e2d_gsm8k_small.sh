#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=../setup_env.sh
source setup_env.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
USE_AR_CHECKPOINT="${USE_AR_CHECKPOINT:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT}/reference}"
RUN_NAME="${RUN_NAME:-e2d-gsm8k-small-block4}"
NUM_DEVICES="${NUM_DEVICES:-1}"
MAX_DURATION="${MAX_DURATION:-20ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10ba}"
CHECKPOINTS_TO_KEEP="${CHECKPOINTS_TO_KEEP:-0}"
ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-false}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
LR="${LR:-1e-5}"
WARMUP="${WARMUP:-1000ba}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-1ba}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

if [ "${USE_AR_CHECKPOINT}" = "true" ]; then
  : "${AR_CHECKPOINT_PATH:?Set AR_CHECKPOINT_PATH when USE_AR_CHECKPOINT=true}"
  AR_INIT_OVERRIDES=(
    "model.config.backbone_config.train_on_ar=true"
    "model.config.backbone_config.ar_checkpoint_path=${AR_CHECKPOINT_PATH}"
  )
else
  AR_INIT_OVERRIDES=(
    "model.config.backbone_config.train_on_ar=false"
    "model.config.backbone_config.ar_checkpoint_path=null"
  )
fi

if [ "${ENABLE_CHECKPOINTING}" = "true" ]; then
  CHECKPOINT_OVERRIDES=(
    "composer.callbacks.hf_compatible_checkpointing.disable_hf=true"
    "~composer.callbacks.save_best_checkpointing"
  )
else
  CHECKPOINT_OVERRIDES=(
    "~composer.callbacks.hf_compatible_checkpointing"
    "~composer.callbacks.save_best_checkpointing"
  )
fi

uv run composer -n "${NUM_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME}" \
  dataset@train_dataset=gsm8k_small_train \
  dataset@eval_dataset=gsm8k_small_eval \
  train_dataset.max_samples="${TRAIN_SAMPLES}" \
  eval_dataset.max_samples="${EVAL_SAMPLES}" \
  +metrics.encoder_loss._target_=src.tasks.metrics.EncoderLoss \
  +metrics.decoder_loss._target_=src.tasks.metrics.DecoderLoss \
  +eval_metrics.encoder_loss._target_=src.tasks.metrics.EncoderLoss \
  +eval_metrics.decoder_loss._target_=src.tasks.metrics.DecoderLoss \
  model=e2d \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen \
  model.config.length=256 \
  model.config.attn_backend=sdpa \
  model.config.backbone_config.num_encoder_layers=28 \
  model.config.backbone_config.num_decoder_layers=2 \
  model.config.backbone_config.keep_top_decoder_layers=true \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.reinit_encoder=false \
  model.config.backbone_config.reinit_decoder=false \
  "${AR_INIT_OVERRIDES[@]}" \
  model.config.decoder_loss_lambda=1.0 \
  block_size=4 \
  eval_block_size=4 \
  training.global_batch_size="${NUM_DEVICES}" \
  training.grad_accum=1 \
  training.autoresume=false \
  composer.optimizer.lr="${LR}" \
  composer.lr_scheduler.t_warmup="${WARMUP}" \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.eval_subset_num_batches=4 \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.trainer.save_num_checkpoints_to_keep="${CHECKPOINTS_TO_KEEP}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval="${CONSOLE_LOG_INTERVAL}" \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  eval_dataloader.batch_size=1 \
  ~composer.algorithms.ema \
  "${CHECKPOINT_OVERRIDES[@]}"
