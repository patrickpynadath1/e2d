#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=../setup_env.sh
source setup_env.sh

: "${AR_CHECKPOINT_PATH:?Set AR_CHECKPOINT_PATH to an AR weights-only checkpoint}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
RUN_NAME="${RUN_NAME:-e2d-gsm8k-block4-30kexamples-2gpu-$(date +%Y%m%d-%H%M%S)}"
NUM_DEVICES="${NUM_DEVICES:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
MAX_DURATION="${MAX_DURATION:-15000ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000ba}"
CHECKPOINTS_TO_KEEP="${CHECKPOINTS_TO_KEEP:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT}/studies}"
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-drafter-study}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_MODE

uv run composer -n "${NUM_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME}" \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  +metrics.encoder_loss._target_=src.tasks.metrics.EncoderLoss \
  +metrics.decoder_loss._target_=src.tasks.metrics.DecoderLoss \
  +eval_metrics.encoder_loss._target_=src.tasks.metrics.EncoderLoss \
  +eval_metrics.decoder_loss._target_=src.tasks.metrics.DecoderLoss \
  model=e2d \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen \
  model.config.length=768 \
  model.config.attn_backend=sdpa \
  model.config.backbone_config.num_encoder_layers=28 \
  model.config.backbone_config.num_decoder_layers=2 \
  model.config.backbone_config.keep_top_decoder_layers=true \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.reinit_encoder=false \
  model.config.backbone_config.reinit_decoder=false \
  model.config.backbone_config.train_on_ar=true \
  model.config.backbone_config.ar_checkpoint_path="${AR_CHECKPOINT_PATH}" \
  model.config.decoder_loss_lambda=1.0 \
  block_size=4 \
  eval_block_size=4 \
  training.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  training.grad_accum=1 \
  training.autoresume=false \
  training.antithetic_sampling=false \
  composer.optimizer.lr=1e-5 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=100ba \
  composer.lr_scheduler.alpha_f=0.5 \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.trainer.save_num_checkpoints_to_keep="${CHECKPOINTS_TO_KEEP}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval=10ba \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  '~composer.callbacks.save_best_checkpointing' \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  eval_dataloader.batch_size=1
