#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=../setup_env.sh
source setup_env.sh

: "${AR_CHECKPOINT_PATH:?Set AR_CHECKPOINT_PATH to the GSM8K AR weights checkpoint}"

MODE="${MODE:-masked}"
case "${MODE}" in
  masked) MODEL_CONFIG=masked_diffusion_e2d ;;
  uniform) MODEL_CONFIG=uniform_diffusion_e2d ;;
  *) echo "MODE must be 'masked' or 'uniform'" >&2; exit 2 ;;
esac

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
NUM_DEVICES="${NUM_DEVICES:-1}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"
MAX_DURATION="${MAX_DURATION:-100ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25ba}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
INFERENCE_STEPS="${INFERENCE_STEPS:-1}"
LR="${LR:-1e-5}"
WARMUP="${WARMUP:-20ba}"
RUN_NAME="${RUN_NAME:-e2d-gsm8k-${MODE}-diffusion-block${BLOCK_SIZE}-${MAX_DURATION}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_CACHE_HOME}/runs}"
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-drafter-study}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_MODE

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
  model="${MODEL_CONFIG}" \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen \
  model.config.length=256 \
  model.config.attn_backend=sdpa \
  model.config.inference_steps="${INFERENCE_STEPS}" \
  model.config.backbone_config.num_encoder_layers=28 \
  model.config.backbone_config.num_decoder_layers=2 \
  model.config.backbone_config.keep_top_decoder_layers=true \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.reinit_encoder=false \
  model.config.backbone_config.reinit_decoder=false \
  model.config.backbone_config.train_on_ar=true \
  model.config.backbone_config.ar_checkpoint_path="${AR_CHECKPOINT_PATH}" \
  block_size="${BLOCK_SIZE}" \
  eval_block_size="${BLOCK_SIZE}" \
  training.global_batch_size="${NUM_DEVICES}" \
  training.grad_accum=1 \
  training.autoresume=false \
  training.antithetic_sampling=false \
  composer.optimizer.lr="${LR}" \
  composer.lr_scheduler.t_warmup="${WARMUP}" \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.eval_subset_num_batches="${EVAL_BATCHES}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval=1ba \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  eval_dataloader.batch_size=1 \
  ~composer.algorithms.ema \
  ~composer.callbacks.hf_compatible_checkpointing \
  ~composer.callbacks.save_best_checkpointing
