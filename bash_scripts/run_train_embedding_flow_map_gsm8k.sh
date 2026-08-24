#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=../setup_env.sh
source setup_env.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_DURATION="${MAX_DURATION:-3ep}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1ep}"
PROGRESS_EVAL_SAMPLES="${PROGRESS_EVAL_SAMPLES:-16}"
PROGRESS_EVAL_TOKENS="${PROGRESS_EVAL_TOKENS:-256}"
PROGRESS_EVAL_STEPS="${PROGRESS_EVAL_STEPS:-1}"
MODEL_LENGTH="${MODEL_LENGTH:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-10ba}"
SHARED_RANK="${SHARED_RANK:-16}"
AR_RANK="${AR_RANK:-16}"
FLOW_RANK="${FLOW_RANK:-32}"
DIAGONAL_MIN_TIME="${DIAGONAL_MIN_TIME:-0.8}"
SEMIGROUP_WEIGHT="${SEMIGROUP_WEIGHT:-0.0}"
MAX_TIME_JUMP="${MAX_TIME_JUMP:-0.25}"
BOUNDARY_PROBABILITY="${BOUNDARY_PROBABILITY:-0.0}"
CURRICULUM_START="${CURRICULUM_START:-100}"
CURRICULUM_END="${CURRICULUM_END:-1000}"
DATASET_MODE="${DATASET_MODE:-small}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
RUN_NAME="${RUN_NAME:-gsm8k-embedding-flow-map-block${BLOCK_SIZE}-${MAX_DURATION}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT}/embedding-flow-map}"
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-embedding-flow-map}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_MODE

if [ "${DATASET_MODE}" = "full" ]; then
  DATASET_OVERRIDES=(
    "dataset@train_dataset=gsm8k_train"
    "dataset@eval_dataset=gsm8k_eval"
  )
else
  DATASET_OVERRIDES=(
    "dataset@train_dataset=gsm8k_small_train"
    "dataset@eval_dataset=gsm8k_small_eval"
    "train_dataset.max_samples=${TRAIN_SAMPLES}"
    "eval_dataset.max_samples=${EVAL_SAMPLES}"
  )
fi

uv run composer -n "${NUM_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME}" \
  "${DATASET_OVERRIDES[@]}" \
  model=embedding_flow_map \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen \
  model.config.length="${MODEL_LENGTH}" \
  model.config.attn_backend=sdpa \
  model.config.backbone_config.num_encoder_layers=-1 \
  model.config.backbone_config.num_decoder_layers=-1 \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.shared_lora_rank="${SHARED_RANK}" \
  model.config.ar_lora_rank="${AR_RANK}" \
  model.config.flow_lora_rank="${FLOW_RANK}" \
  model.config.diagonal_min_time="${DIAGONAL_MIN_TIME}" \
  model.config.semigroup_loss_weight="${SEMIGROUP_WEIGHT}" \
  model.config.max_time_jump="${MAX_TIME_JUMP}" \
  model.config.boundary_probability="${BOUNDARY_PROBABILITY}" \
  +composer/callbacks@composer.callbacks=embedding_flow_map_evaluators \
  composer.callbacks.flow_map_curriculum.start_batch="${CURRICULUM_START}" \
  composer.callbacks.flow_map_curriculum.end_batch="${CURRICULUM_END}" \
  composer.callbacks.gsm8k_progress_evaluator.num_samples="${PROGRESS_EVAL_SAMPLES}" \
  composer.callbacks.gsm8k_progress_evaluator.max_new_tokens="${PROGRESS_EVAL_TOKENS}" \
  composer.callbacks.gsm8k_progress_evaluator.num_steps="${PROGRESS_EVAL_STEPS}" \
  block_size="${BLOCK_SIZE}" \
  training.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  training.grad_accum="${GRAD_ACCUM}" \
  training.antithetic_sampling=false \
  composer.optimizer.lr="${LR}" \
  composer.lr_scheduler.t_warmup="${WARMUP}" \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval=1ba \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  ~metrics.nll \
  ~metrics.bpd \
  ~metrics.perplexity \
  ~eval_metrics.nll \
  ~eval_metrics.bpd \
  ~eval_metrics.perplexity \
  ~composer.algorithms.ema
