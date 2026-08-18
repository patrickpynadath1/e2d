#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=../setup_env.sh
source setup_env.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
FLOW_MODEL_CONFIG="${FLOW_MODEL_CONFIG:-latent_flow_e2d}"
USE_AR_CHECKPOINT="${USE_AR_CHECKPOINT:-false}"
DATASET_MODE="${DATASET_MODE:-small}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${NUM_DEVICES}}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"
MAX_DURATION="${MAX_DURATION:-100ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25ba}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
EVAL_BLOCK_SIZE="${EVAL_BLOCK_SIZE:-${BLOCK_SIZE}}"
MODEL_LENGTH="${MODEL_LENGTH:-256}"
INFERENCE_STEPS="${INFERENCE_STEPS:-1}"
LATENT_STATS_PATH="${LATENT_STATS_PATH:-${PWD}/outputs/diagnostics/qwen-latent-norms/normalization_stats.pt}"
PREDICTION_TYPE="${PREDICTION_TYPE:-velocity}"
ADAPTIVE_TIMESTEP_SAMPLING="${ADAPTIVE_TIMESTEP_SAMPLING:-false}"
ADAPTIVE_NUM_BINS="${ADAPTIVE_NUM_BINS:-50}"
ADAPTIVE_EMA_DECAY="${ADAPTIVE_EMA_DECAY:-0.99}"
ADAPTIVE_UNIFORM_MIX="${ADAPTIVE_UNIFORM_MIX:-0.2}"
ADAPTIVE_MIN_OBSERVATIONS="${ADAPTIVE_MIN_OBSERVATIONS:-100}"
SPEC_EVAL_MAX_NEW_TOKENS="${SPEC_EVAL_MAX_NEW_TOKENS:-64}"
SPEC_EVAL_NUM_PROMPTS="${SPEC_EVAL_NUM_PROMPTS:-8}"
FIXED_TRAINING_NOISE="${FIXED_TRAINING_NOISE:-false}"
FIXED_TRAINING_SEED="${FIXED_TRAINING_SEED:-17}"
LR="${LR:-1e-5}"
WARMUP="${WARMUP:-20ba}"
RUN_NAME="${RUN_NAME:-e2d-gsm8k-latent-flow-block${BLOCK_SIZE}-${MAX_DURATION}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${E2D_OUTPUT_ROOT}/reference}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100ba}"
CHECKPOINTS_TO_KEEP="${CHECKPOINTS_TO_KEEP:-0}"
ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-false}"
AUTORESUME="${AUTORESUME:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-drafter-study}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_MODE

if [ "${PREDICTION_TYPE}" != "velocity" ] && [ "${PREDICTION_TYPE}" != "x0" ]; then
  echo "PREDICTION_TYPE must be 'velocity' or 'x0'" >&2
  exit 2
fi

if [ "${FLOW_MODEL_CONFIG}" = "latent_flow_e2d" ]; then
  if [ ! -f "${LATENT_STATS_PATH}" ]; then
    echo "Latent statistics artifact not found: ${LATENT_STATS_PATH}" >&2
    echo "Run scripts/eval/compare_qwen_latent_gaussian_norms.py first." >&2
    exit 2
  fi
  if [ "${FIXED_TRAINING_NOISE}" = "true" ] && [ "${ADAPTIVE_TIMESTEP_SAMPLING}" = "true" ]; then
    echo "FIXED_TRAINING_NOISE=true cannot be combined with adaptive timestep sampling." >&2
    exit 2
  fi
  FLOW_OVERRIDES=(
    "model.config.latent_stats_path=${LATENT_STATS_PATH}"
    "model.config.prediction_type=${PREDICTION_TYPE}"
    "model.config.adaptive_timestep_sampling=${ADAPTIVE_TIMESTEP_SAMPLING}"
    "model.config.adaptive_num_bins=${ADAPTIVE_NUM_BINS}"
    "model.config.adaptive_ema_decay=${ADAPTIVE_EMA_DECAY}"
    "model.config.adaptive_uniform_mix=${ADAPTIVE_UNIFORM_MIX}"
    "model.config.adaptive_min_observations=${ADAPTIVE_MIN_OBSERVATIONS}"
  )
else
  FLOW_OVERRIDES=()
fi

if [ "${FLOW_MODEL_CONFIG}" = "riemannian_latent_flow_e2d" ]; then
  SPHERICAL_OVERRIDES=(
    "+metrics.direction_loss._target_=src.tasks.metrics.DirectionLoss"
    "+metrics.scalar_loss._target_=src.tasks.metrics.ScalarLoss"
    "+metrics.angular_error._target_=src.tasks.metrics.AngularError"
    "+metrics.radius_relative_error._target_=src.tasks.metrics.RadiusRelativeError"
    "+eval_metrics.direction_loss._target_=src.tasks.metrics.DirectionLoss"
    "+eval_metrics.scalar_loss._target_=src.tasks.metrics.ScalarLoss"
    "+eval_metrics.angular_error._target_=src.tasks.metrics.AngularError"
    "+eval_metrics.radius_relative_error._target_=src.tasks.metrics.RadiusRelativeError"
  )
else
  SPHERICAL_OVERRIDES=()
fi

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
    "~composer.callbacks.hf_compatible_checkpointing"
    "~composer.callbacks.save_best_checkpointing"
  )
else
  CHECKPOINT_OVERRIDES=(
    "composer.trainer.save_folder=null"
    "~composer.callbacks.hf_compatible_checkpointing"
    "~composer.callbacks.save_best_checkpointing"
  )
fi

uv run composer -n "${NUM_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME}" \
  "${DATASET_OVERRIDES[@]}" \
  ~metrics.nll \
  ~metrics.bpd \
  ~metrics.perplexity \
  ~eval_metrics.nll \
  ~eval_metrics.bpd \
  ~eval_metrics.perplexity \
  +metrics.flow_loss._target_=src.tasks.metrics.FlowLoss \
  +eval_metrics.flow_loss._target_=src.tasks.metrics.FlowLoss \
  "${SPHERICAL_OVERRIDES[@]}" \
  model="${FLOW_MODEL_CONFIG}" \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen \
  model.config.length="${MODEL_LENGTH}" \
  model.config.attn_backend=sdpa \
  model.config.inference_steps="${INFERENCE_STEPS}" \
  "${FLOW_OVERRIDES[@]}" \
  model.config.fixed_training_noise="${FIXED_TRAINING_NOISE}" \
  model.config.fixed_training_seed="${FIXED_TRAINING_SEED}" \
  model.config.backbone_config.num_encoder_layers=28 \
  model.config.backbone_config.num_decoder_layers=2 \
  model.config.backbone_config.keep_top_decoder_layers=true \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.reinit_encoder=false \
  model.config.backbone_config.reinit_decoder=false \
  "${AR_INIT_OVERRIDES[@]}" \
  block_size="${BLOCK_SIZE}" \
  eval_block_size="${EVAL_BLOCK_SIZE}" \
  training.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  training.grad_accum="${GRAD_ACCUM}" \
  training.autoresume="${AUTORESUME}" \
  training.antithetic_sampling=false \
  composer.optimizer.lr="${LR}" \
  composer.lr_scheduler.t_warmup="${WARMUP}" \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  +composer/callbacks@composer.callbacks=flow_evaluators \
  composer.callbacks.speculative_generation_evaluator.num_steps="${INFERENCE_STEPS}" \
  composer.callbacks.speculative_generation_evaluator.max_new_tokens="${SPEC_EVAL_MAX_NEW_TOKENS}" \
  composer.callbacks.speculative_generation_evaluator.num_prompts="${SPEC_EVAL_NUM_PROMPTS}" \
  composer.trainer.eval_subset_num_batches="${EVAL_BATCHES}" \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.trainer.save_num_checkpoints_to_keep="${CHECKPOINTS_TO_KEEP}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.console_log_interval=1ba \
  hydra.run.dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0 \
  eval_dataloader.batch_size=1 \
  ~composer.algorithms.ema \
  "${CHECKPOINT_OVERRIDES[@]}"
