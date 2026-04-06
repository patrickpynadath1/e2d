#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
N_LAYERS=28
NUM_FUTURE_HEADS=4
MTP_BLOCK_SIZE=4
MTP_PER_BLOCK=true

# MTP loss weights
TARGET_LOSS_WEIGHT=1.0
FUTURE_LOSS_WEIGHT=1.0

# Hyperparameters
LR=1e-5
WARMUP_DURATION="100ba"
ALPHA_F=0.5
BATCH_SIZE=1
MAX_DURATION="30000ba"
PRECISION="amp_bf16"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base
NUM_SHOT=0
TRAIN_ON_CONTEXT=false

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TAG="mtp"
LAYERS="layers${N_LAYERS}"
RUN_NAME=gsm8k-${NUM_SHOT}shot_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${PRECISION}_${LAYERS}_${TAG}_${TIMESTAMP}

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  train_dataset.num_shot=${NUM_SHOT} \
  composer.optimizer.lr=${LR} \
  composer.trainer.precision=${PRECISION} \
  composer.trainer.eval_interval="1000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  model=mtp \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.length=768 \
  model.config.backbone_config.reinit_model=false \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=false \
  model.config.target_loss_weight=${TARGET_LOSS_WEIGHT} \
  model.config.future_loss_weight=${FUTURE_LOSS_WEIGHT} \
  model.config.num_future_heads=${NUM_FUTURE_HEADS} \
  model.config.speculative_draft_len=${NUM_FUTURE_HEADS} \
  model.config.mtp_per_block=${MTP_PER_BLOCK} \
  model.config.mtp_block_size=${MTP_BLOCK_SIZE} \
  model.config.train_on_context=${TRAIN_ON_CONTEXT} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  training.compile_backbone=false \
  hydra.run.dir=/data/shared_data/hankun/outputs/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  composer.callbacks.save_best_checkpointing.save_local=false \
  eval_dataloader.batch_size=2
