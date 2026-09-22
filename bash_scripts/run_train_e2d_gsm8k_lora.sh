#!/bin/bash
# Frozen Qwen3-1.7B verifier + SEED drafting LoRA, using existing self-distillation.
# Run from any directory. All generation checks must pass before Composer starts.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/gsm8k_qwen3_1p7b_selfdistill_bsz1_maxlen${MAX_SEQ_LEN}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/shared_data/hankun/outputs}
BLOCK_SIZE=${BLOCK_SIZE:-4}
N_DECODER_LAYERS=${N_DECODER_LAYERS:-4}
LORA_RANK=${LORA_RANK:-256}
LORA_ALPHA=${LORA_ALPHA:-512}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LR=${LR:-1e-4}
WARMUP_DURATION=${WARMUP_DURATION:-100ba}
ALPHA_F=${ALPHA_F:-0.5}
MAX_DURATION=${MAX_DURATION:-6ep}
BATCH_SIZE=${BATCH_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000ba}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000ba}
VERIFY=${VERIFY:-false}
VERIFY_ONLY=${VERIFY_ONLY:-false}
# Regenerate every completion with a KV cache. The faster teacher-forced option
# checks all reference argmaxes plus cached spot checks, but different attention
# shapes can change near-tied argmaxes in finite precision.
VERIFICATION_METHOD=${VERIFICATION_METHOD:-generate}
GENERATION_SAMPLES=${GENERATION_SAMPLES:-8}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-1}
else
  IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-${#DEVICES[@]}}
fi
if (( BATCH_SIZE < NUM_VISIBLE_DEVICES * MICRO_BATCH_SIZE || BATCH_SIZE % (NUM_VISIBLE_DEVICES * MICRO_BATCH_SIZE) != 0 )); then
  echo "BATCH_SIZE must be divisible by NUM_VISIBLE_DEVICES * MICRO_BATCH_SIZE." >&2
  exit 1
fi
if (( MAX_SEQ_LEN % BLOCK_SIZE != 0 )); then
  echo "MAX_SEQ_LEN must be divisible by BLOCK_SIZE." >&2
  exit 1
fi

if [[ ! -d "${DISTILL_DATA_ROOT}/train_preprocessed" || ! -d "${DISTILL_DATA_ROOT}/eval_preprocessed" ]]; then
  echo "Missing distilled training/eval splits in ${DISTILL_DATA_ROOT}." >&2
  echo "Run: bash bash_scripts/generate_gsm8k_selfdistill_bsz1.sh" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=${RUN_NAME:-gsm8k_seed_lora_block${BLOCK_SIZE}_TOPdec${N_DECODER_LAYERS}_rank${LORA_RANK}_lr${LR}_bsz${BATCH_SIZE}_${TIMESTAMP}}
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

if [[ "${VERIFY_ONLY}" == true && "${VERIFY}" != true ]]; then
  echo "ERROR: VERIFY_ONLY=true requires VERIFY=true." >&2
  exit 1
fi

if [[ "${VERIFY}" == true ]]; then
  echo "[verify] Starting self-distillation verification..."

  python scripts/verify_gsm8k_selfdistill.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --data-root "${DISTILL_DATA_ROOT}" \
    --max-length "${MAX_SEQ_LEN}" \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --device cuda \
    --verification-method "${VERIFICATION_METHOD}" \
    --max-samples 0 \
    --generation-samples "${GENERATION_SAMPLES}" \
    --report "${RUN_DIR}/selfdistill_verification.json"

  if [[ "${VERIFY_ONLY}" == true ]]; then
    exit 0
  fi
else
  echo "[verify] Skipping self-distillation verification (VERIFY=false)."
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Only low-rank parameters are trainable. Disable EMA because the existing EMA
# algorithm also averages target buffers; checkpoint and eval use raw weights.
composer -n "${NUM_VISIBLE_DEVICES}" scripts/composer_scripts/train_discrete_denoiser.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME_OR_PATH}" \
  dataset@train_dataset=ultrachat_distill_train \
  dataset@eval_dataset=ultrachat_distill_eval \
  train_dataset.dataset_path="${DISTILL_DATA_ROOT}/train_preprocessed" \
  eval_dataset.dataset_path="${DISTILL_DATA_ROOT}/eval_preprocessed" \
  model=e2d \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_seed_lora \
  model.config.length="${MAX_SEQ_LEN}" \
  model.config.attn_backend=sdpa \
  model.config.train_on_context=false \
  model.config.backbone_config.num_decoder_layers="${N_DECODER_LAYERS}" \
  model.config.backbone_config.lora_rank="${LORA_RANK}" \
  model.config.backbone_config.lora_alpha="${LORA_ALPHA}" \
  model.config.backbone_config.lora_dropout="${LORA_DROPOUT}" \
  training.compile_backbone=false \
  training.global_batch_size="${BATCH_SIZE}" \
  training.grad_accum="$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE ))" \
  training.antithetic_sampling=false \
  block_size="${BLOCK_SIZE}" \
  eval_block_size="${BLOCK_SIZE}" \
  'composer/algorithms=[gradient_clipping]' \
  composer.optimizer.lr="${LR}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup="${WARMUP_DURATION}" \
  composer.lr_scheduler.alpha_f="${ALPHA_F}" \
  composer.loggers.name="${RUN_NAME}" \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  composer.callbacks.save_best_checkpointing.save_local=false \
  train_dataloader.num_workers="${NUM_WORKERS}" \
  eval_dataloader.batch_size=1 \
  hydra.run.dir="${RUN_DIR}" \
  "$@"
