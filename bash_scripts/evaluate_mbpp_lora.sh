#!/bin/bash
# Generate with a frozen-target SEED-LoRA checkpoint on all 500 MBPP test problems.
# Generation only: reports outputs, decoding throughput, and draft statistics;
# no generated code is executed or scored.
# From any directory:
#   MODEL_PATH=/path/to/training/run bash bash_scripts/evaluate_mbpp_lora.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# Match the checkpoint/tokenizer defaults in evaluate_math500_lora.sh.
MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_block8_1.7b_dec2_rank512_lr6e-4_bsz32_20260926_204506}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-1.7B}
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-4}
CONF_SEG=${CONF_SEG:-true}
TRACK_ACC_RATE=${TRACK_ACC_RATE:-false}
TREE_ATTN=${TREE_ATTN:-true}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
MBPP_NUM_SAMPLES=${MBPP_NUM_SAMPLES:-null}  # null for all, or an integer
if [[ "${CONF_SEG}" == true && "${TRACK_ACC_RATE}" == true ]]; then
  echo "Set CONF_SEG=false when using TRACK_ACC_RATE=true." >&2
  exit 1
fi
OUTPUT_PATH=${OUTPUT_PATH:-"${MODEL_PATH}/mbpp_output/seed_lora_${CKPT}_L${MAX_NEW_TOKENS}_block${BLOCK_SIZE}_conf${CONF_SEG}_adaptive${TRACK_ACC_RATE}_tree${TREE_ATTN}_sample${DO_SAMPLE}"}
mkdir -p "${OUTPUT_PATH}"

# Reuse MATH-500's non-thinking chat prompt and frozen-target generation.
# Outputs: rank0.json (generations), generation_stats-rank0.json (throughput,
# acceptance), draft_position_acceptance-rank0.{json,txt} (per-position rates).
accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=mbpp \
  task.num_fewshot=0 \
  task.limit="${MBPP_NUM_SAMPLES}" \
  pretrained_model_name_or_path="${MODEL_PATH}" \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=false \
  +task.model.require_frozen_target=true \
  +task.model.is_instruction_model=true \
  '+task.model.gsm8k_chat_template_kwargs={enable_thinking:false}' \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  output_path="${OUTPUT_PATH}" \
  generated_samples_output_path="${OUTPUT_PATH}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  block_size="${BLOCK_SIZE}" \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  generation_config.num_steps="${BLOCK_SIZE}" \
  generation_config.use_cache=true \
  generation_config.align_inputs_to_blocks=false \
  '~generation/logits_processor@logits_processor_list' \
  gen_kwargs.logits_processor=null \
  'generation/stopping_criteria@stopping_criteria_list=[eos_token_criteria,max_length_criteria]' \
  +gen_kwargs.conf_seg="${CONF_SEG}" \
  +gen_kwargs.track_acc_rate="${TRACK_ACC_RATE}" \
  +gen_kwargs.tree_attn="${TREE_ATTN}" \
  "$@"
