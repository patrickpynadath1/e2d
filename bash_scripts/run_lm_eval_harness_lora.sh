#!/bin/bash
# Evaluate a frozen-target SEED-LoRA checkpoint on the GSM8K test set.
# From any directory:
#   MODEL_PATH=/path/to/training/run bash bash_scripts/run_lm_eval_harness_lora.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# ultrachat lr 3e-4, b = 4, dec = 2, rank = 512 (unfinished, 12000 steps, eval loss 1.8)
# 1.62 (82.75%)
# ultrachat lr 1e-4, b = 8, dec = 2, rank = 512 (unfinished, 15500 steps, eval loss 2.0)
# 1.49 (80.38%)
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-1.7B}
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-4}
CONF_SEG=${CONF_SEG:-true}
TRACK_ACC_RATE=${TRACK_ACC_RATE:-false}
TREE_ATTN=${TREE_ATTN:-true}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
if [[ "${CONF_SEG}" == true && "${TRACK_ACC_RATE}" == true ]]; then
  echo "Set CONF_SEG=false when using TRACK_ACC_RATE=true." >&2
  exit 1
fi
OUTPUT_PATH=${OUTPUT_PATH:-"${MODEL_PATH}/lm_eval_harness_output/seed_lora_${CKPT}_L${MAX_NEW_TOKENS}_block${BLOCK_SIZE}_conf${CONF_SEG}_adaptive${TRACK_ACC_RATE}_tree${TREE_ATTN}_sample${DO_SAMPLE}"}
mkdir -p "${OUTPUT_PATH}"

# Keep the existing harness task/metrics and generated-sample reports. Explicit
# non-thinking chat formatting matches the audited self-distillation prompts.
# The frozen target supplies verification, branch continuations, and corrections.
accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  task.num_fewshot=0 \
  pretrained_model_name_or_path="${MODEL_PATH}" \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=false \
  +task.model.require_frozen_target=true \
  +task.model.is_instruction_model=true \
  +task.model.use_chat_template_for_gsm8k=true \
  +task.model.enable_thinking_for_gsm8k=false \
  +task.model.strip_thinking_for_gsm8k=true \
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
  'generation/stopping_criteria@stopping_criteria_list=[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]' \
  +gen_kwargs.conf_seg="${CONF_SEG}" \
  +gen_kwargs.track_acc_rate="${TRACK_ACC_RATE}" \
  +gen_kwargs.tree_attn="${TREE_ATTN}" \
  "$@"
