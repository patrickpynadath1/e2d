#!/bin/bash
# Evaluate the untuned Qwen3-1.7B release on all 500 MATH-500 test problems.
# From any directory:
#   bash /path/to/e2d/bash_scripts/evaluate_math500.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# 29.18 tokens/s

QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-1.7B}
MODEL_PATH=${MODEL_PATH:-${QWEN_MODEL}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
MATH500_NUM_SAMPLES=${MATH500_NUM_SAMPLES:-null}  # null for all, or an integer
MODEL_NAME=${MODEL_PATH##*/}
OUTPUT_PATH=${OUTPUT_PATH:-"${PWD}/outputs/${MODEL_NAME}/math500_output/untuned_L${MAX_NEW_TOKENS}_sample${DO_SAMPLE}"}
mkdir -p "${OUTPUT_PATH}"

# Share the LoRA task, non-thinking chat prompt, boxed-answer scorer, and reports.
# Standard HF generation handles EOS and max_new_tokens itself. The E2D length
# stopper sees only generated tokens; HF stopping criteria see the prompt too.
accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=math500 \
  task.num_fewshot=0 \
  task.limit="${MATH500_NUM_SAMPLES}" \
  pretrained_model_name_or_path="${MODEL_PATH}" \
  task.model.load_ema_weights=false \
  '+task.model.model_config_overrides={torch_dtype:bfloat16}' \
  +task.model.is_instruction_model=true \
  +task.model.use_chat_template_for_gsm8k=true \
  +task.model.enable_thinking_for_gsm8k=false \
  +task.model.strip_thinking_for_gsm8k=true \
  '+task.model.gsm8k_chat_template_kwargs={enable_thinking:false}' \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  output_path="${OUTPUT_PATH}" \
  generated_samples_output_path="${OUTPUT_PATH}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  generation@generation_config=generation_config \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  generation_config.use_cache=true \
  +generation_config.top_k=0 \
  +generation_config.top_p=1.0 \
  '+generation_config.eos_token_id=${eos_token_id}' \
  '+generation_config.pad_token_id=${eos_token_id}' \
  '~generation/logits_processor@logits_processor_list' \
  gen_kwargs.logits_processor=null \
  '~generation/stopping_criteria@stopping_criteria_list' \
  gen_kwargs.stopping_criteria=null \
  "$@"
