#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL_INSTRUCT="Qwen/Qwen3-4B"
QWEN_MODEL="${QWEN_MODEL_INSTRUCT}"

# TODO: Set your model path and decoding settings, then run.

########### Example model (E2D)
# MODEL_PATH="/data/shared_data/hankun/outputs/ultrachat_block8_lr1e-5_bsz16_warm100ba_alphaf0.5_max-dur3ep_amp_bf16_enc28_TOPdec2_e2d_ultrachat_20260421_065938_tie-weights"
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=true

# untuned Qwen3-4B
MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm0ba_alphaf0.5_max-dur0ba_amp_bf16_layers36_ar_20260423_010658"
BLOCK_SIZE=1
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=true
USE_EMA=true

# Your note says all models are instruction models.
IS_INSTRUCTION_MODEL=true
USE_CHAT_TEMPLATE=true

# MATH-500 settings
MATH500_DATASET="HuggingFaceH4/MATH-500"
MATH500_SPLIT="test"
MATH500_NUM_SAMPLES=null  # null for full 500, or set integer for debugging
PROMPT_STYLE="none"  # "none" or "concise"
INSTRUCTION_PREFIX="Solve the following math problem and give only the final answer."

L=512
CKPT="best"
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
T=${BLOCK_SIZE}
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6
ASSISTANT_EARLY_EXIT=${ASSISTANT_EARLY_EXIT:-0}
USE_MTP_SPECULATION=${USE_MTP_SPECULATION:-false}
MTP_DRAFT_LEN=${MTP_DRAFT_LEN:-4}

if [[ "${MODEL_PATH}" = /* ]]; then
  OUTPUT_DIR="${MODEL_PATH}/math500_output"
else
  OUTPUT_DIR="/data/shared_data/hankun/outputs/${MODEL_PATH}/math500_output"
fi

OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-prompt_${PROMPT_STYLE}"
mkdir -p ${OUTPUT_PATH}

uv run python scripts/eval/math500_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  pretrained_model_name_or_path=${MODEL_PATH} \
  ckpt_file="${CKPT}-rank0.pt" \
  load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path=${QWEN_MODEL} \
  output_path=${OUTPUT_PATH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.num_steps=${T} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria]' \
  +math500_dataset_name=${MATH500_DATASET} \
  +math500_split=${MATH500_SPLIT} \
  +math500_num_samples=${MATH500_NUM_SAMPLES} \
  +is_instruction_model=${IS_INSTRUCTION_MODEL} \
  +use_chat_template=${USE_CHAT_TEMPLATE} \
  +prompt_style=${PROMPT_STYLE} \
  +instruction_prefix="${INSTRUCTION_PREFIX}" \
  $(if [ "${ASSISTANT_EARLY_EXIT}" -gt 0 ] 2>/dev/null; then echo "+gen_kwargs.assistant_early_exit=${ASSISTANT_EARLY_EXIT}"; fi) \
  $(if [ "${USE_MTP_SPECULATION}" == "true" ] 2>/dev/null; then echo "+gen_kwargs.use_mtp_speculation=true +gen_kwargs.mtp_draft_len=${MTP_DRAFT_LEN}"; fi)