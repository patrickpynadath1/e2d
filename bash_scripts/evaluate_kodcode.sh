#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Required for code-eval tasks: allows execution of model-generated code
export HF_ALLOW_CODE_EVAL="1"

QWEN_MODEL="Qwen/Qwen3-4B-Base"

# TODO: Uncomment a model and run

########### AR
# Qwen3-1.7B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur50000ba_amp_bf16_layers28_ar_20260303_000744"
# Qwen3-4B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_lr5e-6_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers36_ar_20260322_123604_fsdp"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true

###########  E2D
# Qwen3-1.7B-Base 28(2), block10
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_block10_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260507_200129_tie-weights"
# Qwen3-4B-Base 36(2), block10
MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_block10_lr5e-6_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc36_TOPdec2_e2d_20260508_055449_tie-weights_fsdp"
# Qwen3-1.7B-Base 28(2), General SFT, bsz = 1, lr = 1e-5
# MODEL_PATH="/data/shared_data/hankun/outputs/ultrachat_block8_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur3ep_amp_bf16_enc28_TOPdec2_e2d_ultrachat_20260420_203202_tie-weights"
BLOCK_SIZE=4
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false
USE_EMA=true

###########  E2D2
# Qwen3-1.7B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur50000ba_amp_bf16_enc28_TOPdec4_e2d2_20260303_073356_tie-weights"
# Qwen3-4B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_block4_lr5e-6_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc36_TOPdec4_e2d2_20260324_112023_tie-weights"
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=true

###########  LayerSkip
# Qwen3-1.7B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur50000ba_amp_bf16_layers28_layerskip_20260303_053152"
# Qwen3-4B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/kodcode-0shot_lr5e-6_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers36_layerskip_20260324_075954"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true
# ASSISTANT_EARLY_EXIT=12

# KodCode evaluation settings
KODCODE_DIFFICULTY="easy"       # "easy", "medium", "hard", or null for all
KODCODE_NUM_SAMPLES=null      # null for all, or integer to limit

L=1024
CKPT="best"
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
T=${BLOCK_SIZE}
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6
ASSISTANT_EARLY_EXIT=${ASSISTANT_EARLY_EXIT:-0}

OUTPUT_DIR="${MODEL_PATH}/kodcode_output"
OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-assistant_early_exit${ASSISTANT_EARLY_EXIT}-difficulty_${KODCODE_DIFFICULTY}"
mkdir -p ${OUTPUT_PATH}

uv run python scripts/eval/kodcode_eval.py \
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
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,kodcode_stopping_criteria]' \
  kodcode_difficulty=${KODCODE_DIFFICULTY} \
  kodcode_num_samples=${KODCODE_NUM_SAMPLES} \
  $(if [ "${ASSISTANT_EARLY_EXIT}" -gt 0 ] 2>/dev/null; then echo "+gen_kwargs.assistant_early_exit=${ASSISTANT_EARLY_EXIT}"; fi)
