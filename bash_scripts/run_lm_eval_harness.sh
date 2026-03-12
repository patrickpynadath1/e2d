#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"
# QWEN_MODEL="meta-llama/Llama-3.2-1B"
# QWEN_MODEL="Qwen/Qwen3-8B-Base"
NUM_FEW_SHOT=0

# TODO: Uncomment a model and run

########### AR
# MODEL_PATH="outputs/<PATH_TO_AR_SAVED_MODEL_DIR>"
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_ar_20251201_061752"
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_ar_20251204_032135"
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers16_ar_20260129_004627"
# Qwen3-4B
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers36_ar_20260130_062332"
# Qwen3-8B
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers36_ar_20260202_040844_fsdp"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true

############ MDLM
#MODEL_PATH="outputs/<PATH_TO_MDLM_SAVED_MODEL_DIR>"
#BLOCK_SIZE=64
#KV_CACHING=false
#ALIGN_INPUTS_TO_BLOCKS=false
#USE_EMA=true

############ BD3LM
#MODEL_PATH="outputs/<PATH_TO_BD3LM_SAVED_MODEL_DIR>"
#BLOCK_SIZE=4
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#USE_EMA=true

######## E2D2
# 24(14)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d2_20251201_072658_tie-weights"
# 24(4)
MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec4_e2d2_20260310_220501_tie-weights"
BLOCK_SIZE=4
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=true
USE_EMA=true

######## LayerSkip (self-speculative decoding via early exit)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260212_203501"
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260213_022759"
# qwen3-1.7b-base
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260213_193059"
# llama-3.2-1b
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers16_layerskip_llama_20260214_032750"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true
# ASSISTANT_EARLY_EXIT=8  # Layer index for self-speculative drafting (1 to num_hidden_layers-1); set to 0 or comment out to disable

######## E2D
# MODEL_PATH="kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B"
# enforce_causal_mask
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d_20251219_031056_tie-weights"
# enforce_causal_mask, also let encoder predict the next token
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d_20251220_064902_tie-weights"
# adapter (context bidirectional)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d_20251228_012410_tie-weights"
# adapter (context causal)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur100000ba_amp_bf16_enc28_TOPdec14_e2d_20251229_010955_tie-weights"
# 28(14)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d_20260219_232800_tie-weights"
# 28(12)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec12_e2d_20260219_180924_tie-weights"
# 28(10)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec10_e2d_20260219_180846_tie-weights"
# 28(8)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec8_e2d_20260219_011258_tie-weights"
# 28(6)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec6_e2d_20260218_204901_tie-weights"
# 28(4)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec4_e2d_20251227_234110_tie-weights"
# 28(2)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20251228_064711_tie-weights"
# 28(1)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec1_e2d_20260118_054940_tie-weights"
# 28(4), freeze bottom encoder
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec4_e2d_20260211_041417_tie-weights"
# llama-3.2-1b 16(2)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc16_TOPdec2_e2d_20260129_002614_tie-weights"
# Qwen3-4B 36(2)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc36_TOPdec2_e2d_20260130_071532_tie-weights"
# Qwen3-8B 36(2)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc36_TOPdec2_e2d_20260202_035147_tie-weights_fsdp"
# 26(2), lora r = 16, alpha = 32, lr = 1e-4
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-4_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260222_010759_tie-weights"
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=true

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

OUTPUT_DIR="/data/shared_data/hankun/outputs/${MODEL_PATH}/lm_eval_harness_output"
REVISION=null

mkdir -p ${OUTPUT_DIR}
L=512
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
T=${BLOCK_SIZE}
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6
CKPT="best"
ASSISTANT_EARLY_EXIT=${ASSISTANT_EARLY_EXIT:-0}  # 0 = disabled; set in model section above for LayerSkip


OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}"
OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_${NUM_FEW_SHOT}shot_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
mkdir -p ${OUTPUT_PATH}

accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  task.num_fewshot=${NUM_FEW_SHOT} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path=${QWEN_MODEL} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
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
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]' \
  $(if [ "${ASSISTANT_EARLY_EXIT}" -gt 0 ] 2>/dev/null; then echo "+gen_kwargs.assistant_early_exit=${ASSISTANT_EARLY_EXIT}"; fi)
