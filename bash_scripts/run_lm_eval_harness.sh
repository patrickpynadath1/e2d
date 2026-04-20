#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL_BASE="Qwen/Qwen3-1.7B-Base"
QWEN_MODEL_INSTRUCT="Qwen/Qwen3-1.7B"
QWEN_MODEL="${QWEN_MODEL_BASE}"
# QWEN_MODEL="meta-llama/Llama-3.2-1B"
NUM_FEW_SHOT=0

# TODO: Uncomment a model and run

########### AR
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_ar_20251201_061752"
# Qwen3-4B-Base, LR=5e-6
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr5e-6_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers36_ar_20260322_125401_fsdp"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true

######## E2D2
# Qwen3-1.7B-Base 28(4)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec4_e2d2_20260310_220501_tie-weights"
# Qwen3-4B-Base 36(4)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr5e-6_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc36_TOPdec4_e2d2_20260323_111133_tie-weights_fsdp"
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true

######## LayerSkip (self-speculative decoding via early exit)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260212_203501"
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260213_022759"
# Qwen3-1.7B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260213_193059"
# Qwen3-4B-Base
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr5e-6_bsz2_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers36_layerskip_20260323_111643_fsdp"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true
# ASSISTANT_EARLY_EXIT=8  # Layer index for self-speculative drafting (1 to num_hidden_layers-1); set to 0 or comment out to disable

######## MTP (self-speculative decoding with future-token heads)
# w_MTP = 0.1, 80.61 tokens/s, (61.56%, 2.46), Acc. 57.01%
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_mtp_20260403_193843"
# w_MTP = 0.2, 80.62 tokens/s, (63.92, 2.56), Acc. 55.50%
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_mtp_20260403_194009"
# mtp_per_block, w_MTP = 0.1
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_mtp_20260404_081803"
# mtp_per_block, w_MTP = 1.0
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_mtp_20260404_082144"
# BLOCK_SIZE=1
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true
# USE_MTP_SPECULATION=true
# MTP_DRAFT_LEN=4

######## E2D
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
MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20251228_064711_tie-weights"
# 28(1)
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec1_e2d_20260118_054940_tie-weights"
# 28(4), freeze bottom encoder
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec4_e2d_20260211_041417_tie-weights"
# 28(2), lora r = 16, alpha = 32, lr = 1e-4
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-4_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260222_010759_tie-weights"
# 28(2), block_size=2
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block2_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_003648_tie-weights"
# 28(2), block_size=6
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block6_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_003421_tie-weights"
# 28(2), block_size=8
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block8_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260318_222300_tie-weights"
# 28(2), block_size=10
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block10_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_003211_tie-weights"
# 28(2), block_size=12
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block12_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_002950_tie-weights"
# 28(2), lambda = 2.0
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260406_001643_tie-weights"
# 28(2), lambda = 1.5
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_195525_tie-weights"
# 28(2), lambda = 0.5
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_195642_tie-weights"
# 28(2), lambda = 0.1
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260405_195841_tie-weights"
# Qwen3-4B-Base, 36(2), lr = 5e-6
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr5e-6_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc36_TOPdec2_e2d_20260323_011022_tie-weights"
# ultrachat (instruction-tuned Qwen3-1.7B with thinking mode), 10k, bsz1
# MODEL_PATH="/data/shared_data/hankun/outputs/ultrachat_block8_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur3ep_amp_bf16_enc28_TOPdec2_e2d_ultrachat_20260419_032210_tie-weights"
# fine-tune from AR
# MODEL_PATH="/data/shared_data/hankun/outputs/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec2_e2d_20260418_032322_tie-weights"
BLOCK_SIZE=4
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false
USE_EMA=true

IS_INSTRUCTION_MODEL=false
if [[ "${MODEL_PATH}" == *"ultrachat"* ]]; then
  IS_INSTRUCTION_MODEL=true
  QWEN_MODEL="${QWEN_MODEL_INSTRUCT}"
fi

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

if [[ "${MODEL_PATH}" = /* ]]; then
  OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
else
  OUTPUT_DIR="/data/shared_data/hankun/outputs/${MODEL_PATH}/lm_eval_harness_output"
fi
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
USE_MTP_SPECULATION=${USE_MTP_SPECULATION:-false}
MTP_DRAFT_LEN=${MTP_DRAFT_LEN:-4}


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
  +task.model.is_instruction_model=${IS_INSTRUCTION_MODEL} \
  +task.model.use_chat_template_for_gsm8k=${IS_INSTRUCTION_MODEL} \
  +task.model.enable_thinking_for_gsm8k=${IS_INSTRUCTION_MODEL} \
  +task.model.strip_thinking_for_gsm8k=${IS_INSTRUCTION_MODEL} \
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
  $(if [ "${ASSISTANT_EARLY_EXIT}" -gt 0 ] 2>/dev/null; then echo "+gen_kwargs.assistant_early_exit=${ASSISTANT_EARLY_EXIT}"; fi) \
  $(if [ "${USE_MTP_SPECULATION}" == "true" ] 2>/dev/null; then echo "+gen_kwargs.use_mtp_speculation=true +gen_kwargs.mtp_draft_len=${MTP_DRAFT_LEN}"; fi)
