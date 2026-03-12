#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: Uncomment a model and run

######### AR
# PROMPT_TEXT="Translation: "
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# BLOCK_SIZE=1
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block_lr3e-4_bsz128_warm1000ba_layers32_hidden512_inter1536_ar_target_prompt_20251112_031736"
# finetune
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block_lr3e-4_bsz128_warm1000ba_layers28_hidden1024_inter3072_ar_target_prompt_20251222_165729"

########### MDLM
#PROMPT_TEXT=null
#KV_CACHING=false
#ALIGN_INPUTS_TO_BLOCKS=false
#BLOCK_SIZE=32
#MODEL_PATH="outputs/<PATH_TO_MDLM_SAVED_MODEL_DIR>"

########### BD3LM
#PROMPT_TEXT=null
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#BLOCK_SIZE=4
#MODEL_PATH="outputs/<PATH_TO_BD3LM_SAVED_MODEL_DIR>"

######### E2D2
# PROMPT_TEXT=null
# BLOCK_SIZE=4
# # MODEL_PATH="kuleshov-group/e2d2-wmt"
# # MODEL_PATH="outputs/wmt_block4_lr3e-4_bsz128_warm1000ba_enc28_dec4_hidden512_inter1536_e2d2_20251120_233640"
# # share_kv, finetune
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block4_lr3e-4_bsz128_warm1000ba_enc28_dec4_hidden1024_inter3072_e2d2_20251222_062114"
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false

######### E2D
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block4_lr3e-4_bsz128_warm1000ba_enc28_dec4_hidden512_inter1536_e2d_20251130_073021"
# share_kv_encoder_gen, finetune
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block4_lr3e-4_bsz128_warm1000ba_enc28_dec4_hidden1024_inter3072_e2d_20251221_191010"
# 28(2)
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block4_lr3e-4_bsz128_warm1000ba_enc28_dec2_hidden1024_inter3072_e2d_20251227_234139"
# Qwen3-1.7B-Base
MODEL_PATH="/data/shared_data/hankun/outputs/wmt_block4_lr1e-5_bsz1_warm100ba_enc28_dec2_hidden2048_inter6144_e2d_20260227_064135"
PORT=29503
PROMPT_TEXT="Translation: "
PROMPT_TEXT=null
BLOCK_SIZE=4
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false

######### LayerSkip
# MODEL_PATH="/data/shared_data/hankun/outputs/wmt-lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260227_073342"
# PORT=29502
# PROMPT_TEXT="Translation: "
# KV_CACHING=true
# BLOCK_SIZE=1
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=true
# ASSISTANT_EARLY_EXIT=8

OUTPUT_DIR="/data/shared_data/hankun/outputs/${MODEL_PATH}/wmt"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=256
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" "posterior"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
MAX_LENGTH=1024
CKPT="best"
USE_EMA=true
NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-T${T}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}"

torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=wmt \
  +task.dataset.target_prompt_text=${PROMPT_TEXT} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation/stopping_criteria@stopping_criteria_list='[max_length_criteria,wmt_stop_string_criteria]' \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null
