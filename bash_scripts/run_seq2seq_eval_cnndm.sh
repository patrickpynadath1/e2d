#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: Uncomment a model and run

######## AR
# untrained AR (27.3/11.3/18.6)
# MODEL_PATH="/data/shared_data/hankun/outputs/cnn_lr1e-5_bsz32_warm0ba_layers28_hidden2048_inter6144_ar_20260308_191936"
# 10k training samples, batch_size = 32, constant LR (35.0/14.4/24.9)
# 30k training samples, batch_size = 32, constant LR (35.9/15.4/25.8)
# 60k training samples, batch_size = 32, cosine LR (38.1/17.2/28.1, 34.18 tokens/s)
# 10k training samples, batch_size = 1, constant LR (eval loss keeps going up, 34.4/14.4/25.0)
# 30k training samples, batch_size = 1, constant LR (eval loss keeps going up)
# 60k training samples, batch_size = 1, cosine LR (35.0/15.0/25.6)
# all, batch_size = 32, cosine LR
# MODEL_PATH="/data/shared_data/hankun/outputs/cnn_lr1e-5_bsz32_warm100ba_layers28_hidden2048_inter6144_ar_20260315_075324"
# PROMPT_TEXT="Summary: "
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# BLOCK_SIZE=1
# LEN_PENALTY=1.0
# REGULATION_START=0
# REPETITION_PENALTY=1.0

########### E2D2
MODEL_PATH="/data/shared_data/hankun/outputs/cnn_block4_lr1e-5_bsz32_warm100ba_enc28_dec4_hidden2048_inter6144_e2d2_20260317_052211"
BLOCK_SIZE=4
PROMPT_TEXT="Summary: "
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false
LEN_PENALTY=1.0
REGULATION_START=0
REPETITION_PENALTY=1.0

########### E2D
# 60k training samples, batch_size = 1, cosine LR (1.81 acc. length, conf_seg=0.7: 54.9 token/s)
# MODEL_PATH="/data/shared_data/hankun/outputs/cnn_block4_lr1e-5_bsz1_warm100ba_enc28_dec2_hidden2048_inter6144_e2d_20260306_231602"
# 120k training samples, batch_size = 128, converged (1.91 acc. length, conf_seg=0.7: 55.8 tokens/s)
# MODEL_PATH="/data/shared_data/hankun/outputs/cnn_block4_lr1e-5_bsz128_warm100ba_enc28_dec2_hidden2048_inter6144_e2d_20260313_060652"
# all, batch_size = 32
# MODEL_PATH="/data/shared_data/hankun/outputs/cnn_block4_lr1e-5_bsz32_warm100ba_enc28_dec2_hidden2048_inter6144_e2d_20260314_003028"
# PROMPT_TEXT=null
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# LEN_PENALTY=1.0
# REGULATION_START=0
# REPETITION_PENALTY=1.0

########### LayerSkip
# MODEL_PATH="/data/shared_data/hankun/outputs/cnn_lr1e-5_bsz32_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_layerskip_20260315_075756"
# PROMPT_TEXT="Summary: "
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# BLOCK_SIZE=1
# LEN_PENALTY=1.0
# REGULATION_START=0
# REPETITION_PENALTY=1.0
# ASSISTANT_EARLY_EXIT=8

OUTPUT_DIR="/data/shared_data/hankun/outputs/${MODEL_PATH}/cnn_dailymail"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=256
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" "posterior"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
MAX_LENGTH=4096
CKPT="best"
USE_EMA=true
EVAL_MAX_SAMPLES=1000
ASSISTANT_EARLY_EXIT=${ASSISTANT_EARLY_EXIT:-0}

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}-assistant_early_exit${ASSISTANT_EARLY_EXIT}-rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}"
PORT=29505
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=cnn_dailymail \
  task.dataset.max_samples=${EVAL_MAX_SAMPLES} \
  +task.dataset.target_prompt_text=${PROMPT_TEXT} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +model_config_overrides.length=${MAX_LENGTH} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-1.7B-Base" \
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
  generation/stopping_criteria@stopping_criteria_list='[max_length_criteria,cnndm_stop_string_criteria]' \
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor,exponential_decay_length_penalty]' \
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY} \
  logits_processor_list.exponential_decay_length_penalty.exponential_decay_length_penalty="[${REGULATION_START},${LEN_PENALTY}]" \
  $(if [ "${ASSISTANT_EARLY_EXIT}" -gt 0 ] 2>/dev/null; then echo "+gen_kwargs.assistant_early_exit=${ASSISTANT_EARLY_EXIT}"; fi)
