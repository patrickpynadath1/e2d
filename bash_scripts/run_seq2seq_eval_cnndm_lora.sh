#!/bin/bash
# Evaluate a frozen-target SEED-LoRA checkpoint on CNN/DailyMail.
# From any directory:
#   MODEL_PATH=/path/to/training/run bash bash_scripts/run_seq2seq_eval_cnndm_lora.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# Match the checkpoint/tokenizer defaults in run_lm_eval_harness_lora.sh.
MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_block8_1.7b_dec2_rank512_lr6e-4_bsz32_20260926_204506}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-1.7B}
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_LENGTH=${MAX_LENGTH:-4096}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1000}
BLOCK_SIZE=${BLOCK_SIZE:-4}
CONF_SEG=${CONF_SEG:-true}
TRACK_ACC_RATE=${TRACK_ACC_RATE:-false}
TREE_ATTN=${TREE_ATTN:-true}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
PROMPT_TEXT=${PROMPT_TEXT:-null}
IS_INSTRUCTION_MODEL=${IS_INSTRUCTION_MODEL:-true}
ENABLE_THINKING_FOR_CNNDM=${ENABLE_THINKING_FOR_CNNDM:-false}
STRIP_THINKING_FOR_CNNDM=${STRIP_THINKING_FOR_CNNDM:-${IS_INSTRUCTION_MODEL}}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
LEN_PENALTY=${LEN_PENALTY:-1.0}
REGULATION_START=${REGULATION_START:-0}
PORT=${PORT:-29504}
if [[ "${CONF_SEG}" == true && "${TRACK_ACC_RATE}" == true ]]; then
  echo "Set CONF_SEG=false when using TRACK_ACC_RATE=true." >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-1}
else
  IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-${#DEVICES[@]}}
fi

OUTPUT_PATH=${OUTPUT_PATH:-"${MODEL_PATH}/cnn_dailymail/seed_lora_${CKPT}_L${MAX_NEW_TOKENS}_block${BLOCK_SIZE}_conf${CONF_SEG}_adaptive${TRACK_ACC_RATE}_tree${TREE_ATTN}_sample${DO_SAMPLE}_instruction${IS_INSTRUCTION_MODEL}_rep${REPETITION_PENALTY}_len${LEN_PENALTY}_reg${REGULATION_START}"}
mkdir -p "${OUTPUT_PATH}"

# Keep run_seq2seq_eval_cnndm.sh's distributed evaluation, ROUGE metrics,
# generated-sample reports, stopping criteria, and optional length penalties.
# Use the instruction model's non-thinking chat format and raw LoRA weights.
torchrun --nproc_per_node "${NUM_VISIBLE_DEVICES}" --master_port="${PORT}" scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=cnn_dailymail \
  task.dataset.max_samples="${EVAL_MAX_SAMPLES}" \
  +task.dataset.target_prompt_text="${PROMPT_TEXT}" \
  +task.dataset.is_instruction_model="${IS_INSTRUCTION_MODEL}" \
  +task.dataset.use_chat_template="${IS_INSTRUCTION_MODEL}" \
  +task.dataset.enable_thinking="${ENABLE_THINKING_FOR_CNNDM}" \
  pretrained_model_name_or_path="${MODEL_PATH}" \
  +model_config_overrides.length="${MAX_LENGTH}" \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=false \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  output_path="${OUTPUT_PATH}" \
  generated_samples_output_path="${OUTPUT_PATH}" \
  +strip_thinking="${STRIP_THINKING_FOR_CNNDM}" \
  max_length="${MAX_LENGTH}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  block_size="${BLOCK_SIZE}" \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  generation_config.num_steps="${BLOCK_SIZE}" \
  generation_config.use_cache=true \
  generation_config.align_inputs_to_blocks=false \
  'generation/stopping_criteria@stopping_criteria_list=[max_length_criteria,cnndm_stop_string_criteria]' \
  'generation/logits_processor@logits_processor_list=[repetition_penalty_logits_processor,exponential_decay_length_penalty]' \
  logits_processor_list.repetition_penalty_logits_processor.penalty="${REPETITION_PENALTY}" \
  logits_processor_list.exponential_decay_length_penalty.exponential_decay_length_penalty="[${REGULATION_START},${LEN_PENALTY}]" \
  +gen_kwargs.conf_seg="${CONF_SEG}" \
  +gen_kwargs.track_acc_rate="${TRACK_ACC_RATE}" \
  +gen_kwargs.tree_attn="${TREE_ATTN}" \
  "$@"
