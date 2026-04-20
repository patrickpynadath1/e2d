#!/bin/bash

set -euo pipefail

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# -----------------------------
# Stage 1: Self-distillation
# -----------------------------
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B
MAX_SEQ_LEN=1024
GEN_MAX_SEQ_LEN=${GEN_MAX_SEQ_LEN:-1536}

ULTRACHAT_DATASET_NAME=HuggingFaceH4/ultrachat_200k
ULTRACHAT_SPLITS="train_sft train_gen"

DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/ultrachat_qwen3_1p7b_thinking_maxlen${MAX_SEQ_LEN}}
DISTILL_RAW_FILE="${DISTILL_DATA_ROOT}/ultrachat_qwen3_1p7b_thinking.jsonl"
DISTILL_TRAIN_JSONL="${DISTILL_DATA_ROOT}/ultrachat_qwen3_1p7b_thinking_train.jsonl"
DISTILL_EVAL_JSONL="${DISTILL_DATA_ROOT}/ultrachat_qwen3_1p7b_thinking_eval.jsonl"
DISTILL_TRAIN_PATH="${DISTILL_DATA_ROOT}/train_preprocessed"
DISTILL_EVAL_PATH="${DISTILL_DATA_ROOT}/eval_preprocessed"

# Optional controls
DISTILL_MAX_SAMPLES=${DISTILL_MAX_SAMPLES:-0}  # 0 means all examples from train_sft + train_gen
EVAL_RATIO=${EVAL_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-42}
FORCE_REGENERATE=${FORCE_REGENERATE:-true}
BATCH_GEN_SIZE=${BATCH_GEN_SIZE:-64}
PROGRESS_EVERY_BATCHES=${PROGRESS_EVERY_BATCHES:-1}

if [[ "${FORCE_REGENERATE}" == "true" || ! -d "${DISTILL_TRAIN_PATH}" || ! -d "${DISTILL_EVAL_PATH}" ]]; then
  echo "[Self-Distill] Building distilled UltraChat dataset at ${DISTILL_DATA_ROOT}"
  mkdir -p "${DISTILL_DATA_ROOT}"

    export MODEL_NAME_OR_PATH
    export ULTRACHAT_DATASET_NAME
    export ULTRACHAT_SPLITS
    export MAX_SEQ_LEN
    export GEN_MAX_SEQ_LEN
    export DISTILL_MAX_SAMPLES
    export EVAL_RATIO
    export SPLIT_SEED
    export DISTILL_RAW_FILE
    export DISTILL_TRAIN_JSONL
    export DISTILL_EVAL_JSONL
    export DISTILL_TRAIN_PATH
    export DISTILL_EVAL_PATH
    export BATCH_GEN_SIZE
    export PROGRESS_EVERY_BATCHES

  python - <<'PY'
import json
import os
import random
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = os.environ["MODEL_NAME_OR_PATH"]
dataset_name = os.environ["ULTRACHAT_DATASET_NAME"]
splits = os.environ["ULTRACHAT_SPLITS"].split()
max_seq_len = int(os.environ["MAX_SEQ_LEN"])
gen_max_seq_len = int(os.environ.get("GEN_MAX_SEQ_LEN", "2048"))
distill_max_samples = int(os.environ.get("DISTILL_MAX_SAMPLES", "0"))
eval_ratio = float(os.environ.get("EVAL_RATIO", "0.01"))
split_seed = int(os.environ.get("SPLIT_SEED", "42"))
batch_gen_size = int(os.environ.get("BATCH_GEN_SIZE", "8"))
progress_every_batches = int(os.environ.get("PROGRESS_EVERY_BATCHES", "20"))

if gen_max_seq_len < max_seq_len:
    raise ValueError(
        f"GEN_MAX_SEQ_LEN ({gen_max_seq_len}) must be >= MAX_SEQ_LEN ({max_seq_len})."
    )

distill_raw_file = os.environ["DISTILL_RAW_FILE"]
distill_train_jsonl = os.environ["DISTILL_TRAIN_JSONL"]
distill_eval_jsonl = os.environ["DISTILL_EVAL_JSONL"]
distill_train_path = os.environ["DISTILL_TRAIN_PATH"]
distill_eval_path = os.environ["DISTILL_EVAL_PATH"]

os.makedirs(os.path.dirname(distill_raw_file), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
if not torch.cuda.is_available():
    model = model.to(device)
model.eval()


def apply_chat_template_with_thinking(prompt_text: str) -> str:
    user_messages: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]

    return tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_pair_for_e2d(prompt: str, completion: str, tokenizer_obj: Any, max_len: int) -> dict[str, list[int]]:
    source = prompt
    target = completion

    # Match TuluV2SFTDataset truncation behavior used elsewhere in this repo.
    tokenized = tokenizer_obj.batch_encode_plus(
        [source, target],
        max_length=max_len // 2,
        padding=False,
        add_special_tokens=False,
        truncation=True,
    )

    source_ids = tokenized["input_ids"][0]
    source_mask = tokenized["attention_mask"][0]
    target_ids = tokenized["input_ids"][1]
    target_mask = tokenized["attention_mask"][1]

    return {
        "input_ids": source_ids + target_ids,
        "attention_mask": source_mask + target_mask,
        "context_mask": source_mask + ([0] * len(target_ids)),
    }


rows: list[dict[str, str]] = []
num_seen = 0


def generate_batch(prompt_texts: list[str], split_name: str) -> bool:
    """Generate completions for a batch and append prompt/completion rows.

    Returns True when distill_max_samples is reached and caller should stop.
    """
    global num_seen
    if not prompt_texts:
        return False

    chat_prompts = [apply_chat_template_with_thinking(p) for p in prompt_texts]

    model_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=gen_max_seq_len - 1,
        padding=True,
    )
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    # Batched generation uses one max_new_tokens across the whole batch.
    # Using gen_max_seq_len avoids short generations caused by padding-to-longest
    # when max_seq_len is small for downstream training.
    input_seq_len = int(model_inputs["input_ids"].shape[-1])
    max_new_tokens = max(1, gen_max_seq_len - input_seq_len)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for i, prompt in enumerate(prompt_texts):
        gen_ids = output_ids[i, input_seq_len:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=False)

        if not completion:
            continue

        row = {
            "prompt": apply_chat_template_with_thinking(prompt),
            "completion": completion,
            "source_split": split_name,
        }
        rows.append(row)
        f_raw.write(json.dumps(row, ensure_ascii=False) + "\n")

        num_seen += 1
        if distill_max_samples > 0 and num_seen >= distill_max_samples:
            return True

    return False

with open(distill_raw_file, "w", encoding="utf-8") as f_raw:
    for split in splits:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        split_total = len(ds)
        split_prompt_seen = 0
        split_batch_count = 0
        split_generated = 0
        print(
            f"[Self-Distill][{split}] start: total_rows={split_total}, "
            f"batch_size={batch_gen_size}, max_samples={distill_max_samples}, "
            f"gen_max_seq_len={gen_max_seq_len}, train_max_seq_len={max_seq_len}"
        )
        prompt_buffer: list[str] = []
        for ex in ds:
            prompt = ex.get("prompt", None)
            if not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if not prompt:
                continue

            prompt_buffer.append(prompt)
            if len(prompt_buffer) >= batch_gen_size:
                prompts_in_batch = len(prompt_buffer)
                before_seen = num_seen
                should_stop = generate_batch(prompt_buffer, split)
                prompt_buffer = []

                split_prompt_seen += prompts_in_batch
                split_batch_count += 1
                split_generated += (num_seen - before_seen)

                if (
                    progress_every_batches > 0
                    and split_batch_count % progress_every_batches == 0
                ) or should_stop:
                    print(
                        f"[Self-Distill][{split}] "
                        f"batches={split_batch_count} "
                        f"prompts_processed={split_prompt_seen}/{split_total} "
                        f"pairs_written={split_generated} "
                        f"global_pairs={num_seen}"
                    )

                if should_stop:
                    break

        if prompt_buffer:
            prompts_in_batch = len(prompt_buffer)
            before_seen = num_seen
            should_stop = generate_batch(prompt_buffer, split)
            prompt_buffer = []

            split_prompt_seen += prompts_in_batch
            split_batch_count += 1
            split_generated += (num_seen - before_seen)

            print(
                f"[Self-Distill][{split}] "
                f"batches={split_batch_count} "
                f"prompts_processed={split_prompt_seen}/{split_total} "
                f"pairs_written={split_generated} "
                f"global_pairs={num_seen}"
            )

            if should_stop:
                break

        print(
            f"[Self-Distill][{split}] done: "
            f"batches={split_batch_count}, prompts_processed={split_prompt_seen}/{split_total}, "
            f"pairs_written={split_generated}, global_pairs={num_seen}"
        )

        if distill_max_samples > 0 and num_seen >= distill_max_samples:
            break

if len(rows) < 2:
    raise RuntimeError(
        "Distillation produced fewer than 2 rows; cannot create train/eval split."
    )

random.seed(split_seed)
random.shuffle(rows)
eval_size = max(1, int(len(rows) * eval_ratio))
eval_size = min(eval_size, len(rows) - 1)

eval_rows = rows[:eval_size]
train_rows = rows[eval_size:]

with open(distill_train_jsonl, "w", encoding="utf-8") as f_train:
    for row in train_rows:
        f_train.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(distill_eval_jsonl, "w", encoding="utf-8") as f_eval:
    for row in eval_rows:
        f_eval.write(json.dumps(row, ensure_ascii=False) + "\n")

train_tok = [
    encode_pair_for_e2d(r["prompt"], r["completion"], tokenizer, max_seq_len)
    for r in train_rows
]
eval_tok = [
    encode_pair_for_e2d(r["prompt"], r["completion"], tokenizer, max_seq_len)
    for r in eval_rows
]

train_ds = Dataset.from_list(train_tok)
eval_ds = Dataset.from_list(eval_tok)

if os.path.isdir(distill_train_path):
    import shutil

    shutil.rmtree(distill_train_path)
if os.path.isdir(distill_eval_path):
    import shutil

    shutil.rmtree(distill_eval_path)

train_ds.save_to_disk(distill_train_path)
eval_ds.save_to_disk(distill_eval_path)

print(f"[Self-Distill] Saved full distilled JSONL: {distill_raw_file}")
print(f"[Self-Distill] Saved train JSONL: {distill_train_jsonl}")
print(f"[Self-Distill] Saved eval JSONL: {distill_eval_jsonl}")
print(f"[Self-Distill] Saved train preprocessed dataset: {distill_train_path}")
print(f"[Self-Distill] Saved eval preprocessed dataset: {distill_eval_path}")
print(f"[Self-Distill] Rows: total={len(rows)}, train={len(train_rows)}, eval={len(eval_rows)}")
PY
else
  echo "[Self-Distill] Reusing existing preprocessed datasets:"
  echo "  train: ${DISTILL_TRAIN_PATH}"
  echo "  eval : ${DISTILL_EVAL_PATH}"
fi

# -----------------------------
# Stage 2: Train E2D (1 epoch)
# -----------------------------

# Model arch (same structure as run_train_e2d_gsm8k.sh)
BLOCK_SIZE=8
EVAL_BLOCK_SIZE=8
N_ENCODER_LAYERS=28
ENCODER_TOP_LAYERS=false
N_DECODER_LAYERS=2
DECODER_TOP_LAYERS=true
REINIT_ENCODER=false
REINIT_DECODER=false
TIE_WEIGHTS=true
FREEZE_ENCODER=false
ENCODER_CAUSAL_MASK=false
NULLIFY_SELF_ATTN=false

# Hyperparameters (same setup style as run_train_e2d_gsm8k.sh, but 1 epoch)
LR=1e-5
WARMUP_DURATION="100ba"
ALPHA_F=0.5
DECODER_LOSS_LAMBDA=1.0
BATCH_SIZE=1
MAX_DURATION="3ep"
PRECISION="amp_bf16"

TRAIN_ON_CONTEXT=false
TRAIN_ON_AR=false
AR_CHECKPOINT_PATH=""

TAG="e2d_ultrachat"
if [ "${ENCODER_TOP_LAYERS}" == "true" ]; then
  ENC_LAYERS="TOPenc${N_ENCODER_LAYERS}"
else
  ENC_LAYERS="enc${N_ENCODER_LAYERS}"
fi
if [ "${DECODER_TOP_LAYERS}" == "true" ]; then
  DEC_LAYERS="TOPdec${N_DECODER_LAYERS}"
else
  DEC_LAYERS="dec${N_DECODER_LAYERS}"
fi

# get time stamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RUN_NAME=ultrachat_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${PRECISION}_${ENC_LAYERS}_${DEC_LAYERS}_${TAG}_${TIMESTAMP}
if [ "${TIE_WEIGHTS}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_tie-weights"
fi
if [ "${ENCODER_CAUSAL_MASK}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_encoder-causal-mask"
fi
if [ "${FREEZE_ENCODER}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_freeze-enc"
fi

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_VISIBLE_DEVICES=1
else
  NUM_VISIBLE_DEVICES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
fi

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
    pretrained_model_name_or_path=${MODEL_NAME_OR_PATH} \
    dataset@train_dataset=ultrachat_distill_train \
    dataset@eval_dataset=ultrachat_distill_eval \
  train_dataset.dataset_path=${DISTILL_TRAIN_PATH} \
  eval_dataset.dataset_path=${DISTILL_EVAL_PATH} \
  composer.optimizer.lr=${LR} \
  composer.trainer.precision=${PRECISION} \
  composer.trainer.eval_interval="1000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  model=e2d \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=${MAX_SEQ_LEN} \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen \
  model.config.backbone_config.use_encoder_causal_mask=${ENCODER_CAUSAL_MASK} \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=${TIE_WEIGHTS} \
  model.config.backbone_config.freeze_encoder=${FREEZE_ENCODER} \
  model.config.backbone_config.reinit_decoder=${REINIT_DECODER} \
  model.config.backbone_config.reinit_encoder=${REINIT_ENCODER} \
  model.config.backbone_config.keep_top_decoder_layers=${DECODER_TOP_LAYERS} \
  model.config.backbone_config.keep_top_encoder_layers=${ENCODER_TOP_LAYERS} \
  model.config.backbone_config.use_gradient_checkpointing=false \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=/data/shared_data/hankun/outputs/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  composer.callbacks.save_best_checkpointing.save_local=false \
  eval_dataloader.batch_size=1 \
  model.config.train_on_context=${TRAIN_ON_CONTEXT} \
  model.config.decoder_loss_lambda=${DECODER_LOSS_LAMBDA} \
  model.config.backbone_config.train_on_ar=${TRAIN_ON_AR} \
  model.config.backbone_config.ar_checkpoint_path=${AR_CHECKPOINT_PATH} \
  +model.config.nullify_self_attn=${NULLIFY_SELF_ATTN} \
