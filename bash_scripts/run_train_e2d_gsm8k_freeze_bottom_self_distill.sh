#!/bin/bash

set -euo pipefail

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# -----------------------------
# Stage 1: GSM8K self-distillation
# -----------------------------
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B
MAX_SEQ_LEN=1024
GEN_MAX_SEQ_LEN=${GEN_MAX_SEQ_LEN:-1536}
PROMPT_MAX_SEQ_LEN=${PROMPT_MAX_SEQ_LEN:-512}

GSM8K_DATASET_PATH=openai/gsm8k
GSM8K_CONFIG_NAME=main
GSM8K_SPLIT=train

DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/gsm8k_qwen3_1p7b_selfdistill_maxlen${MAX_SEQ_LEN}}
DISTILL_RAW_FILE="${DISTILL_DATA_ROOT}/gsm8k_qwen3_1p7b_selfdistill.jsonl"
DISTILL_TRAIN_JSONL="${DISTILL_DATA_ROOT}/gsm8k_qwen3_1p7b_selfdistill_train.jsonl"
DISTILL_EVAL_JSONL="${DISTILL_DATA_ROOT}/gsm8k_qwen3_1p7b_selfdistill_eval.jsonl"
DISTILL_TRAIN_PATH="${DISTILL_DATA_ROOT}/train_preprocessed"
DISTILL_EVAL_PATH="${DISTILL_DATA_ROOT}/eval_preprocessed"

# Optional controls
DISTILL_MAX_SAMPLES=${DISTILL_MAX_SAMPLES:-0}  # 0 means use all GSM8K train examples
EVAL_RATIO=${EVAL_RATIO:-0.02}
SPLIT_SEED=${SPLIT_SEED:-42}
FORCE_REGENERATE=${FORCE_REGENERATE:-false}
BATCH_GEN_SIZE=${BATCH_GEN_SIZE:-64}
PROGRESS_EVERY_BATCHES=${PROGRESS_EVERY_BATCHES:-1}
EMPTY_CACHE_EVERY_BATCHES=${EMPTY_CACHE_EVERY_BATCHES:-20}

if [[ "${FORCE_REGENERATE}" == "true" || ! -d "${DISTILL_TRAIN_PATH}" || ! -d "${DISTILL_EVAL_PATH}" ]]; then
  echo "[Self-Distill] Building distilled GSM8K dataset at ${DISTILL_DATA_ROOT}"
  mkdir -p "${DISTILL_DATA_ROOT}"

  export MODEL_NAME_OR_PATH
  export MAX_SEQ_LEN
  export GEN_MAX_SEQ_LEN
  export PROMPT_MAX_SEQ_LEN
  export GSM8K_DATASET_PATH
  export GSM8K_CONFIG_NAME
  export GSM8K_SPLIT
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
  export FORCE_REGENERATE
  export EMPTY_CACHE_EVERY_BATCHES

  python - <<'PY'
import json
import os
import random
from collections import Counter
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = os.environ["MODEL_NAME_OR_PATH"]
max_seq_len = int(os.environ["MAX_SEQ_LEN"])
gen_max_seq_len = int(os.environ.get("GEN_MAX_SEQ_LEN", "1024"))
prompt_max_seq_len = int(os.environ.get("PROMPT_MAX_SEQ_LEN", str(max_seq_len)))
dataset_path = os.environ.get("GSM8K_DATASET_PATH", "openai/gsm8k")
config_name = os.environ.get("GSM8K_CONFIG_NAME", "main")
split_name = os.environ.get("GSM8K_SPLIT", "train")
distill_max_samples = int(os.environ.get("DISTILL_MAX_SAMPLES", "0"))
eval_ratio = float(os.environ.get("EVAL_RATIO", "0.02"))
split_seed = int(os.environ.get("SPLIT_SEED", "42"))
batch_gen_size = int(os.environ.get("BATCH_GEN_SIZE", "8"))
progress_every_batches = int(os.environ.get("PROGRESS_EVERY_BATCHES", "20"))
force_regenerate = os.environ.get("FORCE_REGENERATE", "false").lower() == "true"
empty_cache_every_batches = int(os.environ.get("EMPTY_CACHE_EVERY_BATCHES", "1"))

if gen_max_seq_len < max_seq_len:
    raise ValueError(
        f"GEN_MAX_SEQ_LEN ({gen_max_seq_len}) must be >= MAX_SEQ_LEN ({max_seq_len})."
    )
if prompt_max_seq_len <= 0:
    raise ValueError(f"PROMPT_MAX_SEQ_LEN ({prompt_max_seq_len}) must be > 0.")
if empty_cache_every_batches <= 0:
    raise ValueError(
        f"EMPTY_CACHE_EVERY_BATCHES ({empty_cache_every_batches}) must be > 0."
    )

effective_generation_input_limit = gen_max_seq_len - 1
if prompt_max_seq_len > effective_generation_input_limit:
    print(
        "[Self-Distill] Warning: "
        f"PROMPT_MAX_SEQ_LEN ({prompt_max_seq_len}) is greater than the effective "
        f"generation input limit ({effective_generation_input_limit}). "
        "Prompts near this limit may still be truncated before generation."
    )

distill_raw_file = os.environ["DISTILL_RAW_FILE"]
distill_train_jsonl = os.environ["DISTILL_TRAIN_JSONL"]
distill_eval_jsonl = os.environ["DISTILL_EVAL_JSONL"]
distill_train_path = os.environ["DISTILL_TRAIN_PATH"]
distill_eval_path = os.environ["DISTILL_EVAL_PATH"]

os.makedirs(os.path.dirname(distill_raw_file), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left",
)
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


def apply_chat_template(prompt_text: str) -> str:
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_pair_for_e2d(
    prompt: str,
    completion: str,
    tokenizer_obj: Any,
    max_len: int,
) -> dict[str, list[int]]:
    tokenized = tokenizer_obj.batch_encode_plus(
        [prompt, completion],
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


def strip_batched_generation_padding(gen_ids: torch.Tensor) -> list[int]:
    token_ids = gen_ids.tolist()
    if not token_ids:
        return token_ids

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        return token_ids

    if eos_id is None or pad_id != eos_id:
        return [tid for tid in token_ids if tid != pad_id]

    if eos_id in token_ids:
        first_eos_idx = token_ids.index(eos_id)
        return token_ids[: first_eos_idx + 1]

    return token_ids


def load_existing_rows_for_resume(path: str) -> tuple[list[dict[str, str]], Counter[str]]:
    if force_regenerate or not os.path.isfile(path):
        return [], Counter()

    loaded_rows: list[dict[str, str]] = []
    prompt_counts: Counter[str] = Counter()
    last_valid_offset = 0
    saw_invalid_tail = False

    with open(path, "r", encoding="utf-8") as f_in:
        while True:
            line = f_in.readline()
            if not line:
                break

            stripped = line.strip()
            if not stripped:
                last_valid_offset = f_in.tell()
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                saw_invalid_tail = True
                break

            if isinstance(row, dict):
                loaded_rows.append(row)
                prompt = row.get("prompt")
                if isinstance(prompt, str):
                    prompt_counts[prompt] += 1

            last_valid_offset = f_in.tell()

    if saw_invalid_tail:
        with open(path, "rb+") as f_bin:
            f_bin.truncate(last_valid_offset)
        print(
            "[Self-Distill] Warning: detected truncated JSONL tail; "
            f"truncated to {last_valid_offset} bytes before resuming."
        )

    return loaded_rows, prompt_counts


rows, resume_prompt_counts = load_existing_rows_for_resume(distill_raw_file)
num_seen = len(rows)
num_filtered_by_prompt_len = 0
generate_batch_counter = 0
if num_seen > 0:
    print(
        f"[Self-Distill] Resume mode: loaded {num_seen} existing rows from "
        f"{distill_raw_file}"
    )


def generate_batch(prompt_items: list[str]) -> tuple[bool, int, int]:
    global num_seen, num_filtered_by_prompt_len, generate_batch_counter
    if not prompt_items:
        return False, 0, 0

    prompt_lengths = [
        len(ids)
        for ids in tokenizer(
            prompt_items,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
    ]

    keep_indices = [
        i for i, prompt_len in enumerate(prompt_lengths) if prompt_len <= prompt_max_seq_len
    ]
    filtered_in_batch = len(prompt_items) - len(keep_indices)
    if filtered_in_batch > 0:
        num_filtered_by_prompt_len += filtered_in_batch

    if not keep_indices:
        return False, filtered_in_batch, 0

    filtered_prompts = [prompt_items[i] for i in keep_indices]
    model_inputs = tokenizer(
        filtered_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=gen_max_seq_len - 1,
        padding=True,
    )
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

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

    generated_count = 0
    should_stop = False
    for i, prompt in enumerate(filtered_prompts):
        gen_ids = output_ids[i, input_seq_len:]
        gen_ids = strip_batched_generation_padding(gen_ids)
        if not gen_ids:
            continue

        completion = tokenizer.decode(gen_ids, skip_special_tokens=False)
        if not completion:
            continue

        row = {
            "prompt": prompt,
            "completion": completion,
        }
        rows.append(row)
        f_raw.write(json.dumps(row, ensure_ascii=False) + "\n")

        num_seen += 1
        generated_count += 1
        if distill_max_samples > 0 and num_seen >= distill_max_samples:
            should_stop = True
            break

    del output_ids
    del model_inputs
    if torch.cuda.is_available():
        generate_batch_counter += 1
        if generate_batch_counter % empty_cache_every_batches == 0:
            torch.cuda.empty_cache()

    return should_stop, filtered_in_batch, generated_count


raw_file_mode = "w" if force_regenerate else "a"
with open(distill_raw_file, raw_file_mode, encoding="utf-8") as f_raw:
    if raw_file_mode == "a" and os.path.getsize(distill_raw_file) > 0:
        with open(distill_raw_file, "rb") as f_check:
            f_check.seek(-1, os.SEEK_END)
            if f_check.read(1) != b"\n":
                f_raw.write("\n")

    ds = load_dataset(dataset_path, config_name, split=split_name, trust_remote_code=True)
    total_rows = len(ds)
    prompt_buffer: list[str] = []
    prompt_seen = 0
    batch_count = 0
    generated_total = 0
    filtered_total = 0
    resume_skipped = 0

    print(
        f"[Self-Distill][{split_name}] start: total_rows={total_rows}, "
        f"batch_size={batch_gen_size}, max_samples={distill_max_samples}, "
        f"gen_max_seq_len={gen_max_seq_len}, train_max_seq_len={max_seq_len}, "
        f"prompt_max_seq_len={prompt_max_seq_len}, "
        f"resume_pairs_to_skip={sum(resume_prompt_counts.values())}"
    )

    for ex in ds:
        question = ex.get("question", None)
        if not isinstance(question, str):
            continue

        question = question.strip()
        if not question:
            continue

        prompt = (
            "Please reason step by step, and put your final answer within "
            "$\\boxed{}$. "
            + question
        )
        prompt = apply_chat_template(prompt)

        if resume_prompt_counts.get(prompt, 0) > 0:
            resume_prompt_counts[prompt] -= 1
            resume_skipped += 1
            continue

        prompt_buffer.append(prompt)
        if len(prompt_buffer) >= batch_gen_size:
            prompts_in_batch = len(prompt_buffer)
            batch_count += 1
            should_stop, filtered_in_batch, generated_in_batch = generate_batch(prompt_buffer)
            prompt_buffer = []

            prompt_seen += prompts_in_batch
            filtered_total += filtered_in_batch
            generated_total += generated_in_batch

            if (
                progress_every_batches > 0
                and batch_count % progress_every_batches == 0
            ) or should_stop:
                print(
                    f"[Self-Distill][{split_name}] "
                    f"batches={batch_count} "
                    f"prompts_processed={prompt_seen}/{total_rows} "
                    f"resume_skipped={resume_skipped} "
                    f"filtered_by_prompt_len={filtered_total} "
                    f"pairs_written={generated_total} "
                    f"global_pairs={num_seen}"
                )

            if should_stop:
                break

    if prompt_buffer and (distill_max_samples == 0 or num_seen < distill_max_samples):
        prompts_in_batch = len(prompt_buffer)
        batch_count += 1
        should_stop, filtered_in_batch, generated_in_batch = generate_batch(prompt_buffer)
        prompt_buffer = []

        prompt_seen += prompts_in_batch
        filtered_total += filtered_in_batch
        generated_total += generated_in_batch

        print(
            f"[Self-Distill][{split_name}] "
            f"batches={batch_count} "
            f"prompts_processed={prompt_seen}/{total_rows} "
            f"resume_skipped={resume_skipped} "
            f"filtered_by_prompt_len={filtered_total} "
            f"pairs_written={generated_total} "
            f"global_pairs={num_seen}"
        )

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
print(
    f"[Self-Distill] Rows: total={len(rows)}, train={len(train_rows)}, eval={len(eval_rows)}, "
    f"filtered_by_prompt_len={num_filtered_by_prompt_len}"
)
PY
else
  echo "[Self-Distill] Reusing existing preprocessed datasets:"
  echo "  train: ${DISTILL_TRAIN_PATH}"
  echo "  eval : ${DISTILL_EVAL_PATH}"
fi

# -----------------------------
# Stage 2: Train E2D with frozen-bottom backbone
# -----------------------------

# Model arch
BLOCK_SIZE=4
EVAL_BLOCK_SIZE=4
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

# Hyperparameters
LR=1e-5
WARMUP_DURATION="100ba"
ALPHA_F=0.5
DECODER_LOSS_LAMBDA=1.0
BATCH_SIZE=1
MAX_DURATION=${MAX_DURATION:-5ep}
PRECISION="amp_bf16"

TRAIN_ON_CONTEXT=false

TAG="e2d_gsm8k_self_distill"
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

RUN_NAME=gsm8k-self-distill-${TIMESTAMP}

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_VISIBLE_DEVICES=1
else
  NUM_VISIBLE_DEVICES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen_frozen_bottom \
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
    +model.config.nullify_self_attn=${NULLIFY_SELF_ATTN}