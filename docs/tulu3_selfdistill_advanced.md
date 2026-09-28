# Tulu 3 self-distillation on 1–8 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,2,4,6 PER_DEVICE_BATCH_SIZE=32 \
  bash bash_scripts/generate_tulu3_selfdistill_advanced.sh
```

This runs generation and preprocessing only. Each visible GPU gets one complete
model replica. Whole batches are distributed to available workers, then written
in source order. `DISTILL_MAX_SAMPLES` is a global generated-row limit; `0` uses
all source conversations. At most two batches per GPU are queued, so reaching
the limit may discard a small amount of speculative generation.

Defaults match Stage 1 of `bash_scripts/run_train_e2d_tulu3_distill.sh`:

| Setting | Default |
| --- | --- |
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-1.7B` |
| `DISTILL_SOURCE_NAME` | `allenai/tulu-3-sft-mixture` |
| `DISTILL_SOURCE_SPLITS` | `train` |
| `PER_DEVICE_BATCH_SIZE` | `32` (also accepts `BATCH_GEN_SIZE`) |
| `MAX_SEQ_LEN` | `4096` |
| `GEN_MAX_SEQ_LEN` | `4096` |
| `PROMPT_MAX_SEQ_LEN` | `1024` |
| `EVAL_RATIO`, `SPLIT_SEED` | `0.01`, `42` |
| `DTYPE` | `bfloat16` |

The full conversation prefix before the last reference assistant response is
kept, including previous assistant turns. The final response is replaced using
greedy decoding with thinking disabled. Prompts exceeding `PROMPT_MAX_SEQ_LEN`
are filtered within their original batches. Generation uses left padding and
`GEN_MAX_SEQ_LEN - padded_input_length` new tokens. Training independently
re-tokenizes and truncates the prompt and completion to `MAX_SEQ_LEN // 2` tokens.

With the same per-device batch size, batch composition and generation budgets
match a fresh reference run, regardless of GPU count. GPU kernels, hardware,
library versions, and changing batch size can still change generated tokens.

The default output directory is:

```text
/data/shared_data/hankun/datasets/tulu3_qwen3_1p7b_selfdistill_advanced_bsz32_maxlen4096
```

Override it with `DISTILL_DATA_ROOT`. Outputs use the reference filename stem
`tulu3_qwen3_1p7b_thinking` (overridable with `DISTILL_FILE_STEM`):

```text
tulu3_qwen3_1p7b_thinking.jsonl
tulu3_qwen3_1p7b_thinking_train.jsonl
tulu3_qwen3_1p7b_thinking_eval.jsonl
train_preprocessed/
eval_preprocessed/
selfdistill_metadata.json
```

JSONLs contain exactly `prompt`, `completion`, and `source_split`. Preprocessed
datasets contain `input_ids`, `attention_mask`, and `context_mask`. The seeded
train/eval split uses the same order and rounding as the reference script.
Despite the historical filename containing `thinking`, thinking is disabled.

Worker logs live under `${DISTILL_DATA_ROOT}.shards/shard_*.log`. Checkpoints in
`selfdistill_metadata.json` record committed JSONL bytes and the next batch.
Rerun the same command to resume; an interrupted batch may be generated again.
Completed batches are retained, including empty batches and duplicate prompts.
A worker failure stops generation and prevents preprocessing incomplete output.
Changing generation settings requires a new output directory. Existing outputs
from other generators are not adopted or overwritten.

Extra Python arguments may follow the shell script, for example
`--num-shards 2`, `--attn-implementation sdpa`, or `--local-files-only`.
`PROGRESS_EVERY_BATCHES`, `EMPTY_CACHE_EVERY_BATCHES`, and `PYTHON` are also
supported. The default attention implementation follows Transformers, as in the
reference driver.

To train on the generated dataset, pass the same directory to the existing
training script (its current training length is 4096):

```bash
DISTILL_DATA_ROOT=/data/shared_data/hankun/datasets/tulu3_qwen3_1p7b_selfdistill_advanced_bsz32_maxlen4096 \
  bash bash_scripts/run_train_e2d_tulu3_distill.sh
```

It detects both preprocessed directories and skips its own generation stage.
