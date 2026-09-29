# Tulu 3 self-distillation on 1–8 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,2,4,6 PER_DEVICE_BATCH_SIZE=32 \
  bash bash_scripts/generate_tulu3_selfdistill_advanced.sh
```

This runs generation and preprocessing only. Each visible GPU gets one complete
vLLM model replica (tensor parallel size 1). Whole batches are distributed to
available workers, then written in source order. `DISTILL_MAX_SAMPLES` is a global generated-row limit; `0` uses
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
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` |

The full conversation prefix before the last reference assistant response is
kept, including previous assistant turns. The final response is replaced using
greedy decoding with thinking disabled. Prompts exceeding `PROMPT_MAX_SEQ_LEN`
are filtered within their original batches. vLLM receives the same prompt token
IDs without padding. Every prompt in a batch retains the original shared budget
of `GEN_MAX_SEQ_LEN - longest_input_length` new tokens. Completion token IDs
are decoded with the original tokenizer, retaining EOS and other special tokens.
Training independently re-tokenizes and truncates the prompt and completion to `MAX_SEQ_LEN // 2` tokens.

With the same per-device batch size, batch composition and generation budgets
match a fresh reference run, regardless of GPU count. The JSONL schema, row
ordering, split logic, and preprocessing are unchanged. Generated answer text is
not guaranteed to be identical to Transformers: different inference kernels,
hardware, library versions, and batch sizes can change greedy token choices.

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
Changing generation settings requires a new output directory. The metadata now
records the vLLM backend and version; use a fresh `DISTILL_DATA_ROOT` when
switching from an existing Transformers run to avoid mixing backends. Existing
outputs from other generators are not adopted or overwritten.

Worker exceptions are printed in both the worker log and the main log with the
original traceback. A failed worker does not retry initialization for queued
batches. If engine startup fails around `torch.compile`, try vLLM's eager mode:

```bash
DISTILL_DATA_ROOT=/path/to/new/output_eager \
  nohup bash bash_scripts/generate_tulu3_selfdistill_advanced.sh --enforce-eager \
  > gen-vllm-eager.log 2>&1 &
```

This disables `torch.compile` and CUDA graphs, while retaining vLLM inference,
GPU replicas, and the dataset format. It can reduce throughput and does not
bypass all runtime kernel compilation. The flag is opt-in; use a new output
directory when switching modes. It is a workaround to isolate compilation
failures, not a diagnosis of the underlying compiler/toolchain problem.

Extra Python arguments may follow the shell script, for example
`--num-shards 2`, `--gpu-memory-utilization 0.7`, or `--local-files-only`.
Reduce the memory fraction when sharing a GPU with other workloads.
`PROGRESS_EVERY_BATCHES`, `EMPTY_CACHE_EVERY_BATCHES`, and `PYTHON` are also
supported. vLLM manages its own KV cache; the existing empty-cache interval only
releases unused PyTorch allocations. The legacy `--attn-implementation` flag is
accepted with a warning and ignored because vLLM selects its attention backend.
Workers run the engine in process and compile kernels serially so they remain
compatible with the existing multiprocessing pool.

`environment.yml` pins vLLM 0.9.2 to match the existing PyTorch 2.7.0 and
Transformers 4.52.4 stack. It also uses setuptools 78.1.1 to meet vLLM's Python
3.12 requirement. Creating `e2d-env` from that file installs vLLM and its
transitive dependencies. See the pinned release's
[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.9.2/requirements/cuda.txt)
and [common requirements](https://github.com/vllm-project/vllm/blob/v0.9.2/requirements/common.txt).

To train on the generated dataset, pass the same directory to the existing
training script (its current training length is 4096):

```bash
DISTILL_DATA_ROOT=/data/shared_data/hankun/datasets/tulu3_qwen3_1p7b_selfdistill_advanced_bsz32_maxlen4096 \
  bash bash_scripts/run_train_e2d_tulu3_distill.sh
```

It detects both preprocessed directories and skips its own generation stage.
