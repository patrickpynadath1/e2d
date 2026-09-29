#!/usr/bin/env python3
"""Generate Tulu 3 distillation data using independent vLLM GPU replicas.

Matches Stage 1 of run_train_e2d_tulu3_distill.sh, including its batch boundaries,
text JSONLs, and decode/re-encode training format. A bounded queue distributes
whole batches; the coordinator writes results in source order and checkpoints
each batch. Restarting replays at most the uncommitted batches.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import tempfile
import traceback
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_gsm8k_selfdistill import atomic_json


JSONL_STEM = "tulu3_qwen3_1p7b_thinking"
WORKER = {}


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content", ""))
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return ""


def extract_prompt_messages(example: dict) -> list[dict[str, str]] | None:
    raw_messages = example.get("messages")
    if isinstance(raw_messages, str):
        try:
            raw_messages = json.loads(raw_messages)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_messages, list):
        return None
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = normalize_content(message.get("content", ""))
        if role and content:
            messages.append({"role": role, "content": content})
    last_assistant_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
        -1,
    )
    if last_assistant_idx <= 0:
        return None
    prompt_messages = messages[:last_assistant_idx]
    if prompt_messages[-1]["role"] not in {"user", "tool"}:
        return None
    return prompt_messages


def iter_batches(sources: dict, tokenizer: Any, batch_size: int):
    """Keep the serial driver's pre-length-filter batches, including split tails."""
    batch_index = 0
    for split, source in sources.items():
        prompts = []
        for example in source:
            messages = extract_prompt_messages(example)
            if messages is None:
                continue
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
            if len(prompts) == batch_size:
                yield batch_index, split, prompts
                batch_index += 1
                prompts = []
        if prompts:
            yield batch_index, split, prompts
            batch_index += 1


def strip_batched_generation_padding(token_ids: list[int], tokenizer: Any) -> list[int]:
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    if not token_ids or pad_id is None:
        return token_ids
    if eos_id is None or pad_id != eos_id:
        return [tid for tid in token_ids if tid != pad_id]
    if eos_id in token_ids:
        return token_ids[:token_ids.index(eos_id) + 1]
    return token_ids


def generate_batch(task, tokenizer: Any, model: Any, args) -> tuple[int, list[dict], int]:
    from vllm import SamplingParams

    batch_index, split, prompts = task
    lengths = [len(ids) for ids in tokenizer(
        prompts, add_special_tokens=False, truncation=False, padding=False,
    )["input_ids"]]
    selected = [prompt for prompt, length in zip(prompts, lengths) if length <= args.prompt_max_length]
    filtered = len(prompts) - len(selected)
    if not selected:
        return batch_index, [], filtered
    inputs = tokenizer(
        selected, truncation=True, max_length=args.gen_max_length - 1, padding=False,
    )
    # Keep the old padded-batch budget, even though vLLM takes unpadded IDs.
    # Passing IDs also avoids a second chat template or different BOS handling.
    width = max(len(ids) for ids in inputs["input_ids"])
    sampling = SamplingParams(
        n=1, temperature=0.0, max_tokens=max(1, args.gen_max_length - width),
        # HF explicitly overrode model EOS defaults with the tokenizer EOS.
        # Ignore the engine's EOS defaults and stop only at that same token.
        ignore_eos=True,
        stop_token_ids=[] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id],
        detokenize=False, skip_special_tokens=False,
    )
    outputs = model.generate(
        [{"prompt_token_ids": ids} for ids in inputs["input_ids"]],
        sampling_params=sampling, use_tqdm=False,
    )
    if len(outputs) != len(selected):
        raise ValueError("vLLM returned a different number of outputs than prompts")
    rows = []
    for prompt, output in zip(selected, outputs):
        ids = strip_batched_generation_padding(list(output.outputs[0].token_ids), tokenizer)
        if not ids:
            continue
        completion = tokenizer.decode(ids, skip_special_tokens=False)
        if completion:
            rows.append({"prompt": prompt, "completion": completion, "source_split": split})
    return batch_index, rows, filtered


def load_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left",
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def init_worker(slots, work_root: str, args):
    # Spawn avoids inheriting a CUDA context. Bind before importing torch or
    # loading weights, so every replica sees just its assigned physical GPU.
    index, device = slots.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    # Pool workers are daemonic: keep the single-GPU engine and compilation
    # in this process instead of launching nested multiprocessing workers.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    log_path = Path(work_root) / f"shard_{index:02d}.log"
    with log_path.open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    WORKER.update(args=args, batches=0, device=device, log_path=str(log_path))
    print(f"[distill] Worker {index}: GPU {device}, PID {os.getpid()}", flush=True)


def worker_generate(task):
    """Send plain-text failures across the pool, never compiler exception objects."""
    if "failure" in WORKER:
        # Speculative batches can reach this worker before the coordinator sees
        # its first error. Do not retry a failed engine or allocate weights again.
        raise RuntimeError(WORKER["failure"])
    try:
        return _worker_generate(task)
    except Exception:
        details = traceback.format_exc()
        message = (
            f"[distill] Batch {task[0]} failed on GPU {WORKER.get('device', '?')}; "
            f"worker log: {WORKER.get('log_path', 'unavailable')}\n{details}"
        )
        WORKER["failure"] = message
        print(message, file=sys.stderr, flush=True)
        # PyTorch compiler exceptions can contain unpicklable frame objects.
        # A built-in exception with only a string preserves the original trace
        # in the coordinator's main log without serializing those objects.
        raise RuntimeError(message) from None


def _worker_generate(task):
    import torch
    from vllm import LLM

    args = WORKER["args"]
    # Load inside the task so model-loading exceptions reach the coordinator
    # instead of causing multiprocessing.Pool to repeatedly restart an initializer.
    if "model" not in WORKER:
        WORKER["tokenizer"] = load_tokenizer(args)
        model_path = args.model_name_or_path
        if args.local_files_only and not Path(model_path).is_dir():
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path, local_files_only=True)
        if args.attn_implementation:
            print(
                "[distill] --attn-implementation is ignored; "
                "vLLM selects its attention backend.", flush=True,
            )
        WORKER["model"] = LLM(
            model=model_path, dtype=args.dtype, trust_remote_code=True,
            tensor_parallel_size=1, distributed_executor_backend="uni",
            max_model_len=args.gen_max_length, max_num_seqs=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
            # All tokenization/decoding stays with the existing HF tokenizer.
            skip_tokenizer_init=True, generation_config="vllm",
            enforce_eager=args.enforce_eager,
        )
    result = generate_batch(task, WORKER["tokenizer"], WORKER["model"], args)
    WORKER["batches"] += 1
    if WORKER["batches"] % args.empty_cache_every_batches == 0:
        torch.cuda.empty_cache()
    print(f"[distill] Batch {result[0]}: {len(result[1])} rows; {result[2]} filtered", flush=True)
    return result


def ordered_results(pool, tasks, num_workers):
    """Bound memory and speculative generation while preserving serial order."""
    pending = deque()
    tasks = iter(tasks)
    # Pool replaces workers killed by OOM/signals, losing their pending result.
    # Watch the original processes to fail promptly instead of waiting forever.
    workers = list(pool._pool)

    def submit():
        task = next(tasks, None)
        if task is not None:
            pending.append(pool.apply_async(worker_generate, (task,)))

    for _ in range(2 * num_workers):
        submit()
    while pending:
        result = pending.popleft()
        while True:
            if any(worker.exitcode is not None for worker in workers):
                raise RuntimeError("A generation worker exited unexpectedly; inspect the shard logs and resume.")
            try:
                value = result.get(timeout=0.5)
                break
            except mp.TimeoutError:
                pass
        yield value
        submit()


def commit_results(results, raw_path: Path, metadata_path: Path, settings: dict, state: dict, args):
    """Commit JSONL bytes before advancing the cursor; ignore speculative batches."""
    with raw_path.open("ab") as output:
        for batch_index, rows, filtered in results:
            if batch_index != state["next_batch"]:
                raise ValueError("Generation results are out of source order")
            if args.max_samples:
                rows = rows[:max(0, args.max_samples - state["generated_rows"])]
            output.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
            state.update(
                next_batch=batch_index + 1, raw_bytes=output.tell(),
                generated_rows=state["generated_rows"] + len(rows),
                filtered_prompts=state["filtered_prompts"] + filtered,
            )
            atomic_json(metadata_path, {"settings": settings, **state})
            if args.progress_every_batches and state["next_batch"] % args.progress_every_batches == 0:
                print(f"[distill] {state['next_batch']} batches; {state['generated_rows']} rows; "
                      f"{state['filtered_prompts']} overlong prompts filtered", flush=True)
            if args.max_samples and state["generated_rows"] >= args.max_samples:
                break
    state["status"] = "generated"
    atomic_json(metadata_path, {"settings": settings, **state})


def restore_output(root: Path, settings: dict) -> dict:
    metadata_path = root / "selfdistill_metadata.json"
    raw_path = root / f"{settings['jsonl_stem']}.jsonl"
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("settings") != settings:
            raise ValueError("Resume settings/source differ. Use a new DISTILL_DATA_ROOT.")
        state = {key: value for key, value in saved.items() if key != "settings"}
        size = raw_path.stat().st_size if raw_path.exists() else 0
        if size < state["raw_bytes"]:
            raise ValueError("Generated JSONL is shorter than its committed checkpoint")
        if state["status"] != "generating" and size != state["raw_bytes"]:
            raise ValueError("Completed JSONL was changed after generation")
        if size > state["raw_bytes"]:
            # Only discard bytes written after the last successful checkpoint.
            with raw_path.open("rb+") as output:
                output.truncate(state["raw_bytes"])
        return state
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to overwrite a nonempty output without matching metadata: {root}")
    root.mkdir(parents=True, exist_ok=True)
    state = dict(status="generating", next_batch=0, generated_rows=0, raw_bytes=0, filtered_prompts=0)
    atomic_json(metadata_path, {"settings": settings, **state})
    return state


def encode_pair_for_e2d(prompt: str, completion: str, tokenizer: Any, max_length: int) -> dict:
    tokenized = tokenizer.batch_encode_plus(
        [prompt, completion], max_length=max_length // 2,
        padding=False, add_special_tokens=False, truncation=True,
    )
    source_ids, target_ids = tokenized["input_ids"]
    source_mask, target_mask = tokenized["attention_mask"]
    return {
        "input_ids": source_ids + target_ids,
        "attention_mask": source_mask + target_mask,
        "context_mask": source_mask + [0] * len(target_ids),
    }


def write_dataset(root: Path, tokenizer: Any, args, expected_rows: int) -> dict:
    from datasets import load_dataset

    if expected_rows < 2:
        raise ValueError("At least two generated rows are required for train/eval splits")
    # Memory-map the corpus as in the reference driver; do not collect a million
    # conversations and their token lists in Python memory.
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=root) as temporary:
        staging = Path(temporary)
        raw = load_dataset(
            "json", data_files=str(root / f"{args.jsonl_stem}.jsonl"), split="train",
            cache_dir=str(staging / "cache"),
        )
        if len(raw) != expected_rows:
            raise ValueError("Generated row count differs from the checkpoint")
        indices = list(range(len(raw)))
        random.Random(args.seed).shuffle(indices)
        eval_size = min(len(raw) - 1, max(1, int(len(raw) * args.eval_ratio)))
        sizes = {}
        names = []
        for split, selected in (("train", indices[eval_size:]), ("eval", indices[:eval_size])):
            rows = raw.select(selected)
            jsonl_name = f"{args.jsonl_stem}_{split}.jsonl"
            dataset_name = f"{split}_preprocessed"
            rows.to_json(str(staging / jsonl_name), orient="records", lines=True, force_ascii=False)
            encoded = rows.map(
                lambda row: encode_pair_for_e2d(row["prompt"], row["completion"], tokenizer, args.max_length),
                remove_columns=rows.column_names, keep_in_memory=False,
                load_from_cache_file=False, desc=f"Tokenizing distilled {split} split",
            )
            encoded.save_to_disk(str(staging / dataset_name))
            sizes[split] = len(rows)
            names.extend([jsonl_name, dataset_name])
        for name in names:
            destination = root / name
            if destination.is_dir():
                shutil.rmtree(destination)
            os.replace(staging / name, destination)
    return sizes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--jsonl-stem", default=JSONL_STEM)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--source-dataset", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--source-splits", nargs="+", default=["train"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--gen-max-length", type=int, default=4096)
    parser.add_argument("--prompt-max-length", type=int, default=1024)
    parser.add_argument("--batch-size", "--per-device-batch-size", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=0, help="GPU replicas (1-8); 0 uses all visible GPUs")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"],
                        help="Deprecated compatibility option; vLLM selects its own attention backend")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of each GPU's memory reserved for its vLLM replica")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable vLLM torch.compile and CUDA graphs for troubleshooting")
    parser.add_argument("--max-samples", type=int, default=0, help="Global generated-row limit; 0 uses all rows")
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every-batches", type=int, default=1, help="0 disables progress messages")
    parser.add_argument("--empty-cache-every-batches", type=int, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if (args.max_length < 2 or args.gen_max_length < args.max_length or args.prompt_max_length < 1
            or args.batch_size < 1 or not 0 <= args.num_shards <= 8
            or args.max_samples < 0 or args.max_samples == 1 or not 0 < args.eval_ratio < 1
            or args.progress_every_batches < 0 or args.empty_cache_every_batches < 1
            or not 0 < args.gpu_memory_utilization <= 1):
        parser.error(
            "Invalid lengths, batch/GPU count, sample limit, split ratio, "
            "progress/cache interval, or GPU memory fraction"
        )
    if not args.source_splits or len(set(args.source_splits)) != len(args.source_splits):
        parser.error("Source splits must be nonempty and unique")
    if not args.jsonl_stem or Path(args.jsonl_stem).name != args.jsonl_stem or args.jsonl_stem in {".", ".."}:
        parser.error("jsonl-stem must be a filename stem without directories")
    return args


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from datasets import DownloadConfig, load_dataset

    available = torch.cuda.device_count()
    count = args.num_shards or available
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [device.strip() for device in visible.split(",")] if visible is not None else [
        str(index) for index in range(available)
    ]
    if (not 1 <= count <= 8 or available < count or len(devices) < count
            or any(not device for device in devices) or len(set(devices)) != len(devices)):
        raise ValueError(f"Need 1-8 distinct visible CUDA GPUs; requested={count}, available={available}, "
                         f"CUDA_VISIBLE_DEVICES={visible!r}")
    tokenizer = load_tokenizer(args)
    sources = {
        split: load_dataset(
            args.source_dataset, split=split,
            download_config=DownloadConfig(local_files_only=args.local_files_only),
        ) for split in args.source_splits
    }
    settings = {
        **{key: value for key, value in vars(args).items()
           if key not in {"data_root", "num_shards", "local_files_only", "progress_every_batches",
                          "empty_cache_every_batches", "enforce_eager"}},
        # Preserve compatibility with existing checkpoints in the default mode.
        **({"enforce_eager": True} if args.enforce_eager else {}),
        "schema_version": 1, "prompt_format": "tulu3_messages", "num_shards": count,
        "chat_template_sha256": hashlib.sha256(json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest(),
        "source_fingerprints": {split: getattr(source, "_fingerprint", None) for split, source in sources.items()},
        "source_rows_by_split": {split: len(source) for split, source in sources.items()},
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "inference_backend": "vllm", "vllm_version": version("vllm"),
    }
    root = args.data_root.resolve()
    metadata_path = root / "selfdistill_metadata.json"
    state = restore_output(root, settings)
    if state["status"] == "complete":
        expected = [f"{args.jsonl_stem}_{split}.jsonl" for split in ("train", "eval")]
        expected += [f"{split}_preprocessed" for split in ("train", "eval")]
        if all((root / name).exists() for name in expected):
            print(f"[distill] Already complete: {root}", flush=True)
            return 0
    if state["status"] == "generating" and not (args.max_samples and state["generated_rows"] >= args.max_samples):
        work_root = root.with_name(root.name + ".shards")
        work_root.mkdir(parents=True, exist_ok=True)
        tasks = (task for task in iter_batches(sources, tokenizer, args.batch_size)
                 if task[0] >= state["next_batch"])
        print(f"[distill] Tulu 3 (vLLM): {count} GPUs, batch_size={args.batch_size}, "
              f"resumed={state['generated_rows']}; logs={work_root}", flush=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
        context = mp.get_context("spawn")
        slots = context.Queue()
        for index, device in enumerate(devices[:count]):
            slots.put((index, device))
        try:
            with context.Pool(count, initializer=init_worker, initargs=(slots, str(work_root), args)) as pool:
                commit_results(
                    ordered_results(pool, tasks, count), root / f"{args.jsonl_stem}.jsonl",
                    metadata_path, settings, state, args,
                )
        finally:
            slots.close()
            slots.join_thread()
            signal.signal(signal.SIGTERM, previous_sigterm)
    state["status"] = "generated"
    atomic_json(metadata_path, {"settings": settings, **state})
    sizes = write_dataset(root, tokenizer, args, state["generated_rows"])
    state.update(status="complete", split_rows=sizes, completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(metadata_path, {"settings": settings, **state})
    print(f"[distill] Saved {state['generated_rows']} Tulu 3 rows "
          f"({sizes['train']} train, {sizes['eval']} held out) to {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
