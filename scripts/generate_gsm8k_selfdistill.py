#!/usr/bin/env python3
"""Generate exact-token GSM8K self-distillation, optionally batched and sharded."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_gsm8k_selfdistill import PROMPT_PREFIX, greedy_generation_kwargs


JSONL_STEM = "gsm8k_qwen3_1p7b_selfdistill_bsz1"
DEFAULT_DATA_ROOT = Path("/data/shared_data/hankun/datasets/gsm8k_qwen3_1p7b_selfdistill_bsz1_maxlen1024")


def build_prompt(question: str, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{PROMPT_PREFIX} {question.strip()}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def validate_token_row(row: dict, tokenizer: Any) -> None:
    for text_key, ids_key in (("prompt", "prompt_token_ids"), ("completion", "completion_token_ids")):
        ids = row[ids_key]
        if not ids or any(type(token_id) is not int for token_id in ids):
            raise ValueError(f"Invalid {ids_key} at source index {row.get('source_index')}")
        if text_key == "completion" and tokenizer.decode(ids, skip_special_tokens=False) != row[text_key]:
            raise ValueError(f"{text_key} does not decode exactly from the saved token IDs")
        # The tokenizer can normalize prompt text (e.g. Qwen's Unicode NFC), so
        # decode(encode(prompt)) need not equal the original rendered prompt.
        # Encoding that original text must still reproduce the exact model
        # input IDs. Completion text is decoded from the generated IDs; those
        # IDs remain authoritative even if re-encoding uses a different BPE
        # segmentation of the same text.
        if text_key == "prompt" and tokenizer.encode(row[text_key], add_special_tokens=False) != ids:
            raise ValueError("Prompt does not re-encode exactly to the input token IDs")
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in row["completion_token_ids"][:-1]:
        raise ValueError("Completion contains tokens after EOS")


def generate_row(
    source_index: int,
    question: str,
    tokenizer: Any,
    model: Any,
    *,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    device: str,
) -> dict | None:
    result = generate_prompt_row(
        source_index, build_prompt(question, tokenizer), tokenizer, model,
        max_prompt_tokens=max_prompt_tokens, max_completion_tokens=max_completion_tokens,
        device=device,
    )
    if result is None:
        return None
    return {"source_index": source_index, "question": question.strip(), **result}


def generate_prompt_row(
    source_index: int,
    prompt: str,
    tokenizer: Any,
    model: Any,
    *,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    device: str,
) -> dict | None:
    """Generate from an already-rendered prompt without changing its token IDs."""
    import torch

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_tokens:
        return None
    if not prompt_ids:
        raise ValueError(f"Empty prompt at source index {source_index}")
    # No padding, truncation, batching, or decode/re-encode between generation
    # and recording the token IDs used by training.
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            **greedy_generation_kwargs(tokenizer, max_completion_tokens),
        )
    if output.shape[0] != 1 or output[0, :len(prompt_ids)].tolist() != prompt_ids:
        raise ValueError("Generation did not preserve the batch-one input prompt")
    completion_ids = output[0, len(prompt_ids):].tolist()
    if not completion_ids or len(completion_ids) > max_completion_tokens:
        raise ValueError("Generation returned an invalid completion length")
    result = {
        "source_index": source_index, "prompt": prompt,
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=False),
        "prompt_token_ids": prompt_ids, "completion_token_ids": completion_ids,
    }
    validate_token_row(result, tokenizer)
    return result


def prepare_rows(source: Any, tokenizer: Any, max_prompt_tokens: int, max_samples: int) -> list[dict]:
    """Select the global usable prefix before distributing it across GPUs."""
    rows = []
    for index, example in enumerate(source):
        question = example.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at source index {index}")
        prompt = build_prompt(question, tokenizer)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Empty prompt at source index {index}")
        if len(ids) > max_prompt_tokens:
            continue
        rows.append({
            "source_index": index, "question": question.strip(),
            "prompt": prompt, "prompt_token_ids": ids,
        })
        if max_samples and len(rows) == max_samples:
            break
    return rows


def source_indices_sha256(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices).encode()).hexdigest()


def generate_batch(
    rows: list[dict], tokenizer: Any, model: Any, *, max_completion_tokens: int, device: str,
) -> list[dict]:
    """Left-pad only model inputs; save original prompt IDs and completions through EOS."""
    import torch

    if not rows:
        return []
    kwargs = greedy_generation_kwargs(tokenizer, max_completion_tokens)
    pad_id = kwargs["pad_token_id"]
    if pad_id is None:
        raise ValueError("Batched generation requires a tokenizer pad_token_id or eos_token_id")
    width = max(len(row["prompt_token_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        ids = row["prompt_token_ids"]
        input_ids[index, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, -len(ids):] = 1
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    if output.shape[0] != len(rows) or not torch.equal(output[:, :width], input_ids):
        raise ValueError("Generation did not preserve the padded input prompts")
    completions = output[:, width:].tolist()
    generated = []
    for row, ids in zip(rows, completions):
        if not ids or len(ids) > max_completion_tokens:
            raise ValueError("Generation returned an invalid completion length")
        # Finished sequences receive padding while other batch members continue.
        # Retain EOS itself, even when EOS and padding share the same ID.
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
        # Preserve dataset-specific provenance and the existing JSONL field order.
        result = {key: value for key, value in row.items() if key != "prompt_token_ids"}
        result.update(
            completion=tokenizer.decode(ids, skip_special_tokens=False),
            prompt_token_ids=row["prompt_token_ids"], completion_token_ids=ids,
        )
        validate_token_row(result, tokenizer)
        generated.append(result)
    return generated


def encode_training_row(row: dict, max_length: int) -> dict[str, list[int]]:
    prompt_ids = row["prompt_token_ids"]
    completion_ids = row["completion_token_ids"]
    if not prompt_ids or not completion_ids or len(prompt_ids) > max_length // 2 or len(completion_ids) > max_length // 2:
        raise ValueError("Exact prompt/completion IDs exceed the half-length training budgets or are empty")
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "context_mask": [1] * len(prompt_ids) + [0] * len(completion_ids),
    }


def split_rows(rows: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if len(rows) < 2:
        raise ValueError("At least two generated rows are required for the train/eval split")
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between zero and one")
    ordered = sorted(rows, key=lambda row: row["source_index"])
    random.Random(seed).shuffle(ordered)
    eval_size = min(len(ordered) - 1, max(1, int(len(ordered) * eval_ratio)))
    return ordered[eval_size:], ordered[:eval_size]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_dataset(
    root: Path, rows: list[dict], max_length: int, eval_ratio: float, seed: int,
    *, jsonl_stem: str = JSONL_STEM,
) -> dict:
    from datasets import Dataset, load_from_disk

    root.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = split_rows(rows, eval_ratio, seed)
    sizes = {}
    for split, split_data in (("train", train_rows), ("eval", eval_rows)):
        jsonl_path = root / f"{jsonl_stem}_{split}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_data)
        if jsonl_path.exists() and jsonl_path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite an incompatible split: {jsonl_path}")
        if not jsonl_path.exists():
            temporary = jsonl_path.with_name(jsonl_path.name + f".tmp-{os.getpid()}")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, jsonl_path)
        encoded = [encode_training_row(row, max_length) for row in split_data]
        dataset_path = root / f"{split}_preprocessed"
        if dataset_path.exists():
            saved = load_from_disk(str(dataset_path))
            if len(saved) != len(encoded) or any(old != new for old, new in zip(saved, encoded)):
                raise ValueError(f"Refusing to overwrite incompatible training IDs: {dataset_path}")
        else:
            temporary = root / f".{split}_preprocessed.tmp-{os.getpid()}"
            Dataset.from_list(encoded).save_to_disk(str(temporary))
            temporary.rename(dataset_path)
        sizes[split] = len(split_data)
    return sizes


def load_resume_rows(path: Path, metadata: dict, tokenizer: Any) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed resume JSONL at {path}:{line_number}; preserving the file for inspection") from error
            if type(row.get("source_index")) is not int or row["source_index"] < 0:
                raise ValueError(f"Invalid source index at {path}:{line_number}")
            if rows and row["source_index"] <= rows[-1]["source_index"]:
                raise ValueError("Resume source indices must be unique and increasing")
            validate_token_row(row, tokenizer)
            encode_training_row(row, metadata["max_length"])
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--source-dataset", default="openai/gsm8k")
    parser.add_argument("--source-config", default="main")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses all source examples")
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", "--per-device-batch-size", type=int, default=1,
                        help="Maximum number of prompts generated together on each GPU")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="GPU workers (1-8); 0 automatically uses all visible GPUs")
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_length < 2 or args.max_samples < 0 or args.max_samples == 1 or not 0 < args.eval_ratio < 1 or args.progress_every < 1:
        parser.error("Invalid length, sample count (must be zero or >=2), eval ratio, or progress interval")

    if args.batch_size < 1 or not 0 <= args.num_shards <= 8:
        parser.error("Require batch_size >= 1 and num_shards between 0 (auto) and 8")

    import torch

    if args.num_shards == 0:
        if not args.device.startswith("cuda"):
            parser.error("Automatic GPU discovery requires --device cuda")
        args.num_shards = torch.cuda.device_count()
        if not 1 <= args.num_shards <= 8:
            parser.error(f"Expected 1-8 visible CUDA devices; found {args.num_shards}")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard_index < num_shards")
    if args.num_shards > 1 and args.shard_index is None:
        from scripts.gsm8k_selfdistill_parallel import launch

        return launch(args, sys.argv[1:])
    args.shard_index = args.shard_index or 0
    if args.num_shards == 1:
        work_root = args.data_root.with_name(args.data_root.name + ".shards")
        if (work_root / "run_config.json").exists():
            raise ValueError("This output belongs to a multi-GPU run. Resume with the same GPU count "
                             "or use a new output directory.")

    import transformers
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no distillation started. Run this command on the verifier's GPU host.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    source = load_dataset(
        args.source_dataset, args.source_config, split=args.source_split,
        download_config=DownloadConfig(local_files_only=args.local_files_only),
    )
    selected = prepare_rows(source, tokenizer, args.max_length // 2, args.max_samples)
    if len(selected) < 2:
        raise ValueError("At least two usable prompts are required for the train/eval split")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation, local_files_only=args.local_files_only,
    ).to(args.device).eval()
    metadata = {
        "schema_version": 1, "jsonl_stem": JSONL_STEM,
        "max_length": args.max_length, "max_prompt_tokens": args.max_length // 2,
        "max_completion_tokens": args.max_length // 2,
        "model_name_or_path": args.model_name_or_path,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "dtype": args.dtype, "attn_implementation": args.attn_implementation,
        "device_type": torch.device(args.device).type,
        "batch_size": args.batch_size, "use_cache": True, "do_sample": False, "num_beams": 1,
        "padding": args.batch_size > 1, "add_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": False, "add_generation_prompt": True},
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "prompt_prefix": PROMPT_PREFIX,
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "source_dataset": args.source_dataset, "source_config": args.source_config,
        "source_split": args.source_split, "source_fingerprint": getattr(source, "_fingerprint", None),
        "source_rows": len(source), "max_samples": args.max_samples,
        "split_seed": args.seed, "eval_ratio": args.eval_ratio,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }
    if args.batch_size > 1:
        metadata["padding_side"] = "left"
    if args.num_shards > 1:
        metadata.update(
            num_shards=args.num_shards, shard_index=args.shard_index,
            generation_num_shards=args.num_shards,
            selected_source_rows=len(selected),
            selected_source_indices_sha256=source_indices_sha256([row["source_index"] for row in selected]),
        )
    assigned = [row for row in selected if row["source_index"] % args.num_shards == args.shard_index]
    root = args.data_root
    metadata_path = root / "selfdistill_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (previous.get("num_shards", 1) != args.num_shards
                or previous.get("shard_index", 0) != args.shard_index
                or previous.get("generation_num_shards", 1) != args.num_shards):
            raise ValueError("Resume sharding settings differ. Use a new output directory.")
        changed = [key for key, value in metadata.items() if previous.get(key) != value]
        if changed:
            raise ValueError(f"Resume settings differ ({', '.join(changed)}). Use a new output directory.")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Refusing to use a nonempty directory without generation metadata: {root}")
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(metadata_path, {**metadata, "status": "generating"})
    raw_path = root / f"{JSONL_STEM}.jsonl"
    rows = load_resume_rows(raw_path, metadata, tokenizer)
    if len(rows) > len(assigned):
        raise ValueError("Resume rows exceed the selected source rows")
    for row, expected in zip(rows, assigned):
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Resume prompt/provenance no longer matches source example {expected['source_index']}")
    print(f"[distill] {len(source)} source rows; {len(assigned)} assigned; resumed {len(rows)}; "
          f"batch_size={args.batch_size}; shard={args.shard_index}/{args.num_shards}; "
          f"completion_budget={args.max_length // 2}; {args.dtype}/{args.attn_implementation}", flush=True)
    needs_newline = False
    if raw_path.exists() and raw_path.stat().st_size:
        with raw_path.open("rb") as check:
            check.seek(-1, os.SEEK_END)
            needs_newline = check.read(1) != b"\n"
    with raw_path.open("a", encoding="utf-8") as output:
        if needs_newline:
            output.write("\n")
        for start in range(len(rows), len(assigned), args.batch_size):
            batch = generate_batch(
                assigned[start:start + args.batch_size], tokenizer, model,
                max_completion_tokens=metadata["max_completion_tokens"], device=args.device,
            )
            output.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in batch))
            output.flush()
            os.fsync(output.fileno())
            previous_count = len(rows)
            rows.extend(batch)
            if len(rows) // args.progress_every > previous_count // args.progress_every or len(rows) == len(assigned):
                print(f"[distill] {len(rows)}/{len(assigned)} generated", flush=True)
    if args.num_shards > 1:
        atomic_json(metadata_path, {
            **metadata, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "generated_rows": len(rows),
        })
        print(f"[distill] Shard {args.shard_index}: saved {len(rows)} rows to {root}", flush=True)
        return 0
    sizes = write_dataset(root, rows, args.max_length, args.eval_ratio, args.seed)
    atomic_json(metadata_path, {
        **metadata, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "generated_rows": len(rows), "split_rows": sizes,
    })
    print(f"[distill] Saved {len(rows)} rows ({sizes['train']} train, {sizes['eval']} held out) to {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
