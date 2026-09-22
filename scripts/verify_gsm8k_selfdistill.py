#!/usr/bin/env python3
"""Fail closed unless self-distillation matches the frozen verifier.

The existing JSONL files contain already-rendered Qwen chat prompts. They must
not be wrapped in another chat template. This checks those prompts and the
on-disk training token IDs before checking the original model's greedy outputs.

By default every row is regenerated using greedy decoding with the KV cache.
The faster --verification-method teacher-forced checks each reference token's
causal argmax and additionally regenerates a spread of examples. If each token
is argmax given its reference prefix, induction proves the reference is a
greedy trajectory in exact arithmetic. Different kernels and cache shapes can
change close argmax decisions in finite precision; sampled free-running checks
help detect this, but are not an exhaustive cached generation check.
No data is changed or silently regenerated when a check fails.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROMPT_PREFIX = "Please reason step by step, and put your final answer within $\\boxed{}$."
JSONL_STEM = "gsm8k_qwen3_1p7b_selfdistill_evalprompt"
ULTRACHAT_JSONL_STEM = "ultrachat_qwen3_1p7b_thinking"


def greedy_generation_kwargs(tokenizer: Any, max_new_tokens: int) -> dict:
    """Shared by batch-one distillation and its cached-generation preflight."""
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ),
        # Neutral defaults silence inherited sampling-only config warnings.
        # These values do not alter the greedy argmax.
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(key), str) and row[key] for key in ("prompt", "completion")):
                raise ValueError(f"{path}:{line_number}: expected nonempty prompt/completion strings")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def audit_dataset(args: argparse.Namespace, tokenizer: Any, report: dict) -> list[dict]:
    from datasets import load_from_disk

    examples = []
    all_rows = []
    half_length = args.max_length // 2
    metadata_path = args.data_root / "selfdistill_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    prompt_format = getattr(args, "prompt_format", "gsm8k")
    if prompt_format not in ("gsm8k", "ultrachat"):
        raise ValueError(f"Unsupported prompt format: {prompt_format}")
    legacy_stem = ULTRACHAT_JSONL_STEM if prompt_format == "ultrachat" else JSONL_STEM
    requested_stem = getattr(args, "jsonl_stem", None)
    jsonl_stem = metadata["jsonl_stem"] if metadata is not None else (requested_stem or legacy_stem)
    if requested_stem is not None and requested_stem != jsonl_stem:
        raise ValueError("Requested JSONL stem differs from dataset metadata")
    if Path(jsonl_stem).name != jsonl_stem or not jsonl_stem:
        raise ValueError("Dataset metadata jsonl_stem must be a plain filename stem")
    report["jsonl_stem"] = jsonl_stem
    report["prompt_format"] = prompt_format
    if metadata is not None:
        report["dataset_metadata"] = metadata
        if metadata.get("prompt_format", "gsm8k") != prompt_format:
            raise ValueError("Dataset prompt_format differs from the requested verification format")
        if metadata.get("max_length") != args.max_length:
            raise ValueError("Dataset max_length differs from the training/preflight max_length")
        if metadata.get("status") != "complete":
            raise ValueError("Distillation is incomplete; finish generation before training")
        expected_settings = {
            "batch_size": 1, "do_sample": False, "use_cache": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False, "add_generation_prompt": True},
        }
        for key, expected_setting in expected_settings.items():
            if metadata.get(key) != expected_setting:
                raise ValueError(f"Dataset metadata {key} does not match batch-one greedy verification")
    for split in ("train", "eval"):
        jsonl_path = args.data_root / f"{jsonl_stem}_{split}.jsonl"
        preprocessed_path = args.data_root / f"{split}_preprocessed"
        rows = load_rows(jsonl_path)
        dataset = load_from_disk(str(preprocessed_path))
        if len(rows) != len(dataset):
            raise ValueError(f"{split}: JSONL has {len(rows)} rows but training dataset has {len(dataset)}")
        stats = {
            "rows": len(rows), "jsonl_sha256": sha256(jsonl_path),
            "preprocessed_files_sha256": {
                str(path.relative_to(preprocessed_path)): sha256(path)
                for path in sorted(preprocessed_path.rglob("*")) if path.is_file()
            },
            "max_prompt_tokens": 0, "full_completion_tokens": 0,
            "training_completion_tokens": 0, "truncated_completions": 0,
        }
        report["splits"][split] = stats
        for index, (row, processed) in enumerate(zip(rows, dataset)):
            location = f"{split}[{index}]"
            prompt = row["prompt"]
            start = "<|im_start|>user\n"
            if not prompt.startswith(start) or "<|im_end|>" not in prompt:
                raise ValueError(f"{location}: expected an already-rendered Qwen user chat prompt")
            messages = row.get("prompt_messages")
            if messages is None:
                content = prompt[len(start):].split("<|im_end|>", 1)[0]
                messages = [{"role": "user", "content": content}]
            if (
                not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict) or messages[0].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)
            ):
                raise ValueError(f"{location}: expected a single user message")
            content = messages[0]["content"]
            if prompt_format == "gsm8k" and not content.startswith(PROMPT_PREFIX + " "):
                raise ValueError(f"{location}: prompt does not use the GSM8K evaluation prefix")
            expected_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            if prompt != expected_prompt:
                raise ValueError(f"{location}: saved prompt differs from the verifier's non-thinking chat template")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            has_generated_ids = "completion_token_ids" in row
            completion_ids = (
                row["completion_token_ids"] if has_generated_ids
                else tokenizer.encode(row["completion"], add_special_tokens=False)
            )
            if not isinstance(completion_ids, list) or any(
                not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
                for token_id in completion_ids
            ):
                raise ValueError(f"{location}: completion_token_ids must be a list of nonnegative integers")
            if "prompt_token_ids" in row and row["prompt_token_ids"] != prompt_ids:
                raise ValueError(f"{location}: saved generation prompt_token_ids differ from the verifier tokenizer")
            if metadata is not None and (
                "prompt_token_ids" not in row or not has_generated_ids
            ):
                raise ValueError(f"{location}: missing exact generation token IDs")
            if not prompt_ids or not completion_ids:
                raise ValueError(f"{location}: empty tokenized prompt/completion")
            if len(prompt_ids) > half_length:
                raise ValueError(f"{location}: preprocessing truncated the prompt, changing the verifier context")
            if not has_generated_ids and prompt_ids + completion_ids != tokenizer.encode(prompt + row["completion"], add_special_tokens=False):
                raise ValueError(f"{location}: separately tokenized prompt/completion differ at their boundary")
            if tokenizer.decode(completion_ids, skip_special_tokens=False) != row["completion"]:
                raise ValueError(f"{location}: completion text does not round-trip through the verifier tokenizer")
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and eos_id in completion_ids[:-1]:
                raise ValueError(f"{location}: completion contains tokens after EOS")
            if metadata is not None and len(completion_ids) > half_length:
                raise ValueError(f"{location}: generated completion exceeds the recorded training token budget")
            training_completion = completion_ids[:half_length]
            expected = {
                "input_ids": prompt_ids + training_completion,
                "attention_mask": [1] * (len(prompt_ids) + len(training_completion)),
                "context_mask": [1] * len(prompt_ids) + [0] * len(training_completion),
            }
            for key, value in expected.items():
                if list(processed[key]) != value:
                    raise ValueError(f"{location}: stored {key} differs from the exact raw prompt/completion IDs")
            stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], len(prompt_ids))
            stats["full_completion_tokens"] += len(completion_ids)
            stats["training_completion_tokens"] += len(training_completion)
            stats["truncated_completions"] += len(completion_ids) > half_length
            examples.append({"split": split, "row": index, "prompt_ids": prompt_ids, "completion_ids": completion_ids})
        all_rows.extend(rows)
        print(f"[audit] {split}: {len(rows)} exact templates/token sequences; "
              f"{stats['truncated_completions']} completions truncated to {half_length} training tokens", flush=True)
    raw_path = args.data_root / f"{jsonl_stem}.jsonl"
    if raw_path.exists():
        raw_rows = load_rows(raw_path)
        pair_counts = lambda rows: Counter(
            (
                row["prompt"], row["completion"],
                tuple(row.get("prompt_token_ids", [])),
                tuple(row.get("completion_token_ids", [])),
            )
            for row in rows
        )
        if pair_counts(raw_rows) != pair_counts(all_rows):
            raise ValueError("The full JSONL and the combined train/eval JSONLs contain different prompt/completion pairs")
        report["raw_jsonl_sha256"] = sha256(raw_path)
    return examples


def mismatch_details(expected: list[int], actual: list[int], tokenizer: Any) -> dict | None:
    limit = min(len(expected), len(actual))
    index = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    if index == len(expected) == len(actual):
        return None
    return {
        "completion_token_index": index,
        "expected_token_id": expected[index] if index < len(expected) else None,
        "actual_token_id": actual[index] if index < len(actual) else None,
        "expected_text_near_mismatch": tokenizer.decode(expected[max(0, index - 8):index + 8]),
        "actual_text_near_mismatch": tokenizer.decode(actual[max(0, index - 8):index + 8]),
        "expected_length": len(expected), "actual_length": len(actual),
    }


def check_model(args: argparse.Namespace, tokenizer: Any, examples: list[dict], report: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. No model-output verification was performed. "
                           "Run on a GPU host, or explicitly select --device cpu --dtype float32 for a CPU diagnostic.")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    metadata = report.get("dataset_metadata")
    if metadata is not None:
        for key in ("model_name_or_path", "dtype", "attn_implementation"):
            if metadata.get(key) != getattr(args, key):
                raise ValueError(f"Verifier {key} differs from the recorded distillation settings")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    report["model_commit_hash"] = getattr(model.config, "_commit_hash", None)
    if metadata is not None and metadata.get("model_commit_hash") != report["model_commit_hash"]:
        raise ValueError("Verifier model revision differs from the recorded distillation revision")
    selected = examples if args.max_samples == 0 else examples[:args.max_samples]
    report["total_rows"] = len(examples)
    report["selected_rows"] = len(selected)
    report["teacher_forced_rows_checked"] = 0
    report["generated_rows_checked"] = 0
    report["completion_tokens_checked"] = 0
    generation_count = min(args.generation_samples, len(selected))
    generation_indices = {
        round(i * (len(selected) - 1) / max(1, generation_count - 1)) for i in range(generation_count)
    }
    with torch.inference_mode():
        for ordinal, example in enumerate(selected):
            expected = example["completion_ids"]
            prompt = example["prompt_ids"]
            if args.verification_method == "teacher-forced":
                # Last prompt position predicts completion[0]. Exclude the final
                # completion token from the input so the returned logits align.
                input_ids = torch.tensor([prompt + expected[:-1]], device=args.device)
                logits = model(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, logits_to_keep=len(expected),
                ).logits
                actual = logits.argmax(-1)[0].tolist()
                details = mismatch_details(expected, actual, tokenizer)
                del logits, input_ids
                if details is not None:
                    report["mismatch"] = {"split": example["split"], "row": example["row"],
                                          "method": "teacher-forced", **details}
                    raise ValueError("The frozen verifier's argmax differs from the stored completion: "
                                     + json.dumps(report["mismatch"], ensure_ascii=False)
                                     + ". For near-tied logits, use --verification-method generate to check cached decoding.")
                report["teacher_forced_rows_checked"] += 1
            if args.verification_method == "generate" or ordinal in generation_indices:
                input_ids = torch.tensor([prompt], device=args.device)
                output = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    **greedy_generation_kwargs(tokenizer, len(expected)),
                )
                actual = output[0, len(prompt):].tolist()
                details = mismatch_details(expected, actual, tokenizer)
                del output, input_ids
                if details is not None:
                    report["mismatch"] = {"split": example["split"], "row": example["row"],
                                          "method": "generate", **details}
                    raise ValueError("The frozen verifier's cached greedy generation differs from the stored completion: "
                                     + json.dumps(report["mismatch"], ensure_ascii=False))
                report["generated_rows_checked"] += 1
            report["completion_tokens_checked"] += len(expected)
            if (ordinal + 1) % args.progress_every == 0 or ordinal + 1 == len(selected):
                print(f"[verify] {ordinal + 1}/{len(selected)} exact rows; "
                      f"{report['completion_tokens_checked']} completion tokens", flush=True)
    report["status"] = "verified" if len(selected) == len(examples) else "sample_verified"
    report["full_dataset_verified"] = len(selected) == len(examples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-format", choices=("gsm8k", "ultrachat"), default="gsm8k",
                        help="UltraChat uses the source user prompt without the GSM8K instruction prefix")
    parser.add_argument("--jsonl-stem", help="Legacy JSONL filename stem; new datasets record it in metadata")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--verification-method", choices=("teacher-forced", "generate"), default="generate")
    parser.add_argument("--generation-samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 checks all rows; positive values are diagnostics only")
    parser.add_argument("--audit-only", action="store_true", help="Check templates/token IDs without claiming model-output agreement")
    parser.add_argument("--report", type=Path, help="Write JSON evidence, including failure details")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.max_length < 2 or args.max_samples < 0 or args.generation_samples < 0 or args.progress_every < 1 or args.cpu_threads < 0:
        parser.error("Invalid length, sample count, thread count, or progress interval")
    report = {
        "status": "started", "full_dataset_verified": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": args.model_name_or_path, "data_root": str(args.data_root.resolve()),
        "max_length": args.max_length, "device": args.device, "dtype": args.dtype,
        "attn_implementation": args.attn_implementation, "verification_method": args.verification_method,
        "prompt_format": args.prompt_format,
        "prompt_prefix": PROMPT_PREFIX if args.prompt_format == "gsm8k" else "",
        "chat_template_kwargs": {"enable_thinking": False, "add_generation_prompt": True},
        "tokenization_add_special_tokens": False, "splits": {},
    }
    exit_code = 0
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer

        report["torch_version"] = torch.__version__
        report["transformers_version"] = transformers.__version__
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
        report["chat_template_sha256"] = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        examples = audit_dataset(args, tokenizer, report)
        if args.audit_only:
            report["status"] = "audited_not_verified"
            print("[audit] Templates and training IDs match. Model outputs have NOT been verified.", flush=True)
        else:
            check_model(args, tokenizer, examples, report)
            print(f"[verify] {report['status']}: {report['selected_rows']} rows", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"[verify] FAILED: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[verify] Report: {args.report}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
