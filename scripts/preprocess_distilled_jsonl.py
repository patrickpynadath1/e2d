#!/usr/bin/env python
"""Convert an existing prompt/completion JSONL to the legacy E2D Arrow format.

Uses the same split order and half-length truncation as run_train_e2d_ultrachat.sh.
No model weights or source dataset are loaded. The source JSONL is never changed.
"""

import argparse
import json
import random
import tempfile
from pathlib import Path


def snapshot_jsonl(source: Path, destination: Path) -> dict:
    """Validate a stable snapshot, tolerating only an interrupted final JSON row."""
    before = source.stat()
    counts = {"rows": 0, "blank_lines": 0, "incomplete_final_lines": 0}
    with source.open("rb") as reader, destination.open("wb") as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if reader.tell() == before.st_size and not line.endswith(b"\n"):
                    counts["incomplete_final_lines"] += 1
                    print(f"Skipping incomplete final line {line_number}", flush=True)
                    continue
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from None
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) and row[key]
                for key in ("prompt", "completion")
            ):
                raise ValueError(
                    f"Expected nonempty prompt/completion at {source}:{line_number}"
                )
            writer.write(line.rstrip(b"\r\n") + b"\n")
            counts["rows"] += 1
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(
            "Source JSONL changed during snapshot; stop generation and retry."
        )
    if counts["rows"] < 2:
        raise ValueError(
            "At least two complete records are required for train/eval splits"
        )
    return {
        **counts,
        "source_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
    }


def encode_batch(batch, tokenizer, max_length):
    # Deliberately tokenize each side separately, as in the original distiller.
    options = dict(
        max_length=max_length // 2,
        padding=False,
        add_special_tokens=False,
        truncation=True,
    )
    prompts = tokenizer(batch["prompt"], **options)
    completions = tokenizer(batch["completion"], **options)
    result = {"input_ids": [], "attention_mask": [], "context_mask": []}
    for source_ids, source_mask, target_ids, target_mask in zip(
        prompts["input_ids"],
        prompts["attention_mask"],
        completions["input_ids"],
        completions["attention_mask"],
    ):
        result["input_ids"].append(source_ids + target_ids)
        result["attention_mask"].append(source_mask + target_mask)
        result["context_mask"].append(source_mask + [0] * len(target_ids))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument(
        "--data-root", type=Path, help="Output directory (default: input parent)"
    )
    parser.add_argument("--tokenizer-name-or-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_length < 2 or args.num_proc < 1 or not 0 < args.eval_ratio < 1:
        parser.error("Require max-length >= 2, num-proc >= 1, and 0 < eval-ratio < 1")

    from transformers import AutoTokenizer

    from datasets import Features, Sequence, Value, load_dataset

    source = args.input_jsonl.resolve()
    root = (args.data_root or source.parent).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    names = [f"{stem}_{split}.jsonl" for split in ("train", "eval")]
    names += [f"{split}_preprocessed" for split in ("train", "eval")]
    names += ["preprocessing_metadata.json"]
    existing = [str(root / name) for name in names if (root / name).exists()]
    if existing:
        raise FileExistsError(
            f"Outputs already exist: {existing}. Use a new --data-root."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        local_files_only=args.local_files_only,
    )
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "context_mask": Sequence(Value("int64")),
        }
    )
    # Stage everything on the destination filesystem; publish only after both
    # splits are finished. Temporary Arrow caches and the snapshot are removed.
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=root) as temporary:
        staging = Path(temporary)
        snapshot = staging / "snapshot.jsonl"
        stats = snapshot_jsonl(source, snapshot)
        print(f"Validated {stats['rows']} records from {source}", flush=True)
        raw = load_dataset(
            "json",
            data_files=str(snapshot),
            split="train",
            cache_dir=str(staging / "cache"),
        )
        if len(raw) != stats["rows"]:
            raise RuntimeError("Loaded row count differs from validated snapshot")
        indices = list(range(len(raw)))
        random.Random(args.seed).shuffle(indices)
        eval_size = min(len(raw) - 1, max(1, int(len(raw) * args.eval_ratio)))
        sizes = {}
        for split, selected in (
            ("train", indices[eval_size:]),
            ("eval", indices[:eval_size]),
        ):
            rows = raw.select(selected)
            rows.to_json(
                str(staging / f"{stem}_{split}.jsonl"),
                orient="records",
                lines=True,
                force_ascii=False,
            )
            encoded = rows.map(
                encode_batch,
                batched=True,
                batch_size=256,
                fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
                num_proc=args.num_proc,
                features=features,
                remove_columns=rows.column_names,
                load_from_cache_file=False,
                desc=f"Tokenizing {split}",
            )
            encoded.save_to_disk(str(staging / f"{split}_preprocessed"))
            sizes[split] = len(encoded)
        metadata = {
            "source_jsonl": str(source),
            **stats,
            "splits": sizes,
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "max_length": args.max_length,
            "eval_ratio": args.eval_ratio,
            "seed": args.seed,
            "truncation": "separate_prompt_completion_half_length",
        }
        (staging / "preprocessing_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in names:
            (staging / name).rename(root / name)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
