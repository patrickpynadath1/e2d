"""Independent GPU workers and lossless merging for batch-one UltraChat generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import filecmp
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.generate_gsm8k_selfdistill import atomic_json, load_resume_rows, write_dataset
from scripts.generate_ultrachat_selfdistill import JSONL_STEM, source_location


RUN_FIELDS = {"status", "completed_at", "generated_rows", "split_rows"}
SHARD_FIELDS = {"num_shards", "shard_index", "source_index_stop"}


def read_output_metadata(root: Path) -> dict | None:
    path = root / "selfdistill_metadata.json"
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            metadata.get("prompt_format") != "ultrachat"
            or metadata.get("jsonl_stem") != JSONL_STEM
            or metadata.get("num_shards", 1) != 1
        ):
            raise ValueError(f"Refusing to overwrite a different dataset in {root}; select a new output directory")
        return metadata
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to use a nonempty directory without generation metadata: {root}")
    return None


def merge_shards(root: Path, shard_roots: list[Path], *, local_files_only: bool = False) -> None:
    """Validate all completed shards, restore serial order, and use the serial writer."""
    from transformers import AutoTokenizer

    previous = read_output_metadata(root)
    metadata = None
    tokenizer = None
    stop = None
    rows = []
    for index, shard_root in enumerate(shard_roots):
        shard = json.loads((shard_root / "selfdistill_metadata.json").read_text(encoding="utf-8"))
        if (
            shard.get("status") != "complete"
            or shard.get("num_shards") != len(shard_roots)
            or shard.get("shard_index") != index
            or shard.get("jsonl_stem") != JSONL_STEM
            or shard.get("prompt_format") != "ultrachat"
        ):
            raise ValueError(f"Incomplete or incorrectly assigned shard: {shard_root}")
        common = {key: value for key, value in shard.items() if key not in RUN_FIELDS | SHARD_FIELDS}
        if metadata is None:
            metadata = common
            stop = shard["source_index_stop"]
            tokenizer = AutoTokenizer.from_pretrained(
                metadata["model_name_or_path"], local_files_only=local_files_only,
            )
        elif common != metadata or shard["source_index_stop"] != stop:
            raise ValueError(f"Generation settings differ across shards: {shard_root}")
        raw_path = shard_root / f"{JSONL_STEM}.jsonl"
        if not raw_path.is_file():
            raise ValueError(f"Missing completed shard JSONL: {raw_path}")
        shard_rows = load_resume_rows(raw_path, metadata, tokenizer)
        if len(shard_rows) != shard.get("generated_rows"):
            raise ValueError(f"Completed shard row count differs: {shard_root}")
        sources = {split: range(metadata["source_rows_by_split"][split]) for split in metadata["source_splits"]}
        for row in shard_rows:
            source_index = row["source_index"]
            if source_index >= stop or source_index % len(shard_roots) != index:
                raise ValueError(f"Row {source_index} is outside shard {index}")
            split, source_row_index = source_location(source_index, sources)
            if row.get("source_split") != split or row.get("source_row_index") != source_row_index:
                raise ValueError(f"Row {source_index} has incorrect source provenance")
        rows.extend(shard_rows)
    if metadata is None or len(rows) < 2:
        raise ValueError("At least two generated rows are required for the train/eval split")
    if metadata["max_samples"] and len(rows) > metadata["max_samples"]:
        raise ValueError("Merged shards exceed the global sample limit")
    if previous is not None:
        changed = [key for key, value in metadata.items() if previous.get(key) != value]
        if changed:
            raise ValueError(f"Output settings differ ({', '.join(changed)}). Use a new output directory.")
    rows.sort(key=lambda row: row["source_index"])
    root.mkdir(parents=True, exist_ok=True)
    if previous is None:
        # Record ownership before a potentially long JSONL write so an
        # interrupted merge can be retried without regenerating completions.
        atomic_json(root / "selfdistill_metadata.json", {**metadata, "status": "merging"})
    raw_path = root / f"{JSONL_STEM}.jsonl"
    temporary = raw_path.with_name(raw_path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if raw_path.exists() and not filecmp.cmp(raw_path, temporary, shallow=False):
            raise ValueError(f"Refusing to overwrite incompatible generated rows: {raw_path}")
        atomic_json(root / "selfdistill_metadata.json", {**metadata, "status": "merging"})
        os.replace(temporary, raw_path)
    finally:
        temporary.unlink(missing_ok=True)
    sizes = write_dataset(
        root, rows, metadata["max_length"], metadata["eval_ratio"], metadata["split_seed"],
        jsonl_stem=JSONL_STEM,
    )
    atomic_json(root / "selfdistill_metadata.json", {
        **metadata, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "generated_rows": len(rows), "split_rows": sizes,
    })
    print(f"[distill] Merged {len(rows)} rows ({sizes['train']} train, {sizes['eval']} held out) "
          f"into {root}", flush=True)


def launch(args: argparse.Namespace, arguments: list[str]) -> int:
    """Pin one process to each selected physical GPU; merge only after all succeed."""
    import torch

    if not args.device.startswith("cuda"):
        raise ValueError("Parallel generation requires --device cuda")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [device.strip() for device in visible.split(",")] if visible is not None else [
        str(index) for index in range(torch.cuda.device_count())
    ]
    if (
        len(devices) < args.num_shards or any(not device for device in devices)
        or len(set(devices)) != len(devices) or torch.cuda.device_count() < args.num_shards
    ):
        raise ValueError(f"Need {args.num_shards} distinct visible CUDA devices; "
                         f"CUDA_VISIBLE_DEVICES={visible!r}, available={torch.cuda.device_count()}")
    root = args.data_root.resolve()
    previous = read_output_metadata(root)
    if previous is not None:
        if previous.get("status") == "generating":
            raise ValueError("This output contains an unfinished single-GPU run. "
                             "Select a new DISTILL_DATA_ROOT for parallel generation.")
        requested = {
            "model_name_or_path": args.model_name_or_path, "max_length": args.max_length,
            "dtype": args.dtype, "attn_implementation": args.attn_implementation,
            "source_dataset": args.source_dataset, "source_config": args.source_config,
            "source_splits": args.source_splits, "max_samples": args.max_samples,
            "eval_ratio": args.eval_ratio, "split_seed": args.seed,
        }
        changed = [key for key, value in requested.items() if previous.get(key) != value]
        if changed:
            raise ValueError(f"Output settings differ ({', '.join(changed)}). Use a new output directory.")
    work_root = root.with_name(root.name + ".shards")
    work_root.mkdir(parents=True, exist_ok=True)
    shard_roots = [work_root / f"shard_{index:02d}" for index in range(args.num_shards)]
    script = Path(__file__).with_name("generate_ultrachat_selfdistill.py")
    workers = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, shard_root in enumerate(shard_roots):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[index]}
            environment.setdefault("OMP_NUM_THREADS", "1")
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            log_path = work_root / f"shard_{index:02d}.log"
            command = [sys.executable, "-u", str(script), *arguments,
                       "--shard-index", str(index), "--data-root", str(shard_root), "--device", "cuda:0"]
            with log_path.open("a", encoding="utf-8") as log:
                worker = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            workers.append(worker)
            print(f"[distill] Shard {index}/{args.num_shards}: GPU {devices[index]}, "
                  f"PID {worker.pid}, log {log_path}", flush=True)
        pending = set(range(len(workers)))
        while pending:
            for index in sorted(pending):
                code = workers[index].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"Shard {index} failed (exit {code}); see "
                                       f"{work_root / f'shard_{index:02d}.log'}. "
                                       "Rerun the same command to resume completed rows.")
                pending.remove(index)
                print(f"[distill] Shard {index} finished ({args.num_shards - len(pending)}/{args.num_shards})", flush=True)
            if pending:
                time.sleep(0.5)
        merge_shards(root, shard_roots, local_files_only=args.local_files_only)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0
