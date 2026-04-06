#!/usr/bin/env python3
"""Future-token linear probes on KodCode."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

# Support both direct script execution and package-style execution.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.future_token_probe_utils import add_common_args, run_probe_experiment


PROMPT_PREFIX = "You are an expert Python programmer. Solve the following problem.\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future-token linear probes for KodCode")
    add_common_args(parser)

    parser.add_argument("--dataset_path", type=str, default="KodCode/KodCode-V1-SFT-R1")
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default=None)
    return parser.parse_args()


def _build_prompt(question: str) -> str:
    return f"{PROMPT_PREFIX}{question.strip()}\n[BEGIN]\n"


def _maybe_limit(examples: List[Dict[str, object]], max_examples: int, seed: int) -> List[Dict[str, object]]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(examples)), max_examples))
    return [examples[i] for i in idx]


def _tokenize_example(tokenizer, question: str, solution: str, max_length: int) -> Tuple[List[int], int]:
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    source = bos + _build_prompt(question)
    target = solution + eos

    source_ids = tokenizer(source, add_special_tokens=False, truncation=True, max_length=max_length // 2)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=max_length // 2)["input_ids"]
    return source_ids + target_ids, len(source_ids)


def build_examples(tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    full_dataset = load_dataset(args.dataset_path, split="train", trust_remote_code=True)
    full_dataset = full_dataset.filter(lambda x: x.get("version", "") == "v1.1")

    rng = np.random.RandomState(args.sampling_seed)
    num_samples = len(full_dataset) // 2
    sampled_indices = sorted(rng.choice(len(full_dataset), size=num_samples, replace=False).tolist())
    full_dataset = full_dataset.select(sampled_indices)

    train_size = len(full_dataset) - args.test_size
    train_ds = full_dataset.select(range(train_size))
    eval_ds = full_dataset.select(range(train_size, len(full_dataset)))

    if args.difficulty is not None:
        train_ds = train_ds.filter(lambda x: x.get("gpt_difficulty", "") == args.difficulty)
        eval_ds = eval_ds.filter(lambda x: x.get("gpt_difficulty", "") == args.difficulty)

    train_examples: List[Dict[str, object]] = []
    for ex in train_ds:
        input_ids, context_len = _tokenize_example(tokenizer, ex["question"], ex["solution"], args.max_length)
        if len(input_ids) < 8:
            continue
        train_examples.append({"input_ids": input_ids, "context_len": context_len})

    eval_examples: List[Dict[str, object]] = []
    for ex in eval_ds:
        input_ids, context_len = _tokenize_example(tokenizer, ex["question"], ex["solution"], args.max_length)
        if len(input_ids) < 8:
            continue
        eval_examples.append({"input_ids": input_ids, "context_len": context_len})

    train_examples = _maybe_limit(train_examples, args.max_train_examples, args.seed)
    eval_examples = _maybe_limit(eval_examples, args.max_eval_examples, args.seed + 1)
    return train_examples, eval_examples


def main() -> None:
    args = parse_args()
    run_probe_experiment(args, dataset_label="kodcode", build_examples_fn=build_examples)


if __name__ == "__main__":
    main()
