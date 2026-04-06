#!/usr/bin/env python3
"""Future-token linear probes on ScienceQA (text-only closed-choice)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import concatenate_datasets, load_dataset

# Support both direct script execution and package-style execution.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.future_token_probe_utils import add_common_args, run_probe_experiment

SCIENCEQA_ANSWER_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
SCIENCEQA_PREFIX = (
    "The following is a multiple choice question. Think step by step and then "
    "give your final answer.\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future-token linear probes for ScienceQA")
    add_common_args(parser)

    parser.add_argument("--dataset_path", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--sampling_seed", type=int, default=1)
    parser.add_argument("--source_prompt_text", type=str, default=SCIENCEQA_PREFIX)
    parser.add_argument("--target_prompt_text", type=str, default="Answer: ")
    return parser.parse_args()


def _maybe_limit(examples: List[Dict[str, object]], max_examples: int, seed: int) -> List[Dict[str, object]]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(examples)), max_examples))
    return [examples[i] for i in idx]


def _format_question(question: str, choices: List[str], hint: str | None = None) -> str:
    parts = [question]
    if hint:
        parts.append(f"Hint: {hint}")
    for idx, choice in enumerate(choices):
        parts.append(f"({SCIENCEQA_ANSWER_LETTERS[idx]}) {choice}")
    return "\n".join(parts)


def _format_answer(solution: str, answer_idx: int) -> str:
    letter = SCIENCEQA_ANSWER_LETTERS[answer_idx]
    return f"{solution}\nThe answer is ({letter})."


def _tokenize_example(tokenizer, ex: Dict[str, object], max_length: int, source_prompt_text: str, target_prompt_text: str) -> Tuple[List[int], int]:
    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    q_text = _format_question(ex["question"], ex["choices"], ex.get("hint", ""))
    source = bos + source_prompt_text + q_text + eos
    target = target_prompt_text + _format_answer(ex["solution"], ex["answer"]) + eos

    source_ids = tokenizer(source, add_special_tokens=False, truncation=True, max_length=max_length // 2)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=max_length // 2)["input_ids"]
    return source_ids + target_ids, len(source_ids)


def build_examples(tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    def _is_text_only_closed_choice(example):
        return example["image"] is None and example["task"] == "closed choice"

    train_ds = load_dataset(args.dataset_path, split="train", trust_remote_code=True)
    val_ds = load_dataset(args.dataset_path, split="validation", trust_remote_code=True)
    train_ds = concatenate_datasets([train_ds, val_ds]).filter(_is_text_only_closed_choice)

    test_ds = load_dataset(args.dataset_path, split="test", trust_remote_code=True)
    test_ds = test_ds.filter(_is_text_only_closed_choice)
    if len(test_ds) > args.test_size:
        rng = np.random.RandomState(args.sampling_seed)
        indices = sorted(rng.choice(len(test_ds), size=args.test_size, replace=False).tolist())
        test_ds = test_ds.select(indices)

    train_examples: List[Dict[str, object]] = []
    for ex in train_ds:
        input_ids, context_len = _tokenize_example(
            tokenizer,
            ex,
            args.max_length,
            source_prompt_text=args.source_prompt_text,
            target_prompt_text=args.target_prompt_text,
        )
        if len(input_ids) < 8:
            continue
        train_examples.append({"input_ids": input_ids, "context_len": context_len})

    eval_examples: List[Dict[str, object]] = []
    for ex in test_ds:
        input_ids, context_len = _tokenize_example(
            tokenizer,
            ex,
            args.max_length,
            source_prompt_text=args.source_prompt_text,
            target_prompt_text=args.target_prompt_text,
        )
        if len(input_ids) < 8:
            continue
        eval_examples.append({"input_ids": input_ids, "context_len": context_len})

    train_examples = _maybe_limit(train_examples, args.max_train_examples, args.seed)
    eval_examples = _maybe_limit(eval_examples, args.max_eval_examples, args.seed + 1)
    return train_examples, eval_examples


def main() -> None:
    args = parse_args()
    run_probe_experiment(args, dataset_label="scienceqa", build_examples_fn=build_examples)


if __name__ == "__main__":
    main()
