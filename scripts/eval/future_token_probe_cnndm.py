#!/usr/bin/env python3
"""Future-token linear probes on CNN/DailyMail."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

# Support both direct script execution and package-style execution.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.future_token_probe_utils import add_common_args, run_probe_experiment


SUMMARY_PREFIX = "Summarize the following article: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future-token linear probes for CNN/DailyMail")
    add_common_args(parser)

    parser.add_argument("--dataset_path", type=str, default="abisee/cnn_dailymail")
    parser.add_argument("--dataset_config", type=str, default="3.0.0")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--source_prompt_text", type=str, default=SUMMARY_PREFIX)
    parser.add_argument("--target_prompt_text", type=str, default="Summary: ")
    parser.add_argument("--source_key", type=str, default="article")
    parser.add_argument("--target_key", type=str, default="highlights")
    parser.add_argument("--source_max_length_ratio", type=float, default=0.9)
    parser.add_argument("--target_max_length_ratio", type=float, default=0.1)
    return parser.parse_args()


def _maybe_limit(examples: List[Dict[str, object]], max_examples: int, seed: int) -> List[Dict[str, object]]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(examples)), max_examples))
    return [examples[i] for i in idx]


def _tokenize_example(tokenizer, source_text: str, target_text: str, args: argparse.Namespace) -> Tuple[List[int], int]:
    source = (args.source_prompt_text or "") + source_text
    target = (args.target_prompt_text or "") + target_text
    if tokenizer.bos_token is not None:
        source = tokenizer.bos_token + source
    if tokenizer.eos_token is not None:
        source = source + tokenizer.eos_token
        target = target + tokenizer.eos_token

    source_max_len = int(args.source_max_length_ratio * args.max_length)
    target_max_len = int(args.target_max_length_ratio * args.max_length)

    source_ids = tokenizer(source, add_special_tokens=False, truncation=True, max_length=source_max_len)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=target_max_len)["input_ids"]
    return source_ids + target_ids, len(source_ids)


def build_examples(tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_ds = load_dataset(args.dataset_path, args.dataset_config, split=args.train_split, trust_remote_code=True)
    eval_ds = load_dataset(args.dataset_path, args.dataset_config, split=args.eval_split, trust_remote_code=True)

    train_examples: List[Dict[str, object]] = []
    for ex in train_ds:
        input_ids, context_len = _tokenize_example(tokenizer, ex[args.source_key], ex[args.target_key], args)
        if len(input_ids) < 8:
            continue
        train_examples.append({"input_ids": input_ids, "context_len": context_len})

    eval_examples: List[Dict[str, object]] = []
    for ex in eval_ds:
        input_ids, context_len = _tokenize_example(tokenizer, ex[args.source_key], ex[args.target_key], args)
        if len(input_ids) < 8:
            continue
        eval_examples.append({"input_ids": input_ids, "context_len": context_len})

    train_examples = _maybe_limit(train_examples, args.max_train_examples, args.seed)
    eval_examples = _maybe_limit(eval_examples, args.max_eval_examples, args.seed + 1)
    return train_examples, eval_examples


def main() -> None:
    args = parse_args()
    run_probe_experiment(args, dataset_label="cnn_dailymail", build_examples_fn=build_examples)


if __name__ == "__main__":
    main()
