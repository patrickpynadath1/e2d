#!/usr/bin/env python3
"""Measure decoder degradation along blockwise normalized-latent flow paths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils import maybe_add_missing_special_tokens  # noqa: E402
from src.datasets.tokenize_on_demand import GSM8KDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path.home()
        / ".cache/e2d/outputs/diagnostics/qwen-latent-norms/normalization_stats.pt",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--target-layer-offset", type=int, default=-1)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--times", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / ".cache/e2d/outputs/diagnostics/qwen-latent-interpolation/results.json",
    )
    return parser.parse_args()


def report(message: str, started: float) -> None:
    print(f"[{time.monotonic() - started:8.1f}s] {message}", flush=True)


def additive_causal_mask(length: int, device: torch.device, dtype: torch.dtype):
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    return torch.where(allowed, 0.0, torch.finfo(dtype).min)[None, None].to(dtype)


def decode_final_layer(
    model: torch.nn.Module, residual: torch.Tensor, layer_index: int
) -> torch.Tensor:
    length = residual.shape[1]
    positions = torch.arange(length, device=residual.device)[None]
    position_embeddings = model.model.rotary_emb(residual, positions)
    decoded = model.model.layers[layer_index](
        residual,
        attention_mask=additive_causal_mask(length, residual.device, residual.dtype),
        position_ids=positions,
        use_cache=False,
        cache_position=positions[0],
        position_embeddings=position_embeddings,
    )[0]
    return model.lm_head(model.model.norm(decoded)).float()


def empty_accumulator(block_size: int) -> dict[str, Any]:
    return {
        "latent_squared_error_sum": 0.0,
        "latent_elements": 0.0,
        "kl_sum": 0.0,
        "clean_ce_sum": 0.0,
        "corrupted_ce_sum": 0.0,
        "token_count": 0.0,
        "block_count": 0.0,
        "accepted_length_sum": 0.0,
        "accepted_tokens": 0.0,
        "proposed_tokens": 0.0,
        "fully_accepted_blocks": 0.0,
        "speculative_block_count": 0.0,
        "accepted_length_histogram": [0] * (block_size + 1),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if not args.stats.is_file():
        raise FileNotFoundError(f"normalization statistics not found: {args.stats}")
    if args.num_samples < 1 or any(size < 1 for size in args.block_sizes):
        raise ValueError("sample count and block sizes must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.times):
        raise ValueError("interpolation times must be in [0, 1]")

    dtype = getattr(torch, args.dtype)
    report(f"Loading {args.model}", started)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    layer_index = (
        args.target_layer_offset
        if args.target_layer_offset >= 0
        else len(model.model.layers) + args.target_layer_offset
    )
    if layer_index != len(model.model.layers) - 1:
        raise ValueError(
            "this experiment currently expects the final transformer layer"
        )

    stats = torch.load(args.stats, map_location=args.device, weights_only=True)
    mean = stats["feature_mean"].float().to(args.device)
    std = stats["feature_std"].float().to(args.device).clamp_min(1e-6)
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        split=args.split,
        max_length=args.max_length,
        max_samples=args.num_samples,
        sampling_seed=args.seed,
    )
    accumulators = {
        (block_size, time_value): empty_accumulator(block_size)
        for block_size in args.block_sizes
        for time_value in args.times
    }
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for sample_index in range(len(dataset)):
        example = dataset[sample_index]
        valid = example["attention_mask"].bool()
        input_ids = example["input_ids"][valid].unsqueeze(0).to(args.device)
        context_mask = example["context_mask"][valid].bool().to(args.device)
        output = model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        clean_raw = output.hidden_states[layer_index].detach()
        clean = ((clean_raw.float() - mean) / std).to(dtype)
        clean_logits = decode_final_layer(model, clean_raw, layer_index)
        target_positions = torch.nonzero(~context_mask, as_tuple=False).flatten()
        # Logit i predicts token i+1, so the final sequence position has no target.
        target_positions = target_positions[target_positions < input_ids.shape[1] - 1]

        for block_size in args.block_sizes:
            for start in range(0, target_positions.numel(), block_size):
                block = target_positions[start : start + block_size]
                if not block.numel():
                    continue
                noise = torch.randn(
                    (1, block.numel(), clean.shape[-1]),
                    device=args.device,
                    dtype=dtype,
                    generator=generator,
                )
                clean_block = clean[:, block]
                token_targets = input_ids[:, block + 1]
                clean_block_logits = clean_logits[:, block]
                clean_log_probs = F.log_softmax(clean_block_logits, dim=-1)
                clean_probs = clean_log_probs.exp()
                clean_greedy = clean_block_logits.argmax(dim=-1)
                clean_ce = F.cross_entropy(
                    clean_block_logits.flatten(0, 1),
                    token_targets.flatten(),
                    reduction="sum",
                )

                for time_value in args.times:
                    noisy_block = (1.0 - time_value) * clean_block + time_value * noise
                    corrupted = clean.clone()
                    corrupted[:, block] = noisy_block
                    corrupted_raw = (corrupted.float() * std + mean).to(dtype)
                    corrupted_logits = decode_final_layer(
                        model, corrupted_raw, layer_index
                    )[:, block]
                    corrupted_log_probs = F.log_softmax(corrupted_logits, dim=-1)
                    kl = (clean_probs * (clean_log_probs - corrupted_log_probs)).sum(-1)
                    corrupted_ce = F.cross_entropy(
                        corrupted_logits.flatten(0, 1),
                        token_targets.flatten(),
                        reduction="sum",
                    )
                    accumulator = accumulators[(block_size, time_value)]
                    accumulator["latent_squared_error_sum"] += float(
                        (noisy_block.float() - clean_block.float()).square().sum()
                    )
                    accumulator["latent_elements"] += float(clean_block.numel())
                    accumulator["kl_sum"] += float(kl.sum())
                    accumulator["clean_ce_sum"] += float(clean_ce)
                    accumulator["corrupted_ce_sum"] += float(corrupted_ce)
                    accumulator["token_count"] += float(block.numel())
                    accumulator["block_count"] += 1.0
                    if block.numel() == block_size:
                        corrupted_greedy = corrupted_logits.argmax(dim=-1)
                        matches = (clean_greedy == corrupted_greedy)[0]
                        mismatch = torch.nonzero(~matches, as_tuple=False)
                        accepted_length = (
                            int(mismatch[0, 0]) if mismatch.numel() else block_size
                        )
                        accumulator["accepted_length_sum"] += accepted_length
                        accumulator["accepted_tokens"] += accepted_length
                        accumulator["proposed_tokens"] += block_size
                        accumulator["fully_accepted_blocks"] += int(
                            accepted_length == block_size
                        )
                        accumulator["speculative_block_count"] += 1
                        accumulator["accepted_length_histogram"][accepted_length] += 1
        report(f"Processed sample {sample_index + 1}/{len(dataset)}", started)

    rows = []
    for (block_size, time_value), accumulator in accumulators.items():
        token_count = accumulator["token_count"]
        rows.append(
            {
                "block_size": block_size,
                "time": time_value,
                "normalized_latent_mse": accumulator["latent_squared_error_sum"]
                / accumulator["latent_elements"],
                "clean_to_corrupted_kl": accumulator["kl_sum"] / token_count,
                "clean_token_ce": accumulator["clean_ce_sum"] / token_count,
                "corrupted_token_ce": accumulator["corrupted_ce_sum"] / token_count,
                "token_count": int(token_count),
                "block_count": int(accumulator["block_count"]),
                "speculative_full_block_count": int(
                    accumulator["speculative_block_count"]
                ),
                "expected_accepted_length": accumulator["accepted_length_sum"]
                / accumulator["speculative_block_count"],
                "acceptance_rate": accumulator["accepted_tokens"]
                / accumulator["proposed_tokens"],
                "full_block_acceptance_rate": accumulator["fully_accepted_blocks"]
                / accumulator["speculative_block_count"],
                "accepted_length_histogram": {
                    str(length): count
                    for length, count in enumerate(
                        accumulator["accepted_length_histogram"]
                    )
                },
            }
        )
    results = {
        "model": args.model,
        "stats": str(args.stats),
        "split": args.split,
        "num_samples": len(dataset),
        "target_layer_index": layer_index,
        "interpolation": "xt = (1 - t) * x0 + t * epsilon",
        "normalization": "per hidden coordinate",
        "logit_alignment": "latent/logit position i scores actual token i+1",
        "speculative_reference": "clean-logit greedy argmax",
        "speculative_mode": "teacher-forced clean context per block",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    report(f"Saved results to {args.output}", started)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
