#!/usr/bin/env python3
"""Calibrate per-coordinate Qwen residual statistics and compare with Gaussian noise.

The latent-flow model targets the residual stream entering Qwen's final transformer
layer. The latent-flow model uses dataset mean and standard deviation for every
hidden coordinate. This diagnostic saves those reusable statistics and reports the
held-out distribution actually seen during training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils import maybe_add_missing_special_tokens  # noqa: E402
from src.datasets.tokenize_on_demand import GSM8KDataset  # noqa: E402


def progress(message: str, started_at: float) -> None:
    """Emit an immediately flushed, timestamped calibration status line."""
    elapsed = time.monotonic() - started_at
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional calibration cap; by default the entire split is used.",
    )
    parser.add_argument(
        "--diagnostic-samples",
        type=int,
        default=256,
        help="Examples retained in memory for plots; all examples still fit stats.",
    )
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--target-layer-offset",
        type=int,
        default=-1,
        help="Transformer layer whose input residual is analyzed; -1 is final layer.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "E2D_OUTPUT_ROOT",
                Path.home() / ".cache/e2d/outputs",
            )
        )
        / "diagnostics/qwen-latent-norms",
    )
    return parser.parse_args()


def resolve_layer_index(offset: int, layer_count: int) -> int:
    index = offset if offset >= 0 else layer_count + offset
    if not 0 <= index < layer_count:
        raise ValueError(f"layer offset {offset} is invalid for {layer_count} layers")
    return index


def summarize_norms(vectors: torch.Tensor) -> dict[str, float | int]:
    vectors = vectors.float()
    l2 = vectors.norm(dim=-1)
    rms = vectors.square().mean(dim=-1).sqrt()
    quantiles = torch.quantile(l2, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    return {
        "num_vectors": vectors.shape[0],
        "dimension": vectors.shape[1],
        "coordinate_mean": vectors.mean().item(),
        "coordinate_std": vectors.std().item(),
        "l2_mean": l2.mean().item(),
        "l2_std": l2.std().item(),
        "l2_p01": quantiles[0].item(),
        "l2_p05": quantiles[1].item(),
        "l2_p50": quantiles[2].item(),
        "l2_p95": quantiles[3].item(),
        "l2_p99": quantiles[4].item(),
        "rms_mean": rms.mean().item(),
        "rms_std": rms.std().item(),
    }


def update_running_moments(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Merge a [N, D] batch into float64 Welford feature moments."""
    values = values.double()
    batch_count = values.shape[0]
    if batch_count == 0:
        return count, mean, m2
    batch_mean = values.mean(dim=0)
    batch_m2 = (values - batch_mean).square().sum(dim=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total)
    return total, merged_mean, merged_m2


def _comparison(
    raw: torch.Tensor, normalized: torch.Tensor, gaussian: torch.Tensor
) -> dict[str, Any]:
    raw_summary = summarize_norms(raw)
    normalized_summary = summarize_norms(normalized)
    gaussian_summary = summarize_norms(gaussian)
    return {
        "raw_qwen": raw_summary,
        "coordinate_normalized_qwen": normalized_summary,
        "standard_gaussian": gaussian_summary,
        "raw_to_gaussian_l2_mean_ratio": (
            raw_summary["l2_mean"] / gaussian_summary["l2_mean"]
        ),
        "normalized_to_gaussian_l2_mean_ratio": (
            normalized_summary["l2_mean"] / gaussian_summary["l2_mean"]
        ),
    }


def _plot_norms(
    raw: torch.Tensor,
    normalized: torch.Tensor,
    gaussian: torch.Tensor,
    output_path: Path,
) -> None:
    raw_l2 = raw.float().norm(dim=-1).numpy()
    normalized_l2 = normalized.float().norm(dim=-1).numpy()
    gaussian_l2 = gaussian.float().norm(dim=-1).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(raw_l2, bins=60, density=True, alpha=0.65, label="raw Qwen")
    axes[0].hist(gaussian_l2, bins=60, density=True, alpha=0.65, label="N(0, I)")
    axes[0].set_title("Raw Qwen residual vs Gaussian")
    axes[1].hist(
        normalized_l2,
        bins=60,
        density=True,
        alpha=0.65,
        label="coordinate-normalized Qwen",
    )
    axes[1].hist(gaussian_l2, bins=60, density=True, alpha=0.65, label="N(0, I)")
    axes[1].set_title("Actual flow target vs Gaussian")
    for axis in axes:
        axis.set_xlabel("Per-token L2 norm")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    progress(f"Starting calibration with arguments: {vars(args)}", started_at)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive when provided")
    if args.diagnostic_samples < 1:
        raise ValueError("--diagnostic-samples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device.startswith("cuda"):
        progress(
            f"CUDA available: {torch.cuda.get_device_name(torch.device(args.device))}",
            started_at,
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    progress(f"Loading tokenizer for {args.model}", started_at)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    progress(f"Tokenizer loaded (vocabulary size {len(tokenizer)})", started_at)
    progress(f"Loading {args.model} weights on CPU", started_at)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    progress("Model weights loaded; moving model to requested device", started_at)
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        allocated_gib = torch.cuda.memory_allocated(args.device) / 1024**3
        progress(f"Model moved to GPU ({allocated_gib:.2f} GiB allocated)", started_at)
    else:
        progress(f"Model moved to {args.device}", started_at)
    model.eval()
    layer_count = len(model.model.layers)
    layer_index = resolve_layer_index(args.target_layer_offset, layer_count)

    progress(f"Loading GSM8K {args.split} split", started_at)
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        split=args.split,
        max_length=args.max_length,
        max_samples=args.max_samples,
        sampling_seed=args.seed,
    )
    progress(f"Dataset ready with {len(dataset)} examples", started_at)
    hidden_size = model.config.hidden_size
    count = 0
    feature_mean = torch.zeros(hidden_size, dtype=torch.float64)
    feature_m2 = torch.zeros(hidden_size, dtype=torch.float64)
    diagnostic_latents: list[torch.Tensor] = []
    context_latents: list[torch.Tensor] = []
    answer_latents: list[torch.Tensor] = []
    for index in range(len(dataset)):
        if index == 0:
            progress("Fetching and tokenizing the first example", started_at)
        example = dataset[index]
        input_ids = example["input_ids"].unsqueeze(0).to(args.device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(args.device)
        context_mask = example["context_mask"].bool()
        if index == 0:
            progress(
                f"First example ready ({input_ids.shape[1]} tokens); "
                "starting first Qwen forward pass",
                started_at,
            )
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if index == 0:
            progress("First Qwen forward pass completed", started_at)
        # hidden_states[i] is the residual stream entering transformer layer i.
        latent = output.hidden_states[layer_index][0].float().cpu()
        valid = example["attention_mask"].bool()
        valid_latents = latent[valid]
        count, feature_mean, feature_m2 = update_running_moments(
            count, feature_mean, feature_m2, valid_latents
        )
        if index < args.diagnostic_samples:
            diagnostic_latents.append(valid_latents)
            context_latents.append(latent[valid & context_mask])
            answer_latents.append(latent[valid & ~context_mask])
        elapsed = max(time.monotonic() - started_at, 1e-6)
        examples_per_second = (index + 1) / elapsed
        progress(
            f"Processed example {index + 1}/{len(dataset)} "
            f"({int(valid.sum())} valid tokens; "
            f"{examples_per_second:.2f} examples/s)",
            started_at,
        )

    progress(f"Finished model pass over {len(dataset)} examples", started_at)
    if count < 2:
        raise RuntimeError("calibration requires at least two valid token vectors")
    feature_std = (feature_m2 / (count - 1)).sqrt().clamp_min(1e-6)
    feature_mean = feature_mean.float()
    feature_std = feature_std.float()
    diagnostic_all = torch.cat(diagnostic_latents)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    groups = {
        "diagnostic_all_tokens": diagnostic_all,
        "context_tokens": torch.cat(context_latents),
        "answer_tokens": torch.cat(answer_latents),
    }
    results: dict[str, Any] = {
        "model": args.model,
        "split": args.split,
        "calibration_examples": len(dataset),
        "calibration_token_vectors": count,
        "diagnostic_examples": min(args.diagnostic_samples, len(dataset)),
        "max_length": args.max_length,
        "layer_count": layer_count,
        "target_layer_index_zero_based": layer_index,
        "target_layer_number_one_based": layer_index + 1,
        "hidden_dimension": hidden_size,
        "normalization": "hidden_coordinate",
        "groups": {},
    }
    normalized_groups: dict[str, torch.Tensor] = {}
    gaussian_groups: dict[str, torch.Tensor] = {}
    for name, raw in groups.items():
        normalized = (raw - feature_mean) / feature_std
        gaussian = torch.randn(raw.shape, dtype=torch.float32, generator=generator)
        normalized_groups[name] = normalized
        gaussian_groups[name] = gaussian
        results["groups"][name] = _comparison(raw, normalized, gaussian)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    arrays_path = args.output_dir / "norms.npz"
    stats_path = args.output_dir / "normalization_stats.pt"
    plot_path = args.output_dir / "norm_histograms.png"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        arrays_path,
        raw_diagnostic_l2=groups["diagnostic_all_tokens"].norm(dim=-1).numpy(),
        normalized_diagnostic_l2=normalized_groups["diagnostic_all_tokens"]
        .norm(dim=-1)
        .numpy(),
        gaussian_diagnostic_l2=gaussian_groups["diagnostic_all_tokens"]
        .norm(dim=-1)
        .numpy(),
        raw_context_l2=groups["context_tokens"].norm(dim=-1).numpy(),
        raw_answer_l2=groups["answer_tokens"].norm(dim=-1).numpy(),
    )
    torch.save(
        {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "num_vectors": count,
            "num_examples": len(dataset),
            "model": args.model,
            "target_layer_index": layer_index,
        },
        stats_path,
    )
    progress(f"Saved normalization artifact to {stats_path}", started_at)
    _plot_norms(
        groups["diagnostic_all_tokens"],
        normalized_groups["diagnostic_all_tokens"],
        gaussian_groups["diagnostic_all_tokens"],
        plot_path,
    )

    print(json.dumps(results, indent=2))
    print(f"Saved summary: {json_path}")
    print(f"Saved norm arrays: {arrays_path}")
    print(f"Saved reusable statistics: {stats_path}")
    print(f"Saved histogram: {plot_path}")


if __name__ == "__main__":
    main()
