#!/usr/bin/env python3
"""Compare Qwen residual-stream norms with same-dimensional Gaussian noise.

The latent-flow model targets the residual stream entering Qwen's final transformer
layer. It normalizes those residuals with one scalar mean and standard deviation
before comparing them with N(0, I) noise. This diagnostic reports both raw and
scalar-normalized latent norms so it tests the distribution actually seen in training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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

from src.datasets.tokenize_on_demand import GSM8KDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=8)
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
        default=Path("/workspace/outputs/diagnostics/qwen-latent-norms"),
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


def _comparison(
    raw: torch.Tensor, normalized: torch.Tensor, gaussian: torch.Tensor
) -> dict[str, Any]:
    raw_summary = summarize_norms(raw)
    normalized_summary = summarize_norms(normalized)
    gaussian_summary = summarize_norms(gaussian)
    return {
        "raw_qwen": raw_summary,
        "scalar_normalized_qwen": normalized_summary,
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
        label="scalar-normalized Qwen",
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
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    layer_count = len(model.model.layers)
    layer_index = resolve_layer_index(args.target_layer_offset, layer_count)

    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        split=args.split,
        max_length=args.max_length,
        max_samples=args.num_samples,
        sampling_seed=args.seed,
    )
    all_latents: list[torch.Tensor] = []
    context_latents: list[torch.Tensor] = []
    answer_latents: list[torch.Tensor] = []
    for index in range(len(dataset)):
        example = dataset[index]
        input_ids = example["input_ids"].unsqueeze(0).to(args.device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(args.device)
        context_mask = example["context_mask"].bool()
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states[i] is the residual stream entering transformer layer i.
        latent = output.hidden_states[layer_index][0].float().cpu()
        valid = example["attention_mask"].bool()
        all_latents.append(latent[valid])
        context_latents.append(latent[valid & context_mask])
        answer_latents.append(latent[valid & ~context_mask])
        print(f"Processed sample {index + 1}/{len(dataset)} ({int(valid.sum())} tokens)")

    raw_all = torch.cat(all_latents)
    scalar_mean = raw_all.mean()
    scalar_std = raw_all.std().clamp_min(1e-6)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    groups = {
        "all_tokens": raw_all,
        "context_tokens": torch.cat(context_latents),
        "answer_tokens": torch.cat(answer_latents),
    }
    results: dict[str, Any] = {
        "model": args.model,
        "split": args.split,
        "num_samples": len(dataset),
        "max_length": args.max_length,
        "layer_count": layer_count,
        "target_layer_index_zero_based": layer_index,
        "target_layer_number_one_based": layer_index + 1,
        "hidden_dimension": raw_all.shape[-1],
        "scalar_normalization_mean": scalar_mean.item(),
        "scalar_normalization_std": scalar_std.item(),
        "groups": {},
    }
    normalized_groups: dict[str, torch.Tensor] = {}
    gaussian_groups: dict[str, torch.Tensor] = {}
    for name, raw in groups.items():
        normalized = (raw - scalar_mean) / scalar_std
        gaussian = torch.randn(
            raw.shape, dtype=torch.float32, generator=generator
        )
        normalized_groups[name] = normalized
        gaussian_groups[name] = gaussian
        results["groups"][name] = _comparison(raw, normalized, gaussian)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    arrays_path = args.output_dir / "norms.npz"
    plot_path = args.output_dir / "norm_histograms.png"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        arrays_path,
        raw_all_l2=groups["all_tokens"].norm(dim=-1).numpy(),
        normalized_all_l2=normalized_groups["all_tokens"].norm(dim=-1).numpy(),
        gaussian_all_l2=gaussian_groups["all_tokens"].norm(dim=-1).numpy(),
        raw_context_l2=groups["context_tokens"].norm(dim=-1).numpy(),
        raw_answer_l2=groups["answer_tokens"].norm(dim=-1).numpy(),
    )
    _plot_norms(
        groups["all_tokens"],
        normalized_groups["all_tokens"],
        gaussian_groups["all_tokens"],
        plot_path,
    )

    print(json.dumps(results, indent=2))
    print(f"Saved summary: {json_path}")
    print(f"Saved norm arrays: {arrays_path}")
    print(f"Saved histogram: {plot_path}")


if __name__ == "__main__":
    main()
