#!/usr/bin/env python3
"""Cross-layer linear CKA analysis for GSM8K AR vs LayerSkip checkpoints.

This script:
1. Loads a fixed-size GSM8K subset with the exact tokenization/formatting used by
   the training pipeline.
2. Builds full prompt+response token sequences.
3. Keeps response-token positions only.
4. Samples one shared set of response-token positions across all examples.
5. Extracts hidden states at all transformer layers for AR and LayerSkip.
6. Computes the full cross-layer linear CKA matrix (rows=AR, cols=LayerSkip).
7. Saves the raw matrix and a heatmap.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from transformers import AutoTokenizer

# Headless plotting (for remote servers).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from repo root or directly from scripts/eval.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils import load_model_from_ckpt_dir_path, maybe_add_missing_special_tokens
from src.datasets.tokenize_on_demand import GSM8KDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-layer linear CKA for GSM8K AR vs LayerSkip."
    )
    parser.add_argument("--layerskip_ckpt_dir", type=str, required=True)
    parser.add_argument("--ar_ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str, default="best-rank0.pt")
    parser.add_argument("--load_ema_layerskip", action="store_true", default=False)
    parser.add_argument("--load_ema_ar", action="store_true", default=False)

    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset_path", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--example_subset_seed", type=int, default=1)

    parser.add_argument("--num_sampled_tokens", type=int, default=20000)
    parser.add_argument("--token_sample_seed", type=int, default=7)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cka_device",
        type=str,
        default=None,
        help="Device for CKA math; defaults to --device.",
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="probe_outputs/gsm8k_linear_cka_ar_vs_layerskip",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--heatmap_dpi", type=int, default=180)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_tokenized_examples(
    tokenizer,
    dataset_path: str,
    dataset_config: str,
    split: str,
    max_length: int,
    num_examples: int,
    subset_seed: int,
) -> Tuple[List[Dict[str, List[int]]], List[int]]:
    """Build full prompt+response token sequences using the canonical GSM8K dataset class."""
    ds = GSM8KDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        dataset_path=dataset_path,
        config_name=dataset_config,
        padding=False,
        add_special_tokens=True,
        source_key="question",
        target_key="answer",
        num_shot=0,
    )

    n_total = len(ds)
    all_indices = list(range(n_total))
    if num_examples > 0 and num_examples < n_total:
        rng = random.Random(subset_seed)
        selected_indices = sorted(rng.sample(all_indices, num_examples))
    else:
        selected_indices = all_indices

    examples: List[Dict[str, List[int]]] = []
    kept_indices: List[int] = []
    for idx in selected_indices:
        item = ds[idx]
        input_ids = item["input_ids"].tolist()
        context_mask = item["context_mask"].tolist()
        response_positions = [i for i, m in enumerate(context_mask) if int(m) == 0]

        if not response_positions:
            continue
        examples.append(
            {
                "input_ids": input_ids,
                "response_positions": response_positions,
            }
        )
        kept_indices.append(idx)

    return examples, kept_indices


def sample_shared_response_positions(
    examples: Sequence[Dict[str, List[int]]],
    num_sampled_tokens: int,
    token_sample_seed: int,
) -> Tuple[Dict[int, List[int]], List[int], int]:
    """Sample one shared global set of response-token indices across all examples.

    Returns:
        - per-example list of local response-position indices
        - sampled global response-token indices
        - total number of response tokens before sampling
    """
    total_response_tokens = sum(len(ex["response_positions"]) for ex in examples)
    if total_response_tokens == 0:
        raise ValueError("No response tokens found in selected examples.")

    k = min(num_sampled_tokens, total_response_tokens)
    if k <= 0:
        raise ValueError("--num_sampled_tokens must be > 0.")

    if k == total_response_tokens:
        sampled_global = list(range(total_response_tokens))
    else:
        rng = random.Random(token_sample_seed)
        sampled_global = sorted(rng.sample(range(total_response_tokens), k))

    by_example: Dict[int, List[int]] = defaultdict(list)
    cursor = 0
    offset = 0
    for ex_idx, ex in enumerate(examples):
        n_resp = len(ex["response_positions"])
        while cursor < len(sampled_global) and sampled_global[cursor] < offset + n_resp:
            local_idx = sampled_global[cursor] - offset
            by_example[ex_idx].append(local_idx)
            cursor += 1
        offset += n_resp

    if cursor != len(sampled_global):
        raise RuntimeError("Failed to map all sampled global response positions.")

    return by_example, sampled_global, total_response_tokens


def forward_hidden_states(model_name: str, model, input_ids: torch.Tensor, device: torch.device):
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        if model_name in {"ar", "layerskip"}:
            # LayerSkip is an AR subclass and uses the same backbone path.
            outputs = model.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    return outputs.hidden_states


def collect_layerwise_response_hidden_states(
    model_name: str,
    model,
    examples: Sequence[Dict[str, List[int]]],
    sampled_local_resp_indices: Dict[int, List[int]],
    device: torch.device,
) -> List[torch.Tensor]:
    """Collect sampled response-token hidden states for every transformer layer.

    Returns a list with length `num_layers`, where each entry is shape [N, D].
    """
    model.eval()
    example_ids = sorted(sampled_local_resp_indices.keys())
    if not example_ids:
        raise ValueError("No sampled examples found to extract hidden states.")

    layer_chunks: List[List[torch.Tensor]] | None = None
    total_selected = 0

    for n_done, ex_idx in enumerate(example_ids, start=1):
        ex = examples[ex_idx]
        local_resp_idxs = sampled_local_resp_indices[ex_idx]
        if len(local_resp_idxs) == 0:
            continue

        seq_positions = [ex["response_positions"][i] for i in local_resp_idxs]
        seq_positions_t = torch.tensor(seq_positions, dtype=torch.long, device=device)

        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        hidden_states = forward_hidden_states(
            model_name=model_name,
            model=model,
            input_ids=input_ids,
            device=device,
        )

        # hidden_states[0] is embedding output; hidden_states[1:] are transformer layers.
        if layer_chunks is None:
            num_layers = len(hidden_states) - 1
            if num_layers <= 0:
                raise ValueError("No transformer hidden states found in model outputs.")
            layer_chunks = [[] for _ in range(num_layers)]

        for layer_idx in range(1, len(hidden_states)):
            h = hidden_states[layer_idx].squeeze(0)
            selected = h.index_select(0, seq_positions_t).float().cpu()
            layer_chunks[layer_idx - 1].append(selected)

        total_selected += len(seq_positions)
        if n_done % 100 == 0:
            print(f"[{model_name}] processed {n_done}/{len(example_ids)} sampled examples")

    if layer_chunks is None:
        raise RuntimeError("Failed to collect hidden states.")

    layer_mats: List[torch.Tensor] = []
    for i, chunks in enumerate(layer_chunks, start=1):
        if not chunks:
            raise RuntimeError(f"Layer {i} produced zero sampled tokens.")
        layer_mat = torch.cat(chunks, dim=0)
        layer_mats.append(layer_mat)

    for i, mat in enumerate(layer_mats, start=1):
        if mat.shape[0] != total_selected:
            raise RuntimeError(
                f"Layer {i} has {mat.shape[0]} rows, expected {total_selected}."
            )
    return layer_mats


def center_layers_in_place(layers: Sequence[torch.Tensor]) -> None:
    for i in range(len(layers)):
        layers[i].sub_(layers[i].mean(dim=0, keepdim=True))


def gram_fro_norm(x_centered: torch.Tensor, device: torch.device) -> float:
    x = x_centered.to(device=device, dtype=torch.float32)
    gram = x.transpose(0, 1).matmul(x)
    norm = torch.linalg.norm(gram, ord="fro")
    return float(norm.item())


def compute_linear_cka_matrix(
    ar_layers: Sequence[torch.Tensor],
    layerskip_layers: Sequence[torch.Tensor],
    cka_device: torch.device,
    eps: float,
) -> np.ndarray:
    print("Centering activations...")
    center_layers_in_place(ar_layers)
    center_layers_in_place(layerskip_layers)

    print("Computing per-layer self norms...")
    ar_norms = [gram_fro_norm(x, cka_device) for x in ar_layers]
    layerskip_norms = [gram_fro_norm(y, cka_device) for y in layerskip_layers]

    num_ar = len(ar_layers)
    num_layerskip = len(layerskip_layers)
    out = np.zeros((num_ar, num_layerskip), dtype=np.float64)

    for i in range(num_ar):
        x = ar_layers[i].to(device=cka_device, dtype=torch.float32)
        for j in range(num_layerskip):
            y = layerskip_layers[j].to(device=cka_device, dtype=torch.float32)
            cross = x.transpose(0, 1).matmul(y)
            numerator = float((cross * cross).sum().item())
            denom = max(ar_norms[i] * layerskip_norms[j], eps)
            out[i, j] = numerator / denom
        print(f"Computed CKA row {i + 1}/{num_ar}")

    out = np.clip(out, 0.0, 1.0)
    return out


def save_heatmap(
    cka: np.ndarray,
    out_path: str,
    dpi: int,
    tick_interval: int = 4,   # show every 4th layer index
) -> None:
    n_ar, n_layerskip = cka.shape

    # Make figure square.
    fig_size = 4.5
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        cka,
        aspect="equal",   # keep cells square
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("LayerSkip Layer", fontsize=11)
    ax.set_ylabel("AR Layer", fontsize=11)

    # Show ticks only at a chosen interval.
    xticks = np.unique(np.append(np.arange(0, n_layerskip, tick_interval), n_layerskip - 1))
    yticks = np.unique(np.append(np.arange(0, n_ar, tick_interval), n_ar - 1))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks + 1, fontsize=10)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks + 1, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Linear CKA", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    cka_device = resolve_device(args.cka_device) if args.cka_device else device

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    print("Building tokenized GSM8K examples (training/eval pipeline format)...")
    examples, subset_indices = build_tokenized_examples(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.split,
        max_length=args.max_length,
        num_examples=args.num_examples,
        subset_seed=args.example_subset_seed,
    )
    if not examples:
        raise ValueError("No examples available after tokenization.")
    print(f"Using {len(examples)} tokenized examples.")

    sampled_local_resp_indices, sampled_global_positions, total_response_tokens = (
        sample_shared_response_positions(
            examples=examples,
            num_sampled_tokens=args.num_sampled_tokens,
            token_sample_seed=args.token_sample_seed,
        )
    )
    n_sampled_tokens = len(sampled_global_positions)
    print(
        "Response tokens: "
        f"total={total_response_tokens}, sampled_shared={n_sampled_tokens}"
    )

    print("Loading AR model...")
    ar_model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=args.ar_ckpt_dir,
        ckpt_file=args.ckpt_file,
        load_ema_weights=args.load_ema_ar,
        device=device,
    )
    print("Collecting AR layerwise response hidden states...")
    ar_layers = collect_layerwise_response_hidden_states(
        model_name="ar",
        model=ar_model,
        examples=examples,
        sampled_local_resp_indices=sampled_local_resp_indices,
        device=device,
    )
    del ar_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading LayerSkip model...")
    layerskip_model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=args.layerskip_ckpt_dir,
        ckpt_file=args.ckpt_file,
        load_ema_weights=args.load_ema_layerskip,
        device=device,
    )
    print("Collecting LayerSkip layerwise response hidden states...")
    layerskip_layers = collect_layerwise_response_hidden_states(
        model_name="layerskip",
        model=layerskip_model,
        examples=examples,
        sampled_local_resp_indices=sampled_local_resp_indices,
        device=device,
    )
    del layerskip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i, mat in enumerate(ar_layers, start=1):
        if mat.shape[0] != n_sampled_tokens:
            raise RuntimeError(
                f"AR layer {i} has {mat.shape[0]} sampled tokens, expected {n_sampled_tokens}."
            )
    for i, mat in enumerate(layerskip_layers, start=1):
        if mat.shape[0] != n_sampled_tokens:
            raise RuntimeError(
                "LayerSkip layer "
                f"{i} has {mat.shape[0]} sampled tokens, expected {n_sampled_tokens}."
            )

    print("Computing cross-layer linear CKA matrix...")
    cka = compute_linear_cka_matrix(
        ar_layers=ar_layers,
        layerskip_layers=layerskip_layers,
        cka_device=cka_device,
        eps=args.eps,
    )

    matrix_npy = os.path.join(args.output_dir, "linear_cka_matrix.npy")
    matrix_csv = os.path.join(args.output_dir, "linear_cka_matrix.csv")
    heatmap_png = os.path.join(args.output_dir, "linear_cka_heatmap_ar_layerskip.pdf")
    meta_json = os.path.join(args.output_dir, "linear_cka_metadata.json")
    subset_idx_path = os.path.join(args.output_dir, "selected_dataset_indices.json")
    sampled_global_idx_path = os.path.join(
        args.output_dir,
        "sampled_response_global_indices.npy",
    )

    np.save(matrix_npy, cka)
    np.savetxt(matrix_csv, cka, delimiter=",")
    save_heatmap(cka=cka, out_path=heatmap_png, dpi=args.heatmap_dpi)

    with open(subset_idx_path, "w", encoding="utf-8") as f:
        json.dump(subset_indices, f, indent=2)
    np.save(sampled_global_idx_path, np.asarray(sampled_global_positions, dtype=np.int64))

    metadata = {
        "config": vars(args),
        "num_examples_used": len(examples),
        "num_sampled_tokens": n_sampled_tokens,
        "total_response_tokens_before_sampling": total_response_tokens,
        "ar_num_layers": len(ar_layers),
        "layerskip_num_layers": len(layerskip_layers),
        "ar_hidden_dim": int(ar_layers[0].shape[1]),
        "layerskip_hidden_dim": int(layerskip_layers[0].shape[1]),
        "matrix_shape": [int(cka.shape[0]), int(cka.shape[1])],
        "matrix_npy": matrix_npy,
        "matrix_csv": matrix_csv,
        "heatmap_png": heatmap_png,
        "selected_dataset_indices_json": subset_idx_path,
        "sampled_response_global_indices_npy": sampled_global_idx_path,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"Saved matrix: {matrix_npy}")
    print(f"Saved matrix CSV: {matrix_csv}")
    print(f"Saved heatmap: {heatmap_png}")
    print(f"Saved metadata: {meta_json}")


if __name__ == "__main__":
    main()