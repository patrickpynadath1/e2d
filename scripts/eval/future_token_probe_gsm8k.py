#!/usr/bin/env python3
"""Train linear probes on GSM8K hidden states for future-token prediction.

Modes:
- `--model_type compare`: compare AR vs E2D at the same hidden-state index.
- `--model_type e2d|ar|mtp`: train probes on one selected model.

This script intentionally uses the existing local checkpoint loader to stay compatible
with Composer checkpoints and cached weights-only files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# Allow running from repo root or directly from scripts/eval.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils import load_model_from_ckpt_dir_path, maybe_add_missing_special_tokens


QUESTION_PREFIX = "Please reason step by step, and put your final answer within $\\boxed{}$. "
ANSWER_PREFIX = "Answer: "


@dataclass
class ProbeMetrics:
    horizon: int
    train_loss: float
    eval_loss: float
    train_acc: float
    eval_acc: float
    num_train_samples: int
    num_eval_samples: int
    num_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future-token linear probes for GSM8K")
    parser.add_argument(
        "--model_type",
        type=str,
        default="compare",
        choices=["compare", "e2d", "ar", "mtp"],
        help="`compare` keeps old AR-vs-E2D behavior; choose a single model otherwise.",
    )
    parser.add_argument("--e2d_ckpt_dir", type=str, default=None)
    parser.add_argument("--ar_ckpt_dir", type=str, default=None)
    parser.add_argument("--mtp_ckpt_dir", type=str, default=None)
    parser.add_argument("--ckpt_file", type=str, default="best-rank0.pt")
    parser.add_argument("--load_ema_e2d", action="store_true", default=True)
    parser.add_argument("--load_ema_ar", action="store_true", default=True)
    parser.add_argument("--load_ema_mtp", action="store_true", default=True)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--hidden_state_index",
        type=int,
        default=-1,
        help=(
            "Hidden-state index to probe. Use -1 for auto: E2D uses inferred l', "
            "AR/MTP use --ar_mtp_default_layer (1-based layer number)."
        ),
    )
    parser.add_argument(
        "--ar_mtp_default_layer",
        type=int,
        default=26,
        help=(
            "Default 1-based layer number for AR/MTP when hidden_state_index=-1. "
            "For 28-layer models, 26 means using the output of layer 26."
        ),
    )

    parser.add_argument("--dataset_path", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")

    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_train_examples", type=int, default=7000)
    parser.add_argument("--max_eval_examples", type=int, default=1000)
    parser.add_argument("--max_positions_per_example", type=int, default=64)
    parser.add_argument("--target_only_positions", action="store_true", default=False)

    parser.add_argument("--horizons", type=str, default="1,2,3,4,5")
    parser.add_argument("--probe_vocab_size", type=int, default=8192)
    parser.add_argument("--max_train_samples_per_horizon", type=int, default=200000)
    parser.add_argument("--max_eval_samples_per_horizon", type=int, default=50000)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="probe_outputs")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_horizons(horizons_str: str) -> List[int]:
    horizons = []
    for x in horizons_str.split(","):
        x = x.strip()
        if not x:
            continue
        horizons.append(int(x))
    horizons = sorted(set(horizons))
    if any(h < 1 for h in horizons):
        raise ValueError("All horizons must be >= 1")
    return horizons


def _format_answer(answer: str) -> str:
    # Match existing GSM8K formatting in this repo.
    return re.sub(r"^####\s*(\d+)\s*$", r"$\\boxed{\1}$", answer, flags=re.MULTILINE)


def tokenize_gsm8k_example(
    tokenizer,
    question: str,
    answer: str,
    max_length: int,
) -> Tuple[List[int], int]:
    source = f"{tokenizer.bos_token}{QUESTION_PREFIX}{question}{tokenizer.eos_token}"
    target = f"{ANSWER_PREFIX}{_format_answer(answer)}{tokenizer.eos_token}"

    source_ids = tokenizer(source, add_special_tokens=False, truncation=True, max_length=max_length // 2)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=max_length // 2)["input_ids"]
    input_ids = source_ids + target_ids
    return input_ids, len(source_ids)


def build_examples(
    tokenizer,
    dataset_path: str,
    dataset_config: str,
    split: str,
    max_examples: int,
    max_length: int,
    seed: int,
) -> List[Dict[str, object]]:
    ds = load_dataset(dataset_path, dataset_config, split=split, trust_remote_code=True)
    indices = list(range(len(ds)))
    if max_examples > 0 and len(indices) > max_examples:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_examples))

    out = []
    for i in indices:
        ex = ds[i]
        input_ids, context_len = tokenize_gsm8k_example(
            tokenizer=tokenizer,
            question=ex["question"],
            answer=ex["answer"],
            max_length=max_length,
        )
        if len(input_ids) < 8:
            continue
        out.append(
            {
                "input_ids": input_ids,
                "context_len": context_len,
            }
        )
    return out


def sample_positions(
    examples: List[Dict[str, object]],
    horizons: List[int],
    max_positions_per_example: int,
    target_only_positions: bool,
    seed: int,
) -> List[List[int]]:
    rng = random.Random(seed)
    max_h = max(horizons)
    positions_all: List[List[int]] = []
    for ex in examples:
        ids = ex["input_ids"]
        context_len = int(ex["context_len"])
        max_t = len(ids) - 1 - max_h
        if max_t < 0:
            positions_all.append([])
            continue

        if target_only_positions:
            start = max(0, context_len - 1)
        else:
            start = 0
        cand = list(range(start, max_t + 1))
        if max_positions_per_example > 0 and len(cand) > max_positions_per_example:
            cand = sorted(rng.sample(cand, max_positions_per_example))
        positions_all.append(cand)
    return positions_all


def build_probe_vocab(
    examples: List[Dict[str, object]],
    positions: List[List[int]],
    horizons: List[int],
    probe_vocab_size: int,
) -> Dict[int, Dict[int, int]]:
    counters = {h: Counter() for h in horizons}
    for ex, pos_list in zip(examples, positions):
        ids = ex["input_ids"]
        for t in pos_list:
            for h in horizons:
                y = ids[t + h]
                counters[h][y] += 1

    vocab_maps: Dict[int, Dict[int, int]] = {}
    for h in horizons:
        common = counters[h].most_common(probe_vocab_size)
        vocab_maps[h] = {tok: idx for idx, (tok, _) in enumerate(common)}
    return vocab_maps


def infer_lprime_hidden_index(e2d_model) -> int:
    if not hasattr(e2d_model.backbone, "decoder_layer_idxs"):
        raise ValueError("E2D backbone does not expose decoder_layer_idxs; cannot infer l'.")
    decoder_layer_idxs = sorted(list(e2d_model.backbone.decoder_layer_idxs))
    if len(decoder_layer_idxs) == 0:
        raise ValueError("E2D decoder_layer_idxs is empty.")

    # hidden_states[0] is embeddings, hidden_states[k] is output after layer (k-1).
    # If decoder starts at layer j, the final encoder-below-decoder layer is (j-1),
    # corresponding to hidden_states index j.
    j = decoder_layer_idxs[0]
    if j <= 0:
        raise ValueError("Decoder starts at layer 0; no encoder-below-decoder layer exists.")
    return j


def infer_last_hidden_state_index(model) -> int:
    model_cfg = getattr(getattr(model.backbone, "model", None), "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model_cfg, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(
        "Could not infer final hidden-state index from model config. "
        "Please pass --hidden_state_index explicitly."
    )


def infer_ar_mtp_default_hidden_index(model, default_layer_1based: int) -> int:
    if default_layer_1based < 1:
        raise ValueError("--ar_mtp_default_layer must be >= 1.")

    max_hidden_index = infer_last_hidden_state_index(model)
    if default_layer_1based > max_hidden_index:
        raise ValueError(
            f"--ar_mtp_default_layer={default_layer_1based} exceeds model layers "
            f"({max_hidden_index})."
        )

    # hidden_states[0] is embeddings, and hidden_states[k] is output after layer k.
    # Therefore 1-based layer number N maps to hidden_state_index N.
    return default_layer_1based


def validate_args(args: argparse.Namespace) -> None:
    if args.model_type == "compare":
        if not args.e2d_ckpt_dir or not args.ar_ckpt_dir:
            raise ValueError("--model_type compare requires --e2d_ckpt_dir and --ar_ckpt_dir.")
        return

    required_ckpt = {
        "e2d": args.e2d_ckpt_dir,
        "ar": args.ar_ckpt_dir,
        "mtp": args.mtp_ckpt_dir,
    }[args.model_type]
    if not required_ckpt:
        raise ValueError(
            f"--model_type {args.model_type} requires the matching checkpoint dir argument."
        )


def get_hidden_states_at_layer(
    model_name: str,
    model,
    input_ids: torch.Tensor,
    hidden_state_index: int,
    device: torch.device,
) -> torch.Tensor:
    attn_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        if model_name == "e2d":
            outputs = model.backbone.encoder.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        elif model_name in {"ar", "mtp"}:
            outputs = model.backbone.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    hs = outputs.hidden_states
    if hidden_state_index >= len(hs):
        raise ValueError(
            f"Requested hidden_state_index={hidden_state_index}, "
            f"but model has only {len(hs)-1} layers (hidden_states len={len(hs)})."
        )
    return hs[hidden_state_index].squeeze(0).float().cpu()


def maybe_subsample(
    x: torch.Tensor,
    y: torch.Tensor,
    max_samples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:max_samples]
    return x[idx], y[idx]


def extract_probe_features(
    model_name: str,
    model,
    examples: List[Dict[str, object]],
    positions: List[List[int]],
    horizons: List[int],
    vocab_maps: Dict[int, Dict[int, int]],
    hidden_state_index: int,
    max_samples_per_horizon: int,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    x_store: Dict[int, List[torch.Tensor]] = {h: [] for h in horizons}
    y_store: Dict[int, List[int]] = {h: [] for h in horizons}

    model.eval()
    for ex, pos_list in zip(examples, positions):
        if len(pos_list) == 0:
            continue
        ids = torch.tensor(ex["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        hs = get_hidden_states_at_layer(
            model_name=model_name,
            model=model,
            input_ids=ids,
            hidden_state_index=hidden_state_index,
            device=device,
        )
        ids_list = ex["input_ids"]

        for t in pos_list:
            x_t = hs[t]
            for h in horizons:
                y_tok = ids_list[t + h]
                vocab = vocab_maps[h]
                if y_tok not in vocab:
                    continue
                x_store[h].append(x_t)
                y_store[h].append(vocab[y_tok])

    out_x: Dict[int, torch.Tensor] = {}
    out_y: Dict[int, torch.Tensor] = {}
    for h in horizons:
        if len(x_store[h]) == 0:
            out_x[h] = torch.empty(0, 1)
            out_y[h] = torch.empty(0, dtype=torch.long)
            continue
        x_tensor = torch.stack(x_store[h], dim=0)
        y_tensor = torch.tensor(y_store[h], dtype=torch.long)
        x_tensor, y_tensor = maybe_subsample(
            x_tensor,
            y_tensor,
            max_samples=max_samples_per_horizon,
            seed=seed + h,
        )
        out_x[h] = x_tensor
        out_y[h] = y_tensor
    return out_x, out_y


def evaluate_probe(
    probe: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    if x.shape[0] == 0:
        return float("nan"), float("nan")
    probe.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total = 0
    total_correct = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = ce(logits, yb)
            total_loss += float(loss.item())
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == yb).sum().item())
            total += int(yb.numel())

    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, float, float, float, float]:
    if x_train.shape[0] == 0:
        raise ValueError("No training samples available for this horizon.")

    g = torch.Generator()
    g.manual_seed(seed)

    probe = nn.Linear(x_train.shape[1], num_classes, bias=True).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    for _ in range(epochs):
        probe.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

    train_loss, train_acc = evaluate_probe(probe, x_train, y_train, batch_size, device)
    eval_loss, eval_acc = evaluate_probe(probe, x_eval, y_eval, batch_size, device)
    return probe, train_loss, eval_loss, train_acc, eval_acc


def run_for_model(
    model_name: str,
    model,
    hidden_state_index: int,
    horizons: List[int],
    train_examples: List[Dict[str, object]],
    eval_examples: List[Dict[str, object]],
    train_positions: List[List[int]],
    eval_positions: List[List[int]],
    vocab_maps: Dict[int, Dict[int, int]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[int, ProbeMetrics]:
    print(f"\n[{model_name}] Extracting train features...")
    x_train, y_train = extract_probe_features(
        model_name=model_name,
        model=model,
        examples=train_examples,
        positions=train_positions,
        horizons=horizons,
        vocab_maps=vocab_maps,
        hidden_state_index=hidden_state_index,
        max_samples_per_horizon=args.max_train_samples_per_horizon,
        seed=args.seed,
        device=device,
    )

    print(f"[{model_name}] Extracting eval features...")
    x_eval, y_eval = extract_probe_features(
        model_name=model_name,
        model=model,
        examples=eval_examples,
        positions=eval_positions,
        horizons=horizons,
        vocab_maps=vocab_maps,
        hidden_state_index=hidden_state_index,
        max_samples_per_horizon=args.max_eval_samples_per_horizon,
        seed=args.seed + 1000,
        device=device,
    )

    out: Dict[int, ProbeMetrics] = {}
    for h in horizons:
        num_classes = len(vocab_maps[h])
        print(
            f"[{model_name}] horizon=t+{h}: train={x_train[h].shape[0]} eval={x_eval[h].shape[0]} classes={num_classes}"
        )
        _, train_loss, eval_loss, train_acc, eval_acc = train_probe(
            x_train=x_train[h],
            y_train=y_train[h],
            x_eval=x_eval[h],
            y_eval=y_eval[h],
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + h,
            device=device,
        )
        out[h] = ProbeMetrics(
            horizon=h,
            train_loss=train_loss,
            eval_loss=eval_loss,
            train_acc=train_acc,
            eval_acc=eval_acc,
            num_train_samples=int(x_train[h].shape[0]),
            num_eval_samples=int(x_eval[h].shape[0]),
            num_classes=num_classes,
        )
    return out


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    horizons = parse_horizons(args.horizons)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    print("Preparing GSM8K examples...")
    train_examples = build_examples(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.train_split,
        max_examples=args.max_train_examples,
        max_length=args.max_length,
        seed=args.seed,
    )
    eval_examples = build_examples(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        max_examples=args.max_eval_examples,
        max_length=args.max_length,
        seed=args.seed + 1,
    )

    train_positions = sample_positions(
        examples=train_examples,
        horizons=horizons,
        max_positions_per_example=args.max_positions_per_example,
        target_only_positions=args.target_only_positions,
        seed=args.seed,
    )
    eval_positions = sample_positions(
        examples=eval_examples,
        horizons=horizons,
        max_positions_per_example=args.max_positions_per_example,
        target_only_positions=args.target_only_positions,
        seed=args.seed + 1,
    )

    vocab_maps = build_probe_vocab(
        examples=train_examples,
        positions=train_positions,
        horizons=horizons,
        probe_vocab_size=args.probe_vocab_size,
    )

    if args.model_type == "compare":
        print("Loading E2D model...")
        e2d_model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=args.e2d_ckpt_dir,
            ckpt_file=args.ckpt_file,
            load_ema_weights=args.load_ema_e2d,
            device=device,
        )
        e2d_model.eval()

        if args.hidden_state_index >= 0:
            hidden_state_index = args.hidden_state_index
            print(f"Using user-specified hidden_state_index={hidden_state_index}.")
        else:
            hidden_state_index = infer_lprime_hidden_index(e2d_model)
            lprime_layer = hidden_state_index - 1
            lprime_layer_1based = hidden_state_index
            print(
                "Inferred l' (final encoder-below-decoder layer): "
                f"0-based={lprime_layer}, 1-based={lprime_layer_1based} "
                f"(hidden_state_index={hidden_state_index})."
            )

        e2d_metrics = run_for_model(
            model_name="e2d",
            model=e2d_model,
            hidden_state_index=hidden_state_index,
            horizons=horizons,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_positions=train_positions,
            eval_positions=eval_positions,
            vocab_maps=vocab_maps,
            args=args,
            device=device,
        )

        del e2d_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading AR model...")
        ar_model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=args.ar_ckpt_dir,
            ckpt_file=args.ckpt_file,
            load_ema_weights=args.load_ema_ar,
            device=device,
        )
        ar_model.eval()

        ar_metrics = run_for_model(
            model_name="ar",
            model=ar_model,
            hidden_state_index=hidden_state_index,
            horizons=horizons,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_positions=train_positions,
            eval_positions=eval_positions,
            vocab_maps=vocab_maps,
            args=args,
            device=device,
        )

        rows = []
        for h in horizons:
            rows.append(
                {
                    "horizon": h,
                    "ar_eval_acc": ar_metrics[h].eval_acc,
                    "e2d_eval_acc": e2d_metrics[h].eval_acc,
                    "delta_eval_acc": e2d_metrics[h].eval_acc - ar_metrics[h].eval_acc,
                    "ar_eval_loss": ar_metrics[h].eval_loss,
                    "e2d_eval_loss": e2d_metrics[h].eval_loss,
                    "delta_eval_loss": e2d_metrics[h].eval_loss - ar_metrics[h].eval_loss,
                    "ar_num_eval_samples": ar_metrics[h].num_eval_samples,
                    "e2d_num_eval_samples": e2d_metrics[h].num_eval_samples,
                    "num_classes": ar_metrics[h].num_classes,
                }
            )

        result = {
            "config": vars(args),
            "model_type": "compare",
            "horizons": horizons,
            "hidden_state_index": hidden_state_index,
            "lprime_layer": hidden_state_index - 1,
            "lprime_layer_1based": hidden_state_index,
            "ar": {h: ar_metrics[h].__dict__ for h in horizons},
            "e2d": {h: e2d_metrics[h].__dict__ for h in horizons},
            "summary": rows,
        }

        out_json = os.path.join(args.output_dir, "future_token_probe_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        out_csv = os.path.join(args.output_dir, "future_token_probe_summary.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            header = [
                "horizon",
                "ar_eval_acc",
                "e2d_eval_acc",
                "delta_eval_acc",
                "ar_eval_loss",
                "e2d_eval_loss",
                "delta_eval_loss",
                "ar_num_eval_samples",
                "e2d_num_eval_samples",
                "num_classes",
            ]
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in header) + "\n")

        print("\n=== Future-Token Probe Summary (AR vs E2D, Eval) ===")
        for r in rows:
            print(
                f"t+{r['horizon']}: "
                f"AR_acc={r['ar_eval_acc']:.4f} | "
                f"E2D_acc={r['e2d_eval_acc']:.4f} | "
                f"Delta={r['delta_eval_acc']:+.4f}"
            )
        print(f"\nSaved: {out_json}")
        print(f"Saved: {out_csv}")
        return

    selected_model_name = args.model_type
    selected_ckpt_dir = {
        "e2d": args.e2d_ckpt_dir,
        "ar": args.ar_ckpt_dir,
        "mtp": args.mtp_ckpt_dir,
    }[selected_model_name]
    selected_load_ema = {
        "e2d": args.load_ema_e2d,
        "ar": args.load_ema_ar,
        "mtp": args.load_ema_mtp,
    }[selected_model_name]

    print(f"Loading {selected_model_name.upper()} model...")
    model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=selected_ckpt_dir,
        ckpt_file=args.ckpt_file,
        load_ema_weights=selected_load_ema,
        device=device,
    )
    model.eval()

    lprime_layer: Optional[int] = None
    if args.hidden_state_index >= 0:
        hidden_state_index = args.hidden_state_index
        print(f"Using user-specified hidden_state_index={hidden_state_index}.")
    elif selected_model_name == "e2d":
        hidden_state_index = infer_lprime_hidden_index(model)
        lprime_layer = hidden_state_index - 1
        lprime_layer_1based = hidden_state_index
        print(
            "Inferred l' (final encoder-below-decoder layer): "
            f"0-based={lprime_layer}, 1-based={lprime_layer_1based} "
            f"(hidden_state_index={hidden_state_index})."
        )
    else:
        hidden_state_index = infer_ar_mtp_default_hidden_index(
            model=model,
            default_layer_1based=args.ar_mtp_default_layer,
        )
        print(
            "Auto-selected AR/MTP layer for "
            f"{selected_model_name.upper()}: layer={args.ar_mtp_default_layer} "
            f"(hidden_state_index={hidden_state_index})."
        )

    model_metrics = run_for_model(
        model_name=selected_model_name,
        model=model,
        hidden_state_index=hidden_state_index,
        horizons=horizons,
        train_examples=train_examples,
        eval_examples=eval_examples,
        train_positions=train_positions,
        eval_positions=eval_positions,
        vocab_maps=vocab_maps,
        args=args,
        device=device,
    )

    rows = []
    for h in horizons:
        rows.append(
            {
                "horizon": h,
                "model": selected_model_name,
                "eval_acc": model_metrics[h].eval_acc,
                "eval_loss": model_metrics[h].eval_loss,
                "num_eval_samples": model_metrics[h].num_eval_samples,
                "num_classes": model_metrics[h].num_classes,
            }
        )

    result = {
        "config": vars(args),
        "model_type": selected_model_name,
        "horizons": horizons,
        "hidden_state_index": hidden_state_index,
        "lprime_layer": lprime_layer,
        selected_model_name: {h: model_metrics[h].__dict__ for h in horizons},
        "summary": rows,
    }

    out_json = os.path.join(args.output_dir, "future_token_probe_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    out_csv = os.path.join(args.output_dir, "future_token_probe_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        header = ["horizon", "model", "eval_acc", "eval_loss", "num_eval_samples", "num_classes"]
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in header) + "\n")

    print(f"\n=== Future-Token Probe Summary ({selected_model_name.upper()}, Eval) ===")
    for r in rows:
        print(
            f"t+{r['horizon']}: "
            f"acc={r['eval_acc']:.4f} | "
            f"loss={r['eval_loss']:.4f}"
        )
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
