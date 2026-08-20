#!/usr/bin/env python3
"""Plot latent interpolation and simulated speculative-decoding results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--quality-output", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    return parser.parse_args()


def save_quality_plot(rows: list[dict], output: Path) -> None:
    times = [row["time"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = (
        ("normalized_latent_mse", "Normalized latent MSE"),
        ("clean_to_corrupted_kl", r"$D_{KL}(p_{clean}\,\Vert\,p_{dirty})$"),
        ("corrupted_token_ce", "Token cross-entropy"),
    )
    for axis, (key, label) in zip(axes, panels):
        axis.plot(times, [row[key] for row in rows], marker="o", linewidth=2)
        if key == "corrupted_token_ce":
            axis.axhline(
                rows[0]["clean_token_ce"],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="clean CE",
            )
            axis.legend(frameon=False)
        axis.set_xlabel("Flow interpolation time t")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Qwen final-layer sensitivity to normalized latent noise")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_acceptance_plot(rows: list[dict], output: Path) -> None:
    times = [row["time"] for row in rows]
    figure, left = plt.subplots(figsize=(6.4, 4.2))
    right = left.twinx()
    accepted = left.plot(
        times,
        [row["expected_accepted_length"] for row in rows],
        color="#2563eb",
        marker="o",
        linewidth=2.2,
        label="expected accepted length",
    )
    rates = right.plot(
        times,
        [row["full_block_acceptance_rate"] for row in rows],
        color="#dc2626",
        marker="s",
        linewidth=2.0,
        label="full-block acceptance",
    )
    left.axhline(3.0, color="#2563eb", linestyle=":", linewidth=1.2)
    left.set_xlabel("Flow interpolation time t")
    left.set_ylabel("Expected accepted tokens (block size 4)", color="#2563eb")
    right.set_ylabel("Full-block acceptance rate", color="#dc2626")
    left.set_ylim(0, 4.1)
    right.set_ylim(0, 1.02)
    left.grid(alpha=0.25)
    lines = accepted + rates
    left.legend(lines, [line.get_label() for line in lines], frameon=False)
    figure.suptitle("Simulated speculative acceptance from noised Qwen latents")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results = json.loads(args.input.read_text())
    rows = sorted(results["rows"], key=lambda row: row["time"])
    block_sizes = {row["block_size"] for row in rows}
    if block_sizes != {4}:
        raise ValueError(f"acceptance plot expects block size 4, got {block_sizes}")
    save_quality_plot(rows, args.quality_output)
    save_acceptance_plot(rows, args.acceptance_output)


if __name__ == "__main__":
    main()
