#!/usr/bin/env python3
"""Fail-fast diagnostics for an E2D2 Runpod or local environment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    storage_root = Path(os.environ.get("E2D_STORAGE_ROOT", "/workspace"))
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "storage_root": str(storage_root),
        "storage_exists": storage_root.exists(),
        "storage_writable": os.access(storage_root, os.W_OK),
        "hf_home": os.environ.get("HF_HOME"),
        "checkpoint_root": os.environ.get("E2D_CHECKPOINT_ROOT"),
        "output_root": os.environ.get("E2D_OUTPUT_ROOT"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
    }

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": (f"{properties.major}.{properties.minor}"),
                    "memory_gib": round(properties.total_memory / 1024**3, 2),
                }
            )
        report["devices"] = devices
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        value = torch.ones(1, device="cuda") * 2
        report["cuda_tensor_check"] = value.item()

    print(json.dumps(report, indent=2, sort_keys=True))

    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 is required")
    if os.environ.get("E2D_REQUIRE_CUDA", "0") == "1" and not torch.cuda.is_available():
        errors.append("CUDA is required but PyTorch cannot access a GPU")
    if not storage_root.exists():
        errors.append(f"storage root does not exist: {storage_root}")
    elif not os.access(storage_root, os.W_OK):
        errors.append(f"storage root is not writable: {storage_root}")

    if errors:
        print("Environment check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
