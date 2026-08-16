#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Dependencies are managed by uv. Select one accelerator extra before launching jobs.
# This file only supplies optional runtime environment settings.
export UV_NO_SYNC="${UV_NO_SYNC:-1}"

if [ -f "${HOME}/setup_discdiff.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/setup_discdiff.sh"
fi

# Prefer this checkout over any older e2d2 installation inherited from the image.
E2D_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${E2D_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -d /workspace ]; then
  E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-/workspace}"
  E2D_DEFAULT_CACHE_HOME="${E2D_STORAGE_ROOT}/cache/e2d"
else
  E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-${HOME}/.cache/e2d}"
  E2D_DEFAULT_CACHE_HOME="${E2D_STORAGE_ROOT}"
fi
export E2D_STORAGE_ROOT

export HF_HOME="${HF_HOME:-${E2D_STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# Some Runpod base images enable the optional legacy hf_transfer downloader.
# The locked environment uses huggingface-hub's standard downloader instead.
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "HuggingFace cache set to '${HF_HOME}'."

# Keep large artifacts on persistent storage when /workspace is mounted.
export E2D_CACHE_HOME="${E2D_CACHE_HOME:-${E2D_DEFAULT_CACHE_HOME}}"
export E2D_CHECKPOINT_ROOT="${E2D_CHECKPOINT_ROOT:-${E2D_CACHE_HOME}/checkpoints}"
export E2D_OUTPUT_ROOT="${E2D_OUTPUT_ROOT:-${E2D_STORAGE_ROOT}/outputs}"
export WANDB_DIR="${WANDB_DIR:-${E2D_STORAGE_ROOT}/wandb}"
mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${E2D_CACHE_HOME}" \
  "${E2D_CHECKPOINT_ROOT}" \
  "${E2D_OUTPUT_ROOT}" \
  "${WANDB_DIR}"
echo "E2D checkpoints will be stored under '${E2D_CHECKPOINT_ROOT}'."

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
