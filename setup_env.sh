#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Dependencies are managed by uv. Run `uv sync --frozen` once before launching jobs.
# This file only supplies optional runtime environment settings.
if [ -f "${HOME}/setup_discdiff.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/setup_discdiff.sh"
fi

export HF_HOME="${HF_HOME:-${PWD}/.hf_cache}"
echo "HuggingFace cache set to '${HF_HOME}'."

# Keep large checkpoints on the cache mount rather than the workspace filesystem.
export E2D_CACHE_HOME="${E2D_CACHE_HOME:-${HOME}/.cache/e2d}"
export E2D_CHECKPOINT_ROOT="${E2D_CHECKPOINT_ROOT:-${E2D_CACHE_HOME}/checkpoints}"
mkdir -p "${E2D_CHECKPOINT_ROOT}"
echo "E2D checkpoints will be stored under '${E2D_CHECKPOINT_ROOT}'."

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
