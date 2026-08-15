#!/usr/bin/env bash
set -euo pipefail

# Source this script from the repository root on a Runpod Pod.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export E2D_STORAGE_ROOT="${E2D_STORAGE_ROOT:-/workspace}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/e2d-venv}"

# shellcheck source=../setup_env.sh
source "${REPO_ROOT}/setup_env.sh"

CREDENTIALS_FILE="${E2D_CREDENTIALS_FILE:-${E2D_STORAGE_ROOT}/credentials/e2d.env}"
if [ -f "${CREDENTIALS_FILE}" ]; then
  # shellcheck source=/dev/null
  set -a
  source "${CREDENTIALS_FILE}"
  set +a
fi

mkdir -p \
  "${E2D_STORAGE_ROOT}/datasets" \
  "${E2D_STORAGE_ROOT}/credentials"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Warning: the Runpod checkout has uncommitted changes; leaving them untouched."
fi

echo "Using E2D2 commit $(git rev-parse --short HEAD)."
echo "Synchronizing the locked CUDA 12.8 environment..."
uv sync --frozen --extra cu128

E2D_REQUIRE_CUDA=1 uv run --frozen --extra cu128 \
  python scripts/check_runpod_environment.py

echo "Runpod environment is ready in '${REPO_ROOT}'."
