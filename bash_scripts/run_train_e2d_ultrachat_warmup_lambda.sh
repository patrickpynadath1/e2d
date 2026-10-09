#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Lambda schedule (independent of the learning-rate WARMUP_DURATION).
# Edit these defaults or override them with environment variables when launching.
DECODER_LOSS_LAMBDA_START=${DECODER_LOSS_LAMBDA_START:-0.0}
DECODER_LOSS_LAMBDA=${DECODER_LOSS_LAMBDA:-1.0} # Final drafting-loss weight
DECODER_LOSS_LAMBDA_KEEP_FIRST_RATIO=${DECODER_LOSS_LAMBDA_KEEP_FIRST_RATIO:-0.5}
DECODER_LOSS_LAMBDA_WARMUP_RATIO=${DECODER_LOSS_LAMBDA_WARMUP_RATIO:-0.8}

# Stage 1: hold START until KEEP_FIRST_RATIO of MAX_DURATION.
# Stage 2: linearly ramp START -> DECODER_LOSS_LAMBDA from KEEP_FIRST_RATIO
# to WARMUP_RATIO of MAX_DURATION. WARMUP_RATIO marks the end of the ramp.
# Stage 3: hold DECODER_LOSS_LAMBDA for the remaining training.
# For example, KEEP_FIRST_RATIO=0.3 and WARMUP_RATIO=0.7 means hold for the
# first 30%, ramp from 30% to 70%, then hold the final lambda for the last 30%.
# Require 0 <= KEEP_FIRST_RATIO <= WARMUP_RATIO <= 1 and 0 <= START <= final.
# Equal ratios switch immediately at that point; both 0 use final throughout.
# The existing loss normalization is preserved:
#   (next_token_loss + lambda * draft_loss) / (1 + lambda)
export DECODER_LOSS_LAMBDA
export DISTILL_RUN_TAG="${DISTILL_RUN_TAG:-${DISTILL_SOURCE_SLUG:-ultrachat}}_kf${DECODER_LOSS_LAMBDA_KEEP_FIRST_RATIO}_wu${DECODER_LOSS_LAMBDA_WARMUP_RATIO}_final${DECODER_LOSS_LAMBDA}"

echo "[Lambda Warmup] ${DECODER_LOSS_LAMBDA_START} -> ${DECODER_LOSS_LAMBDA}, keep first ratio=${DECODER_LOSS_LAMBDA_KEEP_FIRST_RATIO}, warmup end ratio=${DECODER_LOSS_LAMBDA_WARMUP_RATIO}, max duration=${MAX_DURATION:-1ep}"

# Reuse the existing dataset preparation, model, and training settings. Extra
# arguments are forwarded to Hydra (e.g. training.load_path=/path/to/checkpoint).
exec bash "${SCRIPT_DIR}/run_train_e2d_ultrachat.sh" \
  +composer/callbacks=decoder_loss_lambda_warmup \
  "composer.callbacks.decoder_loss_lambda_warmup.start_lambda=${DECODER_LOSS_LAMBDA_START}" \
  "composer.callbacks.decoder_loss_lambda_warmup.keep_first_ratio=${DECODER_LOSS_LAMBDA_KEEP_FIRST_RATIO}" \
  "composer.callbacks.decoder_loss_lambda_warmup.warmup_ratio=${DECODER_LOSS_LAMBDA_WARMUP_RATIO}" \
  "$@"