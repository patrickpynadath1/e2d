"""Optional warmup of the E2D drafting-loss coefficient."""

import math

from composer.core import Callback, State, TimeUnit
from composer.loggers import Logger


class DecoderLossLambdaWarmup(Callback):
    """Hold lambda at its start value, linearly ramp it, then hold its final value.

    ``keep_first_ratio`` and ``warmup_ratio`` are fractions of the full training
    duration marking the start and end of the ramp. Equal ratios switch directly
    from start to final lambda at that point. Progress comes from Composer's
    training timestamp, so microbatches and evaluation do not advance it, and
    checkpoint resumes continue with the same max_duration and callback settings.
    """

    def __init__(
        self,
        start_lambda: float,
        final_lambda: float,
        warmup_ratio: float,
        keep_first_ratio: float = 0.0,
    ) -> None:
        self.start_lambda = float(start_lambda)
        self.final_lambda = float(final_lambda)
        self.warmup_ratio = float(warmup_ratio)
        self.keep_first_ratio = float(keep_first_ratio)
        if not all(
            math.isfinite(value)
            for value in (
                self.start_lambda,
                self.final_lambda,
                self.warmup_ratio,
                self.keep_first_ratio,
            )
        ):
            raise ValueError("Lambda warmup settings must be finite.")
        if not 0 <= self.start_lambda <= self.final_lambda:
            raise ValueError("Expected 0 <= start_lambda <= final_lambda.")
        if not 0 <= self.keep_first_ratio <= self.warmup_ratio <= 1:
            raise ValueError("Expected 0 <= keep_first_ratio <= warmup_ratio <= 1.")

    def _current_lambda(self, state: State) -> float:
        if self.warmup_ratio == 0:
            return self.final_lambda

        duration = state.max_duration
        if duration is None or duration.value <= 0:
            raise ValueError("Lambda warmup requires a positive max_duration.")
        if duration.unit == TimeUnit.EPOCH:
            if state.dataloader_len is None or int(state.dataloader_len) <= 0:
                raise ValueError(
                    "Epoch-based lambda warmup requires a sized dataloader."
                )
            # get_elapsed_duration() counts only completed epochs. Include batches
            # within the current epoch so even a one-epoch run ramps smoothly.
            elapsed = (
                state.timestamp.epoch.value
                + state.timestamp.batch_in_epoch.value / int(state.dataloader_len)
            )
        elif duration.unit in (TimeUnit.BATCH, TimeUnit.SAMPLE, TimeUnit.TOKEN):
            elapsed = state.timestamp.get(duration.unit).value
        else:
            raise ValueError(
                "Lambda warmup supports max_duration in ep, ba, sp, or tok."
            )

        progress = elapsed / duration.value
        if progress >= self.warmup_ratio:
            return self.final_lambda
        if progress <= self.keep_first_ratio:
            return self.start_lambda
        fraction = (progress - self.keep_first_ratio) / (
            self.warmup_ratio - self.keep_first_ratio
        )
        return self.start_lambda + (self.final_lambda - self.start_lambda) * fraction

    def _apply(self, state: State) -> float:
        # Unwrap DDP/FSDP, then Composer's HuggingFaceModel, as in the checkpoint
        # callbacks. E2D reads this config field on every loss computation.
        composer_model = (
            state.model.module if hasattr(state.model, "module") else state.model
        )
        model = composer_model.model
        if not hasattr(model.config, "decoder_loss_lambda"):
            raise ValueError(
                "Lambda warmup requires an E2D model with decoder_loss_lambda."
            )
        if getattr(model.backbone, "frozen_target", False):
            raise ValueError(
                "Frozen-target models do not use the joint E2D lambda loss."
            )
        value = self._current_lambda(state)
        model.config.decoder_loss_lambda = value
        return value

    def fit_start(self, state: State, logger: Logger) -> None:
        # Runs after checkpoint loading; no separate schedule counter to restore.
        self._apply(state)

    def batch_start(self, state: State, logger: Logger) -> None:
        value = self._apply(state)
        logger.log_metrics({"loss/train/decoder_loss_lambda": value})

    def batch_end(self, state: State, logger: Logger) -> None:
        # Use the completed training progress for subsequent evaluation and saves.
        self._apply(state)
