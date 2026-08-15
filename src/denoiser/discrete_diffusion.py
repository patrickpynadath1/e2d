"""Dual-decoder discrete-diffusion speculative baselines.

These models reuse E2D2's clean/noisy block attention layout, but expose explicit
masked and uniform corruption objectives and verify every drafted block with the
full causal target model.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.denoiser.base import DenoiserInput, LossAndNllOutput
from src.denoiser.diffusion import E2D2, DiffusionGenerationConfig, E2D2Config
from src.denoiser.speculative import (
    DecoderMode,
    SpeculativeStats,
    longest_matching_prefix,
)


class DiscreteDiffusionE2DConfig(E2D2Config):
    """Configuration shared by masked and uniform dual-decoder baselines."""

    model_type = "discrete_diffusion_e2d"

    def __init__(
        self,
        corruption_mode: str = DecoderMode.MASKED,
        decoder_loss_lambda: float = 1.0,
        inference_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.corruption_mode = DecoderMode(corruption_mode)
        self.decoder_loss_lambda = decoder_loss_lambda
        self.inference_steps = inference_steps


class DiscreteDiffusionE2D(E2D2):
    """Block-bidirectional drafter with full-target speculative verification."""

    config_class = DiscreteDiffusionE2DConfig

    def __init__(self, config: DiscreteDiffusionE2DConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.corruption_mode = DecoderMode(config.corruption_mode)
        if self.corruption_mode not in {DecoderMode.MASKED, DecoderMode.UNIFORM}:
            raise ValueError(f"unsupported corruption mode: {self.corruption_mode}")
        if config.inference_steps < 1:
            raise ValueError("inference_steps must be at least one")
        self.last_speculative_stats = SpeculativeStats()

    def _uniform_replacements(self, x0: torch.LongTensor) -> torch.LongTensor:
        """Sample valid vocabulary IDs, excluding the clean and mask token IDs."""
        replacements = torch.randint(self.vocab_size, x0.shape, device=x0.device)
        invalid = replacements.eq(x0) | replacements.eq(self.mask_token_id)
        while invalid.any():
            replacements[invalid] = torch.randint(
                self.vocab_size,
                (int(invalid.sum().item()),),
                device=x0.device,
            )
            invalid = replacements.eq(x0) | replacements.eq(self.mask_token_id)
        return replacements

    def _sample_q_xt(
        self,
        x0: torch.LongTensor,
        alpha_t: torch.FloatTensor,
        context_mask: torch.FloatTensor,
    ) -> torch.LongTensor:
        if self.corruption_mode == DecoderMode.MASKED:
            return super()._sample_q_xt(x0, alpha_t, context_mask)
        corrupt = (torch.rand(x0.shape, device=x0.device) < (1.0 - alpha_t)) & (
            ~context_mask.bool()
        )
        return torch.where(corrupt, self._uniform_replacements(x0), x0)

    def _ensure_no_unmasked_blocks(
        self,
        x0: torch.LongTensor,
        xt: torch.LongTensor,
        context_mask: torch.FloatTensor,
    ) -> torch.LongTensor:
        if self.corruption_mode == DecoderMode.MASKED:
            return super()._ensure_no_unmasked_blocks(x0, xt, context_mask)
        block_size = self.config.block_size
        if not isinstance(block_size, int):
            return xt
        valid = ~context_mask.bool()
        changed = xt.ne(x0) & valid
        for start in range(0, x0.shape[1], block_size):
            end = min(start + block_size, x0.shape[1])
            needs_corruption = valid[:, start:end].any(-1) & ~changed[:, start:end].any(
                -1
            )
            for batch_idx in needs_corruption.nonzero(as_tuple=False).flatten():
                candidates = (
                    valid[batch_idx, start:end].nonzero(as_tuple=False).flatten()
                )
                if candidates.numel():
                    pos = start + int(candidates[0])
                    xt[batch_idx, pos] = self._uniform_replacements(
                        x0[batch_idx, pos].reshape(1)
                    )[0]
        return xt

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        """AR verifier loss plus same-position corrupted-token decoder loss."""
        del kwargs
        clean = denoiser_inputs.x0
        if clean is None:
            raise ValueError("clean tokens are required for diffusion training")
        seq_len = clean.shape[1]
        decoder_logits = model_output[:, -seq_len:, :]
        valid = (
            denoiser_inputs.tokens_mask.bool()
            if denoiser_inputs.tokens_mask is not None
            else torch.ones_like(clean, dtype=torch.bool)
        )
        corrupted = denoiser_inputs.xt.ne(clean) & valid
        if not corrupted.any():
            raise RuntimeError("discrete diffusion batch contains no corrupted tokens")

        decoder_token_nll = F.cross_entropy(
            decoder_logits.transpose(1, 2), clean, reduction="none"
        )
        decoder_loss = decoder_token_nll[corrupted].mean()
        loss = decoder_loss
        terms: dict[str, torch.Tensor] = {
            "decoder_loss": decoder_loss,
            "corrupted_fraction": corrupted.float().sum()
            / valid.float().sum().clamp_min(1),
        }

        # EncoderGen backbones return clean encoder logits followed by decoder logits.
        if model_output.shape[1] == 2 * seq_len:
            encoder_logits = model_output[:, :seq_len, :]
            encoder_valid = valid[:, 1:]
            encoder_nll = F.cross_entropy(
                encoder_logits[:, :-1].transpose(1, 2), clean[:, 1:], reduction="none"
            )
            encoder_loss = encoder_nll[encoder_valid].mean()
            weight = float(self.config.decoder_loss_lambda)
            if weight < 0:
                raise ValueError("decoder_loss_lambda must be non-negative")
            loss = (encoder_loss + weight * decoder_loss) / (1.0 + weight)
            terms["encoder_loss"] = encoder_loss

        return LossAndNllOutput(
            loss=loss,
            nlls=decoder_token_nll * valid,
            other_loss_terms=terms,
        )

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        """Convert logits to log probabilities without mixing clean/noisy halves."""
        del kwargs
        seq_len = denoiser_inputs.xt.shape[1]
        encoder_logits = (
            backbone_output[:, :-seq_len, :]
            if backbone_output.shape[1] > seq_len
            else None
        )
        decoder_logits = backbone_output[:, -seq_len:, :]
        if self.corruption_mode == DecoderMode.MASKED:
            masked_logits = decoder_logits.clone()
            masked_logits[..., self.mask_token_id] = -1e12
            predicted_log_probs = F.log_softmax(masked_logits, dim=-1)
            corrupted = denoiser_inputs.xt.eq(self.mask_token_id)
            copied_log_probs = torch.full_like(predicted_log_probs, -1e12)
            copied_log_probs = copied_log_probs.scatter(
                -1, denoiser_inputs.xt.unsqueeze(-1), 0.0
            )
            decoder_log_probs = torch.where(
                corrupted.unsqueeze(-1), predicted_log_probs, copied_log_probs
            )
        else:
            decoder_log_probs = F.log_softmax(decoder_logits, dim=-1)
        if encoder_logits is None:
            return decoder_log_probs
        return torch.cat(
            [F.log_softmax(encoder_logits, dim=-1), decoder_log_probs], dim=1
        )

    def _initial_canvas(
        self, batch_size: int, block_size: int, device: torch.device
    ) -> torch.LongTensor:
        if self.corruption_mode == DecoderMode.MASKED:
            return torch.full(
                (batch_size, block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
        canvas = torch.randint(
            self.vocab_size, (batch_size, block_size), dtype=torch.long, device=device
        )
        canvas[canvas == self.mask_token_id] = 0
        return canvas

    def _draft_block(
        self,
        context: torch.LongTensor,
        block_size: int,
        num_steps: int,
    ) -> torch.LongTensor:
        canvas = self._initial_canvas(context.shape[0], block_size, context.device)
        unresolved = torch.ones_like(canvas, dtype=torch.bool)
        for step in range(num_steps):
            denoiser_inputs, cache = self._prepare_inputs_inference(
                input_ids=canvas, context=context, cache=None
            )
            output = self._backbone_forward(denoiser_inputs, **(cache or {}))
            logits = output.logits[:, -block_size:, : self.vocab_size]
            logits[..., self.mask_token_id] = torch.finfo(logits.dtype).min
            probs = logits.softmax(-1)
            confidence, prediction = probs.max(-1)
            if self.corruption_mode == DecoderMode.UNIFORM:
                canvas = prediction
                continue
            remaining_steps = num_steps - step
            reveal_count = math.ceil(unresolved.sum(-1).max().item() / remaining_steps)
            masked_confidence = confidence.masked_fill(~unresolved, -1)
            reveal = masked_confidence.topk(
                min(reveal_count, block_size), dim=-1
            ).indices
            reveal_mask = (
                torch.zeros_like(unresolved).scatter_(1, reveal, True) & unresolved
            )
            canvas = torch.where(reveal_mask, prediction, canvas)
            unresolved &= ~reveal_mask
        if unresolved.any():
            canvas = torch.where(unresolved, prediction, canvas)
        return canvas

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[DiffusionGenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        return_speculative_stats: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor | tuple[torch.LongTensor, SpeculativeStats]:
        """Draft with diffusion and commit only target-greedy verified tokens.

        This initial correctness implementation intentionally verifies without a KV
        cache. Cache reuse can be optimized after output-equivalence tests pass.
        """
        del kwargs
        if inputs is None:
            if batch_size is None:
                batch_size = 1
            target_device = torch.device(device or next(self.parameters()).device)
            inputs = torch.full(
                (batch_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=target_device,
            )
        if inputs.shape[0] != 1:
            raise NotImplementedError(
                "speculative diffusion generation currently supports batch size one"
            )
        generation_config = generation_config or getattr(
            self, "generation_config", None
        )
        if generation_config is None:
            raise ValueError("generation_config is required")
        block_size = int(generation_config.block_size or self.config.eval_block_size)
        num_steps = int(
            getattr(generation_config, "num_steps", self.config.inference_steps)
        )
        max_new_tokens = int(max_new_tokens or generation_config.max_new_tokens)
        generated = inputs
        stats = SpeculativeStats()
        started = time.perf_counter()

        while generated.shape[1] - inputs.shape[1] < max_new_tokens:
            remaining = max_new_tokens - (generated.shape[1] - inputs.shape[1])
            proposal_len = min(block_size, remaining)
            draft_started = time.perf_counter()
            proposal = self._draft_block(generated, proposal_len, num_steps)
            stats.draft_seconds += time.perf_counter() - draft_started
            stats.draft_calls += num_steps
            stats.proposed_tokens += proposal_len

            verify_started = time.perf_counter()
            candidate = torch.cat([generated, proposal], dim=-1)
            verifier_logits = self.backbone.encoder(
                input_ids=candidate, use_cache=False
            ).logits
            start = generated.shape[1] - 1
            target = verifier_logits[:, start : start + proposal_len].argmax(-1)
            accepted = int(longest_matching_prefix(proposal, target)[0].item())
            stats.verifier_seconds += time.perf_counter() - verify_started
            stats.verifier_calls += 1
            stats.accepted_tokens += accepted
            stats.accepted_lengths.append(accepted)

            if accepted:
                generated = torch.cat([generated, proposal[:, :accepted]], dim=-1)
            if (
                accepted < proposal_len
                and generated.shape[1] - inputs.shape[1] < max_new_tokens
            ):
                correction = target[:, accepted : accepted + 1]
                generated = torch.cat([generated, correction], dim=-1)
                stats.correction_tokens += 1
            stats.committed_tokens = generated.shape[1] - inputs.shape[1]
            if (
                self.eos_token_id is not None
                and (generated[:, inputs.shape[1] :] == self.eos_token_id).any()
            ):
                eos_pos = (
                    generated[0, inputs.shape[1] :] == self.eos_token_id
                ).nonzero()[0, 0]
                generated = generated[:, : inputs.shape[1] + int(eos_pos) + 1]
                break

        stats.committed_tokens = generated.shape[1] - inputs.shape[1]
        stats.total_seconds = time.perf_counter() - started
        self.last_speculative_stats = stats
        return (generated, stats) if return_speculative_stats else generated


class MaskedDiffusionE2DConfig(DiscreteDiffusionE2DConfig):
    model_type = "masked_diffusion_e2d"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(corruption_mode=DecoderMode.MASKED, **kwargs)


class MaskedDiffusionE2D(DiscreteDiffusionE2D):
    config_class = MaskedDiffusionE2DConfig


class UniformDiffusionE2DConfig(DiscreteDiffusionE2DConfig):
    model_type = "uniform_diffusion_e2d"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(corruption_mode=DecoderMode.UNIFORM, **kwargs)


class UniformDiffusionE2D(DiscreteDiffusionE2D):
    config_class = UniformDiffusionE2DConfig
