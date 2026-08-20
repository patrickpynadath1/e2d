"""Latent flow-matching drafter with full-target speculative verification."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
from torch import nn

from src.denoiser.base import Denoiser, DenoiserConfig, DenoiserOutput
from src.denoiser.diffusion import DiffusionGenerationConfig
from src.denoiser.speculative import (
    SpeculativeStats,
    linear_flow_sample,
    longest_matching_prefix,
)


def project_to_tangent(point: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Project an ambient vector onto the tangent space at a unit sphere point."""
    return vector - (point * vector).sum(dim=-1, keepdim=True) * point


def spherical_interpolant_and_velocity(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SLERP from clean at t=0 to noise at t=1 and its tangent velocity."""
    clean_float = F.normalize(clean.float(), dim=-1)
    noise_float = F.normalize(noise.float(), dim=-1)
    time_float = time_value.float()
    while time_float.ndim < clean_float.ndim:
        time_float = time_float.unsqueeze(-1)
    cosine = (clean_float * noise_float).sum(dim=-1, keepdim=True)
    cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(cosine)
    sine = torch.sin(angle).clamp_min(eps)
    clean_weight = torch.sin((1.0 - time_float) * angle) / sine
    noise_weight = torch.sin(time_float * angle) / sine
    point = clean_weight * clean_float + noise_weight * noise_float
    velocity = (
        angle
        / sine
        * (
            -torch.cos((1.0 - time_float) * angle) * clean_float
            + torch.cos(time_float * angle) * noise_float
        )
    )
    point = F.normalize(point, dim=-1)
    velocity = project_to_tangent(point, velocity)
    return point.to(clean.dtype), velocity


def sphere_expmap(
    point: torch.Tensor,
    tangent_velocity: torch.Tensor,
    step_size: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Move along a spherical geodesic using the exponential map."""
    point_float = F.normalize(point.float(), dim=-1)
    tangent = project_to_tangent(point_float, tangent_velocity.float())
    tangent_step = step_size * tangent
    step_norm = tangent_step.norm(dim=-1, keepdim=True)
    direction = tangent_step / step_norm.clamp_min(eps)
    updated = torch.cos(step_norm) * point_float + torch.sin(step_norm) * direction
    small = step_norm < eps
    updated = torch.where(small, point_float, updated)
    return F.normalize(updated, dim=-1).to(point.dtype)


class LatentFlowE2DConfig(DenoiserConfig):
    model_type = "latent_flow_e2d"

    def __init__(
        self,
        block_size: int = 4,
        eval_block_size: Optional[int] = None,
        draft_layer_offsets: tuple[int, int] = (-3, -2),
        target_layer_offset: int = -1,
        inference_steps: int = 1,
        stats_epsilon: float = 1e-6,
        latent_stats_path: Optional[str] = None,
        prediction_type: Literal[
            "velocity", "x0", "x0_prev_block_residual"
        ] = "velocity",
        self_conditioning: bool = False,
        self_conditioning_probability: float = 0.5,
        adaptive_timestep_sampling: bool = False,
        adaptive_num_bins: int = 50,
        adaptive_ema_decay: float = 0.99,
        adaptive_uniform_mix: float = 0.2,
        adaptive_min_observations: int = 100,
        fixed_training_noise: bool = False,
        fixed_training_seed: int = 17,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.block_size = block_size
        self.eval_block_size = eval_block_size or block_size
        self.draft_layer_offsets = tuple(draft_layer_offsets)
        self.target_layer_offset = target_layer_offset
        self.inference_steps = inference_steps
        self.stats_epsilon = stats_epsilon
        valid_prediction_types = (
            "velocity",
            "x0",
            "x0_prev_block_residual",
        )
        if prediction_type not in valid_prediction_types:
            raise ValueError(
                "prediction_type must be 'velocity', 'x0', or 'x0_prev_block_residual'"
            )
        if adaptive_num_bins < 2:
            raise ValueError("adaptive_num_bins must be at least two")
        if not 0 <= self_conditioning_probability <= 1:
            raise ValueError("self_conditioning_probability must be in [0, 1]")
        if not 0 <= adaptive_ema_decay < 1:
            raise ValueError("adaptive_ema_decay must be in [0, 1)")
        if not 0 <= adaptive_uniform_mix <= 1:
            raise ValueError("adaptive_uniform_mix must be in [0, 1]")
        if adaptive_min_observations < 0:
            raise ValueError("adaptive_min_observations must be non-negative")
        self.latent_stats_path = latent_stats_path
        self.prediction_type = prediction_type
        self.self_conditioning = self_conditioning
        self.self_conditioning_probability = self_conditioning_probability
        self.adaptive_timestep_sampling = adaptive_timestep_sampling
        self.adaptive_num_bins = adaptive_num_bins
        self.adaptive_ema_decay = adaptive_ema_decay
        self.adaptive_uniform_mix = adaptive_uniform_mix
        self.adaptive_min_observations = adaptive_min_observations
        self.fixed_training_noise = fixed_training_noise
        self.fixed_training_seed = fixed_training_seed


class LatentFlowE2D(Denoiser):
    """Flow matching in the residual stream entering Qwen's final layer.

    The full verifier is frozen. The trainable velocity network is initialized from
    the two transformer layers immediately preceding the final target layer.
    """

    config_class = LatentFlowE2DConfig

    def __init__(self, config: LatentFlowE2DConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.target = self.backbone.encoder
        target_layers = self.target.model.layers
        layer_count = len(target_layers)
        self.target_layer_idx = self._resolve_layer_index(
            config.target_layer_offset, layer_count
        )
        self.draft_layer_idxs = tuple(
            self._resolve_layer_index(offset, layer_count)
            for offset in config.draft_layer_offsets
        )
        if self.target_layer_idx != layer_count - 1:
            raise ValueError(
                "the initial latent-flow baseline requires the final target layer"
            )
        if any(index >= self.target_layer_idx for index in self.draft_layer_idxs):
            raise ValueError("draft layers must precede the target decoder layer")

        self.flow_layers = nn.ModuleList(
            copy.deepcopy(target_layers[index]) for index in self.draft_layer_idxs
        )
        hidden_size = self.target.config.hidden_size
        self.time_mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.self_conditioning_head = (
            nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
            )
            if config.self_conditioning
            else None
        )
        self.velocity_head = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.velocity_head.weight)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.flow_layers.parameters():
            parameter.requires_grad = True
        latent_mean = torch.zeros(hidden_size, dtype=torch.float32)
        latent_std = torch.ones(hidden_size, dtype=torch.float32)
        stats_are_configured = config.latent_stats_path is not None
        if stats_are_configured:
            stats_path = Path(config.latent_stats_path).expanduser()
            if not stats_path.is_file():
                raise FileNotFoundError(
                    f"latent statistics file does not exist: {stats_path}. Run "
                    "scripts/eval/compare_qwen_latent_gaussian_norms.py first."
                )
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            latent_mean = stats["feature_mean"].float()
            latent_std = stats["feature_std"].float()
            expected_shape = (hidden_size,)
            if (
                latent_mean.shape != expected_shape
                or latent_std.shape != expected_shape
            ):
                raise ValueError(
                    "latent feature statistics must have shape "
                    f"{expected_shape}, got {tuple(latent_mean.shape)} and "
                    f"{tuple(latent_std.shape)}"
                )
            latent_std = latent_std.clamp_min(config.stats_epsilon)
        self.register_buffer(
            "latent_mean",
            latent_mean,
        )
        self.register_buffer(
            "latent_std",
            latent_std,
        )
        self.register_buffer(
            "latent_stats_initialized", torch.tensor(stats_are_configured)
        )
        self.register_buffer(
            "adaptive_kl_ema",
            torch.zeros(config.adaptive_num_bins, dtype=torch.float64),
        )
        self.register_buffer(
            "adaptive_kl_counts",
            torch.zeros(config.adaptive_num_bins, dtype=torch.long),
        )
        self.last_speculative_stats = SpeculativeStats()

    @staticmethod
    def _resolve_layer_index(offset: int, layer_count: int) -> int:
        index = offset if offset >= 0 else layer_count + offset
        if not 0 <= index < layer_count:
            raise ValueError(
                f"layer offset {offset} is invalid for {layer_count} layers"
            )
        return index

    @staticmethod
    def _flow_attention_mask(
        seq_len: int, block_size: int, device: torch.device
    ) -> torch.BoolTensor:
        """Return the [2L, 2L] clean/noisy block mask used during training."""
        q_idx = torch.arange(2 * seq_len, device=device)[:, None]
        kv_idx = torch.arange(2 * seq_len, device=device)[None, :]
        q_noisy = q_idx >= seq_len
        kv_noisy = kv_idx >= seq_len
        q_pos = torch.where(q_noisy, q_idx - seq_len, q_idx)
        kv_pos = torch.where(kv_noisy, kv_idx - seq_len, kv_idx)
        q_block = q_pos // block_size
        kv_block = kv_pos // block_size
        clean_rows = ~q_noisy & ~kv_noisy & (q_block >= kv_block)
        noisy_to_clean_history = q_noisy & ~kv_noisy & (q_block > kv_block)
        noisy_current_canvas = q_noisy & kv_noisy & (q_block == kv_block)
        return clean_rows | noisy_to_clean_history | noisy_current_canvas

    @staticmethod
    def _inference_attention_mask(
        context_len: int, block_len: int, device: torch.device
    ) -> torch.BoolTensor:
        total = context_len + block_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        if context_len:
            mask[:context_len, :context_len] = torch.tril(
                torch.ones((context_len, context_len), dtype=torch.bool, device=device)
            )
            mask[context_len:, :context_len] = True
        mask[context_len:, context_len:] = True
        return mask

    @staticmethod
    def _as_additive_mask(mask: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
        additive = torch.where(mask, 0.0, torch.finfo(dtype).min).to(dtype)
        if additive.ndim == 2:
            return additive[None, None]
        if additive.ndim == 3:
            return additive[:, None]
        raise ValueError("attention mask must have shape [L, L] or [B, L, L]")

    @torch.no_grad()
    def extract_clean_latents(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.target.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states[i] is the residual stream entering transformer layer i.
        return output.hidden_states[self.target_layer_idx].detach()

    @torch.no_grad()
    def initialize_latent_stats(
        self, clean_latents: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        del clean_latents, valid_mask
        raise RuntimeError(
            "first-batch latent normalization has been removed; configure a "
            "dataset-calibrated latent_stats_path"
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return ((latents - self.latent_mean) / self.latent_std).to(latents.dtype)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents * self.latent_std + self.latent_mean).to(latents.dtype)

    def _sample_training_path(
        self, clean: torch.Tensor, supplied_time: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = None
        if self.config.fixed_training_noise:
            generator = torch.Generator(device=clean.device)
            generator.manual_seed(self.config.fixed_training_seed)
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        if (
            getattr(self.config, "adaptive_timestep_sampling", False)
            and not self.config.fixed_training_noise
        ):
            block_count = math.ceil(clean.shape[1] / self.config.block_size)
            probabilities = self.adaptive_timestep_probabilities().to(clean.device)
            bin_indices = torch.multinomial(
                probabilities,
                clean.shape[0] * block_count,
                replacement=True,
                generator=generator,
            ).view(clean.shape[0], block_count)
            within_bin = torch.rand(
                clean.shape[0], block_count, device=clean.device, generator=generator
            )
            sampled_time = (bin_indices + within_bin) / self.config.adaptive_num_bins
            sampled_time = sampled_time.repeat_interleave(
                self.config.block_size, dim=-1
            )[:, : clean.shape[1]]
            return noise, sampled_time
        if supplied_time is not None and not self.config.fixed_training_noise:
            return noise, supplied_time
        block_count = math.ceil(clean.shape[1] / self.config.block_size)
        sampled_time = torch.rand(
            clean.shape[0],
            block_count,
            device=clean.device,
            generator=generator,
        )
        sampled_time = sampled_time.repeat_interleave(self.config.block_size, dim=-1)[
            :, : clean.shape[1]
        ]
        return noise, sampled_time

    def adaptive_timestep_probabilities(self) -> torch.Tensor:
        """Return a smoothed KL-slope distribution with uniform exploration."""
        num_bins = self.config.adaptive_num_bins
        uniform = torch.full_like(self.adaptive_kl_ema, 1.0 / num_bins)
        sufficiently_observed = bool(
            (self.adaptive_kl_counts >= self.config.adaptive_min_observations).all()
        )
        if not sufficiently_observed:
            return uniform
        monotone_kl = torch.cummax(self.adaptive_kl_ema, dim=0).values
        # Only changes in difficulty matter; an irreducible KL offset at low
        # noise should not attract extra samples.
        slopes = torch.diff(monotone_kl, prepend=monotone_kl[:1]).clamp_min(0.0)
        if not bool(torch.isfinite(slopes).all()) or float(slopes.sum()) <= 0:
            return uniform
        adaptive = slopes / slopes.sum()
        mix = self.config.adaptive_uniform_mix
        return (1.0 - mix) * adaptive + mix * uniform

    def _run_flow_hidden(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        time_value: torch.Tensor,
        attention_mask: torch.BoolTensor,
        clean_position_ids: torch.LongTensor,
        noisy_position_ids: torch.LongTensor,
        self_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, noisy_len, _ = noisy.shape
        if time_value.ndim == 1:
            time_value = time_value[:, None].expand(batch_size, noisy_len)
        if self.self_conditioning_head is not None:
            if self_conditioning is None:
                self_conditioning = torch.zeros_like(noisy)
            if self_conditioning.shape != noisy.shape:
                raise ValueError(
                    "self-conditioning estimate must have the same shape as noisy "
                    f"latents, got {self_conditioning.shape} and {noisy.shape}"
                )
            noisy = self.self_conditioning_head(
                torch.cat([noisy, self_conditioning.detach()], dim=-1)
            )
        time_features = torch.stack(
            [
                time_value,
                torch.sin(math.pi * time_value),
                torch.cos(math.pi * time_value),
            ],
            dim=-1,
        ).to(noisy.dtype)
        noisy = noisy + self.time_mlp(time_features)
        hidden = torch.cat([clean, noisy], dim=1)
        position_ids = torch.cat([clean_position_ids, noisy_position_ids], dim=1)
        position_embeddings = self.target.model.rotary_emb(hidden, position_ids)
        additive_mask = self._as_additive_mask(attention_mask, hidden.dtype)
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        for layer in self.flow_layers:
            hidden = layer(
                hidden,
                attention_mask=additive_mask,
                position_ids=position_ids,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        return hidden[:, -noisy_len:]

    def _run_flow_layers(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        time_value: torch.Tensor,
        attention_mask: torch.BoolTensor,
        clean_position_ids: torch.LongTensor,
        noisy_position_ids: torch.LongTensor,
        self_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self._run_flow_hidden(
            clean,
            noisy,
            time_value,
            attention_mask,
            clean_position_ids,
            noisy_position_ids,
            self_conditioning,
        )
        return self.velocity_head(hidden)

    @staticmethod
    def _expand_time(time_value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        while time_value.ndim < reference.ndim:
            time_value = time_value.unsqueeze(-1)
        return time_value.to(reference.dtype)

    def prediction_to_x0(
        self,
        noisy: torch.Tensor,
        prediction: torch.Tensor,
        time_value: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert either configured regression target to a clean-latent estimate."""
        expanded_time = self._expand_time(time_value, noisy)
        if self.config.prediction_type == "velocity":
            return noisy - expanded_time * prediction
        if self.config.prediction_type == "x0_prev_block_residual":
            if baseline is None:
                raise ValueError(
                    "x0_prev_block_residual prediction requires a previous-block "
                    "baseline"
                )
            return baseline + prediction
        # The x0 head predicts a residual so its zero initialization is an identity
        # denoiser, while the supervised quantity remains the clean endpoint.
        return noisy + prediction

    def _prediction_target(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        target_velocity: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.prediction_type == "velocity":
            return target_velocity
        if self.config.prediction_type == "x0_prev_block_residual":
            if baseline is None:
                raise ValueError(
                    "x0_prev_block_residual target requires a previous-block baseline"
                )
            return clean - baseline
        return clean - noisy

    @staticmethod
    def _training_previous_block_baseline(
        clean: torch.Tensor,
        valid_mask: torch.BoolTensor,
        context_mask: torch.BoolTensor,
        block_size: int,
    ) -> torch.Tensor:
        """Mean the prompt tail or preceding completion block for each target block."""
        baseline = torch.zeros_like(clean)
        for batch_index in range(clean.shape[0]):
            valid = valid_mask[batch_index]
            context_indices = torch.nonzero(
                valid & context_mask[batch_index], as_tuple=False
            ).flatten()
            target_indices = torch.nonzero(
                valid & ~context_mask[batch_index], as_tuple=False
            ).flatten()
            previous_indices = context_indices[-block_size:]
            for start in range(0, target_indices.numel(), block_size):
                current_indices = target_indices[start : start + block_size]
                if previous_indices.numel():
                    previous_mean = clean[batch_index, previous_indices].mean(
                        dim=0, keepdim=True
                    )
                    baseline[batch_index, current_indices] = previous_mean
                previous_indices = current_indices
        return baseline

    @staticmethod
    def _generation_previous_block_baseline(
        context_latents: torch.Tensor,
        block_len: int,
        previous_block_size: int,
    ) -> torch.Tensor:
        if context_latents.shape[1] == 0:
            return context_latents.new_zeros(
                context_latents.shape[0], block_len, context_latents.shape[-1]
            )
        previous_mean = context_latents[:, -previous_block_size:].mean(
            dim=1, keepdim=True
        )
        return previous_mean.expand(-1, block_len, -1)

    def _decode_logits_from_residual(self, residual: torch.Tensor) -> torch.Tensor:
        seq_len = residual.shape[1]
        positions = torch.arange(seq_len, device=residual.device)[None].expand(
            residual.shape[0], -1
        )
        position_embeddings = self.target.model.rotary_emb(residual, positions)
        causal = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=residual.device)
        )
        decoded = self.target.model.layers[self.target_layer_idx](
            residual,
            attention_mask=self._as_additive_mask(causal, residual.dtype),
            position_ids=positions,
            use_cache=False,
            cache_position=positions[0],
            position_embeddings=position_embeddings,
        )[0]
        return self.target.lm_head(self.target.model.norm(decoded))

    @torch.no_grad()
    def _decoder_kl_per_position(
        self,
        clean: torch.Tensor,
        predicted_x0: torch.Tensor,
        predicted_mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL from clean-latent logits to predicted-latent logits in float32."""
        predicted_sequence = torch.where(
            predicted_mask.unsqueeze(-1), predicted_x0.detach(), clean
        )
        clean_raw = self.denormalize_latents(clean)
        predicted_raw = self.denormalize_latents(predicted_sequence)
        teacher_logits = self._decode_logits_from_residual(clean_raw).float()
        predicted_logits = self._decode_logits_from_residual(predicted_raw).float()
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        predicted_log_probs = F.log_softmax(predicted_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        return (teacher_probs * (teacher_log_probs - predicted_log_probs)).sum(-1)

    @torch.no_grad()
    def _update_adaptive_kl(
        self,
        kl_per_position: torch.Tensor,
        time_value: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if not self.config.adaptive_timestep_sampling:
            return
        # Logit position i predicts token i+1. The timestep belongs to the latent
        # at position i, while validity is determined by that next target token.
        shifted_valid = torch.zeros_like(valid_mask)
        shifted_valid[:, :-1] = valid_mask[:, 1:]
        valid = valid_mask & shifted_valid & torch.isfinite(kl_per_position)
        if not bool(valid.any()):
            return
        bins = (
            torch.floor(
                time_value[valid].float().clamp(0.0, 1.0)
                * self.config.adaptive_num_bins
            )
            .long()
            .clamp_max(self.config.adaptive_num_bins - 1)
        )
        sums = torch.zeros_like(self.adaptive_kl_ema)
        counts = torch.zeros_like(self.adaptive_kl_counts)
        sums.scatter_add_(0, bins, kl_per_position[valid].double())
        counts.scatter_add_(0, bins, torch.ones_like(bins))
        if torch_dist.is_available() and torch_dist.is_initialized():
            torch_dist.all_reduce(sums, op=torch_dist.ReduceOp.SUM)
            torch_dist.all_reduce(counts, op=torch_dist.ReduceOp.SUM)
        observed = counts > 0
        batch_means = sums[observed] / counts[observed].double()
        decay = self.config.adaptive_ema_decay
        previously_seen = self.adaptive_kl_counts[observed] > 0
        old_values = self.adaptive_kl_ema[observed]
        updated = torch.where(
            previously_seen,
            decay * old_values + (1.0 - decay) * batch_means,
            batch_means,
        )
        self.adaptive_kl_ema[observed] = updated
        self.adaptive_kl_counts.add_(counts)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        **kwargs: Any,
    ) -> DenoiserOutput:
        del kwargs, compute_loss
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(input_ids)
        valid = attention_mask.bool()
        loss_mask = valid & ~context_mask.bool()
        clean_raw = self.extract_clean_latents(input_ids, attention_mask)
        if not bool(self.latent_stats_initialized):
            raise RuntimeError(
                "latent normalization statistics are not configured; set "
                "latent_stats_path to statistics from a representative dataset"
            )
        clean = self.normalize_latents(clean_raw)
        noise, t = self._sample_training_path(clean, t)
        noisy, target_velocity = linear_flow_sample(clean, noise, t)
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)[None].expand(
            input_ids.shape[0], -1
        )
        flow_mask = self._flow_attention_mask(
            seq_len, self.config.block_size, input_ids.device
        )
        padding = torch.cat([valid, valid], dim=-1)
        flow_mask = flow_mask[None] & padding[:, :, None] & padding[:, None, :]
        baseline = None
        if self.config.prediction_type == "x0_prev_block_residual":
            baseline = self._training_previous_block_baseline(
                clean,
                valid,
                context_mask.bool(),
                self.config.block_size,
            )
        use_self_conditioning = self.self_conditioning_head is not None and (
            not self.training
            or bool(
                torch.rand((), device=input_ids.device)
                < self.config.self_conditioning_probability
            )
        )
        if use_self_conditioning:
            with torch.no_grad():
                preliminary_prediction = self._run_flow_layers(
                    clean, noisy, t, flow_mask, positions, positions
                )
                self_conditioning = self.prediction_to_x0(
                    noisy, preliminary_prediction, t, baseline
                ).detach()
            prediction = self._run_flow_layers(
                clean,
                noisy,
                t,
                flow_mask,
                positions,
                positions,
                self_conditioning,
            )
        else:
            prediction = self._run_flow_layers(
                clean, noisy, t, flow_mask, positions, positions
            )
        regression_target = self._prediction_target(
            clean, noisy, target_velocity, baseline
        )
        token_mse = F.mse_loss(
            prediction.float(), regression_target.float(), reduction="none"
        ).mean(-1)
        flow_loss = token_mse[loss_mask].mean()
        predicted_x0 = self.prediction_to_x0(noisy, prediction, t, baseline)
        decoder_kl = None
        if self.config.adaptive_timestep_sampling:
            decoder_kl = self._decoder_kl_per_position(clean, predicted_x0, loss_mask)
            self._update_adaptive_kl(decoder_kl, t, loss_mask)
        return DenoiserOutput(
            denoiser_output=prediction,
            tokens_mask=loss_mask.float(),
            loss=flow_loss,
            nlls=token_mse,
            other_loss_terms={
                "flow_loss": flow_loss,
                **(
                    {"decoder_kl": decoder_kl[loss_mask].mean()}
                    if decoder_kl is not None
                    else {}
                ),
            },
            flow_loss=flow_loss,
            flow_timesteps=t,
        )

    def _prepare_inputs(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("LatentFlowE2D implements forward directly")

    def _compute_loss(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("LatentFlowE2D implements forward directly")

    def _draft_latents(
        self,
        context_latents: torch.Tensor,
        block_len: int,
        num_steps: int,
        previous_block_size: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = context_latents.shape[0]
        canvas = torch.randn(
            batch_size,
            block_len,
            context_latents.shape[-1],
            device=context_latents.device,
            dtype=context_latents.dtype,
        )
        context_len = context_latents.shape[1]
        clean_positions = torch.arange(context_len, device=canvas.device)[None].expand(
            batch_size, -1
        )
        noisy_positions = torch.arange(
            context_len, context_len + block_len, device=canvas.device
        )[None].expand(batch_size, -1)
        mask = self._inference_attention_mask(context_len, block_len, canvas.device)
        baseline = None
        if self.config.prediction_type == "x0_prev_block_residual":
            baseline = self._generation_previous_block_baseline(
                context_latents,
                block_len,
                previous_block_size or block_len,
            )
        self_conditioning = None
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_value = torch.full(
                (batch_size, block_len),
                1.0 - step * dt,
                device=canvas.device,
                dtype=canvas.dtype,
            )
            if self.self_conditioning_head is not None and self_conditioning is None:
                preliminary_prediction = self._run_flow_layers(
                    context_latents,
                    canvas,
                    t_value,
                    mask,
                    clean_positions,
                    noisy_positions,
                )
                self_conditioning = self.prediction_to_x0(
                    canvas, preliminary_prediction, t_value, baseline
                ).detach()
            velocity = self._run_flow_layers(
                context_latents,
                canvas,
                t_value,
                mask,
                clean_positions,
                noisy_positions,
                self_conditioning,
            )
            if self.config.prediction_type in ("x0", "x0_prev_block_residual"):
                predicted_x0 = self.prediction_to_x0(
                    canvas, velocity, t_value, baseline
                )
                velocity = (canvas - predicted_x0) / t_value.unsqueeze(-1).clamp_min(
                    self.config.stats_epsilon
                )
                self_conditioning = predicted_x0.detach()
            elif self.self_conditioning_head is not None:
                self_conditioning = self.prediction_to_x0(
                    canvas, velocity, t_value
                ).detach()
            canvas = canvas - dt * velocity
        return canvas

    def decode_latents(
        self, context_latents: torch.Tensor, block_latents: torch.Tensor
    ) -> torch.LongTensor:
        residual = self.denormalize_latents(
            torch.cat([context_latents, block_latents], 1)
        )
        logits = self._decode_logits_from_residual(residual)
        context_len = context_latents.shape[1]
        block_len = block_latents.shape[1]
        # Causal LM logits at position i predict token i+1. The final clean
        # context position therefore supplies the first proposed token.
        return logits[:, context_len - 1 : context_len - 1 + block_len].argmax(-1)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: DiffusionGenerationConfig,
        max_new_tokens: Optional[int] = None,
        return_speculative_stats: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor | tuple[torch.LongTensor, SpeculativeStats]:
        del kwargs
        if inputs.shape[0] != 1:
            raise NotImplementedError(
                "latent speculative generation supports batch size one"
            )
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
            clean_raw = self.extract_clean_latents(generated)
            if not bool(self.latent_stats_initialized):
                raise RuntimeError("latent normalization statistics are not configured")
            context_latents = self.normalize_latents(clean_raw)
            draft_started = time.perf_counter()
            proposed_latents = self._draft_latents(
                context_latents, proposal_len, num_steps, block_size
            )
            proposal = self.decode_latents(context_latents, proposed_latents)
            stats.draft_seconds += time.perf_counter() - draft_started
            stats.draft_calls += num_steps + int(
                self.self_conditioning_head is not None
            )
            stats.proposed_tokens += proposal_len

            verify_started = time.perf_counter()
            candidate = torch.cat([generated, proposal], dim=-1)
            verifier_logits = self.target(input_ids=candidate, use_cache=False).logits
            start = generated.shape[1] - 1
            target_tokens = verifier_logits[:, start : start + proposal_len].argmax(-1)
            accepted = int(longest_matching_prefix(proposal, target_tokens)[0])
            stats.verifier_seconds += time.perf_counter() - verify_started
            stats.verifier_calls += 1
            reached_eos = False
            if self.eos_token_id is not None and accepted:
                accepted_eos = (proposal[0, :accepted] == self.eos_token_id).nonzero(
                    as_tuple=False
                )
                if accepted_eos.numel():
                    accepted = int(accepted_eos[0, 0]) + 1
                    reached_eos = True
            stats.accepted_tokens += accepted
            stats.accepted_lengths.append(accepted)
            if accepted:
                generated = torch.cat([generated, proposal[:, :accepted]], dim=-1)
            if reached_eos:
                break
            if (
                accepted < proposal_len
                and generated.shape[1] - inputs.shape[1] < max_new_tokens
            ):
                generated = torch.cat(
                    [generated, target_tokens[:, accepted : accepted + 1]], dim=-1
                )
                stats.correction_tokens += 1
            if self.eos_token_id is not None and generated[0, -1] == self.eos_token_id:
                break
        stats.committed_tokens = generated.shape[1] - inputs.shape[1]
        stats.total_seconds = time.perf_counter() - started
        self.last_speculative_stats = stats
        return (generated, stats) if return_speculative_stats else generated


class RiemannianLatentFlowE2DConfig(LatentFlowE2DConfig):
    model_type = "riemannian_latent_flow_e2d"

    def __init__(
        self,
        scalar_loss_weight: float = 1.0,
        log_radius_mean: float = 8.02,
        log_radius_std: float = 0.1,
        scalar_prediction_clip: float = 5.0,
        radius_conditioning_width: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if scalar_loss_weight < 0:
            raise ValueError("scalar_loss_weight must be non-negative")
        if log_radius_std <= 0:
            raise ValueError("log_radius_std must be positive")
        if scalar_prediction_clip <= 0:
            raise ValueError("scalar_prediction_clip must be positive")
        if radius_conditioning_width <= 0:
            raise ValueError("radius_conditioning_width must be positive")
        self.scalar_loss_weight = scalar_loss_weight
        self.log_radius_mean = log_radius_mean
        self.log_radius_std = log_radius_std
        self.scalar_prediction_clip = scalar_prediction_clip
        self.radius_conditioning_width = radius_conditioning_width


class RiemannianLatentFlowE2D(LatentFlowE2D):
    """Spherical latent flow with a separate standardized log-radius head."""

    config_class = RiemannianLatentFlowE2DConfig

    def __init__(self, config: RiemannianLatentFlowE2DConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        hidden_size = self.target.config.hidden_size
        scalar_width = max(hidden_size // 4, 1)
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, scalar_width),
            nn.SiLU(),
            nn.Linear(scalar_width, 1),
        )
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)
        self.radius_conditioner = nn.Sequential(
            nn.Linear(1, config.radius_conditioning_width),
            nn.SiLU(),
            nn.Linear(config.radius_conditioning_width, hidden_size),
        )
        # Begin at the unit-direction baseline while allowing the model to learn
        # how previous-token radii should alter attention keys and values.
        nn.init.zeros_(self.radius_conditioner[-1].weight)
        nn.init.zeros_(self.radius_conditioner[-1].bias)
        self.register_buffer(
            "log_radius_mean",
            torch.tensor(config.log_radius_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "log_radius_std",
            torch.tensor(config.log_radius_std, dtype=torch.float32),
        )

    def latent_direction_and_scalar(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radius = latents.float().norm(dim=-1)
        standardized_log_radius = (
            torch.log(radius.clamp_min(self.config.stats_epsilon))
            - self.log_radius_mean
        ) / self.log_radius_std
        direction = F.normalize(latents.float(), dim=-1).to(latents.dtype)
        return direction, standardized_log_radius

    def reconstruct_latents(
        self, direction: torch.Tensor, standardized_log_radius: torch.Tensor
    ) -> torch.Tensor:
        clipped = standardized_log_radius.float().clamp(
            -self.config.scalar_prediction_clip,
            self.config.scalar_prediction_clip,
        )
        radius = torch.exp(self.log_radius_mean + self.log_radius_std * clipped)
        return (F.normalize(direction.float(), dim=-1) * radius.unsqueeze(-1)).to(
            direction.dtype
        )

    def _run_spherical_heads(
        self,
        clean_direction: torch.Tensor,
        clean_log_radius: torch.Tensor,
        noisy_direction: torch.Tensor,
        time_value: torch.Tensor,
        attention_mask: torch.BoolTensor,
        clean_position_ids: torch.LongTensor,
        noisy_position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean_conditioned = self._condition_context(clean_direction, clean_log_radius)
        hidden = self._run_flow_hidden(
            clean_conditioned,
            noisy_direction,
            time_value,
            attention_mask,
            clean_position_ids,
            noisy_position_ids,
        )
        ambient_velocity = self.velocity_head(hidden)
        tangent_velocity = project_to_tangent(
            noisy_direction.float(), ambient_velocity.float()
        )
        scalar_prediction = self.scalar_head(hidden).squeeze(-1).float()
        return tangent_velocity, scalar_prediction

    def _condition_context(
        self,
        clean_direction: torch.Tensor,
        clean_log_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Encode known previous-token radii without rescaling sphere points."""
        clipped_context_radius = clean_log_radius.float().clamp(
            -self.config.scalar_prediction_clip,
            self.config.scalar_prediction_clip,
        )
        radius_conditioning = self.radius_conditioner(
            clipped_context_radius.unsqueeze(-1)
        ).to(clean_direction.dtype)
        return clean_direction + radius_conditioning

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        **kwargs: Any,
    ) -> DenoiserOutput:
        del kwargs, compute_loss
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(input_ids)
        valid = attention_mask.bool()
        loss_mask = valid & ~context_mask.bool()
        clean_raw = self.extract_clean_latents(input_ids, attention_mask)
        clean_direction, scalar_target = self.latent_direction_and_scalar(clean_raw)
        noise, t = self._sample_training_path(clean_direction, t)
        noise_direction = F.normalize(noise.float(), dim=-1).to(clean_direction.dtype)
        path, target_velocity = spherical_interpolant_and_velocity(
            clean_direction, noise_direction, t
        )

        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)[None].expand(
            input_ids.shape[0], -1
        )
        flow_mask = self._flow_attention_mask(
            seq_len, self.config.block_size, input_ids.device
        )
        padding = torch.cat([valid, valid], dim=-1)
        flow_mask = flow_mask[None] & padding[:, :, None] & padding[:, None, :]
        predicted_velocity, scalar_prediction = self._run_spherical_heads(
            clean_direction,
            scalar_target,
            path,
            t,
            flow_mask,
            positions,
            positions,
        )

        direction_token_loss = (
            (predicted_velocity.float() - target_velocity.float()).square().sum(dim=-1)
        )
        scalar_token_loss = F.smooth_l1_loss(
            scalar_prediction,
            scalar_target.float(),
            reduction="none",
        )
        total_token_loss = (
            direction_token_loss + self.config.scalar_loss_weight * scalar_token_loss
        )
        direction_loss = direction_token_loss[loss_mask].mean()
        scalar_loss = scalar_token_loss[loss_mask].mean()
        total_loss = total_token_loss[loss_mask].mean()
        velocity_cosine = F.cosine_similarity(
            predicted_velocity.float(), target_velocity.float(), dim=-1
        ).clamp(-1.0, 1.0)
        angular_error = torch.acos(velocity_cosine)[loss_mask].mean()
        relative_radius_error = (
            (
                torch.exp(
                    self.log_radius_std
                    * (scalar_prediction - scalar_target.float()).clamp(-20.0, 20.0)
                )
                - 1.0
            )
            .abs()[loss_mask]
            .mean()
        )

        return DenoiserOutput(
            denoiser_output=predicted_velocity,
            tokens_mask=loss_mask.float(),
            loss=total_loss,
            nlls=total_token_loss,
            other_loss_terms={
                "flow_loss": total_loss,
                "direction_loss": direction_loss,
                "scalar_loss": scalar_loss,
                "angular_error": angular_error,
                "radius_relative_error": relative_radius_error,
            },
            flow_loss=total_loss,
            direction_loss=direction_loss,
            scalar_loss=scalar_loss,
            angular_error=angular_error,
            radius_relative_error=relative_radius_error,
            flow_timesteps=t,
        )

    def _draft_latents(
        self,
        context_direction: torch.Tensor,
        context_log_radius: torch.Tensor,
        block_len: int,
        num_steps: int,
    ) -> torch.Tensor:
        batch_size = context_direction.shape[0]
        canvas = F.normalize(
            torch.randn(
                batch_size,
                block_len,
                context_direction.shape[-1],
                device=context_direction.device,
                dtype=context_direction.dtype,
            ).float(),
            dim=-1,
        ).to(context_direction.dtype)
        context_len = context_direction.shape[1]
        clean_positions = torch.arange(context_len, device=canvas.device)[None].expand(
            batch_size, -1
        )
        noisy_positions = torch.arange(
            context_len, context_len + block_len, device=canvas.device
        )[None].expand(batch_size, -1)
        mask = self._inference_attention_mask(context_len, block_len, canvas.device)
        step_size = -1.0 / num_steps
        for step in range(num_steps):
            time_value = torch.full(
                (batch_size, block_len),
                1.0 - step / num_steps,
                device=canvas.device,
                dtype=canvas.dtype,
            )
            velocity, _ = self._run_spherical_heads(
                context_direction,
                context_log_radius,
                canvas,
                time_value,
                mask,
                clean_positions,
                noisy_positions,
            )
            canvas = sphere_expmap(canvas, velocity, step_size)

        endpoint_time = torch.zeros(
            (batch_size, block_len), device=canvas.device, dtype=canvas.dtype
        )
        _, scalar_prediction = self._run_spherical_heads(
            context_direction,
            context_log_radius,
            canvas,
            endpoint_time,
            mask,
            clean_positions,
            noisy_positions,
        )
        return self.reconstruct_latents(canvas, scalar_prediction)

    def decode_latents(
        self, context_latents: torch.Tensor, block_latents: torch.Tensor
    ) -> torch.LongTensor:
        residual = torch.cat([context_latents, block_latents], dim=1)
        seq_len = residual.shape[1]
        positions = torch.arange(seq_len, device=residual.device)[None].expand(
            residual.shape[0], -1
        )
        position_embeddings = self.target.model.rotary_emb(residual, positions)
        causal = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=residual.device)
        )
        decoded = self.target.model.layers[self.target_layer_idx](
            residual,
            attention_mask=self._as_additive_mask(causal, residual.dtype),
            position_ids=positions,
            use_cache=False,
            cache_position=positions[0],
            position_embeddings=position_embeddings,
        )[0]
        logits = self.target.lm_head(self.target.model.norm(decoded))
        context_len = context_latents.shape[1]
        block_len = block_latents.shape[1]
        return logits[:, context_len - 1 : context_len - 1 + block_len].argmax(-1)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: DiffusionGenerationConfig,
        max_new_tokens: Optional[int] = None,
        return_speculative_stats: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor | tuple[torch.LongTensor, SpeculativeStats]:
        del kwargs
        if inputs.shape[0] != 1:
            raise NotImplementedError(
                "latent speculative generation supports batch size one"
            )
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
            clean_raw = self.extract_clean_latents(generated)
            context_direction, context_log_radius = self.latent_direction_and_scalar(
                clean_raw
            )
            draft_started = time.perf_counter()
            proposed_latents = self._draft_latents(
                context_direction, context_log_radius, proposal_len, num_steps
            )
            proposal = self.decode_latents(clean_raw, proposed_latents)
            stats.draft_seconds += time.perf_counter() - draft_started
            stats.draft_calls += num_steps + 1
            stats.proposed_tokens += proposal_len

            verify_started = time.perf_counter()
            candidate = torch.cat([generated, proposal], dim=-1)
            verifier_logits = self.target(input_ids=candidate, use_cache=False).logits
            start = generated.shape[1] - 1
            target_tokens = verifier_logits[:, start : start + proposal_len].argmax(-1)
            accepted = int(longest_matching_prefix(proposal, target_tokens)[0])
            stats.verifier_seconds += time.perf_counter() - verify_started
            stats.verifier_calls += 1
            reached_eos = False
            if self.eos_token_id is not None and accepted:
                accepted_eos = (proposal[0, :accepted] == self.eos_token_id).nonzero(
                    as_tuple=False
                )
                if accepted_eos.numel():
                    accepted = int(accepted_eos[0, 0]) + 1
                    reached_eos = True
            stats.accepted_tokens += accepted
            stats.accepted_lengths.append(accepted)
            if accepted:
                generated = torch.cat([generated, proposal[:, :accepted]], dim=-1)
            if reached_eos:
                break
            if (
                accepted < proposal_len
                and generated.shape[1] - inputs.shape[1] < max_new_tokens
            ):
                generated = torch.cat(
                    [generated, target_tokens[:, accepted : accepted + 1]], dim=-1
                )
                stats.correction_tokens += 1
            if self.eos_token_id is not None and generated[0, -1] == self.eos_token_id:
                break
        stats.committed_tokens = generated.shape[1] - inputs.shape[1]
        stats.total_seconds = time.perf_counter() - started
        self.last_speculative_stats = stats
        return (generated, stats) if return_speculative_stats else generated
