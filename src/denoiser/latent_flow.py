"""Latent flow-matching drafter with full-target speculative verification."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.denoiser.base import Denoiser, DenoiserConfig, DenoiserOutput
from src.denoiser.diffusion import DiffusionGenerationConfig
from src.denoiser.speculative import (
    SpeculativeStats,
    linear_flow_sample,
    longest_matching_prefix,
)


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
        self.velocity_head = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.velocity_head.weight)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.flow_layers.parameters():
            parameter.requires_grad = True
        self.register_buffer("latent_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("latent_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("latent_stats_initialized", torch.tensor(False))
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
        if bool(self.latent_stats_initialized):
            return
        values = clean_latents[valid_mask.bool()].float()
        self.latent_mean.copy_(values.mean())
        self.latent_std.copy_(values.std().clamp_min(self.config.stats_epsilon))
        self.latent_stats_initialized.fill_(True)

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

    def _run_flow_layers(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        time_value: torch.Tensor,
        attention_mask: torch.BoolTensor,
        clean_position_ids: torch.LongTensor,
        noisy_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, noisy_len, _ = noisy.shape
        if time_value.ndim == 1:
            time_value = time_value[:, None].expand(batch_size, noisy_len)
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
        return self.velocity_head(hidden[:, -noisy_len:])

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
        self.initialize_latent_stats(clean_raw, valid)
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
        predicted_velocity = self._run_flow_layers(
            clean,
            noisy,
            t,
            flow_mask,
            positions,
            positions,
        )
        token_mse = F.mse_loss(
            predicted_velocity.float(), target_velocity.float(), reduction="none"
        ).mean(-1)
        flow_loss = token_mse[loss_mask].mean()
        return DenoiserOutput(
            denoiser_output=predicted_velocity,
            tokens_mask=loss_mask.float(),
            loss=flow_loss,
            nlls=token_mse,
            other_loss_terms={"flow_loss": flow_loss},
            flow_loss=flow_loss,
            flow_timesteps=t,
        )

    def _prepare_inputs(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("LatentFlowE2D implements forward directly")

    def _compute_loss(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("LatentFlowE2D implements forward directly")

    def _draft_latents(
        self, context_latents: torch.Tensor, block_len: int, num_steps: int
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
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_value = torch.full(
                (batch_size, block_len),
                1.0 - step * dt,
                device=canvas.device,
                dtype=canvas.dtype,
            )
            velocity = self._run_flow_layers(
                context_latents,
                canvas,
                t_value,
                mask,
                clean_positions,
                noisy_positions,
            )
            canvas = canvas - dt * velocity
        return canvas

    def decode_latents(
        self, context_latents: torch.Tensor, block_latents: torch.Tensor
    ) -> torch.LongTensor:
        residual = self.denormalize_latents(
            torch.cat([context_latents, block_latents], 1)
        )
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
                self.initialize_latent_stats(
                    clean_raw, torch.ones(clean_raw.shape[:2], device=clean_raw.device)
                )
            context_latents = self.normalize_latents(clean_raw)
            draft_started = time.perf_counter()
            proposed_latents = self._draft_latents(
                context_latents, proposal_len, num_steps
            )
            proposal = self.decode_latents(context_latents, proposed_latents)
            stats.draft_seconds += time.perf_counter() - draft_started
            stats.draft_calls += num_steps
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
                accepted_eos = (
                    proposal[0, :accepted] == self.eos_token_id
                ).nonzero(as_tuple=False)
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
