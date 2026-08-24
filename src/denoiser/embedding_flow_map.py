"""Joint AR SFT and embedding-space flow-map self-distillation."""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.backbone.position_gated_lora import (
    inject_position_gated_lora,
    set_position_lora_mode,
)
from src.denoiser.base import Denoiser, DenoiserConfig, DenoiserOutput
from src.denoiser.speculative import (
    SpeculativeStats,
    longest_matching_prefix,
)


class EmbeddingFlowMapConfig(DenoiserConfig):
    model_type = "embedding_flow_map"
    auto_map = {
        "AutoConfig": "embedding_flow_map.EmbeddingFlowMapConfig",
        "AutoModel": "embedding_flow_map.EmbeddingFlowMap",
        "AutoModelForCausalLM": "embedding_flow_map.EmbeddingFlowMap",
    }

    def __init__(
        self,
        block_size: int = 8,
        shared_lora_rank: int = 16,
        ar_lora_rank: int = 16,
        flow_lora_rank: int = 32,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        ar_loss_weight: float = 1.0,
        diagonal_loss_weight: float = 1.0,
        semigroup_loss_weight: float = 0.0,
        diagonal_min_time: float = 0.8,
        max_time_jump: float = 0.25,
        boundary_probability: float = 0.0,
        time_epsilon: float = 1e-4,
        generation_mode: str = "verified",
        inference_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0 <= diagonal_min_time <= 1:
            raise ValueError("diagonal_min_time must be in [0, 1]")
        if not 0 < max_time_jump <= 1:
            raise ValueError("max_time_jump must be in (0, 1]")
        if not 0 <= boundary_probability <= 1:
            raise ValueError("boundary_probability must be in [0, 1]")
        self.block_size = block_size
        self.shared_lora_rank = shared_lora_rank
        self.ar_lora_rank = ar_lora_rank
        self.flow_lora_rank = flow_lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = tuple(lora_target_modules)
        self.ar_loss_weight = ar_loss_weight
        self.diagonal_loss_weight = diagonal_loss_weight
        self.semigroup_loss_weight = semigroup_loss_weight
        self.diagonal_min_time = diagonal_min_time
        self.max_time_jump = max_time_jump
        self.boundary_probability = boundary_probability
        self.time_epsilon = time_epsilon
        if generation_mode not in {"ar", "draft", "verified"}:
            raise ValueError("generation_mode must be 'ar', 'draft', or 'verified'")
        self.generation_mode = generation_mode
        self.inference_steps = inference_steps


class EmbeddingFlowMap(Denoiser):
    """A two-time denoiser trained jointly with a causal language model."""

    config_class = EmbeddingFlowMapConfig

    def __init__(self, config: EmbeddingFlowMapConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.target = self.backbone.encoder
        # Some encoder/decoder wrappers keep additional aliases to the same model
        # and may mark decoder parameters trainable during construction. Freeze the
        # complete wrapper before inserting the only trainable transformer weights.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.lora_module_names = inject_position_gated_lora(
            self.target.model,
            config.lora_target_modules,
            shared_rank=config.shared_lora_rank,
            ar_rank=config.ar_lora_rank,
            flow_rank=config.flow_lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        hidden_size = self.target.config.hidden_size
        self.time_mlp = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.flow_head = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.flow_head.weight)

        embedding_weight = self.target.get_input_embeddings().weight.detach().float()
        embedding_mean = embedding_weight.mean(dim=0)
        centered = embedding_weight - embedding_mean
        # A scalar scale preserves angular relationships in the frozen embedding table.
        embedding_scale = centered.square().mean().sqrt().clamp_min(1e-6)
        self.register_buffer("embedding_mean", embedding_mean)
        self.register_buffer("embedding_scale", embedding_scale)
        self.last_speculative_stats = SpeculativeStats()

    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return ((embeddings - self.embedding_mean) / self.embedding_scale).to(
            embeddings.dtype
        )

    def denormalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return (embeddings * self.embedding_scale + self.embedding_mean).to(
            embeddings.dtype
        )

    @staticmethod
    def _as_additive_mask(mask: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.where(mask, 0.0, torch.finfo(dtype).min).to(dtype)[:, None]

    @staticmethod
    def build_block_attention_mask(
        valid: torch.BoolTensor,
        context: torch.BoolTensor,
        block_size: int,
    ) -> torch.BoolTensor:
        """Build strict-causal clean rows and conditioned block-denoising rows."""
        batch_size, seq_len = valid.shape
        device = valid.device
        positions = torch.arange(seq_len, device=device)
        clean_causal = positions[:, None] >= positions[None, :]

        target = valid & ~context
        target_ordinal = target.long().cumsum(dim=-1) - 1
        block_id = torch.where(
            target, target_ordinal // block_size, torch.full_like(target_ordinal, -1)
        )
        noisy_to_clean = context[:, None, :] | (
            target[:, None, :]
            & (block_id[:, None, :] < block_id[:, :, None])
        )
        noisy_canvas = (
            target[:, None, :]
            & target[:, :, None]
            & (block_id[:, None, :] == block_id[:, :, None])
        )

        mask = torch.zeros(
            batch_size, 2 * seq_len, 2 * seq_len, dtype=torch.bool, device=device
        )
        mask[:, :seq_len, :seq_len] = clean_causal
        mask[:, seq_len:, :seq_len] = noisy_to_clean
        mask[:, seq_len:, seq_len:] = noisy_canvas
        doubled_valid = torch.cat([valid, valid], dim=-1)
        return mask & doubled_valid[:, :, None] & doubled_valid[:, None, :]

    @staticmethod
    def _time_features(
        start: torch.Tensor, end: torch.Tensor, length: int
    ) -> torch.Tensor:
        if start.ndim == 1:
            start = start[:, None].expand(-1, length)
        if end.ndim == 1:
            end = end[:, None].expand(-1, length)
        return torch.stack(
            [
                start,
                end,
                end - start,
                torch.sin(math.pi * start),
                torch.cos(math.pi * start),
                torch.sin(math.pi * end),
                torch.cos(math.pi * end),
            ],
            dim=-1,
        )

    def _decode_embedding_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        raw = self.denormalize_embeddings(embeddings)
        return self.target.lm_head(self.target.model.norm(raw))

    def _run_two_time_denoiser(
        self,
        clean_embeddings: torch.Tensor,
        state: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = state.shape
        time_features = self._time_features(start, end, seq_len).to(state.dtype)
        conditioned_state = state + self.time_mlp(time_features)
        inputs = torch.cat([clean_embeddings, conditioned_state], dim=1)
        ar_gate = torch.zeros(batch_size, 2 * seq_len, device=state.device)
        flow_gate = torch.zeros_like(ar_gate)
        ar_gate[:, :seq_len] = 1
        flow_gate[:, seq_len:] = 1
        mode_mask = torch.stack([ar_gate, flow_gate], dim=-1)
        set_position_lora_mode(self.target.model, mode_mask)
        positions = torch.arange(seq_len, device=state.device)[None].expand(
            batch_size, -1
        )
        positions = torch.cat([positions, positions], dim=-1)
        output = self.target.model(
            inputs_embeds=inputs,
            attention_mask=self._as_additive_mask(attention_mask, inputs.dtype),
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        clean_hidden, flow_hidden = output.split(seq_len, dim=1)
        delta = state + self.flow_head(flow_hidden)
        ar_logits = self.target.lm_head(self.target.model.norm(clean_hidden))
        flow_logits = self._decode_embedding_logits(delta)
        return delta, ar_logits, flow_logits

    def flow_map(
        self,
        state: torch.Tensor,
        delta: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        while start.ndim < state.ndim:
            start = start.unsqueeze(-1)
            end = end.unsqueeze(-1)
        denominator = (1.0 - start).clamp_min(self.config.time_epsilon)
        return ((1.0 - end) / denominator) * state + (
            (end - start) / denominator
        ) * delta

    @staticmethod
    def _hard_ce(
        logits: torch.Tensor, labels: torch.LongTensor, mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        if not bool(mask.any()):
            return logits.sum() * 0.0, losses
        return losses[mask].mean(), losses

    @staticmethod
    def _soft_ce(
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        per_token = -(teacher_probs * F.log_softmax(student_logits.float(), -1)).sum(-1)
        if not bool(mask.any()):
            return student_logits.sum() * 0.0
        return per_token[mask].mean()

    def _sample_semigroup_times(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        jump = torch.rand(batch_size, device=device, dtype=dtype)
        jump = jump * self.config.max_time_jump
        jump = jump.clamp_min(self.config.time_epsilon)
        start = torch.rand(batch_size, device=device, dtype=dtype) * (1.0 - jump)
        end = start + jump
        middle = 0.5 * (start + end)
        if self.config.boundary_probability:
            boundary = (
                torch.rand(batch_size, device=device)
                < self.config.boundary_probability
            )
            start = torch.where(boundary, torch.zeros_like(start), start)
            middle = torch.where(boundary, torch.full_like(middle, 0.5), middle)
            end = torch.where(boundary, torch.ones_like(end), end)
        return start, middle, end

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> DenoiserOutput:
        del kwargs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(input_ids)
        valid = attention_mask.bool()
        context = context_mask.bool() & valid
        target_mask = valid & ~context
        clean_raw = self.target.get_input_embeddings()(input_ids).detach()
        clean = self.normalize_embeddings(clean_raw)
        noise = torch.randn_like(clean)
        batch_size = input_ids.shape[0]
        diagonal_time = self.config.diagonal_min_time + (
            1.0 - self.config.diagonal_min_time
        ) * torch.rand(batch_size, device=input_ids.device, dtype=clean.dtype)
        expanded_time = diagonal_time[:, None, None]
        diagonal_state = (1.0 - expanded_time) * noise + expanded_time * clean
        block_mask = self.build_block_attention_mask(
            valid, context, self.config.block_size
        )
        _, ar_logits, diagonal_logits = self._run_two_time_denoiser(
            clean, diagonal_state, diagonal_time, diagonal_time, block_mask
        )

        ar_mask = valid[:, 1:] & ~context[:, 1:]
        ar_loss, ar_nll = self._hard_ce(
            ar_logits[:, :-1], input_ids[:, 1:], ar_mask
        )
        diagonal_loss, diagonal_nll = self._hard_ce(
            diagonal_logits, input_ids, target_mask
        )

        semigroup_loss = diagonal_logits.sum() * 0.0
        if self.config.semigroup_loss_weight > 0:
            start, middle, end = self._sample_semigroup_times(
                batch_size, input_ids.device, clean.dtype
            )
            start_state = (1.0 - start[:, None, None]) * noise + start[
                :, None, None
            ] * clean
            with torch.no_grad():
                first_delta, _, first_logits = self._run_two_time_denoiser(
                    clean, start_state, start, middle, block_mask
                )
                middle_state = self.flow_map(
                    start_state, first_delta, start, middle
                )
                _, _, second_logits = self._run_two_time_denoiser(
                    clean, middle_state, middle, end, block_mask
                )
                first_probs = F.softmax(first_logits.float(), dim=-1)
                second_probs = F.softmax(second_logits.float(), dim=-1)
                gamma = ((1.0 - end) * (middle - start)) / (
                    (1.0 - middle) * (end - start)
                ).clamp_min(self.config.time_epsilon)
                teacher_probs = (
                    gamma[:, None, None] * first_probs
                    + (1.0 - gamma[:, None, None]) * second_probs
                ).detach()
            _, _, direct_logits = self._run_two_time_denoiser(
                clean, start_state, start, end, block_mask
            )
            semigroup_loss = self._soft_ce(
                direct_logits, teacher_probs, target_mask
            )

        total_loss = (
            self.config.ar_loss_weight * ar_loss
            + self.config.diagonal_loss_weight * diagonal_loss
            + self.config.semigroup_loss_weight * semigroup_loss
        )
        nlls = torch.zeros_like(input_ids, dtype=diagonal_nll.dtype)
        nlls[:, 1:] = ar_nll
        return DenoiserOutput(
            logits=ar_logits,
            denoiser_output=diagonal_logits,
            tokens_mask=target_mask.float(),
            loss=total_loss,
            nlls=nlls,
            other_loss_terms={
                "ar_loss": ar_loss,
                "diagonal_ce": diagonal_loss,
                "semigroup_ce": semigroup_loss,
            },
            flow_loss=diagonal_loss + semigroup_loss,
            flow_timesteps=diagonal_time,
        )

    def _prepare_inputs(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("EmbeddingFlowMap implements forward directly")

    def _compute_loss(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("EmbeddingFlowMap implements forward directly")

    def _run_ar_logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Run the jointly trained shared+AR LoRA path with strict causal attention."""
        batch_size, seq_len = input_ids.shape
        embeddings = self.target.get_input_embeddings()(input_ids)
        mode_mask = embeddings.new_zeros(batch_size, seq_len, 2)
        mode_mask[..., 0] = 1
        set_position_lora_mode(self.target.model, mode_mask)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
        )[None].expand(batch_size, -1, -1)
        positions = torch.arange(seq_len, device=input_ids.device)[None].expand(
            batch_size, -1
        )
        hidden = self.target.model(
            inputs_embeds=embeddings,
            attention_mask=self._as_additive_mask(causal, embeddings.dtype),
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self.target.lm_head(self.target.model.norm(hidden))

    @torch.no_grad()
    def generate_ar(
        self, inputs: torch.LongTensor, max_new_tokens: int
    ) -> torch.LongTensor:
        generated = inputs
        batch_size = inputs.shape[0]
        mode_mask = self.target.get_input_embeddings().weight.new_zeros(
            batch_size, inputs.shape[1], 2
        )
        mode_mask[..., 0] = 1
        set_position_lora_mode(self.target.model, mode_mask)
        output = self.target(input_ids=inputs, use_cache=True, return_dict=True)
        past_key_values = output.past_key_values
        token = output.logits[:, -1:].argmax(-1)
        for step in range(max_new_tokens):
            generated = torch.cat([generated, token], dim=-1)
            if self.eos_token_id is not None and bool(
                (token == self.eos_token_id).all()
            ):
                break
            if step + 1 == max_new_tokens:
                break
            mode_mask = mode_mask.new_zeros(batch_size, 1, 2)
            mode_mask[..., 0] = 1
            set_position_lora_mode(self.target.model, mode_mask)
            output = self.target(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            token = output.logits[:, -1:].argmax(-1)
        return generated

    @torch.no_grad()
    def _draft_block(
        self, context_ids: torch.LongTensor, block_len: int, num_steps: int
    ) -> torch.LongTensor:
        context_len = context_ids.shape[1]
        placeholder = torch.full(
            (context_ids.shape[0], block_len),
            self.pad_token_id,
            dtype=context_ids.dtype,
            device=context_ids.device,
        )
        packed_ids = torch.cat([context_ids, placeholder], dim=-1)
        valid = torch.ones_like(packed_ids, dtype=torch.bool)
        context = torch.zeros_like(valid)
        context[:, :context_len] = True
        clean = self.normalize_embeddings(
            self.target.get_input_embeddings()(packed_ids)
        )
        state = torch.zeros_like(clean)
        state[:, context_len:] = torch.randn_like(state[:, context_len:])
        attention = self.build_block_attention_mask(
            valid, context, self.config.block_size
        )
        time_grid = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=context_ids.device,
            dtype=clean.dtype,
        )
        for step in range(num_steps):
            start = time_grid[step].expand(context_ids.shape[0])
            end = time_grid[step + 1].expand(context_ids.shape[0])
            delta, _, _ = self._run_two_time_denoiser(
                clean, state, start, end, attention
            )
            state = self.flow_map(state, delta, start, end)
        logits = self._decode_embedding_logits(state[:, context_len:])
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def generate_draft(
        self,
        inputs: torch.LongTensor,
        max_new_tokens: int,
        block_size: int,
        num_steps: int,
    ) -> torch.LongTensor:
        output_ids = inputs
        finished = torch.zeros(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        generated = 0
        while generated < max_new_tokens and not bool(finished.all()):
            current_block = min(block_size, max_new_tokens - generated)
            next_block = self._draft_block(output_ids, current_block, num_steps)
            if self.eos_token_id is not None:
                # Keep a rectangular batch while preventing tokens after the first EOS
                # from affecting the returned text.
                for batch_index in range(next_block.shape[0]):
                    eos = torch.nonzero(
                        next_block[batch_index] == self.eos_token_id,
                        as_tuple=False,
                    ).flatten()
                    if eos.numel():
                        first = int(eos[0])
                        next_block[batch_index, first + 1 :] = self.pad_token_id
                        finished[batch_index] = True
            output_ids = torch.cat([output_ids, next_block], dim=-1)
            generated += current_block
        return output_ids

    @torch.no_grad()
    def generate_verified(
        self,
        inputs: torch.LongTensor,
        max_new_tokens: int,
        block_size: int,
        num_steps: int,
        return_speculative_stats: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor | tuple[torch.LongTensor, SpeculativeStats]:
        del kwargs
        if inputs.shape[0] != 1:
            raise NotImplementedError(
                "verified generation currently supports batch size one"
            )
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
            verifier_logits = self._run_ar_logits(candidate)
            start_index = generated.shape[1] - 1
            target_tokens = verifier_logits[
                :, start_index : start_index + proposal_len
            ].argmax(-1)
            accepted = int(longest_matching_prefix(proposal, target_tokens)[0])
            stats.verifier_seconds += time.perf_counter() - verify_started
            stats.verifier_calls += 1
            reached_eos = False
            if self.eos_token_id is not None and accepted:
                accepted_eos = torch.nonzero(
                    proposal[0, :accepted] == self.eos_token_id, as_tuple=False
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
            if accepted < proposal_len:
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

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[Any] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        generation_mode: Optional[str] = None,
        return_speculative_stats: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor | tuple[torch.LongTensor, SpeculativeStats]:
        if inputs is None:
            raise ValueError("embedding flow-map generation requires prompt inputs")
        if max_new_tokens is None:
            max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if max_new_tokens is None and max_length is not None:
            max_new_tokens = max(0, max_length - inputs.shape[1])
        max_new_tokens = int(max_new_tokens or self.config.block_size)
        block_size = int(
            getattr(generation_config, "block_size", self.config.block_size)
        )
        num_steps = int(
            getattr(generation_config, "num_steps", self.config.inference_steps)
        )
        mode = generation_mode or self.config.generation_mode
        if mode == "ar":
            return self.generate_ar(inputs, max_new_tokens)
        if mode == "draft":
            return self.generate_draft(inputs, max_new_tokens, block_size, num_steps)
        if mode == "verified":
            return self.generate_verified(
                inputs,
                max_new_tokens,
                block_size,
                num_steps,
                return_speculative_stats=return_speculative_stats,
                **kwargs,
            )
        raise ValueError("generation_mode must be 'ar', 'draft', or 'verified'")
