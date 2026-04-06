from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    DynamicCache,
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput

from src.denoiser.ar import AR, ARConfig
from src.denoiser.base import DenoiserInput, DenoiserOutput, LossAndNllOutput


class MTPConfig(ARConfig):
    """Configuration class for multi-token prediction (MTP) models."""

    model_type = "mtp"
    auto_map = {
        "AutoConfig": "mtp.MTPConfig",
        "AutoModel": "mtp.MTP",
        "AutoModelForCausalLM": "mtp.MTP",
    }

    def __init__(
        self,
        length: Optional[int] = None,
        backbone_config: Optional[Dict[str, Any]] = None,
        tokenization_config: Optional[Dict[str, Any]] = None,
        noise_config: None = None,
        time_conditioned_backbone: Optional[bool] = None,
        target_loss_weight: float = 1.0,
        future_loss_weight: float = 0.1,
        num_future_heads: int = 4,
        speculative_draft_len: int = 4,
        mtp_per_block: bool = False,
        mtp_block_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            length=length,
            backbone_config=backbone_config,
            noise_config=noise_config,
            tokenization_config=tokenization_config,
            time_conditioned_backbone=time_conditioned_backbone,
            **kwargs,
        )
        self.target_loss_weight = target_loss_weight
        self.future_loss_weight = future_loss_weight
        self.num_future_heads = num_future_heads
        self.speculative_draft_len = speculative_draft_len
        self.mtp_per_block = mtp_per_block
        self.mtp_block_size = mtp_block_size


class MTP(AR):
    """Autoregressive model with additional future-token prediction heads.

    The base model head predicts next-token (offset 1) with weight 1.0.
    Additional linear heads predict offsets 2..(num_future_heads + 1), each with
    weight 0.1 by default.
    """

    config_class = MTPConfig

    def __init__(
        self,
        config: MTPConfig,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.target_loss_weight = float(getattr(config, "target_loss_weight", 1.0))
        self.future_loss_weight = float(getattr(config, "future_loss_weight", 0.1))
        self.num_future_heads = int(getattr(config, "num_future_heads", 4))
        self.speculative_draft_len = int(getattr(config, "speculative_draft_len", 4))
        self.mtp_per_block = bool(getattr(config, "mtp_per_block", False))
        configured_block_size = getattr(config, "mtp_block_size", self.num_future_heads)
        if configured_block_size is None:
            configured_block_size = self.num_future_heads
        self.mtp_block_size = max(1, int(configured_block_size))

        hidden_size = getattr(self.backbone.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config.")

        self.future_token_heads = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.num_future_heads)]
        )

    def _get_norm(self):
        return self.backbone.model.model.norm

    def _get_lm_head(self):
        return self.backbone.model.lm_head

    @staticmethod
    def _truncate_cache(cache: DynamicCache, target_cache_len: int) -> DynamicCache:
        for layer_idx in range(len(cache)):
            cache.key_cache[layer_idx] = cache.key_cache[layer_idx][
                ..., :target_cache_len, :
            ]
            cache.value_cache[layer_idx] = cache.value_cache[layer_idx][
                ..., :target_cache_len, :
            ]
        return cache

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        compute_loss: Optional[bool] = True,
        **kwargs,
    ) -> DenoiserOutput:
        denoiser_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            t=t,
        )

        need_hidden_states = bool(compute_loss)
        kwargs = dict(kwargs)
        kwargs.setdefault("output_hidden_states", need_hidden_states)
        backbone_output = self._backbone_forward(denoiser_inputs, **kwargs)

        new_past_key_values = getattr(backbone_output, "past_key_values", None)
        hidden_states = getattr(backbone_output, "hidden_states", None)
        backbone_logits = getattr(backbone_output, "logits", backbone_output[0])

        denoiser_output = self._forward(backbone_logits, denoiser_inputs, **kwargs)

        if compute_loss:
            loss_and_nll = self._compute_loss(
                model_output=denoiser_output,
                denoiser_inputs=denoiser_inputs,
                hidden_states=hidden_states,
                **kwargs,
            )
            loss = loss_and_nll.loss
            nlls = loss_and_nll.nlls
            other_loss_terms = loss_and_nll.other_loss_terms
        else:
            loss, nlls = None, None
            other_loss_terms = {}

        return DenoiserOutput(
            denoiser_output=denoiser_output,
            logits=backbone_logits,
            past_key_values=new_past_key_values,
            tokens_mask=denoiser_inputs.tokens_mask,
            loss=loss,
            nlls=nlls,
            other_loss_terms=other_loss_terms,
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        targets = denoiser_inputs.x0.squeeze(-1)  # (B, L)
        mask = denoiser_inputs.tokens_mask
        if mask is None:
            mask = torch.ones_like(targets, dtype=model_output.dtype)

        def _masked_ce(
            log_probs: torch.FloatTensor,
            local_targets: torch.LongTensor,
            local_mask: torch.FloatTensor,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
            token_nlls = -torch.gather(log_probs, -1, local_targets.unsqueeze(-1)).squeeze(-1)
            masked_nlls = token_nlls * local_mask
            token_count = local_mask.sum(dim=-1).clamp_min(1.0)
            token_loss = (masked_nlls.sum(dim=-1) / token_count).mean()
            return token_loss, masked_nlls

        # Base next-token loss (weight 1.0 by default).
        target_loss, target_nlls = _masked_ce(model_output, targets, mask)

        other_loss_terms: Dict[str, Any] = {
            "target_loss": target_loss.detach(),
        }

        aux_losses: List[torch.Tensor] = []
        if hidden_states is not None and len(hidden_states) > 0:
            final_hidden = hidden_states[-1]
            norm = self._get_norm()
            lm_head = self._get_lm_head()

            block_end_mask_full: Optional[torch.Tensor] = None
            if self.mtp_per_block:
                # Apply future-head loss only at block ends; block size is configurable.
                seq_len_full = targets.shape[1]
                device = targets.device
                block_pos = torch.arange(seq_len_full, device=device)
                block_end_mask_full = ((block_pos + 1) % self.mtp_block_size == 0)
                if seq_len_full > 0:
                    block_end_mask_full[-1] = True
                block_end_mask_full = block_end_mask_full.unsqueeze(0).to(mask.dtype)

            for head_idx in range(self.num_future_heads):
                offset = head_idx + 2  # predict token t+2, ..., t+5 for 4 heads
                if targets.shape[1] <= (offset - 1):
                    continue

                offset_targets = targets[:, (offset - 1) :]
                offset_mask = mask[:, (offset - 1) :]
                seq_len = offset_targets.shape[1]

                if block_end_mask_full is not None:
                    offset_mask = offset_mask * block_end_mask_full[:, :seq_len]

                hidden_slice = final_hidden[:, :seq_len, :]
                projected = self.future_token_heads[head_idx](
                    hidden_slice.to(self.future_token_heads[head_idx].weight.dtype)
                )
                projected = projected.to(norm.weight.dtype)
                offset_logits = lm_head(norm(projected))
                offset_log_probs = F.log_softmax(offset_logits, dim=-1)

                aux_loss, _ = _masked_ce(offset_log_probs, offset_targets, offset_mask)
                aux_losses.append(aux_loss)
                other_loss_terms[f"future_offset_{offset}_loss"] = aux_loss.detach()

        if aux_losses:
            loss = self.target_loss_weight * target_loss + self.future_loss_weight * torch.stack(aux_losses).sum()
        else:
            loss = self.target_loss_weight * target_loss

        return LossAndNllOutput(loss=loss, nlls=target_nlls, other_loss_terms=other_loss_terms)

    def _sample_next_token(
        self,
        logits: torch.FloatTensor,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList],
        do_sample: bool,
    ) -> torch.LongTensor:
        scores = logits
        if logits_processor is not None:
            scores = logits_processor(input_ids, scores)
        if do_sample:
            probs = torch.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return scores.argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        disable_pbar: Optional[bool] = None,
        use_mtp_speculation: Optional[bool] = None,
        mtp_draft_len: Optional[int] = None,
        **kwargs,
    ) -> Union[
        GenerateOutput,
        torch.LongTensor,
        Tuple[
            torch.LongTensor,
            Tuple[int, int],
            Tuple[List[int], int],
        ],
    ]:
        if inputs is None:
            assert batch_size is not None and device is not None
            inputs = torch.full(
                (batch_size, 1),
                fill_value=self.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        if max_new_tokens is None:
            if generation_config is not None and generation_config.max_new_tokens is not None:
                max_new_tokens = generation_config.max_new_tokens
            elif max_length is not None:
                max_new_tokens = max_length - inputs.shape[-1]
            else:
                max_new_tokens = 512

        if max_new_tokens <= 0:
            self._last_draft_position_acceptance = None
            return inputs

        if use_mtp_speculation is None:
            use_mtp_speculation = False
        if not use_mtp_speculation:
            self._last_draft_position_acceptance = None
            return self.backbone.model.generate(
                inputs=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation_config,
                logits_processor=logits_processor,
                # Keep parity with AR.generate: external stopping criteria can be
                # prompt-sensitive (e.g., regex on "\\boxed{...}") and may stop
                # immediately when the prompt already contains a match.
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        do_sample = bool(
            getattr(generation_config, "do_sample", False)
            if generation_config is not None
            else False
        )
        draft_len = (
            self.speculative_draft_len
            if mtp_draft_len is None
            else int(mtp_draft_len)
        )
        draft_len = max(0, min(draft_len, self.num_future_heads))

        batch_size = inputs.shape[0]
        assert batch_size == 1, "Batched MTP speculative decoding is not supported yet"

        generated = inputs.clone()
        prompt_len = inputs.shape[1]
        full_cache = DynamicCache()

        if prompt_len > 1:
            prefill_output = self.backbone.model(
                input_ids=inputs[:, :-1],
                past_key_values=full_cache,
                use_cache=True,
            )
            full_cache = prefill_output.past_key_values

        current_input = inputs[:, -1:]
        warmup_output = self.backbone.model(
            input_ids=current_input,
            past_key_values=full_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        full_cache = warmup_output.past_key_values
        current_anchor_hidden = warmup_output.hidden_states[-1][:, -1, :]
        next_token = self._sample_next_token(
            logits=warmup_output.logits[:, -1, :],
            input_ids=generated,
            logits_processor=logits_processor,
            do_sample=do_sample,
        )

        norm = self._get_norm()
        lm_head = self._get_lm_head()

        total_generated_tokens = 0
        total_accepted_tokens = 0
        total_accepted_lengths: List[int] = []
        accept_counts = 0

        draft_position_attempt_counts: List[int] = [0 for _ in range(draft_len)]
        draft_position_accept_counts: List[int] = [0 for _ in range(draft_len)]

        tokens_generated = 0
        done = False

        pbar = tqdm(
            total=max_new_tokens,
            desc="MTP Generate",
            disable=disable_pbar,
        )

        while tokens_generated < max_new_tokens and not done:
            remaining = max_new_tokens - tokens_generated

            if remaining == 1:
                generated = torch.cat([generated, next_token], dim=-1)
                tokens_generated += 1
                pbar.update(1)
                if stopping_criteria is not None:
                    is_done = stopping_criteria(
                        input_ids=generated[:, prompt_len:],
                        scores=None,
                    )
                    if torch.any(is_done):
                        done = True
                break

            aux_draft_len = min(draft_len, max(0, remaining - 2))
            drafted_aux_tokens: List[torch.LongTensor] = []

            for head_idx in range(aux_draft_len):
                aux_hidden = self.future_token_heads[head_idx](
                    current_anchor_hidden.to(self.future_token_heads[head_idx].weight.dtype)
                )
                aux_hidden = aux_hidden.to(norm.weight.dtype)
                aux_logits = lm_head(norm(aux_hidden))

                aux_context = (
                    torch.cat([generated, next_token], dim=-1)
                    if len(drafted_aux_tokens) == 0
                    else torch.cat(
                        [generated, next_token] + drafted_aux_tokens,
                        dim=-1,
                    )
                )
                aux_token = self._sample_next_token(
                    logits=aux_logits,
                    input_ids=aux_context,
                    logits_processor=logits_processor,
                    do_sample=do_sample,
                )
                drafted_aux_tokens.append(aux_token)

            proposal = (
                next_token
                if len(drafted_aux_tokens) == 0
                else torch.cat([next_token] + drafted_aux_tokens, dim=-1)
            )

            # IMPORTANT: verify on a cloned cache so rejected draft tokens do not
            # leak into the main cache through in-place DynamicCache mutation.
            base_cache_len = full_cache.get_seq_length()
            if len(full_cache) > 0:
                verify_cache = DynamicCache.from_legacy_cache(
                    [
                        (
                            full_cache.key_cache[i].clone(),
                            full_cache.value_cache[i].clone(),
                        )
                        for i in range(len(full_cache))
                    ]
                )
            else:
                verify_cache = DynamicCache()

            verify_output = self.backbone.model(
                input_ids=proposal,
                past_key_values=verify_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            verify_logits = verify_output.logits
            verify_hidden = verify_output.hidden_states[-1]
            verify_cache = verify_output.past_key_values

            n_accepted_aux = 0
            drafted_aux_seq = None
            if len(drafted_aux_tokens) > 0:
                drafted_aux_seq = torch.cat(drafted_aux_tokens, dim=-1)
                verify_aux_preds = verify_logits[:, : drafted_aux_seq.shape[1], :].argmax(dim=-1)
                matches = verify_aux_preds == drafted_aux_seq
                if matches.all():
                    n_accepted_aux = drafted_aux_seq.shape[1]
                else:
                    n_accepted_aux = int(matches.cumprod(dim=1).sum(dim=1).min().item())

                total_generated_tokens += drafted_aux_seq.shape[1]
                total_accepted_tokens += n_accepted_aux
                total_accepted_lengths.append(n_accepted_aux)
                accept_counts += 1
                for pos in range(drafted_aux_seq.shape[1]):
                    draft_position_attempt_counts[pos] += 1
                    if pos < n_accepted_aux:
                        draft_position_accept_counts[pos] += 1

            correction_context_parts = [generated, next_token]
            if drafted_aux_seq is not None and n_accepted_aux > 0:
                correction_context_parts.append(drafted_aux_seq[:, :n_accepted_aux])
            correction_context = torch.cat(correction_context_parts, dim=-1)
            correction_logits = verify_logits[:, n_accepted_aux, :]
            correction_token = self._sample_next_token(
                logits=correction_logits,
                input_ids=correction_context,
                logits_processor=logits_processor,
                do_sample=do_sample,
            )

            accepted_parts = [next_token]
            if drafted_aux_seq is not None and n_accepted_aux > 0:
                accepted_parts.append(drafted_aux_seq[:, :n_accepted_aux])
            accepted_chunk = torch.cat(accepted_parts, dim=-1)

            generated = torch.cat([generated, accepted_chunk], dim=-1)
            tokens_generated += accepted_chunk.shape[1]
            pbar.update(accepted_chunk.shape[1])

            # Keep cache up to accepted speculative prefix (without correction).
            prefix_cache_len = base_cache_len + 1 + n_accepted_aux
            full_cache = self._truncate_cache(verify_cache, prefix_cache_len)

            # Next round reuses verifier outputs: the sampled correction becomes
            # the carried next token, and the anchor hidden is taken from the
            # verifier at the accepted-prefix boundary.
            next_token = correction_token
            current_anchor_hidden = verify_hidden[:, n_accepted_aux, :]

            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=generated[:, prompt_len:],
                    scores=None,
                )
                if torch.any(is_done):
                    done = True

        pbar.close()

        self._last_draft_position_acceptance = {
            "attempt_counts": draft_position_attempt_counts,
            "accept_counts": draft_position_accept_counts,
            "acceptance_rates": [
                (acc / att) if att > 0 else 0.0
                for acc, att in zip(draft_position_accept_counts, draft_position_attempt_counts)
            ],
        }

        if tokenizer is not None:
            acceptance_rate = (
                total_accepted_tokens / total_generated_tokens
                if total_generated_tokens > 0
                else 0.0
            )
            avg_accepted_len = (
                sum(total_accepted_lengths) / accept_counts if accept_counts > 0 else 0.0
            )
            print(f"[MTP] Total drafted tokens: {total_generated_tokens}")
            print(f"[MTP] Total accepted tokens: {total_accepted_tokens}")
            print(f"[MTP] Acceptance rate: {acceptance_rate:.2%}")
            print(f"[MTP] Avg accepted length: {avg_accepted_len:.2f}")
            print(tokenizer.batch_decode(generated))

        return (
            generated,
            (total_generated_tokens, total_accepted_tokens),
            (total_accepted_lengths, accept_counts),
        )