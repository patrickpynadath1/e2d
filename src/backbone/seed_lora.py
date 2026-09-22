"""SEED drafting adapters over an immutable, full-depth target model.

The decoder owns its module graph and LoRA parameters, but shares frozen base
parameters with the target. Adapters are never installed in, or merged into,
the target, including when its embedding and output weights are tied.
"""

import copy
import math

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.backbone.encoder_decoder import LLMasEncoderDecoderShareKV


class LoRALinear(nn.Module):
    """A draft-only low-rank update; ``base`` remains a frozen linear map."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # Keep optimizer state and adapter master weights in float32 under AMP.
        self.lora_A = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=torch.float32
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=torch.float32
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, inputs):
        base_output = self.base(inputs)
        update = nn.functional.linear(
            nn.functional.linear(
                self.dropout(inputs.to(self.lora_A.dtype)), self.lora_A
            ),
            self.lora_B,
        )
        return base_output + (update * self.scaling).to(base_output.dtype)


class LLMasEncoderDecoderFrozenTargetLoRA(LLMasEncoderDecoderShareKV):
    """Original model verifies; its last layers plus LM head draft via LoRA.

    Training returns only drafting logits. The full target forward runs without
    gradients and produces the exact causal KV representations used at inference.
    Existing SEED block attention selects a verified prefix from that cache.
    """

    frozen_target = True

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        attn_backend: str = "sdpa",
        num_decoder_layers: int = 2,
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        **llm_init_kwargs,
    ):
        nn.Module.__init__(self)
        if attn_backend != "sdpa":
            raise ValueError("Frozen-target SEED LoRA currently requires sdpa attention.")
        if lora_rank < 1 or not math.isfinite(lora_alpha) or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        self.encoder = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            attn_implementation=attn_backend,
            **llm_init_kwargs,
        )
        total_layers = len(self.encoder.model.layers)
        if not 1 <= num_decoder_layers <= total_layers:
            raise ValueError(f"num_decoder_layers must be between 1 and {total_layers}.")
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.max_length = max_length
        self.use_encoder_causal_mask = True
        # Existing generation uses this flag to truncate only drafting layers.
        self.tie_encoder_decoder_weights = True
        self.decoder_layer_idxs = list(range(total_layers - num_decoder_layers, total_layers))

        self.decoder = nn.Module()
        self.decoder.model = nn.Module()
        # Clone the module graph, sharing immutable parameters (not adapter modules).
        memo = {id(p): p for p in self.encoder.parameters()}
        self.decoder.model.layers = nn.ModuleList([
            copy.deepcopy(self.encoder.model.layers[i], memo)
            for i in self.decoder_layer_idxs
        ])
        self.decoder.model.norm = copy.deepcopy(self.encoder.model.norm, memo)
        self.decoder.model.rotary_emb = copy.deepcopy(self.encoder.model.rotary_emb, memo)
        for layer in self.decoder.model.layers:
            self._add_adapters(layer, lora_rank, lora_alpha, lora_dropout)
        # A separate wrapper is essential even when the target LM head is tied
        # to its input embedding: only drafting may use the low-rank update.
        self.decoder.lm_head = LoRALinear(
            self.encoder.lm_head, lora_rank, lora_alpha, lora_dropout
        )

    @classmethod
    def _add_adapters(cls, module, rank, alpha, dropout):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            else:
                cls._add_adapters(child, rank, alpha, dropout)

    def train(self, mode=True):
        super().train(mode)
        # In particular, target attention dropout must never affect the cache.
        self.encoder.eval()
        return self

    def unfreeze_encoder(self):
        raise RuntimeError("The verifier must remain frozen in SEED LoRA mode.")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        past_key_values=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_position_ids=None,
        encoder_cache_position=None,
        return_updated_cache=False,
        return_last_hidden_state=False,
        enforce_causal_mask=False,
        preserve_encoder_attention_mask=False,
        **kwargs,
    ):
        if encoder_input_ids is not None:
            if enforce_causal_mask and not preserve_encoder_attention_mask:
                encoder_attention_mask = None
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            with torch.no_grad():
                encoder_output = self.encoder.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                    cache_position=encoder_cache_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = encoder_output.past_key_values
            if return_updated_cache:
                return BaseModelOutputWithPast(
                    last_hidden_state=(
                        encoder_output.last_hidden_state if return_last_hidden_state else None
                    ),
                    past_key_values=past_key_values,
                )

        # Reuse the existing SEED decoder/cache implementation. The target cache
        # is detached; gradients flow only through the drafting adapter modules.
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            **kwargs,
        )
