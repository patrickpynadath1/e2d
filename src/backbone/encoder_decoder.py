from dataclasses import dataclass
from functools import partial
import os
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

from src.backbone.custom_modeling_qwen3 import CustomQwen3ForCausalLM

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


logger = logging.get_logger(__name__)


def _unwrap_fsdp_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_fsdp_wrapped_module", module)


def _get_layer_idx(module: nn.Module) -> int:
    return _unwrap_fsdp_module(module).self_attn.layer_idx


@dataclass
class EncoderBaseModelOutputWithPast(ModelOutput):
    """Custom (encoder) model output.
    Stores previous decoder and updated encoder cache and encoder last hidden state.
    """

    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]] = (
        None
    )
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_past_key_values: Optional[
        Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]
    ] = None


@dataclass
class DecoderCausalLMOutputWithPast(ModelOutput):
    """Custom (decoder) model output.
    Stores previous encoder and updated decoder cache and decoder logits.
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]] = (
        None
    )
    encoder_past_key_values: Optional[
        Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]
    ] = None


class LLMasEncoderDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_encoder: bool = False,
        reinit_decoder: bool = False,
        tie_encoder_decoder_weights: bool = False,
        use_encoder_causal_mask: bool = False,
        num_encoder_layers: int = -1,
        num_decoder_layers: int = -1,
        keep_top_encoder_layers: bool = False,
        keep_top_decoder_layers: bool = False,
        use_gradient_checkpointing: bool = False,
        **llm_init_kwargs,
    ):
        assert not (tie_encoder_decoder_weights and reinit_decoder), (
            "Cannot tie encoder-decoder weights and reinitialize decoder."
        )
        assert not (tie_encoder_decoder_weights and freeze_encoder), (
            "Cannot freeze encoder weights when tying encoder-decoder weights."
        )
        super().__init__()
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights

        if reinit_encoder:
            assert num_encoder_layers > 0
            encoder_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                num_hidden_layers=num_encoder_layers,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            self.encoder = CustomQwen3ForCausalLM(encoder_config)
        else:
            self.encoder = CustomQwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            assert num_encoder_layers <= len(self.encoder.model.layers), (
                f"Cannot keep {num_encoder_layers} layers. "
                f"Pre-trained model only has {len(self.encoder.model.layers)} layers."
            )
            num_encoder_layers = (
                len(self.encoder.model.layers)
                if num_encoder_layers == -1
                else num_encoder_layers
            )
            if keep_top_encoder_layers:
                self.encoder.model.layers = self.encoder.model.layers[
                    -num_encoder_layers:
                ]
            else:
                self.encoder.model.layers = self.encoder.model.layers[
                    :num_encoder_layers
                ]

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "embed_tokens" not in name:
                    param.requires_grad = False
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if tie_encoder_decoder_weights:
            self.decoder = self.encoder
            num_decoder_layers = (
                len(self.decoder.model.layers)
                if num_decoder_layers == -1
                else num_decoder_layers
            )
            assert num_decoder_layers <= len(self.decoder.model.layers), (
                f"Cannot keep {num_decoder_layers} layers. "
                f"Pre-trained model only has {len(self.decoder.model.layers)} layers."
            )
            # Keep **top** layers when tying weights
            self.decoder_layer_idxs = list(range(len(self.encoder.model.layers)))[
                -num_decoder_layers:
            ]

        else:
            if reinit_decoder:
                assert num_decoder_layers > 0
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    num_hidden_layers=num_decoder_layers,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                self.decoder = CustomQwen3ForCausalLM(decoder_config)
            else:
                self.decoder = CustomQwen3ForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                assert num_decoder_layers <= len(self.decoder.model.layers), (
                    f"Cannot keep {num_decoder_layers} layers. "
                    f"Pre-trained model only has {len(self.decoder.layers)} layers."
                )
                if keep_top_decoder_layers:
                    self.decoder.model.layers = self.decoder.model.layers[
                        -num_decoder_layers:
                    ]
                else:
                    self.decoder.model.layers = self.decoder.model.layers[
                        :num_decoder_layers
                    ]
            del self.decoder.model.embed_tokens
            # if in the original LM, the lm_head is weight-tied to embedding,
            # point decoder lm_head to encoder's (instead of initializing separately)
            if (
                self.encoder.lm_head.weight.data_ptr()
                == self.encoder.model.embed_tokens.weight.data_ptr()
            ):
                self.decoder.lm_head = self.encoder.lm_head
            else:
                del self.encoder.lm_head
            if use_gradient_checkpointing:
                self.decoder.gradient_checkpointing_enable()
        self.max_length = max_length

    def freeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = True

    # noinspection PyUnusedLocal
    def forward(
        self,
        # Decoder inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,
        # Encoder inputs
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_past_key_values: Optional[DynamicCache] = None,
        # Additional args
        fix_cache_length: bool = True,  # Not used; compatibility with other backbones
        truncate_cache: bool = True,
        return_updated_cache: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[DecoderCausalLMOutputWithPast, EncoderBaseModelOutputWithPast]:
        # During training/eval encoder_last_hidden_state is None.
        # During generation encoder_last_hidden_state can be not None.
        new_seen_tokens = (
            0
            if encoder_last_hidden_state is None
            else encoder_last_hidden_state.shape[1]
        )
        # Encode clean tokens
        if encoder_input_ids is not None:
            if self.use_encoder_causal_mask:
                encoder_attention_mask = None  # None --> enforces use of causal mask
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            encoder_output = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                use_cache=True,
                past_key_values=encoder_past_key_values,
                cache_position=encoder_cache_position,
            )
            if return_updated_cache:
                # encoder_output.past_key_values now contains latest encoder input
                return EncoderBaseModelOutputWithPast(
                    encoder_last_hidden_state=encoder_output.last_hidden_state,
                    encoder_past_key_values=encoder_output.past_key_values,
                    past_key_values=past_key_values,
                )
            encoder_last_hidden_state = encoder_output.last_hidden_state

        # Run decoder with xattn to clean token hidden states
        if encoder_last_hidden_state is None:  # No new encoder tokens
            q_start_idx = 0
            decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
            if cache_position is None:
                if position_ids is not None:
                    cache_position = position_ids[0]
                else:
                    past_seen_tokens = (
                        past_key_values.get_seq_length()
                        if past_key_values is not None
                        else 0
                    )
                    cache_position = torch.arange(
                        past_seen_tokens,
                        past_seen_tokens + decoder_hidden_states.shape[1],
                        device=decoder_hidden_states.device,
                    )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            decoder_position_embeddings = self.decoder.model.rotary_emb(
                decoder_hidden_states, position_ids
            )
        else:
            q_start_idx = encoder_last_hidden_state.shape[1]
            decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
            decoder_hidden_states = torch.cat(
                [
                    encoder_last_hidden_state,
                    decoder_hidden_states,
                ],
                dim=1,
            )
            if cache_position is None:
                if position_ids is not None:
                    cache_position = position_ids[0]
                else:
                    past_seen_tokens = (
                        past_key_values.get_seq_length()
                        if past_key_values is not None
                        else 0
                    )
                    cache_position = torch.cat(
                        [
                            torch.arange(  # clean token position ids
                                past_seen_tokens,
                                past_seen_tokens + encoder_last_hidden_state.shape[1],
                                device=decoder_hidden_states.device,
                            ),
                            torch.arange(  # noisy position ids
                                past_seen_tokens + new_seen_tokens,
                                past_seen_tokens + new_seen_tokens + input_ids.shape[1],
                                device=decoder_hidden_states.device,
                            ),
                        ],
                        dim=-1,
                    )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            decoder_position_embeddings = self.decoder.model.rotary_emb(
                decoder_hidden_states, position_ids
            )

        if hasattr(self.decoder.model, "_update_causal_mask"):  # bc on transformers
            # noinspection PyProtectedMember
            attention_mask = self.decoder.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=decoder_hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=False,
            )
        for decoder_layer in self.decoder.model.layers:
            layer_idx = _get_layer_idx(decoder_layer)
            if (
                self.tie_encoder_decoder_weights
                and layer_idx not in self.decoder_layer_idxs
            ):
                continue
            # past_key_values gets updated in-place.
            # Record previous length to re-truncate after each layer forward
            if past_key_values is not None and len(past_key_values) > layer_idx:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0
            cache_len = prev_cache_len + new_seen_tokens

            if self.decoder.model.gradient_checkpointing and self.training:
                # noinspection PyProtectedMember
                decoder_hidden_states = self.decoder._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    decoder_hidden_states,  # hidden_states=,
                    attention_mask,  # attention_mask=,
                    position_ids,  # position_ids=,
                    past_key_values,  # past_key_value=,
                    False,  # output_attentions=,
                    True,  # use_cache=,
                    cache_position,  # cache_position=,
                    decoder_position_embeddings,  # position_embeddings=,
                    q_start_idx,  # q_start_idx=
                )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)
            else:
                decoder_hidden_states = decoder_layer(
                    hidden_states=decoder_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=decoder_position_embeddings,
                    q_start_idx=q_start_idx,  # Indicates where to slice output
                    **flash_attn_kwargs,
                )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)
            # Update decoder_hidden_states
            if q_start_idx > 0:
                decoder_hidden_states = torch.cat(
                    [
                        encoder_last_hidden_state,
                        decoder_hidden_states,
                    ],
                    dim=1,
                )
            if truncate_cache and past_key_values is not None:
                # DynamicCache extends along sequence dimension by default;
                # truncate back to original cache len + encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :cache_len, :]
        decoder_hidden_states = self.decoder.model.norm(
            decoder_hidden_states[:, q_start_idx:, :]
        )
        logits = self.decoder.lm_head(decoder_hidden_states)
        return DecoderCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            encoder_past_key_values=encoder_past_key_values,
            # Do not need to store encoder_last_hidden_state.
            # If it was passed in, then it has become part of the past_key_values cache.
        )


class LLMasEncoderDecoderShareKV(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_encoder: bool = False,
        reinit_decoder: bool = False,
        tie_encoder_decoder_weights: bool = False,
        use_encoder_causal_mask: bool = False,
        num_encoder_layers: int = -1,
        num_decoder_layers: int = -1,
        keep_top_encoder_layers: bool = False,
        keep_top_decoder_layers: bool = False,
        use_gradient_checkpointing: bool = False,
        train_on_ar: bool = False,
        ar_checkpoint_path: Optional[str] = None,
        **llm_init_kwargs,
    ):
        assert not (tie_encoder_decoder_weights and reinit_decoder), (
            "Cannot tie encoder-decoder weights and reinitialize decoder."
        )
        assert not (tie_encoder_decoder_weights and freeze_encoder), (
            "Cannot freeze encoder weights when tying encoder-decoder weights."
        )
        assert not (train_on_ar and (reinit_encoder or reinit_decoder)), (
            "Cannot reinitialize encoder/decoder when train_on_ar is enabled."
        )
        super().__init__()
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights

        use_ar_checkpoint = train_on_ar
        ar_state_dict = None
        if use_ar_checkpoint:
            if ar_checkpoint_path is None:
                raise ValueError(
                    "train_on_ar is enabled but ar_checkpoint_path is not set."
                )
            if not os.path.isfile(ar_checkpoint_path):
                raise FileNotFoundError(
                    f"AR checkpoint does not exist: {ar_checkpoint_path}"
                )
            logger.info(f"Loading AR checkpoint from {ar_checkpoint_path}")
            try:
                checkpoint = torch.load(
                    ar_checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(ar_checkpoint_path, map_location="cpu")
            ar_state_dict = self._extract_checkpoint_state_dict(checkpoint)

        if reinit_encoder:
            assert num_encoder_layers > 0
            encoder_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                num_hidden_layers=num_encoder_layers,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            self.encoder = AutoModelForCausalLM.from_config(encoder_config)
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            if use_ar_checkpoint:
                self._load_ar_weights_into_module(
                    module=self.encoder,
                    full_state_dict=ar_state_dict,
                    module_name="encoder",
                    preferred_prefixes=(
                        "model.backbone.encoder.",
                        "backbone.encoder.",
                        "encoder.",
                        "model.backbone.model.",
                        "backbone.model.",
                    ),
                )
            assert num_encoder_layers <= len(self.encoder.model.layers), (
                f"Cannot keep {num_encoder_layers} layers. "
                f"Pre-trained model only has {len(self.encoder.model.layers)} layers."
            )
            num_encoder_layers = (
                len(self.encoder.model.layers)
                if num_encoder_layers == -1
                else num_encoder_layers
            )
            if keep_top_encoder_layers:
                self.encoder.model.layers = self.encoder.model.layers[
                    -num_encoder_layers:
                ]
            else:
                self.encoder.model.layers = self.encoder.model.layers[
                    :num_encoder_layers
                ]

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "embed_tokens" not in name:
                    param.requires_grad = False
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if tie_encoder_decoder_weights:
            self.decoder = self.encoder
            num_decoder_layers = (
                len(self.decoder.model.layers)
                if num_decoder_layers == -1
                else num_decoder_layers
            )
            assert num_decoder_layers <= len(self.decoder.model.layers), (
                f"Cannot keep {num_decoder_layers} layers. "
                f"Pre-trained model only has {len(self.decoder.model.layers)} layers."
            )
            # Keep **top** layers when tying weights
            self.decoder_layer_idxs = list(range(len(self.encoder.model.layers)))[
                -num_decoder_layers:
            ]

        else:
            if reinit_decoder:
                assert num_decoder_layers > 0
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    num_hidden_layers=num_decoder_layers,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                self.decoder = AutoModelForCausalLM.from_config(decoder_config)
            else:
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                if use_ar_checkpoint:
                    self._load_ar_weights_into_module(
                        module=self.decoder,
                        full_state_dict=ar_state_dict,
                        module_name="decoder",
                        preferred_prefixes=(
                            "model.backbone.decoder.",
                            "backbone.decoder.",
                            "decoder.",
                            "model.backbone.model.",
                            "backbone.model.",
                        ),
                    )
                assert num_decoder_layers <= len(self.decoder.model.layers), (
                    f"Cannot keep {num_decoder_layers} layers. "
                    f"Pre-trained model only has {len(self.decoder.layers)} layers."
                )
                if keep_top_decoder_layers:
                    self.decoder.model.layers = self.decoder.model.layers[
                        -num_decoder_layers:
                    ]
                else:
                    self.decoder.model.layers = self.decoder.model.layers[
                        :num_decoder_layers
                    ]
            del self.decoder.model.embed_tokens
            # Even for frozen encoder, ensure embedding tokens are trainable
            self.encoder.model.embed_tokens.requires_grad_(True)
            unused_self_attn_params = ["o_proj", "q_norm", "q_proj"]
            unused_layernorm_params = ["input_layernorm", "post_attention_layernorm"]
            for unused_param in unused_self_attn_params:
                if hasattr(self.encoder.model.layers[-1].self_attn, unused_param):
                    getattr(
                        self.encoder.model.layers[-1].self_attn, unused_param
                    ).requires_grad_(False)
            self.encoder.model.layers[-1].mlp.requires_grad_(False)
            self.encoder.model.norm.requires_grad_(False)
            for unused_param in unused_layernorm_params:
                if hasattr(self.encoder.model.layers[-1], unused_param):
                    getattr(self.encoder.model.layers[-1], unused_param).requires_grad_(
                        False
                    )
            # if in the original LM, the lm_head is weight-tied to embedding,
            # point decoder lm_head to encoder's (instead of initializing separately)
            if (
                self.encoder.lm_head.weight.data_ptr()
                == self.encoder.model.embed_tokens.weight.data_ptr()
            ):
                self.decoder.lm_head = self.encoder.lm_head
            else:
                del self.encoder.lm_head
            if use_gradient_checkpointing:
                self.decoder.gradient_checkpointing_enable()
        self.max_length = max_length

    @staticmethod
    def _extract_checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected AR checkpoint to be a dictionary.")

        candidate = checkpoint
        if (
            "state" in checkpoint
            and isinstance(checkpoint["state"], dict)
            and "model" in checkpoint["state"]
            and isinstance(checkpoint["state"]["model"], dict)
        ):
            candidate = checkpoint["state"]["model"]
        else:
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    candidate = checkpoint[key]
                    break

        tensor_state = {
            key: value
            for key, value in candidate.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        if not tensor_state:
            raise ValueError(
                "Could not find tensor weights in AR checkpoint. "
                "Expected a state dict or nested {'state': {'model': ...}} format."
            )
        return tensor_state

    def _load_ar_weights_into_module(
        self,
        module: nn.Module,
        full_state_dict: Optional[dict[str, torch.Tensor]],
        module_name: str,
        preferred_prefixes: tuple[str, ...],
    ) -> None:
        if full_state_dict is None:
            raise ValueError("AR checkpoint state dict is missing.")

        target_keys = set(module.state_dict().keys())
        matched_state_dict: dict[str, torch.Tensor] = {}

        for raw_key, tensor in full_state_dict.items():
            key = raw_key[7:] if raw_key.startswith("module.") else raw_key
            candidates = [key]

            for prefix in preferred_prefixes:
                if key.startswith(prefix):
                    candidates.append(key[len(prefix) :])

            if key.startswith("model."):
                candidates.append(key[len("model.") :])
            if key.startswith("backbone."):
                candidates.append(key[len("backbone.") :])

            for candidate_key in candidates:
                if candidate_key in target_keys:
                    matched_state_dict[candidate_key] = tensor
                    break

        if not matched_state_dict:
            sample_keys = list(full_state_dict.keys())[:8]
            raise ValueError(
                f"Failed to map AR checkpoint keys for {module_name}. "
                f"Sample keys from checkpoint: {sample_keys}"
            )

        incompatible_keys = module.load_state_dict(matched_state_dict, strict=False)
        if incompatible_keys.missing_keys:
            logger.warning(
                f"Missing {len(incompatible_keys.missing_keys)} {module_name} keys "
                f"when loading AR checkpoint. "
                f"First few: {incompatible_keys.missing_keys[:8]}"
            )
        if incompatible_keys.unexpected_keys:
            logger.warning(
                f"Unexpected {len(incompatible_keys.unexpected_keys)} {module_name} keys "
                f"when loading AR checkpoint. "
                f"First few: {incompatible_keys.unexpected_keys[:8]}"
            )
        logger.info(
            f"Loaded {len(matched_state_dict)} parameters into {module_name} "
            "from AR checkpoint."
        )

    def freeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = True

    # noinspection PyUnusedLocal
    def forward(
        self,
        # Decoder inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,  # Not used
        # Encoder inputs
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_past_key_values: Optional[DynamicCache] = None,  # Not used
        # Additional args
        fix_cache_length: bool = True,  # Not used; compatibility with other backbones
        truncate_cache: bool = True,
        enforce_causal_mask: bool = False,
        return_last_hidden_state: bool = False,
        return_updated_cache: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[CausalLMOutputWithPast, BaseModelOutputWithPast]:
        # Encode clean tokens
        if encoder_input_ids is not None:
            if self.use_encoder_causal_mask or enforce_causal_mask:
                encoder_attention_mask = None  # None --> enforces use of causal mask
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            if return_last_hidden_state:
                encoder_output = self.encoder.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    cache_position=encoder_cache_position,
                )
                past_key_values = encoder_output.past_key_values
                if return_updated_cache:
                    # encoder_output.past_key_values now contains latest encoder input
                    return BaseModelOutputWithPast(
                        last_hidden_state=encoder_output.last_hidden_state,
                        past_key_values=past_key_values,
                    )
            else:
                past_key_values = self.encoder.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    cache_position=encoder_cache_position,
                ).past_key_values
                if return_updated_cache:
                    # encoder_output.past_key_values now contains latest encoder input
                    return BaseModelOutputWithPast(
                        past_key_values=past_key_values,
                    )

        # Run decoder with xattn to clean token hidden states
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
        if cache_position is None:
            if position_ids is not None:
                cache_position = position_ids[0]
            else:  # During training / validation position_ids are not provided
                cache_position = torch.arange(
                    decoder_hidden_states.shape[1],
                    device=decoder_hidden_states.device,
                )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )

        if hasattr(self.decoder.model, "_update_causal_mask"):  # bc on transformers
            # noinspection PyProtectedMember
            attention_mask = self.decoder.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=decoder_hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=False,
            )
        for decoder_layer in self.decoder.model.layers:
            layer_idx = _get_layer_idx(decoder_layer)
            if (
                self.tie_encoder_decoder_weights
                and layer_idx not in self.decoder_layer_idxs
            ):
                continue
            # past_key_values gets updated in-place.
            # Record previous length to truncate after each layer forward
            if past_key_values is not None and len(past_key_values) > layer_idx:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0

            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=position_ids[0],
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)

            if truncate_cache and past_key_values is not None:
                # DynamicCache extends along sequence dimension by default;
                # truncate back to original cache len + encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        logits = self.decoder.lm_head(decoder_hidden_states)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )


class LLMasEncoderDecoderShareKVEncoderGen(LLMasEncoderDecoderShareKV):
    """
    Same as LLMasEncoderDecoderShareKV, but also computes and returns encoder logits
    concatenated with decoder logits, enabling joint training of encoder (next-token prediction)
    and decoder (denoising/generation) so that during inference, we could use the larger encoder
    as verifier and perform speculative decoding.
    """
    # noinspection PyUnusedLocal
    def forward(
        self,
        # Decoder inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,  # Not used
        # Encoder inputs
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_past_key_values: Optional[DynamicCache] = None,  # Not used
        # Additional args
        fix_cache_length: bool = True,  # Not used; compatibility with other backbones
        truncate_cache: bool = True,
        enforce_causal_mask: bool = False,
        return_last_hidden_state: bool = False,
        return_updated_cache: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[CausalLMOutputWithPast, BaseModelOutputWithPast]:
        
        encoder_logits = None

        # Encode clean tokens
        if encoder_input_ids is not None:
            if self.use_encoder_causal_mask or enforce_causal_mask:
                encoder_attention_mask = None  # None --> enforces use of causal mask
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            
            # Always compute last_hidden_state to get logits
            encoder_output = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=encoder_cache_position,
            )
            past_key_values = encoder_output.past_key_values
            
            # Compute encoder logits
            encoder_logits = self.encoder.lm_head(encoder_output.last_hidden_state)

            if return_last_hidden_state:
                if return_updated_cache:
                    # encoder_output.past_key_values now contains latest encoder input
                    return BaseModelOutputWithPast(
                        last_hidden_state=encoder_output.last_hidden_state,
                        past_key_values=past_key_values,
                    )
            
            if return_updated_cache:
                # encoder_output.past_key_values now contains latest encoder input
                return BaseModelOutputWithPast(
                    past_key_values=past_key_values,
                )

        # Run decoder with xattn to clean token hidden states
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
        if cache_position is None:
            if position_ids is not None:
                cache_position = position_ids[0]
            else:  # During training / validation position_ids are not provided
                cache_position = torch.arange(
                    decoder_hidden_states.shape[1],
                    device=decoder_hidden_states.device,
                )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )

        if hasattr(self.decoder.model, "_update_causal_mask"):  # bc on transformers
            # noinspection PyProtectedMember
            attention_mask = self.decoder.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=decoder_hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=False,
            )
        for decoder_layer in self.decoder.model.layers:
            layer_idx = _get_layer_idx(decoder_layer)
            if (
                self.tie_encoder_decoder_weights
                and layer_idx not in self.decoder_layer_idxs
            ):
                continue
            # past_key_values gets updated in-place.
            # Record previous length to truncate after each layer forward
            if past_key_values is not None and len(past_key_values) > layer_idx:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0

            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=position_ids[0],
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)

            if truncate_cache and past_key_values is not None:
                # DynamicCache extends along sequence dimension by default;
                # truncate back to original cache len + encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        logits = self.decoder.lm_head(decoder_hidden_states)
        
        if encoder_logits is not None:
            logits = torch.cat([encoder_logits, logits], dim=1)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

''' -------- Things We Tried But Failed --------
# Keep the entire model frozen, only train a lightweight adapter module
class LLMasEncoderDecoderShareKVAdapter(LLMasEncoderDecoderShareKV):
    def __init__(
        self,
        *args, 
        **kwargs
    ):  
        # Initialize Parent
        super().__init__(*args, **kwargs)

        # MANUAL FREEZE:
        # Since we passed freeze_encoder=False, the model is currently trainable.
        # We iterate over every parameter in the loaded LLM and freeze it.
        for param in self.parameters():
            param.requires_grad = False
            
        # Double check: explicitly ensure embed_tokens is frozen 
        # (The parent class has complex logic regarding this layer, so we overwrite it)
        if hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'embed_tokens'):
             self.encoder.model.embed_tokens.requires_grad_(False)
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'embed_tokens'):
             self.decoder.model.embed_tokens.requires_grad_(False)

        # Define the Adapter
        # Get model dimension
        model_dim = self.encoder.config.hidden_size
        inter_dim = self.encoder.config.intermediate_size

        # 2-layer MLP: Linear -> Activation -> Linear
        self.adapter = nn.Sequential(
            nn.Linear(model_dim, inter_dim),
            nn.GELU(),
            nn.Linear(inter_dim, model_dim) 
        )

        # 5. Enable Gradients for Adapter ONLY
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        # Optional: Initialize adapter weights
        self._init_adapter_weights()

    def _init_adapter_weights(self):
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    # noinspection PyUnusedLocal
    def forward(
        self,
        # Decoder inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,  # Not used
        # Encoder inputs
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_past_key_values: Optional[DynamicCache] = None,  # Not used
        # Additional args
        fix_cache_length: bool = True,  # Not used; compatibility with other backbones
        truncate_cache: bool = True,
        enforce_causal_mask: bool = False,
        return_last_hidden_state: bool = False,
        return_updated_cache: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[CausalLMOutputWithPast, BaseModelOutputWithPast]:
        # Encode clean tokens
        if encoder_input_ids is not None:
            if self.use_encoder_causal_mask or enforce_causal_mask:
                encoder_attention_mask = None  # None --> enforces use of causal mask
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            if return_last_hidden_state:
                encoder_output = self.encoder.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    cache_position=encoder_cache_position,
                )
                past_key_values = encoder_output.past_key_values
                if return_updated_cache:
                    # encoder_output.past_key_values now contains latest encoder input
                    return BaseModelOutputWithPast(
                        last_hidden_state=encoder_output.last_hidden_state,
                        past_key_values=past_key_values,
                    )
            else:
                past_key_values = self.encoder.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    cache_position=encoder_cache_position,
                ).past_key_values
                if return_updated_cache:
                    # encoder_output.past_key_values now contains latest encoder input
                    return BaseModelOutputWithPast(
                        past_key_values=past_key_values,
                    )

        # Run decoder with xattn to clean token hidden states
        # First run the adapter to get adapted (contextualized) embeddings
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
        decoder_hidden_states = self.adapter(decoder_hidden_states)

        if cache_position is None:
            if position_ids is not None:
                cache_position = position_ids[0]
            else:  # During training / validation position_ids are not provided
                cache_position = torch.arange(
                    decoder_hidden_states.shape[1],
                    device=decoder_hidden_states.device,
                )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )

        if hasattr(self.decoder.model, "_update_causal_mask"):  # bc on transformers
            # noinspection PyProtectedMember
            attention_mask = self.decoder.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=decoder_hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=False,
            )
        for decoder_layer in self.decoder.model.layers:
            layer_idx = _get_layer_idx(decoder_layer)
            if (
                self.tie_encoder_decoder_weights
                and layer_idx not in self.decoder_layer_idxs
            ):
                continue
            # past_key_values gets updated in-place.
            # Record previous length to truncate after each layer forward
            if past_key_values is not None and len(past_key_values) > layer_idx:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0

            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=position_ids[0],
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)

            if truncate_cache and past_key_values is not None:
                # DynamicCache extends along sequence dimension by default;
                # truncate back to original cache len + encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        logits = self.decoder.lm_head(decoder_hidden_states)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )
'''

# Keep the bottom encoder layers frozen, only train top layers (decoder)
class LLMasEncoderDecoderShareKVEncoderGenFrozenBottom(LLMasEncoderDecoderShareKVEncoderGen):
    def __init__(self, *args, **kwargs):
        # 1. Initialize the model using the parent class logic
        # This sets up self.encoder, self.decoder, and self.decoder_layer_idxs
        super().__init__(*args, **kwargs)

        # We want to freeze: 
        #   - Embeddings (usually considered bottom)
        #   - Layers that are NOT in the decoder_layer_idxs list
        # We want to keep trainable:
        #   - Layers that ARE in the decoder_layer_idxs list
        #   - The Final LayerNorm (self.encoder.model.norm)
        #   - The LM Head (self.encoder.lm_head)

        # A. Freeze Embeddings
        if hasattr(self.encoder.model, "embed_tokens"):
            self.encoder.model.embed_tokens.requires_grad_(False)

        # B. Freeze Bottom Layers (Encoder-only layers)
        # Convert list to set for O(1) lookups
        decoder_layer_indices = set(self.decoder_layer_idxs)

        for i, layer in enumerate(self.encoder.model.layers):
            if i not in decoder_layer_indices:
                # This layer is used ONLY by the encoder, so we freeze it
                layer.requires_grad_(False)
            else:
                # This layer is used by the decoder (and encoder), keep it trainable.
                # (Explicitly setting True is safer in case default changed elsewhere)
                layer.requires_grad_(True)

        # C. Ensure Head and Norm are trainable (required for Decoder generation)
        if hasattr(self.encoder.model, "norm"):
            self.encoder.model.norm.requires_grad_(True)
            
        if hasattr(self.encoder, "lm_head"):
            self.encoder.lm_head.requires_grad_(True)

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not installed. LoRA functionality will fail if initialized.")

class LLMasEncoderDecoderShareKVEncoderGenLoRA(LLMasEncoderDecoderShareKVEncoderGen):
    """
    Same as LLMasEncoderDecoderShareKVEncoderGen, but wraps the underlying 
    encoder (and decoder) with LoRA adapters for parameter-efficient fine-tuning.
    """
    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        *args, 
        **kwargs
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("Please install 'peft' to use LLMasEncoderDecoderShareKVEncoderGenLoRA.")

        # 1. Initialize the base model (loads weights, sets up encoder/decoder refs)
        super().__init__(*args, **kwargs)

        # 2. Define LoRA Config
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )

        # 3. Apply LoRA to the Encoder
        # Capture the original backbone before wrapping (e.g. LlamaModel)
        # We need this because PeftModel's attribute delegation makes .model point to the CausalLM,
        # breaking the parent class's forward method.
        original_backbone = self.encoder.model

        # Wrap encoder
        self.encoder = get_peft_model(self.encoder, peft_config)

        # FIX: Manually set .model on the PeftModel wrapper to point to the backbone.
        # We use object.__setattr__ to avoid registering it as a PyTorch submodule (which would duplicate params).
        object.__setattr__(self.encoder, "model", original_backbone)

        # 4. Handle the Decoder
        self.decoder = self.encoder

        # 5. Logging
        logger.info("LoRA initialized. Trainable parameters:")
        self.encoder.print_trainable_parameters()

    def save_pretrained(self, save_directory, **kwargs):
        """
        Custom save method to ensure we save the PeftModel adapters correctly.
        """
        # Because self.encoder is a PeftModel, calling save_pretrained on it 
        # will save the `adapter_model.bin` and `adapter_config.json`.
        self.encoder.save_pretrained(save_directory, **kwargs)
        
        if not self.tie_encoder_decoder_weights:
            logger.warning("Encoder and Decoder are not tied. Saving only Encoder adapters to root.")
