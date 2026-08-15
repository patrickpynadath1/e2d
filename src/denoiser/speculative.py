"""Shared, model-independent pieces of speculative dual decoding."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import torch


class DecoderMode(StrEnum):
    AR = "ar_decoder"
    MASKED = "masked_diffusion"
    UNIFORM = "uniform_diffusion"
    MASKED_AR = "masked_diffusion_ar_head"
    UNIFORM_AR = "uniform_diffusion_ar_head"
    LATENT_FLOW = "latent_flow"


@dataclass
class SpeculativeStats:
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    correction_tokens: int = 0
    committed_tokens: int = 0
    draft_calls: int = 0
    verifier_calls: int = 0
    accepted_lengths: list[int] = field(default_factory=list)
    draft_seconds: float = 0.0
    verifier_seconds: float = 0.0
    cache_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if not self.proposed_tokens:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @property
    def average_accepted_length(self) -> float:
        return (
            sum(self.accepted_lengths) / len(self.accepted_lengths)
            if self.accepted_lengths
            else 0.0
        )

    @property
    def tokens_per_second(self) -> float:
        return self.committed_tokens / self.total_seconds if self.total_seconds else 0.0

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            acceptance_rate=self.acceptance_rate,
            average_accepted_length=self.average_accepted_length,
            tokens_per_second=self.tokens_per_second,
        )
        return result


def longest_matching_prefix(
    proposal: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return the accepted prefix length for each batch element."""
    if proposal.shape != target.shape or proposal.ndim != 2:
        raise ValueError("proposal and target must have equal [batch, length] shapes")
    return (proposal == target).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def corrupt_discrete_tokens(
    clean: torch.Tensor,
    corruption_probability: torch.Tensor,
    *,
    mode: DecoderMode,
    vocab_size: int,
    mask_token_id: int | None = None,
    valid_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply masked or uniform token corruption and return data plus corruption mask."""
    if mode not in {DecoderMode.MASKED, DecoderMode.UNIFORM}:
        raise ValueError(f"unsupported discrete corruption mode: {mode}")
    if not torch.is_floating_point(corruption_probability):
        raise TypeError("corruption_probability must be floating point")
    corrupt = torch.rand(
        clean.shape, device=clean.device, generator=generator
    ) < corruption_probability
    if valid_mask is not None:
        corrupt &= valid_mask.bool()
    if mode == DecoderMode.MASKED:
        if mask_token_id is None:
            raise ValueError("mask_token_id is required for masked diffusion")
        replacement = torch.full_like(clean, mask_token_id)
    else:
        replacement = torch.randint(
            vocab_size, clean.shape, device=clean.device, generator=generator
        )
    return torch.where(corrupt, replacement, clean), corrupt


def linear_flow_sample(
    clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear noise interpolation and its constant flow-matching velocity."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    while time.ndim < clean.ndim:
        time = time.unsqueeze(-1)
    noisy = (1.0 - time) * clean + time * noise
    return noisy, noise - clean
