"""Position-routed LoRA layers for joint autoregressive and flow training."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import nn


class _LoRABranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank if rank else 0.0
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(self.dropout(hidden_states))) * self.scaling


class PositionGatedLoRALinear(nn.Module):
    """Frozen linear layer with shared, AR-only, and flow-only LoRA branches.

    ``mode_mask[..., 0]`` routes the AR branch and ``mode_mask[..., 1]`` routes
    the flow branch. The shared branch is active wherever either mode is active.
    The routing tensor is deliberately non-persistent: it describes a forward,
    not model state, and must never enter a checkpoint.
    """

    def __init__(
        self,
        base: nn.Linear,
        shared_rank: int = 16,
        ar_rank: int = 16,
        flow_rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.shared = _LoRABranch(
            base.in_features, base.out_features, shared_rank, alpha, dropout
        )
        self.ar = _LoRABranch(
            base.in_features, base.out_features, ar_rank, alpha, dropout
        )
        self.flow = _LoRABranch(
            base.in_features, base.out_features, flow_rank, alpha, dropout
        )
        self.mode_mask: Optional[torch.Tensor] = None

    def set_mode_mask(self, mode_mask: Optional[torch.Tensor]) -> None:
        self.mode_mask = mode_mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden_states)
        if self.mode_mask is None:
            return output
        mask = self.mode_mask
        if mask.shape[:-1] != hidden_states.shape[:-1] or mask.shape[-1] != 2:
            raise ValueError(
                "LoRA mode mask must have shape hidden_states.shape[:-1] + (2,), "
                f"got {tuple(mask.shape)} for {tuple(hidden_states.shape)}"
            )
        mask = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        ar_gate = mask[..., 0:1]
        flow_gate = mask[..., 1:2]
        shared_gate = (ar_gate + flow_gate).clamp_max(1.0)
        return (
            output
            + shared_gate * self.shared(hidden_states)
            + ar_gate * self.ar(hidden_states)
            + flow_gate * self.flow(hidden_states)
        )


def inject_position_gated_lora(
    module: nn.Module,
    target_modules: Iterable[str],
    shared_rank: int = 16,
    ar_rank: int = 16,
    flow_rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace named linear projections recursively and return their full names."""
    targets = set(target_modules)
    replaced: list[str] = []
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name not in targets or not isinstance(child, nn.Linear):
                continue
            wrapped = PositionGatedLoRALinear(
                child,
                shared_rank=shared_rank,
                ar_rank=ar_rank,
                flow_rank=flow_rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(parent, child_name, wrapped)
            replaced.append(f"{parent_name}.{child_name}".lstrip("."))
    if not replaced:
        raise ValueError(f"no linear modules matched targets {sorted(targets)}")
    return replaced


def set_position_lora_mode(
    module: nn.Module, mode_mask: Optional[torch.Tensor]
) -> None:
    for child in module.modules():
        if isinstance(child, PositionGatedLoRALinear):
            child.set_mode_mask(mode_mask)
