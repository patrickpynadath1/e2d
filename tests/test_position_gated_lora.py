import torch

from src.backbone.position_gated_lora import PositionGatedLoRALinear


def _nonzero_branch(branch, value):
    torch.nn.init.constant_(branch.a.weight, value)
    torch.nn.init.constant_(branch.b.weight, value)


def test_position_gated_lora_routes_ar_and_flow_per_token():
    base = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(base.weight)
    layer = PositionGatedLoRALinear(
        base, shared_rank=1, ar_rank=1, flow_rank=1, alpha=1.0, dropout=0.0
    )
    _nonzero_branch(layer.shared, 1.0)
    _nonzero_branch(layer.ar, 2.0)
    _nonzero_branch(layer.flow, 3.0)
    inputs = torch.ones(1, 2, 2)
    layer.set_mode_mask(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))

    output = layer(inputs)

    # Shared gives 2 at both positions. AR gives 8; flow gives 18.
    assert torch.equal(output[0, 0], torch.full((2,), 10.0))
    assert torch.equal(output[0, 1], torch.full((2,), 20.0))


def test_position_gated_lora_only_backpropagates_through_selected_branch():
    base = torch.nn.Linear(2, 2, bias=False)
    layer = PositionGatedLoRALinear(
        base, shared_rank=1, ar_rank=1, flow_rank=1, alpha=1.0, dropout=0.0
    )
    _nonzero_branch(layer.shared, 1.0)
    _nonzero_branch(layer.ar, 1.0)
    _nonzero_branch(layer.flow, 1.0)
    layer.set_mode_mask(torch.tensor([[[1.0, 0.0]]]))

    layer(torch.ones(1, 1, 2)).sum().backward()

    assert layer.base.weight.grad is None
    assert layer.shared.b.weight.grad is not None
    assert layer.ar.b.weight.grad is not None
    assert layer.flow.b.weight.grad is not None
    assert torch.equal(layer.flow.b.weight.grad, torch.zeros_like(layer.flow.b.weight))


def test_position_gated_lora_rejects_incompatible_mask():
    base = torch.nn.Linear(2, 2)
    layer = PositionGatedLoRALinear(base, 1, 1, 1)
    layer.set_mode_mask(torch.ones(1, 3, 2))
    try:
        layer(torch.ones(1, 2, 2))
    except ValueError as error:
        assert "mode mask" in str(error)
    else:
        raise AssertionError("expected incompatible routing mask to fail")


def test_position_gated_lora_default_ar_activates_shared_and_ar_branches():
    base = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(base.weight)
    layer = PositionGatedLoRALinear(
        base,
        shared_rank=1,
        ar_rank=1,
        flow_rank=1,
        alpha=1.0,
        dropout=0.0,
        default_mode="ar",
    )
    _nonzero_branch(layer.shared, 1.0)
    _nonzero_branch(layer.ar, 2.0)
    _nonzero_branch(layer.flow, 3.0)

    output = layer(torch.ones(1, 3, 2))

    assert torch.equal(output, torch.full((1, 3, 2), 10.0))
