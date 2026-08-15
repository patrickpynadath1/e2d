from types import SimpleNamespace

import pytest
import torch

from src.denoiser.latent_flow import LatentFlowE2D


def _bare_model(fixed: bool) -> LatentFlowE2D:
    model = LatentFlowE2D.__new__(LatentFlowE2D)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        fixed_training_noise=fixed,
        fixed_training_seed=17,
        block_size=4,
    )
    return model


def test_latent_flow_uses_two_layers_before_final_qwen_layer():
    assert LatentFlowE2D._resolve_layer_index(-3, 28) == 25
    assert LatentFlowE2D._resolve_layer_index(-2, 28) == 26
    assert LatentFlowE2D._resolve_layer_index(-1, 28) == 27


def test_invalid_layer_offset_fails_clearly():
    with pytest.raises(ValueError, match="invalid"):
        LatentFlowE2D._resolve_layer_index(-29, 28)


def test_training_flow_mask_uses_clean_history_and_noisy_current_block():
    mask = LatentFlowE2D._flow_attention_mask(4, 2, torch.device("cpu"))
    # Noisy position 2 attends clean block 0 and its noisy block 1 canvas.
    assert mask[6].tolist() == [True, True, False, False, False, False, True, True]
    # It cannot attend the noisy copy of a prior block.
    assert not mask[6, 4:6].any()


def test_inference_flow_mask_has_causal_context_and_bidirectional_canvas():
    mask = LatentFlowE2D._inference_attention_mask(3, 2, torch.device("cpu"))
    assert mask[:3, :3].equal(torch.tril(torch.ones(3, 3, dtype=torch.bool)))
    assert mask[3:, :].all()
    assert not mask[:3, 3:].any()


def test_additive_mask_accepts_batched_and_unbatched_masks():
    mask = torch.eye(3, dtype=torch.bool)
    assert LatentFlowE2D._as_additive_mask(mask, torch.float32).shape == (1, 1, 3, 3)
    assert LatentFlowE2D._as_additive_mask(
        mask[None].repeat(2, 1, 1), torch.float32
    ).shape == (
        2,
        1,
        3,
        3,
    )


def test_fixed_training_path_repeats_noise_and_block_times():
    model = _bare_model(fixed=True)
    clean = torch.zeros(1, 8, 3)
    noise_a, time_a = model._sample_training_path(clean, None)
    noise_b, time_b = model._sample_training_path(clean, None)
    assert torch.equal(noise_a, noise_b)
    assert torch.equal(time_a, time_b)
    assert torch.equal(time_a[:, :4], time_a[:, :1].expand(-1, 4))
    assert torch.equal(time_a[:, 4:], time_a[:, 4:5].expand(-1, 4))


def test_stochastic_training_path_uses_supplied_collator_times():
    model = _bare_model(fixed=False)
    clean = torch.zeros(1, 8, 3)
    supplied = torch.full((1, 8), 0.25)
    _, sampled = model._sample_training_path(clean, supplied)
    assert sampled is supplied
