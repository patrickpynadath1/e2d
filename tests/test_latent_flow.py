from types import SimpleNamespace

import pytest
import torch

from src.denoiser.latent_flow import (
    LatentFlowE2D,
    RiemannianLatentFlowE2D,
    sphere_expmap,
    spherical_interpolant_and_velocity,
)


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


class _GreedyTarget(torch.nn.Module):
    def __init__(self, target_tokens: torch.Tensor):
        super().__init__()
        self.target_tokens = target_tokens

    def forward(self, input_ids, use_cache=False):
        del use_cache
        logits = torch.full((*input_ids.shape, 16), -10.0)
        for position, token in enumerate(self.target_tokens[0]):
            logits[0, position, token] = 10.0
        return SimpleNamespace(logits=logits)


def test_generation_stops_at_eos_inside_fully_accepted_block():
    model = _bare_model(fixed=False)
    model.config.eval_block_size = 4
    model.config.inference_steps = 1
    model.eos_token_id = 9
    model.latent_stats_initialized = torch.tensor(True)
    proposal = torch.tensor([[1, 2, 9, 7]])
    model.target = _GreedyTarget(proposal)
    model.extract_clean_latents = lambda input_ids: torch.zeros(
        input_ids.shape[0], input_ids.shape[1], 2
    )
    model.normalize_latents = lambda latents: latents
    model._draft_latents = lambda context, block_len, num_steps: torch.zeros(
        context.shape[0], block_len, context.shape[-1]
    )
    model.decode_latents = lambda context, block: proposal[:, : block.shape[1]]
    generation_config = SimpleNamespace(
        block_size=4,
        num_steps=1,
        max_new_tokens=4,
    )

    generated, stats = model.generate(
        torch.tensor([[5]]),
        generation_config,
        return_speculative_stats=True,
    )

    assert generated.tolist() == [[5, 1, 2, 9]]
    assert stats.accepted_tokens == 3
    assert stats.accepted_lengths == [3]
    assert stats.committed_tokens == 3


def test_slerp_path_stays_on_sphere_and_velocity_is_tangent():
    clean = torch.tensor([[1.0, 0.0, 0.0]])
    noise = torch.tensor([[0.0, 1.0, 0.0]])
    midpoint, velocity = spherical_interpolant_and_velocity(
        clean, noise, torch.tensor([0.5])
    )

    expected = torch.tensor([[2**-0.5, 2**-0.5, 0.0]])
    assert torch.allclose(midpoint, expected, atol=1e-5)
    assert torch.allclose(midpoint.norm(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose((midpoint * velocity).sum(-1), torch.zeros(1), atol=1e-6)


def test_sphere_expmap_follows_quarter_circle_and_preserves_norm():
    point = torch.tensor([[1.0, 0.0, 0.0]])
    velocity = torch.tensor([[0.0, torch.pi / 2, 0.0]])
    updated = sphere_expmap(point, velocity, step_size=1.0)

    assert torch.allclose(updated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)
    assert torch.allclose(updated.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_log_norm_scalar_round_trip_reconstructs_latents():
    model = RiemannianLatentFlowE2D.__new__(RiemannianLatentFlowE2D)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(stats_epsilon=1e-6, scalar_prediction_clip=100.0)
    model.register_buffer("log_radius_mean", torch.tensor(8.02))
    model.register_buffer("log_radius_std", torch.tensor(0.1))
    latents = torch.randn(2, 3, 16) * torch.tensor([2.0, 4.0, 8.0])[None, :, None]

    direction, scalar = model.latent_direction_and_scalar(latents)
    reconstructed = model.reconstruct_latents(direction, scalar)

    assert torch.allclose(direction.norm(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(reconstructed, latents, rtol=1e-5, atol=1e-5)
