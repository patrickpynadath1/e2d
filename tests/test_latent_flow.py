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
    model.self_conditioning_head = None
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


def test_feature_normalization_broadcasts_across_arbitrary_sequence_lengths():
    model = _bare_model(fixed=False)
    model.register_buffer("latent_mean", torch.tensor([1.0, 2.0, 3.0]))
    model.register_buffer("latent_std", torch.tensor([2.0, 4.0, 8.0]))
    latents = torch.tensor([[[3.0, 6.0, 11.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -5.0]]])

    normalized = model.normalize_latents(latents)
    reconstructed = model.denormalize_latents(normalized)

    assert normalized.shape == latents.shape
    assert torch.allclose(normalized[0, 0], torch.ones(3))
    assert torch.allclose(reconstructed, latents)


@pytest.mark.parametrize(
    "prediction_type", ["velocity", "x0", "x0_prev_block_residual"]
)
def test_prediction_parameterizations_recover_clean_endpoint(prediction_type):
    model = _bare_model(fixed=False)
    model.config.prediction_type = prediction_type
    clean = torch.randn(2, 4, 3)
    noise = torch.randn_like(clean)
    time = torch.rand(2, 4)
    noisy = (1.0 - time.unsqueeze(-1)) * clean + time.unsqueeze(-1) * noise
    target_velocity = noise - clean
    baseline = torch.randn_like(clean)
    target = model._prediction_target(clean, noisy, target_velocity, baseline)

    recovered = model.prediction_to_x0(noisy, target, time, baseline)

    assert torch.allclose(recovered, clean, atol=1e-6)


def test_training_residual_baseline_uses_prompt_then_previous_target_block():
    clean = torch.arange(1, 11, dtype=torch.float32).view(1, 10, 1)
    valid = torch.ones(1, 10, dtype=torch.bool)
    context = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)

    baseline = LatentFlowE2D._training_previous_block_baseline(
        clean, valid, context, block_size=4
    )

    # The first target block uses all three prompt positions: mean([1, 2, 3]) = 2.
    assert torch.equal(baseline[0, 3:7], torch.full((4, 1), 2.0))
    # The second target block uses the prior target block: mean([4, 5, 6, 7]) = 5.5.
    assert torch.equal(baseline[0, 7:10], torch.full((3, 1), 5.5))


def test_training_residual_baseline_ignores_padding_and_truncates_prompt_tail():
    clean = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1)
    valid = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    context = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    baseline = LatentFlowE2D._training_previous_block_baseline(
        clean, valid, context, block_size=3
    )

    # Only the last three valid prompt positions contribute: mean([3, 4, 5]) = 4.
    assert torch.equal(baseline[0, 5:7], torch.full((2, 1), 4.0))
    assert baseline[0, 0].item() == 0.0
    assert baseline[0, 7].item() == 0.0


def test_generation_residual_baseline_uses_last_full_context_block():
    context = torch.arange(1, 8, dtype=torch.float32).view(1, 7, 1)

    baseline = LatentFlowE2D._generation_previous_block_baseline(
        context, block_len=2, previous_block_size=4
    )

    assert torch.equal(baseline, torch.full((1, 2, 1), 5.5))


def test_one_step_sampling_bootstraps_self_conditioning_with_preliminary_x0():
    model = _bare_model(fixed=False)
    model.config.prediction_type = "x0"
    model.config.stats_epsilon = 1e-6
    model.self_conditioning_head = torch.nn.Identity()
    calls = []

    def run_flow(
        clean,
        noisy,
        time,
        mask,
        clean_positions,
        noisy_positions,
        self_conditioning=None,
    ):
        del clean, time, mask, clean_positions, noisy_positions
        calls.append((noisy.clone(), self_conditioning))
        return (
            torch.ones_like(noisy)
            if self_conditioning is None
            else torch.zeros_like(noisy)
        )

    model._run_flow_layers = run_flow
    context = torch.zeros(1, 3, 2)

    model._draft_latents(context, block_len=2, num_steps=1, previous_block_size=2)

    assert len(calls) == 2
    assert calls[0][1] is None
    assert torch.equal(calls[1][1], calls[0][0] + 1.0)


def test_unobserved_adaptive_sampler_is_uniform():
    model = _bare_model(fixed=False)
    model.config.adaptive_num_bins = 4
    model.config.adaptive_min_observations = 2
    model.config.adaptive_uniform_mix = 0.2
    model.register_buffer("adaptive_kl_ema", torch.zeros(4, dtype=torch.float64))
    model.register_buffer("adaptive_kl_counts", torch.zeros(4, dtype=torch.long))

    probabilities = model.adaptive_timestep_probabilities()

    assert torch.equal(probabilities, torch.full((4,), 0.25, dtype=torch.float64))


def test_adaptive_sampler_uses_monotone_kl_slopes_with_uniform_mix():
    model = _bare_model(fixed=False)
    model.config.adaptive_num_bins = 3
    model.config.adaptive_min_observations = 1
    model.config.adaptive_uniform_mix = 0.3
    model.register_buffer(
        "adaptive_kl_ema", torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64)
    )
    model.register_buffer("adaptive_kl_counts", torch.ones(3, dtype=torch.long))

    probabilities = model.adaptive_timestep_probabilities()

    expected_adaptive = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64) / 2.0
    expected = 0.7 * expected_adaptive + 0.3 / 3.0
    assert torch.allclose(probabilities, expected)


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
    model._draft_latents = (
        lambda context, block_len, num_steps, block_size: torch.zeros(
            context.shape[0], block_len, context.shape[-1]
        )
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


def test_previous_radius_conditioning_encodes_distinct_context_features():
    model = RiemannianLatentFlowE2D.__new__(RiemannianLatentFlowE2D)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(scalar_prediction_clip=5.0)
    model.radius_conditioner = torch.nn.Linear(1, 3, bias=False)
    with torch.no_grad():
        model.radius_conditioner.weight.copy_(torch.tensor([[1.0], [0.0], [-1.0]]))
    directions = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    radii = torch.tensor([[1.0, 2.0]])

    conditioned = model._condition_context(directions, radii)

    assert torch.allclose(conditioned[0, 0], torch.tensor([1.0, 1.0, -1.0]))
    assert torch.allclose(conditioned[0, 1], torch.tensor([2.0, 1.0, -2.0]))
