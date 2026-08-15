import pytest
import torch

from src.denoiser.speculative import (
    DecoderMode,
    SpeculativeStats,
    corrupt_discrete_tokens,
    linear_flow_sample,
    longest_matching_prefix,
)


def test_longest_matching_prefix():
    proposal = torch.tensor([[1, 2, 3], [1, 9, 3], [0, 2, 3]])
    target = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert longest_matching_prefix(proposal, target).tolist() == [3, 1, 0]


def test_masked_corruption_respects_valid_mask():
    clean = torch.tensor([[1, 2, 3, 4]])
    noisy, changed = corrupt_discrete_tokens(
        clean,
        torch.ones_like(clean, dtype=torch.float),
        mode=DecoderMode.MASKED,
        vocab_size=10,
        mask_token_id=9,
        valid_mask=torch.tensor([[1, 1, 0, 0]]),
    )
    assert noisy.tolist() == [[9, 9, 3, 4]]
    assert changed.tolist() == [[True, True, False, False]]


def test_uniform_corruption_is_seeded_and_in_vocabulary():
    clean = torch.zeros((2, 16), dtype=torch.long)
    generator = torch.Generator().manual_seed(4)
    noisy, changed = corrupt_discrete_tokens(
        clean,
        torch.ones_like(clean, dtype=torch.float),
        mode=DecoderMode.UNIFORM,
        vocab_size=7,
        generator=generator,
    )
    assert changed.all()
    assert noisy.min() >= 0 and noisy.max() < 7


@pytest.mark.parametrize("time", [0.0, 1.0, 0.4])
def test_linear_flow(time):
    clean = torch.tensor([[[1.0, 2.0]]])
    noise = torch.tensor([[[5.0, 8.0]]])
    noisy, velocity = linear_flow_sample(clean, noise, torch.tensor([time]))
    assert torch.allclose(noisy, (1 - time) * clean + time * noise)
    assert torch.equal(velocity, noise - clean)


def test_stats_derived_metrics_are_safe():
    stats = SpeculativeStats(
        proposed_tokens=8,
        accepted_tokens=6,
        committed_tokens=10,
        accepted_lengths=[4, 2],
        total_seconds=2.0,
    )
    output = stats.to_dict()
    assert output["acceptance_rate"] == 0.75
    assert output["average_accepted_length"] == 3.0
    assert output["tokens_per_second"] == 5.0
