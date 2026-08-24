from types import SimpleNamespace

import torch

from src.denoiser.embedding_flow_map import EmbeddingFlowMap


def _bare_model(epsilon=1e-4):
    model = EmbeddingFlowMap.__new__(EmbeddingFlowMap)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(time_epsilon=epsilon)
    return model


def test_block_mask_has_causal_clean_stream_and_bidirectional_noisy_block():
    valid = torch.ones(1, 6, dtype=torch.bool)
    context = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    mask = EmbeddingFlowMap.build_block_attention_mask(valid, context, block_size=2)

    # The clean copy is strictly causal, including inside a block.
    assert torch.equal(mask[0, :6, :6], torch.tril(torch.ones(6, 6, dtype=torch.bool)))
    # Noisy target position 4 (second target block) sees prompt + first clean
    # target block, and both positions in its own noisy canvas.
    assert mask[0, 10].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_flow_map_is_identity_on_equal_times():
    model = _bare_model()
    state = torch.randn(2, 3, 4)
    delta = torch.randn_like(state)
    time = torch.tensor([0.2, 0.7])

    mapped = model.flow_map(state, delta, time, time)

    assert torch.allclose(mapped, state)


def test_flow_map_reaches_denoiser_at_data_boundary():
    model = _bare_model()
    state = torch.randn(2, 3, 4)
    delta = torch.randn_like(state)
    start = torch.tensor([0.0, 0.4])
    end = torch.ones(2)

    mapped = model.flow_map(state, delta, start, end)

    assert torch.allclose(mapped, delta)


def test_soft_ce_detaches_teacher_but_updates_student():
    student = torch.randn(1, 2, 5, requires_grad=True)
    teacher_logits = torch.randn(1, 2, 5, requires_grad=True)
    teacher = torch.softmax(teacher_logits, -1).detach()
    mask = torch.ones(1, 2, dtype=torch.bool)

    loss = EmbeddingFlowMap._soft_ce(student, teacher, mask)
    loss.backward()

    assert student.grad is not None
    assert teacher_logits.grad is None


def test_semigroup_probability_coefficient_is_convex():
    start = torch.tensor([0.0, 0.2, 0.7])
    middle = torch.tensor([0.25, 0.4, 0.8])
    end = torch.tensor([0.5, 0.9, 0.95])
    gamma = ((1.0 - end) * (middle - start)) / (
        (1.0 - middle) * (end - start)
    )

    assert bool(((gamma >= 0) & (gamma <= 1)).all())


def test_verified_generation_accepts_prefix_then_uses_ar_correction():
    model = _bare_model()
    model.eos_token_id = None
    model.pad_token_id = 0
    proposals = [torch.tensor([[3, 9]]), torch.tensor([[5]])]
    model._draft_block = lambda context, block_len, num_steps: proposals.pop(0)

    def ar_logits(ids):
        vocab = 16
        logits = torch.full((*ids.shape, vocab), -10.0)
        predictions = (ids + 1) % vocab
        logits.scatter_(-1, predictions.unsqueeze(-1), 10.0)
        return logits

    model._run_ar_logits = ar_logits
    output, stats = model.generate_verified(
        torch.tensor([[1, 2]]),
        max_new_tokens=3,
        block_size=2,
        num_steps=1,
        return_speculative_stats=True,
    )

    assert output.tolist() == [[1, 2, 3, 4, 5]]
    assert stats.accepted_lengths == [1, 1]
    assert stats.accepted_tokens == 2
    assert stats.correction_tokens == 1
    assert stats.proposed_tokens == 3


def test_generate_dispatches_between_ar_draft_and_verified_modes():
    model = _bare_model()
    model.config.block_size = 4
    model.config.inference_steps = 1
    model.config.generation_mode = "verified"
    model.generate_ar = lambda inputs, count: torch.tensor([[10]])
    model.generate_draft = lambda inputs, count, block, steps: torch.tensor([[20]])
    model.generate_verified = lambda *args, **kwargs: torch.tensor([[30]])
    inputs = torch.tensor([[1]])

    assert model.generate(inputs, max_new_tokens=1, generation_mode="ar").item() == 10
    assert (
        model.generate(inputs, max_new_tokens=1, generation_mode="draft").item()
        == 20
    )
    assert model.generate(inputs, max_new_tokens=1).item() == 30
