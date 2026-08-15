from types import SimpleNamespace

import torch

from src.denoiser.base import DenoiserInput
from src.denoiser.diffusion import E2D2
from src.denoiser.discrete_diffusion import DiscreteDiffusionE2D
from src.denoiser.speculative import DecoderMode


def _bare_model(mode: DecoderMode = DecoderMode.MASKED) -> DiscreteDiffusionE2D:
    model = DiscreteDiffusionE2D.__new__(DiscreteDiffusionE2D)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(decoder_loss_lambda=1.0)
    model.corruption_mode = mode
    model.vocab_size = 8
    model.mask_token_id = 7
    return model


def test_dual_decoder_mask_uses_clean_history_and_noisy_current_block():
    q_idx = torch.tensor([[2]])
    kv_idx = torch.arange(8)[None, :]
    mask = E2D2._decoder_block_mask(
        None, None, q_idx, kv_idx, block_size=2, seq_length=4
    )
    assert mask.tolist() == [[True, True, False, False, False, False, True, True]]


def test_uniform_replacements_exclude_clean_and_mask_tokens():
    torch.manual_seed(3)
    model = _bare_model(DecoderMode.UNIFORM)
    clean = torch.arange(7).repeat(20, 1)
    replacements = model._uniform_replacements(clean)
    assert not replacements.eq(clean).any()
    assert not replacements.eq(model.mask_token_id).any()
    assert replacements.min() >= 0 and replacements.max() < model.vocab_size


def test_discrete_loss_uses_ar_shift_and_same_position_decoder_targets():
    model = _bare_model()
    clean = torch.tensor([[1, 2, 3]])
    noisy = torch.tensor([[1, 7, 7]])
    logits = torch.full((1, 6, 8), -10.0)
    # Encoder predicts the next token.
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    # Decoder reconstructs the clean token at the same position.
    logits[0, 3, 1] = 10.0
    logits[0, 4, 2] = 10.0
    logits[0, 5, 3] = 10.0
    result = model._compute_loss(
        logits,
        DenoiserInput(
            xt=noisy,
            x0=clean,
            tokens_mask=torch.ones_like(clean),
        ),
    )
    assert result.loss < 1e-5
    assert result.other_loss_terms["encoder_loss"] < 1e-5
    assert result.other_loss_terms["decoder_loss"] < 1e-5
    assert torch.isclose(
        result.other_loss_terms["corrupted_fraction"], torch.tensor(2 / 3)
    )


def test_masked_forward_handles_encoder_and_decoder_halves():
    model = _bare_model()
    clean = torch.tensor([[1, 2]])
    noisy = torch.tensor([[1, 7]])
    output = model._forward(
        torch.randn(1, 4, 8),
        DenoiserInput(xt=noisy, x0=clean),
    )
    assert output.shape == (1, 4, 8)
    assert output[0, 2].argmax() == 1
    assert output[0, 3, model.mask_token_id] < -1e10
