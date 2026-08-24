from types import SimpleNamespace

import torch

from src.custom_composer.callbacks import (
    SpeculativeGenerationEvaluator,
    _extract_gsm8k_answer,
    _read_speculative_stats,
    _target_greedy,
)
from src.denoiser.speculative import SpeculativeStats


class _Target(torch.nn.Module):
    def forward(self, input_ids, use_cache=False):
        del use_cache
        vocab_size = 16
        logits = torch.full((*input_ids.shape, vocab_size), -10.0)
        predictions = (input_ids + 1) % vocab_size
        logits.scatter_(-1, predictions.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits)


class _Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.target = _Target()
        self.eos_token_id = None
        self.last_speculative_stats = SpeculativeStats()

    def generate(self, inputs, max_new_tokens, **kwargs):
        del kwargs
        output = _target_greedy(self, inputs, max_new_tokens)
        self.last_speculative_stats = SpeculativeStats(
            proposed_tokens=8,
            accepted_tokens=6,
            correction_tokens=1,
            committed_tokens=max_new_tokens,
            draft_calls=2,
            verifier_calls=2,
            accepted_lengths=[2, 1],
            draft_seconds=0.5,
            verifier_seconds=0.5,
            total_seconds=2.0,
        )
        return output


class _Tokenizer:
    def __call__(self, text, **kwargs):
        del text, kwargs
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))


class _Logger:
    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


def test_read_speculative_stats_supports_object_and_legacy_dict():
    model = SimpleNamespace(last_speculative_stats=SpeculativeStats(accepted_tokens=3))
    assert _read_speculative_stats(model)["accepted_tokens"] == 3
    legacy = SimpleNamespace(_last_speculative_stats={"accepted_tokens": 4})
    assert _read_speculative_stats(legacy)["accepted_tokens"] == 4


def test_gsm8k_answer_extraction_prefers_boxed_and_normalizes_commas():
    assert _extract_gsm8k_answer("work 12 then $\\boxed{1,234}$") == "1234"
    assert _extract_gsm8k_answer("reasoning\n#### -42") == "-42"
    assert _extract_gsm8k_answer("The final value is 7.5") == "7.5"


def test_callback_logs_acceptance_positions_and_exact_match(monkeypatch):
    monkeypatch.setattr(
        "src.custom_composer.callbacks.dist.get_global_rank", lambda: 0
    )
    denoiser = _Denoiser().train()
    state = SimpleNamespace(
        model=SimpleNamespace(model=denoiser, tokenizer=_Tokenizer())
    )
    logger = _Logger()
    callback = SpeculativeGenerationEvaluator(
        prompts=["one", "two"], max_new_tokens=4, block_size=4
    )

    callback.eval_end(state, logger)

    assert denoiser.training
    assert logger.metrics["speculative/acceptance_rate"] == 0.75
    assert logger.metrics["speculative/average_accepted_length"] == 1.5
    assert logger.metrics["speculative/position_1_acceptance_rate"] == 1.0
    assert logger.metrics["speculative/position_2_acceptance_rate"] == 0.5
    assert logger.metrics["speculative/position_3_acceptance_rate"] == 0.0
    assert logger.metrics["speculative/exact_match_rate"] == 1.0
