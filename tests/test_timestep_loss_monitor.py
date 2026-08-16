from types import SimpleNamespace

import torch

from src.custom_composer.callbacks import TimestepLossMonitor


class _Logger:
    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


def test_timestep_loss_monitor_logs_token_weighted_bins(monkeypatch):
    monkeypatch.setattr(
        "src.custom_composer.callbacks.dist.all_reduce", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.custom_composer.callbacks.dist.get_global_rank", lambda: 0
    )
    output = SimpleNamespace(
        nlls=torch.tensor([[1.0, 3.0, 5.0, 7.0]]),
        flow_timesteps=torch.tensor([[0.1, 0.2, 0.8, 1.0]]),
        tokens_mask=torch.ones(1, 4),
    )
    state = SimpleNamespace(
        outputs=output,
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=100)),
    )
    logger = _Logger()
    monitor = TimestepLossMonitor(num_bins=2)

    monitor.after_forward(state, logger)
    monitor._log_split(state, logger, "train")

    assert logger.metrics["timestep_loss/train/bin_00"] == 2.0
    assert logger.metrics["timestep_loss/train/bin_01"] == 6.0
    assert logger.metrics["timestep_loss/train/count_00"] == 2.0
    assert logger.metrics["timestep_loss/train/count_01"] == 2.0
    assert monitor._train_sums is None
    assert monitor._train_counts is None
