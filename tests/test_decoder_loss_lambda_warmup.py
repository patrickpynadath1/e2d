"""Offline checks for lambda scheduling, loss gradients, and launcher defaults."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from composer import Trainer
from composer.core import Time, Timestamp
from composer.models import ComposerModel
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.custom_composer.decoder_loss_lambda_warmup import DecoderLossLambdaWarmup
from src.denoiser.base import DenoiserInput
from src.denoiser.diffusion import E2D

REPOSITORY = Path(__file__).resolve().parents[1]


def make_state(duration="1ep", *, batch=0, epoch=0, batch_in_epoch=0, wrapped=False):
    model = SimpleNamespace(
        config=SimpleNamespace(decoder_loss_lambda=0.5),
        backbone=SimpleNamespace(frozen_target=False),
    )
    composer_model = SimpleNamespace(model=model)
    return (
        SimpleNamespace(
            model=SimpleNamespace(module=composer_model) if wrapped else composer_model,
            max_duration=Time.from_timestring(duration),
            dataloader_len=Time.from_timestring("100ba"),
            timestamp=Timestamp(
                batch=batch, epoch=epoch, batch_in_epoch=batch_in_epoch
            ),
        ),
        model,
    )


class LambdaWarmupTest(unittest.TestCase):
    def test_initial_hold_ramp_and_final_hold(self):
        for duration in ("1ep", "100ba"):
            with self.subTest(duration=duration):
                callback = DecoderLossLambdaWarmup(0.1, 0.5, 0.7, keep_first_ratio=0.3)
                state, model = make_state(duration)
                for completed, expected in (
                    (0, 0.1),
                    (15, 0.1),
                    (29, 0.1),
                    (30, 0.1),
                    (40, 0.2),
                    (50, 0.3),
                    (60, 0.4),
                    (70, 0.5),
                    (99, 0.5),
                ):
                    state.timestamp = Timestamp(
                        batch=completed, batch_in_epoch=completed
                    )
                    callback.batch_start(state, Mock())
                    self.assertAlmostEqual(model.config.decoder_loss_lambda, expected)

    def test_resume_during_each_stage(self):
        for completed, expected in ((15, 0.1), (50, 0.3), (90, 0.5)):
            with self.subTest(completed=completed):
                state, model = make_state(batch=completed, batch_in_epoch=completed)
                DecoderLossLambdaWarmup(0.1, 0.5, 0.7, keep_first_ratio=0.3).fit_start(
                    state, Mock()
                )
                self.assertAlmostEqual(model.config.decoder_loss_lambda, expected)

    def test_equal_ratios_switch_without_ramp(self):
        callback = DecoderLossLambdaWarmup(0.1, 0.5, 0.3, keep_first_ratio=0.3)
        state, model = make_state()
        for completed, expected in ((0, 0.1), (29, 0.1), (30, 0.5), (99, 0.5)):
            state.timestamp = Timestamp(batch=completed, batch_in_epoch=completed)
            callback.batch_start(state, Mock())
            self.assertEqual(model.config.decoder_loss_lambda, expected)

    def test_single_epoch_ramp_and_plateau(self):
        for wrapped in (False, True):
            callback = DecoderLossLambdaWarmup(0.1, 0.5, 0.5)
            state, model = make_state(wrapped=wrapped)
            callback.fit_start(state, Mock())
            self.assertEqual(model.config.decoder_loss_lambda, 0.1)
            for completed, expected in ((0, 0.1), (25, 0.3), (50, 0.5), (99, 0.5)):
                state.timestamp = Timestamp(batch=completed, batch_in_epoch=completed)
                logger = Mock()
                callback.batch_start(state, logger)
                self.assertAlmostEqual(model.config.decoder_loss_lambda, expected)
                logged = logger.log_metrics.call_args.args[0]
                self.assertAlmostEqual(
                    logged["loss/train/decoder_loss_lambda"], expected
                )

    def test_resumed_fit_uses_restored_progress(self):
        for duration, timestamp in (
            ("100ba", Timestamp(batch=25)),
            ("4ep", Timestamp(epoch=1, batch=100)),
            ("1000sp", Timestamp(sample=250)),
            ("10000tok", Timestamp(token=2500)),
        ):
            with self.subTest(duration=duration):
                state, model = make_state(duration)
                state.timestamp = timestamp
                callback = DecoderLossLambdaWarmup(0.1, 0.5, 0.5)
                callback.fit_start(state, Mock())
                self.assertAlmostEqual(model.config.decoder_loss_lambda, 0.3)

    def test_completed_batch_updates_lambda_for_evaluation_and_checkpoint(self):
        state, model = make_state(batch=49, batch_in_epoch=49)
        callback = DecoderLossLambdaWarmup(0.0, 0.5, 0.5)
        callback.batch_start(state, Mock())
        self.assertAlmostEqual(model.config.decoder_loss_lambda, 0.49)
        state.timestamp = state.timestamp.to_next_batch(samples=32, tokens=128)
        callback.batch_end(state, Mock())
        self.assertEqual(model.config.decoder_loss_lambda, 0.5)

    def test_zero_and_full_duration_warmup(self):
        state, model = make_state()
        DecoderLossLambdaWarmup(0.0, 0.5, 0.0).fit_start(state, Mock())
        self.assertEqual(model.config.decoder_loss_lambda, 0.5)
        state.timestamp = Timestamp(epoch=1, batch=100)
        DecoderLossLambdaWarmup(0.0, 0.5, 1.0).batch_end(state, Mock())
        self.assertEqual(model.config.decoder_loss_lambda, 0.5)

    def test_invalid_settings_and_unsupported_models(self):
        for settings in (
            (-0.1, 0.5, 0.5),
            (1, 0.5, 0.5),
            (0, 1, -1),
            (0, 1, 1.1),
            (0, float("inf"), 0.5),
            (0, 1, float("nan")),
            (0, 1, 0.7, -0.1),
            (0, 1, 0.7, 0.8),
            (0, 1, 0.0, 0.3),
            (0, 1, 0.7, float("nan")),
            (0, 1, 0.7, float("inf")),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DecoderLossLambdaWarmup(*settings)
        state, model = make_state()
        callback = DecoderLossLambdaWarmup(0, 0.5, 0.5)
        model.backbone.frozen_target = True
        with self.assertRaisesRegex(ValueError, "Frozen-target"):
            callback.fit_start(state, Mock())
        model.backbone.frozen_target = False
        state.dataloader_len = None
        with self.assertRaisesRegex(ValueError, "sized dataloader"):
            callback.fit_start(state, Mock())

    def test_schedule_changes_actual_e2d_loss_and_gradients(self):
        state, model = make_state()
        callback = DecoderLossLambdaWarmup(0.0, 0.5, 0.7, keep_first_ratio=0.3)
        tokens = torch.tensor([[1, 2, 3, 4]])
        for step, expected_lambda in (
            (0, 0.0),
            (20, 0.0),
            (30, 0.0),
            (50, 0.25),
            (70, 0.5),
        ):
            state.timestamp = Timestamp(batch=step, batch_in_epoch=step)
            callback.batch_start(state, Mock())
            logits = torch.randn(1, 8, 8, requires_grad=True)
            inputs = DenoiserInput(xt=tokens, x0=tokens)
            loss = E2D._compute_loss(model, logits, inputs).loss
            encoder_loss = F.cross_entropy(
                logits[:, :3].reshape(-1, 8), tokens[:, 1:].reshape(-1)
            )
            decoder_loss = F.cross_entropy(
                logits[:, 4:7].reshape(-1, 8), tokens[:, 1:].reshape(-1)
            )
            expected = (encoder_loss + expected_lambda * decoder_loss) / (
                1 + expected_lambda
            )
            torch.testing.assert_close(loss, expected)
            gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            expected_gradient = torch.autograd.grad(expected, logits)[0]
            torch.testing.assert_close(gradient, expected_gradient)
            if expected_lambda == 0:
                self.assertEqual(torch.count_nonzero(gradient[:, 4:]).item(), 0)
                self.assertGreater(torch.count_nonzero(gradient[:, :3]).item(), 0)

    def test_composer_events_keep_lambda_constant_across_microbatches(self):
        class TinyJointModel(ComposerModel):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(1, 1)
                self.model.config = SimpleNamespace(decoder_loss_lambda=0.5)
                self.model.backbone = SimpleNamespace(frozen_target=False)
                self.seen_lambdas = []

            def forward(self, batch):
                self.seen_lambdas.append(self.model.config.decoder_loss_lambda)
                return self.model(batch).square().mean()

            def loss(self, outputs, batch):
                return outputs

        model = TinyJointModel()
        trainer = Trainer(
            model=model,
            train_dataloader=DataLoader(torch.ones(8, 1), batch_size=2),
            optimizers=torch.optim.SGD(model.parameters(), lr=0.01),
            max_duration="1ep",
            callbacks=[DecoderLossLambdaWarmup(0, 0.5, 0.75, keep_first_ratio=0.25)],
            device="cpu",
            precision="fp32",
            device_train_microbatch_size=1,
            progress_bar=False,
            log_to_console=False,
        )
        try:
            trainer.fit()
            self.assertEqual(model.seen_lambdas, [0, 0, 0, 0, 0.25, 0.25, 0.5, 0.5])
            self.assertEqual(model.model.config.decoder_loss_lambda, 0.5)
        finally:
            trainer.close()


class LambdaLauncherTest(unittest.TestCase):
    def launch(self, script, *, settings=None, extra_args=()):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bash_scripts").mkdir()
            for filename in (
                "run_train_e2d_ultrachat.sh",
                "run_train_e2d_ultrachat_warmup_lambda.sh",
            ):
                shutil.copy(
                    REPOSITORY / "bash_scripts" / filename, root / "bash_scripts"
                )
            (root / "setup_env.sh").write_text("# isolated test environment\n")
            for split in ("train", "eval"):
                (root / "data" / f"{split}_preprocessed").mkdir(parents=True)
            (root / "bin").mkdir()
            executable = root / "bin" / "composer"
            executable.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['TEST_ARGS_PATH'], 'w') as f:\n"
                "    json.dump(sys.argv[1:], f)\n"
            )
            executable.chmod(0o755)
            environment = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("DISTILL_", "DECODER_LOSS_LAMBDA"))
            }
            environment.update(
                {
                    "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                    "TEST_ARGS_PATH": str(root / "args.json"),
                    "DISTILL_DATA_ROOT": str(root / "data"),
                    "CUDA_VISIBLE_DEVICES": "0",
                    "FORCE_REGENERATE": "false",
                    **(settings or {}),
                }
            )
            subprocess.run(
                ["bash", str(root / "bash_scripts" / script), *extra_args],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads((root / "args.json").read_text())

    def compose_launch(self, arguments):
        with initialize_config_dir(
            config_dir=str(REPOSITORY / "configs"), version_base=None
        ):
            return compose(config_name="config", overrides=arguments[3:])

    def test_original_launcher_keeps_fixed_lambda_and_default_callbacks(self):
        arguments = self.launch("run_train_e2d_ultrachat.sh")
        cfg = self.compose_launch(arguments)
        self.assertEqual(cfg.model.config.decoder_loss_lambda, 0.5)
        self.assertNotIn("decoder_loss_lambda_warmup", cfg.composer.callbacks)

    def test_warmup_launcher_configures_schedule_and_forwards_overrides(self):
        arguments = self.launch(
            "run_train_e2d_ultrachat_warmup_lambda.sh",
            settings={
                "DECODER_LOSS_LAMBDA_START": "0.05",
                "DECODER_LOSS_LAMBDA": "1.5",
                "DECODER_LOSS_LAMBDA_KEEP_FIRST_RATIO": "0.3",
                "DECODER_LOSS_LAMBDA_WARMUP_RATIO": "0.7",
            },
            extra_args=("composer.trainer.max_duration=100ba",),
        )
        cfg = self.compose_launch(arguments)
        callback = instantiate(cfg.composer.callbacks.decoder_loss_lambda_warmup)
        self.assertEqual(
            (
                callback.start_lambda,
                callback.final_lambda,
                callback.warmup_ratio,
                callback.keep_first_ratio,
            ),
            (0.05, 1.5, 0.7, 0.3),
        )
        self.assertEqual(cfg.composer.trainer.max_duration, "100ba")
        self.assertIn("hf_compatible_checkpointing", cfg.composer.callbacks)

    def test_warmup_launcher_defaults_to_no_initial_hold(self):
        arguments = self.launch("run_train_e2d_ultrachat_warmup_lambda.sh")
        cfg = self.compose_launch(arguments)
        callback = instantiate(cfg.composer.callbacks.decoder_loss_lambda_warmup)
        self.assertEqual(callback.keep_first_ratio, 0.0)
        self.assertEqual(callback.warmup_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
