import glob
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Literal

import torch
from composer.callbacks import CheckpointSaver
from composer.core import Callback, State, Time, Timestamp
from composer.loggers import Logger
from composer.utils import PartialFilePath, dist, get_save_filename

import wandb
from src.denoiser.diffusion import DiffusionGenerationConfig
from src.denoiser.speculative import SpeculativeStats
from src.utils import (
    fsspec_exists,
    save_pretrained_or_push_to_hub,
    snapshot_repo_to_tmp_dir,
)

log = logging.getLogger(__name__)
__all__ = [
    "DataloaderSpeedMonitor",
    "SpeculativeGenerationEvaluator",
    "TimestepLossMonitor",
]


class TimestepLossMonitor(Callback):
    """Aggregate latent-flow token MSE into fixed timestep bins.

    Curves are reduced across all distributed ranks and emitted at validation end.
    Scalar bin values support ordinary logger dashboards; W&B additionally receives
    a table-backed line chart whose x-axis is the timestep-bin center.
    """

    def __init__(self, num_bins: int = 50) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least two")
        self.num_bins = num_bins
        self._train_sums: torch.Tensor | None = None
        self._train_counts: torch.Tensor | None = None
        self._eval_sums: torch.Tensor | None = None
        self._eval_counts: torch.Tensor | None = None

    def _empty(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_bins, dtype=torch.float64, device=device),
            torch.zeros(self.num_bins, dtype=torch.float64, device=device),
        )

    def _accumulate(self, outputs: Any, split: Literal["train", "eval"]) -> None:
        token_losses = getattr(outputs, "nlls", None)
        timesteps = getattr(outputs, "flow_timesteps", None)
        token_mask = getattr(outputs, "tokens_mask", None)
        if token_losses is None or timesteps is None:
            return

        token_losses = token_losses.detach()
        timesteps = timesteps.detach()
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None].expand_as(token_losses)
        else:
            timesteps = torch.broadcast_to(timesteps, token_losses.shape)
        weights = (
            torch.ones_like(token_losses)
            if token_mask is None
            else torch.broadcast_to(token_mask.detach(), token_losses.shape)
        )
        valid = weights.bool() & torch.isfinite(token_losses) & torch.isfinite(timesteps)
        if not bool(valid.any()):
            return

        indices = torch.floor(timesteps[valid].clamp(0.0, 1.0) * self.num_bins)
        indices = indices.long().clamp_max(self.num_bins - 1)
        weighted_losses = token_losses[valid].double() * weights[valid].double()
        valid_weights = weights[valid].double()

        sums_name = f"_{split}_sums"
        counts_name = f"_{split}_counts"
        sums = getattr(self, sums_name)
        counts = getattr(self, counts_name)
        if sums is None or sums.device != token_losses.device:
            sums, counts = self._empty(token_losses.device)
            setattr(self, sums_name, sums)
            setattr(self, counts_name, counts)
        sums.scatter_add_(0, indices, weighted_losses)
        counts.scatter_add_(0, indices, valid_weights)

    def after_forward(self, state: State, logger: Logger) -> None:
        self._accumulate(state.outputs, "train")

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        self._accumulate(state.outputs, "eval")

    def _log_split(self, state: State, logger: Logger, split: str) -> None:
        sums = getattr(self, f"_{split}_sums")
        counts = getattr(self, f"_{split}_counts")
        if sums is None or counts is None:
            return
        dist.all_reduce(sums, reduce_operation="SUM")
        dist.all_reduce(counts, reduce_operation="SUM")
        means = sums / counts.clamp_min(1.0)

        if dist.get_global_rank() == 0:
            metrics: dict[str, float] = {}
            table_rows: list[list[float | int | None]] = []
            step = int(state.timestamp.batch.value)
            for index in range(self.num_bins):
                low = index / self.num_bins
                high = (index + 1) / self.num_bins
                center = (low + high) / 2
                count = int(counts[index].item())
                mean = float(means[index].item()) if count else None
                metrics[f"timestep_loss/{split}/bin_{index:02d}"] = (
                    float("nan") if mean is None else mean
                )
                metrics[f"timestep_loss/{split}/count_{index:02d}"] = float(count)
                table_rows.append([step, center, mean, count])
            logger.log_metrics(metrics)
            if wandb.run is not None:
                table = wandb.Table(
                    columns=["optimizer_step", "timestep", "mean_flow_loss", "count"],
                    data=table_rows,
                )
                wandb.log(
                    {
                        f"timestep_loss/{split}/curve": wandb.plot.line(
                            table,
                            "timestep",
                            "mean_flow_loss",
                            title=f"{split.title()} flow loss by timestep at step {step}",
                        ),
                        f"timestep_loss/{split}/table": table,
                    },
                    commit=False,
                )

        setattr(self, f"_{split}_sums", None)
        setattr(self, f"_{split}_counts", None)

    def eval_end(self, state: State, logger: Logger) -> None:
        self._log_split(state, logger, "train")
        self._log_split(state, logger, "eval")


def _unwrap_huggingface_model(state_model: Any) -> tuple[Any, Any]:
    """Return the denoiser and tokenizer from Composer's optional DDP wrapper."""
    composer_model = (
        state_model.module if hasattr(state_model, "module") else state_model
    )
    if not hasattr(composer_model, "model") or not hasattr(composer_model, "tokenizer"):
        raise TypeError("speculative evaluation requires Composer HuggingFaceModel")
    return composer_model.model, composer_model.tokenizer


def _target_forward(model: Any, input_ids: torch.LongTensor) -> torch.Tensor:
    """Run the full target model for one uncached greedy step."""
    if hasattr(model, "target"):
        return model.target(input_ids=input_ids, use_cache=False).logits
    if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):
        return model.backbone.encoder(input_ids=input_ids, use_cache=False).logits
    raise TypeError("model does not expose a target verifier")


@torch.no_grad()
def _target_greedy(
    model: Any, input_ids: torch.LongTensor, max_new_tokens: int
) -> torch.LongTensor:
    output = input_ids
    eos_token_id = getattr(model, "eos_token_id", None)
    for _ in range(max_new_tokens):
        token = _target_forward(model, output)[:, -1:].argmax(-1)
        output = torch.cat([output, token], dim=-1)
        if eos_token_id is not None and bool((token == eos_token_id).all()):
            break
    return output


def _read_speculative_stats(model: Any) -> dict[str, Any]:
    stats = getattr(model, "last_speculative_stats", None)
    if stats is None:
        stats = getattr(model, "_last_speculative_stats", None)
    if isinstance(stats, SpeculativeStats):
        return stats.to_dict()
    if isinstance(stats, dict):
        return stats
    raise RuntimeError("model did not expose speculative generation statistics")


class SpeculativeGenerationEvaluator(Callback):
    """Run a small fixed speculative-generation evaluation after validation.

    Every distributed rank executes the same prompts so wrapped model forwards remain
    collective-safe. Only rank zero logs the identical deterministic result.
    """

    def __init__(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        block_size: int = 4,
        num_steps: int = 1,
        seed: int = 17,
        verify_exact_match: bool = True,
        num_prompts: int | None = None,
    ) -> None:
        if not prompts:
            raise ValueError("speculative evaluator requires at least one prompt")
        if max_new_tokens < 1 or block_size < 1 or num_steps < 1:
            raise ValueError("generation lengths and num_steps must be positive")
        if num_prompts is not None and num_prompts < 1:
            raise ValueError("num_prompts must be positive when provided")
        self.prompts = prompts[:num_prompts]
        self.max_new_tokens = max_new_tokens
        self.block_size = block_size
        self.num_steps = num_steps
        self.seed = seed
        self.verify_exact_match = verify_exact_match

    def eval_end(self, state: State, logger: Logger) -> None:
        model, tokenizer = _unwrap_huggingface_model(state.model)
        was_training = model.training
        device = next(model.parameters()).device
        totals: dict[str, float] = {
            "proposed_tokens": 0.0,
            "accepted_tokens": 0.0,
            "correction_tokens": 0.0,
            "committed_tokens": 0.0,
            "draft_calls": 0.0,
            "verifier_calls": 0.0,
            "draft_seconds": 0.0,
            "verifier_seconds": 0.0,
            "cache_seconds": 0.0,
            "total_seconds": 0.0,
        }
        accepted_lengths: list[int] = []
        position_attempts = [0] * self.block_size
        position_accepts = [0] * self.block_size
        exact_matches = 0
        greedy_seconds = 0.0
        speculative_seconds = 0.0
        generation_config = DiffusionGenerationConfig(
            max_new_tokens=self.max_new_tokens,
            block_size=self.block_size,
            num_steps=self.num_steps,
            use_cache=True,
            align_inputs_to_blocks=False,
        )

        model.eval()
        try:
            devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(self.seed)
                for prompt in self.prompts:
                    inputs = tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=True
                    ).input_ids.to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    speculative_started = time.perf_counter()
                    generate = getattr(model, "generate_verified", model.generate)
                    speculative_output = generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        max_new_tokens=self.max_new_tokens,
                        disable_pbar=True,
                        conf_seg=False,
                        tree_attn=False,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    speculative_seconds += time.perf_counter() - speculative_started
                    if isinstance(speculative_output, tuple):
                        speculative_output = speculative_output[0]
                    stats = _read_speculative_stats(model)
                    for key in totals:
                        totals[key] += float(stats.get(key, 0.0))
                    lengths = [
                        int(value) for value in stats.get("accepted_lengths", [])
                    ]
                    accepted_lengths.extend(lengths)
                    for length in lengths:
                        for position in range(self.block_size):
                            position_attempts[position] += 1
                            if position < length:
                                position_accepts[position] += 1

                    generated_count = speculative_output.shape[1] - inputs.shape[1]
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    greedy_started = time.perf_counter()
                    greedy_output = _target_greedy(model, inputs, generated_count)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    greedy_seconds += time.perf_counter() - greedy_started
                    matches = torch.equal(speculative_output, greedy_output)
                    exact_matches += int(matches)
                    if self.verify_exact_match and not matches:
                        raise RuntimeError(
                            "speculative output diverged from target-greedy output: "
                            f"speculative={speculative_output.tolist()}, "
                            f"greedy={greedy_output.tolist()}"
                        )
        finally:
            model.train(was_training)

        if dist.get_global_rank() != 0:
            return
        proposed = totals["proposed_tokens"]
        committed = totals["committed_tokens"]
        metrics = {
            "speculative/acceptance_rate": (
                totals["accepted_tokens"] / proposed if proposed else 0.0
            ),
            "speculative/average_accepted_length": (
                sum(accepted_lengths) / len(accepted_lengths)
                if accepted_lengths
                else 0.0
            ),
            "speculative/proposed_tokens": totals["proposed_tokens"],
            "speculative/accepted_tokens": totals["accepted_tokens"],
            "speculative/correction_tokens": totals["correction_tokens"],
            "speculative/committed_tokens": committed,
            "speculative/draft_calls": totals["draft_calls"],
            "speculative/verifier_calls": totals["verifier_calls"],
            "speculative/draft_seconds": totals["draft_seconds"],
            "speculative/verifier_seconds": totals["verifier_seconds"],
            "speculative/cache_seconds": totals["cache_seconds"],
            "speculative/total_seconds": totals["total_seconds"],
            "speculative/measured_total_seconds": speculative_seconds,
            "speculative/tokens_per_second": (
                committed / speculative_seconds if speculative_seconds else 0.0
            ),
            "speculative/target_greedy_seconds": greedy_seconds,
            "speculative/target_greedy_tokens_per_second": (
                committed / greedy_seconds if greedy_seconds else 0.0
            ),
            "speculative/speedup_vs_target_greedy": (
                greedy_seconds / speculative_seconds if speculative_seconds else 0.0
            ),
            "speculative/exact_match_rate": exact_matches / len(self.prompts),
        }
        for position, (accepted, attempted) in enumerate(
            zip(position_accepts, position_attempts), start=1
        ):
            metrics[f"speculative/position_{position}_acceptance_rate"] = (
                accepted / attempted if attempted else 0.0
            )
        for length in range(self.block_size + 1):
            metrics[f"speculative/accepted_length_{length}_fraction"] = (
                accepted_lengths.count(length) / len(accepted_lengths)
                if accepted_lengths
                else 0.0
            )
        logger.log_metrics(metrics)


class DataloaderSpeedMonitor(Callback):
    """Measure how long it takes to return a batch from the dataloader.

    Copied from:
        https://github.com/AnswerDotAI/ModernBERT/blob/main/src/callbacks/dataloader_speed.py
        Copyright 2024 onwards Answer.AI, LightOn, and contributors
        License: Apache-2.0
    """  # noqa: E501

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_time = time.time_ns()

    def after_dataloader(self, state: State, logger: Logger) -> None:
        self.batch_serve_time = time.time_ns() - self.batch_start_time
        logger.log_metrics(
            {
                "throughput/batch_serve_time_ns": self.batch_serve_time,
                "throughput/batch_serve_time_ms": self.batch_serve_time / 1e6,
            }
        )


class HuggingFaceCompatibleCheckpointing(CheckpointSaver):
    """A checkpoint callback that saves models in a manner in which one can
    `AutoModel.from_pretrained(<ckpt_path>)`.

    """

    def __init__(
        self,
        disable_hf: bool = False,
        save_local: bool = True,
        save_to_hub: bool = False,
        hub_repo_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_to_hub = save_to_hub and not disable_hf
        self.hub_repo_id = hub_repo_id
        if self.save_to_hub and hub_repo_id is None:
            raise ValueError("Saving to hub requires a hub repo id be provided.")
        self.save_local = save_local and not disable_hf
        self.disable_hf = disable_hf or not (self.save_to_hub or self.save_local)
        self.project_root = ""
        self.hf_filename = PartialFilePath(
            f"HF_{self.filename.filename.split('.pt')[0]}", self.filename.folder
        )
        if self.latest_filename is not None:
            self.latest_hf_filename = PartialFilePath(
                f"HF_{self.latest_filename.filename.split('.pt')[0]}",
                self.latest_filename.folder,
            )
        self.saved_hf_checkpoints: list[str] = []
        self.all_saved_hf_checkpoints_to_timestamp: dict[str, Timestamp] = {}

    def fit_start(self, state: State, logger: Logger) -> None:
        super().fit_start(state, logger)
        if dist.get_global_rank() == 0 and not self.disable_hf:
            self.project_root = snapshot_repo_to_tmp_dir(tmp_dir_exists_ok=True)
            log.info(f"Created tmp repo for HF checkpointing at {self.project_root}")
        dist.barrier()  # Holds all the ranks until repo snapshot is done

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()

        all_hf_checkpoints = []
        for (
            save_filename,
            timestamp,
        ) in self.all_saved_hf_checkpoints_to_timestamp.items():
            all_hf_checkpoints.append((save_filename, timestamp.state_dict()))

        state_dict["all_saved_hf_checkpoints_to_timestamp"] = all_hf_checkpoints
        return state_dict

    def load_state_dict(self, state: dict[str, Any]):
        super().load_state_dict(state)
        if "all_saved_hf_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state in state[
                "all_saved_hf_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_hf_checkpoints_to_timestamp[save_filename] = (
                    load_timestamp
                )

    def _save_checkpoint(self, state: State, logger: Logger) -> None:
        """
        Copied / adapted from composer.callbacks.CheckpointSaver._save_checkpoint
            for HF compatibility.
        """
        # TODO: Check that HF saving works with state.fsdp_sharded_state_dict_enabled
        #  (or if we can ignore this scenario).
        # TODO: Do we need to implement HF uploading for remote uploading too?
        if self.disable_hf:
            super()._save_checkpoint(state, logger)  # Perform standard checkpointing
            return

        hf_filename_with_placeholders = self.hf_filename.format(
            state, keep_placeholders=True
        )
        save_hf_filename = get_save_filename(state, hf_filename_with_placeholders)
        self.all_saved_hf_checkpoints_to_timestamp[save_hf_filename] = state.timestamp

        # Adapting `checkpoint.save_checkpoint / ._save_checkpoint` for HF
        saved_hf_path = None
        if dist.get_global_rank() == 0:
            if self.save_local:
                save_pretrained_or_push_to_hub(
                    model=state.model.module.model
                    if hasattr(state.model, "module")
                    else state.model.model,
                    tokenizer=state.model.module.tokenizer
                    if hasattr(state.model, "module")
                    else state.model.tokenizer,
                    repo_id=save_hf_filename,
                    local=True,
                    project_root=self.project_root,
                )
                saved_hf_path = save_hf_filename
                log.debug(f"HF checkpoint locally saved to {saved_hf_path}")
            if self.save_to_hub:
                metrics_str = "Train metrics:\n\t" + "\n\t".join(
                    [
                        f"{k}={v.item():0.4f}"
                        for k, v in state.train_metric_values.items()
                    ]
                )
                if hasattr(state, "eval_metric_values"):
                    metrics_str += "\n\nVal metrics:\n\t" + "\n\t".join(
                        [
                            f"{k}={v.item():0.4f}"
                            for k, v in state.eval_metric_values.items()
                        ]
                    )
                commit_message = (
                    f"Checkpoint @ Epoch {state.timestamp.epoch.value}, "
                    f"Batch {state.timestamp.batch.value}\n\n"
                    f"{metrics_str}\n\n"
                    f"Timestamp:\n"
                    f"\titeration={state.timestamp.iteration.value}\n"
                    f"\tepoch={state.timestamp.epoch.value}\n"
                    f"\tbatch={state.timestamp.batch.value}\n"
                    f"\tsample={state.timestamp.sample.value}\n"
                    f"\ttoken={state.timestamp.token.value}\n"
                    f"\tepoch_in_iteration={state.timestamp.epoch_in_iteration.value}\n"
                    f"\ttoken_in_iteration={state.timestamp.token_in_iteration.value}\n"
                    f"\tbatch_in_epoch={state.timestamp.batch_in_epoch.value}\n"
                    f"\tsample_in_epoch={state.timestamp.sample_in_epoch.value}\n"
                    f"\ttoken_in_epoch={state.timestamp.token_in_epoch.value}"
                )
                save_pretrained_or_push_to_hub(
                    model=state.model.module.model
                    if hasattr(state.model, "module")
                    else state.model.model,
                    tokenizer=state.model.module.tokenizer
                    if hasattr(state.model, "module")
                    else state.model.tokenizer,
                    repo_id=self.hub_repo_id,
                    local=False,
                    project_root=self.project_root,
                    commit_message=commit_message,
                )
            log.debug(f"HF checkpoint pushed to {self.hub_repo_id}")

        if not saved_hf_path:  # not all ranks save
            super()._save_checkpoint(state, logger)  # Perform standard checkpointing
            return

        self.rank_saves_symlinks = (
            dist.get_global_rank() == 0 or not state.fsdp_sharded_state_dict_enabled
        )
        if self.latest_hf_filename is not None and self.num_checkpoints_to_keep != 0:
            symlink = self.latest_hf_filename.format(state)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            # Sharded checkpoints for torch >2.0 use directories not files for
            # load_paths
            if state.fsdp_sharded_state_dict_enabled:
                src_path = str(pathlib.Path(saved_hf_path).parent)
            else:
                src_path = saved_hf_path
            if self.rank_saves_symlinks:
                os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)
        self.saved_hf_checkpoints.append(saved_hf_path)

        if self.num_checkpoints_to_keep >= 0:
            # Adapting `super().__rotate_checkpoints` for HF
            while len(self.saved_hf_checkpoints) > self.num_checkpoints_to_keep:
                checkpoint_to_delete = self.saved_hf_checkpoints.pop(0)
                prefix_dir = str(pathlib.Path(checkpoint_to_delete).parent)
                if not state.fsdp_sharded_state_dict_enabled:
                    shutil.rmtree(checkpoint_to_delete)
                else:
                    if dist.get_global_rank() == 0:
                        shutil.rmtree(prefix_dir)
        super()._save_checkpoint(state, logger)  # Perform standard checkpointing

    def close(self, state: State, logger: Logger) -> None:
        """Clean up tmp repo snapshot"""
        if dist.get_global_rank() == 0:
            if self.project_root and fsspec_exists(self.project_root):
                shutil.rmtree(self.project_root)
        dist.barrier()
        super().close(state, logger)


class SaveBestCheckpointing(HuggingFaceCompatibleCheckpointing):
    """Save the best checkpoint based on a metric."""

    def __init__(
        self,
        metric_to_monitor: str,
        mode: Literal["min", "max"] = "min",
        disable_hf: bool = False,
        save_local: bool = True,
        save_to_hub: bool = False,
        hub_repo_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            disable_hf, save_local, save_to_hub, hub_repo_id, *args, **kwargs
        )

        self.metric_to_monitor = metric_to_monitor
        self.train_or_eval = metric_to_monitor.split("/")[0]
        self.metric_name = "/".join(metric_to_monitor.split("/")[1:])
        self.mode = mode
        self.best_value = None

        self.latest_filename = None
        self.latest_hf_filename = None

    @staticmethod
    def _validate_metric_name(metric_to_monitor: str) -> None:
        invalid_name = False
        split_name = metric_to_monitor.split("/")
        if len(split_name) < 2:
            invalid_name = True
        if split_name[0] not in ["train", "eval"]:
            invalid_name = True

        if invalid_name:
            raise ValueError(
                f"Invalid metric name {metric_to_monitor}. "
                "Expected format is <train|eval>/<metric_name>."
            )

    @property
    def _metric_dict(self) -> dict[str, Any]:
        return {
            "metric_to_monitor": self.metric_to_monitor,
            "mode": self.mode,
            "best_value": self.best_value,
        }

    def _trigger_save(self, metric_value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return metric_value < self.best_value
        return metric_value > self.best_value  # self.mode == "max"

    def eval_end(self, state: State, logger: Logger) -> None:
        metrics = (
            state.train_metric_values
            if self.train_or_eval == "train"
            else state.eval_metric_values
        )
        if self.metric_name not in [k.lower() for k in metrics.keys()]:
            raise ValueError(
                f"Metric {self.metric_name} not found in metrics {metrics.keys()}"
            )
        metrics = {
            key.lower(): (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in metrics.items()
        }
        metric_value = metrics[self.metric_name]
        if self._trigger_save(metric_value):
            filename_with_placeholders = self.filename.format(
                state, keep_placeholders=True
            )
            save_filename = get_save_filename(state, filename_with_placeholders)
            log.info(
                f"Best {self.metric_name} attained: {metric_value}. "
                f"Saving checkpoint to {save_filename}."
            )
            self.best_value = metrics[self.metric_name]
            self._save_checkpoint(state, logger)

    def batch_checkpoint(self, state: State, logger: Logger):
        pass  # Force no saving

    def epoch_checkpoint(self, state: State, logger: Logger):
        pass

    def iteration_checkpoint(self, state: State, logger: Logger):
        pass

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()

        # Add best checkpoint info to state_dict
        all_checkpoints = []
        for save_filename, timestamp in self.all_saved_checkpoints_to_timestamp.items():
            all_checkpoints.append(
                (save_filename, timestamp.state_dict(), self._metric_dict)
            )

        state_dict["all_saved_checkpoints_to_timestamp"] = all_checkpoints
        return state_dict

    def load_state_dict(self, state: dict[str, Any]):
        if "all_saved_hf_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state in state[
                "all_saved_hf_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_hf_checkpoints_to_timestamp[save_filename] = (
                    load_timestamp
                )
        if "all_saved_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state, metrics_dict in state[
                "all_saved_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_checkpoints_to_timestamp[save_filename] = load_timestamp
                self.best_value = metrics_dict["best_value"]  # restore best_value


class LogProfilingTraceToWandb(Callback):
    """Log profiling trace to wandb.

    See: https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
    """  # noqa: E501

    def __init__(
        self,
        composer_prof_folder: str,
        torch_prof_folder: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.composer_prof_folder = composer_prof_folder
        self.torch_prof_folder = torch_prof_folder

    def fit_end(self, state: State, logger: Logger) -> None:
        if dist.get_global_rank() == 0:
            profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
            for trace_file in glob.glob(f"{self.composer_prof_folder}/*.json"):
                profile_art.add_file(trace_file, os.path.basename(trace_file))
            for trace_file in glob.glob(f"{self.torch_prof_folder}/*.pt.trace.json"):
                profile_art.add_file(trace_file, os.path.basename(trace_file))
            wandb.run.log_artifact(profile_art).wait()
        dist.barrier()


class LogSampledTimestep(Callback):
    def batch_end(self, state: State, logger: Logger) -> None:
        sampled_t = state.batch.t
        logger.log_metrics(
            {
                f"sampled_t/{state.dataloader_label}/mean": sampled_t.mean().item(),
                f"sampled_t/{state.dataloader_label}/std": sampled_t.std().item(),
                f"sampled_t/{state.dataloader_label}/max": sampled_t.max().item(),
                f"sampled_t/{state.dataloader_label}/min": sampled_t.min().item(),
            },
        )

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        sampled_t = state.batch["t"]
        logger.log_metrics(
            {
                f"sampled_t/{state.dataloader_label}/mean": sampled_t.mean().item(),
                f"sampled_t/{state.dataloader_label}/std": sampled_t.std().item(),
                f"sampled_t/{state.dataloader_label}/max": sampled_t.max().item(),
                f"sampled_t/{state.dataloader_label}/min": sampled_t.min().item(),
            },
        )


class WarmupWithFrozenEncoder(Callback):
    def __init__(
        self,
        num_warmup_steps: str,
    ):
        super().__init__()
        self.num_warmup_steps = Time.from_timestring(num_warmup_steps)
        self.encoder_is_frozen = False

    def fit_start(self, state: State, logger: Logger) -> None:
        if hasattr(state.model, "module"):
            if not hasattr(state.model.module.model.backbone, "freeze_encoder"):
                raise NotImplementedError(
                    "Model backbone does not have freeze_encoder implemented."
                )
            state.model.module.model.backbone.freeze_encoder()
        else:
            if not hasattr(state.model.model.backbone, "freeze_encoder"):
                raise NotImplementedError(
                    "Model backbone does not have freeze_encoder implemented."
                )
            state.model.model.backbone.freeze_encoder()
        self.encoder_is_frozen = True

    def batch_end(self, state: State, logger: Logger) -> None:
        current_time = state.timestamp.get(self.num_warmup_steps.unit).value
        if self.encoder_is_frozen and self.num_warmup_steps.value >= current_time:
            if hasattr(state.model, "module"):
                state.model.module.model.backbone.unfreeze_encoder()
            else:
                state.model.model.backbone.unfreeze_encoder()
            self.encoder_is_frozen = False
