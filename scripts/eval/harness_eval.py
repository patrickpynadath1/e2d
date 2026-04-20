"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import re
import sys
from typing import Any, List, Tuple

import accelerate
import hydra
import numpy as np
import torch
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
)

from datasets import Dataset
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    format_number,
    count_parameters,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs


class LMEvalHarnessModel(LM):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        generated_samples_output_path: str,
        tokenizer: PreTrainedTokenizer,
        pretrained_model_revision: str | None = None,
        load_ema_weights: bool = False,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        gen_kwargs: Any | None = None,
        accelerator: accelerate.Accelerator | None = None,
        throughput_run: bool = False,
        throughput_samples: int = 100,
        throughput_warmup: int = 100,
        is_instruction_model: bool | None = None,
        use_chat_template_for_gsm8k: bool | None = None,
        enable_thinking_for_gsm8k: bool | None = None,
        strip_thinking_for_gsm8k: bool | None = None,
        model_config_overrides: dict[str, Any] | None = None,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Path to ckpt dir or HF model repo.
            generated_samples_output_path (str): Path to generated samples dir.
            tokenizer (str): Tokenizer name or path.
            pretrained_model_revision (Optional[str]): Revision (e.g., commit id)
                passed to `.from_pretrained` model instantiation.
            load_ema_weights (bool): Whether to load ema weights (for local ckpts).
            ckpt_file (str): Name of ckpt file (for local ckpts).
            gen_kwargs (dict): Generator kwargs.
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
            throughput_run (bool): Whether to run the evaluation throughput.
            is_instruction_model (bool | None): Whether this is an instruction-tuned
                checkpoint. If None, infer from model path (ultrachat => True).
            use_chat_template_for_gsm8k (bool | None): Whether to format GSM8K as
                chat turns. If None, follow `is_instruction_model`.
            enable_thinking_for_gsm8k (bool | None): Whether to request Qwen3
                thinking mode when using chat template. If None, follow
                `is_instruction_model`.
            strip_thinking_for_gsm8k (bool | None): Whether to split <think>
                content from final content before metric extraction. If None,
                follow `is_instruction_model`.
            model_config_overrides (dict[str, Any]): Model config overrides.
        """
        if "fsdp" in pretrained_model_name_or_path:
            load_ema_weights = False
        # E2D and AR trained on GSM8K still used ema
        if ("e2d" in pretrained_model_name_or_path and "e2d2" not in pretrained_model_name_or_path) and "gsm8k" in pretrained_model_name_or_path and "fsdp" in pretrained_model_name_or_path:
            load_ema_weights = True
        super().__init__()
        self.generated_samples_output_path = generated_samples_output_path
        if not fsspec_exists(self.generated_samples_output_path):
            fsspec_mkdirs(self.generated_samples_output_path)
        self.accelerator = accelerator
        if self.accelerator is not None:
            device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rank = 0
            self._world_size = 1
        self.device = torch.device(f"{device}")

        model_config_overrides = (
            {} if model_config_overrides is None else model_config_overrides
        )
        if fsspec_exists(os.path.join(pretrained_model_name_or_path, "config.yaml")):
            model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=pretrained_model_name_or_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
                device=self.device,
                **model_config_overrides,
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
            except:  # Model not compatible with CausalLM
                model = AutoModelForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
        self.model = model.to(self.device)
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
        self.model.eval()
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        # print tokenizer name
        print(f"Using tokenizer: {self.tokenizer.name_or_path}")
        self.gen_kwargs = gen_kwargs
        self.throughput_run = throughput_run
        self.throughput_warmup = throughput_warmup
        self.throughput_samples = throughput_samples
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off", ""}:
                    return False
            return bool(value)

        if is_instruction_model is None:
            is_instruction_model = "ultrachat" in pretrained_model_name_or_path.lower()
        self.is_instruction_model = _coerce_bool(is_instruction_model)

        if use_chat_template_for_gsm8k is None:
            use_chat_template_for_gsm8k = self.is_instruction_model
        if enable_thinking_for_gsm8k is None:
            enable_thinking_for_gsm8k = self.is_instruction_model
        if strip_thinking_for_gsm8k is None:
            strip_thinking_for_gsm8k = self.is_instruction_model
        self.use_chat_template_for_gsm8k = _coerce_bool(use_chat_template_for_gsm8k)
        self.enable_thinking_for_gsm8k = _coerce_bool(enable_thinking_for_gsm8k)
        self.strip_thinking_for_gsm8k = _coerce_bool(strip_thinking_for_gsm8k)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _apply_chat_template_for_gsm8k(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            if self.enable_thinking_for_gsm8k:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    # enable_thinking=True,
                    enable_thinking=False # disable thinking because greedy decoding requires non-thinking mode
                )
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        bos = self.tokenizer.bos_token or ""
        return f"{bos}{user_prompt}"

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    def generate_until(self, requests, **generation_kwargs):
        # Detect task type from the first request
        is_humaneval = (
            len(requests) > 0
            and hasattr(requests[0], "doc")
            and "entry_point" in requests[0].doc
        )
        is_mbpp = (
            len(requests) > 0
            and hasattr(requests[0], "doc")
            and (
                "test_list" in requests[0].doc
                or str(requests[0].doc.get("task_id", "")).lower().startswith("mbpp")
            )
        )
        is_code_eval = is_humaneval or is_mbpp
        is_gsm8k = (
            len(requests) > 0
            and hasattr(requests[0], "doc")
            and "answer" in requests[0].doc
            and not is_code_eval
        )

        # TODO: Move this to utils file / perhaps use chat template
        def _tokenize_gsm8k(
            e,
            prefix_text: str = (
                "Please reason step by step, and put your final answer within "
                + "$\\boxed{}$."
            ),
        ):
            ctx = e["prefix"]
            ctx = re.sub(
                r"^####\s*(\d+)\s*$",
                r"$\\boxed{\1}$" + (self.tokenizer.eos_token or ""),
                ctx,
                flags=re.MULTILINE,
            )

            if self.is_instruction_model:
                user_prompt = ctx.replace("Question: ", f"{prefix_text} ", 1)
                if user_prompt == ctx:
                    user_prompt = f"{prefix_text} {ctx}"
                user_prompt = re.sub(r"\nAnswer:\s*$", "", user_prompt).strip()
                ctx = self._apply_chat_template_for_gsm8k(user_prompt)
            else:
                bos = self.tokenizer.bos_token or ""
                eos = self.tokenizer.eos_token or ""
                ctx = ctx.replace("Question: ", f"{bos}{prefix_text} ")
                if (
                    "E2D" in type(self.model).__name__
                    and "E2D2" not in type(self.model).__name__
                ):
                    ctx = ctx.replace("\nAnswer:", f"{eos}")
                else:
                    ctx = ctx.replace("\nAnswer:", f"{eos}Answer:")
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        def _tokenize_humaneval(e):
            ctx = e["prefix"]
            # For HumanEval, the prompt is already the function signature + docstring
            # Prepend BOS token
            ctx = self.tokenizer.bos_token + ctx
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        def _tokenize_default(e):
            ctx = e["prefix"]
            ctx = self.tokenizer.bos_token + ctx
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        if is_code_eval:
            ds = ds.map(_tokenize_humaneval)
        elif is_gsm8k:
            ds = ds.map(_tokenize_gsm8k)
        else:
            ds = ds.map(_tokenize_default)
        ds = ds.with_format("torch")
        res = []
        res_for_json = []
        correct, total = 0, 0
        tputs = []
        total_generated_tokens = 0
        total_accepted_tokens = 0
        total_accepted_lengths = []
        total_accept_counts = 0
        total_draft_position_attempt_counts: List[int] = []
        total_draft_position_accept_counts: List[int] = []
        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            if (
                self.throughput_run
                and i >= self.throughput_samples + self.throughput_warmup
            ):
                tputs_path = (
                    f"{self.generated_samples_output_path}/throughput-rank{self.rank}"
                )
                with open(f"{tputs_path}.json", "w") as f:
                    json.dump(
                        {
                            "throughput_mean": np.mean(tputs),
                            "throughput_std": np.std(tputs),
                            "throughput_all": tputs,
                        },
                        f,  # type: ignore
                        indent=2,
                    )
                sys.exit(0)
            if self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_event, end_event = None, None
            sample_output = self.model.generate(
                inputs=elem["prefix"][None, ...].to(self.device),
                disable_pbar=(self.rank != 0),
                # tokenizer=self.tokenizer,  # Uncomment for debugging
                **self.gen_kwargs,
            )

            if (
                isinstance(sample_output, tuple)
                and len(sample_output) == 3
                and isinstance(sample_output[1], tuple)
                and isinstance(sample_output[2], tuple)
            ):
                sample, (generated_tokens, accepted_tokens), (accepted_lengths, accept_counts) = sample_output
                draft_position_stats = getattr(self.model, "_last_draft_position_acceptance", None)
            else:
                sample = sample_output
                generated_tokens = int(sample.shape[-1] - elem["prefix"].numel())
                accepted_tokens = generated_tokens
                accepted_lengths = [generated_tokens]
                accept_counts = 1
                draft_position_stats = None
            total_generated_tokens += generated_tokens
            total_accepted_tokens += accepted_tokens
            total_accepted_lengths.extend(accepted_lengths)
            total_accept_counts += accept_counts
            if draft_position_stats is not None:
                attempt_counts = draft_position_stats.get("attempt_counts", [])
                accept_counts_pos = draft_position_stats.get("accept_counts", [])
                max_len = max(
                    len(total_draft_position_attempt_counts),
                    len(attempt_counts),
                    len(total_draft_position_accept_counts),
                    len(accept_counts_pos),
                )
                if len(total_draft_position_attempt_counts) < max_len:
                    total_draft_position_attempt_counts.extend(
                        [0] * (max_len - len(total_draft_position_attempt_counts))
                    )
                if len(total_draft_position_accept_counts) < max_len:
                    total_draft_position_accept_counts.extend(
                        [0] * (max_len - len(total_draft_position_accept_counts))
                    )
                for idx, val in enumerate(attempt_counts):
                    total_draft_position_attempt_counts[idx] += int(val)
                for idx, val in enumerate(accept_counts_pos):
                    total_draft_position_accept_counts[idx] += int(val)
            if self.rank == 0:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_s = start_event.elapsed_time(end_event) / 1000
                tput = (sample.numel() - elem["prefix"].numel()) / elapsed_time_s
                if i >= self.throughput_warmup:
                    tputs.append(tput)
            result = self.tokenizer.decode(sample[0, len(elem["prefix"]) :])
            for until in elem["target"]["until"] + [
                "<|eot_id|>",
                self.tokenizer.eos_token,
            ]:
                result = result.split(until)[0]

            if is_code_eval:
                # For HumanEval: result is code completion after the prompt
                # The lm_eval harness filter will prepend the prompt back
                if self.rank == 0:
                    print("=" * 20)
                    print("prompt: ", elem["prefix_text"])
                    print("generated: ", result)
                    print("=" * 20, end="\n\n")
                res.append(result)
                total += 1
                res_for_json.append(
                    {
                        "task_id": requests[i].doc.get(
                            "task_id", f"CodeEval/{i}"
                        ),
                        "prefix": elem["prefix_text"],
                        "result": result,
                    }
                )
            else:
                # GSM8K / default: extract \boxed{} answer
                raw_result = result
                predicted_ans = None
                if "boxed{" in result:
                    predicted_ans = result.split("boxed{")[1].split("}")[0]
                    result = result.split("boxed{")[0] + "#### " + predicted_ans
                    result = result.replace("$\\", "")
                if self.rank == 0:
                    print("=" * 20)
                    print("Prefix:", elem["prefix_text"])
                    print("Generated:", result)
                    print("(Ground truth):", requests[i].doc["answer"])
                    print("=" * 20, end="\n\n")
                res.append(result)

                # log accuracy
                ground_truth_ans = requests[i].doc["answer"].split("### ")[1]
                if predicted_ans is not None and ground_truth_ans == predicted_ans:
                    correct += 1
                total += 1
                res_for_json.append(
                    (
                        {
                            "prefix": elem["prefix_text"],
                            "result": result,
                            "final_content": raw_result,
                        }
                        if is_gsm8k and self.is_instruction_model
                        else {
                            "prefix": elem["prefix_text"],
                            "result": result,
                        }
                    )
                )
            # torch.cuda.empty_cache()
            if self.rank == 0:
                if is_gsm8k:
                    print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")
                else:
                    print(f"\nCompleted: {total}/{len(ds)}\n")
                acceptance_rate = (
                    total_accepted_tokens / total_generated_tokens
                    if total_generated_tokens > 0
                    else 0.0
                )
                avg_accepted_len = (
                    np.sum(total_accepted_lengths) / total_accept_counts
                    if total_accept_counts > 0
                    else 0.0
                )
                print(f"Total generated tokens: {total_generated_tokens}, Total accepted tokens: {total_accepted_tokens}, Acceptance rate: {acceptance_rate:.2%}")
                print(f"Average accepted length: {avg_accepted_len:.2f}")
                if len(total_draft_position_attempt_counts) > 0:
                    per_pos_strings = []
                    for pos, (acc, att) in enumerate(
                        zip(total_draft_position_accept_counts, total_draft_position_attempt_counts),
                        start=1,
                    ):
                        rate = (acc / att) if att > 0 else 0.0
                        per_pos_strings.append(f"p{pos}:{rate:.2%} ({acc}/{att})")
                    print("Per-position acceptance rate: " + ", ".join(per_pos_strings))
                if i >= self.throughput_warmup:
                    print(
                        f"Thput (tok/s): {np.mean(tputs):0.2f} +/- {np.std(tputs):0.2f}"
                    )
                else:
                    print(f"Thput (tok/s): {tput:0.2f}")

        if self.rank == 0 and len(total_draft_position_attempt_counts) > 0:
            per_position_summary = []
            for pos, (acc, att) in enumerate(
                zip(total_draft_position_accept_counts, total_draft_position_attempt_counts),
                start=1,
            ):
                per_position_summary.append(
                    {
                        "position": pos,
                        "accept_count": int(acc),
                        "attempt_count": int(att),
                        "acceptance_rate": float(acc / att) if att > 0 else 0.0,
                    }
                )
            draft_position_metrics_path = (
                f"{self.generated_samples_output_path}/draft_position_acceptance-rank{self.rank}.json"
            )
            with open(draft_position_metrics_path, "w") as f:
                json.dump(per_position_summary, f, indent=2)

            draft_position_metrics_txt_path = (
                f"{self.generated_samples_output_path}/draft_position_acceptance-rank{self.rank}.txt"
            )
            with open(draft_position_metrics_txt_path, "w") as f:
                for row in per_position_summary:
                    f.write(
                        f"position={row['position']}, acceptance_rate={row['acceptance_rate']:.6f}, "
                        f"accept_count={row['accept_count']}, attempt_count={row['attempt_count']}\n"
                    )

        samples_path = f"{self.generated_samples_output_path}/rank{self.rank}"
        with open(f"{samples_path}.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        print(f"RANK {self.rank} completed!")
        return res


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    accelerator = accelerate.Accelerator()
    accelerator = accelerate.Accelerator() if accelerator.num_processes > 1 else None
    set_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.task.model, accelerator=accelerator)
    results = hydra.utils.call(cfg.task, model=model)
    if results is not None and (
        accelerator is None or accelerator.local_process_index == 0
    ):
        samples = results.pop("samples")
        evaluation_tracker = EvaluationTracker(output_path=cfg.output_path)
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=samples[task_name]
            )
        print(make_table(results))
        metrics_f = f"{cfg.task.model.generated_samples_output_path}/metrics.txt"
        with open(metrics_f, "w") as f:
            f.write(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    register_useful_resolvers()
    main()
