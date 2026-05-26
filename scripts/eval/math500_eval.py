"""
MATH-500 evaluation script.

Evaluates model exact-match accuracy on HuggingFaceH4/MATH-500.
This script uses the same model loading and generation stack as the other
custom evaluation scripts in this repository.
"""

import json
import os
import re
import time
from typing import Any

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.utils import (
    count_parameters,
    format_number,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists


_TEXT_PATTERN = re.compile(r"\\text\{([^{}]*)\}")
_BOXED_PATTERN = re.compile(r"\\boxed\{")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_FINAL_ANSWER_TAG_PATTERN = re.compile(
    r"^\s*final_answer\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


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


def _extract_balanced_braces(text: str, start_idx: int) -> tuple[str | None, int]:
    depth = 0
    for idx in range(start_idx, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            if depth == 0:
                return text[start_idx:idx], idx + 1
            depth -= 1
    return None, len(text)


def _extract_boxed_answer(text: str) -> str | None:
    matches = list(_BOXED_PATTERN.finditer(text))
    if not matches:
        return None

    last = matches[-1]
    content, _ = _extract_balanced_braces(text, last.end())
    if content is None:
        return None
    return content.strip()


def _normalize_answer(text: str) -> str:
    if text is None:
        return ""

    s = text.strip()
    s = s.replace("\r", "\n")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

    boxed = _extract_boxed_answer(s)
    if boxed is not None:
        s = boxed

    s = s.replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\%", "%")

    prev = None
    while prev != s:
        prev = s
        s = _TEXT_PATTERN.sub(r"\1", s)

    s = _WHITESPACE_PATTERN.sub("", s)
    s = s.strip(" .")

    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = s[1:-1].strip()

    return s.lower()


def _extract_final_answer(generated_text: str) -> str:
    text = generated_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Prioritize boxed format \boxed{...} which is most robust.
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip()

    lowered = text.lower()
    triggers = ["final answer is", "answer is", "final answer:", "answer:"]
    best_pos = -1
    best_trigger = None
    for trigger in triggers:
        pos = lowered.rfind(trigger)
        if pos > best_pos:
            best_pos = pos
            best_trigger = trigger
    if best_trigger is not None and best_pos >= 0:
        candidate = text[best_pos + len(best_trigger):].strip()
        candidate = candidate.splitlines()[0].strip()
        return candidate.strip(" .")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1].strip(" .")


def _parse_simple_number(text: str) -> float | None:
    s = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        try:
            return float(s)
        except ValueError:
            return None

    frac_match = re.fullmatch(r"[-+]?\d+/[-+]?\d+", s)
    if frac_match:
        try:
            num, den = s.split("/")
            den_val = float(den)
            if den_val == 0:
                return None
            return float(num) / den_val
        except Exception:
            return None

    latex_frac_match = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", s)
    if latex_frac_match:
        try:
            num = float(latex_frac_match.group(1))
            den = float(latex_frac_match.group(2))
            if den == 0:
                return None
            return num / den
        except Exception:
            return None

    return None


def _answers_match(predicted: str, gold: str) -> bool:
    pred_norm = _normalize_answer(predicted)
    gold_norm = _normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_num = _parse_simple_number(pred_norm)
    gold_num = _parse_simple_number(gold_norm)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) <= 1e-9

    return False


def _build_prompt(
    problem: str,
    prompt_style: str,
    instruction_prefix: str,
    answer_format_instruction: str,
) -> str:
    prefix_blocks = []
    if prompt_style == "concise":
        prefix_blocks.append(instruction_prefix.strip())

    # Always include this format instruction when configured, even with
    # prompt_style="none", so extraction stays robust and consistent.
    if answer_format_instruction.strip():
        prefix_blocks.append(answer_format_instruction.strip())

    if len(prefix_blocks) == 0:
        return problem

    return "\n\n".join(prefix_blocks + [problem])


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="kodcode_eval_config",
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
    ckpt_file = cfg.get("ckpt_file", "best-rank0.pt")
    load_ema = cfg.get("load_ema_weights", True)

    if "fsdp" in pretrained_model_name_or_path:
        load_ema = False

    if fsspec_exists(os.path.join(pretrained_model_name_or_path, "config.yaml")):
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=pretrained_model_name_or_path,
            load_ema_weights=load_ema,
            ckpt_file=ckpt_file,
            device=device,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            revision=cfg.get("pretrained_model_revision", None),
        )

    model = model.to(device)
    model.eval()
    print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    dataset_name = cfg.get("math500_dataset_name", "HuggingFaceH4/MATH-500")
    split = cfg.get("math500_split", "test")
    num_samples = cfg.get("math500_num_samples", None)
    sampling_seed = cfg.get("math500_sampling_seed", 42)
    prompt_style = cfg.get("prompt_style", "none")
    instruction_prefix = cfg.get(
        "instruction_prefix",
        "Solve the following math problem and give only the final answer.",
    )
    use_answer_format_instruction = _coerce_bool(
        cfg.get("use_answer_format_instruction", True)
    )
    answer_format_instruction = cfg.get(
        "answer_format_instruction",
        (
            "After solving, output your final answer inside \\boxed{} like this: "
            "\\boxed{your_answer}"
        ),
    )
    if not use_answer_format_instruction:
        answer_format_instruction = ""
    is_instruction_model = _coerce_bool(cfg.get("is_instruction_model", True))
    use_chat_template = _coerce_bool(
        cfg.get("use_chat_template", is_instruction_model)
    )

    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if num_samples is not None and str(num_samples).lower() != "null":
        n = min(int(num_samples), len(ds))
        if n < len(ds):
            ds = ds.shuffle(seed=sampling_seed).select(range(n))

    print(f"Evaluating on {len(ds)} MATH-500 samples from {dataset_name}:{split}")

    gen_kwargs = {}
    if hasattr(cfg, "gen_kwargs"):
        gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
        if isinstance(gen_kwargs, DictConfig):
            gen_kwargs = OmegaConf.to_container(gen_kwargs, resolve=True)

    output_path = cfg.get("output_path", "./math500_output")
    os.makedirs(output_path, exist_ok=True)

    results_per_problem = []
    tputs = []
    total_correct = 0
    total_problems = 0
    total_generated_tokens = 0
    total_accepted_tokens = 0
    total_accepted_lengths = []
    total_accept_counts = 0
    total_draft_position_attempt_counts: list[int] = []
    total_draft_position_accept_counts: list[int] = []
    warmup = 5

    subject_stats: dict[str, dict[str, int]] = {}

    for idx in tqdm(range(len(ds)), desc="Evaluating MATH-500"):
        example = ds[idx]
        problem = example["problem"]
        gold_answer = example["answer"]
        subject = example.get("subject", "unknown")
        level = int(example.get("level", -1))
        unique_id = example.get("unique_id", str(idx))

        user_prompt = _build_prompt(
            problem,
            prompt_style,
            instruction_prefix,
            answer_format_instruction,
        )

        is_e2d = "E2D" in type(model).__name__ and "E2D2" not in type(model).__name__
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_prompt}]
            try:
                ctx = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                ctx = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            bos = tokenizer.bos_token or ""
            eos = tokenizer.eos_token or ""
            if is_e2d:
                ctx = f"{bos}{user_prompt}{eos}"
            else:
                ctx = f"{bos}{user_prompt}{eos}Answer:"

        prefix_tokens = tokenizer(
            ctx,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        sample_output = model.generate(
            inputs=prefix_tokens,
            disable_pbar=False,
            **gen_kwargs,
        )

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000
        else:
            elapsed_time_s = time.perf_counter() - start_time

        if (
            isinstance(sample_output, tuple)
            and len(sample_output) == 3
            and isinstance(sample_output[1], tuple)
            and isinstance(sample_output[2], tuple)
        ):
            sample, (generated_tokens, accepted_tokens), (
                accepted_lengths,
                accept_counts,
            ) = sample_output
            draft_position_stats = getattr(
                model,
                "_last_draft_position_acceptance",
                None,
            )
        else:
            sample = sample_output
            generated_tokens = int(sample.shape[-1] - prefix_tokens.shape[-1])
            accepted_tokens = generated_tokens
            accepted_lengths = [generated_tokens]
            accept_counts = 1
            draft_position_stats = None

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
            for pos_idx, val in enumerate(attempt_counts):
                total_draft_position_attempt_counts[pos_idx] += int(val)
            for pos_idx, val in enumerate(accept_counts_pos):
                total_draft_position_accept_counts[pos_idx] += int(val)

        total_generated_tokens += generated_tokens
        total_accepted_tokens += accepted_tokens
        total_accepted_lengths.extend(accepted_lengths)
        total_accept_counts += accept_counts

        num_new_tokens = int(sample.shape[-1] - prefix_tokens.shape[-1])
        tput = num_new_tokens / elapsed_time_s if elapsed_time_s > 0 else 0.0
        if idx >= warmup:
            tputs.append(tput)

        generated_text = tokenizer.decode(
            sample[0, prefix_tokens.shape[1]:],
            skip_special_tokens=True,
        )

        for stop_seq in [tokenizer.eos_token, "<|eot_id|>"]:
            if stop_seq:
                generated_text = generated_text.split(stop_seq)[0]

        predicted_answer = _extract_final_answer(generated_text)
        is_correct = _answers_match(predicted_answer, gold_answer)

        total_correct += int(is_correct)
        total_problems += 1

        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(is_correct)

        results_per_problem.append(
            {
                "idx": idx,
                "unique_id": unique_id,
                "subject": subject,
                "level": level,
                "prompt": ctx,
                "problem": problem,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "normalized_gold_answer": _normalize_answer(gold_answer),
                "normalized_predicted_answer": _normalize_answer(predicted_answer),
                "is_correct": is_correct,
                "generated_text": generated_text,
            }
        )

        print("=" * 20)
        print("Prompt:", ctx)
        print("Generated:", generated_text)
        print("Predicted answer:", predicted_answer)
        print("(Ground truth):", gold_answer)
        print("=" * 20, end="\n\n")

        accuracy = total_correct / total_problems
        if idx >= warmup and tputs:
            print(
                f"\n[{idx + 1}/{len(ds)}] "
                f"Accuracy: {total_correct}/{total_problems} = {accuracy:.2%} | "
                f"Thput: {np.mean(tputs):.2f} +/- {np.std(tputs):.2f} tok/s"
            )
        else:
            print(
                f"\n[{idx + 1}/{len(ds)}] "
                f"Accuracy: {total_correct}/{total_problems} = {accuracy:.2%} | "
                f"Thput: {tput:.2f} tok/s"
            )

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
        print(
            f"Total generated tokens: {total_generated_tokens}, "
            f"Total accepted tokens: {total_accepted_tokens}, "
            f"Acceptance rate: {acceptance_rate:.2%}"
        )
        print(f"Average accepted length: {avg_accepted_len:.2f}")
        if len(total_draft_position_attempt_counts) > 0:
            per_pos_strings = []
            for pos, (acc, att) in enumerate(
                zip(
                    total_draft_position_accept_counts,
                    total_draft_position_attempt_counts,
                ),
                start=1,
            ):
                rate = (acc / att) if att > 0 else 0.0
                per_pos_strings.append(f"p{pos}:{rate:.2%} ({acc}/{att})")
            print("Per-position acceptance rate: " + ", ".join(per_pos_strings))

    final_accuracy = total_correct / total_problems if total_problems > 0 else 0.0
    metrics: dict[str, Any] = {
        "dataset": dataset_name,
        "split": split,
        "prompt_style": prompt_style,
        "exact_match_accuracy": final_accuracy,
        "total_correct": total_correct,
        "total_problems": total_problems,
        "per_subject": {
            subject: {
                "accuracy": (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                ),
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for subject, stats in sorted(subject_stats.items())
        },
    }
    if tputs:
        metrics["throughput_mean_tok_s"] = float(np.mean(tputs))
        metrics["throughput_std_tok_s"] = float(np.std(tputs))
    if total_generated_tokens > 0:
        metrics["acceptance_rate"] = total_accepted_tokens / total_generated_tokens
    if total_accept_counts > 0:
        metrics["average_accepted_length"] = float(
            np.sum(total_accepted_lengths) / total_accept_counts
        )
    if len(total_draft_position_attempt_counts) > 0:
        metrics["per_position_acceptance"] = [
            {
                "position": pos,
                "accept_count": int(acc),
                "attempt_count": int(att),
                "acceptance_rate": float(acc / att) if att > 0 else 0.0,
            }
            for pos, (acc, att) in enumerate(
                zip(
                    total_draft_position_accept_counts,
                    total_draft_position_attempt_counts,
                ),
                start=1,
            )
        ]

    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results_per_problem, f, indent=2)

    if len(total_draft_position_attempt_counts) > 0:
        per_position_summary = [
            {
                "position": pos,
                "accept_count": int(acc),
                "attempt_count": int(att),
                "acceptance_rate": float(acc / att) if att > 0 else 0.0,
            }
            for pos, (acc, att) in enumerate(
                zip(
                    total_draft_position_accept_counts,
                    total_draft_position_attempt_counts,
                ),
                start=1,
            )
        ]
        with open(os.path.join(output_path, "draft_position_acceptance.json"), "w") as f:
            json.dump(per_position_summary, f, indent=2)
        with open(os.path.join(output_path, "draft_position_acceptance.txt"), "w") as f:
            for row in per_position_summary:
                f.write(
                    f"position={row['position']}, acceptance_rate={row['acceptance_rate']:.6f}, "
                    f"accept_count={row['accept_count']}, attempt_count={row['attempt_count']}\n"
                )

    print("\n" + "=" * 60)
    print("MATH-500 EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"Exact-match Accuracy: {final_accuracy:.2%} "
        f"({total_correct}/{total_problems})"
    )
    if tputs:
        print(f"Throughput: {np.mean(tputs):.2f} +/- {np.std(tputs):.2f} tok/s")
    if total_generated_tokens > 0:
        print(
            f"Acceptance rate: "
            f"{total_accepted_tokens / total_generated_tokens:.2%}"
        )
    if total_accept_counts > 0:
        print(
            f"Average accepted length: "
            f"{np.sum(total_accepted_lengths) / total_accept_counts:.2f}"
        )
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    register_useful_resolvers()
    main()