"""Boxed-answer scoring for MATH-500 using lm-eval's MATH normalization."""

from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string


def extract_boxed_answer(response: str) -> str | None:
    """Extract the last box, retaining nested LaTeX braces (e.g. fractions)."""
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return None
    if boxed.startswith("\\boxed "):
        answer = boxed[len("\\boxed ") :]
    else:
        # Both \\boxed{...} and \\fbox{...} use the same brace-delimited content.
        answer = boxed[boxed.index("{") + 1 : -1]
    return answer.strip() or None


def boxed_answer_accuracy(predicted_answer: str | None, gold: str) -> bool:
    """Use the harness's normalized exact match, not GSM8K numeric matching."""
    return predicted_answer is not None and is_equiv(predicted_answer, gold)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Use the same scorer for the running log and aggregate harness metrics."""
    return {
        "exact_match": int(
            boxed_answer_accuracy(extract_boxed_answer(results[0]), doc["answer"])
        )
    }
