"""CPU checks for MATH-500 scoring and the shared generation/reporting path."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from lm_eval.api.instance import Instance
from omegaconf import OmegaConf
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from scripts.eval.harness_eval import LMEvalHarnessModel, main
from scripts.eval.math500_metrics import extract_boxed_answer, process_results


class Math500MetricsTest(unittest.TestCase):
    def test_nested_and_last_box(self):
        response = r"First \boxed{2}, corrected to $\boxed{\frac{1}{\sqrt{2}}}$."
        self.assertEqual(extract_boxed_answer(response), r"\frac{1}{\sqrt{2}}")

    def test_invalid_answers(self):
        for response in ["No answer", r"\boxed{}", r"\boxed{\frac{1}{2}"]:
            with self.subTest(response=response):
                self.assertIsNone(extract_boxed_answer(response))
                self.assertEqual(
                    process_results({"answer": "1"}, [response]), {"exact_match": 0}
                )

    def test_math_normalization(self):
        for gold, response in [
            (r"\frac{1}{2}", r"\boxed{0.5}"),
            (r"\frac{14}{3}", r"\boxed{\dfrac{14}{3}}"),
            (r"\left(3, \frac{\pi}{2}\right)", r"\boxed{(3,\frac{\pi}{2})}"),
            (r"3\sqrt{13}", r"\boxed{3\sqrt{13}}"),
            (r"\text{Evelyn}", r"\boxed{\text{Evelyn}}"),
            ("42", r"\fbox{42}"),
        ]:
            with self.subTest(gold=gold):
                self.assertEqual(
                    process_results({"answer": gold}, [response]), {"exact_match": 1}
                )
        self.assertEqual(
            process_results({"answer": "42"}, [r"\boxed{41}"]), {"exact_match": 0}
        )


class ChatTokenizer:
    bos_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.chat_kwargs = kwargs
        return "<user>" + messages[0]["content"] + "<assistant><think>\n\n</think>\n\n"

    def __call__(self, text, **kwargs):
        self.tokenize_kwargs = kwargs
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, tokens):
        return "".join(chr(int(token)) for token in tokens)


class HarnessMath500Test(unittest.TestCase):
    def run_generation(self, directory, doc, response, *, hf_model=None):
        tokenizer = ChatTokenizer()
        adapter = LMEvalHarnessModel.__new__(LMEvalHarnessModel)
        adapter._rank = 0
        adapter._world_size = 1
        adapter.device = torch.device("cpu")
        adapter.tokenizer = tokenizer
        adapter.generated_samples_output_path = directory
        adapter.gen_kwargs = {}
        adapter._is_hf_model = hf_model is not None
        adapter.is_instruction_model = True
        adapter.gsm8k_chat_template_kwargs = {"enable_thinking": False}
        adapter.throughput_run = False
        adapter.throughput_warmup = 0

        def generate(inputs, **kwargs):
            completion = torch.tensor(
                [[ord(char) for char in response + "<eos>ignored"]]
            )
            return (
                torch.cat([inputs, completion], dim=1),
                (8, 6),
                ([3, 3], 2),
                (0.2, 0.5),
            )

        adapter.model = SimpleNamespace(
            generate=generate,
            _last_draft_position_acceptance={
                "draft_lengths": [4, 4],
                "attempt_counts": [2, 2],
                "accept_counts": [2, 1],
            },
        )
        if hf_model is not None:
            adapter.model = hf_model
            adapter.gen_kwargs = {
                "generation_config": GenerationConfig(
                    do_sample=False,
                    max_new_tokens=4,
                    pad_token_id=0,
                ),
                "max_length": None,
                "max_new_tokens": 4,
                "stopping_criteria": None,
                "logits_processor": None,
            }
        question = doc.get("problem", doc.get("question"))
        request = Instance(
            request_type="generate_until",
            doc=doc,
            idx=0,
            arguments=(f"Question: {question}\nAnswer:", {"until": []}),
        )
        event = MagicMock()
        event.elapsed_time.return_value = 1000
        stdout = io.StringIO()
        with (
            patch("torch.cuda.Event", return_value=event),
            patch("torch.cuda.synchronize"),
            redirect_stdout(stdout),
        ):
            results = adapter.generate_until([request])
        return results, tokenizer, stdout.getvalue()

    def test_math500_prompt_scoring_and_reports(self):
        doc = {
            "problem": "Compute 1/2.",
            "solution": "reference",
            "answer": r"\frac{1}{2}",
            "unique_id": "test/1",
        }
        response = r"The result is $\boxed{\frac{1}{2}}$."
        with tempfile.TemporaryDirectory() as directory:
            results, tokenizer, log = self.run_generation(directory, doc, response)
            self.assertEqual(results, [response])
            self.assertEqual(
                tokenizer.messages,
                [
                    {
                        "role": "user",
                        "content": (
                            "Please reason step by step, and put your final answer "
                            r"within $\boxed{}$. Compute 1/2."
                        ),
                    }
                ],
            )
            self.assertEqual(
                tokenizer.chat_kwargs,
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            )
            self.assertEqual(tokenizer.tokenize_kwargs, {"add_special_tokens": False})
            for marker in [
                "Prefix:",
                "Generated:",
                "(Ground truth):",
                "Accuracy: 1/1 = 100.00%",
                "[TIME STATS]",
                "Total generated tokens: 8",
                "Acceptance rate: 75.00%",
                "Average accepted length: 3.00",
                "Running avg draft length: 4.00",
                "Per-position acceptance rate:",
                "Thput (tok/s):",
            ]:
                self.assertIn(marker, log)
            saved = json.loads((Path(directory) / "rank0.json").read_text())[0]
            self.assertEqual(saved["result"], response)
            self.assertEqual(saved["final_content"], response)
            self.assertEqual(saved["predicted_answer"], doc["answer"])
            self.assertTrue(saved["is_correct"])
            self.assertEqual(
                process_results(doc, results)["exact_match"], int(saved["is_correct"])
            )
            stats = json.loads(
                (Path(directory) / "draft_position_acceptance-rank0.json").read_text()
            )
            self.assertEqual(stats[1]["acceptance_rate"], 0.5)
            self.assertTrue(
                (Path(directory) / "draft_position_acceptance-rank0.txt").exists()
            )

    def test_gsm8k_scoring_stays_unchanged(self):
        doc = {"question": "Compute 21+21.", "answer": "21+21 = 42\n#### 42"}
        with tempfile.TemporaryDirectory() as directory:
            results, _, log = self.run_generation(
                directory, doc, r"The result is $\boxed{42}$."
            )
            self.assertEqual(results, ["The result is #### 42"])
            self.assertIn("Accuracy: 1/1 = 100.00%", log)

    def test_plain_qwen_generation_counts_new_tokens_and_ar_steps(self):
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=256,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=512,
                eos_token_id=None,
            )
        ).eval()
        doc = {"problem": "Compute 1/2.", "solution": "reference", "answer": "0.5"}
        with tempfile.TemporaryDirectory() as directory:
            results, tokenizer, log = self.run_generation(
                directory,
                doc,
                "",
                hf_model=model,
            )
            # The real HF generate() rejects unsupported kwargs. A prompt longer
            # than the budget must still produce four NEW tokens, not just one.
            self.assertEqual(len(results[0]), 4)
            self.assertIn("Total generated tokens: 4", log)
            self.assertIn("Acceptance rate: 100.00%", log)
            self.assertIn("Average accepted length: 1.00", log)
            self.assertFalse(tokenizer.chat_kwargs["enable_thinking"])

    def test_main_saves_harness_json_reports(self):
        results = {
            "results": {"math500": {"exact_match,none": 1.0}},
            "configs": {"math500": {"task": "math500"}},
            "versions": {"math500": 1},
            "n-shot": {"math500": 0},
            "higher_is_better": {"math500": {"exact_match": True}},
            "samples": {
                "math500": [
                    {
                        "doc_hash": "d",
                        "prompt_hash": "p",
                        "target_hash": "t",
                        "arguments": [["prompt", {"until": []}]],
                        "resps": [[r"\boxed{42}"]],
                        "filtered_resps": [r"\boxed{42}"],
                        "target": "42",
                    }
                ]
            },
        }
        adapter = SimpleNamespace(tokenizer=ChatTokenizer(), is_instruction_model=True)
        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.create(
                {
                    "seed": 1234,
                    "pretrained_model_name_or_path": "/checkpoints/test-model",
                    "output_path": directory,
                    "task": {"model": {"generated_samples_output_path": directory}},
                }
            )
            with (
                patch(
                    "scripts.eval.harness_eval.accelerate.Accelerator",
                    return_value=SimpleNamespace(num_processes=1),
                ),
                patch(
                    "scripts.eval.harness_eval.hydra.utils.instantiate",
                    return_value=adapter,
                ),
                patch(
                    "scripts.eval.harness_eval.hydra.utils.call", return_value=results
                ),
                redirect_stdout(io.StringIO()),
            ):
                main.__wrapped__(cfg)
            root = Path(directory)
            reports = list(root.rglob("results_*.json"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(
                json.loads(reports[0].read_text())["results"]["math500"][
                    "exact_match,none"
                ],
                1.0,
            )
            self.assertEqual(len(list(root.rglob("samples_math500_*.jsonl"))), 1)
            self.assertIn("math500", (root / "metrics.txt").read_text())


if __name__ == "__main__":
    unittest.main()
