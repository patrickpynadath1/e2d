"""Offline parity with the actual Tulu 3 Stage 1 code, plus resume and dispatch."""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset, load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from scripts import generate_tulu3_selfdistill as generate


REPO = Path(__file__).resolve().parents[1]
REFERENCE = (REPO / "bash_scripts/run_train_e2d_ultrachat.sh").read_text().split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def tiny_tokenizer():
    backend = Tokenizer(WordLevel({
        "[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "answer": 3, "user": 4,
        "assistant": 5, "system": 6, "short": 7, "long": 8, "prior": 9,
        "question": 10, "REPLACE": 11, "tool": 12,
    }, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        padding_side="left", model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' [EOS] ' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tokenizer


class TinyModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.tolist(), attention_mask.tolist(), kwargs))
        budget = kwargs["max_new_tokens"]
        # Depend on the original batch width. Regrouping after filtering changes
        # these answers and therefore fails the reference comparison.
        suffix = torch.full((len(input_ids), budget), 3, dtype=torch.long)
        if len(input_ids) > 1:
            suffix[0, -2:] = torch.tensor([kwargs["eos_token_id"], kwargs["pad_token_id"]])
        return torch.cat([input_ids, suffix], dim=1)


class ThreadPool:
    """Exercise the ordered, bounded dispatcher without requiring CUDA."""

    def __init__(self, workers):
        self.executor = ThreadPoolExecutor(workers)
        self._pool = [SimpleNamespace(exitcode=None) for _ in range(workers)]

    def apply_async(self, fn, args):
        future = self.executor.submit(fn, *args)

        def get(timeout):
            try:
                return future.result(timeout)
            except TimeoutError:
                raise mp.TimeoutError from None

        return SimpleNamespace(get=get)


class TuluGenerationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = tiny_tokenizer()
        examples = [
            {"messages": [{"role": "user", "content": "short " * (i + 1)},
                          {"role": "assistant", "content": "REPLACE"}]}
            for i in range(12)
        ]
        examples[0]["messages"] = []
        examples[2]["messages"][0]["content"] = "long " * 70
        examples[5]["messages"] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "REPLACE"},
        ]
        self.sources = {"train": Dataset.from_list(examples[:7]), "extra": Dataset.from_list(examples[7:])}
        self.args = SimpleNamespace(
            batch_size=3, max_length=24, gen_max_length=40, prompt_max_length=22,
            max_samples=0, eval_ratio=0.25, seed=42, progress_every_batches=0,
            jsonl_stem=generate.JSONL_STEM,
        )

    def reference(self, root):
        root.mkdir()
        env = {
            "MODEL_NAME_OR_PATH": "tiny", "DISTILL_SOURCE_NAME": "tiny-tulu",
            "DISTILL_SOURCE_SPLITS": "train extra", "DISTILL_SOURCE_FORMAT": "messages",
            "MAX_SEQ_LEN": str(self.args.max_length), "GEN_MAX_SEQ_LEN": str(self.args.gen_max_length),
            "PROMPT_MAX_SEQ_LEN": str(self.args.prompt_max_length),
            "DISTILL_MAX_SAMPLES": str(self.args.max_samples), "EVAL_RATIO": str(self.args.eval_ratio),
            "SPLIT_SEED": str(self.args.seed), "BATCH_GEN_SIZE": str(self.args.batch_size),
            "PROGRESS_EVERY_BATCHES": "0", "EMPTY_CACHE_EVERY_BATCHES": "20", "FORCE_REGENERATE": "false",
            "DISTILL_RAW_FILE": str(root / f"{self.args.jsonl_stem}.jsonl"),
        }
        for split in ("train", "eval"):
            env[f"DISTILL_{split.upper()}_JSONL"] = str(root / f"{self.args.jsonl_stem}_{split}.jsonl")
            env[f"DISTILL_{split.upper()}_PATH"] = str(root / f"{split}_preprocessed")
        model = TinyModel()

        def source_or_json(name, *args, split, **kwargs):
            if name == "tiny-tulu":
                return self.sources[split]
            return load_dataset(name, *args, split=split, **kwargs)

        with patch.dict(os.environ, env), patch("torch.cuda.is_available", return_value=False), \
                patch("datasets.load_dataset", side_effect=source_or_json), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
            exec(compile(REFERENCE, "reference_tulu_stage1", "exec"), {})
        return model.calls

    def run_new(self, root, workers):
        settings = {"jsonl_stem": self.args.jsonl_stem}
        state = generate.restore_output(root, settings)
        model = TinyModel()
        tasks = generate.iter_batches(self.sources, self.tokenizer, self.args.batch_size)
        pool = ThreadPool(workers)
        try:
            with patch.object(generate, "worker_generate", side_effect=lambda task: generate.generate_batch(
                task, tiny_tokenizer(), model, self.args,
            )):
                generate.commit_results(
                    generate.ordered_results(pool, tasks, workers), root / f"{self.args.jsonl_stem}.jsonl",
                    root / "selfdistill_metadata.json", settings, state, self.args,
                )
        finally:
            pool.executor.shutdown()
        generate.write_dataset(root, self.tokenizer, self.args, state["generated_rows"])
        return model.calls

    def test_reference_parity_across_worker_counts_and_global_limit(self):
        for limit in (0, 5):
            self.args.max_samples = limit
            reference = self.root / f"reference_{limit}"
            expected_calls = self.reference(reference)
            for workers in (1, 2, 8):
                with self.subTest(limit=limit, workers=workers):
                    output = self.root / f"output_{limit}_{workers}"
                    actual_calls = self.run_new(output, workers)
                    # Unlimited runs must make exactly the same generation calls.
                    if not limit:
                        self.assertCountEqual(actual_calls, expected_calls)
                    for suffix in ("", "_train", "_eval"):
                        name = f"{self.args.jsonl_stem}{suffix}.jsonl"
                        self.assertEqual((output / name).read_bytes(), (reference / name).read_bytes())
                    for split in ("train", "eval"):
                        self.assertEqual(list(load_from_disk(str(output / f"{split}_preprocessed"))),
                                         list(load_from_disk(str(reference / f"{split}_preprocessed"))))

    def test_prompt_edge_cases_match_reference_functions(self):
        tree = ast.parse(REFERENCE)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"normalize_content", "extract_prompt_messages"}]
        namespace = {"Any": object, "json": json, "source_format": "messages"}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "reference", "exec"), namespace)
        examples = [
            {}, {"messages": "bad json"}, {"messages": [{"role": "assistant", "content": "reference"}]},
            {"messages": [{"role": "user", "content": "unanswered"}]},
            {"messages": [None, {"role": " USER ", "content": ["hello", {"text": "world"}]},
                          {"role": "assistant", "content": "reference"}]},
            {"messages": json.dumps([{"role": "tool", "content": "result"},
                                     {"role": "assistant", "content": "reference"}])},
        ]
        for source in self.sources.values():
            examples.extend(source)
        for example in examples:
            self.assertEqual(generate.extract_prompt_messages(example), namespace["extract_prompt_messages"](example))
        messages = generate.extract_prompt_messages(self.sources["train"][5])
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertNotIn("REPLACE", str(messages))

    def test_padding_semantics(self):
        self.assertEqual(generate.strip_batched_generation_padding([3, 1, 2, 1], self.tokenizer), [3, 2])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.assertEqual(generate.strip_batched_generation_padding([3, 2, 2, 2], self.tokenizer), [3, 2])

    def test_resume_after_partial_write_preserves_batch_boundaries(self):
        settings = {"jsonl_stem": self.args.jsonl_stem}
        root = self.root / "resume"
        state = generate.restore_output(root, settings)
        raw = root / f"{self.args.jsonl_stem}.jsonl"
        metadata = root / "selfdistill_metadata.json"
        tasks = list(generate.iter_batches(self.sources, self.tokenizer, self.args.batch_size))
        results = [generate.generate_batch(task, self.tokenizer, TinyModel(), self.args) for task in tasks]

        def interrupted():
            yield results[0]
            raise RuntimeError("worker failed")

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            generate.commit_results(interrupted(), raw, metadata, settings, state, self.args)
        committed = raw.read_bytes()
        with raw.open("ab") as output:
            output.write(b'{"uncommitted":')
        state = generate.restore_output(root, settings)
        self.assertEqual(raw.read_bytes(), committed)
        self.assertEqual(state["next_batch"], 1)
        generate.commit_results(iter(results[1:]), raw, metadata, settings, state, self.args)
        expected = [row for _, rows, _ in results for row in rows]
        self.assertEqual([json.loads(line) for line in raw.read_text().splitlines()], expected)
        with self.assertRaisesRegex(ValueError, "settings/source differ"):
            generate.restore_output(root, {**settings, "batch_size": 7})
        raw.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "shorter"):
            generate.restore_output(root, settings)

    def test_worker_death_does_not_hang(self):
        pool = ThreadPool(1)
        pool._pool[0].exitcode = -9
        try:
            with patch.object(generate, "worker_generate", return_value=(0, [], 0)):
                with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                    list(generate.ordered_results(pool, [(0, "train", ["prompt"])], 1))
        finally:
            pool.executor.shutdown()


    @unittest.skipUnless(os.environ.get("RUN_TULU_GPU_TESTS") == "1", "opt-in two-GPU smoke test")
    def test_real_gpu_workers_with_tiny_local_model(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model_path = self.root / "tiny_model"
        torch.manual_seed(42)
        Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, eos_token_id=2, pad_token_id=1,
        )).save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        source_path = self.root / "source"
        source_path.mkdir()
        for split, source in self.sources.items():
            filename = "validation.jsonl" if split == "extra" else f"{split}.jsonl"
            source.to_json(str(source_path / filename))
        outputs = []
        for workers in (1, 2):
            root = self.root / f"gpu_{workers}"
            command = [
                sys.executable, str(REPO / "scripts/generate_tulu3_selfdistill.py"),
                "--data-root", str(root), "--model-name-or-path", str(model_path),
                "--source-dataset", str(source_path), "--source-splits", "train", "validation",
                "--num-shards", str(workers), "--batch-size", "3", "--dtype", "float32",
                "--max-length", "24", "--gen-max-length", "40", "--prompt-max-length", "22",
                "--eval-ratio", "0.25", "--local-files-only", "--max-samples", "5",
            ]
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata_path = root / "selfdistill_metadata.json"
            metadata = metadata_path.read_bytes()
            self.assertEqual(json.loads(metadata)["generated_rows"], 5)
            rerun = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=30)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertIn("Already complete", rerun.stdout)
            self.assertEqual(metadata_path.read_bytes(), metadata)
            logs = list(root.with_name(root.name + ".shards").glob("shard_*.log"))
            self.assertEqual(len(logs), workers)
            for log in logs:
                self.assertIn("GPU", log.read_text())
            outputs.append(root)
        for suffix in ("", "_train", "_eval"):
            name = f"{self.args.jsonl_stem}{suffix}.jsonl"
            self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())
        for split in ("train", "eval"):
            self.assertEqual(list(load_from_disk(str(outputs[0] / f"{split}_preprocessed"))),
                             list(load_from_disk(str(outputs[1] / f"{split}_preprocessed"))))

    def test_shell_defaults_and_overrides_from_another_directory(self):
        repo = self.root / "repo"
        (repo / "bash_scripts").mkdir(parents=True)
        (repo / "setup_env.sh").write_text("")
        launcher = repo / "bash_scripts/generate_tulu3_selfdistill_advanced.sh"
        shutil.copyfile(REPO / "bash_scripts" / launcher.name, launcher)
        probe = repo / "python_probe"
        probe.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        probe.chmod(0o755)
        env = {"PATH": os.environ["PATH"], "PYTHON": str(probe)}
        for overrides in ({}, {"PER_DEVICE_BATCH_SIZE": "4", "DISTILL_DATA_ROOT": "/tmp/path with spaces"}):
            result = subprocess.run(["bash", str(launcher), "--local-files-only"], cwd=self.root,
                                    env={**env, **overrides}, text=True, capture_output=True, check=True)
            args = json.loads(result.stdout)
            self.assertEqual(args[0], "scripts/generate_tulu3_selfdistill.py")
            for flag, value in (("--source-dataset", "allenai/tulu-3-sft-mixture"),
                                ("--source-splits", "train"), ("--max-length", "4096"),
                                ("--batch-size", overrides.get("PER_DEVICE_BATCH_SIZE", "32")),
                                ("--num-shards", "0")):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertEqual(args[-1], "--local-files-only")
        result = subprocess.run(["bash", str(launcher)], env={**env, "PER_DEVICE_BATCH_SIZE": "0"},
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
