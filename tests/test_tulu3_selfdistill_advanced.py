"""Dataset parity for controlled vLLM outputs, plus resume and GPU dispatch."""

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


class TinyVLLM:
    """Return the same controlled completions as TinyModel, via vLLM's API."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, *, sampling_params, use_tqdm):
        params = sampling_params
        assert params.n == 1 and params.temperature == 0.0
        assert params.ignore_eos and params.stop_token_ids == [2]
        assert not params.detokenize and not params.skip_special_tokens
        assert not use_tqdm
        ids = [prompt["prompt_token_ids"] for prompt in prompts]
        self.calls.append((ids, params.max_tokens))
        results = []
        for index in range(len(prompts)):
            suffix = [3] * params.max_tokens
            if len(prompts) > 1 and index == 0:
                suffix[-2:] = [2]  # EOS is returned, but no batch padding.
            results.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=suffix)]))
        return results


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


class UnpicklableWorkerError(RuntimeError):
    """Reproduce compiler exceptions that retain live Python frames."""

    def __init__(self, message):
        super().__init__(message)
        self.frame = sys._getframe()


def fail_with_unpicklable_error(*args, **kwargs):
    raise UnpicklableWorkerError("original engine failure")


def init_failing_worker(slots, work_root, stage):
    args = SimpleNamespace(
        local_files_only=False, model_name_or_path="unused-model",
        attn_implementation=None, dtype="bfloat16", gen_max_length=40,
        batch_size=3, gpu_memory_utilization=0.9, seed=42, enforce_eager=False,
    )
    generate.init_worker(slots, work_root, args)
    sys.modules["vllm"] = SimpleNamespace(LLM=fail_with_unpicklable_error)
    generate.load_tokenizer = lambda args: None
    if stage == "generation":
        generate.WORKER.update(model=object(), tokenizer=None)
        generate.generate_batch = fail_with_unpicklable_error


class TuluGenerationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = tiny_tokenizer()
        # Offline tests need no vLLM installation or GPU; the opt-in subprocess
        # smoke test below imports the real package and exercises its engine.
        self.vllm = SimpleNamespace(SamplingParams=SimpleNamespace)
        previous_vllm = sys.modules.get("vllm")
        sys.modules["vllm"] = self.vllm

        def restore_vllm():
            if previous_vllm is None:
                sys.modules.pop("vllm", None)
            else:
                sys.modules["vllm"] = previous_vllm

        self.addCleanup(restore_vllm)
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
        model = TinyVLLM()
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
                        expected_inputs = [
                            ([[token for token, active in zip(row, mask) if active]
                              for row, mask in zip(ids, masks)], kwargs["max_new_tokens"])
                            for ids, masks, kwargs in expected_calls
                        ]
                        self.assertCountEqual(actual_calls, expected_inputs)
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
        results = [generate.generate_batch(task, self.tokenizer, TinyVLLM(), self.args) for task in tasks]

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

    def test_prompt_filtering_truncation_and_shared_budget(self):
        from tokenizers.processors import TemplateProcessing

        # The filtering pass excludes special tokens, but generation historically
        # used the tokenizer default (including BOS/EOS). Preserve that difference.
        self.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="[EOS] $A", special_tokens=[("[EOS]", 2)],
        )
        self.args.gen_max_length = 5
        self.args.prompt_max_length = 8
        model = TinyVLLM()
        result = generate.generate_batch(
            (7, "extra", ["short", "long " * 7, "long " * 9]),
            self.tokenizer, model, self.args,
        )
        self.assertEqual(model.calls, [([[2, 7], [2, 8, 8, 8]], 1)])
        self.assertEqual(result[0], 7)
        self.assertEqual(result[2], 1)
        self.assertEqual(result[1][0]["prompt"], "short")
        self.assertEqual(result[1][1]["prompt"], "long " * 7)
        self.assertEqual(result[1][0]["completion"], "[EOS]")
        self.assertEqual(result[1][1]["completion"], "answer")
        self.assertEqual(generate.generate_batch(
            (8, "extra", ["long " * 9]), self.tokenizer, model, self.args,
        ), (8, [], 1))
        self.assertEqual(len(model.calls), 1)

    def test_worker_reuses_one_engine_and_resolves_cached_model(self):
        from unittest.mock import Mock

        args = SimpleNamespace(
            **vars(self.args), model_name_or_path="cached/tiny", local_files_only=True,
            dtype="bfloat16", gpu_memory_utilization=0.6, attn_implementation=None,
            empty_cache_every_batches=2, enforce_eager=False,
        )
        model = TinyVLLM()
        constructor = Mock(return_value=model)
        self.vllm.LLM = constructor
        task = (0, "train", ["short"])
        with patch.dict(generate.WORKER, {"args": args, "batches": 0}, clear=True), \
                patch.object(generate, "load_tokenizer", return_value=self.tokenizer), \
                patch("huggingface_hub.snapshot_download", return_value=str(self.root)) as snapshot, \
                patch("torch.cuda.empty_cache") as empty_cache:
            generate.worker_generate(task)
            generate.worker_generate(task)
        constructor.assert_called_once_with(
            model=str(self.root), dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=1, distributed_executor_backend="uni",
            max_model_len=40, max_num_seqs=3, gpu_memory_utilization=0.6, seed=42,
            skip_tokenizer_init=True, generation_config="vllm", enforce_eager=False,
        )
        snapshot.assert_called_once_with("cached/tiny", local_files_only=True)
        empty_cache.assert_called_once()
        self.assertEqual(len(model.calls), 2)

    def test_eager_mode_is_opt_in_and_forwarded_to_vllm(self):
        from unittest.mock import Mock

        command = ["generate", "--data-root", str(self.root)]
        with patch.object(sys, "argv", command):
            self.assertFalse(generate.parse_args().enforce_eager)
        with patch.object(sys, "argv", [*command, "--enforce-eager"]):
            args = generate.parse_args()
        constructor = Mock(return_value=TinyVLLM())
        self.vllm.LLM = constructor
        with patch.dict(generate.WORKER, {"args": args, "batches": 0}, clear=True), \
                patch.object(generate, "load_tokenizer", return_value=self.tokenizer):
            generate.worker_generate((0, "train", ["short"]))
        self.assertTrue(constructor.call_args.kwargs["enforce_eager"])

    def test_worker_binds_gpu_before_loading_engine(self):
        from unittest.mock import Mock

        slots = Mock()
        slots.get.return_value = (1, "GPU-selected-uuid")
        args = SimpleNamespace(local_files_only=True)
        with patch.dict(os.environ, {}, clear=True), patch.dict(generate.WORKER, {}, clear=True), \
                patch("os.dup2"), patch("signal.signal"):
            generate.init_worker(slots, str(self.root), args)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-selected-uuid")
            self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
            self.assertEqual(os.environ["TORCHINDUCTOR_COMPILE_THREADS"], "1")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("model", generate.WORKER)

    def test_unpicklable_worker_errors_reach_parent_and_shard_log(self):
        context = mp.get_context("spawn")
        for stage in ("initialization", "generation"):
            with self.subTest(stage=stage):
                root = self.root / stage
                root.mkdir()
                slots = context.Queue()
                slots.put((0, "GPU-test-device"))
                try:
                    with context.Pool(
                        1, initializer=init_failing_worker,
                        initargs=(slots, str(root), stage),
                    ) as pool:
                        result = pool.apply_async(
                            generate.worker_generate, ((7, "train", ["prompt"]),),
                        )
                        with self.assertRaises(RuntimeError) as caught:
                            result.get(timeout=30)
                        # A queued batch must report the same error without
                        # attempting another engine initialization/generation.
                        retry = pool.apply_async(
                            generate.worker_generate, ((8, "train", ["prompt"]),),
                        )
                        with self.assertRaises(RuntimeError) as repeated:
                            retry.get(timeout=30)
                    message = str(caught.exception)
                    self.assertEqual(message, str(repeated.exception))
                    for detail in (
                        "original engine failure", "UnpicklableWorkerError",
                        "Traceback", "Batch 7", "GPU-test-device", "shard_00.log",
                    ):
                        self.assertIn(detail, message)
                    log = (root / "shard_00.log").read_text()
                    self.assertIn("original engine failure", log)
                    self.assertIn("UnpicklableWorkerError", log)
                    self.assertEqual(log.count("[distill] Batch 7 failed"), 1)
                finally:
                    slots.close()
                    slots.join_thread()

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
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(self.tokenizer), hidden_size=128, intermediate_size=256,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            max_position_embeddings=64, eos_token_id=2, pad_token_id=1,
            tie_word_embeddings=False,
        ))
        # Ensure nonempty, stable completions. Random untrained weights can emit
        # only PAD, which the production code correctly filters out.
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(0.1)
            model.lm_head.weight.zero_()
            model.lm_head.weight[3].fill_(0.1)
        model.save_pretrained(model_path)
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
                "--num-shards", str(workers), "--batch-size", "3", "--dtype", "bfloat16",
                "--max-length", "24", "--gen-max-length", "40", "--prompt-max-length", "22",
                "--eval-ratio", "0.25", "--local-files-only", "--max-samples", "5",
                "--gpu-memory-utilization", "0.1",
            ]
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=300)
            shard_logs = root.with_name(root.name + ".shards")
            diagnostics = "\n".join(path.read_text() for path in shard_logs.glob("shard_*.log"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + diagnostics)
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
        for overrides in ({}, {"PER_DEVICE_BATCH_SIZE": "4", "DISTILL_DATA_ROOT": "/tmp/path with spaces",
                               "VLLM_GPU_MEMORY_UTILIZATION": "0.6"}):
            result = subprocess.run(["bash", str(launcher), "--local-files-only"], cwd=self.root,
                                    env={**env, **overrides}, text=True, capture_output=True, check=True)
            args = json.loads(result.stdout)
            self.assertEqual(args[0], "scripts/generate_tulu3_selfdistill.py")
            for flag, value in (("--source-dataset", "allenai/tulu-3-sft-mixture"),
                                ("--source-splits", "train"), ("--max-length", "4096"),
                                ("--batch-size", overrides.get("PER_DEVICE_BATCH_SIZE", "32")),
                                ("--num-shards", "0"),
                                ("--gpu-memory-utilization", overrides.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertEqual(args[-1], "--local-files-only")
        result = subprocess.run(["bash", str(launcher)], env={**env, "PER_DEVICE_BATCH_SIZE": "0"},
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
