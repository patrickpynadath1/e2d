"""GPU smoke test for discrete diffusion drafting and greedy verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir

from scripts.utils import register_useful_resolvers
from src.denoiser.diffusion import DiffusionGenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("masked", "uniform"), default="masked")
    parser.add_argument("--ar-checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer:")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1)
    return parser.parse_args()


@torch.no_grad()
def target_greedy(model, input_ids: torch.LongTensor, count: int) -> torch.LongTensor:
    output = input_ids
    for _ in range(count):
        logits = model.backbone.encoder(input_ids=output, use_cache=False).logits
        output = torch.cat([output, logits[:, -1:].argmax(-1)], dim=-1)
    return output


def main() -> None:
    args = parse_args()
    register_useful_resolvers()
    model_choice = f"{args.mode}_diffusion_e2d"
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "run_name=discrete-generation-smoke",
                "dataset@train_dataset=gsm8k_small_train",
                "dataset@eval_dataset=gsm8k_small_eval",
                f"model={model_choice}",
                "model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv_encoder_gen",
                f"pretrained_model_name_or_path={args.model}",
                f"block_size={args.block_size}",
                "model.config.length=256",
                "model.config.backbone_config.num_encoder_layers=28",
                "model.config.backbone_config.num_decoder_layers=2",
                "model.config.backbone_config.keep_top_decoder_layers=true",
                "model.config.backbone_config.tie_encoder_decoder_weights=true",
                "model.config.backbone_config.train_on_ar=true",
                f"model.config.backbone_config.ar_checkpoint_path={args.ar_checkpoint}",
            ],
        )
    model = hydra.utils.instantiate(cfg.model).to("cuda").eval()
    inputs = model.tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    generation_config = DiffusionGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        num_steps=args.num_steps,
        use_cache=False,
    )
    speculative, stats = model.generate(
        inputs,
        generation_config=generation_config,
        return_speculative_stats=True,
    )
    greedy = target_greedy(model, inputs, args.max_new_tokens)
    if not torch.equal(speculative, greedy):
        message = (
            "speculative output differs from target greedy output:\n"
            f"{speculative}\n{greedy}"
        )
        raise AssertionError(
            message
        )
    print(json.dumps(stats.to_dict(), indent=2))
    print(model.tokenizer.decode(speculative[0]))


if __name__ == "__main__":
    main()
