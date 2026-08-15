# ![icon](assets/icon.png) E2D2: Encoder-Decoder Diffusion Language Models for Efficient Training and Inference

[![deploy](https://img.shields.io/badge/Paper_📃-green)](https://arxiv.org/abs/2510.22852)
[![deploy](https://img.shields.io/badge/Blog_📝%20%20-8A2BE2)](https://m-arriola.com/e2d2)
[![deploy](https://img.shields.io/badge/HuggingFace_🤗%20-E2D2%20-orange)](https://huggingface.co/collections/kuleshov-group/e2d2)


This repository contains code and scripts for reproducing experimental results from our
work.

## 0. Getting started

### Setup environment

Dependencies are managed exclusively with [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --frozen --extra cu118
```

Run commands through uv so they use the locked environment:

```bash
uv run --extra cu118 pytest
uv run --extra cu118 ruff check src scripts tests
```

Local GPU runs use the locked CUDA 11.8 variant. CUDA 12.8 environments such as
Runpod use `uv sync --frozen --extra cu128`. The accelerator extras are mutually
exclusive and share one lockfile. [`setup_env.sh`](./setup_env.sh) only supplies
optional runtime settings; it does not install or activate an environment.

When present, `setup_env.sh` loads WandB and Hugging Face credentials from
`~/setup_discdiff.sh`. Credentials are optional for offline tests.
Copy the contents below into a shell script `/home/<YOUR_USER_NAME>/setup_discdiff.sh`
and replace the placeholder tokens with your own:
```shell
# W&B / HF Setup
export WANDB__SERVICE_WAIT=600
export _WANDB_STARTUP_DEBUG="true"
export WANDB_ENTITY="<WANDB_ENTITY>"
export WANDB_API_KEY="<WANDB_API_KEY>"
echo "Logging into W&B as '${WANDB_ENTITY}'."

# HF Setup
export HUGGINGFACE_TOKEN="<HF_TOKEN>"
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```
- WandB token can be found [here](https://wandb.ai/authorize).
- HuggingFace token can be setup [here](https://huggingface.co/settings/tokens).

### Contributing to the repo
We will try to use GitHub issues to track bugs, features, and todos.
To contribute to the repo, please create a new issue and assign it to yourself.
Then [create a new branch from the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-a-branch-for-an-issue)
and open a pull request.


We use [pre-commit](https://pre-commit.com/) to run linters and formatters on the code.
To install the pre-commit hooks, run:

```bash
uv run pre-commit install
```
On every `git commit`,
the pre-commit hooks will run automatically and report any issues / automatic fixes that
were applied.

### Runpod development Pods

Runpod uses the CUDA 12.8 dependency variant and keeps durable data on a volume
mounted at `/workspace`. Build the development image for Runpod's x86-64 hosts:

```bash
docker build --platform=linux/amd64 -f Dockerfile.runpod \
  -t YOUR_REGISTRY/e2d2-runpod:cuda128-v1 .
docker push YOUR_REGISTRY/e2d2-runpod:cuda128-v1
```

Create a private Pod template with that image, a network volume mounted at
`/workspace`, and ports `8888/http` and `22/tcp`. Leave the entrypoint and start
command empty so the base image continues to start Jupyter and SSH.

Clone this repository to `/workspace/e2d` on the first Pod. On each Pod, update
and validate the environment with:

```bash
cd /workspace/e2d
git pull --ff-only
source bash_scripts/runpod_bootstrap.sh
```

The bootstrap uses `/opt/e2d-venv`, loads
`/workspace/credentials/e2d.env` when present, and requires a successful CUDA
tensor operation. Keep the credentials file out of Git and set its mode to 600.

Run the portable reference experiment with persistent outputs and checkpoints:

```bash
AR_CHECKPOINT_PATH=/workspace/checkpoints/ar/weights-only.pt \
  bash bash_scripts/run_train_e2d_gsm8k_small.sh
```

## 1. Code Organization
1. [`bash_scripts`](bash_scripts): These shells scripts can be used to reproduce the
experiments from our work.
See [below](#2-reproducing-experiments).
2. [`configs`](configs): We utilize hydra config files to organize experiments.
   1. [`config.yaml`](configs/config.yaml) This config is the entry point for launching
   training experiments.
   2. [`eval_config.yaml`](configs/eval_config.yaml) This config is the entry point for
   evaluations.
3. [`scripts`](scripts): The main training and evaluation scripts
   1. [`scripts/composer_scripts/train_discrete_denoiser.py`](scripts/composer_scripts/train_discrete_denoiser.py):
   This script is the main training entry point.
   2. [`scripts/evals`](scripts/eval): These scripts run the evaluation for the
   translation, summarization, and math reasoning datasets, as well as any likelihood
   evaluation.
4. [`src`](src):
   1. [`src/denoiser`](src/denoiser): During training, denoisers take in "noisy" inputs
   and predict clean signals.
   At inference, starting from a purely noisy signal, through iterative denoising, these
   classes produce samples that resemble data.
      1. `AR`: We can view autoregressive models within this paradigm.
      Noise is applied by masking tokens one at a time from right-to-left.
      Denoising is done one token at a time, left-to-right.
      2. `Diffusion`: We implement masked diffusion models:
         - `MDLM`: Standard masked diffusion.
         - `BD3LM`: Block diffusion models.
         - `E2D2`: Our encoder-decoder implementation.
   2. [`src/backbone`](src/backbone): These are the underlying neural networks the take
   in noisy inputs and produce logits.
   Each denoiser is parameterized by a backbone.
   The denoiser can optionally, post-process the logit outputs of the backbone to
   produce log-probs over the clean sequence.


## 2. Reproducing Experiments
The shell scripts provided in [`bash_scripts`](bash_scripts) can be used to reproduce
the training and evaluations from our work.
- For training, the files follow a convention where the dataset and denoiser class are
specified.
For example, to train the fine-tuning E2D2 model on the GSM8K dataset, use the following
shell script: [`run_train_e2d2_gsm8k.sh`](bash_scripts/run_train_e2d2_gsm8k.sh).
- Once models have been trained, the provided evaluation scripts can be used to reproduce
tables and figures from our work.
For example, to evaluate models trained on the WMT translation dataset, use the
following shell script: [`run_seq2seq_eval_wmt.sh`](bash_scripts/run_seq2seq_eval_wmt.sh).
In that file, and similar ones for other evaluations, specify the path to the saved
checkpoints, and uncomment the relevant section for a given denoiser class.
We also provide scripts that will produce the generation throughput numbers we report.
These files contain a `_tput` at the end of the script name.

### Small E2D reference run

The first correctness and performance milestone is the existing dual-decoder E2D
method: the full model verifies proposals drafted by its tied top two layers. Run its
portable 128-example GSM8K preset with:

```bash
uv sync --frozen
AR_CHECKPOINT_PATH=/path/to/ar/weights-only.pt \
  bash bash_scripts/run_train_e2d_gsm8k_small.sh
```

`OUTPUT_ROOT`, `MODEL_NAME`, `RUN_NAME`, `NUM_DEVICES`, and `MAX_DURATION` are optional.
The script defaults to online W&B logging, local outputs, block size 4, 20 training
batches, and a 32-example validation subset. It deliberately fails before launching if
the AR checkpoint is unspecified.

E2D generation keeps its historical return tuple and also records normalized metrics
in `model._last_speculative_stats`, including proposals, accepted/committed tokens,
draft and verifier calls, acceptance length, drafting time, total time, and throughput.

Checkpoints default to `~/.cache/e2d/checkpoints/<run_name>`, which is the large cache
mount on the reference machine. Set `E2D_CHECKPOINT_ROOT` to override it. Hydra outputs
and W&B logs remain under the configured run directory; checkpoint callbacks alone use
the cache location. The small smoke launcher disables checkpoints by default; set
`ENABLE_CHECKPOINTING=true CHECKPOINTS_TO_KEEP=1` for a retained training checkpoint.

Below are the evaluation scripts provided for various tasks:
- Text summarization: [`run_seq2seq_eval_cnndm.sh`](bash_scripts/run_seq2seq_eval_cnndm.sh),[`run_seq2seq_eval_cnndm_tput.sh`](bash_scripts/run_seq2seq_eval_cnndm_tput.sh)
- Machine translation: [`run_seq2seq_eval_wmt.sh`](bash_scripts/run_seq2seq_eval_wmt.sh), [`run_seq2seq_eval_wmt_tput.sh`](bash_scripts/run_seq2seq_eval_wmt_tput.sh).
- Mathematical reasoning: [`run_lm_eval_harness.sh`](bash_scripts/run_lm_eval_harness.sh), [`run_lm_eval_harness_tput.sh`](bash_scripts/run_lm_eval_harness_tput.sh), [`run_likelihood_eval_gsm8k.sh`](bash_scripts/run_likelihood_eval_gsm8k.sh)
- Likelihood estimation (trained on OpenWebText): [`run_likelihood_eval_owt.sh`](bash_scripts/run_likelihood_eval_owt.sh)

## 3. HuggingFace Integration
We release the following models on HuggingFace:
- 80M E2D2 for text summarization (trained from scratch):
[`kuleshov-group/e2d2-cnndm`](https://huggingface.co/kuleshov-group/e2d2-cnndm)
- 250M E2D2 for machine translation (trained from scratch):
[`kuleshov-group/e2d2-wmt`](https://huggingface.co/kuleshov-group/e2d2-wmt)
- 1.7B E2D2 for mathematical reasoning (fine-tuned from Qwen3):
[`kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B`](https://huggingface.co/kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B)
- 170M E2D2 trained on OpenWebText (trained from scratch):
[`kuleshov-group/e2d2-owt`](https://huggingface.co/kuleshov-group/e2d2-owt)

To use these models, follow the snippet below:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.generation.stopping_criteria import EosTokenCriteria
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_config_overrides = {}  # Use this to optionally override config parameters
model = AutoModelForMaskedLM.from_pretrained(
    "kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B",  # Use one of the repos from above
    trust_remote_code=True,
    # **model_config_overrides,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
   "kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B",  # Use one of the repos from above
   trust_remote_code=True,
)

system_prompt = "Please reason step by step, and put your final answer within $\\boxed{}$."
user_prompt = "Every day, Wendi feeds each of her chickens three cups of mixed \
chicken feed, containing seeds, mealworms and vegetables to help keep them \
healthy. She gives the chickens their feed in three separate meals. In \
the morning, she gives her flock of chickens 15 cups of feed. In the \
afternoon, she gives her chickens another 25 cups of feed. How many cups \
of feed does she need to give her chickens in the final meal of the day if \
the size of Wendi’s flock is 20 chickens?"

inputs = tokenizer(
   tokenizer.eos_token + system_prompt + user_prompt + tokenizer.eos_token + "Answer:",
   return_tensors="pt").to(device)

with torch.inference_mode():
   output_ids = model.generate(
      inputs=inputs["input_ids"],
      max_new_tokens=256,  # Set a maximum length for the output
      stopping_criteria=EosTokenCriteria(tokenizer.eos_token_id), # Stop generation after next EOS
   )
print(tokenizer.decode(output_ids[0]))
```

These models can also be used in the evaluation scripts by setting
`pretrained_model_name_or_path=` to one of the options above.

## Citation
```
@inproceedings{
arriola2025e2d2,
title={Encoder-Decoder Diffusion Language Models for Efficient Training and Inference},
author={Marianne Arriola and Yair Schiff and Hao Phung and Aaron Gokaslan and Volodymyr Kuleshov},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://arxiv.org/abs/2510.22852}
}
```
