"""The job the Pointwise Kaggle runner notebook executes.

The notebook (notebooks/kaggle_pointwise_job.ipynb) only clones the `jev`
branch, installs dependencies, finds the adapter and calls this module, so the
job is chosen in git and a new job needs no notebook edits: change the job
below, push, then Save & Run All on Kaggle.

Current job: Phase 4 training (docs/designs/pointwise-phase4-plan.md).
1. Build the training data from pinned sources into /tmp (not saved as output).
   Remove Kaggle's preinstalled torchao and check that LoRA can be applied
   (preflight), so an environment problem fails in seconds, not mid-launch.
   Run 3 caps each public source at 2,000 train records (Kev's balanced recipe;
   docs/designs/pointwise-phase4-review.md).
2. Train on every visible GPU with torchrun: a fresh 2-epoch run with Kev's option
   augmentation, lr 1e-4 and weight decay 0.01, checking dev loss every 15 minutes
   and evaluating the best weights (Phase 4 run 3). It doesn't search
   /kaggle/input, so it can't resume an earlier run's checkpoint by accident.
3. If training fails in its first 20 minutes (almost always out of memory on the
   largest batch, which runs first), retry once with half the token budget.

Usage: python -m scripts.kaggle_job --adapter-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

DATA_DIR = "/tmp/pointwise-data-run4"
TOKEN_BUDGET = 4096
EARLY_FAILURE_MINUTES = 20
TRAIN_ARGS = [
    "--epochs", "2",  # sizes the LR schedule and stops there
    "--augment",
    "--checkpoint-minutes", "15",  # the run is ~1h, so 30 would give only one dev check
    "--max-hours", "10.5",  # safety: Kaggle stops sessions at 12h and then keeps no output
]


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(line.startswith("GPU") for line in out.splitlines()))
    except (OSError, subprocess.CalledProcessError):
        return 1


PREFLIGHT = """
import torch, transformers
from peft import LoraConfig, get_peft_model
from src.pointwise.train import LORA_TARGETS
config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=16)
model = get_peft_model(transformers.Qwen3Model(config), LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=LORA_TARGETS))
model(input_ids=torch.tensor([[1, 2, 3]]))
print("preflight ok: LoRA applies and runs", flush=True)
"""


def run(command: list[str], env: dict) -> int:
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", required=True)  # unused: Phase 4 trains a new adapter
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # the notebook pins GPU 0; training uses all of them
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if run([sys.executable, "-m", "src.pointwise.data.build", "--out", DATA_DIR, "--train-cap", "2000"], env):
        sys.exit("data build failed")
    # Kaggle's image ships torchao 0.10; peft 0.21 raises on any torchao below 0.16
    # while applying LoRA (2026-10-06 session 1 failed this way). Pointwise doesn't
    # quantise, and peft treats an absent torchao as fine.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], env)
    if run([sys.executable, "-c", PREFLIGHT], env):
        sys.exit("preflight failed: LoRA can't be applied in this environment")

    gpus = gpu_count()
    budget = TOKEN_BUDGET
    for attempt in range(2):
        started = time.time()
        code = run(
            [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(gpus),
             "-m", "src.pointwise.train", "--data-dir", DATA_DIR, "--output-dir", args.output_dir,
             "--token-budget", str(budget), *TRAIN_ARGS],
            env,
        )
        if code == 0:
            return
        minutes = (time.time() - started) / 60
        if attempt == 0 and minutes < EARLY_FAILURE_MINUTES:
            budget //= 2
            print(f"Training failed after {minutes:.1f} min; retrying with token budget {budget}", flush=True)
            continue
        sys.exit(f"training failed (exit {code})")


if __name__ == "__main__":
    main()
