"""The job the Pointwise Kaggle runner notebook executes.

The notebook (notebooks/kaggle_pointwise_job.ipynb) only clones the `jev`
branch, installs dependencies, finds the Banking77 adapter and calls this
module, so the job is chosen in git and a new job needs no notebook edits:
change the job below, push, then Save & Run All on Kaggle.

Current job: Phase 5 evaluation (docs/designs/pointwise-plan.md). No training.
1. Rebuild the Phase 3 data into /tmp with run 3's settings (train cap 2,000), which reproduces
   run 3's dev and held-out records; scripts.phase5_eval asserts the ids match reports/run3_sample_ids.json.
   Remove Kaggle's preinstalled torchao and check that LoRA can be applied (preflight).
2. Find Phase 4 run 3's best weights under /kaggle/input: attach that notebook version (version 12, session
   358167213) as an input. Only a state.pt at run 3's best step is accepted.
3. Run scripts.phase5_eval on one GPU: Pointwise, Qwen3-1.7B-Base with no training, Qwen2.5-1.5B-Instruct with
   and without the Banking77 LoRA, option-shuffle flip rates and latency.
The Phase 4 training job (runs 2-5) is in git history, last at commit b4c10e7.

Usage: python -m scripts.kaggle_job --adapter-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = "/tmp/pointwise-data-run6"

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


def find_run3_state() -> Path:
    import torch

    from scripts.phase5_eval import RUN3_BEST_STEP

    found = []
    for path in sorted(Path("/kaggle/input").glob("**/pointwise-train/best/state.pt")):
        step = torch.load(path, map_location="cpu", weights_only=True)["step"]
        found.append((str(path), step))
        if step == RUN3_BEST_STEP:
            return path
    sys.exit(f"no run 3 weights (best step {RUN3_BEST_STEP}) under /kaggle/input; found {found}. "
             "Attach Kaggle notebook version 12 of mini-jev-phase-0 as an input.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", required=True, help="Phase 0 Banking77 LoRA")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # one T4: latency is measured on one GPU
    if run([sys.executable, "-m", "src.pointwise.data.build", "--out", DATA_DIR, "--train-cap", "2000"], env):
        sys.exit("data build failed")
    # Kaggle's image ships torchao 0.10; peft 0.21 raises on any torchao below 0.16 while applying LoRA.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], env)
    if run([sys.executable, "-c", PREFLIGHT], env):
        sys.exit("preflight failed: LoRA can't be applied in this environment")
    state = find_run3_state()
    print("Using run 3 weights:", state, flush=True)
    code = run([sys.executable, "-m", "scripts.phase5_eval", "--data-dir", DATA_DIR, "--state", str(state),
                "--adapter-dir", args.adapter_dir, "--output-dir", args.output_dir], env)
    if code:
        sys.exit(f"phase 5 evaluation failed (exit {code})")


if __name__ == "__main__":
    main()
