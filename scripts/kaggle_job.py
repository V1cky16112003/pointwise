"""The job the Pointwise Kaggle runner notebook executes.

The notebook (notebooks/kaggle_pointwise_job.ipynb) only clones the `jev`
branch, installs dependencies, finds the Banking77 adapter and calls this
module, so the job is chosen in git and a new job needs no notebook edits:
change the job below, push, then Save & Run All on Kaggle.

Current job: Phase 6 serving latency (docs/designs/pointwise-plan.md). No training, no data build, so no DATA_DIR.
1. Find Phase 4 run 3's best weights first, before any slow step: any state.pt under /kaggle/input (the
   pointwise-run3-weights dataset, or notebook version 12, session 358167213). Only one at run 3's best step is accepted.
2. Remove Kaggle's preinstalled torchao and check that LoRA can be applied (preflight).
3. Run scripts.phase6_latency on one GPU: one ~512-token state and k = 1/2/4/8 questions, median and p95 over 60 requests,
   for a cold state, a cached state and a cache miss, with the adapter unmerged and merged.
The Phase 5 evaluation job is in git history at a9b86a0, the Phase 4 training job (runs 2-5) at b4c10e7.

Usage: python -m scripts.kaggle_job --adapter-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    # Any path: a Kaggle dataset upload flattens folders, so the file may sit at the dataset's root.
    for path in sorted(Path("/kaggle/input").glob("**/state.pt")):
        try:
            step = torch.load(path, map_location="cpu", weights_only=True)["step"]
        except Exception as e:  # noqa: BLE001  not a Pointwise checkpoint
            found.append((str(path), repr(e)[:60]))
            continue
        found.append((str(path), step))
        if step == RUN3_BEST_STEP:
            return path
    sys.exit(f"no run 3 weights (best step {RUN3_BEST_STEP}) under /kaggle/input; found {found}. "
             "Attach the dataset holding run 3's state.pt (or Kaggle notebook version 12 of mini-jev-phase-0) as an input.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", help="unused: the runner notebook always passes it")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # one T4: latency is measured on one GPU
    state = find_run3_state()  # first: a missing input should fail in seconds, not after the install steps
    print("Using run 3 weights:", state, flush=True)
    # Kaggle's image ships torchao 0.10; peft 0.21 raises on any torchao below 0.16 while applying LoRA.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], env)
    if run([sys.executable, "-c", PREFLIGHT], env):
        sys.exit("preflight failed: LoRA can't be applied in this environment")
    code = run([sys.executable, "-m", "scripts.phase6_latency", "--state", str(state), "--output-dir", args.output_dir], env)
    if code:
        sys.exit(f"phase 6 latency failed (exit {code})")


if __name__ == "__main__":
    main()
