# Pointwise Phase 4: RLCD-lite training (execution plan)

Parent: `pointwise-plan.md` Phase 4. Date: 2026-10-06.

## Goal
Train the pointer head and a LoRA adapter so the Phase 2 decision API gives
calibrated answers, and measure them on sources it never trained on.

## Budget
Kaggle quota at the start: 29.5 GPU-hours left until 2026-10-10. Sessions are
capped at 12h, and a session that times out or is cancelled saves **no output**.
- Session 1: ~11h (setup ~10 min, training 10.5h, evaluation ~10 min).
- Session 2: same, resuming from session 1.
- ~7h left for Phase 5 evaluation.
Both T4s are used (DDP): quota counts session time, not GPUs, so two GPUs train
about twice as much for the same quota.

## Design (`src/pointwise/train.py`)
| Choice | Value | Why |
|---|---|---|
| Base | Qwen3-1.7B-Base, fp16, frozen | T4 has no bf16; Phase 1–2 ran on it |
| Adapter | LoRA r=16, α=32, dropout 0.05, q/k/v/o/gate/up/down, fp32 | Kev's recipe |
| Head | pointer head, fp32 (autocast disabled around it) | small logits differences matter for calibration |
| Loss | log score −Σ q log p (+ optional Brier, off) | strictly proper; equals CE for one-hot targets |
| Optimiser | AdamW, LR 2e-4 (LoRA) / 1e-3 (head), no weight decay, grad clip 1.0, fp16 GradScaler | standard LoRA settings |
| Schedule | 200-step warmup, cosine to 10% over the planned steps | length set from measured step time × 21 planned hours |
| Batches | ≤ 4096 padded tokens per GPU, length-sorted within chunks of 1000 | little padding; largest batch first so OOM appears at once |
| Context | state ≤ 384, state + question ≤ 1024, packed ≤ 2048 tokens; longer records dropped whole | cutting a state could cut the evidence a label depends on |
| Memory | gradient checkpointing (non-reentrant) | fits T4's 15 GB |

Kaggle safety:
- Stops after `--max-hours 10.5` of training and saves; checkpoints every 30 min
  (written to a temp file and renamed, so a kill mid-save can't corrupt it).
- Resumes from the newest `pointwise-train/` (or pre-rename `minijev-train/`) checkpoint under
  `/kaggle/input` (attach the previous version) or its own output dir:
  adapter, head, optimiser, GradScaler, step and schedule length.
- `scripts/kaggle_job.py` builds the data into `/tmp` (~2 min, not saved as
  output), runs torchrun on every GPU, and retries once with half the token
  budget if training fails in its first 20 minutes.

End of session (rank 0): 1,500 dev records and up to 500 records per held-out
source are scored; T is fitted on dev (golden-section search on mean NLL);
`metrics.json` has accuracy / NLL / Brier / ECE per source, raw and scaled, plus
chance accuracy; `heldout_predictions.jsonl` keeps every logit.

## Local verification
- `tests/test_pointwise_metrics.py` (5, CI-safe): softmax, per-question scores,
  temperature fitting recovers a known overconfidence (T = 3), T never changes accuracy.
- `tests/test_pointwise_train.py` (8, tiny random Qwen3 saved to disk, CPU):
  batched forward == one at a time; batches cover every example once within budget,
  largest first; batch stream crosses epochs and gives each rank its own batch;
  LR schedule; a full `main()` run checkpoints and evaluates; a second run resumes
  at the saved step; resume from a "previous session" directory; loss falls on a
  small set; **a real two-process DDP run** (gloo) completes.
- Data build checked with the Kaggle stack (Python 3.11, datasets 3.1.0).
- Bug found by the tests: the training loop passed `Example` wrappers instead of
  encoded requests to the model.

## Gate
On held-out out-of-domain sources (QNLI, PAWS, Emotion, TweetEval sentiment):
ECE ≤ 0.05 after temperature scaling, and accuracy clearly above chance per
source. Reported, not gated: raw ECE, fitted T (Kev's adapters land at 2.2–2.4),
dev accuracy per training source, zero-shot Banking77 accuracy.

## Run 2: shorter, best checkpoint (2026-10-07)
Run 1 overfitted (see run log), so the resume is dropped for a fresh, shorter run:
- `--epochs 2` (~4,900 steps, ~5.5h): the LR schedule decays to 10% over exactly
  two epochs and the run stops there. `--max-hours 10.5` stays as a safety net.
- Every 30 minutes rank 0 scores the 1,500 dev records, fits T and logs dev NLL
  after scaling to `dev_curve.jsonl`. The weights with the lowest scaled dev NLL are
  kept (in memory and in the checkpoint, so a resume keeps them too).
- The final evaluation uses the best weights, saved alone as `best/state.pt`
  (adapter + head) for Phase 5. Dev and held-out samples match run 1's exactly.
- Selection uses scaled NLL, not raw: T is fitted at the end anyway, so raw
  overconfidence alone shouldn't decide. Selection and the final T fit share the
  dev records, so dev numbers are slightly optimistic; held-out isn't used for either.
- No `--resume-search`: it can't pick up run 1's step-9,683 checkpoint by accident.
- Cost: ~6 GPU-hours of the 18.7 left, leaving ~12h for Phase 5.

Tests: `test_epochs_stop_and_best_weights_are_evaluated` (dev checked every step;
the evaluated weights, `best/state.pt` and the reported dev NLL all match the
curve's minimum), plus a two-process DDP run with a dev check every step.

## How to run
Notebook *mini-jev-phase-0*, Accelerator **GPU T4 x2**, Internet on, nothing extra
attached as input, **Save & Run All**.

## Status
- [x] Training code, tests, Kaggle job.
- [x] Run 1, session 1 (gate not met; overfitted; see run log). Session 2 dropped.
- [ ] Run 2: 2 epochs, best checkpoint.

## Run log
- **2026-10-06 05:25 UTC, session 1, attempt 1: failed in 4 minutes, ~4.5 GPU-min used.**
  The data build passed on Kaggle; training died while applying LoRA. Kaggle's image
  ships torchao 0.10.0, and peft 0.21.2 raises on any torchao below 0.16 (an absent
  torchao is fine; it wasn't installed locally, so the tests passed). The OOM retry
  then repeated the same error. Fix: the job uninstalls torchao (Pointwise doesn't
  quantise) and runs a LoRA preflight on a tiny model first, so environment errors
  stop in seconds with a clear message. Reproduced locally with a stub torchao 0.10.0:
  same error with it, preflight passes without.
- **2026-10-06 05:50–16:55 UTC, session 1, attempt 2: completed, ~10.5 GPU-h used.**
  Preflight passed; DDP world 2; setup 174 s. 152,507 train records after the context
  filter (dropped: Yelp 2,135, BoolQ 64, MNLI 1). 4.0 s/step → LR schedule sized to
  18,847 steps for 21 h. Stopped by `--max-hours` at step 9,683 (10.55 h trained),
  checkpointed, evaluated. No OOM, no retry.
  - **The run covered ~4 epochs** (2,451 steps per epoch). Mean train loss per
    epoch: 0.338 → 0.200 → 0.122 → 0.076.
  - **Dev (in-domain, 2,230 questions):** accuracy 0.851, raw NLL 0.679, raw ECE 0.103.
    T fitted on dev = **3.20** (Kev's adapters: 2.2–2.4). Scaled ECE 0.018.
    The gap between train loss (0.08) and dev NLL (0.68) is overfitting, and the
    model is overconfident; a T this high is how that shows up.
  - **Held-out, after T scaling** (accuracy / chance / ECE): QNLI 0.760 / 0.50 / 0.077,
    TweetEval 0.626 / 0.33 / 0.069, Emotion 0.484 / 0.17 / 0.127, PAWS 0.634 / 0.50 /
    0.216, Banking77 zero-shot 0.614 / 0.013 / 0.058. Out-of-domain ECE (all four) 0.074.
  - **Gate: FAIL** (ECE 0.074 > 0.05). Accuracy is above chance on every source;
    PAWS is the weakest (13 points above chance, confidently wrong).
  - Session 2 as planned would take this to ~7.7 epochs; with no dev-loss curve, it
    could deepen the overfitting rather than fix calibration.
- **Run 2, attempt 1** (2026-10-07): DDP deadlock at step 525. Each rank decided
  checkpoint timing on its own clock; rank 1 entered the barrier while rank 0 kept
  training, so both hung until the 2 h NCCL timeout. Fixed in f3a2461 (all-reduce
  the "checkpoint due" flag). No results.
- **Run 2, attempt 2** (2026-10-13, Kaggle session 357990976, commit f3a2461, 2×T4):
  2 epochs, 4,902 steps, 5.3 GPU-hours, finished without error.
  - **Dev curve** (scaled NLL every ~485 steps): 0.403 → 0.368 → 0.338 → 0.352 →
    0.339 → 0.352 → 0.332 → 0.332 → 0.329 → 0.323 → **0.3227 at step 4,902**. The best
    checkpoint is the last step: no overfitting inside 2 epochs (run 1's came from epochs 3–4).
  - **Dev (in-domain, 2,230 questions):** accuracy 0.870, NLL 0.323, ECE 0.022.
    T fitted on dev = **1.32** (run 1: 3.20), so the model is far less overconfident in-domain.
  - **Held-out** (accuracy / chance / raw ECE / ECE after T):
    QNLI 0.866 / 0.50 / 0.094 / 0.138, TweetEval 0.672 / 0.33 / 0.032 / 0.064,
    Emotion 0.504 / 0.17 / 0.163 / 0.099, PAWS 0.654 / 0.50 / 0.255 / 0.223,
    Banking77 zero-shot 0.626 / 0.013 / 0.196 / 0.135. Out-of-domain ECE (all four):
    raw 0.092, **after T 0.080** (run 1: 0.074).
  - **Gate: FAIL** (ECE 0.080 > 0.05). Accuracy is clearly above chance on every source,
    and every source beats run 1.
  - **Likely cause:** in-domain T doesn't carry over out of domain. PAWS stays
    confidently wrong (ECE 0.22, alone enough to break the pooled number), and QNLI gets
    *worse* after scaling (0.094 → 0.138): T > 1 makes an already-accurate source
    underconfident. One global temperature fitted on dev can't fix miscalibration
    that points in different directions on different held-out sources.
  - ~~**Proposed next change (not started):** train with a larger Brier weight (the
    Brier variant set aside in the plan). Brier penalises confident mistakes boundedly
    and tends to give better out-of-distribution calibration than log loss alone. Keep
    2 epochs and the same gate.~~ Withdrawn 2026-10-14: Kev recorded CE + Brier as a
    negative result. See `pointwise-phase4-review.md` for the revised plan.
