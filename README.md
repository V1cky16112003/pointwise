# Pointwise — a small model that returns calibrated decisions in one forward pass

Pointwise is an open, small-scale replication of the "System One" idea: **unstructured state in, typed and
calibrated decisions out, without generating text.** You send a state plus questions with a fixed set of
options, and the model returns a probability for every option. The answer is always one of the declared options,
so an out-of-schema answer cannot happen.

Status: **Phase 5 (evaluation report) is in progress.** Phases 0–4 are done. The Phase 4 calibration gate
**failed**, so the numbers below are honest but not yet good. The comparison against baselines is not written yet.

## How it works

- **Backbone:** Qwen3-1.7B Base with a LoRA adapter (r=16, attention + MLP). No text is generated.
- **Pointer head:** a small head reads the hidden state at a `<decide>` token and scores it against the hidden
  state at the end of each option. A softmax over one question's options gives its probabilities.
- **API:** the request and response shapes follow TypeSafe's public `POST /v1/systemone` contract
  (`src/pointwise/schema.py`), with `choice`, `noul` and `score` question types.
- **Parallel questions:** one state is encoded once, and several questions are answered from it with a block
  attention mask (`src/pointwise/packing.py`).
- **Design:** `docs/designs/pointwise-architecture.md` (architecture and the evidence behind each choice) and
  `docs/designs/pointwise-plan.md` (phases and gates). The design is reconstructed from public clues and from Kev, an
  open Jev-like model by Jared Palmer. TypeSafe has not published Jev's internals.

## Results so far

All numbers below come from the logs in `docs/designs/`.

**Phase 0: logit read-out on Banking77** (Qwen2.5-1.5B-Instruct LoRA, label scoring by full-sequence
log-likelihood instead of `generate()`):

| | Read-out | Constrained generation |
|---|---|---|
| Accuracy | 0.918 | 0.914 |
| ECE | 0.021 | n/a |

The read-out matched generation on accuracy and was well calibrated, but it was **slower** than generation
(0.50× the speed), because all 77 options still have to go through the model. The speed gate failed.

**Phase 4: Pointwise training, run 3 (the model carried forward).** Held-out accuracy and ECE after temperature
scaling (T = 1.52, fitted on a dev sample), 500 questions per source:

| Source | Accuracy | ECE |
|---|---|---|
| QNLI | 0.818 | 0.082 |
| PAWS | 0.750 | 0.130 |
| Emotion | 0.492 | 0.159 |
| TweetEval sentiment | 0.652 | 0.038 |
| Banking77 (zero-shot, never trained on) | 0.626 | 0.113 |
| **Pooled out-of-domain ECE (first four rows)** | | **0.056** |

**The Phase 4 gate failed.** It required pooled out-of-domain ECE ≤ 0.05 with no source more than 3 points below
run 2's accuracy. Run 3 is the best of four runs (runs 2–5 landed between 0.056 and 0.085) but still misses both
conditions: ECE is 0.056 and QNLI accuracy is 4.8 points below run 2. The weak spots are over-confidence on PAWS
(the model leans on word overlap) and on Emotion. Details and the other runs are in
`docs/designs/pointwise-phase4-plan.md`.

## What Phase 5 will add

A report, `reports/pointwise-results.md`, comparing Pointwise with a no-training Qwen3-1.7B baseline, the Phase 0
Banking77 specialist, and Kev, with paired bootstrap confidence intervals, reliability diagrams, risk–coverage
curves and a latency table. This section will be replaced by its findings when it is written.

## Repo layout

```
src/pointwise/        # the model: schema, encoding, packing, pointer head, training, metrics
src/pointwise/data/   # training-data converters (public sources, policy generator, rule records)
src/readout.py        # Phase 0 label read-out for the Banking77 classifier
src/evaluate_core.py  # shared metrics (ECE, Brier, AURC, coverage), CI-tested without torch
scripts/kaggle_job.py # the job the Kaggle notebook runs (settings for each training run)
docs/designs/         # plans, run logs and design notes for every phase
LOGBOOK.md            # chronological record of decisions
```

The repo also holds the original Phase 0 project: a LoRA vs. QLoRA comparison on Banking77 with a config-driven
eval harness (`src/train.py`, `src/evaluate.py`, `configs/task.yaml`, `src/serve.py`, `demo/`). Its design is in
`docs/designs/lora-finetuning-banking-intent-classifier.md`.

## Running it

Training and evaluation run on Kaggle (GPU T4 x2). Import the notebook, attach the Banking77 input, then Save
Version → Save & Run All. The notebook clones the `jev` branch and runs `scripts/kaggle_job.py`.

Tests (no GPU needed):

```bash
pip install -r requirements-core.txt
pytest tests/ -v
```

## Known limitations

- The Phase 4 calibration gate failed (pooled out-of-domain ECE 0.056 against a 0.05 target).
- Each result is a single run with one seed, and each source has 500 held-out questions, so small gaps between
  runs may be noise. Phase 5 adds bootstrap confidence intervals.
- The `/v1/systemone` endpoint is not served yet (Phase 6); `src/serve.py` still serves the Phase 0 classifier.
- Training data are public datasets plus rule-generated records. There is no teacher-labelled synthetic set.
