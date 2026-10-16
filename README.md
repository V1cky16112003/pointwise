# Pointwise — a small model that returns calibrated decisions in one forward pass

Pointwise is an open, small-scale replication of the "System One" idea: **unstructured state in, typed and
calibrated decisions out, without generating text.** You send a state plus questions with a fixed set of
options, and the model returns a probability for every option. The answer is always one of the declared options,
so an out-of-schema answer cannot happen.

Status: **Phase 5 (evaluation) is done except the Kev comparison, which was not run**; Phases 0–4 are done. **Phase 6 (serving) is built and tested on CPU; its T4 latency numbers are not measured yet.** The Phase 4
calibration gate **failed** (pooled out-of-domain ECE 0.056 against a 0.05 target) and run 3 was not retuned.
The full comparison is in `reports/pointwise-results.md`.

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

## Phase 5 results

Run 3 was compared on the same 2,500 held-out questions (500 per source) with a no-training Qwen3-1.7B-Base, the
Phase 0 Banking77 LoRA specialist (Qwen2.5-1.5B) and its no-training base, using paired bootstrap 95% intervals.
Full tables, reliability diagrams and risk–coverage curves: `reports/pointwise-results.md`.

- **Wins:** against its own no-training base, pooled out-of-domain accuracy is 0.678 vs 0.640 (+3.8 points, CI
  [+1.8, +5.8]); Banking77 (zero-shot) +29.8 and TweetEval +20.6. Pooled OOD ECE after temperature is 0.056 vs 0.109
  for that base, but per source it is mixed and Qwen2.5-Instruct is at 0.052 (point estimates, no intervals). The answer changes when options are shuffled on 21% of Banking77 questions vs 61.5% for the base.
- **Latency (T4, one ~512-token state):** 196 ms for 1 question and 261 ms for 8, against 223 ms and 1,890 ms for
  generate-and-parse with the Qwen3 base.
- **Losses:** Emotion accuracy is 8.0 points *below* the no-training Qwen3 base (CI [-11.8, -4.2]). Against the
  no-training Qwen2.5-1.5B-Instruct, pooled OOD is -2.0 points (CI [-4.0, -0.1], borderline), with PAWS and Emotion
  lower. At Cov@5% (the share of questions answerable at <=5% error) Pointwise covers 0.4% pooled OOD vs 17.1% for
  Qwen2.5-Instruct.
- **Not distinguishable from noise:** the Banking77 specialist on pooled OOD (-0.2, CI [-2.1, +1.7], so anything from 2 points worse to 1.7 better); the specialist wins Banking77
  by 30.6 points because that data is in-domain for it.
- **Not run:** Kev-0.8B/4B (Banking77 is in its training data and it needs a separate environment).

## Repo layout

```
src/pointwise/        # the model: schema, encoding, packing, pointer head, training, metrics
src/pointwise/data/   # training-data converters (public sources, policy generator, rule records)
src/readout.py        # Phase 0 label read-out for the Banking77 classifier
src/evaluate_core.py  # shared metrics (ECE, Brier, AURC, coverage), CI-tested without torch
scripts/kaggle_job.py # the job the Kaggle notebook runs (currently the Phase 6 latency measurement)
scripts/phase6_latency.py  # cold / cached / merged serving latency on one T4
src/serve.py          # FastAPI: /v1/systemone (Pointwise, state-prefix cache) and the Phase 0 /predict
scripts/phase5_eval.py   # all arms on one T4; scripts/phase5_report.py builds reports/ from its output
src/pointwise/baselines.py, compare.py  # no-training scoring; arm alignment and paired bootstrap
reports/              # pointwise-results.md (python -m scripts.phase5_report --input artifacts/phase5)
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
  runs may be noise. Phase 5 reports bootstrap confidence intervals; calibration differences (ECE) have none.
- `POST /v1/systemone` exists in `src/serve.py` with a state-prefix cache but has not been run on the real model or a GPU; see `docs/designs/pointwise-plan.md`, Phase 6 log.
- Training data are public datasets plus rule-generated records. There is no teacher-labelled synthetic set.
