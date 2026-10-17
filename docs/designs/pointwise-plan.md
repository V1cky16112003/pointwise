# Pointwise — Implementation Plan

Companion to `pointwise-architecture.md`. Hardware: Kaggle T4 (fp16, ~30 GPU-h/week).
Each phase ends with a measurable gate; don't start the next phase until it passes.
Revised 2026-10-04 after comparison with Kev (architecture §8): pointer-head
read-out instead of slot tokens, TypeSafe's real API contract, re-derived
speed and calibration gates.

## Phase 0 — Logit read-out on Banking77 (done, gate re-scoped)
Goal: prove "decide without generating" on the existing pipeline.
- `src/readout.py`: score each of the 77 labels by its **full-sequence**
  log-likelihood (label tokens + EOS), renormalised over the label set.
  First-token scoring, the original idea, can't tell apart labels that share
  a first token (details in `pointwise-phase0-plan.md`).
- Reuse the existing LoRA checkpoint; compare against `generate()` + `parse_label`.
- Add `ece`, `brier`, `aurc`, `coverage_at` to `src/evaluate_core.py` (+ tests).
- **Gate:** read-out accuracy within 1 pt of generation (PASS: 0.918 vs 0.914);
  ECE reported (PASS: 0.021). Speed (≥5× faster) FAILED at 0.50×. Phase 0c
  profiles why. The speed gate is re-derived in Phase 2, because a read-out
  still has to put all 77 options in the input (`pointwise-phase0c-plan.md`).

Learn: why reading logits gives a probability distribution for free.

## Phase 1 — Shared-prefix parallel questions (≈2 days)
Goal: one state, many questions, one prefill.
- Bump `transformers` on the `jev` branch to a version with Qwen3 (≥ 4.51);
  keep `main`'s pin. Re-run the Phase 0 tests on the new version.
- `src/pointwise/packing.py`: build packed sequence + block attention mask +
  restarted position ids (state, then k question branches). Additive mask with
  `finfo.min`, padded query rows keep their diagonal (as Kev's `branch_mask_batch`).
- Row form: each question as one causal row continuing a cached state;
  rows per pass capped by a token budget.
- Unit tests: packed logits == logits from k separate "state + Qi" runs
  (atol 1e-3); row form == packed form.
- Benchmark on T4: k = 1, 4, 16, 64 questions; compare vs k separate calls.
- **Gate:** equivalence tests pass; latency grows sub-linearly in k.

Learn: KV caching, attention masks, why context budget = state + longest question.

## Phase 2 — Pointer head & the `/v1/systemone` API (≈3 days)
- Delimiters: reuse existing Qwen special tokens for state / question /
  option open / option close / decide; no new embedding rows.
- `src/pointwise/schema.py`: pydantic `Choice`, `Noul`, `Score`,
  `SystemOneRequest` matching TypeSafe's `/v1/systemone` (questions keyed by
  id; choice criteria `{name: description}`; score criteria = ordered list);
  `render()` for object/array states; TypeSafe's confidence formulas; 4-decimal
  rounding (architecture §2).
- `src/pointwise/model.py`: `PointerHead` (`W_q`, `W_k`, `d_p = 256`, fp32),
  reading `<decide>` against each option's closing-token hidden state; masked
  softmax per question; temperature applied only in eval mode.
- Escape `<|…|>` in all caller text before tokenising.
- Tests: answer always within declared options; probs sum to 1; a state or
  option containing a delimiter string cannot add an option; one question's
  text is invisible to its siblings; option-order flip rate (after training).
- **Gate:** API works end-to-end on the untrained model (random but well-typed);
  isolation and boundary-forgery tests pass. Speed, on a ~520-token state with
  1..8 short questions: each extra question costs ≤ 25% of a one-question
  request. A Banking77-shaped request (15-token message, one 77-option
  question, ~480 tokens) is reported next to generation, not gated: there the
  options dominate the input, so sharing the state can't help.
  *(Revised 2026-10-05: the earlier wording gated the extra-question cost on
  77-option questions, which can't pass by construction.)*

## Phase 3 — Training data (≈4 days)
Revised 2026-10-06 after checking the source list against Kev's training data
(details and reasoning in `pointwise-phase3-plan.md`).
- `src/pointwise/data/`: converters → `{request, targets}` JSONL, where `request`
  is a `/v1/systemone` request and `targets` a probability vector per question.
- **Train sources:** MNLI, BoolQ, AG News (capped), SST-5, Yelp (capped),
  CLINC150 minus its banking and credit-card domains and any intent named like a
  Banking77 label, multiple-choice QA (ARC, OpenBookQA, CommonsenseQA), and a
  rule-based policy generator (JSON/prose records, several questions each,
  labels computed by code).
- **Why multiple-choice QA:** every other source has a fixed label set, which a
  model can learn without reading the options. Varied option texts force the
  pointer head to read them, which zero-shot Banking77 depends on.
- **Held out (eval only):** Banking77 (zero-shot), QNLI, PAWS, Emotion,
  TweetEval sentiment. ChaosNLI (calibration vs human disagreement) moves to
  Phase 5; it exists only as an unofficial CC BY-NC mirror.
- Augment: random option subsets, shuffled option order (never for ordered
  Score levels), paraphrased instructions, JSON vs prose state, derived extra
  questions (1–8 per state).
- No paid teacher set in v1: the policy generator covers Jev-shaped requests
  with exact labels at no cost.
- Training context to start: state ≤ 384, question branch ≤ 1024, packed ≤ 2048.
- **Gate:** ≥200k train questions; label-distribution report including where
  the gold option sits (must be ~uniform); no Banking77 text, no excluded CLINC
  intent, no train/eval text overlap (tests assert the checks).

## Phase 4 — RLCD-lite training (2 Kaggle sessions)
Revised 2026-10-06 (details: `pointwise-phase4-plan.md`).
- `src/pointwise/train.py`: LoRA (r=16, attention + MLP) on Qwen3-1.7B Base,
  fp16 backbone with fp32 LoRA, head and loss; loss = log score (soft CE),
  optional Brier term; DDP over both T4s; gradient checkpointing.
- Kaggle-proof: stops itself before the 12h limit (a timed-out session keeps no
  output), checkpoints every 30 min, resumes from the previous notebook version
  attached as input, LR schedule sized from measured step time over 21 planned hours.
- **One main run**, not the 2×2 matrix: the weekly quota (~29h left) fits two
  ~11h sessions plus Phase 5 evaluation. The "SFT cross-entropy baseline" arm
  is dropped: with one-hot targets the log score *is* cross-entropy, so it
  would repeat the main run. Brier and permutation-KL variants wait for a
  later quota week.
- Fit temperature T on dev; store it with the results.
- **Gate:** on held-out out-of-domain sources (QNLI, PAWS, Emotion, TweetEval),
  ECE ≤ 0.05 after temperature scaling, and accuracy clearly above chance per
  source. Report raw ECE, T, and zero-shot Banking77 accuracy (not gated).

Learn: proper scoring rules — why the optimum of log score and Brier is the
true distribution, and why one-hot training still ends up overconfident.

## Phase 5 — Evaluation report (≈2 days)
- Zero-shot Banking77 (never trained on) vs the repo's fine-tuned classifier
  and vs Kev-0.8B / Kev-4B (open weights, same API).
- No-training baseline (SemIf-style): Qwen3-1.7B Base with no LoRA and no
  pointer head, scoring each option by full-sequence log-likelihood as in
  `src/readout.py`, renormalised over the candidate set. Same eval sets,
  prompts and candidates as Pointwise, so the gap is what fine-tuning adds.
  Report accuracy, raw ECE and ECE after its own fitted T per source, with
  paired bootstrap CIs on the accuracy gap.
- Task-specific generative fine-tune: the Phase 0 Banking77 LoRA
  (Qwen2.5-1.5B-Instruct) scored by the same logit read-out, on Banking77 and
  on every other eval source. In-domain specialist vs Pointwise zero-shot on
  Banking77; out of domain, it shows what a single-task fine-tune transfers.
  Its backbone differs, so also run the no-training baseline on
  Qwen2.5-1.5B-Instruct and compare each fine-tune's gain over its own base.
- Reliability diagrams (raw and after T), risk–coverage curves, Cov@5%,
  option-shuffle flip rate.
- Latency table on T4: Pointwise vs generate-and-parse, k questions per state.
- Write `reports/pointwise-results.md`.
- **Gate:** report shows where Pointwise wins and loses, with bootstrap CIs,
  including against the no-training baseline.

**Phase 5 log**
- **2026-10-14, prepared, not yet run.** Model under test is Phase 4 run 3 (Kaggle `mini-jev-phase-0`
  version 12, session 358167213, best step 1541, T = 1.52). Its held-out predictions and weights were saved
  locally (`artifacts/run3/`, git-ignored) and `score_rows` reproduces the logged per-source numbers.
  - Built: `src/pointwise/baselines.py` (option log-likelihood scoring for a plain causal LM, with the prompt
    cache shared across options), `src/pointwise/compare.py` (arm alignment, paired bootstrap resampled inside
    each source, reliability bins, flip rate), `scripts/phase5_eval.py` (all arms on one T4),
    `scripts/phase5_report.py` (tables, plots, `reports/pointwise-results.md`), and `scripts/kaggle_job.py`
    now runs that eval, not training. `reports/run3_sample_ids.json` pins run 3's 1,500 dev and 2,500
    held-out ids; the job asserts it rebuilds exactly them, so every arm answers the same questions.
  - Arms: Pointwise run 3; Qwen3-1.7B-Base with no training; Qwen2.5-1.5B-Instruct with and without the
    Phase 0 Banking77 LoRA (Banking77 read through the specialist's own prompt via `src/readout.py`, other
    sources through the plain-text prompt). Each arm's T is fitted on its own dev rows, never on held-out.
  - Tested without a GPU: option scoring equals scoring each full sequence alone; flip-rate bookkeeping;
    bootstrap; arm alignment; the three causal arms end to end on Qwen3-0.6B / Qwen2.5-0.5B with a
    zero-initialised adapter (which reproduces the base arm exactly). Not tested: the real 1.7B / 1.5B models,
    and Pointwise's flip-rate and latency stages, which only ran partially on CPU (memory-bound on an 8 GB laptop).
  - **Kev not run.** Kev's model card lists Banking77 among its training sets, so Kev's Banking77 number would be
    in-domain, not zero-shot. QNLI, PAWS and Emotion are listed as never trained on. Kev-0.8B uses a
    Qwen3.5 (Gated DeltaNet) base and its own package, so it needs a separate environment and a T4 check.
    Left for a decision; see the hand-off.
  - Estimated cost, one T4 session, about 1-1.5 h wall (~1.5-2 GPU-hours of quota, an estimate from the Phase 0b
    per-example timings, not a measurement): data rebuild and ~10 GB of model downloads, then ~7,000 plain-text
    questions per causal arm, Banking77 read-out for two arms, flip rates and latency.
    The job saves after each arm, so a late failure keeps the earlier ones.

- **2026-10-15, run (Kaggle `mini-jev-phase-0` version 17, commit 3c513b7).** Version 16 ran only the Pointwise arm:
  the baselines raised `NotImplementedError` on the dev set's score questions (policies, SST-5, Yelp; held-out has
  none). Fix 3c513b7: baselines skip them; each arm fits T on its own dev rows. Version 17 ran all four arms.
  Reproduction check passed (Pointwise logits equal run 3's saved predictions, T 1.516, pooled OOD ECE 0.0561); the
  LoRA arm's Banking77 accuracy was 0.932 against Phase 0's 0.918. Kev not run (Banking77 in its training data;
  separate environment). Results (`reports/pointwise-results.md`): pooled OOD accuracy Pointwise 0.678, Qwen3 base
  0.640 (+3.8, CI [+1.8, +5.8]), Banking77 LoRA 0.680 (-0.2, CI [-2.1, +1.7], not distinguishable), Qwen2.5-Instruct 0.698 (-2.0, CI
  [-4.0, -0.1], borderline). Pointwise loses Emotion to the Qwen3 base (-8.0, CI [-11.8, -4.2]). Pooled OOD ECE after T:
  Pointwise 0.056, Qwen3 base 0.109, LoRA 0.081, Instruct 0.052. Cov@5% pooled OOD 0.004 vs 0.171 for Instruct.
  Flip rate (Banking77/Emotion/TweetEval) 0.210/0.075/0.045 vs 0.615/0.190/0.070 for the Qwen3 base. Latency T4, k=1/8:
  196/261 ms vs 223/1,890 ms. Estimated cost was ~1-1.5 h wall (~1.5-2 GPU-h quota); actual not measured here. Kev comparison was not run, so Phase 5's Kev item is open.

## Phase 6 — Serving (≈2 days)
- Extend `src/serve.py` with `POST /v1/systemone`, request/response exactly as
  `schema.py` (TypeSafe's contract), so a Jev or Kev client works unchanged.
- State-prefix cache: reuse a state's KV across requests with the same state.
- Optional: vLLM prefix caching path for throughput.
- **Gate:** p95 latency < 150 ms for 1 state + 8 questions on T4.

**Phase 6 log**
- **2026-10-16 built and tested; 2026-10-17 T4 latency measured (Kaggle `mini-jev-phase-0` version 18, commit 0d56b95).** Model is Phase 4 run 3, unchanged (T 1.516 comes from
  `artifacts/run3/metrics.json` at startup).
  - Built: `POST /v1/systemone` in `src/serve.py` (body validated by `schema.py`, answers from `DecisionModel.decide`,
    422 for malformed input or a state over 32,768 tokens, a generic 500 that leaks nothing, 503 if the model is not
    configured); configured by `POINTWISE_WEIGHTS`, `POINTWISE_METRICS`, `POINTWISE_DEVICE`, `POINTWISE_MERGE_LORA`,
    `POINTWISE_CACHE_ENTRIES`, `POINTWISE_CACHE_TOKENS`, and the server refuses to start if a file is missing or T is not
    positive. `/predict` (Phase 0) still works when `ADAPTER_DIR` is set. `StateCache` in `src/pointwise/packing.py`
    is an LRU of prefilled state KV caches keyed by a hash of the state's token ids, bounded by entries and by total
    tokens. `scripts/phase6_latency.py` times six configurations; `scripts/kaggle_job.py` now runs it.
  - Tested without a GPU, in `tests/test_pointwise_serving.py` (tiny random Qwen3, not the real model): cached and
    uncached probabilities agree (fp32 1e-5; the fp16 tolerance of 5e-3 is an assumption, not a measurement), a smaller
    request after a larger one is unaffected, stored entries hold only their state (the cropped view used to keep the
    question rows alive), two states never share an entry, LRU and token-budget eviction, a LoRA-wrapped and a merged
    backbone, 4xx for malformed input and caller text the encoder refuses, a generic 500 otherwise, startup failures
    (missing files, T not positive, metrics from other weights, bad cache sizes). Ad hoc and not in the suite: a CPU
    smoke of the cache and the latency script on Qwen3-0.6B-Base through `train.build_model`, and a deliberate break
    of the cache key and of the crop, which made the tests fail. Not tested before the T4 job (see Results below for what it then measured): the real 1.7B model loaded through
    `load_pointwise`, anything on a GPU, fp16 agreement on real weights, a state near the 32k-token cap (the block
    mask is O(length^2) and could run out of memory; that would be a 500), a request with hundreds of questions or many long options (only the state length is capped), and the cache at its limits: a state over `POINTWISE_CACHE_TOKENS` (default 16,384) is answered uncached, and a full cache briefly holds one extra state during prefill.
  - The T4 job also recorded how far cached and merged answers drift from the uncached, unmerged ones on the real
    weights (largest probability difference and top-option flips); T 1.516 was fitted on the unmerged model.
  - **Gate (p95 < 150 ms for 1 state + 8 questions on a T4): met only with a cached state; not met cold.** See Results.
  - **Results** (`reports/pointwise-serving.md`, 524-token state, k=8 questions, p95 against the 150 ms target): cold
    unmerged 212 ms (not met, 62 over), cold merged 179 ms (not met, 29 over), cached unmerged 83 ms (met), cache miss
    298 ms (not met), cached merged 56 ms (met). A cache miss is slower than no cache at every k (+144 ms at k=1, +85 ms
    at k=8, by medians), so the unmerged cache pays only if more than about 40% (k=8) to 66% (k=1) of requests reuse a
    stored state (from medians; the merged miss cost was not measured). Largest probability drift on the real weights:
    cached 0.0018, merged 0.0026, no top option changed, over 8 distinct questions on one state (the merged-and-cached
    configuration was not checked). One state stores 57.3 MiB. Not measured: HTTP, concurrency, other state lengths, accuracy or calibration of the
    merged model, the real `load_pointwise` start-up.
  - Cost of the T4 job: no data build, one model load, six configurations x four k x 65 requests. The estimate was
    15-20 minutes, about 0.3 GPU-h; the actual session time was not measured here.

## Phase 7 — Stretch
- RLCD stage 2 with rationales (OpenJev Algorithm 1) on a reasoning task.
- >255 options via 2-stage Noul scoring → Choice.
- Option isolation (permutation-invariant option spans, packed form).
- Qwen3-4B backbone.

## Risks
| Risk | Mitigation |
|---|---|
| Pointer head learns option-position bias | Shuffle options, permutation-KL term, measure flip rate; option isolation in Phase 7 |
| Pinned transformers 4.46 can't load Qwen3 | Bump on `jev` in Phase 1; Phase 0 tests already run on 4.46 and 5.x |
| Speed gain over generation smaller than hoped | Gate on marginal cost per extra question, where shared prefill pays off |
| T4 fp16 overflow | Qwen3 (not Gemma); head and loss in fp32; grad clipping |
| Kaggle quota | Small run matrix; resume checkpoints; short eval sets during dev |
| Teacher cost | Teacher set is optional; real datasets alone suffice for v1 |
| Overclaiming vs Jev | Report results as "Pointwise", never as Jev parity; compare against Kev, which has open weights |

## Out of scope
Pre-training, multimodal input, multi-label outputs, matching Jev's benchmark claims.
