# Pointwise cold path: where the time goes (plan step 4)

Kaggle `mini-jev-phase-0`, one Tesla T4, torch 2.11.0+cu128, commit ef86f40, run 3 weights (step 1541, T 1.516).
`scripts/phase6_profile.py`: one 524-token state, k=8 three-option questions, 60 timed requests per row after 5 warm-up.
Wall = `decide()` without instrumentation; stages are CUDA-event GPU times (medians) from an instrumented pass.
Raw numbers: `reports/phase6_profile.json`. All rows come from this one session; they are not compared with Phase 6.

## Where the time goes (k=8, ms)

| Config | wall median / p95 | encode | mask | backbone forward(s) | head | answers | kernels | GPU idle* |
|---|---|---|---|---|---|---|---|---|
| cold, unmerged | 202.3 / 203.9 | 3.6 | 0.7 | packed 194.0 | 2.6 | 0.1 | 3202 | 43% |
| miss, unmerged | 287.5 / 295.4 | 3.8 | 0.7 | prefill 181.7 + questions 107.6 | 2.9 | 0.1 | 6688 | 57% |
| cached, unmerged | 102.0 / 111.4 | 3.4 | 0.8 | questions 98.8 | 2.7 | 0.1 | 3109 | 79% |
| cold, merged | 151.0 / 153.6 | 3.7 | 0.7 | packed 140.9 | 2.7 | 0.1 | 1823 | 32% |
| miss, merged | 194.5 / 198.2 | 3.5 | 0.6 | prefill 137.3 + questions 53.9 | 2.8 | 0.1 | 3939 | 45% |
| cached, merged | 56.8 / 62.9 | 3.4 | 0.7 | questions 54.6 | 2.7 | 0.1 | 1733 | 73% |

\* From one request under `torch.profiler`: 1 - (GPU kernel time / wall). The profiler itself slows the CPU side, so this
overstates idle time; read it as a ranking, not an amount.

## What it says

1. **The cold path is the backbone, nearly all of it.** The 28 decoder layers are 192 of 202 ms unmerged and 138 of 151 ms
   merged. Tokenising, the mask, the head and the answers together come to about 7 ms. The block mask costs 0.7 ms, so
   building it as Q x L instead of L x L (candidate a) cannot buy anything measurable; dropped.
2. **The LoRA adapter costs ~51 ms of a cold request** (6.87 ms per layer unmerged vs 4.99 merged): every adapted linear
   layer runs two extra small matmuls and the kernel count drops from 3,202 to 1,823 when merged.
3. **A miss is not "prefill + cached + 65 ms".** Measured here, miss = prefill + cached + 3.8 ms unmerged (0.5 ms merged).
   The Phase 6 gap was a cross-configuration artefact; what makes a miss slow is that it runs **two** full passes: it reads
   the state once to store its KV (182 ms), then runs the questions on top (108 ms), while a cold request reads state and
   questions in one pass (194 ms). Candidate (b) fixes exactly this.
4. **The cached path is launch-bound.** Its questions pass is ~150 tokens but takes 99 ms unmerged / 55 ms merged, with the
   GPU idle most of the time (79% / 73%): Python cannot queue ~3,100 small kernels faster than the GPU runs them. This is
   where CUDA graphs (candidate c) would help; the cold path is mostly GPU-bound (32% idle merged), so graphs would buy it
   little.

## Merged-model check (plan step 5, same session; `reports/merged_check.json`)

2,500 Phase 5 held-out questions, served path (eval, no autocast), raw logits scored by `score_rows` at T 1.516:
top option agrees on **99.96%** (1 of 2,500 differs); pooled OOD ECE after T 0.05696 unmerged vs 0.05759 merged
(shift **+0.0006**, limit 0.005); held-out accuracy 0.6676 both; max logit difference 0.064. **Passes: merged may be the
default.** The unmerged OOD ECE here (0.0570) differs from Phase 5's 0.0561 because Phase 5 scored under fp16 autocast and
this check uses the served path (fp32 adapter maths); the ship rule compares merged against unmerged in the same session.

## Concept: fp16 rounding when merging

Merging computes W' = W + B·A once and stores W' in fp16, which keeps about 3 significant digits. The unmerged model
instead computes x·W (fp16) and x·A·B (fp32) separately and adds them. The two orders round differently, which is the
0.064 max logit difference above; it changed one top answer in 2,500.

## Next (A/B in one session)

- (b) On a miss, run the one packed pass a cold request runs, and keep the state rows' KV from it (exact: state tokens only
  attend to state tokens). Expected: miss ≈ cold, about 200 ms unmerged / 150 ms merged instead of 288 / 195.
- (d) Serve merged by default (passes above): cold p95 153.6 ms, 3.6 ms over the target in this session.
- (c) CUDA graphs only for the cached path, if (b) and (d) leave the cold target unmet by more than noise.
