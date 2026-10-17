# Pointwise Phase 6: serving latency on a T4

Model: Phase 4 run 3 (best step 1541, T 1.516), unchanged. Measured on Kaggle (`mini-jev-phase-0` version 18, commit
0d56b95) on one Tesla T4, fp16. Raw numbers: `reports/phase6_latency.json`.

## Verdict against the plan's target (p95 under 150 ms for one state and 8 questions)

| Configuration | p95 at k=8 | Target | Margin |
|---|---|---|---|
| (a) cold state, adapter unmerged | 212 ms | **not met** | 62 ms over |
| (b) cold state, adapter merged | 179 ms | **not met** | 29 ms over |
| (c) cached state, adapter unmerged | 83 ms | **met** | 67 ms under |
| (d) cache miss, adapter unmerged | 298 ms | not met | 148 ms over |
| (e) cached state, adapter merged | 56 ms | met | 94 ms under |

So the target is met only when the state's KV cache is already stored. A request with a new state, which is every
first request, misses it; merging the adapter lowers the cold times but not below the target. (The cache-miss row was
measured only unmerged.) A missed target is a result here: the model and its
temperature were not touched.

## What was measured

One state of 524 tokens (the same Banking77-chat construction as Phase 5) and k three-option choice questions,
k = 1, 2, 4, 8. Each cell is 60 timed calls of `DecisionModel.decide(request, state_cache)` after 5 untimed ones,
GPU-synchronised. That call includes tokenising, the model, the pointer head and building the answers. It does **not**
include HTTP, JSON or FastAPI. Times are milliseconds.

| Configuration | k=1 median / p95 / max | k=2 median / p95 / max | k=4 median / p95 / max | k=8 median / p95 / max |
|---|---|---|---|---|
| a. cold state, adapter unmerged | 151 / 153 / 155 | 157 / 158 / 158 | 169 / 171 / 172 | 209 / 212 / 213 |
| b. cold state, adapter merged | 130 / 132 / 133 | 134 / 136 / 136 | 141 / 144 / 145 | 176 / 179 / 181 |
| c. cached state, adapter unmerged | 78 / 83 / 115 | 78 / 81 / 81 | 80 / 85 / 113 | 80 / 83 / 88 |
| d. cache miss (state not stored yet), unmerged | 294 / 300 / 302 | 288 / 294 / 300 | 289 / 294 / 319 | 295 / 298 / 318 |
| e. cached state, adapter merged | 42 / 44 / 45 | 43 / 44 / 45 | 45 / 47 / 65 | 54 / 56 / 56 |
| Phase 5 method (probs on a pre-encoded request, unmerged) | 159 / 161 / 161 | 166 / 168 / 169 | 182 / 186 / 189 | 232 / 236 / 237 |

For comparison, Phase 5 measured generate-and-parse with the no-training Qwen3-1.7B-Base at 223 / 445 / 989 / 1,890 ms
(medians, k = 1 / 2 / 4 / 8), in a different Kaggle session, so the comparison is indicative only: that session ran
the same Pointwise method 11% to 19% slower than this one (see Caveats). It was not re-run here. At k=8, cold
Pointwise (209 ms median) is about 9x faster than that baseline (about 7x against Phase 5's own 261 ms Pointwise
number), and cached Pointwise (80 ms) about 24x. At k=1 the cold path is only about 1.5x faster (151 vs 223 ms).

## What the numbers say

- **Caching makes the cost nearly independent of k.** Cached, k=1 costs 78 ms and k=8 costs 80 ms (median); cold, it
  grows from 151 to 209 ms. With the state already read, the model only processes the questions. Cached unmerged
  stays near 78-80 ms at every k and cached merged near 42-54 ms, so most of that cost does not grow with k; no profile
  was taken, so what makes up that floor is not measured.
- **Merging the adapter helps both.** Cold k=8 goes from 209 to 176 ms (median); cached from 80 to 54 ms.
- **A cache miss makes a request slower, at every k.** (d) against (a): +144 ms at k=1 (294 vs 151), +85 ms at k=8
  (295 vs 209), by the medians. From the code, a miss reads the state once to store it and then runs the questions as a
  second pass, instead of one packed pass; the timings do not separate those two steps. This is the only request shape that is slower than not having the cache.
- **Break-even.** From the medians, the cache lowers the average cost only if more than about 66% (k=1), 62% (k=2),
  57% (k=4) or 40% (k=8) of requests reuse a state already stored (unmerged: the miss cost was not measured for the merged adapter). A workload where each state is asked about once
  is slower with the cache than without it. This was computed from the medians, not measured on a real request mix.
- **Memory.** One stored 524-token state holds 57.3 MiB (60 MB) of keys and values. Scaling that linearly, the default
  cache (8 entries, 16,384 tokens in total) would hold at most about 1.8 GiB, plus one extra state while a new one is
  prefilled. GPU memory was not measured.

## Do the cached and merged answers differ from the plain ones?

Measured on the real weights over the 15 question slots of the k = 1, 2, 4, 8 requests, at the served T. The k requests share their first
questions, so these are 8 distinct questions on one state, not 15:

| Comparison | Largest difference in any probability | Top option changed (of 15 slots) |
|---|---|---|
| cache miss vs uncached | 0.0018 | 0 |
| cache hit vs uncached | 0.0018 | 0 |
| merged vs unmerged | 0.0026 | 0 |

The tests' assumed fp16 tolerance of 5e-3 holds on this sample. Only two comparisons were run, cached against uncached and merged against unmerged; the merged and cached
configuration (e), the fastest row, was not checked on its own. It is a small sample (8 distinct questions, one state, no
answer margins recorded), so it does not rule out a flipped answer on a near-tie, and it says nothing about accuracy or
calibration after merging: merged weights were not re-scored on the Phase 5 held-out set, and T 1.516 was fitted on
the unmerged model.

## What would be needed to meet the target on a cold request

These are hypotheses, not measured. Cold requests miss by 62 ms (unmerged) or 29 ms (merged), so merging alone
gets close but does not reach it.
1. **Merge the adapter and also trim the cold pass**, for example building the attention mask only for the question
   rows, which is what a cached request already does. Untested.
2. **Make the cache miss cheaper**: store the state's keys and values from the single packed pass instead of reading
   the state a second time, which should bring (d) down towards (a). Untested.
3. **Warm the cache before traffic arrives** when the caller knows the state ahead of the questions. This moves the
   cost off the request rather than reducing it.

## Caveats

- One T4 session, one state length (524 tokens), one question shape (three-option choice). A longer state, more
  options or many more questions were not measured. The O(length²) block mask for very long states and requests with
  hundreds of questions are untested and could run out of memory (a 500 from the endpoint).
- Latency excludes HTTP, JSON and the lock; concurrent requests were not measured. The server serialises GPU work with
  one lock, so concurrent requests queue.
- The Phase 5 method row (pre-encoded request, 159 / 166 / 182 / 232 ms medians) is 19%, 17%, 15% and 11% below the
  196 / 199 / 214 / 261 ms Phase 5 reported for the same method in another session, so treat differences smaller than
  that between sessions with care. Within this run that row is also 23 ms *slower* than (a) at k=8 although it does
  less work, which points to drift during the run; the configurations ran in the order a, p5, c, d, then merged b, e,
  so merged-vs-unmerged differences are partly confounded with run order.
- Every cell times the same request repeated 60 times, so the cached-state results are for a hot, identical request;
  p95 over 60 samples is the 57th value, close to the maximum.
- The real server start-up (`load_pointwise` with the 1.7B model and run 3's `metrics.json`) was not run on a GPU; the
  latency script builds the model the same way and sets T = 1.5159, but it is not the endpoint. The endpoint itself was
  tested only with a tiny random model (`tests/test_pointwise_serving.py`).
- The latency job was run once; there are no intervals across sessions.
