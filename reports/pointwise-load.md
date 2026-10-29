# Pointwise HTTP load test (production plan step 7)

One Kaggle T4 session, 2026-10-29, commit 8f5579b, `scripts/load_test.py`; raw numbers in `reports/load_test.json`.
The real server (`python -m src.serve_main`) ran on run 3 (`state.pt` step 1541, T 1.516, adapter merged, the default)
through `load_pointwise`, with the `requirements-serve.txt` pins (torch 2.11.0+cu128, transformers 5.14.1, peft 0.21.2,
fastapi 0.138.1, uvicorn 0.49.0). Every request: one 526-token support-chat state and k=8 three-option questions.
Measured, not tuned: no model, latency or serving setting was changed.

## First real start

- Start-up: **14.6 s** from launch to `/readyz` 200 (model already downloaded; the download, timed apart, was 21.9 s).
- GPU memory after load: 3,637 MiB (`nvidia-smi`); peak over the whole test 4,569 MiB of the T4's 15 GB.
- Probes as documented, with `POINTWISE_API_KEY` set: `/healthz` 200 and `/readyz` 200 without the key; `/metrics`
  401 without it, 200 with it; `POST /v1/systemone` 401 with no key and with a wrong key, 200 with the right one.
- The server logged `Pointwise loaded on cuda: step 1541, T=1.516, merged=True`.

## Steady load

Closed-loop clients: each sends its next request as soon as its last one answers. 200 requests per row, run in this
order (GPU memory never shrinks, so the memory column is cumulative). "Cached" means half the requests repeat one of
the last 4 new states; the hit counts are the server's own `/metrics` counters.

| Mix | Clients | Status | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Cache hits / misses | Little's law in flight | GPU MiB peak |
|---|---|---|---|---|---|---|---|---|---|
| cold | 1 | 200 x 200 | 5.94 | 136 | 141 | 143 | 0 / 200 | 0.81 | 4,307 |
| cold | 4 | 200 x 200 | 6.91 | 542 | 559 | 568 | 0 / 200 | 3.73 | 4,371 |
| cold | 16 | 200 x 200 | 6.52 | 2,360 | 2,434 | 2,439 | 0 / 200 | 15.04 | 4,549 |
| cached | 1 | 200 x 200 | 8.16 | 148 | 156 | 158 | 95 / 105 | 0.86 | 4,569 |
| cached | 4 | 200 x 200 | 9.60 | 390 | 590 | 619 | 105 / 95 | 3.79 | 4,569 |
| cached | 16 | 200 x 200 | 9.42 | 1,616 | 1,953 | 2,060 | 99 / 101 | 15.19 | 4,569 |

No 503s and no other non-200 in steady load: with 16 closed-loop clients at most 15 wait, inside the 16-slot queue.

**Throughput vs latency.** Throughput is requests finished per second; latency is how long one request takes. One GPU
runs one request at a time, so once it is never idle (from 4 clients on) more clients cannot raise throughput: it
stays at about 6.5-6.9 req/s cold, 9.4-9.6 cached, and each extra client only adds waiting. Latency grows in step with
the queue: at 16 clients a request waits behind about 15 others, 15 x ~145 ms, so p50 is about 2.4 s.

**Little's law** (mean requests in the system = throughput x mean time in the system) is a check on the client:
3.7-3.8 at 4 clients and 15.0-15.2 at 16, as it should be. At 1 client it is 0.81-0.86 because the client spends the
rest of the time building the next request body, outside the timed span.

**Why p99 sits close to p50 here.** Closed-loop clients never send more than the GPU can take, so the queue length is
fixed and the tail is flat. With open-loop traffic (requests arriving on their own schedule) bursts would build the
queue up and down, and p99 would grow much faster than p50 as load approaches 6.5 req/s.

## Correctness under load

Every 200 response from steady load, the burst and the shutdown phase (1,254 responses, 955 distinct states, 10,032
questions) against `decide()` run serially in-process on the same GPU after the server had exited:
**max probability difference 0.0036; 4 top-option flips**, all on 2 states seen twice each, where the reference's top
two options were 0.0002 and 0.0005 apart (fp16 near ties, not mix-ups). A mixed-up request would answer about another
state and be off by tenths. Concurrency, the queue and the cache did not mix up requests.

## Overload burst

56 requests at once (queue 16 + 1 running, under anyio's 40-thread pool + 16): **17 x 200, 39 x 503, no 500s**.
Every 503 carried `Retry-After: 1` and was counted in `pointwise_overload_rejections_total` (39). The 503s answered in
260 ms median (302 ms max): they are cheap but not instant, because each still needs a pool thread and the event loop
is busy with the burst. The 17 served requests took up to 2.9 s.

## Shutdown

SIGTERM after 3 s of 16-client load: `/readyz` on a fresh connection answered **503 within 30 ms**; all 32 requests
sent before the signal finished with 200 (none sent after, none dropped); the server logged `shutdown: no work in
flight` and the process **exited 2.9 s after the signal**, inside the 25 s grace. Exit status is -15 (uvicorn re-raises
the signal after its own clean shutdown).

## HTTP overhead: not resolved

In-process `decide()` on the server's path (fresh StateCache, tokenising included), run after the server exited:
miss p50 156 ms, hit p50 50 ms. Served cold p50 at 1 client was 136 ms, **20 ms faster** than in-process, so this run
cannot separate HTTP overhead from drift: the in-process timing ran last, after ~3 minutes of continuous load, and the
misses inside the cached rows also came out slower than the cold rows run before them, which fits a T4 clocking down
when hot. The bound that holds: HTTP adds less than the session's drift (~20 ms) to a ~140 ms request. An interleaved
A/B (as in step 4) would be needed to measure it.

## What this means for capacity

- One T4 serves about **6.5 req/s** of new states and **9.5 req/s** when half the states repeat. Every request is
  k=8 on ~526 tokens; fewer questions or shorter states are faster.
- Latency is service time x requests ahead of you. To keep p95 under ~600 ms, keep at most ~4 requests in flight per
  instance (cold p95 559 ms at 4); scale out with instances beyond that, not with the queue.
- The default queue (16, 10 s timeout) admits waits of ~2.4 s before it rejects; anything past it gets a fast 503 with
  Retry-After, never a 500 or a hang.

## Limitations of this test

- Closed-loop clients only; no open-loop arrival rate, so tail behaviour under bursty traffic is not measured beyond
  the one burst.
- The image itself was not run: Kaggle cannot run containers, so this tested the server process and its pins.
- One state length (526 tokens) and k=8; T4 thermal drift within the session is likely (above).
- `src/data.py`'s Banking77 loader runs a dataset script that `datasets` 5 refuses; the load test read the Hub's
  Parquet copy instead. Phase 0 evaluation and the data build need Kaggle's preinstalled `datasets`.
