# Pointwise in production: the operator's guide

This guide covers what the service is, how to run it, how to size it, how to operate it, and what it cannot do. It
describes Phase 4 run 3 as shipped: `state.pt` step 1541, temperature T 1.516, with the adapter merged by default.
Every number below is copied from a report or JSON file in `reports/`, and each one names its source. Nothing here was
re-measured for this guide.

## 1. What it is, and what it is not for

You send Pointwise a piece of text (the *state*, such as a support chat) and up to 32 multiple-choice questions about
it. It answers every question in one forward pass of a small model (Qwen3-1.7B-Base plus a LoRA adapter and a pointer
head). Each answer is a probability over the options you declared, and no text is generated. The request and response
follow TypeSafe's public `/v1/systemone` contract, defined in [`src/pointwise/schema.py`](../src/pointwise/schema.py),
so a Jev or Kev client can call it unchanged. There are three question types:

- `choice`: pick one of the named criteria.
- `noul`: return p(yes).
- `score`: return the expected level on an ordered scale.

**Uses it fits:** routing and triage where a wrong answer is cheap to correct. A human or a later step should check the
outcome.

**Uses it must not have:**
- **Automated decisions that rely on the confidence number.** Calibration missed its gate: pooled out-of-domain ECE is
  0.056, against a target of 0.05.
- **"Answer only when confident" filtering.** Ranking answers by confidence barely works (Cov@5% is 0.004).
- **Emotion or paraphrase tasks**, where it loses to untrained baselines.
- **States longer than the 384 tokens it was trained on.** No one has measured accuracy on them.

See section 7 for the numbers and their sources.

## 2. Run it

**Weights.** You need two files from run 3:
- `state.pt`: the LoRA adapter and the pointer head, step 1541.
- `metrics.json`: holds the fitted T 1.516.

The server refuses to start in either of these cases:
- the `best_step` in `metrics.json` is not the step stored in `state.pt`;
- either file is missing, either file is unreadable, or T is not positive.

Both files are mounted at run time and are never baked into the image. To get them, follow README, "Run it in Docker",
step 1.

**Base model and the Hugging Face cache.** On first start, Qwen3-1.7B-Base (about 3.4 GB) downloads into `HF_HOME`.
Keep that directory on a persistent volume so that a restart does not download it again. In the step 7 test, start-up
took **14.6 s** from launch to `/readyz` 200 with the model already cached. The download alone took 21.9 s on Kaggle's
network (`reports/pointwise-load.md`, "First real start").

**GPU.** You need one NVIDIA GPU, a T4 or better. The host driver must support CUDA 12.8 (driver 570 or newer, see the
`Dockerfile` header). Memory use was measured on a T4 (`reports/pointwise-load.md`):
- **3,637 MiB** after the model loads;
- a peak of **4,569 MiB** (about 4.6 GB) under the full load test with the default cache.

Budget at least 5 GB of free GPU memory per instance. Run one instance per GPU. The process holds a single GPU lock, a
single queue and a single state cache, so it must run as one worker.

**Docker.** The README section "Run it in Docker" has the build command, the `docker run` line, an example call and a
clean stop. The image runs `python -m src.serve_main` as PID 1. Its built-in `HEALTHCHECK` probes `/healthz`, with a
start period long enough for the first download.

**Bare process.** On a host that has the `requirements-serve.txt` pins and torch 2.11.0+cu128:

```bash
POINTWISE_WEIGHTS=weights/state.pt POINTWISE_METRICS=weights/metrics.json \
POINTWISE_API_KEY="$(openssl rand -hex 24)" POINTWISE_HOST=0.0.0.0 \
python -m src.serve_main
```

**Apple silicon (development only).** With no CUDA, the server uses the Apple GPU (MPS) in fp16 and runs the same
bare-process command. It works but is slow: about 9 s per request on an 8 GB M1, against about 140 ms on a T4
(`reports/pointwise-mac-smoke.md`). The Docker image cannot use the Apple GPU, and on CPU it needs about 7 GB just for
the fp32 model.

Always start it with `python -m src.serve_main`, not with plain `uvicorn src.serve:app`. Only `serve_main` makes
`/readyz` fail the moment a shutdown signal arrives, and only it keeps the whole shutdown within
`POINTWISE_SHUTDOWN_GRACE_S`.

## 3. Configuration

The server reads every variable once, at start-up. A bad value stops it before it listens, with a line that starts
with `FATAL:`. The table below is checked against the code by `tests/test_production_doc.py`, in both directions,
together with each default.

| Variable | Default | Valid range | Past it |
|---|---|---|---|
| `POINTWISE_WEIGHTS` | unset | path to run 3's `state.pt`; this or `ADAPTER_DIR` is required | FATAL at start if missing or unreadable |
| `POINTWISE_METRICS` | unset | path to the `metrics.json` that matches the weights; required with `POINTWISE_WEIGHTS` | FATAL at start if missing, if its `best_step` is another step, or if T <= 0 |
| `POINTWISE_DEVICE` | unset | a torch device; unset means `cuda` if available, else `cpu` | an invalid device fails at start |
| `POINTWISE_MERGE_LORA` | `1` | `0` or `1`; `1` folds the adapter into the weights | FATAL at start for any other value |
| `POINTWISE_API_KEY` | unset | non-empty ASCII with no surrounding whitespace; unset means **no auth** | FATAL at start if invalid; 401 for a missing or wrong key on every path except `/healthz` and `/readyz` |
| `POINTWISE_MAX_STATE_TOKENS` | `4096` | integer >= 1 and <= `POINTWISE_MAX_REQUEST_TOKENS` | FATAL at start; per request, 422 `context_overflow` |
| `POINTWISE_MAX_REQUEST_TOKENS` | `8192` | integer >= `POINTWISE_MAX_STATE_TOKENS` (the state plus every question) | FATAL at start; per request, 422 `context_overflow` |
| `POINTWISE_MAX_QUESTIONS` | `32` | integer >= 1 | FATAL at start; per request, 422 `too_many_questions` |
| `POINTWISE_MAX_BODY_BYTES` | `1048576` | integer >= 1 | FATAL at start; per request, 413 `body_too_large`, sent before the body is parsed |
| `POINTWISE_MAX_QUEUE` | `16` | integer from 0 to 24 (0 means never wait); requests that may wait for the GPU | FATAL at start; per request, 503 `overloaded` with `Retry-After: 1` when the queue is full |
| `POINTWISE_QUEUE_TIMEOUT_S` | `10` | number > 0 | FATAL at start; per request, 503 `overloaded` with `Retry-After: 1` after waiting this long |
| `POINTWISE_CACHE_ENTRIES` | `8` | integer >= 1; how many states the KV cache holds | FATAL at start |
| `POINTWISE_CACHE_TOKENS` | `16384` | integer >= 1; total state tokens cached | FATAL at start; a single state longer than this is answered uncached |
| `POINTWISE_STUCK_AFTER_S` | `60` | number > 0 | `/readyz` answers 503 once one request has held the GPU this long |
| `POINTWISE_SHUTDOWN_GRACE_S` | `25` | number >= 3; the whole shutdown budget, which must stay below the orchestrator's kill timeout | FATAL at start |
| `POINTWISE_HOST` | `127.0.0.1` | bind address; the image sets `0.0.0.0` | n/a |
| `POINTWISE_PORT` | `8000` | integer from 1 to 65535 | FATAL at start |
| `ADAPTER_DIR` | unset | the Phase 0 Banking77 classifier at `POST /predict`; a demo, not part of the product | FATAL at start if missing, empty or unloadable |
| `HF_HOME` | `/home/app/.cache/huggingface` | read by Hugging Face, not by our code; the image's value is shown, and the library falls back to `~/.cache/huggingface` | n/a |

## 4. Endpoints and status codes

| Endpoint | Auth | What it answers |
|---|---|---|
| `POST /v1/systemone` | key, if set | the decision (schema in `src/pointwise/schema.py`) |
| `GET /healthz` | never | **liveness**: 200 `{"status":"ok"}` whenever the process answers. It never touches the GPU. |
| `GET /readyz` | never | **readiness**: 200 when this instance should get traffic. It answers 503 when no model is loaded, after a shutdown signal, or when one request has held the GPU past `POINTWISE_STUCK_AFTER_S`. The reason goes to the log, not to the response. |
| `GET /metrics` | key, if set | Prometheus text format; it still answers while the server drains |

*Liveness vs readiness:* liveness asks "is the process alive, or should it be restarted?". Readiness asks "should
traffic go here right now?". A server that is loading, draining or stuck is alive but not ready. Restarting it would
not help, and it should simply get no traffic. So point restarts at `/healthz` and routing at `/readyz`.

**Status codes on `/v1/systemone`:**

| Code | Body `detail.error` | When |
|---|---|---|
| 200 | none | answered |
| 401 | `unauthorized` | the key is set, and the `Authorization: Bearer <key>` header is missing or wrong (sent with `WWW-Authenticate: Bearer`) |
| 413 | `body_too_large` | the body is over `POINTWISE_MAX_BODY_BYTES`, whether the size is declared or the body is streamed |
| 422 | pydantic errors, `context_overflow`, `too_many_questions`, `invalid_request` | a malformed body, a request over a token limit, too many questions, or caller text the encoder refuses |
| 503 + `Retry-After: 1` | `overloaded` | the GPU queue is full, or the request waited `POINTWISE_QUEUE_TIMEOUT_S` |
| 503 + `Retry-After: 1` | `shutting_down` | the server is draining after a shutdown signal |
| 503 | `pointwise_not_configured` | the server started without `POINTWISE_WEIGHTS` |
| 500 | `decision_failed` / `internal_error` | an unexpected error; the body never carries caller text or exception text |

**Request ids.** Each request gets an id:
- If the caller sends an `X-Request-Id` of 1-64 characters from `[A-Za-z0-9._-]`, the server keeps it.
- Otherwise the server generates a fresh one.

The id comes back in the `X-Request-Id` response header and appears on every log line made while handling the
request. Use it to match a client's error to a log line.

**What is logged: metadata only, never text.**
- Each data request produces a line with its method, route, status and milliseconds.
- Each `/v1/systemone` answer also logs its question count, input tokens, whether the state cache hit, and its time.
- Errors log the exception type and the file:line frames. They never log the message, which could hold caller text.

The body, the state, the questions and the query string are never logged. Failed-auth lines are capped at 10 per 10 s,
followed by a count of how many were left out. Probes and scrapes are logged only when they fail.

**Metrics** (`/metrics`):

| Metric | Type | Meaning |
|---|---|---|
| `pointwise_requests_total{route,status}` | counter | HTTP requests by route and status |
| `pointwise_request_duration_seconds{route}` | histogram | time to answer, with buckets 0.025-12.8 s |
| `pointwise_ready` | gauge | 1 if `/readyz` would answer 200 |
| `pointwise_queue_waiting` | gauge | requests waiting for the GPU |
| `pointwise_gpu_busy_seconds` | gauge | how long the current request has held the GPU (0 when idle) |
| `pointwise_overload_rejections_total` | counter | overload 503s |
| `pointwise_state_cache_hits_total` / `_misses_total` | counter | state cache hits and misses |
| `pointwise_state_cache_entries` / `_tokens` | gauge | states and state tokens currently cached |

## 5. Capacity and latency (measured on one T4)

All the numbers below are for k=8 three-option questions on a single state of about 525 tokens. Fewer questions or
shorter states are faster.

**One request, in-process** (`decide()` with no HTTP; `reports/pointwise-serving.md`, "Step 4 result", which uses
`reports/coldpath_ab.json`):

| Path (merged, the default) | median ms | p95 ms |
|---|---|---|
| cold (no cache) | 143.7 | 160.4 |
| cache miss (state stored for later) | 149.7 | 170.1 |
| cache hit (state seen before) | 52.2 | 58.4 |

*p95* means 95% of requests finished within that time. It is the usual way to state latency, because the median hides
the slow tail.

**Through HTTP, closed-loop clients** (each client sends its next request as soon as the previous one answers;
`reports/pointwise-load.md`, "Steady load", which uses `reports/load_test.json`):

| Mix | Clients | Throughput req/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| all new states | 1 | 5.94 | 136 | 141 | 143 |
| all new states | 4 | 6.91 | 542 | 559 | 568 |
| all new states | 16 | 6.52 | 2,360 | 2,434 | 2,439 |
| half repeated | 1 | 8.16 | 148 | 156 | 158 |
| half repeated | 4 | 9.60 | 390 | 590 | 619 |
| half repeated | 16 | 9.42 | 1,616 | 1,953 | 2,060 |

No request in these runs got a status other than 200. Answers under load matched serial `decide()` within 0.0036.
There were 4 top-option flips, all on near ties with margins of 0.0005 or less.

**Overload burst** (`reports/pointwise-load.md`, "Overload burst"): of 56 requests sent at once, 17 got 200 and 39 got
503 with `Retry-After: 1`. None got a 500. The 503s took 260 ms median and 302 ms max.

**Shutdown** (`reports/pointwise-load.md`, "Shutdown"): SIGTERM was sent under 16-client load. `/readyz` answered 503
within 30 ms. All 32 in-flight requests finished with 200, and the process exited 2.9 s after the signal (grace 25 s).

**Start-up and memory:** 14.6 s to ready with the model cached; 3,637 MiB after load and a 4,569 MiB peak
(`reports/pointwise-load.md`, "First real start").

**Sizing rule.** One GPU runs one request at a time. Throughput therefore tops out at about **6.5 req/s** of new states
and about **9.5 req/s** when half the states repeat. Past that point, each extra request in flight only adds waiting.
Latency is roughly the service time multiplied by the number of requests ahead of yours. So decide on the p95 you will
accept, and cap the requests in flight per instance to match:

| Acceptable p95 | Max requests in flight per instance | From |
|---|---|---|
| ~150 ms | 1 | p95 141 ms at 1 client (new states) |
| ~600 ms | 4 | p95 559 ms at 4 clients (new states) |
| ~2.5 s | 16 | p95 2,434 ms at 16 clients (new states) |

Add instances to serve more traffic; a longer queue only makes requests wait longer. Instances needed ≈ peak req/s ÷
6.5 (all new states), plus headroom. The default queue (16 waiting, 10 s timeout) admits waits of about 2.4 s before it
starts answering 503 quickly.

## 6. Operating it

**Probes.** Shutdown order with these settings: `/readyz` fails within 30 ms of SIGTERM. The server finishes the work
already running, within `POINTWISE_SHUTDOWN_GRACE_S` (25 s by default). Kubernetes kills the pod only after
`terminationGracePeriodSeconds`. **Keep the grace below the kill timeout**, or requests are cut off mid-answer.

For Kubernetes:

```yaml
terminationGracePeriodSeconds: 30        # > POINTWISE_SHUTDOWN_GRACE_S (25)
startupProbe:   {httpGet: {path: /healthz, port: 8000}, periodSeconds: 10, failureThreshold: 90}  # 15 min for the first download
livenessProbe:  {httpGet: {path: /healthz, port: 8000}, periodSeconds: 15, timeoutSeconds: 3, failureThreshold: 3}
readinessProbe: {httpGet: {path: /readyz,  port: 8000}, periodSeconds: 5,  timeoutSeconds: 2, failureThreshold: 1}
```

Before the model has loaded, nothing is listening on the port. The startup probe covers that time, up to the
Dockerfile's 900 s start period. Point liveness at `/healthz` only. A busy GPU must not get the pod restarted, and
`/readyz` already takes a stuck instance out of rotation.

For a plain load balancer, health-check `/readyz` every 5 s, mark the instance down after 1 failure and up after 2
successes. With Docker, use `docker stop` (README step 4) and give it `--stop-timeout` above the grace.

**Alerting.** An *SLO* (service-level objective) is a target you promise, for example "99% of requests answered, with
p95 under 600 ms". Alert when you are breaking it, not on every individual error. These are suggested starting points:

| Alert on | Expression (PromQL sketch) | Why |
|---|---|---|
| instance not ready | `pointwise_ready == 0` for 2 min | draining, failed load, or stuck GPU |
| stuck GPU | `pointwise_gpu_busy_seconds > 10` | a single request normally takes about 0.15 s |
| overload | `rate(pointwise_overload_rejections_total[5m]) > 0` | clients are getting 503s: add instances |
| errors | `rate(pointwise_requests_total{status=~"5..",route="/v1/systemone"}[5m])`, excluding overload | 500s are bugs |
| latency SLO | p95 from `pointwise_request_duration_seconds_bucket` above your target | load past the sizing rule |
| queue building | `pointwise_queue_waiting > 4` for 5 min | in flight is past the ~600 ms point |
| auth noise | `rate(pointwise_requests_total{status="401"}[5m])` high | a misconfigured client or probing |

**Deploy.** Use rolling replacement:
1. Start the new instance.
2. Wait for `/readyz` 200.
3. Send it a known request and check the answer.
4. Only then send SIGTERM to an old instance.

At start-up the log should say `Pointwise loaded on cuda: step 1541, T=1.516, merged=True`. Any other step or T means
the wrong files are mounted.

*Rollback* means going back to the last version that was known to work, as fast as possible, and finding out why
later. Keep the previous image tag and the previous `state.pt` and `metrics.json` pair together. Swap them back with
the same rolling procedure. The server will not start with a `metrics.json` from other weights, so a wrong pairing
fails loudly instead of serving bad answers.

Two rollbacks need no new artefacts:
- **`POINTWISE_MERGE_LORA=0`** serves the adapter unmerged. That is the configuration T was fitted on. It is about 31%
  slower cold (p95 233.4 ms vs 160.4 ms, `reports/pointwise-serving.md`, "Step 4 result"). Merged and unmerged agree on
  the top option for 99.96% of 2,500 held-out questions (`reports/merged_check.json`). Use it if you suspect the merge.
- **The previous image tag**, for any code regression.

## 7. Known limitations

| Limitation | Number | Source |
|---|---|---|
| Calibration gate missed | pooled OOD ECE after T 0.056 (0.0561) vs target 0.05; merged on the served path 0.0576 | `reports/pointwise-results.md`, `reports/merged_check.json` |
| Weak confidence ranking | Cov@5% on pooled OOD 0.004 (Qwen2.5-1.5B-Instruct 0.171): confidence cannot pick a low-error subset | `reports/pointwise-results.md` |
| Loses on Emotion and PAWS | Emotion -8.0 points [-11.8, -4.2] vs its own untrained base; vs Qwen2.5-1.5B-Instruct, PAWS -5.8 [-9.6, -2.0] and Emotion -4.4 [-8.0, -1.0] | `reports/pointwise-results.md` |
| Long states unvalidated | trained on states <= 384 tokens (`--max-state` 384); latency measured at 524-526 tokens; accuracy above the 384 tokens it was trained on never measured, although the limit allows 4,096 | `src/pointwise/train.py`, `reports/pointwise-load.md` |
| Cold p95 over the 150 ms target | 160.4 ms, 10.4 ms over (cache hit 58.4 ms meets it) | `reports/pointwise-serving.md` |
| Cache miss memory shortcut | a miss builds the full L x L mask, so a miss costs as much memory as a cold request; one 524-token state stores 57.3 MiB of KV | `src/pointwise/packing.py`, `reports/pointwise-serving.md` |
| `/metrics` not rate-limited | protected only by the API key; scrape it from inside your network | `src/serve.py` |
| GPU lock not FIFO | waiters are not served in arrival order, so one unlucky request can wait up to `POINTWISE_QUEUE_TIMEOUT_S` (10 s) while later ones go first | `src/serve.py` (`threading.Lock`) |
| Image never run on a GPU | the image is built and import-checked in CI only; the server process and its pins ran on a T4 | `reports/pointwise-load.md`, "Limitations of this test" |
| Open-loop load untested | closed-loop clients only, plus one 56-request burst; p99 under bursty arrivals is unknown | `reports/pointwise-load.md` |
| HTTP overhead unmeasured | the run could only bound it below the session's drift of about 20 ms | `reports/pointwise-load.md`, "HTTP overhead" |
| `src/data.py` broken under `datasets` 5 | the Banking77 loader runs a dataset script that `datasets` 5 refuses; the Phase 0 evaluation and the data build need Kaggle's preinstalled `datasets`. Serving is unaffected. | `reports/pointwise-load.md` |

Each result above comes from a single training run with one seed. Calibration differences have no confidence interval
(`reports/pointwise-results.md`).
