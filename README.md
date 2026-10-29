# Pointwise

**A small model that answers multiple-choice questions about a piece of text in one forward pass, with a probability
for every option and no generated text.**

You send a *state* (a support chat, a ticket, a message) and up to 32 questions. Each question comes with its own fixed
set of options. Pointwise reads the state once and answers every question from it. Each answer is always one of the
options you declared, so it can never fall outside your schema. The API follows TypeSafe's public `POST /v1/systemone`
contract, so a Jev or Kev client can call it unchanged.

**Status: shipped as a production-ready service.** The served model is Phase 4 run 3 (step 1541, temperature T 1.516,
adapter merged), served by a hardened FastAPI server with:
- request limits;
- an overload queue that answers 503 with `Retry-After`;
- liveness, readiness and Prometheus metrics;
- an optional API key;
- a graceful drain on shutdown;
- a CUDA Docker image.

It has been load-tested on a T4. **Read [`docs/production.md`](docs/production.md) before you deploy it.** That guide
covers every setting, status code, metric and capacity number, plus the known limitations. The most important
limitation: confidence is not well enough calibrated for fully automated decisions.

## Quickstart: call it

Start it with Docker (below), then:

```bash
curl -s localhost:8000/v1/systemone -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also two charges on my card.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Refunds, wrong items", "shipping": null, "billing": "Charges"}},
    "escalate":   {"type": "noul", "instructions": "Does this need urgent human attention?"}
  }}'
```

```json
{"model": "pointwise-latest",
 "answers": {
   "department": {"type": "choice", "choice": "returns", "confidence": 0.71,
                  "probabilities": {"returns": 0.7102, "shipping": 0.1185, "billing": 0.1713}},
   "escalate":   {"type": "noul", "noul": 0.3364}},
 "usage": {"input_tokens": 61, "output_tokens": 0}}
```

The numbers in this response are illustrative. There are three question types:
- `choice`: pick one of the named criteria.
- `noul`: return p(yes).
- `score`: return an expected level, with `score`, `confidence`, `legend` and `probabilities`.

Probabilities are rounded to 4 decimals. Every response carries an `X-Request-Id` header.

## Run it in Docker (one NVIDIA GPU)

**1. Get run 3's weights.** You need two files:
- `state.pt`: the LoRA adapter and the pointer head, step 1541.
- `metrics.json`: the fitted temperature, T 1.516.

The server will not start if the `best_step` in `metrics.json` is not the step stored in the weights. `state.pt` is in
the Kaggle dataset `v1cky1611/pointwise-run3-weights`. Both files are in `artifacts/run3/` on the machine that trained
run 3; they are git-ignored and never baked into the image. On first start, the base model Qwen3-1.7B-Base (about
3.4 GB) downloads from Hugging Face into the mounted cache volume.

```bash
mkdir -p weights
kaggle datasets download v1cky1611/pointwise-run3-weights -p weights --unzip   # state.pt
cp artifacts/run3/metrics.json weights/
```

**2. Build and run.** You need the NVIDIA Container Toolkit and a driver that supports CUDA 12.8 (version 570 or newer).
Use a T4 or better: run 3 used 3,637 MiB after load and peaked at 4,569 MiB under load (`reports/load_test.json`).

```bash
docker build -t pointwise .
export KEY="$(openssl rand -hex 24)"
docker run --rm --gpus 1 -p 8000:8000 --stop-timeout 30 \
  -v "$PWD/weights:/weights:ro" -v pointwise-hf-cache:/home/app/.cache/huggingface \
  -e POINTWISE_WEIGHTS=/weights/state.pt -e POINTWISE_METRICS=/weights/metrics.json \
  -e POINTWISE_API_KEY="$KEY" \
  pointwise
```

Without Docker, run `python -m src.serve_main` with the same env vars, on a host that has the `requirements-serve.txt`
pins installed. Every env var, with its default and its valid range, is in `docs/production.md`, section 3. The most
important ones:

| Variable | Default | What it does |
|---|---|---|
| `POINTWISE_WEIGHTS`, `POINTWISE_METRICS` | required | run 3's `state.pt` and `metrics.json` |
| `POINTWISE_API_KEY` | unset = **no auth** | when set, every path except `/healthz` and `/readyz` needs `Authorization: Bearer <key>`; set it anywhere but your laptop |
| `POINTWISE_MERGE_LORA` | `1` | `1` folds the adapter into the weights (cold p95 233 -> 160 ms, `reports/pointwise-serving.md`); `0` serves it unmerged |
| `POINTWISE_MAX_QUEUE` / `_QUEUE_TIMEOUT_S` | 16 / 10 | requests that may wait for the GPU; past either, the server answers 503 + `Retry-After` |
| `POINTWISE_SHUTDOWN_GRACE_S` | 25 | the drain budget on SIGTERM; keep it under `--stop-timeout` |

**3. Check it.** `/readyz` returns 200 once the model has loaded: 14.6 s on a T4 with the model already cached.
`/healthz` is liveness. `/metrics` serves Prometheus text and needs the key.

**4. Stop it cleanly.** `docker stop <container>` sends SIGTERM. `/readyz` fails at once, so a load balancer stops
routing traffic to the container. New data requests get 503, and requests already running finish within
`POINTWISE_SHUTDOWN_GRACE_S`.

## How it works

- **Backbone:** Qwen3-1.7B-Base with a LoRA adapter (r=16, on attention and MLP). By default the adapter is merged into
  the weights when the server starts.
- **Pointer head:** a small head reads the hidden state at a `<decide>` token. It scores that against the hidden state
  at the end of each option, and a softmax over one question's options gives that question's probabilities.
  Temperature scaling (T 1.516, fitted on dev data) softens over-confident scores.
- **Parallel questions:** the state is encoded once, and all the questions are answered from it in one pass, using a
  block attention mask (`src/pointwise/packing.py`).
- **State cache:** the server keeps the key/value tensors of recent states. A repeated state then costs only its
  questions (in-process p95 58 ms instead of 160 ms, `reports/pointwise-serving.md`).
- **Design:** `docs/designs/pointwise-architecture.md` gives the architecture and the evidence behind each choice;
  `docs/designs/pointwise-plan.md` gives the phases, gates and run logs. The design is reconstructed from public clues
  and from Kev, an open Jev-like model by Jared Palmer. TypeSafe has not published Jev's internals.

## Results

Accuracy and calibration were measured on 2,500 held-out questions, 500 per source, with paired bootstrap 95%
intervals (`reports/pointwise-results.md`).

| Source | Accuracy | ECE after T |
|---|---|---|
| QNLI | 0.818 | 0.082 |
| PAWS | 0.750 | 0.130 |
| Emotion | 0.492 | 0.159 |
| TweetEval sentiment | 0.652 | 0.038 |
| Banking77 (zero-shot, never trained on) | 0.626 | 0.113 |
| **Pooled out-of-domain** | **0.678** | **0.056** |

*ECE* (expected calibration error) measures how far the stated confidence is from the actual accuracy. Lower is
better.

**Wins:**
- **Against its own untrained base:** +3.8 points pooled out-of-domain accuracy (CI [+1.8, +5.8]).
- **On two sources:** +29.8 on Banking77 and +20.6 on TweetEval.
- **Option order:** shuffling the options changes the answer far less often (21% vs 61.5% on Banking77).
- **Calibration:** ECE after T is 0.056, against 0.109 for the base.

**Losses:**
- **Emotion:** -8.0 points against the untrained base.
- **Against untrained Qwen2.5-1.5B-Instruct:** PAWS -5.8 and Emotion -4.4.
- **Calibration target:** missed (0.056 against 0.05).
- **Ranking by confidence:** nearly useless. Cov@5% is 0.004, against 0.171 for Qwen2.5-Instruct.

**Serving on one T4**, with k=8 questions on one state.

One request at a time, in-process (no HTTP), on a 524-token state (`reports/pointwise-serving.md`):

| Path | p95 |
|---|---|
| new state | 160.4 ms (the 150 ms target is missed by 10.4) |
| cached state | 58.4 ms |

Through HTTP, with closed-loop clients on a 526-token state (`reports/pointwise-load.md`). Each client sends its next
request as soon as the last one answers.

| Mix | Clients | Throughput | p95 |
|---|---|---|---|
| all new states | 1 | 5.94 req/s | 141 ms |
| all new states | 4 | 6.91 req/s | 559 ms |
| all new states | 16 | 6.52 req/s | 2,434 ms |
| half repeated | 4 | 9.60 req/s | 590 ms |

Past about 4 clients, throughput stops rising and each extra client only adds waiting.

**Under load:**
- **Correctness:** every answer matched the serial reference, with a maximum probability difference of 0.0036 over
  10,032 questions.
- **Overload:** of a burst of 56 requests, 17 got 200 and 39 got 503 with `Retry-After` (260 ms median); none got a 500.
- **Shutdown:** a SIGTERM under load finished every in-flight request and exited in 2.9 s.

**Token efficiency** (`reports/pointwise-tokens.md`): 8 questions on one state take 680 input tokens, against 4,365
for generate-and-parse (6.4x fewer). On a cached state they take 156 tokens. Pointwise generates no tokens at all.

The Phase 0 read-out experiment (Banking77 label scoring: accuracy 0.918, ECE 0.021) and the four training runs are
logged in `docs/designs/`.

## Known limitations

The full table, with every number and its source, is in `docs/production.md`, section 7. In short:
- **Calibration:** confidence is approximately calibrated (ECE 0.056) but misses the 0.05 gate, and it cannot pick out
  a low-error subset. Keep a human or a later check in the loop.
- **Weak tasks:** emotion and paraphrase tasks are weak.
- **Long states:** accuracy on states longer than the 384 tokens it was trained on was never measured. The server's limit allows 4,096.
- **Cold latency:** cold p95 is 160 ms, against the 150 ms target.
- **Untested:** the Docker image itself has never run on a GPU (the same server process and pins have), and
  open-loop load is untested.
- **Evidence base:** one training run with one seed; calibration differences have no confidence intervals.

## Repo layout

```
src/pointwise/          the model: schema (API contract), encoding, packing, pointer head, training, metrics
src/pointwise/data/     training-data converters (public sources, policy generator, rule records)
src/serve.py            FastAPI: /v1/systemone, /healthz, /readyz, /metrics (and the Phase 0 /predict demo)
src/serve_ops.py        request ids, API key, drain flag, metrics middleware
src/serve_main.py       `python -m src.serve_main`: the one-process runner that drains on SIGTERM
Dockerfile              CUDA 12.8 image for one GPU
docs/production.md      the operator's guide
docs/designs/           architecture, plans and run logs for every phase
reports/                evaluation, latency, load and token reports, with their raw JSON
scripts/                Kaggle job, evaluation, latency profiling, load test, token counts
tests/                  CPU tests (tiny random models, no GPU)
```

The repo also holds the original Phase 0 project, a LoRA vs QLoRA Banking77 classifier (`src/train.py`,
`src/evaluate.py`, `configs/task.yaml`, `demo/`; design in `docs/designs/lora-finetuning-banking-intent-classifier.md`).
The code from before Pointwise is kept on the `legacy-main` branch.

## Development

**Tests** run on the CPU. Install torch from the index that matches your machine (see the header of
`requirements-serve.txt`):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-serve.txt
pytest tests -q          # about 300 passed, 1 skipped
```

If the count is about half that, the serving tests were skipped. They need `peft`, `httpx` and the
`Qwen/Qwen2.5-1.5B-Instruct` tokenizer in the Hugging Face cache. `tests/test_production_doc.py` checks that the env var
table in `docs/production.md` matches the code.

**GPU jobs** run on Kaggle (T4 x2). The runner notebook `notebooks/kaggle_pointwise_job.ipynb` clones the repo and runs
`scripts/kaggle_job.py`. That script decides which job runs, so a new job needs no notebook edit. To run one, use Save
Version → Save & Run All.
