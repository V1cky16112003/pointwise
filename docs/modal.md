# Deploying Pointwise on Modal

One T4, scales to zero. The app is `deploy/modal_app.py`; it starts `python -m src.serve_main` (so the startup checks and
the shutdown drain in `docs/production.md` still apply) behind `@modal.web_server(8000)`.

## What it sets up

| Piece | Name | Holds |
|---|---|---|
| Secret | `pointwise-key` | `POINTWISE_API_KEY` |
| Volume | `pointwise-weights` | `state.pt` and `metrics.json` from the same run (the server refuses a mismatched pair) |
| Image | built from `requirements-serve.txt`, torch 2.11.0 (`cu128`) | the code, plus the Qwen3-1.7B-Base download, baked in at build time |
| App | `pointwise`, GPU `T4`, `min_containers=0`, `max_containers=1`, `scaledown_window=60`, `max_inputs=16` | one server; the server has a single GPU lock and cache, so never raise `max_containers` |

## Deploy

```bash
pip install modal && modal setup
export KEY=$(openssl rand -hex 24)                       # keep it; never paste it into logs or chats
modal secret create pointwise-key POINTWISE_API_KEY=$KEY
modal volume create pointwise-weights
modal volume put pointwise-weights weights/state.pt /state.pt
modal volume put pointwise-weights weights/metrics.json /metrics.json
modal serve deploy/modal_app.py      # temporary URL, stops with Ctrl-C
modal deploy deploy/modal_app.py     # permanent URL
```

Use the URL that ends in `.modal.run`, not the `modal.com/apps/...` dashboard link (that one returns "Method Not Allowed").
`state.pt` is in Git LFS: `ls -lh weights/state.pt` must show about 74 MB, not a pointer file.

## Check it

`/healthz` answers 200, `POST /v1/systemone` with the key answers 200, and with a wrong key answers 401. Use
`-d @file.json` for the body (a body that starts with `@` is read as a file name by curl).

## Measured (one session, 2026-11-03; the request body in the README quickstart, 60 input tokens)

| What | Result |
|---|---|
| Cold start (container asleep, `GET /healthz`) | 19.4 s in Modal's log; 21.0 s from a Mac for a full request |
| First request after the cold start, server side | 702 ms (state cache miss, GPU just started) |
| Same request again, from a Mac | 0.51 s total, which includes the network |

These are single samples, not a distribution. The load numbers in `docs/production.md` (about 2 req/s per T4) were
measured on Kaggle, not on Modal.

## Run and update

| Task | Command |
|---|---|
| Logs | `modal app logs pointwise` |
| What is running | `modal container list` |
| Switch off | `modal app stop pointwise` |
| New code | `modal deploy deploy/modal_app.py` |
| New weights | `modal volume put pointwise-weights <file> /<file> --force`, for both files |
| New key | `modal secret create pointwise-key POINTWISE_API_KEY=<new> --force`, then redeploy |

## Cost and limits

- A T4 is $0.000164 per second (about $0.59 per hour), billed only while a container runs. Set a spend limit in the Modal
  dashboard; `max_containers=1` caps the rate.
- Anyone with the URL can wake the GPU (`/healthz` needs no key), which costs money. Keep the URL private, and rely on the
  spend limit as the backstop.
- `scaledown_window=60`: a user arriving after a minute idle waits about 20 s. A longer window means fewer waits and more
  idle GPU seconds.
- Not checked: whether `serve_main`'s shutdown drain runs here. It starts as a child process of the Modal function, not as
  PID 1, so a scale-down may not reach it.
