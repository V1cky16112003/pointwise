# Pointwise on an Apple M1 (smoke test, 2026-10-30)

A functional check, not a performance one. MacBook Air M1, 8 GB, macOS; run 3 (`state.pt` step 1541, T 1.516,
adapter merged); torch 2.12.1 with the MPS backend (Apple's GPU), fp16. No model or serving setting was changed.

## Docker: does not fit, by design
`docker build` worked on arm64, and the container downloaded the base model and loaded all 310 base weight tensors.
Docker on macOS cannot reach the Apple GPU, so the server fell back to CPU, where the model loads in fp32 (about 6.9 GB
for 1.7B parameters). The container was killed right after "Loading weights 100%" with Docker given 7 GB. The CUDA
image is meant for NVIDIA hosts; it is still unrun on a GPU (`docs/production.md`, section 7).

## Native on the M1 GPU (`python -m src.serve_main`)
Code change: the server picks `mps` when there is no CUDA, and loads in fp16 there (`src/serve.py`,
`src/pointwise/train.py`); test `test_startup_picks_the_apple_gpu_when_there_is_no_cuda`.

| Check | Result |
|---|---|
| Start-up log | `Pointwise loaded on mps: step 1541, T=1.516, merged=True` (62 s, model cached) |
| `/metrics` and `/v1/systemone` without the key, or with a wrong one | 401 |
| `/metrics` with the key | 200 (checked on a second start) |
| README example request | 200: department `returns` 0.548 (billing 0.350, shipping 0.102), escalate 0.588, 60 tokens |
| Same request twice, in-process | identical answers (max difference 0.0) |
| Ctrl+C (SIGINT) | `shutdown: no work in flight`, `Application shutdown complete`; the trailing `KeyboardInterrupt` traceback is uvicorn re-raising the signal after its clean exit |

**Latency:** the README request took 9,353 ms, of which 9,176 ms was the model. In-process, on a ~25-line state with
k=8, the first call took 36 s (MPS compiling its kernels), then 7.4, 8.1 and 2.8 s. With the fp16 model (~3.4 GB) and
macOS sharing 8 GB, the machine swaps. A T4 takes about 140 ms. Use an M1 for development and demos, not for serving.

**Not measured:** MPS answers against the T4 on the same request, memory use, and anything under load. The README
example's numbers (0.71 / 0.34) are illustrative (README says so), not a T4 measurement to compare against.
