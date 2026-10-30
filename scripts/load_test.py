"""Step 7: HTTP load test of the real server on one T4 (docs/designs/pointwise-plan.md). Measures; changes nothing.

1. Start `python -m src.serve_main` on run 3's weights (merged, the default) and time it until /readyz is 200.
   Check /healthz, /readyz and /metrics, and the API key (401 without it, 200 with it).
2. Steady load: CLIENTS concurrent closed-loop clients (each sends its next request when the last one answers), at
   1, 4 and 16, with two mixes: every state new (cold) and half the requests repeating a recent state (cached).
   PER_LEVEL requests each; k=8 three-option questions on ~512-token states. /metrics is read before and after each
   run, so the hit rate reported is the one the server saw, not the one intended.
3. Overload: BURST requests at once, more than the queue (16) holds. Expect fast 503s with Retry-After, no 500s.
4. Shutdown: SIGTERM while 16 clients are busy. /readyz must fail at once, requests already sent must finish (200)
   or get a 503, and the process must exit inside POINTWISE_SHUTDOWN_GRACE_S.
With --open-loop, step 2 and 3 are replaced by open-loop traffic: requests arrive on their own Poisson schedule at
   RATES req/s (OPEN_LOOP_S seconds each, every state new), whether or not earlier ones have answered, so bursts build
   the queue up and down as real traffic does.
5. After the server has exited, the same weights in-process on the same GPU: every 200's answers against
   decide() run serially (correctness), and decide()'s own latency on the path the server takes (HTTP overhead).

Usage: python -m scripts.load_test --weights <state.pt> --metrics <metrics.json> --output-dir <dir> [--open-loop]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

CLIENTS = (1, 4, 16)  # fixed order: the GPU allocator never hands memory back, so peak memory is cumulative
PER_LEVEL = 200
REPEAT_SHARE = 0.5  # the "cached" mix
RECENT = 4  # repeats are drawn from the last few new states, well inside the cache's 8 entries
BURST = 56  # above queue 16 + 1 running; below anyio's 40-thread pool + 16 would let some wait unseen, so stay near it
K = 8
STATE_TOKENS = 512
API_KEY = "load-test-key"
GRACE_S = 25.0
PORT = 8000
REFERENCE_REPS = 60
RATES = (2.0, 4.0, 6.0, 8.0)  # around the measured cold capacity of ~6.5 req/s on one T4
OPEN_LOOP_S = 60.0


def percentile(samples: list[float], q: float) -> float:
    """Nearest-rank percentile, as phase6_latency.summarise: the smallest sample with at least q% at or below it."""
    ordered = sorted(samples)
    return ordered[max(math.ceil(q / 100 * len(ordered)), 1) - 1]


def summarise(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"n": 0}
    return {"n": len(latencies_ms), "mean": statistics.fmean(latencies_ms),
            **{f"p{q}": percentile(latencies_ms, q) for q in (50, 95, 99)}, "max": max(latencies_ms)}


def request_mix(n: int, repeat_share: float, seed: int, first_new: int = 0) -> list[int]:
    """n state ids: a new id (never sent before) or, with probability repeat_share, one of the last RECENT new ids.
    New ids start at first_new so separate runs never share a state. The first request is always new."""
    rng, ids, recent, nxt = random.Random(seed), [], [], first_new
    for _ in range(n):
        if recent and rng.random() < repeat_share:
            ids.append(rng.choice(recent))
        else:
            ids.append(nxt)
            recent = (recent + [nxt])[-RECENT:]
            nxt += 1
    return ids


def make_state(i: int, messages: list[str], count_tokens, target: int = STATE_TOKENS) -> str:
    """A support chat of about `target` tokens, unique to i: a ticket header, then messages from a rotating offset."""
    lines, j = [f"Ticket {i}"], (i * 7) % len(messages)
    while count_tokens("\n".join(lines)) < target:
        lines.append(f"Customer: {messages[j % len(messages)]}")
        j += 1
    return "\n".join(lines) + "\n"


def payload(state: str, k: int = K) -> dict:
    from scripts.bench_decide import shared_state_request

    return shared_state_request(state, k)


def compare_answers(served: dict, reference: dict) -> dict:
    """Two responses' answers (both rounded as served). A mixed-up request answers about another state, so its
    probabilities are off by tenths; float noise stays near 1e-3. A flip records the reference's top-two margin,
    so a near tie is not mistaken for a bug."""
    diff, flips = 0.0, []
    for key, ref in reference.items():
        got = served[key]
        diff = max([diff] + [abs(got["probabilities"][o] - p) for o, p in ref["probabilities"].items()])
        if got["choice"] != ref["choice"]:
            top = sorted(ref["probabilities"].values(), reverse=True)
            flips.append({"question": key, "margin": top[0] - top[1]})
    return {"max_prob_diff": diff, "flips": flips, "questions": len(reference)}


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {'name{labels}': value}, comments skipped."""
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            out[name] = float(value)
    return out


def metrics_delta(before: dict, after: dict) -> dict:
    """The counters the report needs, as after minus before."""
    keys = [k for k in after if k.startswith(("pointwise_state_cache_hits_total", "pointwise_state_cache_misses_total",
                                              "pointwise_overload_rejections_total"))
            or re.match(r'pointwise_requests_total\{route="/v1/systemone"', k)]
    return {k: after[k] - before.get(k, 0.0) for k in keys}


async def send(client: httpx.AsyncClient, state_id: int, body: dict, headers: dict) -> dict:
    """One request -> a record. Never raises: a transport error is recorded as status None with its class name."""
    start = time.perf_counter()
    try:
        r = await client.post("/v1/systemone", json=body, headers=headers)
    except httpx.HTTPError as exc:
        return {"state": state_id, "status": None, "error": type(exc).__name__, "sent": start,
                "ms": 1000 * (time.perf_counter() - start)}
    record = {"state": state_id, "status": r.status_code, "sent": start, "ms": 1000 * (time.perf_counter() - start),
              "retry_after": r.headers.get("retry-after")}
    if r.status_code == 200:
        record["answers"] = r.json()["answers"]
    return record


async def closed_loop(client, state_ids: list[int], body_for, clients: int, headers: dict,
                      stop: asyncio.Event | None = None) -> tuple[list[dict], float]:
    """`clients` workers share one list of requests; each sends its next one when the last answers.
    -> (records, wall seconds). With `stop`, workers stop taking new requests once it is set."""
    queue, records = list(reversed(state_ids)), []

    async def worker():
        while queue and not (stop and stop.is_set()):
            i = queue.pop()
            records.append(await send(client, i, body_for(i), headers))

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(clients)))
    return records, time.perf_counter() - start


def arrival_times(rate: float, duration_s: float, seed: int) -> list[float]:
    """Poisson arrivals: exponential gaps with mean 1/rate, as many as fall inside duration_s."""
    rng, t, out = random.Random(seed), 0.0, []
    while (t := t + rng.expovariate(rate)) < duration_s:
        out.append(t)
    return out


async def open_loop(client, state_ids: list[int], times: list[float], body_for, headers: dict) -> tuple[list[dict], float]:
    """Send state_ids[i] at times[i] seconds from now, never waiting for answers. -> (records, wall seconds).
    Bodies are built before the clock starts so building one never delays an arrival."""
    bodies = [body_for(i) for i in state_ids]
    start = time.perf_counter()

    async def at(t, i, body):
        await asyncio.sleep(max(0.0, start + t - time.perf_counter()))
        return await send(client, i, body, headers)

    records = await asyncio.gather(*(at(t, i, b) for t, i, b in zip(times, state_ids, bodies)))
    return list(records), time.perf_counter() - start


def level_row(records: list[dict], wall_s: float, clients: int) -> dict:
    ok = [r["ms"] for r in records if r["status"] == 200]
    statuses: dict[str, int] = {}
    for r in records:
        key = str(r["status"] if r["status"] is not None else r["error"])
        statuses[key] = statuses.get(key, 0) + 1
    throughput = len(ok) / wall_s if wall_s else 0.0
    row = {"clients": clients, "requests": len(records), "statuses": statuses, "wall_s": wall_s,
           "throughput_rps": throughput, "latency_ms": summarise(ok)}
    if ok:  # Little's law: requests in flight = throughput x mean time in the system; ~= clients if measured right
        row["littles_law_in_flight"] = throughput * statistics.fmean(ok) / 1000
    return row


class GpuMemory:
    """Peak nvidia-smi memory.used on GPU 0 while running, sampled every 0.2 s in a thread (MiB)."""

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()

    def sample(self) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True, timeout=5).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return 0

    def __enter__(self):
        self.peak, self._stop = self.sample(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(0.2):
            self.peak = max(self.peak, self.sample())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


async def wait_ready(client: httpx.AsyncClient, proc: subprocess.Popen, timeout_s: float) -> float:
    """Seconds until /readyz is 200. The model loads before uvicorn listens, so refused connections are expected."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode} during startup")
        try:
            if (await client.get("/readyz")).status_code == 200:
                return time.perf_counter() - start
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.5)
    raise SystemExit(f"server not ready after {timeout_s:.0f}s")


async def probes(client: httpx.AsyncClient, body: dict) -> dict:
    """The documented behaviour of the open and keyed routes, as status codes."""
    auth = {"Authorization": f"Bearer {API_KEY}"}
    return {
        "healthz_no_key": (await client.get("/healthz")).status_code,
        "readyz_no_key": (await client.get("/readyz")).status_code,
        "metrics_no_key": (await client.get("/metrics")).status_code,
        "metrics_key": (await client.get("/metrics", headers=auth)).status_code,
        "systemone_no_key": (await client.post("/v1/systemone", json=body)).status_code,
        "systemone_wrong_key": (await client.post("/v1/systemone", json=body,
                                                  headers={"Authorization": "Bearer wrong"})).status_code,
        "systemone_key": (await client.post("/v1/systemone", json=body, headers=auth)).status_code,
    }


async def scrape(client) -> dict:
    return parse_metrics((await client.get("/metrics", headers={"Authorization": f"Bearer {API_KEY}"})).text)


async def shutdown_check(client, proc, body_for, first_new: int, headers: dict) -> tuple[dict, list[dict]]:
    """SIGTERM after 3 s of 16-client load. Only requests sent before the signal count as in flight."""
    stop = asyncio.Event()
    ids = list(range(first_new, first_new + 10_000))  # more than can finish: the signal ends the run
    task = asyncio.create_task(closed_loop(client, ids, body_for, 16, headers, stop))
    await asyncio.sleep(3.0)
    sigterm_at = time.perf_counter()
    proc.send_signal(signal.SIGTERM)
    readyz = None
    async with httpx.AsyncClient(base_url=str(client.base_url), timeout=2) as fresh:  # a new connection, as a probe makes
        try:
            readyz = (await fresh.get("/readyz")).status_code
        except httpx.TransportError as exc:
            readyz = type(exc).__name__  # refused: also not ready
    readyz_ms = 1000 * (time.perf_counter() - sigterm_at)
    stop.set()
    records, _ = await task
    exit_s = None
    for _ in range(int(GRACE_S * 2 + 20)):
        if proc.poll() is not None:
            exit_s = time.perf_counter() - sigterm_at
            break
        await asyncio.sleep(0.5)
    before = [r for r in records if r["sent"] < sigterm_at]
    after = [r for r in records if r["sent"] >= sigterm_at]
    count = lambda rs: level_row(rs, 1, 16)["statuses"]  # noqa: E731
    return {"readyz_after_sigterm": readyz, "readyz_probe_ms": readyz_ms, "exit_s": exit_s,
            "exit_code": proc.returncode, "grace_s": GRACE_S, "exited_inside_grace": exit_s is not None and exit_s <= GRACE_S,
            "sent_before_sigterm": count(before), "sent_after_sigterm": count(after),
            "sent_before_all_200_or_503": all(r["status"] in (200, 503) for r in before)}, records


def reference_check(weights: str, metrics: str, served: list[dict], body_for) -> tuple[dict, dict]:
    """Every 200 against decide() run serially in-process (same loader, same GPU, the server gone), then decide()'s
    latency on the server's own path: a fresh StateCache per call (a miss) and a warmed one (a hit), tokenising timed."""
    import torch

    from src import serve
    from src.pointwise.packing import StateCache
    from src.pointwise.schema import SystemOneRequest

    serve.load_pointwise(weights, metrics)
    model = serve._pointwise
    refs, diffs, flips, n = {}, 0.0, [], 0
    for r in served:
        if r["state"] not in refs:
            refs[r["state"]] = model.decide(body_for(r["state"]), state_cache=StateCache())["answers"]
        c = compare_answers(r["answers"], refs[r["state"]])
        diffs, n = max(diffs, c["max_prob_diff"]), n + c["questions"]
        flips += [{"state": r["state"], **f} for f in c["flips"]]
    correctness = {"responses": len(served), "distinct_states": len(refs), "questions": n,
                   "max_prob_diff": diffs, "top_option_flips": flips}

    def time_it(make_cache, reps=REFERENCE_REPS):
        request, out = SystemOneRequest.model_validate(body_for(10**6)), []
        warm = StateCache()
        for i in range(reps + 5):
            cache = make_cache(warm)
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.decide(request, state_cache=cache)
            torch.cuda.synchronize()
            if i >= 5:
                out.append(1000 * (time.perf_counter() - start))
        return summarise(out)

    in_process = {"miss": time_it(lambda warm: StateCache()), "hit": time_it(lambda warm: warm)}
    return correctness, in_process


async def run_server_phase(args, env: dict, body_for) -> dict:
    out: dict = {}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    limits = httpx.Limits(max_connections=BURST + 8, max_keepalive_connections=BURST + 8)
    proc = subprocess.Popen([sys.executable, "-m", "src.serve_main"], env=env)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=60, limits=limits) as client:
            with GpuMemory() as mem:
                out["startup_s"] = await wait_ready(client, proc, timeout_s=900)
            out["gpu_mib_after_load"] = mem.sample()
            out["probes"] = await probes(client, body_for(-1))
            warm, _ = await closed_loop(client, [-2 - i for i in range(5)], body_for, 1, headers)  # CUDA warm-up, untimed
            out["levels"], served, nxt = [], [r for r in warm if r["status"] == 200], 0
            for rate in RATES if args.open_loop else ():
                times = arrival_times(rate, OPEN_LOOP_S, seed=int(rate * 10))
                ids = list(range(nxt, nxt + len(times)))
                nxt += len(times)
                before = await scrape(client)
                records, wall = await open_loop(client, ids, times, body_for, headers)
                row = {"mix": "open-loop cold", "rate_rps": rate, "offered_rps": len(times) / OPEN_LOOP_S,
                       **level_row(records, wall, 0), "server_counters": metrics_delta(before, await scrape(client))}
                out["levels"].append(row)
                served += [r for r in records if r["status"] == 200]
                print(json.dumps({k: row[k] for k in ("rate_rps", "statuses", "throughput_rps", "latency_ms")}), flush=True)
            for mix, share in () if args.open_loop else (("cold", 0.0), ("cached", REPEAT_SHARE)):
                for clients in CLIENTS:
                    ids = request_mix(PER_LEVEL, share, seed=clients, first_new=nxt)
                    nxt = max(ids) + 1
                    before = await scrape(client)
                    with GpuMemory() as mem:
                        records, wall = await closed_loop(client, ids, body_for, clients, headers)
                    row = {"mix": mix, "intended_repeat_share": share, **level_row(records, wall, clients),
                           "server_counters": metrics_delta(before, await scrape(client)), "gpu_mib_peak": mem.peak}
                    out["levels"].append(row)
                    served += [r for r in records if r["status"] == 200]
                    print(json.dumps({k: row[k] for k in ("mix", "clients", "statuses", "throughput_rps", "latency_ms")}), flush=True)
            if args.open_loop:  # the burst was measured in the closed-loop run
                out["shutdown"], records = await shutdown_check(client, proc, body_for, nxt, headers)
                served += [r for r in records if r["status"] == 200]
                out["served"] = served
                return out
            before = await scrape(client)
            burst = await asyncio.gather(*(send(client, nxt + i, body_for(nxt + i), headers) for i in range(BURST)))
            nxt += BURST
            out["burst"] = {**level_row(list(burst), 1, BURST), "server_counters": metrics_delta(before, await scrape(client)),
                            "retry_after_on_every_503": all(r["retry_after"] for r in burst if r["status"] == 503),
                            "ms_503": summarise([r["ms"] for r in burst if r["status"] == 503])}
            served += [r for r in burst if r["status"] == 200]
            out["shutdown"], records = await shutdown_check(client, proc, body_for, nxt, headers)
            served += [r for r in records if r["status"] == 200]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    out["served"] = served
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--open-loop", action="store_true", help="Poisson arrivals at RATES instead of closed-loop clients")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from src.config import load_task_config
    from src.pointwise.train import MODEL_NAME

    if not torch.cuda.is_available():
        raise SystemExit("no GPU: refusing to write CPU timings")
    start = time.perf_counter()
    snapshot_download(MODEL_NAME)  # timed apart, so startup_s is the server's own start, not the download
    download_s = time.perf_counter() - start
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # The Hub's Parquet copy: datasets 5 (requirements-serve.txt) no longer runs banking77.py, the loader src.data uses
    messages = load_dataset(load_task_config().dataset_hf_name, revision="refs/convert/parquet", split="test")["text"]
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    bodies: dict[int, dict] = {}

    def body_for(i: int) -> dict:
        if i not in bodies:
            bodies[i] = payload(make_state(i, messages, count))
        return bodies[i]

    env = {**os.environ, "POINTWISE_WEIGHTS": args.weights, "POINTWISE_METRICS": args.metrics,
           "POINTWISE_API_KEY": API_KEY, "POINTWISE_PORT": str(PORT), "POINTWISE_SHUTDOWN_GRACE_S": str(GRACE_S)}
    result = asyncio.run(run_server_phase(args, env, body_for))
    served = result.pop("served")
    result["correctness"], result["in_process"] = reference_check(args.weights, args.metrics, served, body_for)
    result["env"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "download_s": download_s,
                     "state_tokens": count(body_for(0)["state"]), "k": K, "per_level": PER_LEVEL, "burst": BURST,
                     "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                text=True).stdout.splitlines()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ("load_test_open.json" if args.open_loop else "load_test.json")).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
