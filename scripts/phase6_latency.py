"""Phase 6 serving latency on one T4 (docs/designs/pointwise-plan.md).

One ~512-token state and k three-option questions (k = 1, 2, 4, 8), the same request shape as the Phase 5
latency table. For every configuration the timed call is DecisionModel.decide(request, state_cache): tokenising,
the model, the pointer head and building the answers; JSON and HTTP are not included. Each cell is the median and
p95 over REPS requests after WARMUP untimed ones, GPU-synchronised.

  a  cold state, adapter unmerged: no cache, the whole request in one packed pass
  c  cached state, adapter unmerged: the state's KV is stored, only the questions run
  d  cache miss, adapter unmerged: the cache is on but empty for this state (prefill, then questions)
  b  cold state, adapter merged
  e  cached state, adapter merged
  p5 a with a pre-encoded request: the Phase 5 method, to reconcile with its 261 ms at k=8

Merging is in place and cannot be undone, so the unmerged rows run first.
Usage: python -m scripts.phase6_latency --state <run 3 state.pt> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from src.pointwise.model import DecisionModel
from src.pointwise.packing import StateCache
from src.pointwise.schema import SystemOneRequest

KS = (1, 2, 4, 8)
RUN3_TEMPERATURE = 1.5159  # run 3's metrics.json, so probability drift is measured as served
GATE_MS = 150.0  # the plan's p95 target for one state and 8 questions
REPS, WARMUP = 60, 5  # the goal asks for at least 50 timed requests per cell


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {"n": len(ordered), "median": statistics.median(ordered), "mean": statistics.fmean(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "max": ordered[-1]}


def timed(fn, device, reps: int, warmup: int, before=lambda: None) -> dict:
    for _ in range(warmup):
        before()
        fn()
    samples = []
    for _ in range(reps):
        before()  # outside the timer: e.g. emptying the cache
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append(1000 * (time.perf_counter() - start))
    return summarise(samples)


def requests_for(state: str, ks) -> dict[int, SystemOneRequest]:
    from scripts.bench_decide import shared_state_request

    return {k: SystemOneRequest.model_validate(shared_state_request(state, k)) for k in ks}


def agreement(a: list[list[float]], b: list[list[float]]) -> dict:
    """How far two sets of per-question probabilities are apart, and how often the top option differs."""
    top = lambda p: max(range(len(p)), key=p.__getitem__)  # noqa: E731
    return {"max_prob_diff": max(abs(x - y) for qa, qb in zip(a, b) for x, y in zip(qa, qb)),
            "top_option_flips": sum(top(qa) != top(qb) for qa, qb in zip(a, b)), "questions": len(a)}


def pooled(parts: list[dict]) -> dict:
    return {"max_prob_diff": max(p["max_prob_diff"] for p in parts),
            "top_option_flips": sum(p["top_option_flips"] for p in parts), "questions": sum(p["questions"] for p in parts)}


def measure(model: DecisionModel, state: str, ks=KS, reps: int = REPS, warmup: int = WARMUP) -> dict:
    """All configurations for one model. Merges the adapter part-way through if it has one."""
    from peft import PeftModel

    device = model.backbone.device
    requests = requests_for(state, ks)
    state_tokens = len(model.encode(requests[ks[0]]).state_ids)
    out: dict = {"state_tokens": state_tokens, "reps": reps, "warmup": warmup, "results": {}}
    encoded = {k: model.encode(r) for k, r in requests.items()}
    reference = {k: model.probs(e) for k, e in encoded.items()}  # uncached, unmerged: what model.decide() returns
    miss, hit = [], []
    for k, e in encoded.items():
        fresh = StateCache()
        miss.append(agreement(model.probs(e, fresh), reference[k]))  # prefill, then the questions
        hit.append(agreement(model.probs(e, fresh), reference[k]))  # the stored state, as a second request sees it
    out["agreement"] = {"cache_miss_vs_uncached": pooled(miss), "cache_hit_vs_uncached": pooled(hit)}

    def run(name: str, make_fn, cache=None, before=lambda: None) -> None:
        out["results"][name] = {
            k: timed(make_fn(request, cache), device, reps, warmup, before) for k, request in requests.items()
        }

    def decide(request, cache):
        return lambda: model.decide(request, cache)

    def warmed(request, cache):  # one untimed request stores the state; every timed one then hits
        model.decide(request, cache)
        return decide(request, cache)

    run("a", decide)
    run("p5", lambda request, _: (lambda e=model.encode(request): model.probs(e)))
    cache = StateCache(max_entries=1, max_tokens=1 << 20)
    run("c", warmed, cache)
    out["cache_mb_per_state"] = cache.nbytes() / 2**20  # the keys and values of one state of state_tokens tokens
    miss_cache = StateCache(max_entries=1, max_tokens=1 << 20)
    run("d", decide, miss_cache, before=miss_cache.clear)
    if isinstance(model.backbone, PeftModel):
        model.backbone = model.backbone.merge_and_unload()
        out["merged"] = True
        out["agreement"]["merged_vs_unmerged"] = pooled([
            agreement(model.probs(e), reference[k]) for k, e in encoded.items()])
        run("b", decide)
        run("e", warmed, StateCache(max_entries=1, max_tokens=1 << 20))
    return out


def table(result: dict) -> str:
    names = {"a": "a. cold, unmerged", "c": "c. cached state, unmerged", "d": "d. cache miss, unmerged",
             "b": "b. cold, merged", "e": "e. cached state, merged", "p5": "Phase 5 method (probs, pre-encoded)"}
    ks = list(next(iter(result["results"].values())))
    lines = ["| Configuration | " + " | ".join(f"k={k} median / p95" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for key in ("a", "p5", "c", "d", "b", "e"):
        if key in result["results"]:
            row = result["results"][key]
            lines.append(f"| {names[key]} | " + " | ".join(f"{row[k]['median']:.0f} / {row[k]['p95']:.0f}" for k in ks) + " |")
    gate = [f"- {names[key]}: p95 {result['results'][key][8]['p95']:.0f} ms at k=8, "
            f"{'met' if result['results'][key][8]['p95'] < GATE_MS else 'not met'} against the {GATE_MS:.0f} ms target"
            for key in ("a", "c", "d", "b", "e") if key in result["results"] and 8 in result["results"][key]]
    return "\n".join(lines + [""] + gate)


def main() -> None:
    from scripts.bench_packing import build_request
    from scripts.phase5_eval import RUN3_BEST_STEP
    from src.pointwise import train as T

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", type=Path, required=True, help="run 3's best/state.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU: these numbers are judged against a T4 target, so refusing to write CPU timings")
    device = torch.device("cuda")
    model = T.build_model(T.parse_args(["--data-dir", ".", "--output-dir", "."]), device)
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    assert state["step"] == RUN3_BEST_STEP, f"state.pt is step {state['step']}, not run 3's best step {RUN3_BEST_STEP}"
    T.load_trainable(model, state)
    model.head.temperature = RUN3_TEMPERATURE
    model.eval()
    ids, _ = build_request(model.tokenizer, 512, 1)
    result = measure(model.eval(), model.tokenizer.decode(ids))
    result["env"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase6_latency.json").write_text(json.dumps(result, indent=1))
    (args.output_dir / "phase6_latency.md").write_text(table(result) + "\n")
    print(table(result), flush=True)


if __name__ == "__main__":
    main()
