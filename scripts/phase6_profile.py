"""Where a cold Pointwise request spends its time (cold-path latency, plan step 4).

One ~512-token state and k=8 three-option questions, the Phase 6 request. The timed call is the served one,
DecisionModel.decide(request, state_cache), with timed wrappers around the functions it calls, so the stages add up
to the real path, not a copy of it:

  encode        tokenising and laying out the request (CPU)
  mask          building the additive block attention mask (block_mask_batch)
  packed        the cold path's one forward pass over state + questions
  prefill       (before step 4 b only) a miss's pass over the state alone; a miss is now one packed pass
  questions_on_kv  the questions' forward pass on top of the stored state KV (cached)
  .../mask, .../layer  inside those: mask building, and the decoder layers summed (per-layer median printed apart)
  head          the pointer head on the question features
  answers       turning probabilities into the response body

Each stage has a GPU time (CUDA events: when the GPU reached the start and the end) and a CPU time (how long Python
took to issue it, no synchronising). When CPU time is close to GPU time for a stage the GPU was waiting on Python: that
is kernel launch overhead (each small GPU operation costs a few microseconds just to queue, whatever its size). One
request per configuration also runs under torch.profiler, to count kernels and sum the time the GPU was actually busy.

Configurations, adapter unmerged then merged (merging is in place): cold (no cache), miss (cache on, emptied before
each request) and cached (state already stored). Median and p95 over REPS requests after WARMUP. The miss row is
accounted against a cold request (one-pass miss), or prefill + cached on the old two-pass path.
Usage: python -m scripts.phase6_profile --state <run 3 state.pt> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import torch

import src.pointwise.model as model_module
import src.pointwise.packing as packing
from scripts.phase6_latency import GATE_MS, REPS, WARMUP, load_run3, requests_for, summarise, sync
from src.pointwise.model import DecisionModel
from src.pointwise.packing import StateCache

K = 8


class Recorder:
    """Nested timed spans for one request. GPU times come from CUDA events, read after the request syncs."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.stack: list[str] = []
        self.spans: list[tuple[str, object, object, float, float]] = []  # path, start, end, cpu start, cpu end

    def _stamp(self):
        if not self.cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @contextmanager
    def span(self, name: str):
        self.stack.append(name)
        path = "/".join(self.stack)
        start, cpu_start = self._stamp(), time.perf_counter()
        try:
            yield
        finally:
            end, cpu_end = self._stamp(), time.perf_counter()
            self.stack.pop()
            self.spans.append((path, start, end, cpu_start, cpu_end))

    def totals(self) -> dict[str, dict[str, float]]:
        """Per span path, GPU and CPU ms summed over its occurrences in this request. Call after a sync."""
        out: dict[str, dict[str, float]] = {}
        for path, start, end, cpu_start, cpu_end in self.spans:
            cpu = 1000 * (cpu_end - cpu_start)
            gpu = start.elapsed_time(end) if self.cuda else cpu  # no GPU: wall time stands in for both
            row = out.setdefault(path, {"gpu": 0.0, "cpu": 0.0, "count": 0})
            row["gpu"] += gpu
            row["cpu"] += cpu
            row["count"] += 1
        return out

    def layer_times(self) -> list[float]:
        return [s.elapsed_time(e) if self.cuda else 1000 * (ce - cs)
                for path, s, e, cs, ce in self.spans if path.endswith("/layer")]


def decoder_layers(backbone) -> list[torch.nn.Module]:
    layers = [m for name, m in backbone.named_modules() if name.split(".")[-2:-1] == ["layers"] and name.split(".")[-1].isdigit()]
    if not layers:
        raise ValueError("no decoder layers found (expected modules named ...layers.<i>)")
    return layers


@contextmanager
def instrumented(model: DecisionModel, rec_box: list[Recorder]):
    """Timed wrappers around every stage of decide(); rec_box[0] is the current request's recorder."""

    def wrap(name, fn):
        def timed(*a, **kw):
            with rec_box[0].span(name):
                return fn(*a, **kw)
        return timed

    layer_spans: list = []

    def pre(module, args):
        cm = rec_box[0].span("layer")
        cm.__enter__()
        layer_spans.append(cm)

    def post(module, args, output):
        layer_spans.pop().__exit__(None, None, None)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(model, "encode", wrap("encode", model.encode)))
        stack.enter_context(mock.patch.object(packing, "block_mask_batch", wrap("mask", packing.block_mask_batch)))
        stack.enter_context(mock.patch.object(packing, "prefill_state", wrap("prefill", packing.prefill_state)))
        stack.enter_context(mock.patch.object(packing, "branch_features", wrap("questions_on_kv", packing.branch_features)))
        packed = wrap("packed", packing.packed_forward)
        stack.enter_context(mock.patch.object(packing, "packed_forward", packed))
        stack.enter_context(mock.patch.object(model_module, "packed_forward", packed))  # the cold path's own import
        stack.enter_context(mock.patch.object(packing, "packed_with_state_cache",
                                              wrap("packed", packing.packed_with_state_cache)))  # a miss since step 4 (b)
        stack.enter_context(mock.patch.object(model, "_head_logits", wrap("head", model._head_logits)))
        stack.enter_context(mock.patch.object(model_module, "to_answers", wrap("answers", model_module.to_answers)))
        handles = []
        for layer in decoder_layers(model.backbone):
            handles += [layer.register_forward_pre_hook(pre), layer.register_forward_hook(post)]
        stack.callback(lambda: [h.remove() for h in handles])
        yield


def kernel_stats(fn) -> dict:
    """One call under torch.profiler: kernels launched and the GPU's busy time, against the call's wall time."""
    from torch.profiler import ProfilerActivity, profile

    sync(torch.device("cuda"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()  # inside: profiler start-up is not request time
        fn()
        sync(torch.device("cuda"))
    wall = 1000 * (time.perf_counter() - start)
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    busy = sum(e.time_range.elapsed_us() for e in kernels) / 1000
    return {"kernels": len(kernels), "gpu_busy_ms": busy, "profiled_wall_ms": wall,
            "launch_bound_share": max(0.0, 1 - busy / wall) if wall else 0.0}


def profile_config(model: DecisionModel, request, cache: StateCache | None, before, reps: int, warmup: int) -> dict:
    device = model.backbone.device
    box: list[Recorder] = [Recorder(device)]
    walls, stages, layers = [], [], []
    with instrumented(model, box):
        for i in range(warmup + reps):
            before()
            box[0] = Recorder(device)
            sync(device)
            start = time.perf_counter()
            model.decide(request, cache)
            sync(device)
            if i >= warmup:
                walls.append(1000 * (time.perf_counter() - start))
                stages.append(box[0].totals())
                layers += box[0].layer_times()
    paths = sorted({p for s in stages for p in s}, key=lambda p: (p.split("/")[0], p))
    out = {"wall": summarise(walls), "stages": {}, "layer_ms": summarise(layers) if layers else None}
    for path in paths:
        out["stages"][path] = {kind: summarise([s.get(path, {}).get(kind, 0.0) for s in stages]) for kind in ("gpu", "cpu")}
        out["stages"][path]["count"] = stages[0].get(path, {}).get("count", 0)
    # Uninstrumented, to show what the wrappers and hooks cost.
    plain = []
    for _ in range(reps):
        before()
        sync(device)
        start = time.perf_counter()
        model.decide(request, cache)
        sync(device)
        plain.append(1000 * (time.perf_counter() - start))
    out["wall_uninstrumented"] = summarise(plain)
    if device.type == "cuda":
        before()
        out["kernels"] = kernel_stats(lambda: model.decide(request, cache))
    return out


def profile(model: DecisionModel, state: str, reps: int = REPS, warmup: int = WARMUP) -> dict:
    from peft import PeftModel

    request = requests_for(state, (K,))[K]
    out: dict = {"k": K, "state_tokens": len(model.encode(request).state_ids), "reps": reps, "configs": {}}

    def run_all(suffix: str) -> None:
        out["configs"][f"cold{suffix}"] = profile_config(model, request, None, lambda: None, reps, warmup)
        miss = StateCache(max_entries=1, max_tokens=1 << 20)
        out["configs"][f"miss{suffix}"] = profile_config(model, request, miss, miss.clear, reps, warmup)
        hit = StateCache(max_entries=1, max_tokens=1 << 20)
        model.decide(request, hit)  # stores the state; every timed request then hits
        out["configs"][f"cached{suffix}"] = profile_config(model, request, hit, lambda: None, reps, warmup)
        out[f"miss_accounting{suffix}"] = miss_accounting(out["configs"], suffix)

    run_all("")
    if isinstance(model.backbone, PeftModel):
        model.backbone = model.backbone.merge_and_unload()
        run_all("_merged")
    return out


def miss_accounting(configs: dict, suffix: str) -> dict:
    """Medians: what a miss costs beyond prefilling the state and then serving it as a cached request. Walls are the
    uninstrumented ones; prefill is its GPU span (the only clock for one stage), so its CPU-only cost lands in extra."""
    miss, cached = configs[f"miss{suffix}"], configs[f"cached{suffix}"]
    if "prefill" not in miss["stages"]:  # one-pass miss (step 4 b): compare with a cold request instead
        cold = configs[f"cold{suffix}"]["wall_uninstrumented"]["median"]
        return {"miss_ms": miss["wall_uninstrumented"]["median"], "cold_ms": cold,
                "extra_ms": miss["wall_uninstrumented"]["median"] - cold}
    prefill = miss["stages"]["prefill"]["gpu"]["median"]
    extra = miss["wall_uninstrumented"]["median"] - prefill - cached["wall_uninstrumented"]["median"]
    q_miss = miss["stages"]["questions_on_kv"]["gpu"]["median"]
    q_hit = cached["stages"]["questions_on_kv"]["gpu"]["median"]
    return {"miss_ms": miss["wall_uninstrumented"]["median"], "prefill_ms": prefill,
            "cached_ms": cached["wall_uninstrumented"]["median"],
            "extra_ms": extra, "questions_on_kv_miss_ms": q_miss, "questions_on_kv_cached_ms": q_hit}


STAGE_ORDER = ("encode", "packed", "prefill", "questions_on_kv", "head", "answers")


def table(result: dict) -> str:
    lines = [f"k={result['k']}, state {result['state_tokens']} tokens, {result['reps']} requests per configuration; "
             "GPU ms (CPU ms to issue), median / p95.", ""]
    for name, cfg in result["configs"].items():
        lines += [f"### {name}: wall median {cfg['wall']['median']:.1f} / p95 {cfg['wall']['p95']:.1f} ms "
                  f"(uninstrumented {cfg['wall_uninstrumented']['median']:.1f} / {cfg['wall_uninstrumented']['p95']:.1f}; "
                  f"target {GATE_MS:.0f} ms p95)", "", "| Stage | calls | GPU median / p95 | CPU median / p95 |", "|---|---|---|---|"]
        for path, s in sorted(cfg["stages"].items(), key=lambda kv: (_rank(kv[0]), kv[0])):
            indent = "&nbsp;&nbsp;" * path.count("/")
            lines.append(f"| {indent}{path.split('/')[-1]} | {s['count']} | {s['gpu']['median']:.1f} / {s['gpu']['p95']:.1f} "
                         f"| {s['cpu']['median']:.1f} / {s['cpu']['p95']:.1f} |")
        if cfg.get("layer_ms"):
            lines.append(f"\nOne decoder layer: median {cfg['layer_ms']['median']:.2f} ms, p95 {cfg['layer_ms']['p95']:.2f} ms.")
        if cfg.get("kernels"):
            kn = cfg["kernels"]
            lines.append(f"Profiled request: {kn['kernels']} kernels, GPU busy {kn['gpu_busy_ms']:.1f} of "
                         f"{kn['profiled_wall_ms']:.1f} ms ({100 * kn['launch_bound_share']:.0f}% idle, waiting on the CPU).")
        lines.append("")
    for key, acc in result.items():
        if key.startswith("miss_accounting") and "cold_ms" in acc:
            lines.append(f"- {key}: one-pass miss {acc['miss_ms']:.1f} vs cold {acc['cold_ms']:.1f} ms (extra {acc['extra_ms']:.1f})")
        elif key.startswith("miss_accounting"):
            lines.append(f"- {key}: miss {acc['miss_ms']:.1f} = prefill {acc['prefill_ms']:.1f} + cached {acc['cached_ms']:.1f} "
                         f"+ extra {acc['extra_ms']:.1f} ms; questions-on-KV pass {acc['questions_on_kv_miss_ms']:.1f} on a miss "
                         f"vs {acc['questions_on_kv_cached_ms']:.1f} cached")
    return "\n".join(lines)


def _rank(path: str) -> int:
    top = path.split("/")[0]
    return STAGE_ORDER.index(top) if top in STAGE_ORDER else len(STAGE_ORDER)


def main() -> None:
    from scripts.bench_packing import build_request

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", type=Path, required=True, help="run 3's best/state.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: the profile is judged against a T4 target, so refusing to write CPU timings")
    model = load_run3(args.state)
    ids, _ = build_request(model.tokenizer, 512, 1)
    result = profile(model, model.tokenizer.decode(ids))
    result["env"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase6_profile.json").write_text(json.dumps(result, indent=1))
    (args.output_dir / "phase6_profile.md").write_text(table(result) + "\n")
    print(table(result), flush=True)


if __name__ == "__main__":
    main()
