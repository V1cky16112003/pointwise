"""Step 4 A/B: the kept cold-path changes against the old path, in one Kaggle session (docs/designs/pointwise-plan.md).

Two copies of run 3 on one T4: unmerged and merged (the new default). One ~512-token state, k=8 three-option questions,
every variant timed round by round (phase6_latency.interleaved), so session drift hits all alike; ratios are to the
first variant, the old default (cold, unmerged). Variants: cold, miss with the old two-pass path (prefill_state then
branch_features), miss with the new one-pass path (StateCache), cached; each unmerged and merged.

Correctness on the real weights: probabilities of the new miss, the hit, and the merged cold/miss/hit paths against the
old reference (unmerged, uncached), over the k=1/2/4/8 latency requests and the first 100 Phase 5 held-out records.
Usage: python -m scripts.coldpath_ab --data-dir <built data> --state <run 3 state.pt> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.phase5_eval import load_samples
from scripts.phase6_latency import GATE_MS, agreement, interleaved, load_run3, pooled, requests_for
from src.pointwise.model import DecisionModel
from src.pointwise.schema import to_answers
from src.pointwise.packing import StateCache, branch_features, pack, prefill_state

K = 8
HELDOUT_RECORDS = 100


@torch.no_grad()
def old_miss_probs(model: DecisionModel, encoded) -> list[list[float]]:
    """The pre-step-4 cache miss: read the state alone, then the questions on its KV (two passes)."""
    packed = pack(encoded.state_ids, [q.ids for q in encoded.questions])
    cache = prefill_state(model.backbone, encoded.state_ids)
    return [z.softmax(-1).tolist() for z in model._head_logits(encoded, branch_features(model.backbone, cache, packed))]


def variants(models: dict[str, DecisionModel], request) -> dict:
    out = {}
    for tag, m in models.items():
        miss, hit = StateCache(max_entries=1, max_tokens=1 << 20), StateCache(max_entries=1, max_tokens=1 << 20)
        m.decide(request, hit)  # stored once; every timed request hits
        out[f"cold_{tag}"] = (lambda m=m: m.decide(request), lambda: None)
        out[f"miss_old_{tag}"] = (lambda m=m: to_answers(request, old_miss_probs(m, m.encode(request))), lambda: None)
        out[f"miss_new_{tag}"] = (lambda m=m, c=miss: m.decide(request, c), miss.clear)
        out[f"cached_{tag}"] = (lambda m=m, c=hit: m.decide(request, c), lambda: None)
    return out


def correctness(models: dict[str, DecisionModel], encoded_requests: list) -> dict:
    """Each path against the old reference: unmerged, uncached (what decide() returned before step 4)."""
    base = models["unmerged"]
    parts: dict[str, list] = {}
    for e in encoded_requests:
        ref = base.probs(e)
        parts.setdefault("old_miss_vs_reference", []).append(agreement(old_miss_probs(base, e), ref))
        for tag, m in models.items():
            cache = StateCache(max_entries=1, max_tokens=1 << 20)
            if tag != "unmerged":
                parts.setdefault(f"cold_{tag}_vs_reference", []).append(agreement(m.probs(e), ref))
            parts.setdefault(f"miss_{tag}_vs_reference", []).append(agreement(m.probs(e, cache), ref))
            parts.setdefault(f"hit_{tag}_vs_reference", []).append(agreement(m.probs(e, cache), ref))
    return {name: pooled(p) for name, p in parts.items()}


def table(result: dict) -> str:
    lines = ["| Variant | median | p95 | p95 / old default | vs 150 ms p95 |", "|---|---|---|---|---|"]
    for name, row in result["timing"].items():
        verdict = "met" if row["p95"] < GATE_MS else f"not met by {row['p95'] - GATE_MS:.1f}"
        lines.append(f"| {name} | {row['median']:.1f} | {row['p95']:.1f} | {row['p95_ratio']:.2f} | {verdict} |")
    lines += ["", "| Path vs old reference | max prob diff | top-option flips | questions |", "|---|---|---|---|"]
    lines += [f"| {n} | {a['max_prob_diff']:.4f} | {a['top_option_flips']} | {a['questions']} |"
              for n, a in result["correctness"].items()]
    return "\n".join(lines)


def main() -> None:
    from scripts.bench_packing import build_request

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="run 3's best/state.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke run: held-out records per source (load_samples)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: the A/B is judged against a T4 target, so refusing to write CPU timings")
    models = {"unmerged": load_run3(args.state)}
    models["merged"] = copy.deepcopy(models["unmerged"])  # one load; ~3.4 GB more on the T4
    models["merged"].backbone = models["merged"].backbone.merge_and_unload()
    ids, _ = build_request(models["unmerged"].tokenizer, 512, 1)
    state = models["unmerged"].tokenizer.decode(ids)
    request = requests_for(state, (K,))[K]
    timing = interleaved(variants(models, request), torch.device("cuda"))
    _, _, held, _ = load_samples(args, models["unmerged"].tokenizer)
    encoded = [models["unmerged"].encode(r) for r in requests_for(state, (1, 2, 4, 8)).values()]
    encoded += [ex.encoded for ex in held[:HELDOUT_RECORDS]]
    result = {"k": K, "state_tokens": len(models["unmerged"].encode(request).state_ids), "timing": timing,
              "correctness": correctness(models, encoded), "correctness_records": len(encoded),
              "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coldpath_ab.json").write_text(json.dumps(result, indent=1))
    (args.output_dir / "coldpath_ab.md").write_text(table(result) + "\n")
    print(table(result), flush=True)


if __name__ == "__main__":
    main()
