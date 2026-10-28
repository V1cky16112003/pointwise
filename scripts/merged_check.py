"""Merged-model check (plan step 5): does serving with the LoRA adapter merged change run 3's answers?

Merging folds the adapter into the fp16 base weights (W + B @ A, rounded once to fp16), so the merged model computes
slightly different numbers than base + adapter computed separately. Temperature 1.516 was fitted on the unmerged model.
Scores the 2,500 Phase 5 held-out questions (run 3's sample, checked against reports/run3_sample_ids.json) with the
adapter unmerged, then merged, in the same session, through src.pointwise.train.score_rows at run 3's temperature.
Logits come from the served path (question_logits: eval mode, no autocast, so the fp32 adapter runs in fp32 as it does
when serving), with the head's temperature at 1 so score_rows applies T exactly once.

Ship merged as the serving default only if the top option agrees on >= 99.5% of questions AND pooled out-of-domain
ECE after T moves by at most 0.005 (from Phase 5's 0.0561); otherwise serve unmerged.
Usage: python -m scripts.merged_check --data-dir <built data> --state <run 3 state.pt> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.phase5_eval import load_samples, write_rows
from scripts.phase6_latency import RUN3_TEMPERATURE, load_run3
from src.pointwise import train as T

MIN_TOP_AGREEMENT = 0.995
MAX_ECE_SHIFT = 0.005


def top(logits: list[float]) -> int:
    return max(range(len(logits)), key=logits.__getitem__)


def compare(unmerged: list[dict], merged: list[dict], temperature: float = RUN3_TEMPERATURE) -> dict:
    """Top-option agreement and pooled OOD ECE after T, unmerged vs merged. Rows must be the same questions."""
    key = lambda r: (r["id"], r["question"])  # noqa: E731
    a, b = {key(r): r for r in unmerged}, {key(r): r for r in merged}
    if a.keys() != b.keys():
        raise ValueError("unmerged and merged rows cover different questions")
    agree = sum(top(a[k]["logits"]) == top(b[k]["logits"]) for k in a) / len(a)
    ood = lambda rows: [r for r in rows if r["source"] != "banking77"]  # noqa: E731
    ece_a = T.score_rows(ood(unmerged), temperature)["all"]["ece"]
    ece_b = T.score_rows(ood(merged), temperature)["all"]["ece"]
    max_diff = max(abs(x - y) for k in a for x, y in zip(a[k]["logits"], b[k]["logits"]))
    return {"questions": len(a), "top_agreement": agree, "ood_ece_unmerged": ece_a, "ood_ece_merged": ece_b,
            "ece_shift": ece_b - ece_a, "max_logit_diff": max_diff,
            "ship_merged": agree >= MIN_TOP_AGREEMENT and abs(ece_b - ece_a) <= MAX_ECE_SHIFT,
            "heldout_unmerged": T.score_rows(unmerged, temperature)["all"],
            "heldout_merged": T.score_rows(merged, temperature)["all"]}


@torch.no_grad()
def served_logits(model, examples) -> list[dict]:
    """One row per question, in the collect_logits row format, but computed the way decide() computes them."""
    rows = []
    for ex in examples:
        for k, (z, target) in enumerate(zip(model.question_logits(ex.encoded), ex.targets)):
            rows.append({"source": ex.source, "id": ex.id, "question": k, "logits": z.tolist(), "target": target})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="run 3's best/state.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke run: this many records per source")
    args = parser.parse_args()

    model = load_run3(args.state)
    model.head.temperature = 1.0  # raw logits: score_rows applies run 3's T
    _, _, held, _ = load_samples(args, model.tokenizer)
    unmerged = served_logits(model, held)
    model.backbone = model.backbone.merge_and_unload()
    merged = served_logits(model, held)
    out = args.output_dir
    write_rows(out / "merged_check" / "heldout_unmerged.jsonl", unmerged)
    write_rows(out / "merged_check" / "heldout_merged.jsonl", merged)
    result = compare(unmerged, merged)
    (out / "merged_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
