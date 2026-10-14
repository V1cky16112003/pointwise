"""Build the Pointwise Phase 3 data (docs/designs/pointwise-phase3-plan.md).

Writes to --out:
    train.jsonl    training records, sources mixed
    dev.jsonl      held-back rows of the training sources (checkpoint selection)
    test.jsonl     more held-back rows of the training sources (read once per model)
    heldout.jsonl  sources never trained on: Banking77 (zero-shot), QNLI, PAWS, Emotion, TweetEval
    report.json    counts, label report, leakage checks and the gate

Usage (from the repo root):
    python -m src.pointwise.data.build --out data/pointwise
    python -m src.pointwise.data.build --out /tmp/pointwise-smoke --limit-rows 200   # quick check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

from src.pointwise.data import records as R
from src.pointwise.data import sources as S
from src.pointwise.data.policies import policy_records

GATE_MIN_TRAIN_QUESTIONS = 200_000
# A source's gold-first rate may differ from the random expectation by this much,
# or by 3 standard errors if more: a capped source (2,000 records) misses 0.02 by chance.
GATE_POSITION_TOLERANCE = 0.02
EVAL_CAP = 1000  # records per source per dev/test split, and per held-out source
POLICY_RECORDS = {"train": 15_000, "dev": 500, "test": 500}

# name: (converter, train split, eval splits, train row cap, extra converter args for train)
TRAIN_SOURCES = {
    "mnli": (S.mnli, "train", ["validation_matched"], 30_000, {}),
    "boolq": (S.boolq, "train", ["validation"], None, {}),
    "agnews": (S.agnews, "train", ["test"], 25_000, {}),
    "sst5": (S.sst5, "train", ["validation", "test"], None, {}),
    "yelp": (S.yelp, "train", ["test"], 25_000, {}),
    "clinc": (S.clinc, "train", ["validation", "test"], None, {"copies": 2}),
    "arc_easy": (S.arc, "train", ["validation", "test"], None, {}),
    "arc_challenge": (S.arc, "train", ["validation", "test"], None, {}),
    "openbookqa": (S.openbookqa, "train", ["validation", "test"], None, {}),
    "commonsenseqa": (S.commonsenseqa, "train", ["validation"], None, {}),
}
HELDOUT_SOURCES = {
    "qnli": (S.qnli, "validation"),
    "paws": (S.paws, "test"),
    "emotion": (S.emotion, "test"),
    "tweet_sentiment": (S.tweet_sentiment, "test"),
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _rng(*parts) -> random.Random:
    return random.Random(":".join(str(p) for p in parts))


def _sample(rows: list, cap: int | None, rng: random.Random, limit: int | None) -> list:
    rows = list(rows)
    rng.shuffle(rows)
    for bound in (cap, limit):
        if bound is not None:
            rows = rows[:bound]
    return rows


def _dev_or_test(text: str) -> str:
    """Stable split by text hash, so the same row always lands on the same side."""
    return "dev" if hashlib.sha256(normalise(text).encode()).digest()[0] % 2 == 0 else "test"


def _renumber(recs: list[dict], split: str) -> list[dict]:
    for n, rec in enumerate(recs):
        rec["split"] = split
        rec["id"] = f"{rec['source']}/{split}/{n}"
    return recs


def build(seed: int, limit_rows: int | None, load_rows=S.load_rows, banking=None,
          train_cap: int | None = None) -> tuple[dict, dict]:
    """-> ({split: records}, report). load_rows and banking are injectable for tests.
    train_cap: at most this many train records per public source (Kev's balanced recipe);
    policies keep their own count."""
    from src.config import load_task_config

    config = load_task_config()
    if banking is None:
        from src.data import load_banking77_splits

        splits = load_banking77_splits(config)
        banking = {"labels": splits.labels, "texts": [r["text"] for s in (splits.train, splits.val, splits.test) for r in s], "test": list(splits.test)}

    out = {"train": [], "dev": [], "test": [], "heldout": []}
    clinc_excluded: frozenset = frozenset()
    for name, (convert, train_split, eval_splits, cap, train_kwargs) in TRAIN_SOURCES.items():
        rows, names = load_rows(name, train_split)
        kwargs = {}
        if name == "clinc":
            clinc_excluded = S.clinc_excluded(names, banking["labels"])
            kwargs["excluded"] = clinc_excluded
        rows = _sample(rows, cap, _rng(name, "rows", seed), limit_rows)
        out["train"] += convert(rows, names, _rng(name, "train", seed), "train", **kwargs, **train_kwargs)[:train_cap]

        eval_rows = []
        for split in eval_splits:
            eval_rows += load_rows(name, split)[0]
        eval_recs = convert(_sample(eval_rows, None, _rng(name, "eval-rows", seed), limit_rows), names, _rng(name, "eval", seed), "eval", **kwargs)
        for side in ("dev", "test"):
            out[side] += [r for r in eval_recs if _dev_or_test(r["_text"]) == side][:EVAL_CAP]

    scale = 1 if limit_rows is None else min(1.0, limit_rows / 1000)
    for split, n in POLICY_RECORDS.items():
        out[split] += policy_records(seed, split, max(1, int(n * scale)))

    out["heldout"] += S.banking77(banking["test"], banking["labels"], _rng("banking77", seed), "heldout")
    for name, (convert, split) in HELDOUT_SOURCES.items():
        rows, names = load_rows(name, split)
        rows = _sample(rows, EVAL_CAP, _rng(name, "rows", seed), limit_rows)
        out["heldout"] += convert(rows, names, _rng(name, "heldout", seed), "heldout")

    checks = leakage_checks(out, banking["texts"], clinc_excluded)
    for split in out:
        out[split] = _renumber(out[split], split)
    _rng("shuffle", seed).shuffle(out["train"])
    for recs in out.values():
        for rec in recs:
            rec.pop("_text", None)
            R.validate(rec)
    report = make_report(out, checks, full_size=limit_rows is None and train_cap is None)
    report["train_cap"] = train_cap
    return out, report


def leakage_checks(out: dict, banking_texts: list[str], clinc_excluded: frozenset) -> dict:
    """Drops leaking records in place and reports how many were found.
    - Banking77 text in any non-Banking77 record: breaks the zero-shot claim.
    - Train text also in dev/test/held-out: would inflate every eval.
    - An excluded CLINC intent as a CLINC option: a Banking77-like label."""
    banking = {normalise(t) for t in banking_texts}
    banking_leaks = 0
    for split, recs in out.items():
        keep = []
        for rec in recs:
            if rec["source"] != "banking77" and normalise(rec.get("_text", "")) in banking:
                banking_leaks += 1
            else:
                keep.append(rec)
        out[split] = keep

    eval_texts = {normalise(r["_text"]) for s in ("dev", "test", "heldout") for r in out[s] if "_text" in r}
    before = len(out["train"])
    out["train"] = [r for r in out["train"] if "_text" not in r or normalise(r["_text"]) not in eval_texts]

    shown = {n.replace("_", " ") for n in clinc_excluded}
    clinc_hits = sum(
        1
        for recs in out.values()
        for r in recs
        if r["source"] == "clinc"
        for q in r["request"]["questions"].values()
        if q["type"] == "choice" and shown & set(q["criteria"])
    )
    return {
        "banking77_text_dropped": banking_leaks,
        "train_eval_overlap_dropped": before - len(out["train"]),
        "clinc_excluded_intents": sorted(clinc_excluded),
        "clinc_excluded_intent_options": clinc_hits,
    }


def make_report(out: dict, checks: dict, full_size: bool) -> dict:
    labels = {split: R.label_report(recs) for split, recs in out.items()}
    train_questions = sum(s["questions"] for s in labels["train"].values())
    skewed = sorted(
        source
        for source, s in labels["train"].items()
        if s.get("choice_total", 0) >= 1000
        and abs(s["gold_first_rate"] - s["expected_first_rate"])
        > max(GATE_POSITION_TOLERANCE, 3 * math.sqrt(s["expected_first_rate"] * (1 - s["expected_first_rate"]) / s["choice_total"]))
    )
    gate = {
        "train_questions": train_questions,
        "enough_questions": train_questions >= GATE_MIN_TRAIN_QUESTIONS or not full_size,
        "no_clinc_excluded_options": checks["clinc_excluded_intent_options"] == 0,
        "gold_position_unbiased": not skewed,
        "position_skewed_sources": skewed,
    }
    gate["pass"] = gate["enough_questions"] and gate["no_clinc_excluded_options"] and gate["gold_position_unbiased"]
    return {
        "records": {split: len(recs) for split, recs in out.items()},
        "questions": {split: sum(s["questions"] for s in labels[split].values()) for split in out},
        "leakage": checks,
        "gate": gate,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/pointwise"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=None, help="rows per source split, for a quick check")
    parser.add_argument("--train-cap", type=int, default=None, help="train records per public source (run 3: 2000)")
    args = parser.parse_args()

    out, report = build(args.seed, args.limit_rows, train_cap=args.train_cap)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, recs in out.items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("records", "questions", "leakage", "gate")}, indent=2))
    if not report["gate"]["pass"]:
        raise SystemExit("Phase 3 gate failed; see report.json")


if __name__ == "__main__":
    main()
