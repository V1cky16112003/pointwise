"""Comparing arms on the same held-out questions (Pointwise Phase 5). Pure Python, runs in CI.

Every arm writes rows shaped like the Phase 4 held-out predictions:
    {"source", "id", "question", "logits", "target"}
A log-likelihood arm puts each option's summed log-probability in `logits`, so
`metrics.softmax` renormalises over the candidates and `train.score_rows` scores it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.pointwise import metrics as M


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def row_key(row: dict) -> tuple:
    return row["source"], row["id"], row["question"]


def align(arms: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Every arm's rows sorted by key. Raises unless all arms hold exactly the same questions with the
    same targets: a paired bootstrap on different questions would be meaningless."""
    out = {name: sorted(rows, key=row_key) for name, rows in arms.items()}
    names = list(out)
    ref = out[names[0]]
    for name in names[1:]:
        rows = out[name]
        if [row_key(r) for r in rows] != [row_key(r) for r in ref]:
            raise ValueError(f"arm {name!r} does not hold the same questions as {names[0]!r}")
        if any(a["target"] != b["target"] for a, b in zip(rows, ref)):
            raise ValueError(f"arm {name!r} has different targets from {names[0]!r}")
    return out


def paired_bootstrap(a: list[bool], b: list[bool], clusters: list, n_boot: int = 2000, seed: int = 0) -> tuple:
    """(mean(a) - mean(b), 95% low, 95% high), resampling whole clusters with replacement.
    A cluster is one source's question ids (pass (source, id) pairs); sampling stays inside each
    source, so every resample keeps the real number of questions per source."""
    members: dict = {}
    for i, c in enumerate(clusters):
        members.setdefault(c, []).append(i)
    by_source: dict = {}
    for c in members:
        by_source.setdefault(c[0], []).append(c)
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        total = 0
        for cs in by_source.values():
            for c in rng.choices(cs, k=len(cs)):
                total += sum(a[i] - b[i] for i in members[c])
        diffs.append(total / n)
    diffs.sort()
    point = sum(x - y for x, y in zip(a, b)) / n
    return point, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def reliability(confidences: list[float], correct: list[bool], n_bins: int = 10) -> list[dict]:
    """Per confidence bin: how many questions, mean confidence, and the share answered correctly."""
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        bins[min(n_bins - 1, max(0, int(c * n_bins - 1e-12)))].append(i)
    return [
        {"lo": k / n_bins, "hi": (k + 1) / n_bins, "n": len(m),
         "confidence": sum(confidences[i] for i in m) / len(m), "accuracy": sum(correct[i] for i in m) / len(m)}
        for k, m in enumerate(bins) if m
    ]


def fit_arm_temperature(dev_rows: list[dict]) -> float:
    """One T fitted on the arm's own dev questions, the same protocol as Pointwise."""
    return M.fit_temperature([r["logits"] for r in dev_rows], [r["target"] for r in dev_rows])
