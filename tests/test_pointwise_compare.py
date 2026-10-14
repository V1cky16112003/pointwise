"""Arm alignment, paired bootstrap, reliability bins and flip rate. Pure Python, runs in CI."""
import random

import pytest

from src.pointwise import compare as C


def row(source, id, logits, target):
    return {"source": source, "id": id, "question": 0, "logits": logits, "target": target}


def test_align_sorts_and_rejects_mismatches():
    a = [row("s", "2", [0, 1], [0, 1]), row("s", "1", [1, 0], [1, 0])]
    b = [row("s", "1", [0, 1], [1, 0]), row("s", "2", [1, 0], [0, 1])]
    aligned = C.align({"a": a, "b": b})
    assert [r["id"] for r in aligned["a"]] == ["1", "2"] == [r["id"] for r in aligned["b"]]
    with pytest.raises(ValueError, match="same questions"):
        C.align({"a": a, "b": b[:1]})
    with pytest.raises(ValueError, match="different targets"):
        C.align({"a": a, "b": [row("s", "1", [0, 1], [0, 1]), row("s", "2", [1, 0], [0, 1])]})


def test_paired_bootstrap_detects_a_real_gap_and_not_a_tie():
    rng = random.Random(1)
    clusters = [("s", i) for i in range(400)]
    base = [rng.random() < 0.5 for _ in clusters]
    better = [True if not x and rng.random() < 0.4 else x for x in base]  # fixes 40% of the misses
    point, lo, hi = C.paired_bootstrap(better, base, clusters, n_boot=500)
    assert point > 0 and lo > 0 and lo <= point <= hi
    point, lo, hi = C.paired_bootstrap(base, base, clusters, n_boot=200)
    assert point == lo == hi == 0


def test_paired_bootstrap_resamples_inside_each_source():
    # Source "x" always agrees, source "y" always differs: every resample must keep both sizes.
    clusters = [("x", i) for i in range(10)] + [("y", i) for i in range(10)]
    a = [True] * 10 + [True] * 10
    b = [True] * 10 + [False] * 10
    point, lo, hi = C.paired_bootstrap(a, b, clusters, n_boot=100)
    assert point == lo == hi == 0.5


def test_reliability():
    bins = C.reliability([0.95, 0.92, 0.15], [True, False, False], n_bins=10)
    assert [b["n"] for b in bins] == [1, 2] and bins[1]["accuracy"] == 0.5
