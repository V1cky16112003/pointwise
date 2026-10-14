"""Option shuffling and flip-rate bookkeeping in scripts/phase5_eval.py."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts import phase5_eval as P

OPTIONS = ["alpha", "beta", "gamma", "delta", "epsilon"]


def record(i, gold=2):
    return {"source": "emotion", "id": f"emotion/heldout/{i}",
            "request": {"state": "text", "questions": {"emotion": {"type": "choice", "instructions": "?", "criteria": {o: None for o in OPTIONS}}}},
            "targets": {"emotion": [1.0 if j == gold else 0.0 for j in range(len(OPTIONS))]}}


def test_permuted_keeps_gold_on_the_same_option():
    rec = record(0, gold=3)
    shuffled, order = P.permuted(rec)
    names = list(shuffled["request"]["questions"]["emotion"]["criteria"])
    gold_now = shuffled["targets"]["emotion"].index(1.0)
    assert names[gold_now] == "delta" and sorted(order) == list(range(5)) and names != OPTIONS
    assert rec["request"]["questions"]["emotion"]["criteria"] == {o: None for o in OPTIONS}  # input untouched


def test_flip_rate_is_zero_for_an_order_blind_scorer_and_positive_for_a_biased_one():
    records = [record(i) for i in range(40)]

    def blind(rec):  # the same score for an option wherever it sits
        return [float(len(o)) + 0.01 * OPTIONS.index(o) for o in rec["request"]["questions"]["emotion"]["criteria"]]

    def first_option_wins(rec):
        return [1.0] + [0.0] * (len(OPTIONS) - 1)

    assert P.flip_rates(records, blind)["emotion"] == {"n": 40, "flip_rate": 0.0}
    assert P.flip_rates(records, first_option_wins)["emotion"]["flip_rate"] > 0.5


def test_flip_rates_skip_sources_with_a_fixed_option_order():
    rec = {**record(0), "source": "qnli"}
    assert P.flip_rates([rec], lambda r: [0.0] * 5) == {}
