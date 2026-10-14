"""Pointwise Phase 3 data: builders, policy rules, converters, and a full build
with a fake dataset loader. Pure Python (pydantic + pyyaml), so it runs in CI
with no network."""
import random
import re

import pytest

from src.pointwise.data import build as B
from src.pointwise.data import policies as P
from src.pointwise.data import records as R
from src.pointwise.data import sources as S


def _gold(target):
    return max(range(len(target)), key=target.__getitem__)


# --- records ------------------------------------------------------------------


def test_choice_keeps_gold_and_subsets():
    rng = random.Random(0)
    options = [f"o{i}" for i in range(20)]
    q, t = R.choice(rng, "pick", options, "o7", k=5)
    names = list(q["criteria"])
    assert len(names) == 5 and "o7" in names
    assert names[_gold(t)] == "o7" and sum(t) == 1


def test_choice_rejects_bad_input():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        R.choice(rng, "pick", ["a", "b"], "c")
    with pytest.raises(ValueError):
        R.choice(rng, "pick", ["a", "a", "b"], "a")


def test_choice_gold_position_is_spread():
    rng = random.Random(0)
    positions = [_gold(R.choice(rng, "pick", ["a", "b", "c", "d"], "a")[1]) for _ in range(4000)]
    for p in range(4):
        assert abs(positions.count(p) / 4000 - 0.25) < 0.03


def test_score_never_shuffles_levels():
    q, t = R.score("rate", ["low", "mid", "high"], 2)
    assert q["criteria"] == ["low", "mid", "high"] and _gold(t) == 2


def test_validate_catches_target_mismatch():
    rec = R.record("x", "train", 0, "s", {"q": R.noul("ok?", True)})
    R.validate(rec)
    rec["targets"]["q"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        R.validate(rec)


def test_label_report_gold_position_stats():
    recs = [R.record("x", "train", i, "s", {"q": ({"type": "choice", "instructions": "", "criteria": {"a": None, "b": None}}, R.one_hot(2, i % 2))}) for i in range(10)]
    report = R.label_report(recs)["x"]
    assert report["gold_first_rate"] == 0.5
    assert report["expected_first_rate"] == 0.5
    assert report["gold_mean_relative_position"] == 0.5


# --- policies -----------------------------------------------------------------


def test_condition_holds_at_ties():
    obj = {"n": 10, "c": "gold"}
    assert not P.Condition("", "n", ">", 10).holds(obj)
    assert P.Condition("", "n", "<=", 10).holds(obj)
    assert P.Condition("", "c", "in", ("gold", "silver")).holds(obj)
    assert not P.Condition("", "c", "in", ("basic",)).holds(obj)


def _numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]


@pytest.mark.parametrize("domain", P.DOMAINS, ids=[d.name for d in P.DOMAINS])
def test_band_gold_matches_level_text(domain):
    """Re-derives every band label from the level wording alone."""
    rng = random.Random(1)
    for _ in range(300):
        obj = {f.name: f.sample(rng) for f in domain.fields}
        q, t = P._band(rng, domain, obj)
        field = next(f for f in domain.fields if f.phrase in q["instructions"])
        value, levels = obj[field.name], q["criteria"]
        a = _numbers(levels[0])[0]
        b = _numbers(levels[2])[0]
        want = 0 if (value < a if field.money else value <= a) else 1 if value <= b else 2
        assert _gold(t) == want, (levels, value)


def test_decision_applies_rules_in_order(monkeypatch):
    domain = P.DOMAINS[0]
    yes = P.Condition("yes", "price", ">", -1)
    no = P.Condition("no", "price", "<=", -1)
    obj = {f.name: f.sample(random.Random(0)) for f in domain.fields}
    for conds, want in [((yes, yes, no), 0), ((yes, no, yes), 1), ((no, yes, no), 2), ((yes, no, no), 2)]:
        it = iter(conds)
        monkeypatch.setattr(P, "condition", lambda rng, d, avoid=frozenset(): next(it))
        q, t = P._decision(random.Random(0), domain, obj)
        assert list(q["criteria"])[_gold(t)] == domain.outcomes[want]


def test_compound_rules_never_repeat_a_field():
    rng = random.Random(2)
    for domain in P.DOMAINS:
        for _ in range(200):
            fields = [c.field for c in P._distinct(rng, domain, 3)]
            assert len(set(fields)) == 3


def test_article_before_vowel_nouns():
    assert P._a("order") == "an order" and P._a("support ticket") == "a support ticket"


def test_clean_fixes_source_artifacts():
    assert S.clean("Woods #39; record") == "Woods' record"
    assert S.clean(r"line one\nline two") == "line one\nline two"
    assert S.clean("a &amp; b quot;hiquot;") == 'a & b "hi"'


def test_policy_records_valid_and_deterministic():
    recs = P.policy_records(0, "train", 300)
    for rec in recs:
        R.validate(rec)
        assert 1 <= len(rec["request"]["questions"]) <= 6
    assert recs == P.policy_records(0, "train", 300)
    assert recs != P.policy_records(0, "dev", 300)
    assert any(isinstance(r["request"]["state"], dict) for r in recs)
    assert any(isinstance(r["request"]["state"], str) for r in recs)


def test_policy_json_state_carries_the_object():
    rng = random.Random(3)
    domain = P.DOMAINS[1]
    obj = {f.name: f.sample(rng) for f in domain.fields}
    for _ in range(20):
        state = P.render_state(rng, domain, obj, "APP-1")
        if isinstance(state, dict):
            assert all(state[f.name] == obj[f.name] for f in domain.fields)
            return
    pytest.fail("never rendered JSON")


# --- converters -----------------------------------------------------------------

CLINC_NAMES = ["weather", "timer", "exchange_rate", "play_music", "oos"] + sorted(S.CLINC_BANKING_DOMAINS)


def test_clinc_excluded_covers_domains_banking77_names_and_oos():
    excluded = S.clinc_excluded(CLINC_NAMES, ["exchange_rate", "top_up_failed"])
    assert S.CLINC_BANKING_DOMAINS <= excluded
    assert {"exchange_rate", "oos"} <= excluded and "weather" not in excluded
    with pytest.raises(ValueError):
        S.clinc_excluded(["weather"], [])


def test_clinc_drops_excluded_rows_and_options():
    excluded = S.clinc_excluded(CLINC_NAMES, ["exchange_rate"])
    rows = [{"text": f"msg {i}", "intent": i} for i in range(len(CLINC_NAMES))]
    recs = S.clinc(rows, CLINC_NAMES, random.Random(0), "train", excluded=excluded, copies=2)
    assert len(recs) == 2 * 3  # weather, timer, play music; two option subsets each
    shown = {n.replace("_", " ") for n in excluded}
    for rec in recs:
        R.validate(rec)
        assert not shown & set(rec["request"]["questions"]["intent"]["criteria"])


def test_mnli_skips_unlabelled_rows():
    rows = [{"premise": "p", "hypothesis": "h", "label": -1}, {"premise": "p2", "hypothesis": "h2", "label": 0}]
    recs = S.mnli(rows, S.NLI_OPTIONS, random.Random(0), "train")
    assert len(recs) == 1
    R.validate(recs[0])


def test_mcqa_skips_duplicate_options_and_bad_keys():
    good = {"question": "q?", "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]}, "answerKey": "B"}
    dup = {**good, "choices": {"text": ["a", "a", "c"], "label": ["A", "B", "C"]}}
    bad = {**good, "answerKey": "E"}
    recs = S.arc([good, dup, bad], None, random.Random(0), "train")
    assert len(recs) == 1
    q, t = recs[0]["request"]["questions"]["answer"], recs[0]["targets"]["answer"]
    assert list(q["criteria"])[_gold(t)] == "b"


def test_yelp_recommend_only_for_clear_ratings():
    rows = [{"text": f"r{i}", "label": i % 5} for i in range(500)]
    for rec in S.yelp(rows, None, random.Random(0), "train"):
        R.validate(rec)
        if "recommend" in rec["request"]["questions"]:
            stars = _gold(rec["targets"]["rating"])
            assert stars != 2 and _gold(rec["targets"]["recommend"]) == int(stars >= 3)


# --- full build with a fake loader -----------------------------------------------

BANKING = {
    "labels": ["card_arrival", "exchange_rate", "top_up_failed"],
    "texts": ["where is my card", "what is the exchange rate", "top up failed"],
    "test": [{"text": "what is the exchange rate", "label": 1}, {"text": "top up failed", "label": 2}],
}
LEAK_BANKING = "Where is my card"  # a Banking77 message planted in AG News train
LEAK_EVAL = "shared news text"  # planted in both AG News train and its test split


def fake_load_rows(name, split):
    n = 30
    if name == "mnli":
        return [{"premise": f"p{split}{i}", "hypothesis": "h", "label": i % 3} for i in range(n)], S.NLI_OPTIONS
    if name == "boolq":
        return [{"question": f"is it {split}{i}", "passage": "text", "answer": i % 2 == 0} for i in range(n)], None
    if name == "agnews":
        rows = [{"text": f"news {split} {i}", "label": i % 4} for i in range(n)]
        rows.append({"text": LEAK_EVAL, "label": 0})
        if split == "train":
            rows.append({"text": LEAK_BANKING, "label": 1})
        return rows, ["World", "Sports", "Business", "Sci/Tech"]
    if name in ("sst5", "yelp"):
        return [{"text": f"{name} {split} {i}", "label": i % 5} for i in range(n)], None
    if name == "clinc":
        return [{"text": f"clinc {split} {i}", "intent": i % len(CLINC_NAMES)} for i in range(n)], CLINC_NAMES
    if name in ("arc_easy", "arc_challenge", "openbookqa", "commonsenseqa"):
        key = "question_stem" if name == "openbookqa" else "question"
        return [{key: f"{name} {split} {i}?", "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]}, "answerKey": "C"} for i in range(n)], None
    if name == "qnli":
        return [{"question": f"q{i}", "sentence": "s", "label": i % 2} for i in range(n)], ["entailment", "not_entailment"]
    if name == "paws":
        return [{"sentence1": f"a{i}", "sentence2": "b", "label": i % 2} for i in range(n)], ["0", "1"]
    if name in ("emotion", "tweet_sentiment"):
        names = ["sadness", "joy", "anger"] if name == "emotion" else ["negative", "neutral", "positive"]
        return [{"text": f"{name} {i}", "label": i % 3} for i in range(n)], names
    raise KeyError(name)


@pytest.fixture(scope="module")
def built():
    return B.build(seed=0, limit_rows=40, load_rows=fake_load_rows, banking=BANKING)


def test_build_writes_every_split_with_valid_records(built):
    out, report = built
    for split in ("train", "dev", "test", "heldout"):
        assert out[split], split
        for rec in out[split]:
            assert "_text" not in rec
            assert rec["split"] == split
            R.validate(rec)
    assert len({r["id"] for s in out.values() for r in s}) == sum(len(s) for s in out.values())
    sources = {r["source"] for r in out["train"]}
    assert {"mnli", "boolq", "agnews", "sst5", "yelp", "clinc", "arc", "openbookqa", "commonsenseqa", "policies"} <= sources
    assert {r["source"] for r in out["heldout"]} == {"banking77", "qnli", "paws", "emotion", "tweet_sentiment"}


def test_build_drops_planted_leaks(built):
    out, report = built
    assert report["leakage"]["banking77_text_dropped"] >= 1
    assert report["leakage"]["train_eval_overlap_dropped"] >= 1
    states = [str(r["request"]["state"]) for r in out["train"]]
    assert not any(LEAK_BANKING in s or LEAK_EVAL in s for s in states)
    assert report["leakage"]["clinc_excluded_intent_options"] == 0


def test_build_never_trains_on_banking77_or_heldout_sources(built):
    out, _ = built
    train_sources = {r["source"] for s in ("train", "dev", "test") for r in out[s]}
    assert train_sources.isdisjoint({"banking77", "qnli", "paws", "emotion", "tweet_sentiment"})


def test_banking77_heldout_offers_every_label(built):
    out, _ = built
    for rec in (r for r in out["heldout"] if r["source"] == "banking77"):
        assert set(rec["request"]["questions"]["intent"]["criteria"]) == {"card arrival", "exchange rate", "top up failed"}


def test_build_report_gate_shape(built):
    _, report = built
    assert set(report["gate"]) >= {"train_questions", "enough_questions", "gold_position_unbiased", "pass"}
    assert report["gate"]["enough_questions"]  # small builds don't apply the size gate


def test_train_cap_limits_each_public_source_and_keeps_policies():
    out, report = B.build(seed=0, limit_rows=40, load_rows=fake_load_rows, banking=BANKING, train_cap=5)
    counts = {}
    for r in out["train"]:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    # arc_easy and arc_challenge share the "arc" source name, so arc may hold two capped sources
    assert all(n <= (10 if s == "arc" else 5) for s, n in counts.items() if s != "policies")
    assert counts["policies"] > 5
    assert report["train_cap"] == 5 and report["gate"]["pass"]


# --- Kev's option augmentation ----------------------------------------------------

def _choice_rec(n_options=4, gold=1):
    keys = [f"opt{i}" for i in range(n_options)]
    return {"id": "t/train/0", "source": "t", "split": "train",
            "request": {"state": "s", "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {k: None for k in keys}},
                                                    "yn": {"type": "noul", "instructions": "yes?"}}},
            "targets": {"q": R.one_hot(n_options, gold), "yn": [0.0, 1.0]}}


def _aug_gold(rec, key="q"):
    return list(rec["request"]["questions"][key]["criteria"])[rec["targets"][key].index(1.0)]


class _Fixed(random.Random):
    """random() always returns r, so each augmentation branch can be forced."""
    def __init__(self, r):
        super().__init__(0)
        self.r = r

    def random(self):
        return self.r


def test_augment_none_option_replaces_gold_and_becomes_answer():
    rec = _choice_rec()
    out = R.augment(rec, _Fixed(0.05))
    crit = out["request"]["questions"]["q"]["criteria"]
    assert "opt1" not in crit and len(crit) == 4
    assert _aug_gold(out) in {k for k, _ in R.NONE_OPTIONS}
    R.validate(out)


def test_augment_none_distractor_keeps_aug_gold():
    out = R.augment(_choice_rec(), _Fixed(0.15))
    crit = out["request"]["questions"]["q"]["criteria"]
    assert len(crit) == 5 and _aug_gold(out) == "opt1"
    assert set(crit) - {f"opt{i}" for i in range(4)} <= {k for k, _ in R.NONE_OPTIONS}
    R.validate(out)


def test_augment_distractor_adds_irrelevant_option_keeps_aug_gold():
    out = R.augment(_choice_rec(), _Fixed(0.30))
    crit = out["request"]["questions"]["q"]["criteria"]
    assert len(crit) == 5 and _aug_gold(out) == "opt1"
    assert (set(crit) - {f"opt{i}" for i in range(4)}) <= set(R.DISTRACTORS)
    R.validate(out)


def test_augment_leaves_noul_soft_targets_and_two_option_none_alone():
    rec = _choice_rec(n_options=2)
    rec["targets"]["yn"] = [0.0, 1.0]
    out = R.augment(rec, _Fixed(0.05))  # none-replaces-gold needs > 2 options
    assert set(out["request"]["questions"]["q"]["criteria"]) == {"opt0", "opt1"} and _aug_gold(out) == "opt1"
    assert out["request"]["questions"]["yn"] == rec["request"]["questions"]["yn"]
    soft = _choice_rec()
    soft["targets"]["q"] = [0.25, 0.25, 0.25, 0.25]
    assert R.augment(soft, _Fixed(0.05))["request"]["questions"]["q"] == soft["request"]["questions"]["q"]


def test_none_pair_differs_only_in_gold_presence():
    present, absent = R.none_pair(_choice_rec(), random.Random(3))
    pc, ac = present["request"]["questions"]["q"]["criteria"], absent["request"]["questions"]["q"]["criteria"]
    assert _aug_gold(present) == "opt1" and "opt1" not in ac
    assert list(ac) == [k for k in pc if k != "opt1"]  # same order, gold removed
    assert _aug_gold(absent) in {k for k, _ in R.NONE_OPTIONS} and _aug_gold(absent) in pc
    assert present["id"] != absent["id"]
    for r in (present, absent):
        R.validate(r)
    assert R.none_pair(_choice_rec(n_options=2), random.Random(0)) == []


def test_position_gate_allows_chance_on_small_sources_but_not_bias():
    def report_for(n, first):
        recs = []
        for i in range(n):
            gold = 0 if i < first else 1 + i % 2
            recs.append({"id": f"x/train/{i}", "source": "x", "split": "train",
                         "request": {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None, "c": None}}}},
                         "targets": {"q": R.one_hot(3, gold)}})
        return B.make_report({"train": recs}, {"clinc_excluded_intent_options": 0}, full_size=False)["gate"]
    assert report_for(2000, 620)["gold_position_unbiased"]  # 0.310 vs 0.333: 2.2 SE, chance
    assert not report_for(2000, 560)["gold_position_unbiased"]  # 0.280: 5 SE, a bias
    assert not report_for(30000, 9300)["gold_position_unbiased"]  # 0.310 at full size: still flagged
