"""Record format, question builders and the label report (pure Python, CI-safe).

A record is one /v1/systemone request plus a target distribution per question:

    {"id": "agnews/train/17", "source": "agnews", "split": "train",
     "request": {"state": ..., "questions": {key: {type, instructions, criteria}}},
     "targets": {key: [p over the question's options, in option order]}}

Option order for a target is schema.question_options(): choice criteria order,
[false, true] for noul, level order for score. Targets are one-hot here; the
format allows soft targets (Phase 5 ChaosNLI, teacher labels).

Every builder takes the record's random.Random, so a build is reproducible
from one seed.
"""
from __future__ import annotations

import random
from collections import Counter

from src.pointwise.schema import SystemOneRequest, question_options


def one_hot(n: int, index: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(n)]


def choice(
    rng: random.Random,
    instructions: str,
    options: list[str],
    gold: str,
    k: int | None = None,
    descriptions: dict[str, str] | None = None,
) -> tuple[dict, list[float]]:
    """A Choice question with the options shuffled. With k, keep only k options:
    the gold plus k-1 random distractors. Shuffling every time is what stops the
    gold from sitting at a predictable position."""
    if gold not in options:
        raise ValueError(f"gold {gold!r} is not an option")
    if len(set(options)) != len(options):
        raise ValueError("options must be unique")
    pool = [o for o in options if o != gold]
    if k is not None:
        pool = rng.sample(pool, k - 1)
    picked = pool + [gold]
    rng.shuffle(picked)
    descriptions = descriptions or {}
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": {o: descriptions.get(o) for o in picked},
    }
    return question, one_hot(len(picked), picked.index(gold))


def noul(instructions: str, answer: bool, criteria: dict | None = None) -> tuple[dict, list[float]]:
    question = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    return question, one_hot(2, int(answer))


def score(instructions: str, levels: list[str], gold_level: int) -> tuple[dict, list[float]]:
    """An ordered Score question. Levels are never shuffled: their order is the scale."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}, one_hot(
        len(levels), gold_level
    )


def record(source: str, split: str, index: int, state, questions: dict[str, tuple[dict, list[float]]]) -> dict:
    return {
        "id": f"{source}/{split}/{index}",
        "source": source,
        "split": split,
        "request": {"state": state, "questions": {key: q for key, (q, _) in questions.items()}},
        "targets": {key: t for key, (_, t) in questions.items()},
    }


def validate(rec: dict) -> None:
    """Raises unless the request is a valid /v1/systemone request and every
    target is a distribution over exactly that question's options."""
    request = SystemOneRequest.model_validate(rec["request"])
    if set(rec["targets"]) != set(request.questions):
        raise ValueError(f"{rec['id']}: target keys differ from question keys")
    for key, question in request.questions.items():
        target = rec["targets"][key]
        names, _ = question_options(question)
        if len(target) != len(names):
            raise ValueError(f"{rec['id']}/{key}: {len(target)} targets for {len(names)} options")
        if abs(sum(target) - 1) > 1e-6 or min(target) < 0:
            raise ValueError(f"{rec['id']}/{key}: target is not a distribution")


# Ported from Kev's kev/data.py: "none" options appear both as the right answer and as a
# wrong one, in varied wording, or the model learns "this wording => pick it".
NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
                ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
                ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
                ("other", "An answer not covered by the other options"), ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes",
               "taxes": "Unrelated: quarterly tax filing"}


def _hard_gold(target: list[float]) -> int | None:
    """Index of a one-hot target's gold option; None for a soft target, which augmentation leaves alone
    (adding or removing options would change what it means)."""
    return target.index(1.0) if sorted(target)[-2:] == [0.0, 1.0] else None


def _choice(q: dict, crit: dict, gold: str, rng: random.Random) -> tuple[dict, list[float]]:
    keys = list(crit)
    rng.shuffle(keys)
    return {**q, "criteria": {k: crit[k] for k in keys}}, one_hot(len(keys), keys.index(gold))


def augment(rec: dict, rng: random.Random, p_none=0.10, p_none_distract=0.12, p_distract=0.15) -> dict:
    """Kev's augment on one record's Choice questions, each drawn once: with p_none a none option replaces the
    gold and becomes the answer (only with > 2 options); with p_none_distract a none option is added as a wrong
    answer; with p_distract an irrelevant distractor is added. Options are reshuffled either way."""
    questions, targets = {}, {}
    for key, q in rec["request"]["questions"].items():
        target = rec["targets"][key]
        gi = _hard_gold(target) if q["type"] == "choice" else None
        if gi is None:
            questions[key], targets[key] = q, target
            continue
        crit, gold = dict(q["criteria"]), list(q["criteria"])[gi]
        nones = [o for o in NONE_OPTIONS if o[0] not in crit]
        extra = [k for k in DISTRACTORS if k not in crit]
        r = rng.random()
        if r < p_none and len(crit) > 2 and nones:
            nk, nd = rng.choice(nones)
            crit.pop(gold)
            crit[nk], gold = nd, nk
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and nones:
            nk, nd = rng.choice(nones)
            crit[nk] = nd
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and extra:
            k = rng.choice(extra)
            crit[k] = DISTRACTORS[k]
        questions[key], targets[key] = _choice(q, crit, gold, rng)
    return {**rec, "request": {**rec["request"], "questions": questions}, "targets": targets}


def none_pair(rec: dict, rng: random.Random) -> list[dict]:
    """Kev's minimal pair: one eligible Choice (>= 3 options, one-hot target) asked twice with the same added none
    option and option order, once with the gold present (none is wrong) and once with it removed (none is right).
    [] when no question is eligible."""
    eligible = [k for k, q in rec["request"]["questions"].items()
                if q["type"] == "choice" and len(q["criteria"]) >= 3 and _hard_gold(rec["targets"][k]) is not None]
    if not eligible:
        return []
    key = rng.choice(eligible)
    q = rec["request"]["questions"][key]
    gold = list(q["criteria"])[_hard_gold(rec["targets"][key])]
    nk, nd = rng.choice([o for o in NONE_OPTIONS if o[0] not in q["criteria"]] or [("none_of_these", None)])
    keys = list(q["criteria"]) + [nk]
    rng.shuffle(keys)
    crit = {k: (nd if k == nk else q["criteria"][k]) for k in keys}
    absent = {k: v for k, v in crit.items() if k != gold}
    pair = []
    for tag, c, answer in (("present", crit, gold), ("absent", absent, nk)):
        pair.append({**rec, "id": f"{rec['id']}#none-{tag}",
                     "request": {"state": rec["request"]["state"], "questions": {key: {**q, "criteria": c}}},
                     "targets": {key: one_hot(len(c), list(c).index(answer))}})
    return pair


def pick(rng: random.Random, templates: list[str], **fields) -> str:
    """A random paraphrase of an instruction."""
    return rng.choice(templates).format(**fields)


def label_report(records: list[dict]) -> dict:
    """Per source: records, questions, question types, option counts, and where
    the gold option sits. For unordered choices the gold position should be
    close to uniform; a skew would let a model score by position alone."""
    report: dict = {}
    for rec in records:
        src = report.setdefault(
            rec["source"],
            {
                "records": 0,
                "questions": 0,
                "types": Counter(),
                "options": Counter(),
                "questions_per_record": Counter(),
                "choice_total": 0,
                "gold_first": 0,  # gold is the first option
                "expected_first": 0.0,  # sum of 1/K: gold_first if position were random
                "gold_relative_position": 0.0,  # sum of index/(K-1); mean 0.5 if random
                "noul_true": 0,
                "noul_total": 0,
            },
        )
        src["records"] += 1
        qs = rec["request"]["questions"]
        src["questions_per_record"][len(qs)] += 1
        for key, q in qs.items():
            target = rec["targets"][key]
            gold = max(range(len(target)), key=target.__getitem__)
            src["questions"] += 1
            src["types"][q["type"]] += 1
            src["options"][_bucket(len(target))] += 1
            if q["type"] == "choice" and len(target) > 1:
                src["choice_total"] += 1
                src["gold_first"] += gold == 0
                src["expected_first"] += 1 / len(target)
                src["gold_relative_position"] += gold / (len(target) - 1)
            elif q["type"] == "noul":
                src["noul_total"] += 1
                src["noul_true"] += gold
    for src in report.values():
        for key in ("types", "options", "questions_per_record"):
            src[key] = dict(sorted(src[key].items()))
        if src["choice_total"]:
            n = src["choice_total"]
            src["gold_first_rate"] = src.pop("gold_first") / n
            src["expected_first_rate"] = src.pop("expected_first") / n
            src["gold_mean_relative_position"] = src.pop("gold_relative_position") / n
        else:
            for key in ("gold_first", "expected_first", "gold_relative_position"):
                src.pop(key)
    return report


def _bucket(n: int) -> str:
    for hi in (2, 3, 5, 10, 25, 50, 100):
        if n <= hi:
            return f"<= {hi}"
    return "> 100"
