"""Rule-structure records: Kev's compositional policy groups, ported to Pointwise records.

Adapted from jaredpalmer/kev `kev/composition.py` (Apache-2.0, main @ 0a6d3fa) with
Kev-0.8B's base-recipe settings: 60 random rule trees, 7 training groups each, render
styles 0/1/3/4. Each group is two minimal pairs over one policy: a relevant pair whose
single fact edit flips accept/reject, and an irrelevant pair that only changes a routing
reference the policy says to ignore. See docs/designs/pointwise-phase4-review.md.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from src.pointwise.data import records as R

STRUCTURES = 60
GROUPS_PER_STRUCTURE = 7
STYLES = (0, 1, 3, 4)
# Kev's held-out eval shapes (decision-v8 manifest); no training tree may equal one of them
EXCLUDED_KEYS = frozenset({
    "and(0,if(1,2,3))", "and(0,or(1,2))", "and(not(0),or(not(1),not(2)))", "if(0,not(1),2)",
    "not(or(0,and(1,2)))", "or(0,and(1,not(2)))", "or(0,unless(1,2))", "or(and(0,1),not(2))",
})
KINDS = ("lt", "le", "gt", "ge", "eq", "range", "match", "elapsed", "flag")


def atom_value(atom, facts):
    if any(key not in facts for key in atom["fields"]):
        return None
    values = [facts[k] for k in atom["fields"]]
    kind, t, x = atom["kind"], atom["threshold"], values[0]
    if kind == "lt": return x < t
    if kind == "le": return x <= t
    if kind == "gt": return x > t
    if kind == "ge": return x >= t
    if kind == "eq": return x == t
    if kind == "range": return t <= x <= t + 10
    if kind == "match": return x == values[1]
    if kind == "elapsed": return (date.fromisoformat(values[1]) - date.fromisoformat(x)).days <= t
    return x  # flag


def evaluate(tree, atoms, facts):
    """Three-valued: None when a missing fact leaves the answer open."""
    if isinstance(tree, int):
        return atom_value(atoms[tree], facts)
    op, *children = tree
    values = [evaluate(t, atoms, facts) for t in children]
    if op == "not": return None if values[0] is None else not values[0]
    if op == "unless":
        values, op = [values[0], None if values[1] is None else not values[1]], "and"
    if op == "and": return False if False in values else None if None in values else True
    if op == "or": return True if True in values else None if None in values else False
    condition, yes, no = values  # if
    return yes if condition is True else no if condition is False else yes if yes == no else None


def atom_text(atom):
    kind, fields, t = atom["kind"], atom["fields"], atom["threshold"]
    words = {"lt": "less than", "le": "at most", "gt": "greater than", "ge": "at least", "eq": "equal to"}
    if kind in words: return f"{fields[0]} is {words[kind]} {t}"
    if kind == "range": return f"{fields[0]} is between {t} and {t + 10}, including both endpoints"
    if kind == "match": return f"{fields[0]} is the same person as {fields[1]}"
    if kind == "elapsed": return f"the elapsed days from {fields[0]} to {fields[1]} are at most {t}"
    return f"{fields[0]} is yes"


def render(tree, atoms, style):
    """0 = logic-like, 1 = 'both/at least one of' lists, 3 = plain prose, 4 = clause per line."""
    if isinstance(tree, int): return atom_text(atoms[tree])
    op, *children = tree
    p = [render(c, atoms, style) for c in children]
    if style == 3:
        if op == "not": return f"the condition \"{p[0]}\" fails"
        if op == "and": return f"{p[0]}, and also {p[1]}"
        if op == "or": return f"either {p[0]}, or else {p[1]}"
        if op == "unless": return f"{p[0]}, except when {p[1]}"
        return f"when {p[0]} the requirement is that {p[1]}, and when it is not the requirement is that {p[2]}"
    if style == 4:
        if op == "not": return f"[NOT: {p[0]}]"
        if op in ("and", "or"): return f"[{'ALL' if op == 'and' else 'ANY'} of: {p[0]} | {p[1]}]"
        if op == "unless": return f"[{p[0]} UNLESS {p[1]}]"
        return f"[IF {p[0]} THEN {p[1]} ELSE {p[2]}]"
    if op == "not": return f"NOT ({p[0]})" if style == 0 else f"it is not the case that ({p[0]})"
    if op in ("and", "or"):
        if style == 0: return f"({p[0]}) {op.upper()} ({p[1]})"
        return f"{'both' if op == 'and' else 'at least one of'} these conditions hold: [({p[0]}); ({p[1]})]"
    if op == "unless": return f"({p[0]}) holds and the exception ({p[1]}) does not hold"
    return f"if ({p[0]}), use ({p[1]}); otherwise use ({p[2]})"


WRAPPERS = {
    0: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    1: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    3: "A case is approved when {rule}; any other case is denied. Routing references play no part in the decision.",
    4: "DECISION RULE {rule} -> approve; otherwise reject. Ignore the routing reference.",
}


# --- structure keys, so equivalent trees count once and held-out shapes stay out ---

def _skeleton(tree):
    if isinstance(tree, int): return "_"
    op, *children = tree
    parts = [_skeleton(c) for c in children]
    return f"{op}({','.join(sorted(parts) if op in ('and', 'or') else parts)})"


def _sort_commutative(tree):
    if isinstance(tree, int): return tree
    op, *children = tree
    children = [_sort_commutative(c) for c in children]
    return (op, *(sorted(children, key=_skeleton) if op in ("and", "or") else children))


def relabel(tree):
    """Renumber leaves in first-appearance order."""
    mapping = {}
    def walk(t):
        if isinstance(t, int): return mapping.setdefault(t, len(mapping))
        return (t[0], *[walk(c) for c in t[1:]])
    return walk(tree)


def canonical(tree) -> str:
    def text(t):
        return str(t) if isinstance(t, int) else f"{t[0]}({','.join(text(c) for c in t[1:])})"
    return text(relabel(_sort_commutative(tree)))


def _push_negation(tree):
    """De Morgan normal form."""
    if isinstance(tree, int): return tree
    op, *children = tree
    if op == "not":
        inner = children[0]
        if isinstance(inner, int): return tree
        iop, *ic = inner
        if iop == "not": return _push_negation(ic[0])
        if iop in ("and", "or"): return ("or" if iop == "and" else "and", *[_push_negation(("not", c)) for c in ic])
        if iop == "unless": return _push_negation(("or", ("not", ic[0]), ic[1]))
        return ("not", _push_negation(inner))
    if op == "unless": return ("and", _push_negation(children[0]), _push_negation(("not", children[1])))
    return (op, *[_push_negation(c) for c in children])


def structure_keys(tree) -> set[str]:
    return {canonical(tree), canonical(_push_negation(tree))}


def _random_tree(rng, depth, next_leaf):
    if depth == 0 or (depth < 3 and rng.random() < 0.25):
        return next_leaf()
    op = rng.choice(["and", "or", "not", "unless", "if", "and", "or"])
    arity = {"not": 1, "if": 3}.get(op, 2)
    return (op, *[_random_tree(rng, depth - 1, next_leaf) for _ in range(arity)])


def sample_trees(n: int, seed) -> list:
    """n distinct structures with 2-4 leaves, none equal to an excluded shape."""
    rng, trees, keys = random.Random(f"{seed}:trees"), [], set()
    for _ in range(20000):
        if len(trees) >= n: break
        counter = [0]
        def next_leaf():
            counter[0] += 1
            return counter[0] - 1
        t = _random_tree(rng, rng.choice([2, 2, 3]), next_leaf)
        if isinstance(t, int) or not 2 <= counter[0] <= 4: continue
        k = structure_keys(t)
        if k & EXCLUDED_KEYS or k & keys: continue
        keys |= k
        trees.append(relabel(t))
    if len(trees) < n: raise ValueError(f"only {len(trees)} distinct structures found")
    return trees


def _leaves(tree) -> set[int]:
    return {tree} if isinstance(tree, int) else set().union(*(_leaves(t) for t in tree[1:]))


def _atoms(tree, rng):
    nouns = rng.sample(["request", "account", "package", "review", "member", "shipment", "entry", "case"], 4)
    atoms = []
    for i in range(max(_leaves(tree)) + 1):
        kind, prefix = rng.choice(KINDS), nouns[i]
        fields = {"match": [f"{prefix} signer", f"{prefix} designated approver"],
                  "elapsed": [f"{prefix} start date", f"{prefix} end date"],
                  "flag": [f"{prefix} verified"]}.get(kind, [f"{prefix} value"])
        atoms.append({"kind": kind, "fields": fields, "threshold": rng.randint(5, 60)})
    return atoms


def _domains(atoms, rng):
    domains = {}
    for a in atoms:
        kind, fs, t = a["kind"], a["fields"], a["threshold"]
        if kind == "match":
            people = rng.sample(["Mira", "Noah", "Aiko", "Ravi", "Sana", "Elin", "Tomas", "Kofi"], 3)
            domains.update({k: people for k in fs})
        elif kind == "elapsed":
            day = date(2027, rng.randint(1, 8), rng.randint(1, 28))
            domains[fs[0]] = [day.isoformat()]
            domains[fs[1]] = [(day + timedelta(days=n)).isoformat() for n in (max(0, t - 1), t, t + 1, t + 10)]
        elif kind == "flag":
            domains[fs[0]] = [False, True]
        else:
            domains[fs[0]] = [t - 1, t, t + 1, t + 10, t + 11]
    domains["routing reference"] = [rng.randint(100, 500), rng.randint(501, 999)]
    return domains


def _case_text(facts, order):
    def value(v):
        return "yes" if v is True else "no" if v is False else str(v)
    return " ".join(f"The {k} is {value(facts[k])}." for k in order)


def group(tree, rng) -> list[tuple[dict, bool]]:
    """Four (state, accept) pairs: relevant pair (labels differ), irrelevant pair (labels equal)."""
    atoms = _atoms(tree, rng)
    domains = _domains(atoms, rng)
    changed = None
    for _ in range(1000):
        facts = {k: rng.choice(v) for k, v in domains.items()}
        label = evaluate(tree, atoms, facts)
        keys = list(domains)
        rng.shuffle(keys)
        for key in keys:
            for value in domains[key]:
                # the edited field must decide the answer: without it the rule is undetermined
                if evaluate(tree, atoms, {**facts, key: value}) != label and \
                        evaluate(tree, atoms, {k: v for k, v in facts.items() if k != key}) is None:
                    changed = {**facts, key: value}
                    break
            if changed: break
        if changed: break
    if changed is None:
        raise ValueError(f"cannot create a decisive edit for {canonical(tree)}")
    other_ref = next(v for v in domains["routing reference"] if v != facts["routing reference"])
    nuisance = {**facts, "routing reference": other_ref}
    order = list(facts)
    rng.shuffle(order)
    style = rng.choice(STYLES)
    policy = WRAPPERS[style].format(rule=render(tree, atoms, style))
    return [({"policy": policy, "case": _case_text(values, order)}, evaluate(tree, atoms, values))
            for values in (facts, changed, facts, nuisance)]


def rule_records(seed: int, n: int | None = None) -> list[dict]:
    """Train-split records in group order; n truncates to whole groups of 4."""
    out = []
    for s, tree in enumerate(sample_trees(STRUCTURES, f"pointwise-rules:{seed}")):
        rng = random.Random(f"pointwise-rules:{seed}:{s}")
        for _ in range(GROUPS_PER_STRUCTURE):
            for state, accept in group(tree, rng):
                question = R.choice(rng, "Apply the policy to this case.", ["accept", "reject"],
                                    "accept" if accept else "reject",
                                    descriptions={"accept": "The policy permits this case",
                                                  "reject": "The policy does not permit this case"})
                out.append(R.record("rules", "train", len(out), state, {"decision": question}))
    return out if n is None else out[: n - n % 4]
