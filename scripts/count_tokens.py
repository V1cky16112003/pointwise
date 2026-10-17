"""Token counts per request shape: Pointwise (one pass, and with the state already cached) against generate-and-parse.

Counts are model input tokens, from the real tokenizer, for the Phase 6 latency request (one ~512-token state, k
three-option choice questions) and for the held-out sources (one question per record).
  pointwise one pass   state + every question, read once (what usage.input_tokens reports)
  pointwise cached     the questions only, the state's KV cache already stored
  generate-and-parse   one prompt per question, each carrying the whole state again (the Phase 5 prompt format)
Generate-and-parse is also capped at 8 new tokens per question (the Phase 5 setting); what it actually generated was
not recorded, and it is not counted here.
Usage: python -m scripts.count_tokens --heldout /tmp/pointwise-data-run5/heldout.jsonl > reports/pointwise-tokens.md
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

KS = (1, 2, 4, 8)


def ratios(state: int, questions: list[int], prompts: list[int]) -> dict:
    """Token counts for one request and how many times fewer Pointwise reads than generate-and-parse."""
    one_pass, cached, gp = state + sum(questions), sum(questions), sum(prompts)
    return {"one_pass": one_pass, "cached": cached, "generate_and_parse": gp,
            "x_one_pass": gp / one_pass, "x_cached": gp / cached}


def main() -> None:
    import transformers

    import src.pointwise.baselines as B
    from scripts.bench_decide import shared_state_request
    from scripts.bench_packing import build_request
    from src.pointwise.encoding import encode_request
    from src.pointwise.schema import SystemOneRequest, question_options

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--heldout", required=True, help="Phase 3 heldout.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    args = parser.parse_args()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)

    ids, _ = build_request(tok, 512, 1)
    state_text = tok.decode(ids)
    print("# Pointwise token consumption\n")
    print(f"Tokenizer: `{args.model}`. Reproduce: `python -m scripts.count_tokens --heldout <heldout.jsonl>`.\n")
    print("## One state, k questions (Phase 6 latency request: ~524-token state, three-option choice questions)\n")
    print("| k | Pointwise, one pass | Pointwise, state cached | Generate-and-parse prompts | x fewer (one pass) | x fewer (cached) |")
    print("|---|---|---|---|---|---|")
    for k in KS:
        req = SystemOneRequest.model_validate(shared_state_request(state_text, k))
        e = encode_request(tok, req)
        prompts = [len(tok(B.question_prompt(req, key)[0], add_special_tokens=False).input_ids) for key in req.questions]
        r = ratios(len(e.state_ids), [len(q.ids) for q in e.questions], prompts)
        print(f"| {k} | {r['one_pass']} | {r['cached']} | {r['generate_and_parse']} | {r['x_one_pass']:.1f}x | {r['x_cached']:.1f}x |")
    print("\n## Held-out sources (one question per record; Pointwise's tokens only, generate-and-parse was not counted here)\n")
    print("| Source | Options | State tokens | Question + option tokens | Total | Question + options share |")
    print("|---|---|---|---|---|---|")
    by = collections.defaultdict(list)
    for line in open(args.heldout):
        rec = json.loads(line)
        assert len(rec["targets"]) == 1, "held-out records have one question each"
        request = SystemOneRequest.model_validate(rec["request"])
        e = encode_request(tok, request)
        n_options = len(question_options(next(iter(request.questions.values())))[1])
        by[rec["source"]].append((len(e.state_ids), sum(len(q.ids) for q in e.questions), n_options))
    for source, v in sorted(by.items()):
        s, q, n = (statistics.mean(x[i] for x in v) for i in range(3))
        print(f"| {source} | {n:.0f} | {s:.1f} | {q:.1f} | {s + q:.1f} | {100 * q / (s + q):.0f}% |")


if __name__ == "__main__":
    main()
