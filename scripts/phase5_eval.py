"""Phase 5 evaluation arms (docs/designs/pointwise-plan.md, Phase 5).

Scores the same dev and held-out questions that Phase 4 run 3 used, with every arm:
    pointwise    run 3's best weights (adapter + head)
    qwen3_base   Qwen3-1.7B-Base, no LoRA, no head: option log-likelihoods (src/pointwise/baselines.py)
    qwen25_base  Qwen2.5-1.5B-Instruct, no adapter
    qwen25_lora  the same model with the Phase 0 Banking77 LoRA
The Qwen2.5 arms read Banking77 through the specialist's own prompt (src/readout.py) and the other
sources through the plain-text prompt. Writes <output>/arms/<arm>_{dev,heldout}.jsonl, flip_rates.json
and latency.json; scripts/phase5_report.py turns them into the report.

Usage (from the repo root):
    python -m scripts.phase5_eval --data-dir D --state run3/best/state.pt --adapter-dir A --output-dir O [--limit 6]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
import transformers

from src.config import load_task_config
from src.pointwise import baselines as B
from src.pointwise import train as T
from src.pointwise.schema import SystemOneRequest, question_options
from src.readout import LabelScorer

QWEN3 = "Qwen/Qwen3-1.7B-Base"
QWEN25 = "Qwen/Qwen2.5-1.5B-Instruct"
FLIP_SOURCES = ("banking77", "emotion", "tweet_sentiment")  # choice questions; noul has a fixed [no, yes] order
FLIP_QUESTIONS = 200  # per source
RUN3_BEST_STEP = 1541
LATENCY_KS = (1, 2, 4, 8)


def log(*a) -> None:
    print(*a, flush=True)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def load_samples(args, tokenizer):
    """Run 3's dev and held-out examples, reproduced exactly as T.main drew them, and their records."""
    targs = T.parse_args(["--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir)])
    rng = random.Random(targs.seed)  # one rng for dev then held-out, as in train.main
    dev, _ = T.load_examples(args.data_dir / "dev.jsonl", tokenizer, targs)
    rng.shuffle(dev)
    dev = dev[: targs.eval_dev_records]
    held, _ = T.load_examples(args.data_dir / "heldout.jsonl", tokenizer, targs)
    rng.shuffle(held)
    held = T._stratified(held, targs.eval_heldout_per_source)
    expected = json.loads(Path("reports/run3_sample_ids.json").read_text())
    assert sorted(e.id for e in dev) == expected["dev"], "dev sample differs from Phase 4 run 3"
    assert sorted(e.id for e in held) == expected["heldout"], "held-out sample differs from Phase 4 run 3"
    if args.limit:  # smoke runs only: a few records per held-out source, so the sample check above still ran in full
        dev = dev[: args.limit]
        per_source: dict = {}
        held = [e for e in held if per_source.setdefault(e.source, []).append(1) or len(per_source[e.source]) <= args.limit]
    wanted = {e.id for e in dev} | {e.id for e in held}
    records = {}
    for name in ("dev", "heldout"):
        for line in (args.data_dir / f"{name}.jsonl").open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["id"] in wanted:
                records[rec["id"]] = rec
    return targs, dev, held, records


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def permuted(rec: dict) -> tuple[dict, list[int]]:
    """The record with its (single) choice question's options in a seeded random order, and that order."""
    out = json.loads(json.dumps(rec))
    key = next(iter(out["request"]["questions"]))
    criteria = out["request"]["questions"][key]["criteria"]
    names = list(criteria)
    order = list(range(len(names)))
    random.Random(f"flip:{rec['id']}").shuffle(order)
    out["request"]["questions"][key]["criteria"] = {names[i]: criteria[names[i]] for i in order}
    out["targets"][key] = [rec["targets"][key][i] for i in order]
    return out, order


def flip_rates(records: list[dict], scores) -> dict:
    """Per source: share of questions whose answer changes when the options are reordered.
    scores(rec) -> one score per option of the record's question, in the record's option order."""
    by_source: dict = {}
    for rec in records:
        if rec["source"] in FLIP_SOURCES and len(by_source.setdefault(rec["source"], [])) < FLIP_QUESTIONS:
            by_source[rec["source"]].append(rec)
    out = {}
    for source, recs in by_source.items():
        flips = 0
        for rec in recs:
            before = max(range(len(s := scores(rec))), key=s.__getitem__)
            shuffled, order = permuted(rec)
            after = max(range(len(s2 := scores(shuffled))), key=s2.__getitem__)
            flips += order[after] != before  # order[i] is the original index of the option now at i
        out[source] = {"n": len(recs), "flip_rate": flips / len(recs)}
    return out


def pointwise_arm(args, targs, dev, held, records, out: Path, report: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(targs, device)
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    assert state["step"] == RUN3_BEST_STEP, f"state.pt is step {state['step']}, not run 3's best step {RUN3_BEST_STEP}"
    T.load_trainable(model, state)
    model.eval()
    for name, examples in (("dev", dev), ("heldout", held)):
        write_rows(out / "arms" / f"pointwise_{name}.jsonl", T.collect_logits(model, examples, targs.token_budget))
    model.eval()  # collect_logits leaves the model in train mode (LoRA dropout on), which would corrupt flips and latency
    probs = lambda rec: model.probs(model.encode(SystemOneRequest.model_validate(rec["request"])))[0]  # noqa: E731
    report["pointwise"] = flip_rates([records[e.id] for e in held], probs)
    report["pointwise_latency"] = pointwise_latency(model)
    del model
    free()


def causal_model(name: str, adapter: Path | None):
    device, dtype = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tokenizer


def causal_arm(arm: str, model_name: str, adapter: Path | None, native_banking: bool, dev, held, records, out: Path, report: dict):
    model, tokenizer = causal_model(model_name, adapter)
    config = load_task_config()
    scorer = LabelScorer(model, tokenizer, config) if native_banking else None
    labels = [label.replace("_", " ") for label in config.labels]

    def option_scores(rec: dict) -> list[float]:
        """One score per option in the record's option order, the same procedure for rows and flips."""
        (key,) = rec["request"]["questions"]
        if scorer is None or rec["source"] != "banking77":
            (row,) = B.score_record(model, tokenizer, rec)
            return row["logits"]
        options = list(rec["request"]["questions"][key]["criteria"])
        (scores,) = scorer.score([rec["request"]["state"]])
        return [scores[labels.index(o)] for o in options]

    def rows_for(rec: dict) -> list[dict]:
        if scorer is None or rec["source"] != "banking77":
            return B.score_record(model, tokenizer, rec)
        (key,) = rec["request"]["questions"]
        return [{"source": rec["source"], "id": rec["id"], "question": 0, "logits": option_scores(rec),
                 "target": rec["targets"][key]}]

    for name, examples in (("dev", dev), ("heldout", held)):
        started = time.time()
        write_rows(out / "arms" / f"{arm}_{name}.jsonl", [r for e in examples for r in rows_for(records[e.id])])
        log(f"{arm} {name}: {len(examples)} records in {time.time() - started:.0f}s")

    report[arm] = flip_rates([records[e.id] for e in held], option_scores)
    if arm == "qwen3_base":
        report["generate_latency"] = generate_latency(model, tokenizer)
    del model, scorer
    free()


# --- latency ---------------------------------------------------------------------------------

def pointwise_latency(model) -> dict:
    from scripts.bench_decide import shared_state_request
    from scripts.bench_packing import build_request, time_ms

    state_ids, _ = build_request(model.tokenizer, 512, 1)
    state = model.tokenizer.decode(state_ids)
    device = model.backbone.device
    out = {}
    for k in LATENCY_KS:
        encoded = model.encode(SystemOneRequest.model_validate(shared_state_request(state, k)))
        out[k] = time_ms(lambda e=encoded: model.probs(e), device)
    return out


def generate_latency(model, tokenizer) -> dict:
    """Generate-and-parse: one greedy generation per question, each re-reading the state."""
    from scripts.bench_decide import shared_state_request
    from scripts.bench_packing import build_request, time_ms

    state_ids, _ = build_request(tokenizer, 512, 1)
    state = tokenizer.decode(state_ids)
    out = {}
    for k in LATENCY_KS:
        request = SystemOneRequest.model_validate(shared_state_request(state, k))
        prompts = [B.question_prompt(request, key)[0] for key in request.questions]

        def run(prompts=prompts) -> None:
            for prompt in prompts:
                ids = tokenizer(prompt, return_tensors="pt").to(model.device)
                model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        out[k] = time_ms(run, model.device)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="run 3's best/state.pt")
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Phase 0 Banking77 LoRA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke run: this many records")
    args = parser.parse_args()
    if args.limit:  # CPU smoke run: time one tiny request once
        global LATENCY_KS
        LATENCY_KS = (1,)
        import scripts.bench_packing as bench_packing

        bench_packing.REPS = 1

    tokenizer = transformers.AutoTokenizer.from_pretrained(T.MODEL_NAME)
    targs, dev, held, records = load_samples(args, tokenizer)
    log(f"dev {len(dev)}, held-out {len(held)} examples (matches Phase 4 run 3)")
    report: dict = {"env": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                            "banking77_adapter": str(args.adapter_dir), "run3_state": str(args.state)}}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stages = [
        ("pointwise", lambda: pointwise_arm(args, targs, dev, held, records, out, report)),
        ("qwen3_base", lambda: causal_arm("qwen3_base", QWEN3, None, False, dev, held, records, out, report)),
        ("qwen25_base", lambda: causal_arm("qwen25_base", QWEN25, None, True, dev, held, records, out, report)),
        ("qwen25_lora", lambda: causal_arm("qwen25_lora", QWEN25, args.adapter_dir, True, dev, held, records, out, report)),
    ]
    for name, stage in stages:  # saved after every stage, so a late failure keeps the earlier arms
        failure = None
        if name == "pointwise":  # the primary arm: a missing or wrong checkpoint must stop the run, not be skipped
            stage()
        else:
            try:
                stage()
            except Exception as e:  # noqa: BLE001
                failure = repr(e)
        if failure:  # outside the except block, so the failed stage's model is no longer referenced by a traceback
            report[f"{name}_error"] = failure
            log(f"{name} FAILED: {failure}")
            free()
        (out / "phase5_report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
