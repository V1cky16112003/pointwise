"""Phase 5 report: arms' dev/held-out rows -> reports/pointwise-results.md and plots.

Reads <input>/arms/<arm>_{dev,heldout}.jsonl and <input>/phase5_report.json (written by scripts.phase5_eval).
Per arm, T is fitted on that arm's own dev rows, then held-out questions are scored with
src.pointwise.train.score_rows, so every arm gets the same protocol as Pointwise. Accuracy gaps use a paired
bootstrap over held-out records (resampled inside each source).

Usage (from the repo root):
    python -m scripts.phase5_report --input artifacts/phase5 --output-dir reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluate_core import aurc, coverage_at_risk, risk_coverage_curve  # noqa: E402
from src.pointwise import compare as C  # noqa: E402
from src.pointwise import metrics as M  # noqa: E402
from src.pointwise.train import score_rows  # noqa: E402

SOURCES = ["banking77", "qnli", "paws", "emotion", "tweet_sentiment"]
OOD = [s for s in SOURCES if s != "banking77"]
ARM_ORDER = ["pointwise", "qwen3_base", "qwen25_lora", "qwen25_base", "kev"]
LABEL = {"pointwise": "Pointwise (run 3)", "qwen3_base": "Qwen3-1.7B-Base, no training",
         "qwen25_lora": "Qwen2.5-1.5B + Banking77 LoRA", "qwen25_base": "Qwen2.5-1.5B-Instruct, no training",
         "kev": "Kev"}
GAPS = [("pointwise", "qwen3_base"), ("pointwise", "qwen25_lora"), ("qwen25_lora", "qwen25_base"), ("pointwise", "kev")]


def load_arms(path: Path) -> dict:
    arms = {}
    for arm in ARM_ORDER:
        dev, held = path / "arms" / f"{arm}_dev.jsonl", path / "arms" / f"{arm}_heldout.jsonl"
        if dev.exists() and held.exists():
            arms[arm] = {"dev": C.read_rows(dev), "heldout": C.read_rows(held)}
    return arms


def arm_stats(rows: dict, temperature: float) -> dict:
    scored = [M.question_scores(r["logits"], r["target"], temperature) for r in rows["heldout"]]
    ood = [s for s, r in zip(scored, rows["heldout"]) if r["source"] != "banking77"]
    raw_ood = [M.question_scores(r["logits"], r["target"]) for r in rows["heldout"] if r["source"] != "banking77"]
    conf = [s["confidence"] for s in ood]
    ok = [s["correct"] for s in ood]
    return {"T": temperature, "by_source": score_rows(rows["heldout"], temperature),
            "raw_by_source": score_rows(rows["heldout"], 1.0),
            "pooled_ood": M.summarise(ood), "pooled_ood_raw": M.summarise(raw_ood),
            "aurc": aurc(conf, ok), "cov5": coverage_at_risk(conf, ok, 0.05),
            "confidence": conf, "correct": ok, "scored": scored}


def gap_table(arms: dict, a: str, b: str) -> list[str]:
    rows_a, rows_b = C.align({a: arms[a]["heldout"], b: arms[b]["heldout"]}).values()
    out = []
    for source in SOURCES + ["all out-of-domain"]:
        keep = [i for i, r in enumerate(rows_a) if (r["source"] != "banking77" if source.startswith("all") else r["source"] == source)]
        ca = [M.question_scores(rows_a[i]["logits"], rows_a[i]["target"])["correct"] for i in keep]
        cb = [M.question_scores(rows_b[i]["logits"], rows_b[i]["target"])["correct"] for i in keep]
        clusters = [(rows_a[i]["source"], rows_a[i]["id"]) for i in keep]
        point, lo, hi = C.paired_bootstrap(ca, cb, clusters)
        verdict = "unclear" if lo <= 0 <= hi else (f"{LABEL[a]} wins" if point > 0 else f"{LABEL[b]} wins")
        out.append(f"| {source} | {len(keep)} | {100 * point:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {verdict} |")
    return out


def plot_reliability(stats: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(stats), figsize=(3.6 * len(stats), 3.6), squeeze=False, sharey=True)
    for ax, (arm, s) in zip(axes[0], stats.items()):
        for label, conf, style in (("raw", s["raw_conf"], "--"), ("after T", s["confidence"], "-")):
            pts = C.reliability(conf, s["correct"])  # accuracy does not depend on T, only the bins do
            ax.plot([p["confidence"] for p in pts], [p["accuracy"] for p in pts], style, marker="o", label=label)
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.set_title(LABEL[arm], fontsize=8)
        ax.set_xlabel("confidence")
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_risk_coverage(stats: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for arm, s in stats.items():
        cov, risk = zip(*risk_coverage_curve(s["confidence"], s["correct"]))
        ax.plot(cov, risk, label=LABEL[arm])
    ax.axhline(0.05, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("coverage (share answered, most confident first)")
    ax.set_ylabel("error rate among answered")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    arms = load_arms(args.input)
    aligned = C.align({a: v["heldout"] for a, v in arms.items()})
    for a in arms:
        arms[a]["heldout"] = aligned[a]
    stats = {}
    for arm, rows in arms.items():
        s = arm_stats(rows, C.fit_arm_temperature(rows["dev"]))
        s["raw_conf"] = [M.question_scores(r["logits"], r["target"])["confidence"] for r in rows["heldout"] if r["source"] != "banking77"]
        stats[arm] = s
    run = json.loads((args.input / "phase5_report.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(stats, args.output_dir / "reliability.png")
    plot_risk_coverage(stats, args.output_dir / "risk_coverage.png")

    md = ["# Pointwise Phase 5 results", "",
          "All arms score the same 2,500 held-out questions (500 per source) that Phase 4 run 3 used. Temperature T is fitted "
          "per arm on that arm's own 1,500 dev questions, never on held-out. Banking77 is zero-shot for Pointwise and Qwen3; "
          "it is in-domain for the Banking77 LoRA arm and for Kev (Kev's model card lists Banking77 among its training sets). "
          "Sanity check: the LoRA arm's Banking77 accuracy should be near the Phase 0 read-out's 0.918; if it is far lower, "
          "the scoring path is wrong and nothing below should be trusted.", "",
          "## How to read this", "",
          "- **Accuracy** counts a question right when the highest-probability option is the gold one.",
          "- **Calibration** asks whether confidence matches accuracy: of questions answered with ~80% confidence, ~80% should be right. "
          "**ECE** is the average gap, weighted by how many questions fall in each confidence bin (0 is perfect).",
          "- **Temperature T** divides every logit by one number fitted on dev. It softens or sharpens confidence without ever changing "
          "which option wins, so accuracy is untouched.",
          "- **Reliability diagram:** one point per confidence bin, accuracy against confidence. On the diagonal is calibrated; below it "
          "is over-confident.",
          "- **Risk-coverage curve:** answer only the most confident questions first. Coverage is the share answered and risk is the error "
          "rate among them. **Cov@5%** is the largest coverage whose error rate stays within 5%. **AURC** is the area under the curve (lower is better).",
          "- **Paired bootstrap:** the same held-out questions are resampled with replacement 2,000 times and the accuracy gap recomputed "
          "each time. If the 95% interval excludes 0, the gap is unlikely to be sampling noise. 500 questions per source is small, "
          "so intervals are wide and a gap of a few points will often read 'unclear'.", "",
          "## Accuracy and calibration per source (accuracy / raw ECE / ECE after T)", "",
          "| Arm | T | " + " | ".join(SOURCES) + " | pooled OOD acc | pooled OOD ECE raw | pooled OOD ECE after T | AURC | Cov@5% |",
          "|---|---|" + "---|" * (len(SOURCES) + 5)]
    for arm, s in stats.items():
        cells = [f"{s['by_source'][src]['accuracy']:.3f} / {s['raw_by_source'][src]['ece']:.3f} / {s['by_source'][src]['ece']:.3f}" for src in SOURCES]
        p = s["pooled_ood"]
        md.append(f"| {LABEL[arm]} | {s['T']:.2f} | " + " | ".join(cells) +
                  f" | {p['accuracy']:.3f} | {s['pooled_ood_raw']['ece']:.3f} | {p['ece']:.3f} | {s['aurc']:.3f} | {s['cov5']:.3f} |")
    md += ["", "![Reliability](reliability.png)", "", "![Risk-coverage](risk_coverage.png)", "",
           "## Accuracy gaps with paired bootstrap 95% intervals (percentage points, 2,000 resamples)", ""]
    for a, b in GAPS:
        if a in arms and b in arms:
            md += [f"### {LABEL[a]} minus {LABEL[b]}", "", "| Source | n | gap | 95% CI | reading |", "|---|---|---|---|---|",
                   *gap_table(arms, a, b), ""]
    md += ["## Option-order flip rate", "", "Share of questions whose answer changes when the options are shuffled "
           "(200 questions per source; QNLI and PAWS have a fixed yes/no order).", "",
           "| Arm | " + " | ".join(["banking77", "emotion", "tweet_sentiment"]) + " |", "|---|---|---|---|"]
    for arm in stats:
        flips = run.get(arm, {})
        md.append(f"| {LABEL[arm]} | " + " | ".join(f"{flips[s]['flip_rate']:.3f}" if s in flips else "n/a" for s in ["banking77", "emotion", "tweet_sentiment"]) + " |")
    errors = {k: v for k, v in run.items() if k.endswith("_error")}
    if errors:
        md += ["", "## Stages that failed", "", "Arms or tables missing above did not run:", ""] + [f"- `{k}`: {v}" for k, v in errors.items()]
    if "pointwise_latency" in run and "generate_latency" in run:
        md += ["", "## Latency on one state with k questions (ms, median)", "", f"GPU: {run['env']['gpu']}. fp16, one state of about 512 tokens, k three-option questions. Pointwise runs with the LoRA unmerged "
               "(a merged adapter would be a little faster); generate-and-parse is the no-training Qwen3-1.7B-Base reading the same state again for each question. "
               "Generation is capped at 8 new tokens and a base model may not stop early, so it can be slightly slow. Median of 5 runs after one warm-up.", "",
               "| k | Pointwise (one pass) | Generate-and-parse (Qwen3-1.7B-Base, one greedy generation per question) |", "|---|---|---|"]
        for k, ms in run["pointwise_latency"].items():
            md.append(f"| {k} | {ms:.0f} | {run['generate_latency'][k]:.0f} |")
    (args.output_dir / "pointwise-results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
