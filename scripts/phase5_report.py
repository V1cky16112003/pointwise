"""Phase 5 report: arms' dev/held-out rows -> reports/pointwise-results.md and plots.

Reads <input>/arms/<arm>_{dev,heldout}.jsonl and <input>/phase5_report.json (written by scripts.phase5_eval).
Per arm, T is fitted on that arm's own dev rows, then held-out questions are scored with
src.pointwise.train.score_rows, so every arm gets the same protocol as Pointwise. Accuracy gaps use a paired
bootstrap over held-out records (resampled inside each source), for every pair of arms present.

Usage (from the repo root):
    python -m scripts.phase5_report --input artifacts/phase5 --output-dir reports
"""
from __future__ import annotations

import argparse
import itertools
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
GROUPS = SOURCES + ["pooled OOD"]  # pooled OOD = every source except banking77
ARM_ORDER = ["pointwise", "qwen3_base", "qwen25_lora", "qwen25_base", "kev"]
LABEL = {"pointwise": "Pointwise (run 3)", "qwen3_base": "Qwen3-1.7B-Base, no training",
         "qwen25_lora": "Qwen2.5-1.5B + Banking77 LoRA", "qwen25_base": "Qwen2.5-1.5B-Instruct, no training",
         "kev": "Kev"}


def in_group(source: str, group: str) -> bool:
    return source != "banking77" if group == "pooled OOD" else source == group


def load_arms(path: Path) -> dict:
    arms = {}
    for arm in ARM_ORDER:
        dev, held = path / "arms" / f"{arm}_dev.jsonl", path / "arms" / f"{arm}_heldout.jsonl"
        if dev.exists() and held.exists():
            arms[arm] = {"dev": C.read_rows(dev), "heldout": C.read_rows(held)}
    return arms


def arm_stats(rows: dict, temperature: float) -> dict:
    held = rows["heldout"]
    scaled = [M.question_scores(r["logits"], r["target"], temperature) for r in held]
    raw = [M.question_scores(r["logits"], r["target"]) for r in held]
    groups = {}
    for g in GROUPS:
        keep = [i for i, r in enumerate(held) if in_group(r["source"], g)]
        conf = [scaled[i]["confidence"] for i in keep]
        ok = [scaled[i]["correct"] for i in keep]
        groups[g] = {"conf": conf, "raw_conf": [raw[i]["confidence"] for i in keep], "ok": ok,
                     "aurc": aurc(conf, ok), "cov5": coverage_at_risk(conf, ok, 0.05),
                     "summary": M.summarise([scaled[i] for i in keep]), "raw_summary": M.summarise([raw[i] for i in keep])}
    return {"T": temperature, "by_source": score_rows(held, temperature), "raw_by_source": score_rows(held, 1.0), "groups": groups}


def gap_table(arms: dict, a: str, b: str) -> list[str]:
    rows_a, rows_b = C.align({a: arms[a]["heldout"], b: arms[b]["heldout"]}).values()
    out = []
    for group in GROUPS:
        keep = [i for i, r in enumerate(rows_a) if in_group(r["source"], group)]
        ca = [M.question_scores(rows_a[i]["logits"], rows_a[i]["target"])["correct"] for i in keep]
        cb = [M.question_scores(rows_b[i]["logits"], rows_b[i]["target"])["correct"] for i in keep]
        clusters = [(rows_a[i]["source"], rows_a[i]["id"]) for i in keep]
        point, lo, hi = C.paired_bootstrap(ca, cb, clusters)
        verdict = "unclear" if lo <= 0 <= hi else (f"{LABEL[a]} wins" if point > 0 else f"{LABEL[b]} wins")
        out.append(f"| {group} | {len(keep)} | {100 * point:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {verdict} |")
    return out


def plot_reliability(stats: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(GROUPS), len(stats), figsize=(2.8 * len(stats), 2.4 * len(GROUPS)), squeeze=False, sharex=True, sharey=True)
    for row, group in enumerate(GROUPS):
        for col, (arm, s) in enumerate(stats.items()):
            ax, g = axes[row][col], s["groups"][group]
            for label, conf, style in (("raw", g["raw_conf"], "--"), ("after T", g["conf"], "-")):
                pts = C.reliability(conf, g["ok"])
                ax.plot([p["confidence"] for p in pts], [p["accuracy"] for p in pts], style, marker="o", ms=3, label=label)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
            if row == 0:
                ax.set_title(LABEL[arm], fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{group}\naccuracy", fontsize=7)
            if row == len(GROUPS) - 1:
                ax.set_xlabel("confidence", fontsize=7)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_risk_coverage(stats: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
    for ax, group in zip(axes.flat, GROUPS):
        for arm, s in stats.items():
            cov, risk = zip(*risk_coverage_curve(s["groups"][group]["conf"], s["groups"][group]["ok"]))
            ax.plot(cov, risk, label=LABEL[arm])
        ax.axhline(0.05, color="grey", lw=0.8, ls=":")
        ax.set_title(group, fontsize=9)
        ax.set_xlabel("coverage (most confident first)", fontsize=7)
        ax.set_ylabel("error rate among answered", fontsize=7)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def reproduction_check(arms: dict, stats: dict, reference: Path, reference_metrics: Path) -> list[str]:
    """Pointwise rows from this run against Phase 4 run 3's own saved predictions: the same weights and
    questions should give the same logits (up to fp16 noise) and a dev-fitted T close to run 3's."""
    if "pointwise" not in arms or not reference.exists():
        return []
    new, old = C.align({"new": arms["pointwise"]["heldout"], "old": C.read_rows(reference)}).values()
    diff = max(abs(x - y) for a, b in zip(new, old) for x, y in zip(a["logits"], b["logits"]))
    agree = sum(max(range(len(a["logits"])), key=a["logits"].__getitem__) == max(range(len(b["logits"])), key=b["logits"].__getitem__)
                for a, b in zip(new, old)) / len(new)
    out = ["", "## Reproduction check: Pointwise against Phase 4 run 3", "",
           f"- Largest difference in any held-out logit: {diff:.4f}; same top option on {100 * agree:.1f}% of {len(new)} questions.",
           f"- T refitted here on the rebuilt dev rows: {stats['pointwise']['T']:.3f}."]
    if reference_metrics.exists():
        saved = json.loads(reference_metrics.read_text())
        ood = [M.question_scores(r["logits"], r["target"], saved["temperature"]) for r in new if r["source"] != "banking77"]
        out += [f"- Run 3's saved T: {saved['temperature']:.3f}. Pooled OOD ECE with this run's logits and that T: "
                f"{M.summarise(ood)['ece']:.4f} (run 3 logged {saved['gate']['heldout_ood_ece_scaled']:.4f})."]
    return out + ["- If these differ a lot, the weights or the question sample are not run 3's and every arm's dev-fitted T is suspect.", ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--run3-dir", type=Path, default=Path("artifacts/run3"), help="run 3's heldout_predictions.jsonl and metrics.json")
    args = parser.parse_args()

    arms = load_arms(args.input)
    aligned = C.align({a: v["heldout"] for a, v in arms.items()})
    for a in arms:
        arms[a]["heldout"] = aligned[a]
    stats = {arm: arm_stats(rows, C.fit_arm_temperature(rows["dev"])) for arm, rows in arms.items()}
    run = json.loads((args.input / "phase5_report.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(stats, args.output_dir / "reliability.png")
    plot_risk_coverage(stats, args.output_dir / "risk_coverage.png")

    n_dev = len(next(iter(arms.values()))["dev"])
    md = ["# Pointwise Phase 5 results", "",
          "All arms score the same 2,500 held-out questions (500 per source) that Phase 4 run 3 used. Temperature T is fitted "
          f"per arm on that arm's own dev rows ({n_dev:,} questions from 1,500 dev records), never on held-out. Banking77 is zero-shot "
          "for Pointwise and Qwen3; it is in-domain for the Banking77 LoRA arm and for Kev (Kev's model card lists Banking77 among "
          "its training sets). Sanity check: the LoRA arm's Banking77 accuracy should be near the Phase 0 read-out's 0.918; if it is "
          "far lower, the scoring path is wrong and nothing below should be trusted.", "",
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
          "| Arm | T | " + " | ".join(SOURCES) + " | pooled OOD acc | pooled OOD ECE raw | pooled OOD ECE after T |",
          "|---|---|" + "---|" * (len(SOURCES) + 3)]
    for arm, s in stats.items():
        cells = [f"{s['by_source'][src]['accuracy']:.3f} / {s['raw_by_source'][src]['ece']:.3f} / {s['by_source'][src]['ece']:.3f}" for src in SOURCES]
        pooled = s["groups"]["pooled OOD"]
        md.append(f"| {LABEL[arm]} | {s['T']:.2f} | " + " | ".join(cells) +
                  f" | {pooled['summary']['accuracy']:.3f} | {pooled['raw_summary']['ece']:.3f} | {pooled['summary']['ece']:.3f} |")
    md += ["", "## Selective prediction after T: Cov@5% / AURC", "",
           "| Arm | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    for arm, s in stats.items():
        md.append(f"| {LABEL[arm]} | " + " | ".join(f"{s['groups'][g]['cov5']:.3f} / {s['groups'][g]['aurc']:.3f}" for g in GROUPS) + " |")
    md += ["", "![Reliability](reliability.png)", "", "![Risk-coverage](risk_coverage.png)",
           *reproduction_check(arms, stats, args.run3_dir / "heldout_predictions.jsonl", args.run3_dir / "metrics.json"),
           "", "## Accuracy gaps with paired bootstrap 95% intervals (percentage points, 2,000 resamples)", ""]
    for a, b in itertools.combinations(arms, 2):
        md += [f"### {LABEL[a]} minus {LABEL[b]}", "", "| Source | n | gap | 95% CI | reading |", "|---|---|---|---|---|",
               *gap_table(arms, a, b), ""]
    md += ["## Option-order flip rate", "", "Share of questions whose answer changes when the options are shuffled "
           "(up to 200 questions per source; QNLI and PAWS have a fixed yes/no order). The Qwen2.5 arms read Banking77 through "
           "the specialist's own prompt for these too.", "",
           "| Arm | " + " | ".join(["banking77", "emotion", "tweet_sentiment"]) + " |", "|---|---|---|---|"]
    for arm in stats:
        flips = run.get(arm, {})
        md.append(f"| {LABEL[arm]} | " + " | ".join(f"{flips[s]['flip_rate']:.3f}" if s in flips else "n/a" for s in ["banking77", "emotion", "tweet_sentiment"]) + " |")
    errors = {k: v for k, v in run.items() if k.endswith("_error")}
    if errors:
        md += ["", "## Stages that failed", "", "Arms or tables missing above did not run:", ""] + [f"- `{k}`: {v}" for k, v in errors.items()]
    if "pointwise_latency" in run and "generate_latency" in run:
        md += ["", "## Latency on one state with k questions (ms, median)", "",
               f"GPU: {run['env']['gpu']}. fp16, one state of about 512 tokens, k three-option questions. Pointwise runs with the LoRA unmerged "
               "(a merged adapter would be a little faster); generate-and-parse is the no-training Qwen3-1.7B-Base reading the same state again for each question. "
               "Generation is capped at 8 new tokens and a base model may not stop early, so it can be slightly slow. Median of 5 runs after one warm-up.", "",
               "| k | Pointwise (one pass) | Generate-and-parse (one greedy generation per question) |", "|---|---|---|"]
        for k, ms in run["pointwise_latency"].items():
            md.append(f"| {k} | {ms:.0f} | {run['generate_latency'][k]:.0f} |")
    (args.output_dir / "pointwise-results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
