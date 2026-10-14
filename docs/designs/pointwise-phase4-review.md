# Pointwise Phase 4 — review against Kev and Jev (2026-10-14)

Why: runs 1 and 2 both failed the gate (held-out OOD ECE 0.074, 0.080 > 0.05).
Question: is the training process wrong, or is the gate out of reach?

Sources read: Kev `kev/train.py`, `README.md`, `PLAN.md` and the Kev-0.8B model card
(github.com/jaredpalmer/kev, main, read 2026-10-14); Archer Hume, "Jev's Architecture
Unmasked" (2026-09-17). Kev's numbers are self-reported.

## Verdict

**The architecture and loss are on track; the data recipe is not.** The gate is
reachable: Kev-0.8B (smaller than our 1.7B) scores ECE 0.049 dev / 0.045 locked test
on transfer-v4 (QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion + held-out policies) at
accuracy 0.648 / 0.697. Ours: 0.080 at 0.674. Same accuracy, worse calibration.

## What matches Kev

| | Pointwise | Kev-0.8B base recipe |
|---|---|---|
| Backbone | Qwen3-1.7B-Base, frozen | Qwen3.5-0.8B-Base, frozen |
| Adapter | LoRA r16, attention + MLP | LoRA r16, α 32, attention + MLP (+ DeltaNet) |
| Read-out | pointer head over option tokens | pointer head (`</opt>` vs `<decide>`) |
| Loss | cross-entropy (`--brier-weight 0`) | cross-entropy |
| Epochs | 2 | 2 |
| Calibration | one T fitted on in-domain dev | one T fitted on in-domain dev (T 2.35) |

## Where we differ (most likely causes, in order)

1. **Data volume and balance.** We train on 152,507 records / 239k questions, skewed
   (policies 52k questions, MNLI 39k, AG News 38k). Kev trains on 12,576 records:
   1,000 per public source + 896 policy pairs + 1,680 rule structures. Same 2 epochs,
   so we take ~12× more steps on a few dominant sources.
2. **No option augmentation.** We only shuffle options. Kev adds a "none of these"
   option (p 0.10), distractor options (p 0.15), none + distractor (p 0.12) and
   none-present / none-absent minimal pairs (p 0.25). These teach the model what
   missing evidence looks like, which is what over-confident errors lack.
3. **Learning rate and regularisation.** Ours: LoRA lr 2e-4, head lr 1e-3, weight
   decay 0. Kev-0.8B: lr 1e-4 for both, weight decay 0.01, OneCycle with 10% warm-up.

## Diagnostics on run 2's held-out predictions (offline, no GPU)

- **Miscalibration points in opposite directions**, so no single temperature fixes it.
  Mean confidence − accuracy after T = 1.32: PAWS +0.223, Emotion +0.089 (over);
  QNLI −0.138, TweetEval −0.055 (under). The best pooled OOD ECE over any T in
  [0.5, 3.0] is 0.080 at T = 1.30, which is exactly what we got. Temperature is
  already optimal; the problem is in the trained model.
- **Fitting T on held-out datasets makes it worse**, not better: leave-one-source-out
  T gives pooled ECE 0.113. Kev's finding that a held-out-datasets T helps (lesson 5)
  does not apply when sources disagree in direction.
- **PAWS is a shortcut, not noise.** Of 299 non-paraphrase pairs, 169 are called
  paraphrases; only 4 of 201 true paraphrases are missed. PAWS pairs share almost all
  their words, so this is the lexical-overlap → "same meaning" heuristic that heavy
  MNLI training is known to teach (McCoy et al. 2019, HANS). We have 30k MNLI records;
  Kev has 1k.

## Correction to the run-2 log

The proposed next change there, "train with a larger Brier weight", is withdrawn.
Kev tried label smoothing, CE + Brier and focal loss against plain CE and recorded
all three as negative results (`PLAN.md`, "Negative results worth remembering").

## Proposed run 3 (not started; needs approval)

One change set, matching Kev's base recipe, on the same 2 epochs and the same gate:
- cap each public source at ~1,000–2,000 records (keep policies and rules);
- add Kev-style option augmentation (none option, distractors, none pairs);
- LoRA and head lr 1e-4, weight decay 0.01.

The smaller data set also cuts the run from ~5.3 to well under 1 GPU-hour, so a
second seed fits the budget. Success = pooled OOD ECE ≤ 0.05 with no source's
accuracy below run 2's by more than 3 points.
