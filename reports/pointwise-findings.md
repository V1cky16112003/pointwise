## Findings

Comparisons are on 500 held-out questions per source (pooled out-of-domain, "OOD" = 2,000). A gap is called distinguishable from noise only when its paired-bootstrap 95% interval excludes 0. 36 per-source and pooled intervals were computed without any multiple-comparison correction, so a result whose interval barely excludes 0 (marked "borderline") should be read as weak. Calibration differences (ECE) have no interval in this report, so they are point estimates only.

### Where Pointwise wins
- **Against its own no-training base (Qwen3-1.7B-Base):** pooled OOD accuracy 0.678 vs 0.640, gap +3.8 points, CI [+1.8, +5.8], distinguishable from noise. Banking77 (zero-shot for Pointwise) +29.8 [+25.2, +34.6] and TweetEval +20.6 [+15.0, +26.0] are large; QNLI (+2.8) and PAWS (-0.4) are noise.
- **Pooled OOD calibration after T (mixed, point estimates only):** 0.056 vs 0.109 for the Qwen3 base and 0.081 for the LoRA specialist, but Qwen2.5-Instruct is 0.052. Raw pooled ECE barely differs (0.129 vs 0.134 and 0.146). Pooling hides per-source errors that cancel: Pointwise's after-T ECE is worse than the Qwen3 base on Banking77 (0.113 vs 0.046) and PAWS (0.130 vs 0.080), and worse than Instruct on QNLI, PAWS and Emotion. The clear per-source gain is TweetEval (0.038 vs 0.288 for the Qwen3 base).
- **Option-order flip rate:** Pointwise 0.210 / 0.075 / 0.045 on Banking77 / Emotion / TweetEval, against 0.615 / 0.190 / 0.070 for the Qwen3 base. The no-training baseline is far more sensitive to option order on Banking77.
- **Latency on a T4 (one ~512-token state, k questions):** one pass costs 196 ms for k=1 and 261 ms for k=8. Generate-and-parse with the Qwen3 base costs 223 ms and 1,890 ms, about 7x slower at k=8. At k=1 they are close (196 vs 223 ms).
- **Against the no-training Qwen2.5-1.5B-Instruct on Banking77 and TweetEval:** +19.0 [+14.0, +24.2] and +5.8 [+0.8, +10.6] (TweetEval borderline).

### Where Pointwise loses
- **Emotion, against its own no-training base:** -8.0 points, CI [-11.8, -4.2]. The untrained Qwen3 base is more accurate here. This loss is not hidden.
- **Against the no-training Qwen2.5-1.5B-Instruct (a different base model):** pooled OOD -2.0 [-4.0, -0.1] (borderline), PAWS -5.8 [-9.6, -2.0], Emotion -4.4 [-8.0, -1.0]. QNLI -3.6 [-7.2, 0.0] is not distinguishable.
- **Selective prediction:** Cov@5% (share of questions answerable at <=5% error) on pooled OOD is 0.004 for Pointwise, essentially nothing, against 0.171 for Qwen2.5-Instruct. Pointwise's after-T ECE (0.056) is about the same as Qwen2.5-Instruct's (0.052), but Instruct ranks its confident answers much better (AURC 0.148 vs 0.209). Cov@5% is near zero for three of four arms on pooled OOD (Pointwise 0.004, Qwen3 base 0.004, LoRA 0.011; about 8 of 2,000 questions), so only Instruct stands out; Pointwise's AURC beats the Qwen3 base's (0.209 vs 0.246).
- **The Phase 4 gate (pooled OOD ECE after T <= 0.05) was missed (0.056) and is still missed.** This report has no numeric gate.
- **Banking77 against the specialist:** the Banking77 LoRA gets 0.932 against Pointwise's 0.626, gap -30.6 [-34.8, -26.0]. That is expected: Banking77 is in-domain for the LoRA and zero-shot for Pointwise.

### Not distinguishable from noise
- Pointwise vs the Banking77 LoRA on pooled OOD: -0.2 [-2.1, +1.7]; per-source QNLI, PAWS, Emotion and TweetEval all include 0 (Emotion -3.6 [-7.2, 0.0] only just).
- Pointwise vs Qwen3 base on QNLI and PAWS.

### Fine-tune vs its own base
- Pointwise vs Qwen3-1.7B-Base: win on pooled OOD (above).
- Banking77 LoRA vs Qwen2.5-1.5B-Instruct: Banking77 +49.6 [+44.8, +54.4], but pooled OOD -1.8 [-3.4, -0.2] (borderline): the specialist lost accuracy on QNLI (-3.2 [-5.8, -0.4]) and PAWS (-5.4 [-9.2, -1.8]), both distinguishable from noise, offset in the pool by Emotion and TweetEval.

### Caveats
- **Flip rate for the Qwen2.5 arms on Banking77 (0.000) is not comparable.** That path scores the fixed label names from the message alone and never shows the option list, so reordering options cannot change it. It says nothing about robustness.
- **Pointwise and the Qwen2.5 arms use different base models,** so those gaps mix "method" with "base model". The own-base comparisons are the cleaner ones.
- **Latency:** Pointwise ran with the LoRA unmerged. Generate-and-parse is capped at 8 new tokens and the base model may not stop early. One state of about 512 tokens, median of 5 runs, one T4.
- **Reproduction check passed:** Pointwise logits match Phase 4 run 3's saved held-out predictions (largest difference 0.0000, same top option on all 2,500 questions, T 1.516, pooled OOD ECE 0.0561). The LoRA arm's Banking77 accuracy is 0.932, near the Phase 0 read-out's 0.918, so the scoring path is sound.
- **Kev-0.8B / Kev-4B were not run.** Kev's model card lists Banking77 among its training sets, so Banking77 would be in-domain for Kev, and it needs a separate Qwen3.5 Gated-DeltaNet environment that was not set up. QNLI, PAWS and Emotion would have been fair comparisons.
- Rerun note: this findings text is hand-written for the version-17 numbers; regenerate it if the arms change.
- Dev questions of type score (policies, SST-5, Yelp) are answered only by Pointwise, so the baselines' T is fitted on 1,844 of Pointwise's 2,230 dev questions.

## Protocol
