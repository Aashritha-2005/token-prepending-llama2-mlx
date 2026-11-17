# Combination (run 10 = Enhancement 1 + 2): reading (S5.4, 2026-11-05 16:37)

- C4 (primary) run 10 - run 5: +0.23, 95% CI [+0.11, +0.37], Bonferroni [+0.06, +0.40], 4/7 positive -> **no resolvable difference**.
- E9 (exploratory) run10 - run8: +1.97 [+1.74, +2.22], 6/7 positive.
- E10 (exploratory) run10 - r3: +2.32 [+2.03, +2.65], 6/7 positive.
- E11 (exploratory) run10 - run4: +0.89 [+0.70, +1.06], 7/7 positive.
- Reading (plan 14.2): Run 10 ~ run 5: reversal dominates.
- Caveats: not tuned; CP setting from dev (forward view, addendum 1); keep_final chosen after dev; run 4 batch-1 vanilla, run 5 and r3 batched TP, run 8 and run 10 batch 1; CP x reversed view untested before the test pass.

---

## Stage 5 closing notes (AGENT-RUN, 2026-11-05; notes only, nothing recomputed into any state file)

The block above was written by `live_step.py gate-s5.4` at 16:37:00 (sha256 of that version f99c1326a85cb47a24391db5b296ef60f7f1512868856356d19aff63734d7e05). It is kept unchanged; everything below is added.

### Verdict (pre-registered wording, notes/s5_preregistration.md sha256 7823735e…87cb122)

- **C4 (primary) run 10 − run 5: +0.23, 95% CI [+0.11, +0.37], Bonferroni [+0.06, +0.40], positive 4/7 → "no resolvable difference".**
  The 95% interval excludes 0, but the sign rule (≥ 5 of 7 datasets positive) is not met (4/7: STS12, STS-B and SICK-R are negative). Under the pre-registered rule this is not "improves".
- Reading under plan 14.2: **run 10 ≈ run 5, "reversal dominates".** Adding CP to both views does not add a resolvable gain on top of Enhancement 1.

### Exploratory (labelled exploratory; no claims are made from them)

| ID | Comparison | Diff | 95% CI | Bonferroni | Positive |
|---|---|---|---|---|---|
| E9 | run 10 − run 8 | +1.97 | [+1.74, +2.22] | [+1.69, +2.30] | 6/7 |
| E10 | run 10 − r3 | +2.32 | [+2.03, +2.65] | [+1.96, +2.73] | 6/7 |
| E11 | run 10 − run 4 | +0.89 | [+0.70, +1.06] | [+0.65, +1.12] | 7/7 |

(From results.jsonl lines S5.4-E9/E10/E11 and gates.json S5.4.)

### Full per-dataset test table (Spearman × 100, layer 27 unless stated)

Built by a read-only script from results/test/scores_*.json and notes/reference_numbers.json (no number typed). [PAPER] rows are published values, not measured here.

| Run | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg | Δ r3 |
|---|---|---|---|---|---|---|---|---|---|
| PromptEOL [PAPER] | 58.81 | 77.01 | 66.34 | 73.22 | 73.56 | 71.66 | 69.64 | 70.03 | -7.22 |
| r1 PromptEOL, final layer | 58.72 | 77.11 | 66.42 | 73.34 | 73.50 | 71.78 | 69.65 | 70.07 | -7.18 |
| r2 PromptEOL, layer 27 | 64.26 | 82.09 | 72.27 | 78.95 | 77.24 | 76.84 | 72.41 | 74.87 | -2.39 |
| PromptEOL+TP [PAPER] | 66.90 | 83.12 | 74.31 | 79.87 | 80.03 | 80.67 | 75.40 | 77.19 | -0.06 |
| r3 TP e 7 | 66.88 | 83.28 | 74.40 | 80.08 | 79.97 | 80.72 | 75.44 | 77.25 | +0.00 |
| r3' TP e 8 | 67.02 | 82.55 | 73.57 | 79.27 | 79.51 | 80.29 | 75.18 | 76.77 | -0.49 |
| run 4 ReverseEOL (vanilla) | 70.58 | 84.45 | 76.17 | 83.40 | 80.61 | 81.82 | 73.76 | 78.68 | +1.43 |
| run 5 Enh 1 (fwd+rev, TP) | 70.86 | 84.64 | 76.41 | 83.02 | 81.81 | 83.19 | 75.45 | 79.34 | +2.09 |
| run 7 CP (vanilla) | 66.24 | 82.82 | 73.35 | 79.52 | 78.48 | 78.59 | 73.95 | 76.14 | -1.12 |
| run 8 Enh 2 (CP+TP) | 66.44 | 83.94 | 74.75 | 81.39 | 80.27 | 80.51 | 75.91 | 77.60 | +0.35 |
| control 9 (CP+TP, sentiment aux) | 65.42 | 83.76 | 73.83 | 80.83 | 80.22 | 78.01 | 74.63 | 76.67 | -0.58 |
| run 10 Enh 1+2 | 70.69 | 85.30 | 76.92 | 84.04 | 82.26 | 82.98 | 74.81 | 79.57 | +2.32 |

Sources: r1/r2 scores_r1r2_vanilla.json ("final" / "27"); r3 scores_r3_tp_e7.json; r3' scores_r3p_tp_e8.json; run 4 scores_r4_reverseeol.json; run 5 scores_r5_enh1_fwd_rev_tp.json; run 7 scores_r7_cp.json; run 8 scores_r8_cptp.json; control 9 scores_r9_cptp_sent.json; run 10 scores_r10_comb.json (reproduced from pairs_r10_comb.npz to 1e-9 at closing).

### Cost (s/sentence computed, aux pass included; logs/S5.4.json and the pass logs)

| Pass / run | Pass s/sent | Run s/sent (cumulative) | × r3 |
|---|---|---|---|
| r3 TP, forward (batched) | 0.092 | 0.092 | 1.00× |
| run 5 = r3 + reversed TP pass (batched) | 0.092 | 0.184 | 2.00× |
| run 8 CP+TP forward (batch 1) | 0.159 | 0.159 | 1.73× |
| run 10 = run 8 + reversed CP+TP pass (batch 1) | 0.166 | 0.325 | 3.54× |

Per unit of cost (point estimates over r3, average Spearman × 100): Enhancement 1 buys +2.09 for one extra r3-sized pass (+1.00× r3); Enhancement 2 buys +0.35 for +0.73× r3; the combination buys +2.32 for +2.54× r3, i.e. +0.23 over run 5 (not resolvable under C4) for a further +1.54× r3. Enhancement 1 is by far the best value; adding CP on top of it costs most of the extra time and returns no resolvable gain. (S3.7 logged 2.045× for run 5 from its own pass timings; the 2.00× here uses the rounded per-pass figures that gate-s5.4 reads.)

### Caveats

- The combination was not tuned. Every setting is inherited from frozen enh1 and enh2 (comb entry sha256 7cb2341e…32a2, S5.1).
- The CP setting (index 4 = paper ℓ 5, α 4, aux with TP) was selected on dev for the **forward view only**, as the best of 17 cells after one extension of the grid (addendum 1, sha256 8ceb9443…6063d7). Its use on the reversed view was not tuned or checked on dev beyond the S5.2 sanity gate.
- Punctuation keep_final (enh1) was chosen after seeing dev results (the `full` option failed the rev-only check; logged deviation).
- Batching differs between runs: run 4 is batch-1 vanilla; run 5 and r3 are batched TP; run 8 and run 10 are batch 1. C4 compares batch-1 CP+TP states with batched TP states.
- The bootstrap intervals cover test-pair sampling only (B = 2000, seed 0). They do not cover seed, prompt, quantisation or selection variance.
- 8-bit group-64 MLX model with fp16 activations, not the fp16/bf16 weights of the papers.
- CP's published gain is against the last-layer PromptEOL baseline; here every CP run is at layer 27, so run 7's gain over r2 and CP's published number are not the same comparison.

### What can and cannot be claimed (plan 13.3)

Can claim:
- TP's gain was reproduced on an 8-bit Llama-2-7B on a laptop (r3 77.25 vs 77.19 [PAPER]); the layer-matched baseline r2 (74.87) separates early exit (+4.79 over r1) from TP's own gain (+2.39, C1, robust).
- **Enhancement 1** (ReverseEOL + TP, run 5 79.34): improves over r3, robust (C2 +2.09, 7/7). The shuffled-word control 6a recovers 1.51 of the 2.09 (exploratory), so "a second view helps" is better supported than "reversal specifically helps".
- **Enhancement 2** (CP + TP, run 8 77.60): improves over r3, robust (C3 +0.35, 5/7). Small effect; below r3 on STS12 and STS-B; selected after one grid extension.
- **Combination** (run 10 79.57): no resolvable difference from Enhancement 1 alone (C4); reading "reversal dominates".

Do not claim:
- That the combination improves over Enhancement 1 (C4 sign rule not met), or that CP and reversal are complementary.
- That Enhancement 2 adds anything on top of Enhancement 1 (E8 run 8 − run 5 −1.74 and C4 both say no).
- Anything from E9–E11 (exploratory), including that run 10 beats run 4 or run 8.
- That ReverseEOL's 78.69 applies to TP, or that CP's published 75.27 is comparable to the layer-27 runs.
- That L5 (tuning cost) is solved: every enhancement needed dev selection.
- Anything about other backbones, prompts, tasks or precisions.
