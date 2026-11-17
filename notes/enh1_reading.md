# S3.5 Enhancement 1 (ReverseEOL + TP) — statistics and reading

Numbers: results/test/S3_enh1.json, logs/S3.4.log (scripts/s3_4_enh1_test.py). LLaMA2-7B 8-bit group-64 MLX, layer 27.
Frozen 'enh1' (hash checked against S3.3): fwd + rev, α 0.6 on fwd, normalisation on, keep_final, TP s=1 e=7, batched, <PST> row 3407.
**keep_final was chosen after seeing the dev results** (deviation logged 2026-11-04: under `full`, rev-only failed the S3.2 collapse
check). α = 0.6 is the top of the plan's grid. Paths: run 5, 6a, 6b, 6c and r3 are batched TP; run 4 and r2 are batch-1 vanilla.

## Test scores (Spearman x100)

| Run | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg |
|---|---|---|---|---|---|---|---|---|
| Paper PromptEOL+TP | 66.90 | 83.12 | 74.31 | 79.87 | 80.03 | 80.67 | 75.40 | 77.19 |
| r2 vanilla L27 | 64.26 | 82.09 | 72.27 | 78.95 | 77.24 | 76.84 | 72.41 | 74.87 |
| r3 TP e=7 | 66.88 | 83.28 | 74.40 | 80.08 | 79.97 | 80.72 | 75.44 | 77.25 |
| run 4 ReverseEOL (vanilla) | 70.58 | 84.45 | 76.17 | 83.40 | 80.61 | 81.82 | 73.76 | 78.68 |
| **run 5** fwd+rev TP | 70.86 | 84.64 | 76.41 | 83.02 | 81.81 | 83.19 | 75.45 | **79.34** |
| 6a fwd+shuf TP | 69.92 | 84.56 | 76.33 | 82.42 | 80.97 | 81.76 | 75.36 | 78.76 |
| 6b fwd+fwd_revprompt TP | 66.56 | 83.00 | 74.30 | 79.59 | 80.19 | 81.22 | 75.54 | 77.20 |
| 6c rev only TP | 64.80 | 72.01 | 66.52 | 76.47 | 74.32 | 75.66 | 63.28 | 70.44 |

S3.4 sanity: 6b − r3 = −0.05 (within ±1.0).

## Bootstrap (B = 2000, seed 0, per-dataset resampling, STS12-16 pooled)

| ID | Comparison | Role | Diff | 95% CI | Bonferroni CI | + signs / 7 | 8.4 |
|---|---|---|---|---|---|---|---|
| C2 | run 5 vs r3 | primary | +2.09 | [+1.84, +2.33] | [+1.79, +2.42] | 7 | improves, robust |
| E1 | 6a vs run 5 | exploratory | −0.58 | [−0.77, −0.39] | [−0.82, −0.35] | 0 (7/7 negative) | 6a lower |
| E2 | 6b vs run 5 | exploratory | −2.14 | [−2.37, −1.91] | [−2.45, −1.85] | 1 (6/7 negative) | 6b lower |
| E3 | 6c vs run 5 | exploratory | −8.90 | [−9.48, −8.34] | [−9.60, −8.17] | 0 (7/7 negative) | 6c lower |
| E4 | run 5 vs run 4 | exploratory | +0.66 | [+0.48, +0.84] | [+0.43, +0.90] | 6 | run 5 higher |

C2's SICK-R difference is +0.0009 (positive, but essentially zero).

## Reading (plan 14.2, Enhancement 1 table)

- **C2 (primary): run 5 improves over r3, robust.** +2.09; the 95% and the Bonferroni intervals both exclude 0; positive on 7/7 datasets (SICK-R only by +0.0009).
- **Matching row: "Run 5 > r3 (C2 differs), and 6a, 6b, 6c all below run 5" → reversal adds information beyond TP.** All three controls are below run 5, with intervals excluding 0 and at least 6/7 signs agreeing (exploratory).
- **Rows that do not apply:**
  - "Run 5 ≈ run 4 > r3": run 5 is above run 4 (+0.66, exploratory), so reversal does not subsume TP.
  - "Run 5 ≈ r3": no.
  - "6a ≈ run 5": no, but 6a is close. Descriptively, a shuffled second view gives +1.51 over r3, against run 5's +2.09. So most of the gain over TP comes from adding any second view. The reversal-specific part is 0.58 (E1, exploratory).
  - "6b ≈ run 5": no; 6b ≈ r3 (−0.05), so the reversed wording alone adds nothing.
  - "6c ≈ run 5": no; the reversed TP view alone is far lower (70.44), below r2.
- **Caveats:**
  - keep_final and α were selected on dev, and the punctuation restriction was made after seeing the dev results.
  - The exploratory comparisons carry no multiplicity claim.
  - Run 4 vs run 5 compares different paths (batch-1 vanilla vs batched TP; S0.13 path cosines ≥ 0.999996).
  - 6c's test score (70.44) is much lower relative to its dev score (77.96) than the combined runs are; it is reported, not acted on.


## Additions (2026-11-04)

**(1) Share of run 5's gain that a shuffled second view already gets (exploratory).** Over r3, control 6a gains
+1.51 and run 5 gains +2.09 (test averages), so 6a reaches 72% of run 5's gain. The part specific to reversal
is +0.58 (E1, exploratory, no claim).

**(2) Caveats.**
- keep_final was chosen after seeing the dev results. The original rule picked `full`, which failed the rev-only check
  (76.01 < 76.72). The deviation is logged in gate_log.md.
- alpha = 0.6 is the edge of the plan's grid; values above it were not tried.
- Run 4 is vanilla at batch 1, while run 5 and r3 are TP with equal-length batches. Run 5 vs run 4 (E4) compares different numerical
  paths (S0.13: batched vs batch-1 cosines >= 0.999996).
- The `full` rule failed because the sentence-final "." moves to the front, directly after <PST>, in most reversed texts
  (count in T7 and logs/S3.2_debug.log).

**(3) Unexplained observation: 6c dev to test.** The reversed TP view alone scores 77.96 on STS-B dev and 70.44 as the
7-set test average (75.66 on STS-B test). This drop is larger than for the combined runs. The cause is not known and no causal
claim is made.
