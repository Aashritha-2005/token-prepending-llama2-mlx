# Enhancement 2 (CP + TP): reading (S4.7, 2026-11-05 14:09)

- C3 (primary) run 8 - r3: +0.35, 95% CI [+0.18, +0.52], Bonferroni [+0.14, +0.57], 5/7 positive -> **improves, robust**.
- E6 (exploratory) run8 - run7: +1.47 [+1.26, +1.67], 7/7 positive.
- E7 (exploratory) run9 - run8: -0.93 [-1.15, -0.71], 0/7 positive.
- E8 (exploratory) run8 - run5: -1.74 [-1.99, -1.49], 1/7 positive.
- Control 9 below run 8: yes (76.67 vs 77.60).
- Reading (plan 14.2): Run 8 > r3 (C3 differs): CP filters content TP does not.
- Caveat: r3 is batched TP (equal-length batches), run 8 is batch 1.

## Stage 4 closing notes (2026-11-05, written after S4.8 passed)
- Run 8 (CP + TP, frozen enh2: index 4 / paper l 5, alpha 4, aux with TP, scaled, TP s 1 e 7, layer 27, batch 1): test average **77.60**, +0.35 vs r3 (77.25).
- **C3 (primary) run 8 − r3: +0.35, 95% CI [+0.18, +0.52], Bonferroni [+0.14, +0.57], 5/7 positive → "improves, robust"** (pre-registered rules, notes/s4_preregistration.md section 6).
- Control 9 (sentiment auxiliary prompt): 76.67, below run 8 on 7/7 datasets (E7 0/7 positive), so the control check passed.
- Exploratory (no claims): E6 run 8 − run 7 +1.47 (7/7 positive); E7 run 9 − run 8 −0.93 (0/7); E8 run 8 − run 5 −1.74 (1/7). Run 5 (Enhancement 1, 79.34) stays the better enhancement on test.
- Caveats:
  - the effect is small (+0.35 on the average);
  - run 8 is below r3 on STS12 (66.44 vs 66.88) and STS-B (80.51 vs 80.72);
  - the setting is the best of 17 dev cells after one grid extension (notes/s4_preregistration_addendum1.md), so the dev score is optimistic; the test set was untouched until run 8;
  - the bootstrap covers test-pair sampling only, not dev-selection optimism;
  - run 8 is batch 1 while r3 is batched TP (S4.2b: TP batch 1 vs batched on dev −0.0036);
  - 8-bit model (fp16 activations).
- Cost (S4.8): run 8 0.159 s/sentence (batch 1, aux included) = 1.73× r3 (batched) and 1.07× TP batch 1.
