# Rebuttal analyses on saved predictions

All numbers are computed from the saved prediction files (Hi-EF test: 409 samples, 8 episodes; Hi-EF development CV: 2,421 samples, 45 episodes; MELD test: 1,341 windows, 223 dialogues). Intervals are 95% percentile intervals from a bootstrap that resamples whole episodes (dialogues for MELD), 2,000 draws, on the seed ensembles. `ep+` is the number of episodes in which PartiSet beats the comparison on UAR. Scripts: `experiments/rtt/analysis/`.

## 1. R1: Hi-EF test, PartiSet against the ViT models

| PartiSet − | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) | SD over seeds of ΔUAR |
|---|---|---|---|---|
| **Baseline** (21.30 / 31.30) | **+3.94 [1.22, 7.91]** | **+5.13 [0.67, 10.29]** | 6/8 | 0.35 |
| Strongest configuration, Trans/Avg (24.23 / 32.76) | +1.01 [−1.98, 4.60] | +3.67 [−0.38, 8.61] | 5/8 | 1.72 |
| F,B,S (21.62 / 32.27) | +3.63 [0.49, 8.14] | +4.16 [−0.60, 10.09] | 6/8 | 1.57 |
| F,B,S,A (22.72 / 33.74) | +2.52 [−0.06, 6.80] | +2.69 [−2.40, 8.50] | 5/8 | 1.59 |
| Avg/Avg (22.84 / 32.52) | +2.40 [−0.23, 6.31] | +3.91 [0.21, 8.43] | 5/8 | 1.61 |
| Qwen2.5-VL zero-shot (15.75 / 21.76) | +9.49 [3.39, 13.95] | +14.67 [9.00, 18.32] | 7/8 | 1.50 |
| Qwen2.5-VL few-shot, set 0 (18.97 / 24.45) | +6.27 [−1.55, 11.66] | +11.98 [3.36, 18.18] | 6/8 | 1.50 |

## 2. R1: development CV (5 folds, 5 seeds)

| PartiSet − | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|
| Baseline with PartiSet's features (23.25 / 32.71) | +3.37 [1.70, 4.85] | +5.20 [3.37, 7.01] | 29/45 |
| Baseline, original features (21.27 / 31.68) | +5.35 [3.20, 7.41] | +6.24 [3.15, 9.37] | 32/45 |

## 3. M1: matched full model against variants (same run, 10 seeds, 45 episodes)

| Full − variant | UAR (full / variant) | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|---|
| Faces pooled per clip (no slots) | 26.24 / 25.80 | **+0.44 [−0.74, 1.84]** | +0.12 [−1.91, 2.19] | 23/45 |
| Without listener tokens | 26.24 / 24.20 | **+2.04 [1.10, 3.07]** | +2.81 [1.36, 4.30] | 31/45 |
| Mean pooling instead of query | 26.24 / 26.42 | −0.18 [−1.26, 0.93] | +0.29 [−1.12, 1.82] | 19/45 |

## 4. R2: MELD test

| PartiSet − | ΔUAR [95% CI] | ΔWAR [95% CI] |
|---|---|---|
| Baseline (15.27 / 45.34) | +1.58 [−0.35, 3.90] | **−6.04 [−8.22, −3.73]** |
| Strongest configuration, Trans/Avg (16.42 / 38.85) | +0.43 [−2.29, 3.35] | +0.45 [−1.98, 2.99] |

Recall by class and share of *neutral* predictions (%):

| Model | angry | disgust | fear | happy | neutral | sad | surprise | predicted neutral |
|---|---|---|---|---|---|---|---|---|
| Baseline | 15.5 | 0 | 0 | 0 | 91.4 | 0 | 0 | **90.8** |
| ViT Avg/Avg | 3.5 | 0 | 0 | 2.4 | 95.0 | 0 | 0 | 95.7 |
| ViT Trans/Avg | 12.0 | 2.4 | 5.0 | 13.2 | 71.5 | 4.8 | 6.1 | 70.2 |
| **PartiSet** | 17.6 | 0 | 5.0 | 17.9 | 70.5 | 1.9 | 5.0 | **70.5** |
| Qwen2.5-VL zero-shot | 49.3 | 17.1 | 0 | 29.7 | 23.4 | 19.0 | 20.0 | 20.7 |

*Neutral* is 47.8% of the test labels; an always-*neutral* predictor has WAR 47.80 and UAR 14.29.

## 5. T1: mirror and non-mirror samples (development CV, 2,421 samples)

| Group | n | PartiSet acc / UAR | Baseline with PartiSet's features | Baseline |
|---|---|---|---|---|
| Mirror (target = emotion of A) | 860 | 55.9 / 32.3 | 45.8 / 26.5 | 45.6 / 25.4 |
| Non-mirror | 1,561 | 28.0 / 22.1 | 25.5 / 20.3 | 24.0 / 18.4 |

PartiSet minus Baseline with PartiSet's features, by group:

| Group | ΔUAR [95% CI] | Δaccuracy [95% CI] |
|---|---|---|
| Mirror | +5.84 [2.88, 9.60] | +10.12 [6.43, 13.96] |
| Non-mirror | +1.81 [0.25, 3.37] | +2.50 [0.68, 4.38] |
| Mirror − non-mirror | +4.04 [0.84, 7.87] | +7.62 [3.65, 11.87] |

Errors shared between models (same samples): 69–73% of the samples that either model gets wrong; correlation of correctness 0.44–0.56.

## 6. T2: responder groups (development CV, against Baseline with PartiSet's features)

| Group | n | PartiSet UAR | Baseline with PartiSet's features | ΔUAR [95% CI] |
|---|---|---|---|---|
| B seen = proxy | 990 | 25.93 | 22.91 | +3.02 [0.55, 5.64] |
| B seen ≠ proxy | 1,114 | 27.79 | 24.42 | +3.37 [0.54, 5.88] |
| B not seen | 225 | 23.80 | 19.86 | +3.94 [−0.08, 8.68] |
| uncertain | 92 | 24.61 | 18.98 | +5.63 [−2.54, 15.21] |

## Not done

- R3: manual audit of the estimated responder oracle (needs annotators).
- R4: end-to-end profiling of pre-processing and fusion.
- M2, M3: need retraining.
- Qwen few-shot on MELD from predictions: the uploaded file only has the zero-shot run.

## 7. M2 and M3: input controls (development CV, 10 seeds, 45 episodes; run `k1_partiset_controls.ipynb`)

The `Full` arm reproduces the earlier ten-seed run exactly (maximum absolute difference of the probabilities 0.0).

| Arm | Ensemble UAR | Ensemble WAR | Per-seed UAR (mean ± SD) |
|---|---|---|---|
| Full | 26.24 | 38.04 | 24.74 ± 0.97 |
| Listener slot emptied (masked) | 24.51 | 35.98 | 23.12 ± 0.66 |
| Listener slot replaced by the other-participants slot | 23.98 | 34.82 | 22.86 ± 0.70 |
| Listener slot taken from another sample | 23.36 | 33.62 | 22.51 ± 0.88 |
| Continuity of slots across turns broken (clips I/II permuted) | 26.46 | 38.74 | 24.71 ± 1.10 |
| One permutation for all turns (continuity kept, slot meaning changed) | 26.42 | 38.50 | 24.56 ± 0.47 |

Paired differences in UAR (a − b), 95% interval from resampling whole episodes:

| a − b | ΔUAR [95% CI] | episodes a better |
|---|---|---|
| Full − listener emptied | +1.73 [0.67, 2.84] | 25/45 |
| Full − listener from other participants | +2.26 [1.07, 3.40] | 29/45 |
| Full − listener from another sample | +2.88 [1.39, 4.22] | 34/45 |
| Full − continuity broken | −0.22 [−1.13, 0.72] | 22/45 |
| Full − one permutation for all turns | −0.18 [−1.01, 0.68] | 20/45 |
| One permutation for all turns − continuity broken | −0.03 [−0.87, 0.79] | 20/45 |
| Listener from other participants − listener emptied | −0.53 [−1.33, 0.30] | 20/45 |
| Listener from another sample − listener emptied | −1.15 [−2.19, −0.06] | 14/45 |

## 8. Source-preservation controls (development CV, 10 seeds, 45 episodes; `k1_partiset_controls.ipynb`, session c)

Same features, same fusion model, same number of face tokens (nine), same training protocol; only the grouping of the face evidence into slots changes. *Replicated pooling*: every slot of a turn holds the same clip-level frames (up to 32). *Random grouping*: each face frame of a turn goes to one of the three slots at random (fixed seed, identity not used). In both, the per-slot synchrony entries are replaced by their mean over slots, so no grouping leaks. `Full` reproduces the earlier ten-seed run exactly (maximum absolute difference 0.0).

| Arm | Ensemble UAR | Ensemble WAR | Per-seed UAR (mean ± SD) |
|---|---|---|---|
| Full PartiSet (participant slots) | 26.24 | 38.04 | 24.74 ± 0.97 |
| Replicated pooling | 24.87 | 36.76 | 23.42 ± 0.65 |
| Random grouping | 25.04 | 36.89 | 23.36 ± 0.70 |

| a − b | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|
| Full − replicated pooling | +1.37 [0.23, 2.56] | +1.28 [−0.75, 3.16] | 27/45 |
| Full − random grouping | +1.20 [0.00, 2.43] | +1.16 [−0.64, 3.01] | 26/45 |
| Replicated pooling − random grouping | −0.17 [−1.04, 0.71] | −0.12 [−1.24, 0.97] | 19/45 |

Reading: Full is above both controls in UAR (the lower interval end is 0.23 and 0.00, so the random-grouping difference is borderline), and not significantly above in WAR. The two controls do not differ from each other. These controls use 32 frames per slot (replicated) or a disjoint split of them (random), against up to 24 frames per cell in Full; slot embeddings are kept in all arms. Not corrected for multiple comparisons.

The retrained sham-listener control (listener taken from another sample, in training and evaluation) is the row "Listener slot taken from another sample" above: Full − sham = +2.88 [1.39, 4.22] UAR, +4.42 [2.39, 6.38] WAR; the sham is also below the emptied-listener arm (−1.15 [−2.19, −0.06]).
