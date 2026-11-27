# Rebuttal: all result tables

All numbers are computed from the saved prediction files and measurements (Hi-EF test: 409 samples, 8 episodes; Hi-EF development CV: 2,421 samples, 45 episodes; MELD test: 1,341 windows, 223 dialogues; timings on a Tesla T4). Intervals are 95% percentile intervals from a bootstrap that resamples whole episodes (dialogues for MELD), 2,000 draws, on the seed ensembles. `ep+` is the number of episodes in which the first model beats the second on UAR.

| Table | Content | Reviewer concern it answers |
|---|---|---|
| 1 | Baseline and ViT rows recomputed from the saved test predictions | consistency of Table 1 and Figure 9; the strongest retrained configuration |
| 2 | Hi-EF test, paired differences with intervals | documentation of uncertainty; independence unit |
| 3 | Development CV, paired differences | main comparison with more episodes |
| 4 | Full against variants (10 seeds, same run) | contribution of participant separation |
| 5 | Input controls (listener, continuity) | whether the gain comes from content or from extra tokens |
| 6 | MELD: differences, per-class recall, predicted classes | metric-dependent result |
| 7 | Mirror and non-mirror samples | difficulty shared across models |
| 8 | Responder groups | role of responder observability |
| 9 | Cost of fusion, grouping and pre-processing | efficiency scope |

## 1. Baseline and ViT rows recomputed from the saved predictions

| ViT configuration | Inputs | Intra | Inter | Hi-EF UAR | Hi-EF WAR | MELD UAR | MELD WAR |
|---|---|---|---|---|---|---|---|
| ViT | F,B,S | Trans | LSTM+Trans | 21.62 | 32.27 | 14.87 | 46.09 |
| ViT | F,B,S,A | Trans | LSTM+Trans | 22.72 | 33.74 | 15.05 | 46.16 |
| ViT | F,B,S,T | Trans | LSTM+Trans | 20.82 | 30.81 | 14.97 | 45.64 |
| ViT | F,B,S,T,A | Avg | Avg | 22.84 | 32.52 | 14.41 | 46.16 |
| ViT | F,B,S,T,A | Trans | Avg | **24.23** | 32.76 | **16.42** | 38.85 |
| ViT | F,B,S,T,A | Avg | LSTM+Trans | 19.01 | 28.85 | 15.14 | 40.64 |
| ViT (Baseline) | F,B,S,T,A | Trans | LSTM+Trans | 21.30 | 31.30 | 15.27 | 45.34 |
| Qwen2.5-VL zero-shot | V,T | – | – | 15.75 | 21.76 | 22.65 | 25.80 |
| Qwen2.5-VL few-shot | V,T | – | – | 17.80 | 23.88 | 24.85 | 23.19 |
| PartiSet | F,S,T,A | Set | Set | **25.24** | **36.43** | 16.85 | 39.30 |

Hi-EF WAR is a multiple of 100/409 (Baseline 128 of 409 correct = 31.30; PartiSet 149 of 409 = 36.43); MELD WAR is a multiple of 100/1,341 (Baseline 608 of 1,341 = 45.34; PartiSet 527 of 1,341 = 39.30). The few-shot MELD value comes from the summary table of the run, because the uploaded probabilities contain only the zero-shot run.

## 2. Hi-EF test: PartiSet against the other models

| PartiSet − | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) | SD over seeds of ΔUAR |
|---|---|---|---|---|
| **Baseline** (21.30 / 31.30) | **+3.94 [1.22, 7.91]** | **+5.13 [0.67, 10.29]** | 6/8 | 0.35 |
| Strongest configuration, Trans/Avg (24.23 / 32.76) | +1.01 [−1.98, 4.60] | +3.67 [−0.38, 8.61] | 5/8 | 1.72 |
| F,B,S (21.62 / 32.27) | +3.63 [0.49, 8.14] | +4.16 [−0.60, 10.09] | 6/8 | 1.57 |
| F,B,S,A (22.72 / 33.74) | +2.52 [−0.06, 6.80] | +2.69 [−2.40, 8.50] | 5/8 | 1.59 |
| Avg/Avg (22.84 / 32.52) | +2.40 [−0.23, 6.31] | +3.91 [0.21, 8.43] | 5/8 | 1.61 |
| PartiSet without slots, faces pooled per clip, same test run (23.61 / 34.23) | +1.64 [−1.67, 4.53] | +2.20 [−1.33, 5.29] | 4/8 | 1.60 |
| Qwen2.5-VL zero-shot (15.75 / 21.76) | +9.49 [3.39, 13.95] | +14.67 [9.00, 18.32] | 7/8 | 1.50 |
| Qwen2.5-VL few-shot, set 0 (18.97 / 24.45) | +6.27 [−1.55, 11.66] | +11.98 [3.36, 18.18] | 6/8 | 1.50 |

## 3. Development CV (5 folds, 5 seeds): PartiSet against the Baseline

| PartiSet − | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|
| Baseline with PartiSet's features (23.25 / 32.71) | +3.37 [1.70, 4.85] | +5.20 [3.37, 7.01] | 29/45 |
| Baseline, original features (21.27 / 31.68) | +5.35 [3.20, 7.41] | +6.24 [3.15, 9.37] | 32/45 |

## 4. Matched full model against variants (same run, 10 seeds, 45 episodes)

| Full − variant | UAR (full / variant) | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|---|
| Faces pooled per clip (no slots) | 26.24 / 25.80 | **+0.44 [−0.74, 1.84]** | +0.12 [−1.91, 2.19] | 23/45 |
| Without listener tokens | 26.24 / 24.20 | **+2.04 [1.10, 3.07]** | +2.81 [1.36, 4.30] | 31/45 |
| Mean pooling instead of query | 26.24 / 26.42 | −0.18 [−1.26, 0.93] | +0.29 [−1.12, 1.82] | 19/45 |
| Without slot embeddings | 26.24 / 26.41 | −0.17 [−1.03, 0.61] | −0.29 [−1.34, 0.72] | 21/45 |
| A and L exchanged in a random half of the samples | 26.24 / 26.91 | −0.67 [−1.57, 0.23] | −0.99 [−2.81, 0.63] | 26/45 |

The last two rows were run later with the same folds, seeds and early-stopping episodes; `Full` is reproduced exactly (maximum absolute difference of the probabilities 0.0), so the pairing is valid. Without slot embeddings the ensemble UAR is 26.41, the value of Table 4 of the manuscript. For the exchange of A and L the matched run gives 26.91, whereas Table 4 of the manuscript lists 26.25 (a different run and implementation of the random half); in both, exchanging the slots does not lower UAR.

The pooled-face value in this matched run is 25.80; Table 4 of the manuscript lists 25.20 (a different run), so the manuscript row should be paired with the full-model score of its own run. Not available as matched runs: a random non-speaker as listener (needs identity labels of the faces) and the oracle responder (needs the future clip).

## 5. Input controls (development CV, 10 seeds, 45 episodes)

`Full` reproduces the earlier ten-seed run exactly (maximum absolute difference of the probabilities 0.0).

| Arm | Ensemble UAR | Ensemble WAR | Per-seed UAR (mean ± SD) |
|---|---|---|---|
| Full | 26.24 | 38.04 | 24.74 ± 0.97 |
| Listener slot emptied (masked) | 24.51 | 35.98 | 23.12 ± 0.66 |
| Listener slot replaced by the other-participants slot | 23.98 | 34.82 | 22.86 ± 0.70 |
| Listener slot taken from another sample | 23.36 | 33.62 | 22.51 ± 0.88 |
| Continuity of slots across turns broken (clips I/II permuted) | 26.46 | 38.74 | 24.71 ± 1.10 |
| One permutation for all turns (continuity kept, slot meaning changed) | 26.42 | 38.50 | 24.56 ± 0.47 |

| a − b | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|
| Full − listener emptied | +1.73 [0.67, 2.84] | +2.07 [0.55, 3.69] | 25/45 |
| Full − listener from other participants | +2.26 [1.07, 3.40] | +3.22 [1.67, 4.75] | 29/45 |
| Full − listener from another sample | +2.88 [1.39, 4.22] | +4.42 [2.39, 6.38] | 34/45 |
| Full − continuity broken | −0.22 [−1.13, 0.72] | −0.70 [−2.25, 0.77] | 22/45 |
| Full − one permutation for all turns | −0.18 [−1.01, 0.68] | −0.45 [−1.76, 0.87] | 20/45 |
| One permutation for all turns − continuity broken | −0.03 [−0.87, 0.79] | −0.25 [−1.21, 0.75] | 20/45 |
| Listener from other participants − listener emptied | −0.53 [−1.33, 0.30] | −1.16 [−2.27, −0.04] | 20/45 |
| Listener from another sample − listener emptied | −1.15 [−2.19, −0.06] | −2.35 [−4.13, −0.56] | 14/45 |

With the other-participants slot the listener's faces are absent from the input altogether, which is why it behaves like removing the listener. Five comparisons with the full model are reported without correction for multiple comparisons.

## 6. MELD test (1,341 windows, 223 dialogues)

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

## 7. Mirror and non-mirror samples (development CV, 2,421 samples)

| Group | n | PartiSet acc / UAR | Baseline with PartiSet's features | Baseline |
|---|---|---|---|---|
| Mirror (target = emotion of A) | 860 | 55.9 / 32.3 | 45.8 / 26.5 | 45.6 / 25.4 |
| Non-mirror | 1,561 | 28.0 / 22.1 | 25.5 / 20.3 | 24.0 / 18.4 |

| PartiSet − Baseline with PartiSet's features | ΔUAR [95% CI] | Δaccuracy [95% CI] |
|---|---|---|
| Mirror | +5.84 [2.88, 9.60] | +10.12 [6.43, 13.96] |
| Non-mirror | +1.81 [0.25, 3.37] | +2.50 [0.68, 4.38] |
| Mirror − non-mirror | +4.04 [0.84, 7.87] | +7.62 [3.65, 11.87] |

Errors shared between models (same samples): 69–73% of the samples that either model gets wrong; correlation of correctness 0.44–0.56. The groups are defined with the true labels and are used only to analyse predictions.

## 8. Responder groups (development CV, against the Baseline with PartiSet's features)

| Group | n | PartiSet UAR | Baseline with PartiSet's features | ΔUAR [95% CI] |
|---|---|---|---|---|
| B seen = proxy | 990 | 25.93 | 22.91 | +3.02 [0.55, 5.64] |
| B seen ≠ proxy | 1,114 | 27.79 | 24.42 | +3.37 [0.54, 5.88] |
| B not seen | 225 | 23.80 | 19.86 | +3.94 [−0.08, 8.68] |
| uncertain | 92 | 24.61 | 18.98 | +5.63 [−2.54, 15.21] |

The groups use the future clip and serve only as an analysis; the two small groups have wide intervals.

## 9. Cost (Tesla T4)

Fusion stage (precomputed features, batch 64, synthetic inputs of the real shapes):

| | Baseline | PartiSet |
|---|---|---|
| Parameters | 34.057M | 0.709M |
| FLOPs per sample (train-mode count) | 1.886 G | 0.030 G (24 frames per cell); 17.0 M (4 frames per cell) |
| Forward, median (p95), ms per sample | 0.643 (0.650) | 0.042 (0.048) |
| Forward + backward, median (p95), ms per sample | 2.130 (2.148) | 0.162 (0.170) |
| Peak GPU memory in training | 977 MiB | 239 MiB |

Pre-processing per clip (extraction code of the experiments, 40 random clips):

| Stage | Mean (s) | Median (s) | p95 (s) |
|---|---|---|---|
| Video decoding and frame sampling | 0.25 | 0.24 | 0.35 |
| Face detection + ArcFace + HSEmotion | 2.94 | 2.08 | 6.98 |
| Audio + ECAPA | 0.46 | 0.09 | 0.19 |

The mean of the audio stage contains one outlier (most likely the first call); the median is representative. Participant grouping takes 2.5 ms per sample (clustering and slot assignment) and 9.9 ms per sample for the face tensors (PCA fitted on 200 samples only; the real per-fold PCA fit takes about 7 s). Unit costs: CLIP ViT-B/32 on 16 images 0.112 s; YOLOv8-small on 16 frames 0.328 s. The pre-processing of the Baseline was not measured end to end.

## Not done

- Manual audit of the estimated responder oracle (needs annotators); the oracle statistics (2,127 of 2,421 samples matched, proxy equal to the responder in 993 of 1,451 assigned cases) come from the analysis logs.
- Qwen few-shot on MELD from predictions (the uploaded file contains only the zero-shot run).
