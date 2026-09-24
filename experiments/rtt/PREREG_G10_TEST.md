# Preregistration: G10, the single locked-test evaluation

This file was written and committed **before** `g10_test_preregistered.ipynb` was run and before any G10 test
prediction existed. The test split (409 MCIS, 8 episodes) has not been used for any model development. Earlier
test access is recorded in `README.md`: the G4 runs of the RtT/trajectory models, which is a different model family.

## Models (all frozen; hyper-parameters exactly as in the CV notebooks)

| Arm | Definition |
|---|---|
| `PaperBest` | The Hi-EF paper's best configuration as replicated in the user's baseline notebook (`EFBaseline`): per-clip temporal Transformers over the face and whole-frame features, cross-attention type fusion, then cross-attention modality fusion with text and audio, then LSTM (3 layers) + Transformer (2 layers) inter-clip fusion, then an MLP head. AdamW (lr 1e-4, weight decay 1e-5), batch 32, 50 epochs, ReduceLROnPlateau on the selection loss (patience 5, factor 0.5), best selection-UAR checkpoint. |
| `B1` | The benchmark-style forecaster used from G3 on (early stopping, patience 8). |
| `LateFusion` | B1 ⊕ unbalanced face LR on the G6b-style role means, equal log-space weight 0.5. |
| `RoleNet` | The G8b model, unchanged. |
| `RoleNet-noRole` | The G8b ablation, unchanged. |

## Training protocol

- Training uses the 45 train+val episodes. Early stopping and checkpoint selection use 5 of them, drawn with
  `random.Random(2026)`. The model is fit on the other 40.
- The PCA of the HSEmotion embedding, the face-LR scaler and C, and the class prior used for logit adjustment are all
  fitted on the 45 training episodes only.
- **5 seeds** per neural arm: 42, 123, 456, 789, 1024. The seed ensemble averages probabilities.
- Scoring:
  - **LA**: argmax(log p − log π_train), applied once to the seed-averaged probabilities.
  - **plain**: argmax of the same averaged probabilities.

## Primary and secondary analyses

**Confirmatory contrasts, tested in this fixed order** (fixed-sequence procedure, two-sided α = 0.05). The metric is
ΔUAR under LA, 7 classes, seed ensemble. The CI is a 95% bootstrap over the 8 test episodes (2,000 resamples). A
contrast counts as confirmed if the CI lower bound is > 0. Testing stops at the first unconfirmed contrast; later
contrasts are then reported as descriptive only.

1. **Primary:** `RoleNet` − `PaperBest`
2. `RoleNet` − `B1`
3. `RoleNet` − `LateFusion`
4. `RoleNet` − `RoleNet-noRole` (the role-assignment claim)

**Descriptive, no confirmatory claims:**
- plain scores and WAR;
- 6-class macro-recall without fear (the test set has very few fear examples);
- per-class recall;
- per-episode wins;
- listener visible vs. not visible in clip III;
- per-seed results;
- the paper's published numbers (UAR 23.72, WAR 35.19). These are on a different split and are **not** directly
  comparable.

## Commitments

- The notebook is run **once**. If it crashes for a technical reason before any test prediction is printed, it may be
  re-run unchanged. After that, no model, hyper-parameter, seed or scoring change is allowed.
- All arms and all numbers are reported, including unfavourable ones.
- With only 8 test episodes the confidence intervals will be wide. A positive but unconfirmed contrast is reported as
  *consistent with the CV result*, not as confirmation.
