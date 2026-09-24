# Recognize-then-Transition (RtT) gates

Hypothesis: on Hi-EF, forecasting B's emotion splits into (a) recognizing A's state in clip III and
(b) an A→B transition. Part (b) is nearly solved by a 7×7 table: with gold A emotion, `P(B | E_A)`
reaches 28.10 UAR / 39.72 WAR on the locked validation split, above every deployable model so far.
Part (a) is the bottleneck, so these gates test ways to recognize A better.

| File | What it does |
|---|---|
| `cheap_baselines.py` | Majority, Copy-A, Markov (gold A) and TF-IDF baselines, train → val, source-bootstrap CIs |
| `g1_llm_recognition.ipynb` | G1: an OpenAI LLM reads subtitle lines I–III, predicts A's emotion/polarity and B directly; A posterior → `P(B \| E_A)` |
| `g2_recognizer_all_labels.ipynb` | G2: multimodal A recognizer trained on clip III only vs. all labeled clips (III ∪ IV), 5 seeds; A posterior → `P(B \| E_A)` |
| `g3_trajectory_forecaster.ipynb` | G3: cross-fitted recognizer gives soft emotion/polarity/confidence for clips I–III; forecaster arms B1, B1+certw, traj_only, B1+traj, B1+traj+certw with paired bootstrap vs B1 |
| `g3b_robustness.ipynb` | G3b: robustness of `traj_only` — per-fold eval trajectories (no train/eval feature mismatch), 3 recognizer seeds, both selection protocols (inner-dev / val), clip ablation III · II–III · I–III, logistic-regression forecaster, gate summary |
| `g4_test_preregistered.ipynb` | G4: the single preregistered test run — selection on val, evaluation on test (see below) |
| `g5_episode_cv.ipynb` | G5: preregistered secondary — the primary contrast under 5-fold cross-validation over all 53 episodes (run after G4) |
| `g6a_listener_visibility.ipynb` | G6a: gate for the listener-aware formulation — face detection + identity clustering on clips I–IV (train+val only) to measure how often B is visible while listening in clip III or spoke in clip I/II, plus annotated montages |
| `g6b_listener_expression.ipynb` | G6b: does the listener's face in clip III (HSEmotion expression + valence/arousal, clips I–III only) forecast B's emotion beyond A's face? Logistic regressions train → val, overall and on listener-visible MCIS, with an early-frames-only boundary check and the listener identity tracked into clips I/II |
| `g6c_listener_cv.py` | G6c: re-analysis of the G6b features — zero-shot listener vs A face, balanced logistic regression out-of-fold over all 45 train+val episodes, presence-only control, boundary split |
| `g7b_faces_into_b1.ipynb` | G7b: B1 + role-grounded face features (A, listener, listener in I/II, dominant faces of I/II) through a zero-initialised branch; early-frame and presence-only arms; both selection protocols, 5 seeds (needs the `role-features` dataset) |
| `g7_late_fusion.py` | Late fusion of B1 (G3b val probabilities) with a face-only balanced LR, equal weight, presence-only control |
| `g8a_role_features.ipynb` | G8a: role-agnostic extraction for every clip used as I–III (train/val/test inputs, no labels read): per-face box, pose, ArcFace, HSEmotion logits + 1280-d embedding, mouth opening, landmarks; per-clip ECAPA (whole + 1.5 s windows) and a 10 Hz energy envelope; resumable shards |
| `g8b_rolenet_cv.ipynb` | G8b: RoleNet (role-tagged face tokens, speech/scene tokens, balance aids) vs B1, B1+faces-joint (the G7b design), LateFusion with an unbalanced face LR, and four ablations; 5-fold CV over the 45 train+val episodes, PCA fitted per fold, scores plain and with one post-hoc logit adjustment; diagnostics for the design-debate hypotheses (split identities, mouth–audio sync AUC, person-specific inertia via clip-IV voice used for analysis only) |
| `g9_rolenet_plus_cv.ipynb` | G9 (round 1 of at most 2): RoleNet+ adds speaker-role tags on the speech tokens (A by voice vs clip III; B by a cross-fitted B-pointer whose training labels come from clip-IV voice) and a B-previous-utterance token with an auxiliary head. Arms: RoleNet, +spk, RoleNet+, RoleNet+ without face roles, and an ORACLE-pointer analysis arm; same folds/seeds as G8b; preregistered adoption rule |
| `build_notebooks.py` | Regenerates the notebooks |

All notebooks use the locked split `source_folder_split_seed42.csv` (1,993 / 428 / 409 MCIS, 37 / 8 / 8 episodes).
The test split raises an error unless `UNLOCK_TEST = True`.

## Running on Kaggle

1. Attach `hi-ef-dataset`, `hi-ef-features-v2` and a dataset holding `source_folder_split_seed42.csv`;
   fix the paths in each notebook's CONFIG cell if they differ.
2. G1 only: add the Kaggle secret `OPENAI_API_KEY` and set `MODEL`. Responses are cached in `/kaggle/working`.
3. Run G2 first if you want G1's optional ensemble cell to pick up `g2_val_probs_ALL.npz`.

## Gate criteria

* G1: LLM `emotion_A` UAR above the frozen recognizer (23.21).
* G2: arm `ALL` beats arm `III` on A recognition UAR, across seeds.
* Either gate passing → the RtT forecast should move toward the oracle (≈28 UAR / ≈40 WAR);
  it must beat B1 (24.65 / 35.79) with a source-bootstrap CI that excludes zero before the test split is opened.
* G3: `B1+traj` (or `B1+traj+certw`) beats `B1` with a paired source-bootstrap CI on ΔUAR that excludes zero
  and wins on most seeds. Early stopping uses inner-dev training episodes, so G3's B1 is not numerically
  identical to the report's B1 (selected on val); compare arms within G3.
* G3b: (1) `traj_I-III` beats `B1` under inner-dev selection with a paired CI on ΔUAR above zero,
  (2) `traj_I-III@val` is at least as good as `B1@val` (report protocol), (3) recognizer-seed spread of
  `traj_I-III` UAR ≤ 1.5 points. Only after this gate is the test split opened, once, for a preregistered set of models.

## Preregistered test plan (frozen after G3b, before any test access)

* **Method:** `traj_I-III` — forecaster on the soft trajectory of clips I–III (emotion + polarity posteriors,
  max-prob, entropy per clip) from a recognizer cross-fitted over training episodes (5 folds × 3 seeds, seed
  posteriors averaged; evaluation rows predicted once per fold model, predictions averaged). Early stopping on val.
* **Primary contrast:** `traj_I-III@val` − `B1@val` on test, ΔUAR of the 5-seed ensemble, 95% paired bootstrap over
  test episodes. Confirmed if ΔUAR > 0 with CI lower bound > 0; directional if ΔUAR > 0 with CI including 0.
* **Secondary (descriptive):** ΔWAR, macro-F1, certain-label rows, per-episode wins, inner-dev pair, A's contribution
  (`traj_I-III@val` vs `traj_I-II@val`, `B1@val` vs `B0@val`), clip ablation, logistic regression, deployable
  recognize-then-transition, majority / Copy-A oracle / Markov oracle.
* G3b validation evidence: inner-dev ΔUAR +3.32 [+0.62, +5.42] (5/5 seeds); val-selected pair tied
  (26.23 vs 25.77 seed-mean UAR; ensemble ΔUAR −0.62 [−2.98, +2.52]); recognizer-seed spread 1.06 UAR.
* **Preregistered secondary analysis (G5), added before test access:** the same contrast (`traj_I-III` vs `B1`,
  both early-stopped on a selection split of 8 episodes) under 5-fold episode-level cross-validation over all 53
  episodes (outer folds balanced by MCIS count; recognizer cross-fitted inside each outer training set with one seed
  per inner fold; 3 forecaster seeds). Readout: pooled out-of-fold ΔUAR with 95% bootstrap over episodes, per-fold Δ,
  and the same numbers on the 45 non-test and the 8 locked-test episodes. **Run order: G4 first, then G5**, because
  the G5 folds evaluate on locked-test episodes. If G4 and G5 disagree, both are reported.

## Test-run log

* **G4 run 1 (invalid for trajectory arms):** a cache-key bug in `make_T` let the trajectory arms reuse the all-zero
  trajectory tensor built for the raw arm with the same clips, so `traj_I-III@val`, `traj_I-III` and `traj_I-II@val`
  were trained on constant inputs (val UAR 16.05 vs 26.23 for the identical configuration in G3b). Rows without a
  preceding raw arm on the same clips (`B1@val`, `B0@val`, `B1`, `traj_II-III@val`, `traj_III@val`, LR, RtT, references)
  were unaffected and reproduced G3b's validation numbers exactly.
* **Fix:** the cache key now records whether a trajectory is attached, and an assertion checks the tensor. No method,
  hyper-parameter, seed or selection change. G4 is re-run once with the fix; the fixed `traj_I-III@val` must reproduce
  G3b's validation UAR (26.23 seed mean) before its test number is read. Both runs are reported.

## G6a results (train+val, 600 MCIS)

* Faces: B visible in clip III with a usable face 47.9%; B and A in the same frame only 0.7% (reaction shots);
  B frames cover ~12% of clip III; the no-clip-IV listener rule is 81.4% precise.
* Voices: clip III and IV sound like the same speaker in 1.3%; B spoke in clip II 33.4%, in clip I 27.4%;
  B observable (usable face in III or a voice turn in I/II) 72.1%.
* The clip I/II speaker is a third person (neither A nor B) in 36.7% / 40.5% of MCIS, so the rule
  "context speaker ≠ A ⇒ B" is only 47% / 40% precise. B must be anchored on the clip-III listener, not on turn-taking.

## G6b / G6c results (train+val only)

* HSEmotion reads Hi-EF faces: zero-shot A face → A label UAR 23.1 (chance 14.3).
* Zero-shot, same 1,234 MCIS with a visible listener: listener face → B's next label UAR 20.18 vs A face 16.46,
  ΔUAR +3.72 [+0.60, +7.24] (episode bootstrap over 45 episodes).
* Balanced LR out-of-fold: A+listener − A = +2.1 / +2.6 / +1.9 UAR on all MCIS (3 fold shuffles), +3.7 / +5.6 / +4.6
  on listener-visible MCIS; presence/frame-share only gives ≈ 0, so the gain comes from the expression.
* 51% of listener frames lie in the last 20% of clip III; early-frame-only gains are smaller (+1.1 to +2.2, CI includes 0).
  Decision: keep the original Hi-EF protocol (clip III is a legal input) and report the early-frame arm descriptively.
* Adding the listener on top of context faces gains little overall (+0.3) because in 74% of listener-visible MCIS the
  same person is already in clips I/II; where it is not, +2.3 / +7.7 / +3.8 UAR. The signal is B's own face anywhere
  in the input, which motivates G7b.

## G7 late fusion (val, test untouched)

* Joint training inside B1 (G7b, zero-initialised face branch) started below B1 on the first seed.
* Late fusion, equal weight, B1 (inner-dev, 5 seeds) with a face-only balanced LR: seed-ensemble UAR 21.88 → 26.38,
  ΔUAR +4.50 [+0.33, +7.30]; per seed +1.30 / +4.89 / +2.33 / +2.58 / +5.38; presence-only control +0.43 [−1.18, +1.79].
  Larger on listener-visible MCIS (+5.06) than on the rest (+2.37).
* Reading: the face information is useful but a large jointly trained encoder crowds it out (modality imbalance).
  The weight 0.5 was not tuned but was not preregistered either; the next evaluation fixes it in advance and uses
  cross-validation over all 45 train+val episodes.

## G8 plan (fixed before running G8b)

* Development and model comparison use 5-fold cross-validation over the 45 train+val episodes (folds balanced by
  MCIS count, 5 inner episodes for early stopping, 3 seeds). The test split stays locked.
* RoleNet hyper-parameters are set a priori in the notebook's CONFIG and are not tuned on the CV results.
* Every model is trained with plain cross-entropy; seed-averaged probabilities are scored plain and with one post-hoc
  logit adjustment (log p − log π of the training fold, τ = 1). Adjusting only once follows the G7 check, where
  class-balanced training plus adjustment cancelled the gain.
* Primary contrast: RoleNet − B1 under the logit adjustment, pooled out-of-fold seed-ensemble ΔUAR, 95% bootstrap over
  the 45 episodes.
  Secondary: RoleNet vs LateFusion (weight 0.5), vs RoleNet-noRole, vs RoleNet-noBalance; faces-only vs context-only.

## Design debate (four critic agents, two rounds) and the logit-adjustment check

* Check run on val (test untouched): B1 with a post-hoc logit adjustment alone gives +1.77 [−3.55, +8.32] and is very
  seed-unstable. B1 ⊕ an **unbalanced** face LR gives +2.10 [+0.09, +4.52] (WAR +4.4). About half of the earlier
  +4.5 late-fusion gain was therefore prior correction, and about +2 UAR is face information.
* The critics read a written summary, not the data. Their claims are therefore tested in G8b rather than adopted:
  joint face training inside B1 (was one seed in G7b), person-specific inertia (was based on a weak proxy),
  mouth–audio sync quality, split identities.
* Not changed yet: the State–Transition head. The debate recommends a staged, cross-fitted, log-linear version with two
  paths (persistence, reaction to A). It is built only if the G8b diagnostics and results support it.

## G8b results (5-fold CV over the 45 train+val episodes, 2,421 MCIS, 3 seeds; test untouched)

**Diagnostics:**
- Split identities are rare: the A-vs-L centroid cosine has a median of 0.01 and only 0.8% of values fall in [0.30, 0.45).
- Mouth–audio sync is a weak cue (AUC 0.613).
- Emotional persistence (clip-IV voice, analysis only):
  - clip II: P(y_IV = y_II) is 50.6% when B spoke, 45.2% when a third person spoke, and about 35% when A spoke; B vs others Δ +8.4 [−0.5, +16.5];
  - clip I: 43.0 / 42.6 / ~35%.

**Main results:** seed ensemble, ΔUAR with 95% episode bootstrap. "LA" means scored with one post-hoc logit adjustment. The 6-class column is macro-recall without fear (fear has 32 examples; one extra fear hit is worth 0.45 UAR).

| Contrast | LA, 7-class | LA, 6-class | plain, 7-class |
|---|---|---|---|
| **RoleNet − B1 (primary)** | **+4.83 [+2.40, +7.13]**, positive in 5/5 folds | +6.16 [+4.02, +7.96] | +4.44 [+2.64, +6.08] |
| RoleNet − LateFusion | +3.15 [+0.97, +5.31] | +4.20 [+2.07, +6.18] | +2.13 [+0.78, +3.35] |
| B1+faces-joint − B1 | +3.66 [+0.39, +6.85]* | +0.10 [−2.08, +2.07] | +0.51 [−1.38, +2.30] |
| RoleNet − RoleNet-noRole | −1.14 [−3.13, +1.19]* | +0.75 [−0.78, +2.24] | +1.17 [−0.01, +2.29] |
| RoleNet − RoleNet-noBalance | −0.02 | +1.02 [−0.85, +2.90] | +0.78 [−0.58, +2.06] |
| facesOnly − ctxOnly | +3.05 [+0.68, +5.70] | +3.04 [+0.80, +5.26] | +2.98 [+1.25, +4.65] |

\* Fear-driven: B1+faces-joint gets 10/32 fear hits, noRole 5, RoleNet 1.

**Subsets:** RoleNet − B1 (LA) is +5.91 [+3.00, +8.88] where the listener is visible in clip III (n = 1,451) and +2.41 [−1.19, +6.03] where it is not (n = 970).

**Reading:**
- The compact role/expression model beats B1 and late fusion robustly.
- Joint face training inside B1 does not help.
- Faces carry more signal than the compact context.
- The specific contribution of role assignment (about +1) and of the balance aids (about +1) is not established; per-fold 6-class Δ for role is −2.0 / +1.2 / +5.3 / −3.0 / +3.2.
- CV results are exploratory, because the design was informed by analyses on the same episodes. The hyper-parameters and the primary contrast were fixed before the run.

## G9 plan (round 1 of at most 2, fixed before running)

* **Adoption rule.** Adopt RoleNet+ over RoleNet for the single test run only if all three hold:
  - the seed-ensemble ΔUAR (LA) > 0;
  - the 6-class Δ (without fear) > 0;
  - the 6-class Δ is positive in ≥ 4/5 folds.
* **Clip IV.** Its voice gives the B-pointer's training labels and the B-previous-utterance targets. It is never an
  input; the ORACLE arm is analysis only.
* **After round 2** (if any), no further architecture changes. Then one preregistered test run.

## G9 results (round 1): preregistered decision is KEEP RoleNet

**Setup checks:**
- The RoleNet arm reproduces G8b exactly, per seed and per fold.
- B-pointer AUC, out-of-fold: 0.821 overall (clip I 0.833, clip II 0.803).

**Contrasts**, seed ensemble, ΔUAR with 95% episode bootstrap:

| Contrast | LA, 7-class | LA, 6-class | plain, 7-class |
|---|---|---|---|
| RoleNet+ − RoleNet | +0.11 [−2.23, +2.03] | +0.13 | −0.10 |
| RoleNet+spk − RoleNet | −0.48 | — | — |
| RoleNet+ ORACLE − RoleNet+ | −0.64 [−1.94, +0.50] | — | — |
| RoleNet+ − RoleNet+ −faceRoles | +0.61 | +1.75 [−0.05, +3.69] | +0.85 |

- **Decision.** RoleNet+ − RoleNet is positive in only 2/5 folds, so RoleNet+ is not adopted.
- **The ORACLE arm is flat.** Even a perfect "B spoke in I/II" pointer adds nothing on top of RoleNet. The label-level
  B-specific persistence is therefore not exploitable by tagging context speech with speaker roles in this
  architecture.
- **Face-role assignment** gives a consistent but borderline benefit:
  - G8b, RoleNet − noRole: plain +1.17 [−0.01, +2.29];
  - G9, RoleNet+ vs −faceRoles: 6-class +1.75 [−0.05, +3.69];
  - G9, same contrast on listener-visible MCIS: plain +1.66 [+0.19, +3.11].
