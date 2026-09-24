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
