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
| `PREREG_G10_TEST.md` | Preregistration of the single locked-test run, committed before G10 was run |
| `g10_test_preregistered.ipynb` | G10: trains PaperBest (the Hi-EF paper's best configuration, from the replication notebook), B1, LateFusion, RoleNet and RoleNet-noRole on the 45 train+val episodes (5 seeds) and evaluates once on test with the preregistered fixed-sequence contrasts |
| `g11_channel_ablations_cv.ipynb` | G11: channel ablations of RoleNet in the G8b CV protocol (same folds and seeds). Face tokens are grouped into simulation (A + speech/scene), strict surrogation (bystanders O) and target observation (listener L); each group is removed from attention one at a time and in combination (9 arms). Hypotheses H1–H4 and the reading rule for the paper framing are fixed in the notebook header; plain scoring; test untouched |
| `g12_cs_rolenet_cv.ipynb` | G12: CS-RoleNet — role-specific states (B-self, Event, Scene) updated in clip order by separate GRU cells, with an additive per-path logit decomposition (optional penalised interaction term). Arms: RoleNet, CS-add, CS-int, and retrained path ablations; decision rule and faithfulness check (contribution share vs retrain drop) fixed in the header; G8b folds/seeds; plain scoring; test untouched |
| `g13_token_ablations_cv.ipynb` | G13: token-level ablations of RoleNet in the G8b CV protocol — readout (mean-pool, listener-token readout, read-only query, 3 queries), embeddings (role, clip, absent token vs mask), speech/scene tokens and speech components (text/audio/voice, clip-III speech), time horizon (clip III only, no clip I), plus a re-seeded `Full` as the noise floor; reading rules fixed in the header; plain scoring; test untouched |
| `g14_ten_seed_confirmation_cv.ipynb` | G14: 10-seed re-run of `Full`, `Q-meanpool`, `minus-L`, `T-clipIIIonly` and `noRole` in the G8b CV protocol; intervals from a two-level bootstrap over seeds and episodes; decision rules (confirmed / non-inferiority margin 1 UAR for dropping the query) fixed in the header; plain scoring; test untouched |
| `g15_dynamics_gates.ipynb` | G15: gates for the latent affect-dynamics formulation (DynAffect), analysis only: (1) does the information in the listener's face about B's next emotion decay with time before clip IV (near vs far observation, same MCIS); (2) does B's last labelled emotion (B identified by clip-IV voice, analysis only) persist less after a longer gap, and toward which baseline; decision rules fixed in the header; test untouched |
| `g16_time_vs_type.ipynb` | G16: separates elapsed time from the type of observation behind G15 gate 1 — listener face in clip II vs clip I when B spoke in neither (primary), clip III vs a non-speaking far clip, and far observations split by whether B spoke; B's speaking from the clip-IV voice (analysis only); decision rules fixed in the header; test untouched |
| `g17_listening_vs_speaking.ipynb` | G17: paired check of constraint (iii) — the same person's face in a context clip where B is silent vs one where B speaks, same MCIS (B's speaking from the clip-IV voice, analysis only); separate and pooled logistic regressions; decision rule for keeping (iii) in the problem statement fixed in the header; test untouched |
| `g19_forecastable_distinctions_cv.ipynb` | G19: which pairwise emotion distinctions are forecastable from clips I–II vs I–III — pair AUCs (21 pairs; pairs with a class under 50 MCIS reported only) for RoleNet (10 seeds, token mask per information set) and a pooled-feature logistic regression; two-level bootstrap; decision rules (cross-predictor Spearman ≥ 0.7; non-uniform gain from clip III) fixed in the header; test untouched |
| `SYNTHESIS.md` | Evidence synthesis after G20: scope decisions, test-access record, feature-pipeline audit, findings, directions by status, screening checklist |
| `g20_interaction_gate_cv.ipynb` | G20: gate for forecast value of text × audio-visual non-additivity — `Additive` (a(X)+b(Z), jointly trained) vs `Local` (+ same-clip text × audio bilinear term, main test) and `Window` (full RoleNet / LR window products, secondary) in a neural family (10 seeds, temperature-scaled on the early-stopping episodes) and an LR family (C and temperature by inner CV); decision on Δ_NLL in both families; late-fusion control; EMAP on RoleNet logits; test untouched |
| `g23_negative_separability_cv.ipynb` | G23: are angry/sad/disgust separable at all? Pair AUCs of the same LR on the same features for forecasting (I–III → B at IV) vs recognition of B at clip IV (diagnostic) and of A at clip III; reading rules (INVALID / PERCEPTION LIMIT / FORECASTING-SPECIFIC / INTERMEDIATE) fixed in the header; test untouched |
| `g24_appraisal_reaction_cv.ipynb` | G24: does an LLM appraisal reading of subtitles I–III (10 dimensions, zero-shot, no labels, clip IV text never sent) add information on *which* negative emotion B reacts with? LR as G23 `FC` with/without appraisal; NEG pair AUC on the shift subset (B ≠ A); PASS/STOP rule fixed in the header; direct LLM forecast as control; contamination probe; test untouched |
| `g25_listener_state_vs_reaction.ipynb` | G25: does the listener's face show B's ongoing state or B's reaction to A? LR on the listener's HSEmotion expression in clip III (`NOW`), clips I/II (`HIST`), their average (`AVG`, one state) or both separately (`SEP`, allows change); REACTION SIGNAL if `SEP` beats `AVG` on shift MCIS; validity check vs the class prior; optional correlation with RoleNet's listener gain (attach `g14_oof_probs.npz`); test untouched |
| `g26a_clip4_faces.ipynb` | G26a: the unchanged G8a extraction on clip IV of train+val MCIS (test clip IV not read); used only as the responder target / oracle in G26 |
| `g26_responder_pointer_cv.ipynb` | G26: Responder-Pointer RoleNet — does knowing who responds matter (Oracle vs rule, stage 1) and can a pointer over candidate identities, supervised by clip IV at training only, learn it (stage 2)? 10 seeds, two-level bootstrap, rules fixed in the header; test untouched |
| `g27_mention_aggregation_pilot.ipynb` | G27 pilot: early (B) vs late (C) aggregation of contextualised mention representations (frozen RoBERTa-large, pronoun + NER person mentions per clip, whole-context candidate), same gate and head, plus whole-context arm A; shared fold-fitted audio-visual vector; 3 seeds; primary NLL(B) − NLL(C); test untouched. Tests mention aggregation, not full event-role perspectives |
| `g28_masked_listener_training_cv.ipynb` | G28: does training RoleNet with the listener's face features masked (C1 all L, C2 L-III, C3 L history; drawn with p = 0.5 among conditions that change the MCIS) improve the forecast under the same masking, and at what cost on the data as they are? Same architecture; R-std vs R-mask, 10 seeds; shared checkpoint rule J_dev = 0.5·NLL(C0) + 0.5·mean NLL(Ck); primary Δ_rec = mean per-seed NLL difference, two-level bootstrap with shared draws; auxiliary: do the 9 presence flags forecast B's emotion (logistic vs prior); test untouched |
| `g29_context_pooling_cv.ipynb` | G29: context-conditioned face-evidence pooling before compression — P0 FramePool (RoleNet), P1 mean pool, P2 stronger context-free scorer, P3 context added after pooling, P4 context in the frame scores (proposed; shared g_θ with P3, zero-initialised context maps), R frame tokens without pooling (reference); 10 seeds, G14 checkpoint rule (dev UAR), deciding metric mean per-seed NLL, confirmed only if NLL(P3) − NLL(P4) and NLL(P2) − NLL(P4) both have CIs above 0; step 0 frames-per-cell and within-cell variance (descriptive); time/memory reported; test untouched |
| `g30_aux_supervision_cv.ipynb` | **Cancelled by the authors during fold 0 (no saved output, no decision).** Partial fold-0 log, 7 seeds with S4, mean ΔNLL (positive = S4 better): S3 − S4 +0.0016 (S4 better 3/7), S2 − S4 +0.0006, S1 − S4 +0.0067, S0 − S4 +0.034 (−0.13 to +0.15 across seeds); NLL-selected checkpoints at epoch 0–4. Descriptive only. G30: RoleNet's face auxiliary head — S0 original (mean of 9 face tokens), S1 none, S2 L only, S3 L + context, S4 L + stop-gradient(context); same h_ψ for S2–S4, auxiliary losses on pre-dropout tokens, A and context heads unchanged; 10 seeds; NLL for checkpoint selection and evaluation; method supported iff NLL(S3) − NLL(S4) CI > 0, worth adding iff also vs S0 and S1; per-model probe of z_L's conditional gain (diagnostic); saves checkpoints and inner-dev/outer logits; test untouched |
| `g31_mirror_mixture_linear_check.ipynb` | G31, step 1 of direction A (mechanism queries): linear check whether a "B mirrors A" / "rest" mixture p = g·p_mirror + (1 − g)·p_rest beats additive logits with the same information; p_mirror learned from A's clip-III label (cross-fitted), L2 by grouped inner CV, 5-fold episode CV, out-of-fold NLL, episode bootstrap; passed iff ΔNLL(additive − mixture) CI > 0 and AUC(g → Y_B = Y_A) CI > 0.5; noRole variant descriptive; no GPU training; test untouched |
| `g32_mirror_diagnostics.ipynb` | **Not run (dry run on synthetic data only); no result.** G32 (second look after G31): (i) oracle mirror — p_mirror from A's gold clip-III label, smoothing ε chosen by each model; (ii) A-vs-B split — p_mirror from A's evidence only, p_own from L only, gate reads everything; additive control with the same raw information; wider L2 grids, separate gate penalty, gate initialised at the mirror share; per diagnostic passed iff ΔNLL CI > 0 and AUC(g → mirror) CI > 0.5; joint reading table fixed before running; grid-edge hits reported; linear, no GPU; test untouched |
| `g33a_temporal_audit_cv.ipynb` | G33a (campaign G33, with G33b): temporal structure audit — `ordered` (R0, exported for G33b), test-time swap I↔II, `bag12`, `bag123`; bag defined operationally (numerical invariance under clip permutations < 1e-5, plus removal of the voice-cosine-to-clip-III feature for bag123); D_reliance, U_order, U_boundary; checkpoint by inner-dev UAR, temperature per arm × fold × seed on inner dev, primary = mean per-seed calibrated NLL on held-out folds; swap TV / flip rate as diagnostics; conclusions about explicit clip identity only; test untouched |
| `g33b_relational_bias_cv.ipynb` | G33b: factorial TIME × PERSON × EVIDENCE attention biases (zero-initialised, 112 parameters) on RoleNet — R-full, leave-one-family-out, R-random (per-family label permutation with a fixed seed, marginals kept); 5 seeds × 3 folds (authors' budget rule; R0 retrained here because the G33a manifest no longer matches); same selection / calibration / metric as G33a; unlocked only if NLL(R0) − NLL(R-full) and NLL(R-random) − NLL(R-full) both have CIs > 0; test untouched |
| `g33c_order_probe.ipynb` | G33c: does clip content reveal the order of clips I and II? Pairwise logistic probe on the per-clip inputs RoleNet gets (faces per role, text, audio, scene, voice row; no clip label), both orders, 3-fold episode CV, AUC with episode bootstrap, per feature group; fixed reading: AUC CI < 0.60 → G33a's bag12 removed order information, CI > 0.60 → G33a tested only explicit order labels; test untouched |
| `g34_hypothesis_queries_cv.ipynb` | **Withdrawn before running (see SYNTHESIS §5, run gate).** G34: emotion-hypothesis queries (screening, 5 seeds × 3 folds). Same RoleNet backbone; readout T0 (query token → linear), G (7 generic queries cross-attend the 15 evidence outputs, mean → linear) or H (same module, query c bound to class c, s_c = u_cᵀ·LN(h_c) + b_c; G and H have equal parameter counts). G33 protocol (inner-dev UAR checkpoint, per-fold temperature, calibrated held-out NLL, seeds × episodes bootstrap). Gate: Δ_base = NLL(T0) − NLL(H) and Δ_specificity = NLL(G) − NLL(H) both with CI > 0 → re-run at 10 × 5, otherwise stop this line. Descriptive: query differentiation D_JS and per-query source profile; test untouched |
| `g35_group_emap_cv.ipynb` | G35: does trained RoleNet use non-additive interactions between evidence groups? Three-group EMAP (L = listener faces I–III; H = clips I–II A/O faces, speech, scene; E = clip III A/O faces, speech, scene), partners drawn independently per group from the same evaluation rows (M = 32), plus one-group-vs-rest EMAPs; RoleNet trained with the G33 recipe, 5 seeds × 3 folds; each predictor temperature-scaled on its inner-dev logits; primary D_int = NLL(EMAP3) − NLL(full) on held-out folds, seeds × episodes bootstrap; fixed reading: CI > 0 → interactions carry value, CI upper < 0.01 → no interaction value above 0.01 NLL, else inconclusive; test untouched |
| `g36_resampling_cv_test.ipynb` | G36: class-resampled RoleNet (β = 0 / 0.5 / 1, sampling ∝ n_class^−β). Part A: development CV (45 episodes, 5 folds, 10 seeds, plain UAR, per-class recall, ΔUAR vs RoleNet with seeds × episodes bootstrap); fixed rule: the better resampling arm is selected and the test is read only if its CV ΔUAR > 0. Part B: **second, disclosed test read** of the selected arm with the G10 protocol (5 seeds), compared with the saved G10 predictions of PaperBest and RoleNet (attach `g10_test_probs.npz`) |
| `g36b_test_upsqrt.ipynb` | G36b: part B of G36 as a standalone notebook — G36 part A (10 seeds × 5 folds) selected `Up-sqrt` (CV UAR 26.95 vs 26.10, ΔUAR +0.86 [−0.94, +2.18]); trains it with the G10 protocol and reads the test once (**second, disclosed read**), compared with the saved G10 predictions of the baseline and RoleNet (`g10_test_probs.npz` or the `.npy` folder); first cell checks all inputs |
| `g37_case_frames.ipynb` | G37: motivation / case-study frames — from the candidates of `analysis/case_study_select.py` (saved test predictions; final model vs baseline) keeps MCIS with ≥ 3 people in clips I–III and ≥ 2 in clip III (ArcFace identities as in RoleNet), renders a one-row strip with face boxes coloured by role (A / L / O) and transcripts, plus a 4 × 3 grid; figures only; inputs: `hi-ef-dataset`, `g8a-features`, CPU |
| `g38_resampling_noregress.ipynb` | G38: class resampling that protects every class — arms RoleNet (uniform), Up-quarter (∝ n_c^−0.25), Tail-median (classes below the median count oversampled to it, ≤ 3×); development CV (10 seeds × 5 folds); fixed rule: an arm is eligible only if no class loses > 5 recall points vs RoleNet on CV, the best eligible arm is selected, the test is read only if its CV ΔUAR > 0 (**third, disclosed read**, G10 protocol, compared with the saved G10 predictions); first cell checks all inputs |
| `g39_party_statistics.ipynb` | G39: party statistics of the 2,421 development MCIS that motivate RoleNet (no model; test not read) — number of people in clips I–III and in clip III (ArcFace identities seen in ≥ 2 frames), where the responder B is before clip IV (listening in III / only in I–II / not seen; B = dominant clip-IV face matched as in G26, analysis only), B's visibility rank in clip III (proxy = rank 2), whether B spoke in I/II (voice); outputs `g39_numbers.json`, `g39_people.pdf`, `g39_responder.pdf`; inputs: `hi-ef-dataset`, `hi-ef-split`, `g8a-features`, G26a output; CPU |
| `m1_meld_prepare_features.ipynb` | M1: MELD in the Hi-EF format — MCIS windows (I–III → IV by a different speaker), Hi-EF-layout annotation file, `hi-ef-features-v2`-style clip features (CLIP ViT-B/32 face/frame/text, AudioCLIP ESResNeXt-FBSP 527) for clips I–III |
| `m2_meld_g8a_features.ipynb` | M2: the unchanged G8a face/voice extraction on MELD clips I–III, split into chunks (`N_CHUNKS`, `CHUNK`) |
| `m3_meld_rolenet.ipynb` | M3: G10's RoleNet / RoleNet-noRole / B1 / LateFusion / PaperBest on MELD (train → fit, dev → early stopping, test once; 3 seeds; no speaker names as input), plus mirroring-vs-shift analysis |
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

## G10: the single preregistered test run (409 MCIS, 8 episodes). Run once, as registered in `PREREG_G10_TEST.md`.

**Preregistered fixed-sequence result** (ΔUAR, LA, 7-class, seed ensemble, 95% bootstrap over the 8 test episodes):

| # | Contrast | ΔUAR [95% CI] | Outcome |
|---|---|---|---|
| 1 | RoleNet − PaperBest (primary) | +3.77 [−7.49, +10.12] | **NOT CONFIRMED**; the sequence stops here |
| 2 | RoleNet − B1 | +2.45 [−0.44, +5.39] | descriptive only |
| 3 | RoleNet − LateFusion | +2.29 [−0.38, +5.52] | descriptive only |
| 4 | RoleNet − RoleNet-noRole | −1.93 [−7.31, +4.77] | descriptive only |

**Seed-ensemble test scores:**

| Model | UAR, LA | UAR, plain | WAR, plain |
|---|---|---|---|
| B1 | 20.86 | 23.82 | 34.23 |
| RoleNet | 23.31 | **25.24** | **36.43** |
| RoleNet-noRole | 25.24 | 23.61 | 34.23 |
| PaperBest | 19.55 | 21.40 | 32.52 |
| LateFusion | 21.02 | 21.82 | 32.03 |

**Descriptive analyses, specified in the preregistration:**
- RoleNet − PaperBest: plain +3.84 [+1.2, +6.8]; 6-class LA +9.95 [+6.6, +13.6]; RoleNet wins 6/8 episodes.
- RoleNet − LateFusion: plain +3.42 [+1.6, +5.9].
- RoleNet − B1: plain +1.42 [−0.8, +4.9].
- PaperBest − B1: plain −2.42 [−3.5, −0.6].

**Post-hoc diagnosis** (labelled as such, not a confirmatory result):
- The test set has only 6 fear examples, so one fear hit is worth 2.38 UAR points.
- Under LA, PaperBest predicts fear for 27.6% of test MCIS (true rate 1.5%) and angry for 1.2% (true rate 18.8%). It gets 2/6 fear hits (+4.76 UAR) while its angry recall drops to 1.3%. RoleNet gets 0/6; RoleNet-noRole gets 1/6.
- The width of the primary CI and the sign flip of contrast 4 are therefore driven by the rare fear class under the preregistered LA scoring.
- Without adjustment (plain), no model predicts fear. On that common footing RoleNet is the best model, and its gaps to PaperBest and to LateFusion have CIs above 0.
- The choice of LA-scored 7-class UAR as the primary test metric was a design error for a test set with 6 fear examples. It is reported, not replaced.

### Reporting decision after G10: plain (benchmark-standard) scoring as the main metric

- Plain argmax scoring is the metric used by the Hi-EF paper and prior work, and it was listed in the preregistration
  as a descriptive analysis.
- The paper will state that the preregistered primary metric was LA-scored UAR and that it was not confirmed
  (+3.77 [−7.49, +10.12]), with the fear-class diagnosis above.

**Test, plain, seed ensemble** (95% bootstrap over the 8 test episodes):

| Model | UAR [95% CI] | WAR |
|---|---|---|
| PaperBest | 21.40 [16.2, 24.7] | 32.52 |
| B1 | 23.82 [18.1, 27.3] | 34.23 |
| LateFusion | 21.82 [16.8, 24.8] | 32.03 |
| RoleNet-noRole | 23.61 [18.4, 27.5] | 34.23 |
| **RoleNet** | **25.24 [20.8, 28.2]** | **36.43** |

| Contrast | ΔUAR [95% CI] | ΔWAR [95% CI] | Episodes won |
|---|---|---|---|
| RoleNet − PaperBest | **+3.84 [+1.24, +6.76]** | +3.91 [+0.40, +7.79] | 7/8 |
| RoleNet − B1 | +1.42 [−0.78, +4.94] | +2.20 [−1.32, +6.65] | 6/8 |
| RoleNet − LateFusion | **+3.42 [+1.64, +5.88]** | +4.40 [+1.61, +7.29] | 6/8 |
| RoleNet − RoleNet-noRole | +1.64 [−1.67, +4.53] | +2.20 [−1.33, +5.29] | 4/8 |

**CV, plain** (45 train+val episodes, from G8b):

| Contrast | ΔUAR [95% CI] |
|---|---|
| RoleNet − B1 | +4.44 [+2.64, +6.08] |
| RoleNet − LateFusion | +2.13 [+0.78, +3.35] |
| RoleNet − noRole | +1.17 [−0.01, +2.29] |

PaperBest was not part of the CV.

## G11 results: channel ablations of RoleNet (5-fold CV, 2,421 MCIS, 45 episodes, 3 seeds, plain; test untouched)

Question: which evidence does RoleNet use? Face tokens are grouped as **simulation** (A's face + speech/scene
tokens: the event), **strict surrogation** (bystanders O: other people's reactions, Gilbert et al. 2009) and **target
observation** (the listener L, i.e. B in ~81% of MCIS). Removed groups are masked out of attention at training and
evaluation. `Full` re-runs G8b's RoleNet: 26.00 UAR (G8b 25.92).

Evidence available: listener in III 59.9%, listener in I/II 44.2%, O anywhere 69.1%, O in III 16.2%,
O present with no listener in III 35.7%.

| Arm (seed ensemble) | UAR [95% CI] | WAR |
|---|---|---|
| Full | 26.00 [23.8, 28.1] | 37.30 |
| minus-A (A's face removed) | 25.66 | 35.89 |
| minus-L3 (listener in clip III removed) | 25.00 | 36.31 |
| minus-O (bystanders removed) | 24.64 | 35.85 |
| minus-L (listener removed everywhere) | 23.48 | 33.75 |
| Sim-only (A + speech/scene) | 23.22 | 32.96 |
| Obs-only (L + O faces) | 23.10 | 31.89 |
| Self-only (L faces) | 20.60 | 30.86 |
| Surr-only (O faces) | 17.39 | 24.95 |

Preregistered hypotheses (ΔUAR, 95% episode bootstrap):
- **H1 strict surrogation**, Full − minus-O on O-present MCIS: +1.38 [−0.46, +3.38] → directional only.
  H1b, Surr-only − Sim-only where only bystanders are seen: **−3.91 [−6.83, −0.37]** → the opposite of surrogation > simulation.
- **H2 target observation**, Full − minus-L: **+2.52 [+1.35, +3.63]**, positive in 5/5 folds → holds
  (+3.13 [+1.46, +4.70] where the listener is visible in III).
- **H3 current reaction**, Full − minus-L3 where the listener is in III: +0.61 [−0.84, +2.07] → directional only.
  The listener's clip I/II appearances carry most of it: minus-L3 − minus-L on listener-in-I/II MCIS +2.76 [+0.54, +5.10].
- **H4 observation beats simulation**, Obs-only − Sim-only: −0.12 [−1.93, +1.68] → not supported.

Other contrasts: Full − Sim-only +2.78 [+1.41, +4.06] (5/5 folds); Full − Obs-only +2.90 [+0.74, +4.89];
Full − minus-A +0.34 [−1.32, +1.88]; Self-only − Surr-only +3.22 [+1.17, +5.04].

**Post-hoc controls (analysis of the saved probabilities, not preregistered).** Removing a token group also changes
training, so the ablation deltas contain a generic part. It shows up where the removed tokens do not even exist:
- Full − minus-O on MCIS **without** any O face: +1.14 [−1.17, +3.55], about the same as with O (+1.36).
  Difference-in-differences (O present vs absent): **+0.24 [−2.59, +3.01]** → no O-specific information.
- Full − minus-L on O-present MCIS **without** a listener: +1.26 [−0.74, +3.30]; with a listener +2.51 [+0.54, +4.39].
  Difference-in-differences: +1.25 [−1.76, +3.98] → the listener-specific part is about half of the raw H2 effect
  and not significant on this subset.
- The notebook's automatic "Framing" line ("observation beats simulation") follows the rule "H2 or H4", but H4 itself
  failed; the supported reading is complementarity, see below.
- Note: in the saved npz of this run, `has_listener_in_III` actually holds the "listener in I/II" mask (key
  collision, fixed in the notebook afterwards). The logged hypothesis results were computed in memory and are unaffected.

**Reading.**
1. Strict surrogation (bystanders' reactions) carries no detectable information for B's next emotion in Hi-EF, and
   on its own it is clearly worse than simulation. The Gilbert-style "surrogation beats simulation" claim is **not**
   supported here.
2. Observing the target helps (H2, 5/5 folds), mostly through B's appearances across the context, not specifically
   the clip-III reaction; part of the raw effect is a generic ablation cost.
3. The event and observation channels are **complementary**: each alone is ~23 UAR and together 26 (both +2.8–2.9).
4. A's face adds nothing beyond the speech/scene tokens.

## G12 results: CS-RoleNet (additive, clip-ordered role paths) — decision KEEP RoleNet

5-fold CV, 2,421 MCIS, 45 episodes, 3 seeds, plain scoring; test untouched. `RoleNet` reproduces G8b exactly (25.92).

| Arm (seed ensemble) | UAR [95% CI] | WAR |
|---|---|---|
| RoleNet | 25.92 [23.9, 27.6] | 37.79 |
| CS-int (additive + penalised interaction) | 24.60 [22.7, 26.4] | 35.48 |
| CS-add -Event | 24.50 | 35.65 |
| CS-add -B | 24.32 | 35.36 |
| CS-add (additive) | 23.78 [21.5, 26.1] | 33.87 |
| CS-add -Scene | 22.76 | 32.88 |

- CS-add − RoleNet: **−2.14 [−3.79, −0.29]** (1/5 folds > 0); CS-int − RoleNet: −1.32 [−2.80, +0.34] (2/5);
  CS-int − CS-add: +0.82 [−0.38, +2.08] (4/5). Interaction share in CS-int: 13.6%.
- **Decision (fixed rule): KEEP RoleNet.** Neither CS arm reaches RoleNet's UAR.
- Path ablations of CS-add (retrained): removing B −0.54 and removing Event −0.73 *raise* UAR slightly; removing Scene
  lowers it by 1.02 [−0.22, +2.26] (4/5 folds).
- Contribution shares in CS-add: B 25.1%, Event 32.9%, Scene 42.0%. The B share falls from 35.6% (listener in III) to
  9.4% when no listener is visible, so the decomposition behaves sensibly.
- Faithfulness is weak: rank agreement between shares and retrain drops is +0.50 over 3 paths. B and Event receive
  sizeable shares but are not *necessary* (the other paths compensate when they are retrained without them).

**Reading.** Forcing the evidence into separate, additive, clip-ordered paths costs about 2 UAR. RoleNet's joint
attention across roles and modalities carries information that the additive structure removes; the penalised
interaction term recovers only part of it. Path contributions of the additive model are exact but describe a weaker
model, and they are not faithful to what each path is needed for (redundancy between paths).

## G13 results: token ablations of RoleNet, including the query token (5-fold CV, 2,421 MCIS, 3 seeds, plain; test untouched)

Every arm is retrained from scratch. `Full` is the same architecture as RoleNet, rebuilt in `RoleNetTok`.
`Full-reseed` is the same model with other seeds; it gives the noise floor.

| Family | Arm | UAR | Full − arm [95% CI] | folds Full better |
|---|---|---|---|---|
| Noise | Full / Full-reseed | 25.00 / 26.17 | −1.17 [−2.69, +0.57] | 2/5 |
| Query / readout | Q-meanpool (no query, mean of tokens) | 25.57 | −0.57 [−2.21, +1.21] | 3/5 |
| | Q-readL3 (read the listener-III token) | 25.10 | −0.10 [−1.63, +1.67] | 3/5 |
| | Q-readonly (tokens cannot attend to the query) | 25.67 | −0.66 [−1.71, +0.41] | 1/5 |
| | Q-3queries | 25.86 | −0.85 [−2.27, +0.56] | 1/5 |
| Embeddings | E-noRoleEmb | 25.80 | −0.79 [−1.71, +0.16] | 2/5 |
| | E-noClipEmb | 25.28 | −0.28 [−1.51, +0.90] | 2/5 |
| | E-absentMask (mask absent roles instead of an "absent" token) | 25.56 | −0.56 [−2.02, +0.89] | 2/5 |
| Speech / scene | C-noSpeech | 25.52 | −0.52 [−1.96, +0.93] | 3/5 |
| | C-noScene | 25.20 | −0.20 [−1.79, +1.42] | 3/5 |
| | C-noSpeechIII | 26.17 | −1.17 [−2.80, +0.46] | 1/5 |
| | C-noText | 26.25 | −1.25 [−2.86, +0.34] | 1/5 |
| | C-noAudio | 25.07 | −0.06 [−1.12, +1.12] | 2/5 |
| | C-noVoice | 26.00 | −1.00 [−2.18, +0.11] | 1/5 |
| Time horizon | T-clipIIIonly | 23.08 | **+1.92 [−0.26, +4.08]** | 3/5 |
| | T-noClipI | 24.70 | +0.30 [−1.21, +1.77] | 4/5 |

**Run-to-run noise (post hoc).** Four runs of the same model exist with 3 seeds each: G11 Full 26.00, G12 RoleNet
25.92, G13 Full 25.00, G13 Full-reseed 26.17. They disagree on 15–30% of argmax predictions. Summation order changes
between implementations, so even identical seeds give different training runs. All 220 three-seed ensembles drawn from
these 12 runs give UAR 25.55 ± 0.40 (range 24.59–26.42; 95% of combinations in [24.75, 26.23]). A 12-seed
super-ensemble gives 26.35. Against this distribution:

- **T-clipIIIonly (23.08) is below every one of the 220 same-model ensembles**, 2.47 under their mean. Dropping clips
  I and II hurts; this is the only token ablation clearly outside noise. T-noClipI (24.70, 1.8th percentile) points
  the same way but is weaker, so clip II carries most of the extra context.
- Every other arm sits inside the same-model range (14th–98th percentile). C-noText (97.7th) and C-noSpeechIII (95.9th)
  are at the high edge, but with 15 arms one such value is expected by chance.

**Reading.**
1. The query token is not needed. Mean-pooling, reading the listener's clip-III token, a read-only query and three
   queries all score within noise of Full.
2. Role and clip embeddings and the explicit "absent" token are replaceable. RoleNet has no positional encoding, so
   without the role embedding the Transformer cannot tell A's, L's and O's face tokens apart (except through the
   per-role absent token). The score does not drop, so the model uses *what* the faces show more than *whose* face it
   is. This does not contradict G11: removing the listener's tokens removes their content, not only their label.
3. No single speech or scene sub-channel is needed on its own; the model compensates with the others.
4. Context from earlier clips (I, II) matters; together with the G11 listener effect (+2.52) these are the effects
   that survive the noise floor.
5. Contrasts of about 1 UAR between 3-seed runs are within noise. That includes RoleNet − noRole in G8b (+1.17) and
   minus-O in G11. Final claims should use more seeds (≥10) or report against the same-model noise distribution.

## G14 results: ten-seed confirmation (5-fold CV, 2,421 MCIS, 10 seeds, plain; test untouched)

The first three `Full` seeds reproduce G13 `Full` bit for bit (25.00), so the run is deterministic per implementation.

| Arm | 10-seed ensemble UAR | per-seed UAR (mean ± SD) | Full − arm, seeds + episodes [95%] | episodes only | folds Full better | Verdict (fixed rule) |
|---|---|---|---|---|---|---|
| Full | 26.24 | 24.74 ± 0.97 | | | | |
| Q-meanpool | 26.42 | 24.18 ± 0.97 | −0.18 [−1.55, +1.62] | [−1.26, +0.93] | 3/5 | inconclusive → keep query |
| minus-L | 24.20 | 22.80 ± 0.75 | **+2.04 [+0.21, +3.46]** | [+1.10, +3.07] | 5/5 | **CONFIRMED** |
| T-clipIIIonly | 24.11 | 22.52 ± 0.55 | **+2.13 [+0.20, +4.08]** | [+0.50, +3.51] | 4/5 | **CONFIRMED** |
| noRole | 25.80 | 24.11 ± 0.78 | +0.44 [−1.21, +2.16] | [−0.74, +1.84] | 4/5 | not confirmed |

- Three-seed sub-ensembles of the 10 `Full` seeds: 25.49 ± 0.56 (range 24.30–26.96), which matches the G13 estimate.
- The listener effect sits where the listener is visible: Full − minus-L is +2.80 on the 1,451 MCIS with the
  listener in clip III and +0.54 on the 970 without.

**Reading.**
1. Two effects survive seed and episode noise: **the listener's face tokens** (+2.0) and **the context of clips I–II**
   (+2.1).
2. **Splitting the faces into A / L / O roles is not established** (+0.44, CI includes 0). `noRole` still sees the
   listener's face, pooled with the others. With G13 `E-noRoleEmb`, this says the gain comes from *having the
   listener's face* in the input, not from labelling it as a separate role.
3. Dropping the query token neither helps nor hurts clearly. Non-inferiority within 1 UAR is not shown, so RoleNet
   keeps the query token.

## G15 results: gates for the latent affect-dynamics formulation (train+val, analysis only; test untouched)

| Gate | Result | Fixed-rule verdict |
|---|---|---|
| G1a, listener face near (clip III) vs far (clip II/I), same 1,070 MCIS, equal frames | IG near 0.104 bits, far 0.026 bits; Δ **+0.079 [+0.025, +0.135]** bits (all frames +0.069); median time to clip IV 0.32 s vs 3.98 s | **PASS** |
| G1b, late vs early half of L's frames in clip III (331 MCIS) | +0.007 [−0.068, +0.080] bits | no within-clip effect |
| G2a, persistence of B's last labelled emotion, near (480) vs far (156) | β 1.51 vs 1.19; Δβ +0.32 [−0.04, +0.70]; within the same 33 MCIS β(y_II) − β(y_I) +0.30 [−0.16, +0.98] | **DIRECTIONAL** |
| G2b, return to the episode baseline | D +0.068 [−0.112, +0.253] bits | home base: **global** |

Decision under the fixed rule: gate 1 PASS and gate 2 DIRECTIONAL, so the result is reported and the authors decide.

**Reading.**
1. A recent observation of the listener carries about four times more information about B's next emotion than
   an observation of the same person one or two clips earlier. This is the core prediction of the formulation.
2. The gate does not separate *elapsed time* from *type of observation*: the near observation is the listener
   reacting in clip III, the far one is the same person in clips I/II (possibly speaking). G1b, the only
   within-clip time contrast (a few seconds), shows nothing. So the evidence supports recency at the clip level,
   not yet a continuous-time decay law.
3. Label-level persistence also weakens with the gap, in the predicted direction (in both the between- and the
   within-MCIS contrast), but the CI touches 0. The decay is toward the population prior; an episode-specific home
   base is not supported.

## G16 results: elapsed time vs type of observation (train+val, analysis only; test untouched)

| Contrast | n | IG a / b (bits) | a − b [95%] |
|---|---|---|---|
| **T1** L in II vs L in I, B silent in both (primary) | 262 | −0.090 / −0.073 | **−0.017 [−0.115, +0.078]** |
| T1-all, L in II vs L in I | 616 | +0.026 / +0.055 | −0.029 [−0.083, +0.023] |
| T2, L in III vs far clip where B was silent | 623 | +0.030 / +0.010 | +0.020 [−0.052, +0.098] |
| T3, G1a far observation when B spoke in that clip | 447 | far −0.002, near +0.125 | descriptive |
| T3, G1a far observation when B was silent in that clip | 623 | far +0.046, near +0.090 | descriptive |

**Decision (fixed rule): no decay beyond the clip-III reaction → discrete-step state, not an OU process.**

Post hoc, model-based (G14 10-seed probabilities, same MCIS): the per-MCIS gain from the listener tokens
(log p Full − log p minus-L) does not depend on the time between L's last clip-III frame and clip IV
(Spearman ρ +0.005, partial for clip-III duration and L frame count +0.003; 1,451 MCIS).

**Reading.**
1. With the type of observation held (non-speaking), a clip-II face is not more informative than a clip-I face,
   and clip III is not clearly better than a non-speaking far clip. The G15 near > far gap is mostly driven by
   far observations in which B was *speaking* (information ≈ 0), not by elapsed time.
2. Continuous-time decay (OU / DynAffect in real time) is not supported at the time scales of Hi-EF.
3. What matters is *what kind* of observation of the target is available: a listening/non-speaking face carries
   information about the next emotion, a speaking face much less. Label-level persistence (G15 gate 2) remains
   directional.

## G17 results: listening vs speaking observations of the same person (paired; analysis only; test untouched)

| Contrast | n | silent − speaking (bits) [95%] |
|---|---|---|
| **P1** same MCIS, separate models, equal frames (primary) | 257 | **−0.007 [−0.131, +0.120]** |
| P1, silent clip later / earlier | 95 / 162 | −0.051 [−0.333, +0.242] / +0.004 [−0.182, +0.220] |
| P2 paired, one model over all context observations | 257 | −0.008 [−0.068, +0.056] |
| P2 all observations: silent / speaking (unpaired) | 1,047 / 639 | +0.066 [+0.014, +0.120] / +0.052 [−0.015, +0.113] |

**Decision (fixed rule): constraint (iii) is dropped.** Paired within the same MCIS and person, a face observed while
B is silent is not more informative than one observed while B speaks. The G16 T3 gap (≈ 0 vs +0.046 bits) was a
between-MCIS composition effect. Together with G16, the G15 near > far gap is explained neither by elapsed time nor
by listening vs speaking; only a small, non-significant advantage of the clip-III reaction remains (G16 T2 +0.020).

## G19 results: forecastable pairwise distinctions, I–II vs I–III (5-fold CV, train+val, RoleNet 10 seeds + LR; test untouched)

**Decision (fixed rule): CONTINUE.** Rule 1: Spearman(RoleNet, LR) of pair AUCs at I–III = 0.75 (≥ 0.7; at I–II 0.86).
Rule 2: 4 eligible pairs gain from clip III with CI > 0, 11 pairs are above chance at I–II with a Δ CI including 0.

| Pair (RoleNet) | AUC I–II | AUC I–III | Δ [95%] |
|---|---|---|---|
| happy/sad | 0.702 | 0.770 | **+0.068 [+0.040, +0.090]** (LR +0.049 [+0.025, +0.075]) |
| happy/surprise | 0.702 | 0.753 | **+0.051 [+0.024, +0.084]** |
| angry/happy | 0.756 | 0.804 | **+0.048 [+0.028, +0.069]** |
| disgust/happy | 0.717 | 0.763 | **+0.046 [+0.016, +0.083]** |
| happy/neutral, neutral/sad, angry/neutral | 0.75–0.77 | 0.75–0.79 | +0.007 to +0.017, CI incl. 0 |
| angry/sad, angry/disgust, disgust/sad | 0.56–0.59 | 0.59–0.62 | CI incl. 0 |

Post hoc (not part of the rules):
- All four pairs that gain from clip III involve **happy** (mean Δ over happy pairs +0.046, other pairs +0.010).
- Distinctions inside the negative emotions (angry/sad, angry/disgust, disgust/sad) and surprise vs
  neutral/disgust stay near chance (AUC ≈ 0.58–0.62) with or without clip III.
- Which pairs gain from clip III is **not** stable across predictor families: Spearman of Δ between RoleNet and LR
  = 0.25. LR confirms only happy/sad; with clip III added, LR loses AUC on several pairs (e.g. neutral/sad −0.031,
  disgust/neutral −0.035), plausibly because its feature count grows by half.
- Rule 1 passed narrowly (0.75 vs 0.70).

### G19 follow-up: structure of the forecastable distinctions (PIC analysis; exploratory, train+val OOF only; test untouched)

Scripts: `g19_pic/pic.py` (raw probabilities) and `g19_pic/pic_cal.py` (temperature-calibrated: T fitted on the 4
discovery folds, applied to the held-out fold). Input: `g19_oof_probs.npz` and `g19_lr_oof_probs.npz` from G19.
Bootstrap: source folders are resampled, and every pair/statistic is recomputed in each draw (1,000 draws). Seed
noise is **not** included (RoleNet = mean of 10 seeds).

**(0) Direct test, happy-pair gain minus other-pair gain** (mean Δ over 5 happy pairs − mean over 10 other pairs):
RoleNet +0.036 [+0.016, +0.059]; LR +0.007 [−0.009, +0.024]. The difference holds for RoleNet only.
(Note: the 49% "happy stays happy" figure is P(y_IV = happy | y_III = happy), i.e. A's label at III → B's label at
IV, not the same person.)

**(A) Spectrum** (Cg = λ D_π g, constant removed):
- RoleNet, raw: λ1 ≈ λ2 (I–II 0.325/0.282; I–III 0.322/0.269). The top-1 direction rotates between I–II and I–III
  (|cos| 0.43), but the top-2 subspace is stable (principal cosines 1.00, 0.99). It is spanned by
  happy-vs-rest and negative-vs-neutral. Calibrated: I–II 0.138/0.119, I–III 0.171/0.142.
- LR, raw probabilities are badly miscalibrated (flat spectrum, fear loading −8.6; T = 3.5–5.2). Even after
  calibration, the LR top-1 direction is fear-dominated (rare-class artifact).
- Across families, the top-2 subspaces share only about one dimension (cosines [0.64, 0.05] at I–II, [0.51, 0.15]
  at I–III).

**(B) Directions orthogonal to the standardized happy axis h:** the RoleNet residual r1 is
"angry/sad/fear/disgust vs neutral", stable across information sets (cos 0.99). The LR residual is fear-dominated;
the cross-family |cos| is 0.29–0.34.

**(C) Held-out validation.** Directions are found on 4 folds (I–III probabilities) and evaluated on the 5th fold
with true labels. R² is relative to a constant baseline; Δ_g = R²(I–III) − R²(I–II), which equals
(MSE_I–II − MSE_I–III)/Var g.

| Evaluated on | Direction | R² I–II | R² I–III [95%] | Δ_g [95%] |
|---|---|---|---|---|
| RoleNet | h (happy vs rest) | 0.118 | 0.178 [0.142, 0.213] | **+0.061 [+0.038, +0.083]** |
| RoleNet | r1 found on RoleNet | 0.131 | 0.140 [0.082, 0.193] | +0.009 [−0.012, +0.027] |
| LR | h | 0.035 | 0.035 [0.007, 0.066] | −0.000 [−0.010, +0.011] |
| LR | r1 found on RoleNet | 0.078 | 0.063 [0.038, 0.097] | −0.016 [−0.029, −0.001] |
| either | directions found on LR | ≤ 0 | ≤ 0 | — |

**Reading against the pre-stated decision table: row 2.** A stable direction off the happy axis exists
(negative emotions vs neutral). Both families predict it on held-out sources when it is discovered from RoleNet.
But the gain from clip III does not replicate across families: for happy it appears in RoleNet only, and for the
residual direction in neither. → Keep the difficulty-structure conclusion; do **not** confirm the
information-added-by-direction hypothesis. No objective is designed. If an objective round is ever reached, the
mandatory control is CE + an auxiliary happy/non-happy target.

## G20 results: interaction gate (5-fold CV, train+val; neural 10 seeds + LR; test untouched)

**Decision (fixed rule): STOP** this direction on Hi-EF. The main test fails in both families. Reading: no
sufficiently strong evidence of forecast value from text × audio-visual non-additivity with these data and model
classes; this does not show that the true distribution has no interaction.

| Contrast | Neural Δ [95%] | LR Δ [95%] |
|---|---|---|
| **Main**: NLL(Additive) − NLL(Local) | −0.008 [−0.030, +0.017] | **−0.016 [−0.022, −0.011]** (interaction worse) |
| UAR(Local) − UAR(Additive) | −0.38 [−1.76, +1.18] | −0.49 [−1.31, +0.31] |
| Window: NLL(Additive) − NLL(Window) | +0.001 [−0.022, +0.022] | −0.006 [−0.012, −0.001] |
| Control: NLL(LateFusion) − NLL(Additive) | +0.018 [−0.006, +0.043] | — |
| EMAP: NLL(EMAP of RoleNet) − NLL(RoleNet) | −0.002 [−0.004, +0.000] | — |

Pooled scores (NLL / UAR; class-prior NLL from the training folds = 1.790):

| Arm | Neural | LR |
|---|---|---|
| Additive | 1.662 / 25.49 | 1.699 / 22.78 |
| Local | 1.670 / 25.11 | 1.715 / 22.28 |
| Window (neural = full RoleNet) | 1.661 / 25.98 | 1.705 / 22.65 |
| X-only (text) | 1.814 / 18.48 | — |
| Z-only (no text) | 1.672 / 27.02 | — |
| Late fusion | 1.681 / 25.86 | — |
| EMAP of RoleNet | 1.659 / 26.17 | — |

Notes:
- EMAP: projecting full RoleNet onto an additive text + rest form does not lower its score (NLL even slightly
  better), so no added forecast value of RoleNet's non-additive part has been seen.
- The text branch alone is worse than the class prior in NLL (1.814 vs 1.790) and is only 18.5 UAR. This agrees with
  G13 (removing text does not hurt). The X side of the test is weak, which limits what this gate can detect.
- LR: C was chosen at the lower end of the grid (0.003) in every fold and arm, so even stronger regularisation might be
  preferred. The interaction arms are worse at the same C.

## G23 results: negative-emotion separability, recognition vs forecasting (LR, train+val; test untouched)

**Reading (fixed rule): INVALID.** The positive control failed narrowly: VAL(RB) = 0.772 [0.733, 0.808], lower bound
≤ 0.75. No conclusion of the pre-registered type is drawn. Even setting validity aside, NEG(RB) = 0.692 [0.661, 0.725]
would fall in INTERMEDIATE (not ≥ 0.70 at the lower bound, not < 0.65 at the upper bound).

| Task | UAR | angry/sad | angry/disgust | disgust/sad | NEG [95%] | VAL [95%] |
|---|---|---|---|---|---|---|
| FC (I–III → B at IV) | 22.0 | 0.590 | 0.496 | 0.593 | 0.560 [0.526, 0.592] | 0.645 [0.608, 0.679] |
| RB (IV → B at IV) | 26.4 | 0.706 | 0.635 | 0.734 | 0.692 [0.661, 0.725] | 0.772 [0.733, 0.808] |
| RA (III → A at III) | 24.6 | 0.720 | 0.588 | 0.682 | 0.663 [0.631, 0.693] | 0.758 [0.717, 0.794] |
| FC+expr | 23.2 | 0.602 | 0.507 | 0.605 | 0.571 | 0.721 |
| RA+expr | 28.8 | 0.755 | 0.610 | 0.696 | 0.687 | 0.850 [0.823, 0.876] |
| RB-face / text / audio / scene | 23.8 / 17.7 / 16.2 / 21.1 | | | | 0.651 / 0.594 / 0.631 / 0.614 | 0.758 / 0.619 / 0.577 / 0.681 |

Descriptive (not a rule outcome): Δ = NEG(RB) − NEG(FC) = +0.132 [+0.107, +0.160]. With the same features and model,
the target's own clip-IV moment separates the negative emotions clearly better than the context does. Angry/disgust
is weak everywhere (≤ 0.64). Face and audio carry most of the clip-IV separation; HSEmotion adds mainly valence
(RA VAL 0.758 → 0.850). C was selected inside the grid for every task.

## G24 results: LLM appraisal of subtitles I–III vs reaction direction (gpt-4o-mini, LR, train+val; test untouched)

**Decision (fixed rule): STOP.** Δ_shift = NEG_shift(FC+app) − NEG_shift(FC) = +0.002 [−0.005, +0.010];
NEG_shift(APP) = 0.496 [0.458, 0.532] (lower bound not > 0.5).

| Arm | NEG all [95%] | NEG shift [95%] | UAR all / shift |
|---|---|---|---|
| FC | 0.560 [0.526, 0.592] | 0.538 [0.501, 0.571] | 22.0 / 19.2 |
| FC+app | 0.562 [0.530, 0.593] | 0.540 [0.503, 0.573] | 22.5 / 19.2 |
| APP | 0.514 [0.477, 0.549] | 0.496 [0.458, 0.532] | 15.5 / 14.8 |
| FC+llm (direct LLM forecast) | 0.565 [0.531, 0.597] | 0.534 [0.498, 0.566] | 22.4 / 19.2 |

Measurement checks (descriptive):
- All 2,421 responses parsed. The ratings are nearly degenerate: caused_by_other 0.97 ± 0.15, caused_by_responder
  0.00 ± 0.00, most other dimensions close to 0/1.
- event_valence carries valence (happy vs negative AUC 0.64 for A's own clip-III label, 0.62 for B), but the
  dimensions that should separate negative emotions do not, even for A's emotion in the very text the LLM read
  (angry/sad by other-causation 0.50, by control 0.51; sad/angry by loss 0.54; disgust/angry by norm violation 0.52).
- Inner-CV NLL improves slightly with appraisal or with the direct forecast (≈ 0.01), i.e. valence information only.
- Contamination probe: the LLM named House of Cards for 2/30 items.

Reading: with this zero-shot appraisal reading of the subtitles, no information about the direction of B's negative
reaction was found. The measurement itself is weak (degenerate ratings; no within-negative separation even for the
speaker of the read text), so a better appraisal measure is not excluded.

## MELD in the Hi-EF setting (M1–M3)

Order: M1 (Internet; downloads MELD.Raw ≈ 10 GB into `/tmp`, or attach a copy via `MELD_LOCAL`) → make its output a
dataset → M2 once per chunk (attach M1) → M3 (attach M1 and all M2 outputs). Speaker names are never an input; clip IV
is never processed; the Hi-EF features are re-implemented because the original extraction script is not available.
Local checks: M1 ran end to end on synthetic MELD videos with the real CLIP and AudioCLIP models (AudioCLIP weights load
with no missing keys) and a mocked face detector; M2's MELD-specific cells and its G8a clip listing ran (the G8a model
cells are unchanged); M3 ran end to end on mock M1/M2 outputs built from the real MELD transcripts.

**M3 results (MELD train → fit, dev → early stopping, test; 3 seeds; plain UAR; chance 14.29).** M1 used the Kaggle
copy of MELD.Raw (train/dev/test videos 9,989 / 1,112 / 2,615; MCIS 5,405 / 595 / 1,341; `dia125_utt3` unreadable).
The first M3 run gave constant RoleNet outputs (UAR 14.29 for every seed) because 3 VOI values (3 of 7,341 MCIS) were
NaN; M3 now zeroes non-finite inputs and was re-run, so **MELD test was read twice** (the first read produced no usable
RoleNet result; nothing else changed). Roles: A in III 99.8%, listener in III 81.2%, listener in I/II 53.6%.

| Split | RoleNet UAR | PaperBest UAR |
|---|---|---|
| train (fit) | not measured | not measured |
| val = MELD dev (best epoch per seed, selection-biased) | 20.88 (20.20 / 21.10 / 21.34) | 17.66 (16.78 / 18.79 / 17.40) |
| test, mean of seeds | 16.72 (15.98 / 17.64 / 16.55) | 15.56 (16.15 / 15.43 / 15.09) |
| test, 3-seed ensemble | 16.85 (WAR 39.30) | 15.50 (WAR 46.09) |

- RoleNet − PaperBest on test (ensemble UAR, bootstrap over seeds and dialogues): **+1.35 [−1.02, +4.24]**, not
  established. PaperBest predicts mostly neutral (recall 93%; all other classes 0 except angry 15.5%); RoleNet spreads
  its predictions (happy 17.9%, angry 17.6%) but disgust, sad, fear and surprise stay near 0.
- Ensemble NLL: PaperBest 1.543 vs RoleNet 1.695 (RoleNet − PaperBest +0.152 [+0.062, +0.395]): RoleNet is worse
  calibrated on MELD.
- Test labels for these numbers were rebuilt from the MELD test CSV with the M1 window rule; the per-seed UARs match
  `m3_test_per_seed.csv` exactly.

## G25 results: listener state vs reaction (train+val, analysis only; test untouched)

S = 1,070 MCIS with the listener visible in clip III and in clip I/II (694 shift, 376 mirror).

| Contrast (NLL difference, > 0 = second model better) | Subset | Δ [95%] |
|---|---|---|
| validity: prior − NOW | all S | +0.089 [+0.062, +0.116] |
| **change: AVG − SEP** (pre-registered decision) | **shift** | **+0.0125 [+0.0027, +0.0230]** |
| change: AVG − SEP | all S / mirror | +0.0088 [+0.0027, +0.0153] / +0.0020 [−0.0085, +0.0127] |
| recency: HIST − NOW | all S / shift | +0.035 [+0.012, +0.058] / +0.029 [+0.005, +0.052] |

**Decision (fixed rule): REACTION SIGNAL.**

**Caveat found after the run (my design flaw).** The pre-registered control `AVG` weights both observations equally, but
the clip-III observation is more informative (recency row), so `SEP` can beat `AVG` without any change information.
Post-hoc check (`analysis/g25_posthoc.py`, own grouped folds, not pre-registered): against a weighted average with the
weight tuned in inner CV (chosen w = 0.6–0.8 on clip III), SEP − WAVG on shift MCIS = +0.0076 [+0.0003, +0.0149] (all S
+0.0047 [−0.0015, +0.0103]); SEP vs clip III alone (`NOW`) +0.0046 [−0.0045, +0.0133]. SEP coefficients for clips III and
I/II point in the same direction for angry, fear, happy, neutral and sad (correlation 0.45–0.91), in opposite directions
for disgust (−0.08) and surprise (−0.28).

Reading: most of the listener information is B's state, weighted towards the most recent observation; a change
component beyond that is small and borderline on shift MCIS, possibly specific to surprise/disgust. A confirmation needs
the stricter control (tuned weighted average) pre-registered on new data (e.g. MELD via M1/M2).
- Descriptive: L's clip-III argmax expression equals B's clip-IV label in 29.5% of mirror and 22.5% of shift MCIS
  (A's label: 29.5% / 17.6%); expression-change size is similar in shift and mirror (0.574 vs 0.566).

## G26 results: does knowing the responder help? (5-fold CV, train+val, 10 seeds; test untouched)

**Decision (fixed rule): STOP — knowing the responder does not improve the forecast on Hi-EF.** Stage 1:
UAR(Oracle) − UAR(Heuristic) = −0.04 [−1.50, +1.85] (two-level bootstrap). The pointer was therefore not trained.

Responder coverage (clip IV, target/oracle only): B found in clip IV for 2,420 / 2,421 MCIS; matched to an identity in
clips I–III for 2,127 (88%); the rule's L is B in 993 of its 1,451 picks (68%; G6a measured 81% on 600 MCIS with a
different matching).

| Subset | n | UAR Heuristic / Oracle | NLL Heuristic / Oracle |
|---|---|---|---|
| all | 2,421 | 26.24 / 26.20 | 1.691 / 1.682 |
| B seen in I–III | 2,107 | 26.98 / 26.90 | 1.670 / 1.664 |
| B seen, rule picked someone else or nobody | 390 | 28.33 / 28.15 | 1.580 / 1.594 |
| B not seen in I–III | 314 | 21.20 / 21.29 | 1.836 / 1.804 |

Reading: giving RoleNet the true responder's face as the listener does not change the forecast, even on the 390 MCIS
where the rule picks the wrong person. Together with noRole (G14 +0.44, n.s.), the information RoleNet uses is in the
faces themselves, not in knowing whose face it is.

## G27 results: mention-aggregation pilot (5-fold CV, train+val, 3 seeds; test untouched)

**Decision (fixed rule): NO EVIDENCE THAT LATE AGGREGATION HELPS (this design).** Δ = NLL(B) − NLL(C) =
−0.0066 [−0.0225, +0.0096]. Scope: aggregation of contextualised mention representations, not event-role perspectives.

| Arm | NLL | UAR |
|---|---|---|
| A — whole-context candidate only | **1.6655** | **24.50** |
| B — early aggregation over mentions | 1.6749 | 23.77 |
| C — late aggregation over mentions | 1.6815 | 23.69 |
| (class prior, in-sample) | 1.776 | — |

| Contrast (> 0 = second arm better) | all | with mention | no mention |
|---|---|---|---|
| NLL(B) − NLL(C) | −0.0066 [−0.0225, +0.0096] | −0.0074 [−0.0241, +0.0094] | +0.024 [−0.043, +0.088] |
| NLL(A) − NLL(C) | −0.0160 [−0.0323, +0.0006] | −0.0167 [−0.0331, +0.0000] | +0.012 |
| NLL(A) − NLL(B) | −0.0094 [−0.0196, +0.0001] | −0.0093 [−0.0189, +0.0000] | −0.012 |

- 97.4% of windows have at least one mention (mean 3.8). Gates put 41% (B) / 30% (C) of the weight on the
  whole-context candidate.
- Adding mention candidates does not help either way; both mention arms are slightly worse than the whole-context arm
  (CIs touch 0). With frozen RoBERTa-large, the whole-context arm reaches NLL 1.666 / UAR 24.5, close to RoleNet's CV
  level (G20 full RoleNet NLL 1.661 / UAR 26.0) with a far simpler audio-visual vector.

## G28 results: masked-listener training (5-fold CV, train+val, 10 seeds; test untouched)

Same RoleNet; `R-mask` masks L's face features (before tokenisation) with p = 0.5 among applicable conditions. Actual
masked share of training draws 29.7% (C1 50.8%, C2 24.6%, C3 24.6% of masked draws); 15.0% of masked draws also had
the whole face branch dropped. Both arms selected by J_dev (NLL); best epoch ≈ 3 in both (G13 selected by dev UAR).
Eligible MCIS: C1 1,451; C2/C3 1,070. Role-rule checks 0/0.

| Quantity (mean of per-seed NLL; two-level bootstrap, shared draws) | Estimate [95% CI] |
|---|---|
| **Δ_rec** = NLL_std − NLL_mask, mean over C1–C3 (primary) | **+0.0037 [−0.0066, +0.0153]** |
| Δ_rec,1 (C1, no L) | +0.0127 [−0.0001, +0.0256] |
| Δ_rec,2 (C2, no L-III) | −0.0002 [−0.0109, +0.0115] |
| Δ_rec,3 (C3, no L history) | −0.0014 [−0.0113, +0.0096] |
| Δ_clean = NLL_mask − NLL_std on C0 | −0.0010 [−0.0090, +0.0078] |
| Sensitivity R-std (Ck − C0 on the Ck set): C1 / C2 / C3 | +0.0715 [+0.045, +0.099] / +0.0222 [+0.009, +0.036] / +0.0200 [+0.004, +0.038] |
| Sensitivity R-mask: C1 / C2 / C3 | +0.0575 / +0.0191 / +0.0181 |
| Recovery share Δ_rec,k / sensitivity_k: C1 / C2 / C3 | 0.18 [−0.00, 0.32] / −0.01 [−0.80, 0.49] / −0.07 [−1.22, 0.52] |

UAR (10-seed ensemble, descriptive): C0 25.72 vs 25.36; C1 23.44 vs 24.33; C2 26.38 vs 26.73; C3 26.32 vs 25.83 (R-std vs R-mask).

**Reading (fixed rule): CI of Δ_rec contains 0 → not shown that this policy helps; the data cannot tell whether the
drop under masking is mostly lost evidence or lack of adaptation.** No cost on the data as they are (Δ_clean ≈ 0).
Removing L's face features costs NLL in every condition (largest for all of L, C1), and masked training leaves most of
that drop in place (R-mask sensitivity on C1 still +0.058). The only hint is C1 (lower bound −0.0001, not the primary
estimand). Limitation: NLL-based selection stopped both arms at about epoch 3, which leaves masked training little
time to adapt.

**Auxiliary: presence flags.** 113 distinct presence patterns. Logistic on the 9 flags vs fold prior: ΔNLL −0.0047
[−0.0112, +0.0011]; with pairwise interactions −0.0030 [−0.0131, +0.0069] → no association with B's label shown.

## G29 results: context-conditioned evidence pooling (5-fold CV, train+val, 10 seeds; test untouched)

Step 0 (from the saved frame counts; the variance shares were printed in the run log only): share of (MCIS, clip)
cells with 0 / 1 / 2 / >2 kept frames — A 0.22 / 0.07 / 0.04 / 0.68, L 0.57 / 0.11 / 0.06 / 0.26, O 0.58 / 0.05 /
0.03 / 0.34. Cells P4 can reweight (≥ 2 frames): 46.9% of all cells, 86.2% of observed cells.

| Quantity (mean of per-seed NLL; seeds × episodes bootstrap, shared draws) | Estimate [95% CI] |
|---|---|
| **Δ_43 = NLL(P3) − NLL(P4)** | **+0.018 [−0.081, +0.103]** |
| **Δ_42 = NLL(P2) − NLL(P4)** | **+0.028 [−0.064, +0.106]** |
| NLL(P0) − NLL(P4) | −0.018 [−0.098, +0.060] |
| NLL(R) − NLL(P4) | +0.079 [−0.059, +0.235] |
| NLL(P1) − NLL(P0) | +0.021 [−0.060, +0.100] |

| Arm | NLL per-seed mean | NLL ensemble | UAR per-seed mean ± SD | UAR ensemble | s / run | peak MB |
|---|---|---|---|---|---|---|
| P0 FramePool | 1.897 | 1.679 | 24.63 ± 0.73 | 26.88 | 8.1 | 858 |
| P1 mean | 1.918 | 1.687 | 24.46 ± 0.55 | 25.68 | 7.8 | 858 |
| P2 MLP scorer | 1.943 | 1.690 | 24.62 ± 0.78 | 26.61 | 8.7 | 858 |
| P3 context after | 1.933 | 1.683 | 24.40 ± 0.42 | 26.63 | 9.9 | 860 |
| P4 context in scores | 1.915 | 1.685 | 24.76 ± 0.77 | 26.33 | 10.4 | 861 |
| R frame tokens | 1.994 | 1.733 | 22.81 ± 1.08 | 25.23 | 20.4 | 1,456 |

**Reading (fixed rule): NOT CONFIRMED** — neither primary CI is above 0. P4 is not distinguishable from the current
RoleNet (P0), so it cannot be called an improvement. No pooling variant separates from FramePool; the uncompressed
frame-token reference R is the weakest arm (descriptively) at 2.5× the time and 1.7× the memory.

Power caveat: with checkpoints chosen by dev UAR, per-seed probabilities are poorly calibrated (per-seed NLL ≈ 1.90–1.99
vs ensemble ≈ 1.68–1.73) and vary strongly across folds and seeds, so the CIs are about ±0.09 NLL, roughly ten times
wider than in G28 (NLL selection). Differences of a few hundredths of NLL cannot be detected with this estimand; the
estimand was fixed before running and is not replaced after the fact.

## G31 results: step 1, mirror/rest mixture vs additive logits (linear, train+val; test untouched)

| | role | noRole (descriptive) |
|---|---|---|
| NLL mixture / additive / plain LR / copy p_mirror | 1.7087 / 1.7056 / 1.9215 / 1.8816 | 1.7126 / 1.7108 / 1.8156 / 1.7770 |
| **ΔNLL = NLL(additive) − NLL(mixture)** | **−0.0031 [−0.0061, −0.0001]** | −0.0018 [−0.0043, +0.0007] |
| **AUC(g → Y_B = Y_A)** | **0.526 [0.493, 0.560]** | 0.513 [0.478, 0.545] |
| p_mirror UAR on A's label (fold mean) | 27.1 | 26.1 |

**Fixed rule: NOT PASSED** (neither condition holds; the mixture is slightly worse than additive logits).
Diagnostics that limit what this shows: the gate collapsed to g ≈ 1e-8 in every fold (the mixture reduced to p_rest),
the L2 strength of the mixture/additive heads was the largest grid value (0.1) in every fold, and p_mirror's C was the
smallest grid value (0.01) in every fold, so p_mirror was strongly flattened. In addition, p_rest contained clip-III
text and audio (A's own turn), so the design compared "A" with "everything including A" rather than A with B. G31 is
therefore uninformative about the mirror/own-state mechanism itself; it only shows that this linear mixture did not help.

## G33a results: temporal structure audit (5-fold CV, train+val, 10 seeds; test untouched)

Invariance (bag definition): max |Δ logit| under clip permutations 1.4e-6 (bag12: I↔II; bag123: three permutations),
below the 1e-5 tolerance in every fold and seed → both bags valid. Checkpoint by inner-dev UAR; temperature fitted on
inner dev (median T 1.73–1.85, i.e. the UAR-selected checkpoints are over-confident; calibration lowers the held-out
NLL from 1.89–1.93 to 1.70–1.71).

| Quantity (mean per-seed calibrated NLL, held-out; seeds × episodes bootstrap) | Estimate [95% CI] |
|---|---|
| NLL ordered / bag12 / bag123 / ordered + swap | 1.7018 / 1.7076 / 1.7097 / 1.7032 |
| **D_reliance** = NLL(ordered + swap) − NLL(ordered) | **+0.0014 [−0.0007, +0.0035]** |
| **U_order** = NLL(bag12) − NLL(ordered) | **+0.0058 [−0.0101, +0.0219]** (positive in 4/5 folds; fold 4 −0.014) |
| **U_boundary** = NLL(bag123) − NLL(bag12) | **+0.0021 [−0.0108, +0.0180]** |

Diagnostics: swapping I↔II changes the ordered model's predicted distribution by a mean total variation of 0.019
(95th percentile 0.062) and flips 4.8% of argmax predictions. UAR (10-seed ensemble): ordered 26.25, bag12 26.46,
bag123 26.67 (descriptive).

**Reading (fixed rule): history helps as context (G14 +2.13); no evidence for temporal modelling** — neither the
order of the past clips nor an explicit current-vs-past designation is shown to be useful, and the trained model
barely relies on I/II order. Conclusions concern explicit clip identity only (roles defined from clip III remain an
implicit marker). The CI of U_order spans about 0.032 NLL, so order effects below roughly 0.016 NLL cannot be
detected with this design.

## G33b results: relational bias study (5-fold CV, train+val, 10 seeds, R0 loaded from G33a; test untouched)

This run used the original configuration (10 seeds × 5 folds; R0 = G33a `ordered`, manifest identical).

| Arm | calibrated NLL | raw NLL | UAR ensemble | UAR seed mean |
|---|---|---|---|---|
| R0 | 1.7018 | 1.8912 | 26.35 | 24.52 |
| R-full | 1.7027 | 1.9269 | 26.35 | 24.71 |
| R−time | 1.7073 | 1.9096 | 25.64 | 24.41 |
| R−person | 1.7043 | 1.9070 | 26.40 | 24.57 |
| R−evidence | 1.7047 | 1.9297 | 26.51 | 24.67 |
| R-random | 1.7083 | 1.9201 | 26.36 | 24.58 |

| Contrast (calibrated NLL; seeds × episodes bootstrap) | Estimate [95% CI] |
|---|---|
| **NLL(R0) − NLL(R-full)** | **−0.0009 [−0.0165, +0.0136]** |
| **NLL(R-random) − NLL(R-full)** | **+0.0056 [−0.0067, +0.0185]** |
| NLL(R−time / R−person / R−evidence) − NLL(R-full) (descriptive) | +0.0047 / +0.0017 / +0.0021, all CIs contain 0 |

**Reading (fixed rule): not unlocked — no evidence for relational inductive bias; no method claim.**

Optimisation diagnostics: bias gradients stayed non-zero (R-full grad norm 0.012 → 0.036); the learned biases are small
(|b| mean ≈ 0.025–0.03, max ≈ 0.12–0.15; median maximum within-head spread 0.22, below the ≈ 0.3 reading aid).
**Layer-2 biases are exactly zero, by construction, not by an implementation error.** Verified on the code: they are
in the forward pass and in the optimiser, but the prediction reads only the query token after the last layer, and the
query row carries a single type per family (query-involved / non-face / other), which softmax cancels. Their gradient is
≈ 1e-8 and setting them to large values changes the logits by < 1e-6. The relational biases therefore act in layer 1
only (layer-1 means are about twice the reported layer-averaged |b| and spread means). This is a property of the
pre-registered ontology (no query-specific relation types), so G33b remains a valid test of that design; a version with
query-target relations would be a new hypothesis.

## G35 results: group-level EMAP of RoleNet (5 seeds × 3 folds, train+val; test untouched)

| Quantity (mean per-seed calibrated NLL, held-out; seeds × episodes bootstrap) | Estimate [95% CI] |
|---|---|
| NLL full / EMAP3 / EMAP L\|rest / EMAP H\|rest / EMAP E\|rest | 1.7006 / 1.7112 / 1.7053 / 1.7064 / 1.7061 |
| **D_int** = NLL(EMAP3) − NLL(full) | **+0.0107 [+0.0051, +0.0157]** |
| NLL(EMAP L\|rest) − NLL(full) | +0.0048 [+0.0005, +0.0097] |
| NLL(EMAP H\|rest) − NLL(full) | +0.0059 [+0.0011, +0.0098] |
| NLL(EMAP E\|rest) − NLL(full) | +0.0055 [+0.0008, +0.0096] |

Interaction share of the row-centred held-out logit variance: 5.3% (EMAP3), 2.7–3.4% (one group vs rest).
UAR (seed ensemble, descriptive): full 24.75, EMAP3 24.48. Median temperature: full 1.62, EMAP3 1.78.

**Reading (fixed rule): INTERACTIONS** — the best additive (no-interaction) approximation of the trained RoleNet
over listener / history / clip-III event is worse by 0.011 NLL, and each group interacts with the rest. The
interactions are small (about 5% of the logit variance) and are consistent with G20 (no value from a text × rest
interaction): the non-additive part lies among the face / context groups, not in text.

## G36 / G36b results: class resampling (development CV, then a second, disclosed test read)

**Part A — development CV** (45 episodes, 5 folds, 10 seeds, plain, seed ensembles):

| Arm | UAR | WAR | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|---|---|
| RoleNet (uniform) | 26.10 | 38.29 | 46.2 | 0.9 | 0.0 | 60.0 | 54.0 | 21.2 | 0.4 |
| Up-sqrt (∝ n_c^−0.5) | **26.95** | 37.88 | 37.7 | 12.3 | 0.0 | 58.2 | 53.3 | 24.2 | 3.0 |
| Up-bal (∝ n_c^−1) | 25.06 | 33.17 | 29.9 | 17.3 | 0.0 | 52.0 | 38.2 | 27.6 | 10.6 |

Up-sqrt − RoleNet +0.86 [−0.94, +2.18]; Up-bal − RoleNet −1.03 [−3.22, +1.24]. Fixed rule: **Up-sqrt selected, test read.**

**Part B (G36b) — second test read** (G10 protocol, 5 seeds; baseline and RoleNet = saved G10 predictions):

| Model | UAR | WAR | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|---|---|
| Up-sqrt | **25.64** | 33.74 | 20.8 | 22.5 | 0.0 | 47.0 | 49.4 | 31.7 | 8.1 |
| Baseline (PaperBest) | 21.40 | 32.52 | 44.2 | 0.0 | 0.0 | 54.0 | 48.3 | 3.3 | 0.0 |
| RoleNet (G10) | 25.24 | 36.43 | 48.1 | 2.5 | 0.0 | 50.0 | 52.8 | 23.3 | 0.0 |

| Contrast | ΔUAR [95% CI] | ΔWAR [95% CI] | Episodes won (UAR) |
|---|---|---|---|
| Up-sqrt − Baseline | **+4.24 [+2.02, +6.81]** | +1.22 [−1.69, +4.66] | 6/8 |
| Up-sqrt − RoleNet (G10) | +0.40 [−1.98, +1.93] | −2.69 [−5.27, +0.01] | 3/8 |

Per-seed test UAR 22.84–24.27. Reading: resampling spreads the forecasts over the rare classes (disgust 2.5 → 22.5,
surprise 0 → 8.1, sad 23.3 → 31.7) at the cost of angry (48.1 → 20.8) and WAR; fear stays at 0. Against the
baseline the gain is +4.24 UAR; against the preregistered RoleNet it is not significant. This is the second read of
the test split for RoleNet and is reported as such. Files: `results/g36/`.

## G38 results: class resampling with a no-regression rule (development CV; test not read)

| Arm (CV, 10-seed ensembles) | UAR | WAR | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|---|---|
| RoleNet (uniform) | 26.10 | 38.29 | 46.2 | 0.9 | 0.0 | 60.0 | 54.0 | 21.2 | 0.4 |
| Up-quarter (∝ n_c^−0.25) | 26.17 | 37.26 | 34.4 | 8.6 | 0.0 | 61.7 | 50.8 | 25.1 | 2.5 |
| Tail-median (≤ 3×) | 25.96 | 36.27 | 39.3 | 16.4 | 0.0 | 58.0 | 48.8 | 15.9 | 3.4 |

Fixed rule: no arm eligible (Up-quarter angry −11.8; Tail-median angry −6.9, sad −5.3) → **no test read**.
Decision (authors): the paper keeps the **preregistered RoleNet (G10)**; G36b (Up-sqrt) is reported only as an analysis
of the rare-class / angry trade-off; fear and surprise (test recall 0) are future work.
