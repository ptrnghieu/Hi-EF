# Hi-EF / RoleNet: synthesis of the evidence (state after G20)

Purpose: one place that separates what was tested and failed (and in which scope), what was set aside because it is
equivalent to a known method, what could not be tested, and what is still open. It is written to screen new research
candidates. Sources: `README.md` (G1–G20) and the exploratory analyses that were run during the discussions but not
yet written into the README (marked **[chat]**).

## 0. Scope decisions (fixed by the authors)

- Hi-EF is the main dataset. **MELD** is used to check generality. When a forecasting task is built on MELD, the
  identity of the future speaker is not given unless the corresponding information exists in Hi-EF. Any reused
  checkpoint must be checked for overlap between its training data and the evaluation data.
- Features may be **re-extracted** and the text/audio representations replaced, but only to serve a specific research
  question. A better encoder on its own does not count as novelty.
- The target contribution is a **new method** that starts from a problem with evidence and a theoretical basis. The
  existing results define the problem and bound the claims. A paper made of negative results is not the goal. A new
  mathematical tool is not required if the method contribution is clear.
- Constraints: no new human labels; clip IV is never an input at inference; B's identity is not given as an input;
  no assumption that only two people take part.

## 1. Protocol status: test access

The test split (8 episodes, 409 MCIS) is **not** untouched. Recorded accesses:

| When | What | Model family |
|---|---|---|
| G4 run 1 | single preregistered test run; invalid for the trajectory arms (cache-key bug), valid for the other rows | RtT / trajectory forecasters, B1 |
| G4 re-run | announced in the README after the bug fix; **its outcome is not recorded in the repo** | same |
| G5 | 5-fold CV over all 53 episodes, so the folds evaluate on test episodes; prepared to run after G4; **whether it was run is not recorded in the repo** | same |
| G10 | single preregistered test run (`PREREG_G10_TEST.md`) | RoleNet, PaperBest, B1, LateFusion, RoleNet-noRole |

After G10 the main reporting metric was switched from LA-scored UAR (the preregistered primary, not confirmed:
+3.77 [−7.49, +10.12]) to plain UAR (benchmark standard, preregistered as descriptive). Both are reported. All
experiments after G10 (G11–G20) use train+val only.

Consequence for new work: any new method must be selected on train+val. A later test number must be described as a
further use of a test split that has already been read for model selection decisions at G4/G10.

## 2. Feature pipeline audit (original Hi-EF code in this repo)

| Stream | Original pipeline (from `model_sirv_first.py`, `audioclip.py`, `esresnet/`) | Cached tensor (`hi-ef-features-v2`) |
|---|---|---|
| Faces / frames | CLIP ViT-B/32 image encoder, 512-d per frame | `face_features`, `ori_features` [16, 512] |
| Text | CLIP text encoder on the clip transcript (`clip.tokenize`, 77-token context, captions-style pretraining) | `text_feature` [512] |
| Audio | ESResNeXt-FBSP from AudioCLIP (`ESRNXFBSP.pt`), `num_classes = 527`: AudioSet class outputs, then a linear map 527→512 in the model | `audio_feature` [527], `audio_found` |

- The cached features match these dimensions. The script that produced `hi-ef-features-v2` is **not in the repo**, so
  freezing, checkpoint and truncation details are inferred, not verified.
- Neither text nor audio is a representation made for emotion or prosody. The audio feature is a 527-way sound-event
  vector, not a speech/paralinguistic embedding.
- Our own extraction (G8a): HSEmotion per face (PCA 128), 18 geometric features, ArcFace identities clustered per
  window (cosine 0.45), voice embeddings for who-speaks cues, mouth/audio-envelope sync.
- Consistent with this: removing text does not hurt RoleNet (G13); text alone is worse than the class prior in NLL
  (G20, 1.814 vs 1.790).

## 3. Established findings (train+val CV unless stated)

| # | Finding | Evidence | Scope / caveat |
|---|---|---|---|
| F1 | RoleNet (role-structured face tokens + context) beats B1 and late fusion | CV plain +4.44 [+2.64, +6.08] vs B1; test plain +3.84 [+1.24, +6.76] vs PaperBest | design informed by the same episodes (CV exploratory) |
| F2 | B's own face is the main extra signal | G11 minus-L +2.52 [+1.35, +3.63]; G14 +2.04 (10 seeds) | about half of it is a generic ablation cost (G11 DiD +1.25, n.s.) |
| F3 | Earlier clips matter | G13 T-clipIIIonly below all 220 same-model ensembles; B's I/II appearances +2.76 [+0.54, +5.10] when B is absent in III | history enters additively (F7) |
| F4 | Bystanders carry no detectable specific information | G11 DiD +0.24 [−2.59, +3.01] | face channel only |
| F5 | Who-is-who is weakly used | G13: role embedding, clip embedding, absent token all replaceable; G8b noRole +1.17 [−0.01, +2.29] | assignment is a heuristic (L = B in 81.4% vs a clip-IV proxy) |
| F6 | No continuous-time decay; listening vs speaking does not matter | G16 T1 −0.017; G17 P1 −0.007 | discrete clip steps only |
| F7 | History adds information but does not change the response rule | **[chat]** hysteresis: additive history +0.149 bits; no A × history interaction at any C | LR on current features |
| F8 | B shows label inertia | **[chat]** β_B − β_other = +0.39 [+0.03, +0.73] at clip II; G15 G2a directional | only where B's earlier label exists (B spoke) |
| F9 | Gold earlier B labels are very informative but not recoverable | **[chat]** on the 26% of MCIS with an earlier labelled B turn: copy-B (gold) 39.07 vs RoleNet 27.98 UAR; needs a recognizer ≥ 50–60%, current ≈ 31% | G9: a perfect "B spoke" pointer adds nothing inside RoleNet |
| F10 | Clip III adds mainly happy-vs-other separability (RoleNet only) | G19 + PIC: happy minus other gain RN +0.036 [+0.016, +0.059], LR +0.007 [−0.009, +0.024] | not replicated across families |
| F11 | No forecast value of text × audio-visual non-additivity | G20: Δ_NLL neural −0.008 [−0.030, +0.017], LR −0.016 [−0.022, −0.011]; EMAP of RoleNet −0.002 | weak text representation (§2) |
| F12 | Seed noise is large | 3-seed ensembles vary by ±0.4–0.56 UAR (SD); G13 Full vs Full-reseed 1.17 | contrasts below ~1.5 UAR need 10 seeds |

## 4. Directions tried, by status

### 4a. Tested and failed, within a stated scope

| Direction | Test | Scope of the failure |
|---|---|---|
| Recognize-then-transition (better A recognition) | G1–G5 | recognizers of A on CLIP/AudioCLIP features; trajectory forecaster |
| Speaker-role tagging of context speech | G9 (RoleNet+, ORACLE flat) | inside RoleNet |
| Additive clip-ordered role paths (CS-RoleNet) | G12 | — |
| Strict surrogation (bystanders) | G11 | face channel |
| Continuous-time affect dynamics (OU / decay) | G15–G17 | clip-level steps of Hi-EF |
| History-dependent response (hysteresis) | **[chat]** | LR, current features |
| Text × AV synergy | G20 | current text/audio features, neural + LR |
| Direction-wise information added by clip III | G19 + PIC | two predictor families disagree |

### 4b. Set aside because they reduce to a known method

| Proposal | Reduces to |
|---|---|
| Max-entropy pairwise-preserving contrast Q₂ + density-ratio discriminator | additive-logit model + residual interaction (GA²M / EMAP / PID estimators); the discriminator is a noisier estimator of the same quantity |
| Composition-consistent reduced models (missing participants) | automatic for marginals of any joint model; enforcing it needs a latent-variable / state-space model (VAEAC, MMIN, neural processes) |
| PIC-weighted auxiliary loss | proper weighted Brier score; mandatory control CE + happy/non-happy auxiliary |

Note: reducing to a known form is not by itself a reason to reject. What must still be compared is the assumptions,
the estimator, the supervision it needs and the guarantees. These three were set aside mainly together with the
evidence in 4a/§3 (F7, F4, F11).

### 4c. Could not be tested with the available information

| Direction | Missing |
|---|---|
| Appraisal / goal-state of B (z_B) | human labels (no labelling staff) |
| Two-step recognize-B-then-forecast | a B recognizer ≥ 50–60% on Hi-EF clips (F9) |
| Any claim about text or prosody content | an emotion/paralinguistic text and audio representation (§2) |
| Effect of identity-assignment errors | ground truth of who is who (only a clip-IV proxy, evaluation-only) |

### 4d. Still open, no sufficiently new solution yet

| Problem | Evidence for the problem | What is missing |
|---|---|---|
| Forecasting under uncertain person–observation assignment (G22 proposal) | no identity ground truth; L heuristic 81.4%; F2 vs F5 tension (B's face matters, whose face is weakly used) | evidence that decisions flip under plausible alternative assignments; a method beyond enumeration (Π is probably small on Hi-EF) |
| Exploiting B's label inertia without gold labels (F8/F9) | large oracle gap (39.07 vs 27.98 on 26% of MCIS) | a recognizer good enough on Hi-EF, or a way to use the inertia without explicit recognition |
| Missing-participant memory (G21 proposal) | F3 supports history; F7, F4, F5 argue against pattern-dependent rules and bystander influence | a premise not already contradicted |

## 5. Screening checklist for new candidates

Each candidate must state, before any experiment:

1. Problem statement: the specific failure it addresses, tied to a row of §3 or §4.
2. Grounded knowledge: the theory and the prediction it makes.
3. Proposed solution: the learning/estimation principle.
4. Closest prior work: what it does, and the exact remaining difference in assumptions, estimator, supervision or
   guarantees.
5. Testability without new labels: which data separate the new solution from the closest prior.
6. "If the proposal is replaced by the closest method with reasonable tuning, what is lost?"
7. Which finding in §3 it would contradict, and why that finding does not already rule it out.
