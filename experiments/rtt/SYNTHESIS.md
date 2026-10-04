# Hi-EF / RoleNet: synthesis of the evidence (state after G20)

Purpose: one place that separates what was tested and failed (and in which scope), what was set aside because it is
equivalent to a known method, what could not be tested, and what is still open. It is written to screen new research
candidates. Sources: `README.md` (G1–G20) and the exploratory analyses that were run during the discussions but not
yet written into the README (marked **[chat]**).

## 0. Scope decisions (fixed by the authors)

- Hi-EF is the main dataset. **MELD** is used to check generality. When a forecasting task is built on MELD, the
  identity of the future speaker is not given unless the corresponding information exists in Hi-EF. Any reused
  checkpoint must be checked for overlap between its training data and the evaluation data.
- **Comparisons with the paper:** report RoleNet against the paper's best baseline (`PaperBest`) only. B1 and other
  internal baselines are not added to comparisons or tables unless the authors ask for them.
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
| G4 re-run | announced in the README after the bug fix; **execution not verified** (no record in the repo or in the conversation history) | same |
| G5 | 5-fold CV over all 53 episodes, so the folds evaluate on test episodes; prepared to run after G4; **execution not verified** (no record in the repo or in the conversation history) | same |
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

- The cached features match these dimensions. The script that produced `hi-ef-features-v2` was **not found** (not in
  the repo, not located by the authors), so freezing, checkpoint and truncation details are inferred, not verified.
- Neither text nor audio is a representation made for emotion or prosody. The audio feature is a 527-way sound-event
  vector, not a speech/paralinguistic embedding.
- Our own extraction (G8a): HSEmotion per face (PCA 128), 18 geometric features, ArcFace identities clustered per
  window (cosine 0.45), voice embeddings for who-speaks cues, mouth/audio-envelope sync.
- Observed with these features: removing text does not hurt RoleNet (G13); text alone is worse than the class prior
  in NLL (G20, 1.814 vs 1.790). This does not show that the encoder is the cause.

**Annotation columns not used so far** (`annotation.csv`, labelled clips only: every clip III/IV, 30.8% of clip I,
40.5% of clip II):

| Column | Content | Values (all labelled clips) |
|---|---|---|
| 2 | upper-face action description | e.g. brow lower 1,220, outer brow raiser 1,054, "Nan" 781 |
| 3 | lower-face action description | e.g. lower lip depressor & lips part 1,772, lip stretcher & lips part 1,448 |
| 4 | scene | work 2,360, daily 1,899, social 463, entertainment 61 |
| 5 | interaction polarity | negative 2,234, positive 1,318, neutral 1,231 (used only as an auxiliary target in G2/G3) |
| 6 | intensity | weak 3,252, powerful 1,531 |
| 8 | label uncertainty (1 certain – 3 uncertain) | 3,587 / 559 / 637; level 3 is frequent for sad (177/699), disgust (112/416), fear (35/64) (used only as clip-IV weights in G3 `certw`) |

These are human annotations: they may serve as training-time supervision; using them as inputs for clips I–III would
be the same kind of oracle as the gold A label and is outside the protocol.

## 3. Established findings (train+val CV unless stated)

| # | Finding | Evidence | Scope / caveat |
|---|---|---|---|
| F1 | RoleNet (role-structured face tokens + context) beats B1 and late fusion | CV plain +4.44 [+2.64, +6.08] vs B1; test plain +3.84 [+1.24, +6.76] vs PaperBest | design informed by the same episodes (CV exploratory) |
| F2 | B's own face is the main extra signal | G11 minus-L +2.52 [+1.35, +3.63]; G14 +2.04 (10 seeds) | about half of it is a generic ablation cost (G11 DiD +1.25, n.s.) |
| F3 | Earlier clips matter | G13 T-clipIIIonly below all 220 same-model ensembles; B's I/II appearances +2.76 [+0.54, +5.10] when B is absent in III | history enters additively (F7) |
| F4 | Bystanders carry no detectable specific information | G11 DiD +0.24 [−2.59, +3.01] | face channel only |
| F5 | No benefit established for the role *representations* tried | G13: removing role embedding, clip embedding or the absent token stays within noise; G8b noRole +1.17 [−0.01, +2.29] | removing the embedding keeps the earlier selection and grouping of observations into role tokens; non-significant differences do not show the models are equivalent; L = B in 81.4% vs a clip-IV proxy |
| F6 | No difference detected for elapsed time or for listening vs speaking observations | G16 T1 −0.017 [−0.115, +0.078]; G17 P1 −0.007 [−0.131, +0.120] | only the contrasts and time scales tested; does not show that the true dynamics are discrete or that the two kinds of observation are equivalent |
| F7 | History adds information; no benefit detected from an A × history interaction | **[chat]** hysteresis: additive history +0.149 bits; the interaction model adds nothing at any C | LR on current features only |
| F8 | B shows label inertia | **[chat]** β_B − β_other = +0.39 [+0.03, +0.73] at clip II; G15 G2a directional | only where B's earlier label exists (B spoke) |
| F9 | Gold earlier B labels are very informative; the current recognizer does not recover them | **[chat]** on the 26% of MCIS with an earlier labelled B turn: copy-B (gold) 39.07 vs RoleNet 27.98 UAR; current clip recognizer ≈ 31% | the "≥ 50–60%" figure is the threshold of one error simulation of one pipeline, not a general condition (recognizers with equal UAR can give different forecasts); G9: a perfect "B spoke" pointer adds nothing inside RoleNet |
| F10 | Clip III adds mainly happy-vs-other separability (RoleNet only) | G19 + PIC: happy minus other gain RN +0.036 [+0.016, +0.059], LR +0.007 [−0.009, +0.024] | not replicated across families |
| F11 | No forecast value of text × audio-visual non-additivity | G20: Δ_NLL neural −0.008 [−0.030, +0.017], LR −0.016 [−0.022, −0.011]; EMAP of RoleNet −0.002 | weak text representation (§2) |
| F12 | Seed noise is large | 3-seed ensembles vary by ±0.4–0.56 UAR (SD); G13 Full vs Full-reseed 1.17 | contrasts below ~1.5 UAR need 10 seeds |
| F13 | Target-label ambiguity does not explain the near-chance negative distinctions | **[chat]** `analysis/target_ambiguity.py` on G19 OOF (RoleNet I–III, 10 seeds): 25.8% of clip-IV labels are borderline/uncertain (sad 41%, disgust 47%, fear 56%, neutral 15%); NLL 1.62 / 1.77 / 2.00 and UAR 26.8 / 26.1 / 20.7 for levels 1 / 2 / 3, but within-class NLL differences are mixed (angry +0.21, happy +0.27, sad +0.15, disgust −0.05, neutral −0.33); pair AUC on certain targets only: angry/sad +0.04 [+0.01, +0.07], angry/disgust and disgust/sad no change; forecast entropy hardly detects an uncertain target (AUC 0.54); an uncertain clip-III label raises the chance of an uncertain clip-IV label (39% vs 22%) | uncertainty levels are defined only as Certain / Borderline / Uncertain ("multi-modal uncertainty", 5 annotators + a reviewing professor; guideline in the paper's appendix, not available); uncertain labels are mostly weak intensity (91% at level 3) with more emotion–polarity inconsistency (22% vs 8.5%); G3 certainty weighting of the clip-IV loss: B1+certw 23.15 ± 1.75 vs B1 21.65 ± 1.48 seed-mean UAR, no gain with trajectories |
| F14 | Forecasting success is mostly mirroring; shifts are hardly forecast | **[chat]** `analysis/mirror_vs_shift.py` on G14 OOF (Full, 10 seeds): B's label equals A's clip-III label in 35.5% of MCIS (diagonal lift 1.85–2.8; off-diagonal lifts weak, largest neutral→surprise 1.51, surprise→neutral 1.44); accuracy on mirror samples 56.9% vs shift samples 27.7%; on shifts the model still predicts A's label 29%; the listener-token gain is concentrated on mirror samples (mean log p +0.136 with listener visible) vs shifts (+0.058, and −0.028 without listener); whether a shift happens is partly predictable (AUC 0.70); on shifts, recall of disgust 1.6%, surprise 2.8%, fear 0%, sad 17% | stratified by A's gold label (analysis only, not an input); same picture against the clip-II speaker's label (57.0% vs 28.2%, 990 MCIS) |
| F15 | The target moment separates negative emotions better than the context, but the pre-registered test is INVALID | G23 (LR, same features): NEG at clip IV 0.692 [0.661, 0.725] vs forecast 0.560 [0.526, 0.592], Δ +0.132 [+0.107, +0.160]; positive control VAL(RB) 0.772 [0.733, 0.808] missed the 0.75 validity bound | descriptive only; NEG(RB) would be INTERMEDIATE anyway; angry/disgust weak everywhere |
| F16 | Label-level findings replicate on MELD | **[chat]** `meld/labels_replication.py` (gold labels; 6,001 train+dev / 1,341 test windows; an earlier version merged train/dev dialogues sharing an id, fixed, test numbers unchanged, B ≠ A): mirror rate 34.7% (Hi-EF 35.5%); off-diagonal transition lifts weak except a few (disgust→disgust 3.85, fear→disgust 2.73); B's own earlier label beats A's label: copy-B − copy-A accuracy +0.076 [+0.037, +0.117] on the 70.8% of windows where B spoke earlier (B's turn is further back than A's); no A × history interaction (test NLL 1.4626 additive vs 1.4645) | labels only, no features; MELD = one sitcom (Friends), speaker identities known |
| F17 | MELD text-only feasibility of P1/P3/P5/P6: all rules negative, but the assay failed | `meld/feasibility_text.py` (rules committed before running; `meld/feasibility_text_v1.json`): P6 NOT FEASIBLE (assignment tags raise NLL by 0.018 [0.013, 0.023]; flip rate under a wrong assignment 10.5%), P1 NOT FEASIBLE (relational terms +0.020 NLL), P5 NOT FEASIBLE (DiD −0.051), P3 STOP (Spearman LR–GBM 0.46) | **not informative**: the base text forecaster (mpnet embeddings + LR) is no better than the class prior (test NLL 1.525–1.566 vs prior 1.526; UAR ≈ 15) while a gold-label model reaches 1.461, so added structure only adds variance; a validity check "base model beats the prior" was not pre-registered and should have been |
| F18 | LLM appraisal of the subtitles does not inform the direction of B's negative reaction | G24 STOP: Δ_shift +0.002 [−0.005, +0.010], APP alone 0.496 on shifts | ratings nearly degenerate (gpt-4o-mini, zero-shot); appraisal dimensions do not separate even A's own negative emotions in the read text; only valence is captured |
| F19 | Listener information is mostly B's recent state; a change (reaction) component is small and borderline | G25: rule outcome REACTION SIGNAL (AVG − SEP on shifts +0.0125 [+0.0027, +0.0230]), but against a tuned weighted average only +0.0076 [+0.0003, +0.0149] (post hoc) and SEP ≈ clip III alone; recency clear (HIST − NOW +0.035 [+0.012, +0.058]) | the pre-registered control was too weak (equal weights); opposite-sign coefficients only for surprise and disgust |
| F20 | Knowing who the responder is does not improve the forecast | G26 stage 1: Oracle − Heuristic −0.04 [−1.50, +1.85] UAR (10 seeds); no gain on the 390 MCIS where the rule picks the wrong person (28.15 vs 28.33) | responder identity from clip IV (train/oracle only); together with noRole n.s., the role/identity component of RoleNet is not what makes it work |
| F21 | Keeping mention representations separate (late aggregation) does not help; mentions add nothing over the whole context | G27: NLL(B) − NLL(C) −0.0066 [−0.0225, +0.0096]; whole-context arm A best (A − C −0.016 [−0.032, +0.001]) | frozen RoBERTa-large, pronoun/NER mentions, no event roles or coreference; result limited to this design |
| F22 | Training with masked listener features was not shown to improve the forecast under that masking | G28: Δ_rec +0.0037 [−0.0066, +0.0153] NLL (10 seeds); C1 alone +0.0127 [−0.0001, +0.0256]; no cost on C0 (−0.0010); R-std loses +0.072 NLL when all L face features are removed; presence flags alone do not forecast the label (ΔNLL −0.0047 [−0.0112, +0.0011]) | policy matched to the evaluation masking, L face branch only, masking after role assignment; NLL selection stopped at ≈ epoch 3; cannot separate lost evidence from lack of adaptation |
| F23 | Letting the context choose face frames before compression is not shown to help | G29: NLL(P3) − NLL(P4) +0.018 [−0.081, +0.103], NLL(P2) − NLL(P4) +0.028 [−0.064, +0.106]; P4 vs RoleNet −0.018 [−0.098, +0.060]; no pooling variant separates from FramePool; frame tokens without pooling (R) weakest | UAR checkpoint selection makes per-seed NLL noisy (CIs ±0.09), so the test has low power for small effects; L cells: 57% empty, 11% single frame |
| F24 | A linear mirror/rest mixture did not beat additive logits | G31: ΔNLL(add − mix) −0.0031 [−0.0061, −0.0001]; AUC(g → mirror) 0.526 [0.493, 0.560] | gate collapsed to ≈ 0 in every fold, regularisation at the grid edge, p_mirror flattened, p_rest contained A's clip-III turn → uninformative about the mechanism |
| F25 | Explicit clip order / current-vs-past designation is not shown to matter; history acts as context | G33a: U_order +0.0058 [−0.0101, +0.0219], U_boundary +0.0021 [−0.0108, +0.0180], D_reliance +0.0014 [−0.0007, +0.0035] (calibrated NLL, 10 seeds; bags verified invariant to 1.4e-6) | explicit clip identity only (roles from clip III stay implicit); effects below ~0.016 NLL not detectable; the earlier E-noClipEmb test leaked clip identity and is superseded |
| F26 | Factorial TIME × PERSON × EVIDENCE attention biases do not improve RoleNet and their semantics are not shown to matter | G33b: NLL(R0) − NLL(R-full) −0.0009 [−0.0165, +0.0136]; NLL(R-random) − NLL(R-full) +0.0056 [−0.0067, +0.0185]; UAR ensemble 26.35 vs 26.35 | biases learned but small (max within-head spread ≈ 0.22); they act in layer 1 only because the query row has one type per family (layer-2 biases inert by construction, verified); pattern with F5, F20, F25: explicit structural encodings add no detectable utility beyond evidence availability and flexible joint attention (G12) |
| F27 | The value of B's earlier labelled emotion is persistence; the context adds little once it is known | **[chat]** `analysis/prevB_headroom.py` on the 636 MCIS (26%, 45 episodes) where B's earlier turn is labelled (B found by clip-IV voice, analysis only); episode-fold LR, episode bootstrap: stay rate 48.6%; copy-Z 39.07 UAR, learned P(Y∣Z) 35.90, RoleNet (refit) 28.50, additive RoleNet + Z 36.19; X+Z − Z: ΔUAR +0.28 [−4.68, +6.32], ΔNLL +0.028 [−0.014, +0.068]; X+Z − X: ΔUAR +7.69 [+0.69, +13.43], ΔNLL +0.113 [+0.045, +0.179] | X = RoleNet's 7-way OOF output, not the full context; additive only (an interaction is underpowered at n = 636); gold Z is not available at inference, so the deployable headroom is set by how well Z can be recognised from clips I/II (clip recognizer ≈ 31%, F9; G9 pointer null) |

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
| Target-label ambiguity as the reason negative emotions are not forecastable | **[chat]** F13 | descriptive, one model family (RoleNet OOF) |

### 4b. Set aside because they reduce to a known method

| Proposal | Reduces to |
|---|---|
| Max-entropy pairwise-preserving contrast Q₂ + density-ratio discriminator | additive-logit model + residual interaction (GA²M / EMAP / PID estimators); the discriminator is a noisier estimator of the same quantity |
| Composition-consistent reduced models (missing participants) | automatic for marginals of any joint model; enforcing it needs a latent-variable / state-space model (VAEAC, MMIN, neural processes) |
| PIC-weighted auxiliary loss | proper weighted Brier score; mandatory control CE + happy/non-happy auxiliary |
| Forecast-aware recognition loss with uncertainty-adaptive weights (softmin over candidate transition matrices T_m + KL correction) | **withdrawn by the authors.** The loss is strictly proper (excess-risk identity checked numerically, gap ≈ 5e-18), but with limited capacity it can prefer a recognizer that is worse under every T_m: T₁ = [[1,0,0],[0,1,1]], T₂ = [[0,1,0],[1,0,1]], q = (.25,.25,.5); r₁ = (.27,.23,.5) has recognition error 0.0008, propagated error 0.0008 under both T_m and 0 for the averaged forecaster, r₂ = (.22,.22,.56) has 0.0054 / 0.0018 / 0.0018; the softmin loss (τ = 0.01, λ = 0.1) gives excess 0.00691 for r₁ vs 0.00234 for r₂ (re-computed here). Cause: the curvature of the softmin entropy adds Cov_w(∇H_m)/τ, a penalty on disagreement between the T_m. The corrected form, ‖T̄(C)(r − e_s)‖² + λ‖r − e_s‖² with T̄(C) fixed, measures the deployed forecaster's error but is a context-dependent proper loss (Hepburn et al. 2018; Plaud et al. ICML 2026, as cited by the authors) and is kept only as a baseline |

Note: reducing to a known form is not by itself a reason to reject. What must still be compared is the assumptions,
the estimator, the supervision it needs and the guarantees. These three were set aside mainly together with the
evidence in 4a/§3 (F7, F4, F11).

### 4c. Could not be tested with the available information

| Direction | Missing |
|---|---|
| Appraisal / goal-state of B (z_B) | human labels (no labelling staff) |
| Two-step recognize-B-then-forecast | a recognizer of earlier states whose errors matter less for the forecast (F9); the cause of the weak recognition is not separated: person assignment, domain, observation quality, or the gap between utterance-level labels and facial expression (HSEmotion is already face-specific) |
| Any claim about text or prosody content | an emotion/paralinguistic text and audio representation (§2); this is a separate gap from the two-step row, not necessarily the same cause |
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

**Run gate (fixed by the authors after G34; applies on top of the checklist).** "Worth a try" and exploratory
screening rounds are no longer reasons to spend compute. An experiment is run only if, before running:

1. a bottleneck is clearly indicated by the current evidence;
2. the intervention acts on that bottleneck, not just a plausible architectural variant;
3. there is a quantitative or mechanistic reason to expect an effect large enough to matter, not only "possible";
4. a win would be a substantial contribution (a +0.2–0.5 UAR gain with weak novelty does not qualify);
5. earlier results do not already lower its prior strongly.

Current reading of the evidence (**working hypothesis, not an established bottleneck**): what matters is **the right
evidence being available to the model**, especially listener evidence and the immediately preceding context.
Structural changes to attention / readout / relations (G12–G13, G28–G33) did not help detectably, so this is the
preferred direction. The firm evidence identifies *which sources are needed*; it does not yet identify *which error in
acquiring or using them is fixable*.

**Ablation drop ≠ recoverable headroom.** The G14 effects (Full − minus-L +2.04, Full − T-clipIIIonly +2.13) are the
advantage of having a source over removing it. They do not show that the vanilla model under-uses that source; it may
already extract most of its useful signal. G12 supports joint processing, but it changed grouping, encoder and fusion
point at once, so it does not identify missing higher-order interaction as the cause.

**Before a model change can be motivated, one of these must be shown (evaluation conditions, not proposals):**

| To be shown | Only then is this kind of model change motivated |
|---|---|
| Useful signal is present in the features but lost in aggregation | changing the aggregation can recover it |
| Evidence is noisy or missing, with observable indicators of its reliability | modelling reliability can improve inference |
| With the same evidence, the forecasting rule makes systematic errors | changing the rule can fix exactly those errors |

None of the three is established by the current results. A new direction must start from **a fixable error of the
vanilla model**, with evidence that the information to fix it is present in the allowed inputs.

**Research state (authors, after G34):** supported: listener face, preceding context and joint processing have value.
Not supported: a specific deficiency of the vanilla forecasting function that a mathematical intervention could fix with
a substantial effect. Comparisons: PaperBest is the only reference; all further analyses use train+val; the single test
read (G10) is complete, and its positive plain result does not replace the not-confirmed primary LA 7-class contrast.

*G34 (emotion-hypothesis queries) was withdrawn by the authors before running under this gate:* G13 showed the readout
is insensitive; it adds no information, only another readout of the same representation; the rare-class bottleneck
appears to lie partly in the input signal; the prior art (label-query attention, Query2Label) is close, so a small gain
would be a weak contribution. The notebook is kept in the repo, not run.

*Also withdrawn before running under this gate:*
* **Belief-state / affective-transition forecasting with privileged previous-B labels** — F27: the value of B's earlier
  label is persistence (X+Z ≈ Z), so a context-conditioned transition is not supported; the deployable headroom is set
  by recognising Z from clips I/II (≈ 31%, F9; G9 pointer null) and covers 26% of MCIS.
* **Low-rank higher-order interaction among listener / history / event evidence** — G12 does not isolate
  non-additivity, and its CS-int arm (MLP interaction of the three path summaries, a superset of a CP-trilinear term)
  recovered only +0.82 [−0.38, +2.08]; F7, F11 (EMAP −0.002), F16 and F24 found no interaction value; the form is LMF /
  TFN from multimodal emotion recognition, so a small gain would be a weak contribution.

## 6. Problem statements proposed by the authors (chronological) and their status

| # | Problem statement (authors) | What was done on Hi-EF | Status | Testable on MELD? |
|---|---|---|---|---|
| P1 | **Situation of B, not only B's expression**: forecasting needs z_B = B's goals and how the conversation supports or blocks them (who sided with B, who broke a commitment); third parties matter through this relation chain | not run; needed relation/goal labels | untested (labels) | partly: MELD has text and speaker IDs at training, so a **B-relative** reading of the context (which turns are B's, who responds to B) can be compared with a speaker-agnostic one without new labels; LLM reading is risky on Friends (contamination) |
| P2 | Observability-first version of P1: separate the mechanism, what is observable, and evidence the model learns it (blind evaluation, steps A/B/C) | appraisal pilot prepared, dropped | untested (labels) | as P1 |
| P3 | **Predictable distinctions**: which emotion distinctions are forecastable from the past and how clip III changes them (ρ_X(g), principal inertia components) | G19 (CONTINUE) + PIC follow-up | structure stable within RoleNet; gain from clip III not replicated across predictor families (row 2) | yes: same pair-AUC / PIC analysis with a text forecaster and ~3× more windows |
| P4 | **Pragmatic combination of content and expression** (pairwise-preserving contrast → additive vs interaction) | G20 | STOP in both families; text features weak | yes, with a stronger text encoder and speech audio (more power) |
| P5 | **Memory from missing observations** (absent participants keep an influence; composition-consistent reductions) | G21 proposed, not run | premise contradicted on Hi-EF (hysteresis, G11, G13) | partly done: no A × history interaction at label level (F16); MELD speaker IDs allow testing absent-speaker influence directly |
| P6 | **Forecasting under uncertain person–observation assignment** (certify decisions over valid assignments) | G22 proposed, not run; no identity ground truth on Hi-EF | open; could not measure whether assignment matters | **yes, and only there**: MELD speaker IDs give the true assignment, so the value of correct vs perturbed assignment and decision flips can be measured |
| P7 | Forecast-aware recognition with uncertainty-adaptive supervision | analysis only | withdrawn (counterexample, §4b) | — |
| — | Related open findings: mirroring vs shift (F14), B inertia (F8, F16), appraisal from content (G24, running) | | | inertia and mirroring replicate at label level (F16) |
