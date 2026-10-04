# RoleNet pipeline audit: where information is dropped (train+val only)

Code references are to `experiments/rtt/build_notebooks.py`. "Measured" numbers come from local artifacts
(G16 per-MCIS table, annotation.csv) or from earlier logs where stated.

| # | Stage (code) | What is dropped or simplified | Measured / existing evidence | Status |
|---|---|---|---|---|
| 1 | Frame sampling, G8a: 4 fps, ≤ 32 frames per clip (`:2809`); ≤ 24 frames per role cell (`pick_frames`, `:3259`) | frames beyond the caps | clip durations median 1.9–2.0 s, none > 8 s (cap never binds); cell cap of 24 hit in 0.04% of MCIS | no loss |
| 2 | Per-frame face features (`:3247–3252`) | detection score not kept; HSEmotion embedding reduced by PCA to 128 | not measurable locally | minor, not pursued |
| 3 | Roles (`:3302–3304`): A = most frames in clip III, L = second most in clip III, everyone else O | L exists only if a second face is in clip III; otherwise B's clip I/II faces go to O | no second face in III: 40.1% of MCIS, and then nL1 + nL2 = 0 in 100% of them (G16 table); A = B only 2.5% (G6a, 600 MCIS) | **tested**: oracle L = true responder incl. I/II-only cases, G26 −0.04 [−1.50, +1.85]; no gain on the 390 MCIS where the rule is wrong |
| 4 | FramePool (`:3491–3495`) | number of frames (softmax weights sum to 1); only a presence flag kept | L-III frames 0 / 1 / 2–3 / 4+ = 40 / 26 / 21 / 14% | **tested post hoc** F29 (no gain); G29 pooling variants and per-frame tokens no gain; end-to-end count input untested |
| 5 | Text: CLIP text encoder, 77-token context | truncation; caption-style encoder | utterances ≈ 8 tokens median, q99 ≈ 16, max ≈ 20: no truncation | no loss from truncation; encoder weakness (text alone 18.5 UAR, G20) is out of the chosen scope |
| 6 | Audio: AudioSet 527-way probabilities, L2-normalised (`TXT, AUD, AFD`, G13 cell 8) | overall level; G8a RMS envelope `env` (10 Hz) and ECAPA are used only for mouth–audio sync and voice identity (VOI); pitch never extracted | not measurable locally | **untested**; prior low (C-noAudio −0.06 in G13; clip-IV audio alone 16.2 UAR in G23) |
| 7 | Turn structure (VOI): voice cosine of clips I/II to clip III, mouth–audio sync per role | "speaker of clip k is L" not explicit | — | **tested**: G9 perfect "B spoke" pointer adds nothing |
| 8 | Time: clip durations and gaps not inputs; no order needed | elapsed time | — | **tested**: G16 (no time effect), G33a (order not used) |
| 9 | Training supervision: clip-III label only supervises A's face token (`head_A(h[:, 0, 2])`, `:3535`); clip I/II speaker labels unused; speech tokens never supervised for recognition | training-time labels of the context clips | — | **untested inside RoleNet**; prior lowered by G1–G3b (recognizer on all labels 32.6 UAR, RtT forecast ≈ 20.5) and F27/F28 (deployable gains limited by recognition) |

Summary: every stage is either not lossy (1, 5) or already tested without gain (3, 4, 7, 8). Two dropped signals are
untested: the speech energy envelope / loudness (6) and the unused training-time labels of clips I–III (9). Both have
low priors from earlier results; neither is shown to carry forecast information beyond RoleNet.
