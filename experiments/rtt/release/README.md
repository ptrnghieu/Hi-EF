# PartiSet: participant-centric evidence for emotion forecasting (code release)

Code, notebooks and result tables for the paper *Watch the Listener: Participant-Centric Evidence for Emotion Forecasting
in Multi-Party Video* (anonymous submission). The notebooks were run on Kaggle (GPU); paths that point to Kaggle datasets
use `USER` as a placeholder for the dataset owner and can be edited in each notebook's CONFIG cell.

## Data (not included)
* Hi-EF benchmark (videos, audio, annotations) and the MELD videos are not redistributed here; follow the licences and
  download pages of the original authors. The Hi-EF repository with the original code is the one cited in the paper.
* The episode-level split of Hi-EF is a CSV (`source_folder_split_seed42.csv`, columns `sample_id, split, source_folder,
  clip1..clip4, clip3_emotion, clip4_emotion`); see Appendix A of the paper for how it is built and used. MELD windows are built
  by `notebooks/m1_meld_prepare_features.ipynb`.
* Pre-computed features (CLIP ViT-B/32 face and whole-frame vectors, text and AudioCLIP vectors) follow the Hi-EF feature layout;
  our face/voice features come from `notebooks/g8a_role_features.ipynb` (Hi-EF) and `m2_meld_g8a_features.ipynb` (MELD).

## Layout
```
notebooks/    notebooks behind the numbers in the paper
analysis/     scripts that draw the figures and tables from the saved results
tools/        scripts that generate some of the notebooks (build_*.py)
results/      saved result tables (CSV) used in the paper
docs/         preregistration of the single test run and the list of test-split accesses
```

## Notebooks in `notebooks/`
* **features**: `g8a_role_features`, `m1_meld_prepare_features`, `m2_meld_g8a_features`, `b1_body_features`
* **model, development CV and test**: `g8b_rolenet_cv`, `g10_test_preregistered`, `g14_ten_seed_confirmation_cv`
* **ablations and analyses**: `g11_channel_ablations_cv`, `g12_cs_rolenet_cv`, `g13_token_ablations_cv`, `g33a_temporal_audit_cv`, `g33b_relational_bias_cv`, `g33c_order_probe`, `g35_group_emap_cv`, `g40_clip3_truncation_cv`, `g41_feature_matched_baseline_cv`, `g43_listener_history_vs_current_cv`, `g44_oracle_listener_truncation_cv`, `g44a_clip4_faces_dev`, `g26a_clip4_faces`, `g47_organisation_controls_cv`, `g48_proxy_oracle_groups`, `g32_mirror_diagnostics`, `g45_emotion_features_gate_cv`, `g6a_listener_visibility`, `g6b_listener_expression`, `g39_party_statistics`, `g37_case_frames`
* **baselines and other benchmarks**: `b2_vit_baselines_body`, `b3_released_code_baseline`, `l1_qwen_vl_zero_few_shot`, `m3_meld_rolenet`

## From the paper to the code
| Paper | Where |
|---|---|
| Table 1 (main results) | PartiSet: `g10_test_preregistered` (Hi-EF), `m3_meld_rolenet` (MELD); ViT baselines: `b2_vit_baselines_body`; released-code baseline: `b3_released_code_baseline`; Qwen2.5-VL: `l1_qwen_vl_zero_few_shot` |
| Table 2 (where the gain comes from) | `g41_feature_matched_baseline_cv`, `g43_listener_history_vs_current_cv`, `g40_clip3_truncation_cv`, `g44_oracle_listener_truncation_cv` |
| Table 3 (ablations) | `g8b_rolenet_cv`, `g11`, `g13`, `g14`, `g28`, `g29`, `g30`, `g47` |
| Table 4 (proxy vs oracle) | `g44`, `g48_proxy_oracle_groups` |
| Table 5 (structure, EMAP) | `g33a`, `g33b`, `g33c`, `g35_group_emap_cv` |
| Figures 1, 3, 4, 6, 7, 8 | `analysis/motivation_figure.py` (with `g37_case_frames`), `analysis/data_stats_figure.py` and `analysis/paper_figures_v2.py` (from `g6a_listener_visibility`, `g39_party_statistics` and the saved predictions), `analysis/architecture_figure.py` |

## Notes
* Seeds: 10 for development CV, 5 for the Hi-EF test run, 3 for MELD. Results use seed ensembles.
* The Hi-EF test split was read by the notebooks listed in `docs/test_accesses.md` (including earlier work that is not part of
  this release); `docs/PREREG_G10_TEST.md` is the plan fixed before the preregistered run.
* The language-model notebook needs a GPU with about 16 GB per device and internet access for the weights.
* No identity is linked to a name: faces and voices are only grouped inside one sample.
