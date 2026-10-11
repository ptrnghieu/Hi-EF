# PartiSet

Implementation of **PartiSet**, a participant-centric model for emotion forecasting in multi-party video (anonymous submission).
The repository contains code only: no data, no trained weights and no result tables.

## Method in one paragraph
The faces of a sample are clustered into people and sorted into three slots: **A** (the speaker of the last turn), **L** (the
second most visible person, the likely listener) and **O** (everyone else). Each (slot, turn) cell is summarised by one
attention-pooled token; speech (text, audio, voice descriptor) and scene give one token per turn. A learned query and the 15
tokens are read by a 2-layer pre-norm transformer (d = 128), trained with auxiliary heads and modality dropout.

## Layout
```
partiset/     the method
  participants.py   face clustering -> slots A / L / O, frame selection, turn descriptors
  model.py          PartiSetCore (the model and its ablation switches), PartiSet (token-level switches)
  train.py          training loop (AdamW, early stopping on the UAR of held-out training episodes)
  cv.py             episode-level splits (early-stopping episodes, cross-validation folds)
  metrics.py        WAR / UAR, episode bootstrap, two-level bootstrap over seeds and episodes
  data.py, pipeline.py, config.py   tables, feature loading, hyper-parameters
scripts/      run_partiset_cv.py (5-fold episode CV, 10 seeds), run_partiset_test.py (test split, 5 seeds)
notebooks/    feature extraction and the baselines
tests/        checks that the package matches the first implementation (synthetic data, no dataset needed)
```

## Notebooks
| Notebook | Purpose |
|---|---|
| `extract_participant_features` | per-face (ArcFace, HSEmotion, pose, mouth) and per-voice (ECAPA) features that `partiset/` reads |
| `meld_prepare`, `meld_participant_features` | build MELD in the Hi-EF format and extract the same features |
| `baseline_person_features` | person-crop (body) CLIP features used by the ViT baselines |
| `baseline_vit` | our retraining of the Hi-EF paper's ViT baselines on our splits |
| `baseline_authors_code` | the authors' released model ported to precomputed features |
| `baseline_qwen_vl` | Qwen2.5-VL-7B zero- and few-shot |

The notebooks were written for Kaggle (GPU); dataset paths are set in the configuration cell at the top.

## Data (not included)
Hi-EF and MELD must be obtained from their owners. `partiset/` expects the split table of the paper, the pre-extracted CLIP /
AudioCLIP clip features of Hi-EF, and the output folder of `extract_participant_features` (`shard_*.pkl`).

## Running
```
pip install -r requirements.txt
python -m pytest tests
python scripts/run_partiset_cv.py   --dataset D --features F --split-csv S --participant-features P --out out/
python scripts/run_partiset_test.py --dataset D --features F --split-csv S --participant-features P --out out/ --unlock-test
```
`run_partiset_test.py` is meant to be run once; it refuses to read the test split without `--unlock-test`.
`tests/` check the model, the fold construction and the bootstrap against the first implementation. `participants.py` needs the real
feature shards and is not covered by a synthetic test.
