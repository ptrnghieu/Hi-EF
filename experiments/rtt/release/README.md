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
notebooks/    the baselines (ViT, Qwen2.5-VL)
tests/        checks that the package matches the first implementation (synthetic data, no dataset needed)
```

## Notebooks
| Notebook | Purpose |
|---|---|
| `baseline_vit` | our retraining of the Hi-EF paper's ViT baselines on our splits |
| `baseline_qwen_vl` | Qwen2.5-VL-7B zero- and few-shot |

The notebooks were written for Kaggle (GPU); dataset paths are set in the configuration cell at the top.

## Inputs (not included)
Feature extraction is not part of this repository. `partiset/` and the notebooks expect:
* the split table (`source_folder_split_seed42.csv`) and `annotation.csv` of Hi-EF;
* one `<clip>.pt` per clip with the pre-extracted CLIP ViT-B/32 features of the dataset (`face_features`, `ori_features`,
  `text_feature`, AudioCLIP `audio_feature`);
* for PartiSet, `shard_*.pkl` files with, per clip, `faces` (a list of per-frame face records with `fer`, `fer_emb`, `arc`,
  `box`, `pose`, `mouth`, `t`, `frame`), `audio` (`env`, `ecapa`) and `meta` (`duration`, `n_sampled`);
* for the ViT baselines, `body_<SOURCE>.pt` = `{clip_id: (features [16, 512], valid mask [16])}`, CLIP ViT-B/32 features of the
  largest detected person in 16 frames per clip.

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
