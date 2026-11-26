# Cost of PartiSet and of the Baseline (Tesla T4, measured with k2_profile_pipeline.ipynb)

## Fusion stage (precomputed features, batch 64, synthetic inputs of the real shapes)

| | Baseline | PartiSet |
|---|---|---|
| Parameters | 34.057M | 0.709M |
| FLOPs per sample (train-mode count) | 1.886 G | 0.030 G (24 frames per cell); 17.0 M (4 frames per cell) |
| Forward, median (p95), ms per sample | 0.643 (0.650) | 0.042 (0.048) |
| Forward + backward, median (p95), ms per sample | 2.130 (2.148) | 0.162 (0.170) |
| Peak GPU memory in training | 977 MiB | 239 MiB |

FLOPs are counted in train mode because the PyTorch FLOP counter does not see the fused kernel that `nn.TransformerEncoder` uses in eval mode.

## Pre-processing per clip (extraction code of the experiments, 40 random clips)

| Stage | Mean (s) | Median (s) | p95 (s) |
|---|---|---|---|
| Video decoding and frame sampling | 0.25 | 0.24 | 0.35 |
| Face detection + ArcFace + HSEmotion | 2.94 | 2.08 | 6.98 |
| Audio + ECAPA | 0.46 | 0.09 | 0.19 |

The mean of the audio stage is far above its p95, so it contains one outlier (most likely the first call, which initialises the models); the median is the representative value.

## Participant grouping (PartiSet), 200 development samples

| Step | Cost |
|---|---|
| Clustering of the faces and slot assignment | 2.5 ms per sample |
| Face tensors (PCA transform and arrays) | 9.9 ms per sample (PCA fitted on the 200 samples only; the real per-fold PCA fit takes about 7 s) |
| Reading all feature shards | 6.8 s |

## Unit costs (16 images or frames)

| | Time |
|---|---|
| CLIP ViT-B/32 on 16 images | 0.112 s |
| YOLOv8-small on 16 frames | 0.328 s |
