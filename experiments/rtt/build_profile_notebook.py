"""Writes k2_profile_pipeline.ipynb: cost of the fusion stage (PartiSet vs Baseline, same GPU), of participant grouping and of the
pre-processing stages per clip. The extraction cells are copied verbatim from g8a_role_features.ipynb."""
import json

g8a = json.load(open('g8a_role_features.ipynb'))
G = lambda i: ''.join(g8a['cells'][i]['source'])
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip('\n').splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip('\n').splitlines(True)}

CONFIG_C = '''
# ======== CONFIG (the constants of g8a_role_features, with the paths found automatically) ========
import glob, os
def find(pattern, d=False):
    hits = [p for p in sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)) if (os.path.isdir(p) if d else True)]
    assert hits, f'attach the dataset holding {pattern}'
    return hits[0]
DATASET_DIR = find('hi-ef-dataset', d=True)
SPLIT_CSV = find('source_folder_split_seed42.csv')
OUT_DIR = "/kaggle/working"
SHARD_DIR = os.path.dirname(find('shard_*.pkl'))
SAMPLE_FPS, MAX_FRAMES = 4, 32
DET_SIZE = (640, 640)
MIN_DET_SCORE, MIN_FACE_PX = 0.5, 24
AUDIO_SR = 16000
WIN_S, HOP_S = 1.5, 0.75
ENV_HZ = 10
N_DECODE_THREADS = 4
FER_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/"
           "enet_b0_8_va_mtl.onnx")
N_CLIPS, N_SAMPLES = 40, 200      # clips timed per stage, samples used to time the grouping
'''

cells = [md("""
# Cost of PartiSet and of the Baseline: fusion stage, participant grouping and pre-processing per clip

All numbers are measured in this session on the GPU shown in the first cell, so that they can be reported side by side:

1. **Fusion stage** (precomputed features): parameters, FLOPs per sample, forward and forward+backward time per sample, peak memory, for PartiSet and the Baseline.
2. **Participant grouping** of PartiSet (clustering of the faces of a sample, slot assignment, turn descriptors), per sample.
3. **Pre-processing per clip** with the extraction code of the experiments (the cells are copied verbatim from `g8a_role_features`): video decoding, face detection + ArcFace + HSEmotion, audio + ECAPA.
4. **Unit costs of CLIP** (16 images) and of the YOLOv8-small person detector (16 frames), used by the Baseline's body branch. AudioCLIP is not timed here.

Attach: the code dataset, the Hi-EF dataset, the split table and the participant-feature shards. GPU on, Internet on (model weights are downloaded).
"""),
code("""
import glob, os, shutil, subprocess, sys, time, json
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| torch', torch.__version__)
CODE_SRC = next(p for p in glob.glob('/kaggle/input/**/partiset', recursive=True) if os.path.isdir(p))
CODE_ROOT = '/kaggle/working/code'
shutil.copytree(os.path.dirname(CODE_SRC), CODE_ROOT, dirs_exist_ok=True)
sys.path.insert(0, CODE_ROOT)
"""),
code(CONFIG_C),
md("## 1. Fusion stage"),
code('''
import torch.nn as nn, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from partiset import BASE, Batch, PartiSet
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, FD, MAXF, REPEATS = 64, 146, 24, 30

# Baseline classes: copied from the baseline notebook, executed on synthetic features of the real shapes
nb = json.load(open(f'{CODE_ROOT}/notebooks/baseline_vit.ipynb'))
src = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'class Baseline' in ''.join(c['source']))
FEAT = {'face': torch.randn(3 * B, 16, 512, device=dev), 'ori': torch.randn(3 * B, 16, 512, device=dev),
        'body': torch.randn(3 * B, 16, 512, device=dev), 'text': torch.randn(3 * B, 512, device=dev), 'audio': torch.randn(3 * B, 527, device=dev)}
CLIPIDX = torch.arange(3 * B, device=dev).reshape(B, 3)
def gather(idx):
    flat = idx.reshape(-1)
    return {k: v[flat].reshape(*idx.shape, *v.shape[1:]) for k, v in FEAT.items()}
ns = dict(torch=torch, nn=nn, F=F, FEAT=FEAT, CLIPIDX=CLIPIDX, gather=gather)
import ast
exec(compile(ast.Module([n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)], []), 'baseline', 'exec'), ns)    # classes only
baseline = ns['Baseline'](True, True, 'trans', 'lstm').to(dev)       # text + audio, Transformer intra-clip, LSTM+Transformer inter-clip
ix = torch.arange(B, device=dev); yb = torch.randint(0, 7, (B,), device=dev)

b = Batch(torch.randn(B, 3, 3, MAXF, FD, device=dev), torch.randn(B, 1, 3, 32, FD, device=dev),
          torch.rand(B, 3, 3, MAXF, device=dev) > 0.5, torch.rand(B, 1, 3, 32, device=dev) > 0.5,
          torch.randn(B, 3, 512, device=dev), F.normalize(torch.randn(B, 3, 527, device=dev), dim=-1), torch.ones(B, 3, device=dev),
          torch.randn(B, 3, 1024, device=dev), torch.randn(B, 3, 9, device=dev), yb, yb)
partiset = PartiSet(BASE, FD, 9).to(dev)

def timed(fn):
    for _ in range(3): fn()
    if dev == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn()
        if dev == 'cuda': torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) / B * 1000)
    t.sort(); return t[len(t) // 2], t[int(0.95 * (len(t) - 1))]

def profile(name, model, fwd, train_loss):
    n = sum(p.numel() for p in model.parameters())
    # FLOPs are counted in train mode: in eval mode nn.TransformerEncoder uses a fused kernel that the FLOP counter does not see
    model.train()
    with FlopCounterMode(display=False) as fc: fwd()
    flops = fc.get_total_flops() / B
    model.eval()
    with torch.no_grad():
        f = timed(fwd)
    if dev == 'cuda': fm = torch.cuda.max_memory_allocated() / 2**20
    model.train()
    def step():
        train_loss().backward(); model.zero_grad()
    t = timed(step)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev == 'cuda' else float('nan')
    return dict(model=name, params_M=n / 1e6, GFLOPs_per_sample=flops / 1e9, fwd_ms_median=f[0], fwd_ms_p95=f[1], train_ms_median=t[0], train_ms_p95=t[1], peak_MiB_train=mem)

def ps_loss():
    logits, aux = partiset(b, train=True)
    return F.cross_entropy(logits, yb) + sum(w * F.cross_entropy(l, t.clamp(min=0)) for l, t, w in aux.values())
rows = [profile('Baseline (F,B,S,T,A; Trans, LSTM+Trans)', baseline, lambda: baseline(ix), lambda: F.cross_entropy(baseline(ix), yb)),
        profile('PartiSet', partiset, lambda: partiset(b), ps_loss)]
# PartiSet: the FLOPs depend on the number of face frames kept per (slot, turn) cell
for mf in (4, 24):
    bb = Batch(torch.randn(B, 3, 3, mf, FD, device=dev), b.pool, torch.rand(B, 3, 3, mf, device=dev) > 0.5, *b[3:])
    partiset.train()
    with FlopCounterMode(display=False) as fc: partiset(bb)
    print(f'PartiSet, {mf} frames per cell: {fc.get_total_flops() / B / 1e6:.1f} MFLOPs per sample')
import pandas as pd
FUSION = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(FUSION.round(3).to_string(index=False)); FUSION.to_csv('/kaggle/working/cost_fusion.csv', index=False)
'''),
md("## 2. Participant grouping (PartiSet), per sample"),
code("""
import numpy as np
from partiset.data import load_split
from partiset.participants import ParticipantBuilder, load_feature_shards
_, SP = load_split(DATASET_DIR, SPLIT_CSV)
SP = SP[SP.split != 'test'].reset_index(drop=True)
sub = SP.sample(N_SAMPLES, random_state=0).reset_index(drop=True)
t0 = time.time(); SH = load_feature_shards(SHARD_DIR); t_load = time.time() - t0
t0 = time.time(); PB = ParticipantBuilder(sub, SH); t_group = time.time() - t0
clips_sub = sorted(set(sub[['clip1', 'clip2', 'clip3']].values.ravel()))
t0 = time.time(); PB.build_face_tensors(clips_sub, 'cpu'); t_tensor = time.time() - t0
GROUP = dict(samples=N_SAMPLES, load_shards_s=t_load, grouping_ms_per_sample=t_group / N_SAMPLES * 1000, face_tensors_ms_per_sample=t_tensor / N_SAMPLES * 1000)
print(GROUP); json.dump(GROUP, open('/kaggle/working/cost_grouping.json', 'w'))
"""),
md("## 3. Pre-processing per clip (extraction code of the experiments, copied verbatim)"),
code(G(1)),
code(G(3)),
code(G(4)),
code(G(5)),
code("""
# decode / face detection + ArcFace + HSEmotion / audio + ECAPA, per clip (GPU synchronised)
rng = np.random.default_rng(0)
sel = list(rng.choice(clips, N_CLIPS, replace=False))
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
_ = read_frames(find_media(VIDEO_ROOTS, sel[0], ('.mp4', '.avi', '.mkv', '.mov')))      # warm-up
_ = analyse_frames(*_[:2]) if False else None
T = {'decode': [], 'faces': [], 'audio': []}; FRAMES = []
for c in sel:
    pv = find_media(VIDEO_ROOTS, c, ('.mp4', '.avi', '.mkv', '.mov')); pa = find_media(AUDIO_ROOTS, c, ('.mp3', '.wav', '.flac', '.m4a'))
    t0 = time.perf_counter(); frames, times, meta = read_frames(pv); T['decode'].append(time.perf_counter() - t0)
    t0 = time.perf_counter(); faces = analyse_frames(frames, times); sync(); T['faces'].append(time.perf_counter() - t0)
    t0 = time.perf_counter(); au = analyse_audio(pa); sync(); T['audio'].append(time.perf_counter() - t0)
    FRAMES.append(frames)
EXTRACT = pd.DataFrame({k: pd.Series(v).describe(percentiles=[.5, .95])[['mean', '50%', '95%']] for k, v in T.items()}).T
EXTRACT['total_clip_s'] = EXTRACT['mean']
print(EXTRACT.round(3).to_string())
tot = np.array(T['decode']) + np.array(T['faces']) + np.array(T['audio'])
print(f'per clip: mean {tot.mean():.2f} s, median {np.median(tot):.2f} s, p95 {np.percentile(tot, 95):.2f} s | '
      f'clips of the development and test tables: {len(clips)} -> about {tot.mean() * len(clips) / 3600:.1f} GPU-hours at the mean')
EXTRACT.to_csv('/kaggle/working/cost_extraction_per_clip.csv')
"""),
md("## 4. Unit costs of CLIP (16 images) and of the YOLOv8-small person detector (16 frames)"),
code("""
!pip install -q ultralytics transformers
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
dev2 = 'cuda' if torch.cuda.is_available() else 'cpu'
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev2).eval().half() if dev2 == 'cuda' else CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval()
proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
imgs = [Image.fromarray(f[:, :, ::-1]) for f in next(fr for fr in FRAMES if len(fr) >= 16)[:16]]
def clip_img():
    x = proc(images=imgs, return_tensors='pt')['pixel_values'].to(dev2)
    with torch.no_grad(): clip.get_image_features(pixel_values=x.half() if dev2 == 'cuda' else x)
    sync()
from ultralytics import YOLO
yolo = YOLO('yolov8s.pt')
frames16 = [np.array(i)[:, :, ::-1] for i in imgs]
def yolo16():
    yolo.predict(frames16, classes=[0], conf=0.4, verbose=False); sync()
def med(fn, n=15):
    fn(); fn(); t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t))
UNIT = dict(clip_16_images_s=med(clip_img), yolov8s_16_frames_s=med(yolo16))
print(UNIT); json.dump(UNIT, open('/kaggle/working/cost_units.json', 'w'))
"""),
md("""
## Summary for the paper

Report next to the fusion table: the GPU, the fusion rows of section 1, the grouping time per sample (section 2), the per-clip pre-processing (section 3) and the CLIP / YOLO unit costs (section 4).
The Baseline's body branch needs the YOLOv8-small detector and CLIP on person crops; PartiSet does not use body features. AudioCLIP features (both models) are not timed here.
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('k2_profile_pipeline.ipynb', 'w'), indent=1)
print('written k2_profile_pipeline.ipynb with', len(cells), 'cells')
