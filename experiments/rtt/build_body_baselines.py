"""Build the body-branch baseline notebooks.

  python experiments/rtt/build_body_baselines.py
    b1_body_features.ipynb      B1: YOLO person crops -> CLIP ViT-B/32 'body' features for Hi-EF or MELD clips I-III
    b2_vit_baselines_body.ipynb B2: the seven ViT configurations of the Hi-EF paper (Table 2) with face + body + scene
                                    on the Hi-EF split (G10 protocol) or on MELD (M3 protocol)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    return {"cells": [{"cell_type": k, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if k == "code" else {})} for k, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


m1 = json.load(open(HERE / "m1_meld_prepare_features.ipynb"))
M1_FETCH = "".join(m1["cells"][3]["source"])          # MELD download / unpack / video index code, reused verbatim
M1_FETCH = M1_FETCH[:M1_FETCH.index("vids, csvs = fetch_meld()")]   # keep the definitions, drop M1's own call

# ------------------------------------------------------------------------------------------------ B1
B1 = [
("markdown", """
# B1 — Body features (YOLO person crops, CLIP ViT-B/32) for Hi-EF or MELD

The Hi-EF paper's visual branches are face, whole frame (scene) and **person (body)**. The released features
(`hi-ef-features-v2`) hold face and whole-frame CLIP features only, so this notebook adds the missing branch:
16 frames per clip (evenly spaced, as in M1), the largest detected person in each frame (ultralytics YOLO, class
*person*), CLIP ViT-B/32 image features of the crop (the same encoder as the face and scene features).
Only clips I–III are processed; clip IV is never read. Set `SOURCE = 'hief'` (Hi-EF clips) or `'meld'` (MELD clips;
attach M1's output and, optionally, a MELD.Raw dataset via `MELD_LOCAL`).

Output: `body_<SOURCE>.pt` = `{clip_id: (features [16, 512] fp16, valid mask [16] bool)}`.
Frames without a detected person have mask False and zero features. The detector is not the authors' (their script
is not in the repository), so the crops are similar in kind, not identical.
"""),
("code", "!pip install -q ultralytics"),
("code", '''
# ======== CONFIG ========
import os, glob
SOURCE = "hief"                                   # "hief" or "meld"
OUT_DIR = "/kaggle/working"
N_FRAMES = 16
YOLO_WEIGHTS = "yolov8s.pt"                       # downloaded by ultralytics (internet on)
PERSON_CONF, MIN_PERSON_H = 0.4, 64               # confidence and minimum crop height in pixels
CROP_MARGIN = 0.05
DEBUG_N = None                                    # e.g. 40 clips for a smoke test

DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
MELD_LOCAL = None                                 # MELD.Raw dataset folder; None = auto-detect an attached MELD dataset


def find_meld_raw():
    # an attached dataset whose folder name contains "meld", other than the M1 output (meld-m1 / meld_hief / meld_features)
    out = []
    for base in ["/kaggle/input"] + sorted(glob.glob("/kaggle/input/datasets/*")):
        for d in sorted(glob.glob(base + "/*")):
            n = os.path.basename(d).lower()
            if os.path.isdir(d) and "meld" in n and "m1" not in n and "hief" not in n and "features" not in n and "probs" not in n:
                has = next(glob.iglob(d + "/**/*.mp4", recursive=True), None) or next(glob.iglob(d + "/**/*.tar.gz", recursive=True), None)
                if has:
                    out.append(d)
    return out


if SOURCE == "meld" and MELD_LOCAL is None:
    _c = find_meld_raw()
    assert _c, "attach the MELD.Raw dataset (the Kaggle 'Multimodal EmotionLines Dataset (MELD)') or set MELD_LOCAL"
    MELD_LOCAL = _c[0]
    print("MELD_LOCAL =", MELD_LOCAL)
WORK = "/tmp/meld"


def find_one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"attach the dataset holding {pattern}")
    return hits[0]
'''),
("code", "import numpy as np, pandas as pd\n" + M1_FETCH),
("code", '''
import time, torch, cv2
from PIL import Image
if SOURCE == "hief":
    roots = sorted(glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF")))
    VIDEO_ROOTS = [os.path.join(r, "video") for r in roots if os.path.isdir(os.path.join(r, "video"))]

    def video_of(c):
        ep, num = c.split('/')
        for root in VIDEO_ROOTS:
            p = os.path.join(root, ep, num + ".mp4")
            if os.path.exists(p):
                return p
        return None
    sp = pd.read_csv(SPLIT_CSV, dtype=str, usecols=['sample_id', 'split', 'clip1', 'clip2', 'clip3'])
else:
    vids, csvs = fetch_meld()
    sp = pd.read_csv(find_one("meld_split.csv"), dtype=str)

    def video_of(c):
        return vids.get((c[:2], int(c[2:7]), int(c[8:])))
CLIPS = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))
if DEBUG_N:
    CLIPS = CLIPS[:DEBUG_N]
print(f"{SOURCE}: clips I-III {len(CLIPS)} | without video {sum(video_of(c) is None for c in CLIPS)}")
'''),
("code", '''
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO(YOLO_WEIGHTS)
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float().cpu()


def frames_of(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    keep = set(np.linspace(0, max(n - 1, 0), N_FRAMES).astype(int).tolist()) if n else set()
    out, i = [], 0
    while len(out) < len(keep):
        ok, fr = cap.read()
        if not ok:
            break
        if i in keep:
            out.append(fr)
        i += 1
    cap.release()
    return out


def person_crops(frs):
    """largest person (class 0) per frame, or None"""
    res = yolo(frs, classes=[0], conf=PERSON_CONF, verbose=False, device=0 if DEVICE == "cuda" else "cpu")
    crops = []
    for fr, r in zip(frs, res):
        H, W = fr.shape[:2]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        b = [x for x in b if x[3] - x[1] >= MIN_PERSON_H]
        if not b:
            crops.append(None)
            continue
        x1, y1, x2, y2 = max(b, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        mw, mh = CROP_MARGIN * (x2 - x1), CROP_MARGIN * (y2 - y1)
        crops.append(fr[int(max(0, y1 - mh)):int(min(H, y2 + mh)), int(max(0, x1 - mw)):int(min(W, x2 + mw))])
    return crops
'''),
("code", '''
OUT = f"{OUT_DIR}/body_{SOURCE}.pt"
BODY = torch.load(OUT, map_location="cpu") if os.path.exists(OUT) else {}
todo = [c for c in CLIPS if c not in BODY]
print(f"to extract: {len(todo)} / {len(CLIPS)}")
stats = {"no_video": 0, "frames": 0, "body_frames": 0}
t0 = time.time()
for n, c in enumerate(todo):
    p = video_of(c)
    frs = frames_of(p) if p else []
    stats["no_video"] += not frs
    f = torch.zeros(N_FRAMES, 512); m = torch.zeros(N_FRAMES, dtype=torch.bool)
    if frs:
        crops = person_crops(frs)
        ok = [k for k, cr in enumerate(crops) if cr is not None and cr.size]
        if ok:
            imgs = [Image.fromarray(cv2.cvtColor(crops[k], cv2.COLOR_BGR2RGB)) for k in ok]
            with torch.no_grad():
                o = feat(clip.get_image_features(pixel_values=proc(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)))
            for j, k in enumerate(ok):
                f[k], m[k] = o[j], True
        stats["frames"] += len(frs); stats["body_frames"] += int(m.sum())
    BODY[c] = (f.half(), m)
    if (n + 1) % 500 == 0:
        torch.save(BODY, OUT)
        el = (time.time() - t0) / 60
        print(f"{n + 1}/{len(todo)} | {el:.1f} min | ETA {el / (n + 1) * (len(todo) - n - 1):.0f} min | {stats}", flush=True)
torch.save(BODY, OUT)
print("done", stats)
'''),
("markdown", "## Sanity checks (no labels)"),
("code", '''
valid = torch.stack([BODY[c][1] for c in CLIPS]).float()
print(f"clips with at least one body frame: {(valid.sum(1) > 0).float().mean() * 100:.1f}% | "
      f"mean valid frames per clip {valid.sum(1).mean():.1f} / {N_FRAMES}")
'''),
]

# ------------------------------------------------------------------------------------------------ B2
B2 = [
("markdown", """
# B2 — The Hi-EF paper's ViT baselines with face + body + scene (Table 2, rows 5–11), retrained

Reimplementation of the paper's ViT rows on frozen CLIP features, in the G10 `PaperBest` code, extended by a **body**
branch (person crops from B1) so that the visual input is face + body + scene (whole frame), as in the paper. Rows:

| # | text | audio | intra-clip | inter-clip |
|---|---|---|---|---|
| 5 | – | – | Transformer | LSTM+Transformer |
| 6 | – | AudioCLIP | Transformer | LSTM+Transformer |
| 7 | CLIP | – | Transformer | LSTM+Transformer |
| 8 | CLIP | AudioCLIP | Average | Average |
| 9 | CLIP | AudioCLIP | Transformer | Average |
| 10 | CLIP | AudioCLIP | Average | LSTM+Transformer |
| 11 | CLIP | AudioCLIP | Transformer | LSTM+Transformer (the paper's best, "Baseline") |

Definitions (our reading of the paper; the released code has no separate class for them): *intra Average* = mean over
frames of each visual branch, then mean over the branch and modality vectors; *intra Transformer* = the PaperBest
temporal transformers and cross-attention fusion; *inter Average* = mean of the three clip vectors; *inter
Transformer* = positional encoding + 2-layer transformer, mean pooled; *LSTM+Transformer* = as PaperBest.

* `SOURCE = 'hief'`: G10 protocol. Fit on the 45 train+val episodes minus 5 selection episodes (`random.Random(2026)`),
  early stopping on those, **test read here** (409 MCIS; an additional test access, to be listed in the paper's
  appendix), 5 seeds. `SOURCE = 'meld'`: M3 protocol, MELD train → fit, dev → early stopping, test, 3 seeds.
* Plain argmax UAR / WAR of the seed ensemble and the per-seed mean are reported; no logit adjustment.
"""),
("code", '''
# ======== CONFIG ========
import os, glob
SOURCE = "hief"                       # "hief" or "meld"
UNLOCK_TEST = True                    # the test split is read by this notebook
OUT_DIR = "/kaggle/working"


def find_one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"attach the dataset holding {pattern}")
    return hits[0]


if SOURCE == "hief":
    DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
    FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
    SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
    SEEDS = [42, 123, 456, 789, 1024]
else:
    DATASET_DIR = find_one("meld_hief")
    FEATURES_DIR = find_one("meld_features")
    SPLIT_CSV = find_one("meld_split.csv")
    SEEDS = [42, 123, 456]
BODY_PT = find_one(f"body_{SOURCE}.pt")        # output of B1
N_INNER_DEV, SELECT_SEED = 5, 2026
LR, WEIGHT_DECAY = 1e-4, 1e-5
EPOCHS, BATCH = 50, 32                # as PaperBest in G10 / M3
DEBUG_PER_EPISODE = None
# name, row in Table 2, use text, use audio, intra, inter
CONFIGS = [
    ("ViT FBS / T,LSTM+T",      5, False, False, "trans", "lstm"),
    ("ViT FBS + A / T,LSTM+T",  6, False, True,  "trans", "lstm"),
    ("ViT FBS + T / T,LSTM+T",  7, True,  False, "trans", "lstm"),
    ("ViT FBS+T+A / Avg,Avg",   8, True,  True,  "avg",   "avg"),
    ("ViT FBS+T+A / T,Avg",     9, True,  True,  "trans", "avg"),
    ("ViT FBS+T+A / Avg,LSTM+T", 10, True, True, "avg",   "lstm"),
    ("ViT FBS+T+A / T,LSTM+T (best)", 11, True, True, "trans", "lstm"),
]
'''),
("code", '''
import random, time, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
if not UNLOCK_TEST:
    raise RuntimeError("Set UNLOCK_TEST = True: this notebook reads the test split.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def war_uar(pred, y, k=7):
    pred, y = np.asarray(pred), np.asarray(y)
    return (pred == y).mean() * 100, np.mean([(pred[y == c] == c).mean() * 100 for c in range(k) if (y == c).any()])


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


sp = pd.read_csv(SPLIT_CSV, dtype=str)
sp['yB'] = sp['clip4_emotion'].map(E2I)
assert sp['yB'].notna().all()
if DEBUG_PER_EPISODE:
    sp = sp.groupby('source_folder', group_keys=False).head(DEBUG_PER_EPISODE)
DEV = sp.reset_index(drop=True)
N = len(DEV)
IS_TEST = (DEV.split == 'test').values
src = DEV.source_folder.values
if SOURCE == "hief":
    EPS = np.array(sorted(set(src[~IS_TEST])))
    sel_eps = sorted(random.Random(SELECT_SEED).sample(list(EPS), N_INNER_DEV))
    fit_rows = np.where(~IS_TEST & ~np.isin(src, sel_eps))[0]
    dev_rows = np.where(np.isin(src, sel_eps))[0]
    if not DEBUG_PER_EPISODE:
        assert len(EPS) == 45 and IS_TEST.sum() == 409
else:
    sel_eps = 'MELD dev'
    fit_rows = np.where((DEV.split == 'train').values)[0]
    dev_rows = np.where((DEV.split == 'val').values)[0]
te_rows = np.where(IS_TEST)[0]
assert not set(src[te_rows]) & set(src[~IS_TEST])
print(f"fit {len(fit_rows)} | selection {len(dev_rows)} ({sel_eps}) | test {len(te_rows)}")
y_all = DEV.yB.values
'''),
("code", '''
# ---- features of clips I-III: face, whole frame (ori), body (B1), text, audio -- as G10 / M3 plus the body branch
all_clips = sorted(set(DEV[['clip1', 'clip2', 'clip3']].values.ravel()))
CIDX = {c: i for i, c in enumerate(all_clips)}
BODY = torch.load(BODY_PT, map_location='cpu')
bufs = {k: [] for k in ('face', 'ori', 'body', 'text', 'audio')}
no_body = 0
for c in tqdm(all_clips, desc='loading features'):
    d = torch.load(os.path.join(FEATURES_DIR, c.replace('/', '_') + '.pt'), map_location='cpu', weights_only=False)
    bufs['face'].append(d['face_features'].float())
    bufs['ori'].append(d['ori_features'].float())
    if c in BODY:
        bufs['body'].append(BODY[c][0].float())
    else:
        bufs['body'].append(torch.zeros(16, 512)); no_body += 1
    bufs['text'].append(d['text_feature'].float().reshape(-1))
    bufs['audio'].append(d.get('audio_feature', torch.zeros(527)).float().reshape(-1))
FEAT = {k: torch.nan_to_num(torch.stack(v)).to(DEVICE) for k, v in bufs.items()}
del bufs
print({k: tuple(v.shape) for k, v in FEAT.items()}, "| clips without body features:", no_body)
CLIPIDX = torch.tensor([[CIDX[c] for c in r] for r in DEV[['clip1', 'clip2', 'clip3']].values], device=DEVICE)
YB = torch.tensor(y_all, device=DEVICE)


def gather(idx):
    flat = idx.reshape(-1)
    return {k: v[flat].reshape(*idx.shape, *v.shape[1:]) for k, v in FEAT.items()}
'''),
("code", '''
class TemporalTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x):
        return self.transformer(x + self.pos_encoding[:, :x.size(1), :])


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                                     for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, query, key_values):
        x = query
        for attn, norm in zip(self.layers, self.norms):
            attended, _ = attn(x, key_values, key_values)
            x = norm(x + attended)
        return x


class Intra(nn.Module):
    """One clip -> one 512-d vector. Branches: face, whole frame (ori), body; optional text and audio."""

    def __init__(self, text, audio, mode, d=512, audio_dim=527):
        super().__init__()
        self.text, self.audio, self.mode = text, audio, mode
        if mode == 'trans':
            self.face_t, self.ori_t, self.body_t = (TemporalTransformer(d) for _ in range(3))
            self.type_fusion = CrossAttentionFusion(d)
            self.modality_fusion = CrossAttentionFusion(d)
        self.audio_proj = nn.Linear(audio_dim, d)

    def forward(self, face, ori, body, text, audio):
        if self.mode == 'trans':
            f, o, b = (m(x).mean(1, keepdim=True) for m, x in ((self.face_t, face), (self.ori_t, ori), (self.body_t, body)))
            video = self.type_fusion(f, torch.cat([f, o, b], 1))
        else:
            video = torch.stack([face.mean(1), ori.mean(1), body.mean(1)], 1).mean(1, keepdim=True)
        extra = [video]
        if self.text:
            extra.append(text.unsqueeze(1))
        if self.audio:
            extra.append(self.audio_proj(F.normalize(audio, dim=-1)).unsqueeze(1))
        if self.mode == 'trans':
            return self.modality_fusion(video, torch.cat(extra, 1)).squeeze(1)
        return torch.cat(extra, 1).mean(1)


class Inter(nn.Module):
    def __init__(self, mode, d=512):
        super().__init__()
        self.mode = mode
        if mode == 'lstm':
            self.lstm = nn.LSTM(input_size=d, hidden_size=d, num_layers=3, batch_first=False, dropout=0.1)
        if mode in ('lstm', 'trans'):
            self.pos_encoding = nn.Parameter(torch.randn(1, 3, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=d * 4, dropout=0.1, batch_first=True,
                                               norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, c1, c2, c3):
        if self.mode == 'avg':
            return torch.stack([c1, c2, c3], 1).mean(1)
        if self.mode == 'lstm':
            x = self.lstm(torch.stack([c1, c2, c3], 0))[0].permute(1, 0, 2)
        else:
            x = torch.stack([c1, c2, c3], 1)
        return self.transformer(x + self.pos_encoding).mean(1)


class Baseline(nn.Module):
    def __init__(self, text, audio, intra, inter, d=512, n_classes=7):
        super().__init__()
        self.intra, self.inter = Intra(text, audio, intra), Inter(inter)
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, d // 2), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d // 2, n_classes))

    def forward(self, ix):
        f = gather(CLIPIDX[ix])
        clips = [self.intra(f['face'][:, k], f['ori'][:, k], f['body'][:, k], f['text'][:, k],
                            f['audio'][:, k]) for k in range(3)]
        return self.classifier(self.inter(*clips))


def predict(model, ix, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            out.append(F.softmax(model(ix[i:i + bs]), -1).cpu())
    return torch.cat(out).numpy()


def train_one(cfg, tr, dev, te, seed):
    _, _, text, audio, intra, inter = cfg
    seed_all(seed)
    model = Baseline(text, audio, intra, inter).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    y_dev = YB[dev]
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            loss = F.cross_entropy(model(j), YB[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        p = predict(model, dev)
        sched.step(F.nll_loss(torch.log(torch.tensor(p) + 1e-9), y_dev.cpu()).item())
        u = war_uar(p.argmax(1), y_dev.cpu().numpy())[1]
        if u > best:
            best = u
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model, te), best


for cfg in CONFIGS:
    m = Baseline(*cfg[2:])
    print(f"row {cfg[1]:>2} {cfg[0]:<32} {sum(p.numel() for p in m.parameters()) / 1e6:5.1f}M parameters")
'''),
("markdown", "## Train on the fit rows, select on the selection rows, evaluate once on test"),
("code", '''
tr, dev, te = (torch.tensor(a, device=DEVICE) for a in (fit_rows, dev_rows, te_rows))
yt = y_all[te_rows]
PROBS, log = {}, []
t0 = time.time()
for cfg in CONFIGS:
    name = cfg[0]
    PROBS[name] = []
    for seed in SEEDS:
        p, sel = train_one(cfg, tr, dev, te, seed)
        PROBS[name].append(p)
        w, u = war_uar(p.argmax(1), yt)
        log.append({'row': cfg[1], 'config': name, 'seed': seed, 'sel_UAR': sel, 'test_UAR': u, 'test_WAR': w})
        print(f"row {cfg[1]:>2} {name:<32} seed {seed}: sel {sel:5.2f} | test UAR {u:5.2f} WAR {w:5.2f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        torch.cuda.empty_cache()
    PROBS[name] = np.stack(PROBS[name])
pd.DataFrame(log).to_csv(f"{OUT_DIR}/b2_{SOURCE}_per_seed.csv", index=False)
np.savez(f"{OUT_DIR}/b2_{SOURCE}_probs.npz", sample_id=DEV.sample_id.values[te_rows],
         **{f"row{c[1]}": PROBS[c[0]] for c in CONFIGS})
'''),
("markdown", "## Summary table (plain argmax; seed ensemble and mean over seeds)"),
("code", '''
rows = []
for cfg in CONFIGS:
    P = PROBS[cfg[0]]
    ens = war_uar(P.mean(0).argmax(1), yt)
    per = np.array([war_uar(p.argmax(1), yt) for p in P])
    rows.append({'row': cfg[1], 'config': cfg[0], 'UAR_ens': ens[1], 'WAR_ens': ens[0],
                 'UAR_mean': per[:, 1].mean(), 'WAR_mean': per[:, 0].mean()})
summ = pd.DataFrame(rows)
summ.to_csv(f"{OUT_DIR}/b2_{SOURCE}_summary.csv", index=False)
print(summ.round(2).to_string(index=False))
print("\\nreference (no body branch, same code, G10 / M3): Hi-EF test UAR 21.40 / WAR 32.52; MELD test UAR 15.50 / WAR 46.09 (ensembles)")
'''),
]

(HERE / "b1_body_features.ipynb").write_text(json.dumps(nb(B1), indent=1))
(HERE / "b2_vit_baselines_body.ipynb").write_text(json.dumps(nb(B2), indent=1))
print("written")
