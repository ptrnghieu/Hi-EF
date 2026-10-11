"""Build b3_released_code_baseline.ipynb: the Hi-EF authors' released three-clip model (model_sirv_second.py + train_mmmm_second.py)
ported to precomputed CLIP/AudioCLIP features, trained on our split protocol (Hi-EF G10 protocol or MELD M3 protocol)."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    return {"cells": [{"cell_type": k, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if k == "code" else {})} for k, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


CELLS = [
("markdown", """
# B3 — The released Hi-EF three-clip model, ported to precomputed features, on our split protocol

A line-by-line port of the authors' released model for clips I, II, III → IV (`model_sirv_second.py`, trained by
`train_mmmm_second.py`), so that the baseline can be compared with our reimplementation (B2). Inputs are the same
precomputed CLIP ViT-B/32 face / whole-frame / text features and AudioCLIP features as in G10 and B2 (the frozen encoders of the
released model are replaced by these features); the split protocol is ours (Hi-EF: G10 protocol, MELD: M3 protocol).

What is kept from the released code:
* two 12-layer temporal transformers (face, whole frame; width 512, 8 heads, CLIP-style blocks with TF-style LayerNorm
  eps 1e-12, QuickGELU, learned frame position embeddings, additive key mask), residual connection and masked mean pooling;
  **no body branch** (the released code passes `p_frames = None`, so the body vector is zero);
* two 12-layer query/key-value transformers (`a_`, `b_transformerClip`) and the fusion of image, text and audio
  (audio: a trainable `Linear(527, 512)` on the AudioCLIP vector);
* a 5-layer LSTM over the three clip vectors (last output), L2 normalisation, an adapter (bottleneck 128, residual 0.2)
  plus a bias-free linear layer on the normalised vector, logits multiplied by exp(4.60517) = 100;
* Adam (lr 2e-4, weight decay 1e-6), ReduceLROnPlateau (factor 0.1, patience 5, threshold 0.001) on the validation loss,
  batch 32, 50 epochs, cross-entropy.

Two quirks of the released code are **reproduced by default** and can be switched off (variant `fixed`):
1. `o_visual_output = f_video_features`: the whole-frame branch re-reads the *face* features (flag `ORI_FROM_FACE`);
2. `QKVTransformerClip.forward` returns `resblocks((q, kv))[0]`, which is the unchanged query, so the two 12-layer
   cross-attention stacks have no effect on the output (flag `QKV_RETURNS_QUERY`); the clip vector is then a fixed
   weighted sum of the branch vectors.

Evaluation as in B2 (fit / early-stopping / test rows, plain argmax UAR and WAR of the seed ensemble). Because the
released script selects the checkpoint by validation loss but tests the final-epoch model, three selections are
reported per run: best validation loss, last epoch, best validation UAR. **Reads the test split** (an additional access).
"""),
("code", '''
# ======== CONFIG ========
import os, glob
SOURCE = "hief"                       # "hief" or "meld"
UNLOCK_TEST = True
OUT_DIR = "/kaggle/working"


def find_one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"attach the dataset holding {pattern}")
    return hits[0]


if SOURCE == "hief":
    FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
    SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
    SEEDS = [42, 123, 456, 789, 1024]
else:
    FEATURES_DIR = find_one("meld_features")
    SPLIT_CSV = find_one("meld_split.csv")
    SEEDS = [42, 123, 456]
N_INNER_DEV, SELECT_SEED = 5, 2026
LR, WEIGHT_DECAY = 2e-4, 1e-6          # released train script
EPOCHS, BATCH = 50, 32
LOGIT = 4.60517
DEBUG_PER_EPISODE = None
# name, ORI_FROM_FACE, QKV_RETURNS_QUERY
VARIANTS = [("released", True, True), ("fixed", False, False)]
'''),
("code", '''
import random, time, glob
import numpy as np, pandas as pd
from collections import OrderedDict
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

all_clips = sorted(set(DEV[['clip1', 'clip2', 'clip3']].values.ravel()))
CIDX = {c: i for i, c in enumerate(all_clips)}
bufs = {k: [] for k in ('face', 'fmask', 'ori', 'text', 'audio')}
for c in tqdm(all_clips, desc='loading features'):
    d = torch.load(os.path.join(FEATURES_DIR, c.replace('/', '_') + '.pt'), map_location='cpu', weights_only=False)
    face = d['face_features'].float()
    fm = d.get('face_valid_mask')
    bufs['face'].append(face)
    bufs['fmask'].append(torch.ones(face.shape[0], dtype=torch.bool) if fm is None else torch.as_tensor(fm).bool().reshape(-1))
    bufs['ori'].append(d['ori_features'].float())
    bufs['text'].append(d['text_feature'].float().reshape(-1))
    bufs['audio'].append(d.get('audio_feature', torch.zeros(527)).float().reshape(-1))
FEAT = {k: torch.nan_to_num(torch.stack(v).float()).to(DEVICE) for k, v in bufs.items()}
del bufs
print({k: tuple(v.shape) for k, v in FEAT.items()})
CLIPIDX = torch.tensor([[CIDX[c] for c in r] for r in DEV[['clip1', 'clip2', 'clip3']].values], device=DEVICE)
YB = torch.tensor(y_all, device=DEVICE)


def gather(idx):
    flat = idx.reshape(-1)
    return {k: v[flat].reshape(*idx.shape, *v.shape[1:]) for k, v in FEAT.items()}
'''),
("markdown", "## The released model (port)"),
("code", '''
FEA, HEADS, LAYERS = 512, 8, 12


class LayerNormTF(nn.Module):                       # released `LayerNorm` (TF style, eps inside the square root)
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)); self.bias = nn.Parameter(torch.zeros(hidden_size)); self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.weight * ((x - u) / torch.sqrt(s + self.eps)) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def mlp(d):
    return nn.Sequential(OrderedDict([("c_fc", nn.Linear(d, d * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(d * 4, d))]))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_head); self.ln_1 = LayerNormTF(d); self.mlp = mlp(d); self.ln_2 = LayerNormTF(d)
        self.n_head = n_head

    def forward(self, tup):
        x, m = tup
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=m.repeat_interleave(self.n_head, dim=0))[0]
        return (x + self.mlp(self.ln_2(x)), m)


class TransformerClip(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d, heads) for _ in range(layers)])

    def forward(self, x, m):
        return self.resblocks((x, m))[0]


class QKVResidualAttentionBlock(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_head); self.ln_11 = LayerNormTF(d); self.ln_12 = LayerNormTF(d)
        self.mlp = mlp(d); self.ln_2 = LayerNormTF(d)

    def forward(self, tup):
        q, kv = tup
        k = self.ln_12(kv)
        x = kv + self.attn(self.ln_11(q), k, k, need_weights=False)[0]
        return (q, x + self.mlp(self.ln_2(x)))          # released: returns (q, x)


class QKVTransformerClip(nn.Module):
    def __init__(self, d, layers, heads, returns_query):
        super().__init__()
        self.resblocks = nn.Sequential(*[QKVResidualAttentionBlock(d, heads) for _ in range(layers)])
        self.idx = 0 if returns_query else 1           # released code takes [0] = the query

    def forward(self, q, kv):
        return self.resblocks((q, kv))[self.idx]


def masked_mean(x, valid):                              # released `_mean_pooling_for_similarity_visual`
    v = valid.to(dtype=torch.float).unsqueeze(-1)
    s = v.sum(1)
    s[s == 0.] = 1.
    return (x * v).sum(1) / s


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super().__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(nn.Linear(c_in, c_in // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(c_in // reduction, c_in, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.residual_ratio * self.fc(x) + (1 - self.residual_ratio) * x


class Released(nn.Module):
    def __init__(self, ori_from_face, qkv_returns_query):
        super().__init__()
        self.ori_from_face = ori_from_face
        self.f_pos, self.o_pos = nn.Embedding(16, FEA), nn.Embedding(16, FEA)
        self.f_tf, self.o_tf = TransformerClip(FEA, LAYERS, HEADS), TransformerClip(FEA, LAYERS, HEADS)
        self.a_tf = QKVTransformerClip(FEA, LAYERS, HEADS, qkv_returns_query)
        self.b_tf = QKVTransformerClip(FEA, LAYERS, HEADS, qkv_returns_query)
        self.transshape = nn.Linear(527, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=5, bidirectional=False)
        self.head = nn.Sequential(Adapter(512, residual_ratio=0.2), nn.Linear(512, 7, bias=False))

    def branch(self, x, valid, pos, tf):
        L = x.size(1)
        mask = ((1.0 - valid.float()).unsqueeze(1) * -1000000.0).expand(-1, L, -1)
        h = x + pos(torch.arange(L, device=x.device)).unsqueeze(0)
        h = tf(h.permute(1, 0, 2), mask).permute(1, 0, 2) + x
        return masked_mean(h, valid)

    def clip_vector(self, face, ori, fvalid, text, audio):
        B = face.size(0)
        f = self.branch(face, fvalid, self.f_pos, self.f_tf)
        ovalid = torch.ones(B, ori.size(1), device=face.device)
        o = self.branch(face if self.ori_from_face else ori, ovalid, self.o_pos, self.o_tf)
        img0 = torch.stack([f, o, torch.zeros_like(f)], 0)                      # [3, B, 512]; body branch: zeros
        v3 = torch.tensor([1., 1., 0.], device=face.device).repeat(B, 1)
        img = self.a_tf(f.unsqueeze(0).repeat(3, 1, 1), img0) + img0
        img = masked_mean(img.permute(1, 0, 2), v3)                              # [B, 512]
        mm0 = torch.stack([img, text, self.transshape(audio)], 0)
        mm = self.b_tf(img.unsqueeze(0).repeat(3, 1, 1), mm0) + mm0
        return masked_mean(mm.permute(1, 0, 2), torch.ones(B, 3, device=face.device))

    def forward(self, ix):
        f = gather(CLIPIDX[ix])
        clips = [self.clip_vector(f['face'][:, k], f['ori'][:, k], f['fmask'][:, k], f['text'][:, k], f['audio'][:, k]) for k in range(3)]
        out, _ = self.lstm(torch.stack(clips, 0))
        x = F.normalize(out[-1], dim=-1)
        x = F.normalize(x, dim=1)                                                # LogitHead normalises again
        return self.head(x) * float(np.exp(LOGIT))


for name, a, b_ in VARIANTS:
    print(name, f"{sum(p.numel() for p in Released(a, b_).parameters()) / 1e6:.1f}M parameters (the two QKV stacks are idle when QKV_RETURNS_QUERY)")
'''),
("code", '''
def predict(model, ix, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            out.append(F.softmax(model(ix[i:i + bs]), -1).cpu())
    return torch.cat(out).numpy()


def train_one(variant, tr, dev, te, seed):
    _, a, b_ = variant
    seed_all(seed)
    model = Released(a, b_).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, threshold=0.001)
    y_dev = YB[dev].cpu().numpy()
    best_loss, best_uar = (1e9, None), (-1, None)
    for ep in range(EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            loss = F.cross_entropy(model(j), YB[j])
            opt.zero_grad(); loss.backward(); opt.step()
        p = predict(model, dev)
        vloss = float(F.nll_loss(torch.log(torch.tensor(p) + 1e-9), torch.tensor(y_dev)))
        u = war_uar(p.argmax(1), y_dev)[1]
        sched.step(vloss)
        snap = lambda: {k: v.detach().clone() for k, v in model.state_dict().items()}
        if vloss < best_loss[0]:
            best_loss = (vloss, snap())
        if u > best_uar[0]:
            best_uar = (u, snap())
    out = {'last': predict(model, te)}
    for nm, (_, st) in (('valloss', best_loss), ('valuar', best_uar)):
        model.load_state_dict(st)
        out[nm] = predict(model, te)
    return out, (best_loss[0], best_uar[0])


tr, dev, te = (torch.tensor(a, device=DEVICE) for a in (fit_rows, dev_rows, te_rows))
yt = y_all[te_rows]
SEL = ['valloss', 'last', 'valuar']
PROBS = {(v[0], s): [] for v in VARIANTS for s in SEL}
log, t0 = [], time.time()
for v in VARIANTS:
    for seed in SEEDS:
        out, (bl, bu) = train_one(v, tr, dev, te, seed)
        for s in SEL:
            PROBS[(v[0], s)].append(out[s])
            w, u = war_uar(out[s].argmax(1), yt)
            log.append({'variant': v[0], 'selection': s, 'seed': seed, 'dev_best_loss': bl, 'dev_best_UAR': bu, 'test_UAR': u, 'test_WAR': w})
        print(f"{v[0]:<9} seed {seed}: " + " | ".join(f"{s} UAR {log[-3 + k]['test_UAR']:5.2f} WAR {log[-3 + k]['test_WAR']:5.2f}" for k, s in enumerate(SEL))
              + f" | dev loss {bl:.3f} UAR {bu:5.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.cuda.empty_cache()
    for s in SEL:
        PROBS[(v[0], s)] = np.stack(PROBS[(v[0], s)])
pd.DataFrame(log).to_csv(f"{OUT_DIR}/b3_{SOURCE}_per_seed.csv", index=False)
np.savez(f"{OUT_DIR}/b3_{SOURCE}_probs.npz", sample_id=DEV.sample_id.values[te_rows], **{f"{k[0]}_{k[1]}": v for k, v in PROBS.items()})
'''),
("markdown", "## Summary (plain argmax; seed ensemble and mean over seeds)"),
("code", '''
rows = []
for (name, s), P in PROBS.items():
    ens = war_uar(P.mean(0).argmax(1), yt)
    per = np.array([war_uar(p.argmax(1), yt) for p in P])
    rows.append({'variant': name, 'selection': s, 'UAR_ens': ens[1], 'WAR_ens': ens[0], 'UAR_mean': per[:, 1].mean(),
                 'UAR_sd': per[:, 1].std(), 'WAR_mean': per[:, 0].mean()})
S = pd.DataFrame(rows)
S.to_csv(f"{OUT_DIR}/b3_{SOURCE}_summary.csv", index=False)
print(S.round(2).to_string(index=False))
print("\\nreferences on the same protocol (B2 / G10 / M3): Baseline with body 21.30 / 31.30 (Hi-EF), 15.27 / 45.34 (MELD); "
      "without body 21.40 / 32.52, 15.50 / 46.09; best retrained configuration 24.23 / 32.76, 16.42 / 38.85")
'''),
]
(HERE / "b3_released_code_baseline.ipynb").write_text(json.dumps(nb(CELLS), indent=1))
print("written")
