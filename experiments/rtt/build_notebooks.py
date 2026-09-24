"""Build the Recognize-then-Transition (RtT) gate notebooks.

Run `python experiments/rtt/build_notebooks.py` to regenerate:
  - g1_llm_recognition.ipynb        (G1: OpenAI LLM reads dialogue I-III)
  - g2_recognizer_all_labels.ipynb  (G2: clip-III-only vs all-labeled-clip recognizer)

Both notebooks evaluate on the locked source-folder split and refuse to touch
the test split unless UNLOCK_TEST is set explicitly.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    return {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, src in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


# ---------------------------------------------------------------- shared code
COMMON = r'''
import os, json, math, random, time
import numpy as np
import pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
POL = ['positive', 'neutral', 'negative']
E2I = {e: i for i, e in enumerate(EMO)}
P2I = {p: i for i, p in enumerate(POL)}

# Reference numbers from the locked-split report (validation, 5-seed mean)
REPORT_REF = {'B1_full': (24.65, 35.79), 'T1_future_KL': (25.34, 35.65),
              'Frozen A recognizer (E_A)': (23.21, 33.41)}


def load_tables(annot_csv, split_csv):
    """annotation.csv has no header: 0 clip_id, 1 text, 5 polarity, 6 intensity, 7 emotion, 8 uncertainty."""
    ann = pd.read_csv(annot_csv, header=None, dtype=str).set_index(0)
    sp = pd.read_csv(split_csv, dtype=str)

    def text(c):
        t = ann.at[c, 1] if c in ann.index else None
        return t if isinstance(t, str) else ''

    for k in (1, 2, 3):
        sp[f't{k}'] = sp[f'clip{k}'].map(text)
    sp['yA'] = sp['clip3_emotion'].map(E2I)
    sp['yB'] = sp['clip4_emotion'].map(E2I)
    sp['pA'] = sp['clip3'].map(lambda c: P2I.get(ann.at[c, 5], -1))
    assert sp[['yA', 'yB']].notna().all().all(), 'missing A/B emotion labels'
    return ann, sp


def eval_rows(sp, split, unlock_test=False):
    if split == 'test' and not unlock_test:
        raise RuntimeError('Test split is locked. Set UNLOCK_TEST = True only for the final, preregistered run.')
    return sp[sp['split'] == split].reset_index(drop=True)


def war_uar(pred, y, k):
    pred, y = np.asarray(pred), np.asarray(y)
    war = (pred == y).mean() * 100
    uar = np.mean([(pred[y == c] == c).mean() * 100 for c in range(k) if (y == c).any()])
    return war, uar


def source_boot_ci(pred, y, src, k, n_boot=2000, seed=0):
    """95% CI by resampling whole source folders (episodes) with replacement."""
    pred, y, src = np.asarray(pred), np.asarray(y), np.asarray(src)
    rng = np.random.default_rng(seed)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        stats.append(war_uar(pred[idx], y[idx], k))
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
    return lo, hi


def report(name, pred, y, src, k=7):
    war, uar = war_uar(pred, y, k)
    lo, hi = source_boot_ci(pred, y, src, k)
    print(f'{name:<46} UAR {uar:5.2f} [{lo[1]:5.1f},{hi[1]:5.1f}]   WAR {war:5.2f} [{lo[0]:5.1f},{hi[0]:5.1f}]')
    return {'name': name, 'UAR': uar, 'WAR': war, 'UAR_lo': lo[1], 'UAR_hi': hi[1], 'WAR_lo': lo[0], 'WAR_hi': hi[0]}


def transition_tables(train_rows, alpha=1.0):
    """P(B | E_A) and P(B | E_A, P_A) estimated on TRAIN gold pairs, add-alpha smoothing."""
    T = np.full((7, 7), alpha)
    TP = np.full((7, 3, 7), alpha)
    for a, p, b in zip(train_rows['yA'], train_rows['pA'], train_rows['yB']):
        T[a, b] += 1
        if p >= 0:
            TP[a, p, b] += 1
    return T / T.sum(1, keepdims=True), TP / TP.sum(2, keepdims=True)


def rtt_forecast(pA_emo, T, pA_pol=None, TP=None):
    """Recognize-then-Transition: B distribution from A posteriors.
    Returns hard (argmax of transition row of argmax A) and soft (expected) B predictions."""
    hard = T[pA_emo.argmax(1)].argmax(1)
    if pA_pol is not None and TP is not None:
        pB = np.einsum('na,np,apb->nb', pA_emo, pA_pol, TP)  # assumes E_A and P_A posteriors independent
    else:
        pB = pA_emo @ T
    return hard, pB.argmax(1), pB
'''

# ---------------------------------------------------------------- G1: LLM
G1 = [
    ("markdown", r'''
# G1 — LLM recognizes Party A from dialogue text (Recognize-then-Transition gate)

**Question:** can an LLM reading subtitle lines I–III recognize A's emotion (clip III) better than the frozen
multimodal recognizer (val UAR 23.21)? If yes, plugging its posterior into a train-estimated transition table
P(B | E_A) should move the deployable forecast toward the Markov-oracle level (~28 UAR / ~40 WAR on val).

The LLM also gives a *direct* forecast of B (zero-shot baseline for the paper).

* Evaluates on **val only**; test stays locked.
* Clip IV text is never sent to the model.
* All responses are cached, so re-running costs nothing.
'''),
    ("code", r'''
!pip install -q openai
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
SPLIT_CSV = "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv"   # upload the locked manifest as a Kaggle dataset
G2_PROBS = "/kaggle/working/g2_val_probs_ALL.npz"                        # optional: output of the G2 notebook

MODEL = "gpt-4o-mini"      # change to the OpenAI model you want to use
N_SHOT_PER_CLASS = 0       # 0 = zero-shot; 1 = one train example per A-emotion class (7 examples)
TEMPERATURE = 0.0          # set to None for models that reject the temperature parameter
MAX_WORKERS = 8
EVAL_SPLIT = "val"
UNLOCK_TEST = False
CACHE = f"/kaggle/working/llm_cache_{MODEL.replace('/', '_')}_{N_SHOT_PER_CLASS}shot_{EVAL_SPLIT}.jsonl"
'''),
    ("code", COMMON + r'''
ANNOT_CSV = os.path.join(DATASET_DIR, "Hi-EF-20260829T071606Z-1-001", "Hi-EF", "annotation.csv")
ann, sp = load_tables(ANNOT_CSV, SPLIT_CSV)
train = sp[sp.split == 'train'].reset_index(drop=True)
ev = eval_rows(sp, EVAL_SPLIT, UNLOCK_TEST)
T, TP = transition_tables(train)
print(f"train {len(train)} | {EVAL_SPLIT} {len(ev)} | sources in {EVAL_SPLIT}: {ev.source_folder.nunique()}")
'''),
    ("code", r'''
from openai import OpenAI
try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("OPENAI_API_KEY", UserSecretsClient().get_secret("OPENAI_API_KEY"))
except Exception:
    pass  # fall back to an OPENAI_API_KEY environment variable
client = OpenAI()

SYSTEM = ("You are an expert annotator of emotions in TV drama dialogue. You only see subtitle text, "
          "which may contain transcription errors such as missing letters or apostrophes.")

TASK = """Below are three consecutive subtitle lines from a two-person interaction in a TV drama.
Lines [1] and [2] are preceding context (their speakers are not given). Line [3] is spoken by person A.
The next line (not shown) will be spoken by a different person, B, who is responding to A.
{examples}
[1] {t1}
[2] {t2}
[3] (A) {t3}

Tasks:
1. emotion_A: the emotion A expresses in line [3].
2. polarity_A: the polarity of A's interaction in line [3].
3. emotion_B_next: the emotion B will most likely express in the next line.

Emotion labels: angry, disgust, fear, happy, neutral, sad, surprise.
Polarity labels: positive, neutral, negative.
Return JSON only, giving a probability for every label (each group sums to 1):
{{"emotion_A": {{"angry": p, ...}}, "polarity_A": {{"positive": p, ...}}, "emotion_B_next": {{"angry": p, ...}}}}"""


def build_examples(k, seed=0):
    if k == 0:
        return ""
    rng = random.Random(seed)
    rows = []
    for e in range(7):
        pool = train[(train.yA == e) & (train.t3.str.len() > 0)]
        rows += pool.sample(n=min(k, len(pool)), random_state=rng.randint(0, 10**6)).to_dict('records')
    lines = ["\nExamples (with gold labels):"]
    for r in rows:
        lines.append(f"[1] {r['t1']}\n[2] {r['t2']}\n[3] (A) {r['t3']}\n"
                     f"-> emotion_A={EMO[r['yA']]}, polarity_A={POL[r['pA']] if r['pA'] >= 0 else 'neutral'}, "
                     f"emotion_B_next={EMO[r['yB']]}\n")
    lines.append("Now the item to annotate:")
    return "\n".join(lines)


EXAMPLES = build_examples(N_SHOT_PER_CLASS)


def call_llm(row, retries=5):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK.format(examples=EXAMPLES, t1=row['t1'] or '(no text)',
                                                    t2=row['t2'] or '(no text)', t3=row['t3'] or '(no text)')}]
    kwargs = dict(model=MODEL, messages=msgs, response_format={"type": "json_object"})
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE
    for attempt in range(retries):
        try:
            out = client.chat.completions.create(**kwargs)
            return json.loads(out.choices[0].message.content)
        except Exception as e:
            if "temperature" in str(e) and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            time.sleep(2 ** attempt)
    return None
'''),
    ("code", r'''
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

cache = {}
if os.path.exists(CACHE):
    with open(CACHE) as f:
        for line in f:
            d = json.loads(line)
            cache[d['sample_id']] = d['response']
todo = [r for r in ev.to_dict('records') if r['sample_id'] not in cache]
print(f"cached {len(cache)} | to query {len(todo)}")

with ThreadPoolExecutor(MAX_WORKERS) as pool, open(CACHE, 'a') as f:
    futs = {pool.submit(call_llm, r): r['sample_id'] for r in todo}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        sid, resp = futs[fut], fut.result()
        if resp is not None:
            cache[sid] = resp
            f.write(json.dumps({'sample_id': sid, 'response': resp}) + "\n")
print(f"responses: {sum(s in cache for s in ev.sample_id)}/{len(ev)}")
'''),
    ("code", r'''
def to_probs(d, labels):
    d = {str(k).strip().lower(): v for k, v in (d or {}).items()} if isinstance(d, dict) else {}
    p = np.array([float(d.get(l, 0) or 0) for l in labels], dtype=float)
    p = np.clip(p, 0, None)
    return p / p.sum() if p.sum() > 0 else np.full(len(labels), 1 / len(labels))

n_fail = sum(s not in cache for s in ev.sample_id)
pA = np.stack([to_probs(cache.get(s, {}).get('emotion_A'), EMO) for s in ev.sample_id])
pP = np.stack([to_probs(cache.get(s, {}).get('polarity_A'), POL) for s in ev.sample_id])
pB = np.stack([to_probs(cache.get(s, {}).get('emotion_B_next'), EMO) for s in ev.sample_id])
print(f"failed/missing responses (scored as uniform): {n_fail}")

src = ev.source_folder.values
rows = []
print(f"\n== Recognize A (clip III) — {EVAL_SPLIT} ==")
rows.append(report('LLM emotion_A', pA.argmax(1), ev.yA, src))
m = ev.pA.values >= 0
rows.append(report('LLM polarity_A (3-class)', pP[m].argmax(1), ev.pA.values[m], src[m], k=3))
print("   (reference: frozen multimodal A recognizer UAR 23.21 / WAR 33.41)")

print(f"\n== Forecast B (clip IV) — {EVAL_SPLIT} ==")
rows.append(report('LLM direct forecast (emotion_B_next)', pB.argmax(1), ev.yB, src))
hard, soft, _ = rtt_forecast(pA, T)
rows.append(report('RtT: LLM E_A -> P(B|E_A) hard', hard, ev.yB, src))
rows.append(report('RtT: LLM E_A -> P(B|E_A) soft', soft, ev.yB, src))
_, soft_p, _ = rtt_forecast(pA, T, pP, TP)
rows.append(report('RtT: LLM E_A,P_A -> P(B|E_A,P_A) soft', soft_p, ev.yB, src))
oracle = np.eye(7)[ev.yA.values]
rows.append(report('[oracle] gold E_A -> P(B|E_A) hard', rtt_forecast(oracle, T)[0], ev.yB, src))
for k, (u, w) in REPORT_REF.items():
    print(f"   (report reference) {k:<30} UAR {u:5.2f}   WAR {w:5.2f}")
pd.DataFrame(rows).to_csv(f"/kaggle/working/g1_results_{MODEL.replace('/', '_')}_{N_SHOT_PER_CLASS}shot.csv", index=False)
np.savez(f"/kaggle/working/g1_{EVAL_SPLIT}_probs_{MODEL.replace('/', '_')}_{N_SHOT_PER_CLASS}shot.npz",
         sample_id=ev.sample_id.values, pA=pA, pP=pP, pB=pB)
'''),
    ("markdown", r'''
## Optional: ensemble with the G2 multimodal recognizer
Runs only if the G2 notebook output (`g2_val_probs_ALL.npz`, seed-averaged A posteriors) is attached.
'''),
    ("code", r'''
if os.path.exists(G2_PROBS):
    g2 = np.load(G2_PROBS, allow_pickle=True)
    idx = {s: i for i, s in enumerate(g2['sample_id'])}
    pA_rec = g2['pA_mean'][[idx[s] for s in ev.sample_id]]
    for w in (0.25, 0.5, 0.75):
        mix = w * pA + (1 - w) * pA_rec
        print(f"\n-- LLM weight {w} --")
        report('RtT ensemble: A recognition', mix.argmax(1), ev.yA, src)
        report('RtT ensemble: forecast B (soft)', rtt_forecast(mix, T)[1], ev.yB, src)
else:
    print("G2 output not found; skipping ensemble.")
'''),
    ("markdown", r'''
## Contamination probe (for the paper's limitations section)
Hi-EF comes from a single, well-known series. This asks the LLM to name the show from 30 random TRAIN items.
A high hit rate means LLM results may partly reflect memorized plot knowledge.
'''),
    ("code", r'''
probe = train.sample(n=30, random_state=0)
hits = 0
for r in probe.to_dict('records'):
    q = (f"Which TV series are these subtitle lines from? Answer with the series name only, or 'unknown'.\n"
         f"[1] {r['t1']}\n[2] {r['t2']}\n[3] {r['t3']}")
    kw = dict(model=MODEL, messages=[{"role": "user", "content": q}])
    if TEMPERATURE is not None:
        kw["temperature"] = TEMPERATURE
    try:
        ans = client.chat.completions.create(**kw).choices[0].message.content.strip()
    except Exception as e:
        ans = f"error: {e}"
    hits += "house of cards" in ans.lower()
print(f"Named 'House of Cards' in {hits}/30 train items")
'''),
]

# ---------------------------------------------------------------- G2: recognizer
G2 = [
    ("markdown", r'''
# G2 — Does training the A recognizer on *all* labeled clips help? (Recognize-then-Transition gate)

Every labeled clip in Hi-EF is clip III or IV of some MCIS. The current A recognizer uses only clip III
(1,993 train rows); training sources contain **3,353** labeled clips (III ∪ IV). This notebook compares:

| Arm | Training clips |
|---|---|
| `III` | unique clip III of train MCIS |
| `ALL` | every labeled clip (III ∪ IV) in train sources |

Early stopping uses an **inner-dev** set (clip III of 5 held-out *train* sources), so the report-only
validation split is used only for the final numbers. Each arm's A posterior is then plugged into the
train-estimated transition table P(B | E_A) → deployable Recognize-then-Transition forecast.

Features are the cached CLIP / AudioCLIP tensors (`hi-ef-features-v2`). Test stays locked.
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv"   # upload the locked manifest as a Kaggle dataset
OUT_DIR = "/kaggle/working"

SEEDS = [42, 123, 456, 789, 1024]
ARMS = ["III", "ALL"]
N_INNER_DEV_SOURCES = 5
EPOCHS, PATIENCE, BATCH = 60, 8, 64
LR, WEIGHT_DECAY = 1e-4, 1e-5
POL_WEIGHT = 0.3           # auxiliary polarity loss weight
CLASS_BALANCED = False     # True: inverse-frequency class weights in the emotion CE
EVAL_SPLIT = "val"
UNLOCK_TEST = False
'''),
    ("code", COMMON + r'''
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOT_CSV = os.path.join(DATASET_DIR, "Hi-EF-20260829T071606Z-1-001", "Hi-EF", "annotation.csv")
ann, sp = load_tables(ANNOT_CSV, SPLIT_CSV)
train_all = sp[sp.split == 'train'].reset_index(drop=True)
ev = eval_rows(sp, EVAL_SPLIT, UNLOCK_TEST)
T, TP = transition_tables(train_all)   # transition tables use every train source (labels only)

train_sources = sorted(train_all.source_folder.unique())
inner_dev_sources = sorted(random.Random(0).sample(train_sources, N_INNER_DEV_SOURCES))
fit_sources = [s for s in train_sources if s not in inner_dev_sources]

# clip-level labels
lab = ann[ann[7].notna()].copy()
lab['ep'] = [c.split('/')[0] for c in lab.index]
lab['y_e'] = lab[7].map(E2I)
lab['y_p'] = lab[5].map(lambda p: P2I.get(p, -1))
lab = lab[lab.y_e.notna()]

fit_mcis = train_all[train_all.source_folder.isin(fit_sources)]
arm_clips = {
    'III': sorted(set(fit_mcis.clip3)),
    'ALL': sorted(lab.index[lab.ep.isin(fit_sources)]),
}
dev_clips = sorted(set(train_all[train_all.source_folder.isin(inner_dev_sources)].clip3))
eval_clips = list(ev.clip3)

assert not set(eval_clips) & set(arm_clips['ALL']), 'eval clip found in training clips'
assert not set(dev_clips) & set(arm_clips['ALL']), 'inner-dev clip found in training clips'
print(f"fit sources {len(fit_sources)} | inner-dev sources {inner_dev_sources}")
for a in ARMS:
    print(f"arm {a:<3}: {len(arm_clips[a])} training clips")
print(f"inner-dev clips {len(dev_clips)} | {EVAL_SPLIT} clips {len(eval_clips)}")
'''),
    ("code", r'''
def load_clip(cid):
    d = torch.load(os.path.join(FEATURES_DIR, cid.replace('/', '_') + '.pt'), map_location='cpu', weights_only=False)
    face = d['face_features'].float()
    fmask = d.get('face_valid_mask')
    fmask = torch.ones(face.shape[0], dtype=torch.bool) if fmask is None else torch.as_tensor(fmask).bool().reshape(-1)
    return {
        'face': face, 'fmask': fmask, 'ori': d['ori_features'].float(),
        'text': d['text_feature'].float().reshape(-1),
        'audio': d.get('audio_feature', torch.zeros(527)).float().reshape(-1),
        'afound': torch.tensor(bool(d.get('audio_found', True))),
    }

needed = sorted(set(arm_clips['ALL']) | set(arm_clips['III']) | set(dev_clips) | set(eval_clips))
FEATS = {c: load_clip(c) for c in tqdm(needed, desc='loading features')}


def batchify(clips):
    out = {k: torch.stack([FEATS[c][k] for c in clips]) for k in FEATS[clips[0]]}
    out['y_e'] = torch.tensor([int(lab.at[c, 'y_e']) for c in clips])
    out['y_p'] = torch.tensor([int(lab.at[c, 'y_p']) for c in clips])
    return out
'''),
    ("code", r'''
class TemporalEncoder(nn.Module):
    def __init__(self, d=512, n_frames=16, layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n_frames, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x, mask):  # mask: True = valid frame
        mask = mask.clone()
        mask[~mask.any(1), 0] = True          # keep one token for clips with no valid frame
        h = self.enc(x + self.pos[:, :x.size(1)], src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)


class ClipRecognizer(nn.Module):
    """Face/original temporal encoders + text/audio tokens -> 1-layer fusion Transformer -> E and P heads."""

    def __init__(self, d=512):
        super().__init__()
        self.face = TemporalEncoder(d)
        self.ori = TemporalEncoder(d)
        self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
        self.modality = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3))
        self.emo = nn.Linear(d, 7)
        self.pol = nn.Linear(d, 3)

    def forward(self, b):
        ori_mask = torch.ones(b['ori'].shape[:2], dtype=torch.bool, device=b['ori'].device)
        tokens = torch.stack([self.face(b['face'], b['fmask']), self.ori(b['ori'], ori_mask),
                              self.text(b['text']), self.audio(F.normalize(b['audio'], dim=-1))], 1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        valid[:, 3] = b['afound']
        h = self.fusion(tokens + self.modality, src_key_padding_mask=~valid)
        m = valid.unsqueeze(-1).float()
        h = self.head((h * m).sum(1) / m.sum(1))
        return self.emo(h), self.pol(h)
'''),
    ("code", r'''
def predict(model, clips, bs=256):
    model.eval()
    pe, pp = [], []
    with torch.no_grad():
        for i in range(0, len(clips), bs):
            b = {k: v.to(DEVICE) for k, v in batchify(clips[i:i + bs]).items()}
            le, lp = model(b)
            pe.append(F.softmax(le, -1).cpu()); pp.append(F.softmax(lp, -1).cpu())
    return torch.cat(pe).numpy(), torch.cat(pp).numpy()


def train_arm(arm, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    clips = arm_clips[arm]
    y_all = np.array([int(lab.at[c, 'y_e']) for c in clips])
    w = None
    if CLASS_BALANCED:
        cnt = np.bincount(y_all, minlength=7).astype(float)
        w = torch.tensor(cnt.sum() / (7 * np.maximum(cnt, 1)), dtype=torch.float, device=DEVICE)
    model = ClipRecognizer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dev_y = np.array([int(lab.at[c, 'y_e']) for c in dev_clips])
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(clips))
        for i in range(0, len(order), BATCH):
            b = {k: v.to(DEVICE) for k, v in batchify([clips[j] for j in order[i:i + BATCH]]).items()}
            le, lp = model(b)
            loss = F.cross_entropy(le, b['y_e'], weight=w)
            if (b['y_p'] >= 0).any():
                loss = loss + POL_WEIGHT * F.cross_entropy(lp, b['y_p'], ignore_index=-1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        dev_uar = war_uar(predict(model, dev_clips)[0].argmax(1), dev_y, 7)[1]
        if dev_uar > best:
            best, bad = dev_uar, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best
'''),
    ("code", r'''
src = ev.source_folder.values
results, probs = [], {a: [] for a in ARMS}
for arm in ARMS:
    for seed in SEEDS:
        model, dev_uar = train_arm(arm, seed)
        pA, pP = predict(model, eval_clips)
        probs[arm].append((pA, pP))
        wA, uA = war_uar(pA.argmax(1), ev.yA, 7)
        hard, soft, _ = rtt_forecast(pA, T)
        _, soft_p, _ = rtt_forecast(pA, T, pP, TP)
        r = {'arm': arm, 'seed': seed, 'inner_dev_UAR_A': dev_uar, 'A_UAR': uA, 'A_WAR': wA}
        for name, pred in [('B_hard', hard), ('B_soft', soft), ('B_soft_EP', soft_p)]:
            wB, uB = war_uar(pred, ev.yB, 7)
            r[f'{name}_UAR'], r[f'{name}_WAR'] = uB, wB
        results.append(r)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})

res = pd.DataFrame(results)
res.to_csv(f"{OUT_DIR}/g2_results_per_seed.csv", index=False)
print("\n== mean ± std over seeds ==")
print(res.drop(columns='seed').groupby('arm').agg(['mean', 'std']).round(2).T.to_string())
'''),
    ("code", r'''
print(f"== Seed-averaged posteriors, {EVAL_SPLIT}, with source-bootstrap 95% CI ==")
for arm in ARMS:
    pA = np.mean([p[0] for p in probs[arm]], 0)
    pP = np.mean([p[1] for p in probs[arm]], 0)
    print(f"\n-- arm {arm} --")
    report('Recognize A (emotion)', pA.argmax(1), ev.yA, src)
    report('RtT forecast B: P(B|E_A) hard', rtt_forecast(pA, T)[0], ev.yB, src)
    report('RtT forecast B: P(B|E_A) soft', rtt_forecast(pA, T)[1], ev.yB, src)
    report('RtT forecast B: P(B|E_A,P_A) soft', rtt_forecast(pA, T, pP, TP)[1], ev.yB, src)
    np.savez(f"{OUT_DIR}/g2_{EVAL_SPLIT}_probs_{arm}.npz", sample_id=ev.sample_id.values, pA_mean=pA, pP_mean=pP)

print("\n-- references --")
report('[oracle] gold E_A -> P(B|E_A) hard', rtt_forecast(np.eye(7)[ev.yA.values], T)[0], ev.yB, src)
for k, (u, w) in REPORT_REF.items():
    print(f"   (report reference) {k:<30} UAR {u:5.2f}   WAR {w:5.2f}")
'''),
]


if __name__ == "__main__":
    for name, cells in [("g1_llm_recognition.ipynb", G1), ("g2_recognizer_all_labels.ipynb", G2)]:
        (HERE / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False))
        print("wrote", HERE / name)
