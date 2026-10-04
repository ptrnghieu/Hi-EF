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
            ERRORS.append(f"{type(e).__name__}: {e}")
            if "temperature" in str(e) and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            time.sleep(2 ** attempt)
    return None


# Pre-flight: one real call so key / model / Internet problems surface here instead of as uniform scores
ERRORS = []
probe = call_llm(train.iloc[0].to_dict(), retries=2)
if probe is None:
    raise RuntimeError("Pre-flight call failed. Check: Kaggle 'Internet on', secret OPENAI_API_KEY attached, "
                       f"MODEL name.\nLast error: {ERRORS[-1] if ERRORS else 'unknown'}")
print("pre-flight OK:", json.dumps(probe)[:200])
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
n_ok = sum(s in cache for s in ev.sample_id)
print(f"responses: {n_ok}/{len(ev)}")
if ERRORS:
    print(f"{len(ERRORS)} API errors; first ones:", *sorted(set(ERRORS))[:3], sep="\n  ")
if n_ok < 0.95 * len(ev):
    raise RuntimeError("More than 5% of items have no response; re-run this cell (cached items are skipped) "
                       "before scoring, otherwise missing items are scored as uniform.")
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


# ---------------------------------------------------------------- G3: trajectory forecaster
G3 = [
    ("markdown", r'''
# G3 — Two-party affective trajectory forecaster

Label-level analysis on the locked split showed:

* gold labels of clip III explain only ~6–10% of the uncertainty about B, so routing through A's label caps out;
* clip II carries about as much information about B as A does (B's own previous turn / scene mood);
* the recognizer's errors are mostly valence flips, and its confidence is informative;
* label ambiguity ("uncertain") concentrates in fear / disgust / sad.

So instead of *replacing* the end-to-end model with "recognize A → table", this notebook **adds** a
soft affective trajectory to it:

1. **Stage 1 — cross-fitted recognizer.** The G2 recognizer (all labeled clips) is trained in 5 folds over the
   37 training episodes. Every training MCIS gets recognizer posteriors for clips I, II, III from a model that
   never saw its episode; validation uses the average of the 5 fold models.
   Per clip: emotion posterior (7) + polarity posterior (3) + max-prob + entropy = 12 numbers.
2. **Stage 2 — forecaster arms** (same seeds, same early stopping):

| Arm | Raw clip encoder (B1) | Trajectory I–III | Certainty-weighted loss |
|---|---|---|---|
| `B1` | ✓ | | |
| `B1+certw` | ✓ | | ✓ |
| `traj_only` | | ✓ | |
| `B1+traj` | ✓ | ✓ | |
| `B1+traj+certw` | ✓ | ✓ | ✓ |

Early stopping uses **inner-dev** episodes held out from training (not the report-only validation split).
Validation reports per-seed mean ± std, seed-ensemble scores with source-bootstrap CIs, **paired** bootstrap
differences against `B1`, and a secondary score on rows whose B label is marked *certain*. Test stays locked.
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"

SEEDS = [42, 123, 456, 789, 1024]
ARMS = ["B1", "B1+certw", "traj_only", "B1+traj", "B1+traj+certw"]
N_FOLDS = 5                 # cross-fitting folds over training episodes (stage 1)
N_INNER_DEV_SOURCES = 5     # training episodes held out for forecaster early stopping (stage 2)
REC_SEED = 42
REC_EPOCHS, FC_EPOCHS, PATIENCE = 60, 50, 8
REC_BATCH, FC_BATCH = 64, 32
LR, WEIGHT_DECAY = 1e-4, 1e-5
POL_WEIGHT = 0.3
CERT_WEIGHTS = {'1': 1.0, '2': 0.75, '3': 0.5}   # annotation uncertainty of the TARGET clip IV
SELECT_ON = "inner_dev"     # "val" reproduces the report's protocol (selection on the reported split)
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

train_sources = sorted(train_all.source_folder.unique())
rng = random.Random(0)
shuffled = train_sources[:]
rng.shuffle(shuffled)
FOLD_OF = {s: i % N_FOLDS for i, s in enumerate(shuffled)}
inner_dev_sources = sorted(random.Random(1).sample(train_sources, N_INNER_DEV_SOURCES))

lab = ann[ann[7].notna()].copy()
lab['ep'] = [c.split('/')[0] for c in lab.index]
lab['y_e'] = lab[7].map(E2I)
lab['y_p'] = lab[5].map(lambda p: P2I.get(p, -1))
lab = lab[lab.y_e.notna()]

for d in (train_all, ev):
    d['w_cert'] = d['clip4'].map(lambda c: CERT_WEIGHTS.get(str(ann.at[c, 8]), 1.0))
    d['unc_B'] = d['clip4'].map(lambda c: str(ann.at[c, 8]))
print(f"train {len(train_all)} | {EVAL_SPLIT} {len(ev)} | folds: "
      f"{[sorted(s for s in train_sources if FOLD_OF[s] == k) for k in range(N_FOLDS)]}")
print(f"forecaster inner-dev episodes: {inner_dev_sources}")
'''),
    ("code", r'''
# ---- load every clip used by any MCIS (I-IV) once, keep it on the GPU
all_clips = sorted(set(sp[['clip1', 'clip2', 'clip3', 'clip4']].values.ravel()) & set(
    f[:-3].replace('_', '/', 1) for f in os.listdir(FEATURES_DIR) if f.endswith('.pt')))
CIDX = {c: i for i, c in enumerate(all_clips)}
missing = [c for c in set(train_all[['clip1', 'clip2', 'clip3']].values.ravel()) | set(ev[['clip1', 'clip2', 'clip3']].values.ravel())
           if c not in CIDX]
assert not missing, f"{len(missing)} clips without features, e.g. {missing[:3]}"

bufs = {k: [] for k in ('face', 'fmask', 'ori', 'text', 'audio', 'afound')}
for c in tqdm(all_clips, desc='loading features'):
    d = torch.load(os.path.join(FEATURES_DIR, c.replace('/', '_') + '.pt'), map_location='cpu', weights_only=False)
    face = d['face_features'].float()
    fm = d.get('face_valid_mask')
    bufs['face'].append(face)
    bufs['fmask'].append(torch.ones(face.shape[0], dtype=torch.bool) if fm is None else torch.as_tensor(fm).bool().reshape(-1))
    bufs['ori'].append(d['ori_features'].float())
    bufs['text'].append(d['text_feature'].float().reshape(-1))
    bufs['audio'].append(d.get('audio_feature', torch.zeros(527)).float().reshape(-1))
    bufs['afound'].append(torch.tensor(bool(d.get('audio_found', True))))
FEAT = {k: torch.stack(v).to(DEVICE) for k, v in bufs.items()}
del bufs
print({k: tuple(v.shape) for k, v in FEAT.items()})


def gather(idx):
    """idx: LongTensor of clip indices (any shape) -> dict of feature tensors with that leading shape."""
    flat = idx.reshape(-1)
    return {k: v[flat].reshape(*idx.shape, *v.shape[1:]) for k, v in FEAT.items()}
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
        mask[~mask.any(1), 0] = True
        h = self.enc(x + self.pos[:, :x.size(1)], src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)


class ClipEncoder(nn.Module):
    """Face/original temporal encoders + text/audio tokens -> 1-layer fusion Transformer -> one 512-d vector."""

    def __init__(self, d=512):
        super().__init__()
        self.face, self.ori = TemporalEncoder(d), TemporalEncoder(d)
        self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
        self.modality = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        ori_mask = torch.ones(b['ori'].shape[:2], dtype=torch.bool, device=b['ori'].device)
        tokens = torch.stack([self.face(b['face'], b['fmask']), self.ori(b['ori'], ori_mask),
                              self.text(b['text']), self.audio(F.normalize(b['audio'], dim=-1))], 1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        valid[:, 3] = b['afound']
        h = self.fusion(tokens + self.modality, src_key_padding_mask=~valid)
        m = valid.unsqueeze(-1).float()
        return self.norm((h * m).sum(1) / m.sum(1))


class ClipRecognizer(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.enc = ClipEncoder(d)
        self.drop = nn.Dropout(0.3)
        self.emo, self.pol = nn.Linear(d, 7), nn.Linear(d, 3)

    def forward(self, b):
        h = self.drop(self.enc(b))
        return self.emo(h), self.pol(h)


N_REC = 12   # 7 emotion probs + 3 polarity probs + max prob + entropy


class Forecaster(nn.Module):
    def __init__(self, use_raw=True, use_traj=False, d=512, positions=None):
        super().__init__()
        self.use_raw, self.use_traj = use_raw, use_traj
        self.positions = positions   # clip positions (0=I, 1=II, 2=III); None = the last n clips
        self.enc = ClipEncoder(d) if use_raw else None
        self.traj = nn.Sequential(nn.LayerNorm(N_REC), nn.Linear(N_REC, d), nn.GELU(), nn.Linear(d, d)) if use_traj else None
        self.clip_pos = nn.Parameter(torch.randn(1, 3, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.inter = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, d // 2), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(d // 2, 7))

    def forward(self, clip_idx, rec):  # clip_idx [B,n], rec [B,n,N_REC], n <= 3 clips in temporal order
        B, n = clip_idx.shape
        tok = 0
        if self.use_raw:
            feats = gather(clip_idx)
            flat = {k: v.reshape(B * n, *v.shape[2:]) for k, v in feats.items()}
            tok = self.enc(flat).reshape(B, n, -1)
        if self.use_traj:
            tok = tok + self.traj(rec)
        pos = self.clip_pos[:, list(self.positions)] if self.positions is not None else self.clip_pos[:, 3 - n:]
        h = self.inter(tok + pos)
        return self.head(h.mean(1))
'''),
    ("markdown", r'''
## Stage 1 — cross-fitted recognizer posteriors for clips I, II, III
'''),
    ("code", r'''
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def rec_predict(model, clips, bs=512):
    model.eval()
    pe, pp = [], []
    with torch.no_grad():
        for i in range(0, len(clips), bs):
            idx = torch.tensor([CIDX[c] for c in clips[i:i + bs]], device=DEVICE)
            le, lp = model(gather(idx))
            pe.append(F.softmax(le, -1).cpu()); pp.append(F.softmax(lp, -1).cpu())
    return torch.cat(pe).numpy(), torch.cat(pp).numpy()


def train_recognizer(fit_sources, dev_sources, seed):
    seed_all(seed)
    clips = sorted(lab.index[lab.ep.isin(fit_sources)])
    dev = sorted(set(train_all[train_all.source_folder.isin(dev_sources)].clip3))
    y_e = torch.tensor([int(lab.at[c, 'y_e']) for c in clips], device=DEVICE)
    y_p = torch.tensor([int(lab.at[c, 'y_p']) for c in clips], device=DEVICE)
    cidx = torch.tensor([CIDX[c] for c in clips], device=DEVICE)
    dev_y = np.array([int(lab.at[c, 'y_e']) for c in dev])
    model = ClipRecognizer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, bad = -1, None, 0
    for ep in range(REC_EPOCHS):
        model.train()
        perm = torch.randperm(len(clips), device=DEVICE)
        for i in range(0, len(perm), REC_BATCH):
            j = perm[i:i + REC_BATCH]
            le, lp = model(gather(cidx[j]))
            loss = F.cross_entropy(le, y_e[j])
            if (y_p[j] >= 0).any():
                loss = loss + POL_WEIGHT * F.cross_entropy(lp, y_p[j], ignore_index=-1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        dev_uar = war_uar(rec_predict(model, dev)[0].argmax(1), dev_y, 7)[1]
        if dev_uar > best:
            best, bad = dev_uar, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best


def rec_vector(pe, pp):
    ent = -(pe * np.log(np.clip(pe, 1e-9, 1))).sum(1, keepdims=True)
    return np.concatenate([pe, pp, pe.max(1, keepdims=True), ent], 1).astype(np.float32)
'''),
    ("code", r'''
REC = {}                       # clip id -> 12-d trajectory vector (out-of-fold for training episodes)
fold_models = []
ev_ctx = sorted(set(ev[['clip1', 'clip2', 'clip3']].values.ravel()))
ev_acc = np.zeros((len(ev_ctx), N_REC), dtype=np.float32)
for k in range(N_FOLDS):
    held = [s for s in train_sources if FOLD_OF[s] == k]
    rest = [s for s in train_sources if FOLD_OF[s] != k]
    rdev = sorted(random.Random(100 + k).sample(rest, 4))
    fit = [s for s in rest if s not in rdev]
    model, dev_uar = train_recognizer(fit, rdev, REC_SEED + k)
    assert not set(lab.index[lab.ep.isin(held)]) & set(lab.index[lab.ep.isin(fit)])
    held_clips = sorted(set(train_all[train_all.source_folder.isin(held)][['clip1', 'clip2', 'clip3']].values.ravel()))
    pe, pp = rec_predict(model, held_clips)
    REC.update(zip(held_clips, rec_vector(pe, pp)))
    pe, pp = rec_predict(model, ev_ctx)
    ev_acc += rec_vector(pe, pp) / N_FOLDS   # average of fold models for the evaluation split
    print(f"fold {k}: held {held} | recognizer inner-dev UAR {dev_uar:.2f}")
REC.update(zip(ev_ctx, ev_acc))

for name, d in [('train (OOF)', train_all), (EVAL_SPLIT, ev)]:
    pe = np.stack([REC[c][:7] for c in d.clip3])
    w, u = war_uar(pe.argmax(1), d.yA, 7)
    print(f"recognizer on clip III, {name}: UAR {u:.2f}  WAR {w:.2f}")
np.savez(f"{OUT_DIR}/g3_rec_trajectory.npz", clips=np.array(list(REC)), vecs=np.stack(list(REC.values())))
'''),
    ("markdown", r'''
## Stage 2 — forecaster arms
'''),
    ("code", r'''
fc_train = train_all[~train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
fc_dev = train_all[train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
sel_rows = ev if SELECT_ON == "val" else fc_dev


def tensors(d):
    idx = torch.tensor([[CIDX[c] for c in r] for r in d[['clip1', 'clip2', 'clip3']].values], device=DEVICE)
    rec = torch.tensor(np.stack([np.stack([REC[c] for c in r]) for r in d[['clip1', 'clip2', 'clip3']].values]), device=DEVICE)
    y = torch.tensor(d.yB.values, device=DEVICE)
    w = torch.tensor(d.w_cert.values, dtype=torch.float, device=DEVICE)
    return idx, rec, y, w


T_TR, T_SEL, T_EV = tensors(fc_train), tensors(sel_rows), tensors(ev)


def fc_predict(model, T, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(T[0]), bs):
            out.append(F.softmax(model(T[0][i:i + bs], T[1][i:i + bs]), -1).cpu())
    return torch.cat(out).numpy()


def train_forecaster(arm, seed):
    seed_all(seed)
    model = Forecaster(use_raw='B1' in arm, use_traj='traj' in arm).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    idx, rec, y, w = T_TR
    use_w = 'certw' in arm
    best, best_state, bad = -1, None, 0
    for ep in range(FC_EPOCHS):
        model.train()
        perm = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(perm), FC_BATCH):
            j = perm[i:i + FC_BATCH]
            loss = F.cross_entropy(model(idx[j], rec[j]), y[j], reduction='none')
            loss = (loss * w[j]).sum() / w[j].sum() if use_w else loss.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sel_uar = war_uar(fc_predict(model, T_SEL).argmax(1), T_SEL[2].cpu().numpy(), 7)[1]
        if sel_uar > best:
            best, bad = sel_uar, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best


yB = ev.yB.values
cert = (ev.unc_B == '1').values
results, PROBS = [], {a: [] for a in ARMS}
for arm in ARMS:
    for seed in SEEDS:
        model, sel_uar = train_forecaster(arm, seed)
        p = fc_predict(model, T_EV)
        PROBS[arm].append(p)
        w, u = war_uar(p.argmax(1), yB, 7)
        wc, uc = war_uar(p[cert].argmax(1), yB[cert], 7)
        results.append({'arm': arm, 'seed': seed, 'sel_UAR': sel_uar, 'UAR': u, 'WAR': w, 'UAR_certain': uc, 'WAR_certain': wc})
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in results[-1].items()})

res = pd.DataFrame(results)
res.to_csv(f"{OUT_DIR}/g3_results_per_seed.csv", index=False)
np.savez(f"{OUT_DIR}/g3_{EVAL_SPLIT}_probs.npz", sample_id=ev.sample_id.values,
         **{a.replace('+', '_'): np.stack(PROBS[a]) for a in ARMS})
print("\n== mean ± std over seeds ==")
print(res.drop(columns='seed').groupby('arm', sort=False).agg(['mean', 'std']).round(2).to_string())
'''),
    ("markdown", r'''
## Paired comparison against B1 and seed-ensemble scores
'''),
    ("code", r'''
def paired_diff_ci(pa, pb, y, src, n_boot=2000, seed=0):
    """Bootstrap over episodes of (arm A - arm B) for UAR and WAR, same resampled episodes for both arms."""
    rng = np.random.default_rng(seed)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7)
        wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


src = ev.source_folder.values
ens = {a: np.mean(PROBS[a], 0).argmax(1) for a in ARMS}
print(f"== seed-ensemble, {EVAL_SPLIT} ==")
for a in ARMS:
    report(a, ens[a], yB, src)
print("\n== paired differences vs B1 (seed-ensemble; per-seed wins in brackets) ==")
base = res[res.arm == 'B1'].set_index('seed')
for a in ARMS:
    if a == 'B1':
        continue
    lo, hi = paired_diff_ci(ens[a], ens['B1'], yB, src)
    wa, ua = war_uar(ens[a], yB, 7); wb, ub = war_uar(ens['B1'], yB, 7)
    per = res[res.arm == a].set_index('seed')
    wins_u = int(((per.UAR - base.UAR) > 0).sum()); wins_w = int(((per.WAR - base.WAR) > 0).sum())
    print(f"{a:<16} ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}] ({wins_u}/{len(SEEDS)} seeds)   "
          f"ΔWAR {wa - wb:+5.2f} [{lo[1]:+5.2f},{hi[1]:+5.2f}] ({wins_w}/{len(SEEDS)} seeds)")
print("\n-- references --")
for k, (u, w) in REPORT_REF.items():
    print(f"   (report reference, selected on val) {k:<28} UAR {u:5.2f}   WAR {w:5.2f}")
'''),
]


# ---------------------------------------------------------------- G3b: robustness of traj_only
G3B = [
    ("markdown", r'''
# G3b — Robustness checks for the affective-trajectory forecaster

In G3, `traj_only` (a forecaster on the 12-d soft recognizer outputs of clips I–III) beat end-to-end `B1`.
Before the test split is opened, this notebook checks whether that survives four objections:

1. **Feature-quality mismatch.** In G3, training rows got trajectories from *one* fold model while validation
   rows got the *average of 5 fold models*. Here every validation prediction is made once per fold model's
   trajectory and the **predictions** are averaged, so train and eval features come from the same kind of model.
   The old feature-averaged score is still reported as a diagnostic.
2. **Recognizer instability.** Stage 1 trains 3 recognizer seeds per fold. The main trajectory averages them;
   each single-seed trajectory is also run on its own.
3. **Selection protocol.** Every comparison is run under two protocols:
   `inner_dev` (train on 32 episodes, early-stop on 5 held-out training episodes) and
   `val` (train on all 37 episodes, early-stop on validation — the report's protocol, optimistic for everyone).
4. **Which clips matter.** Clip ablation for the trajectory model: III, II+III, I+II+III.

A logistic regression on the same trajectory (C chosen by episode-grouped CV on train only) is included as the
simplest possible forecaster. Test stays locked.
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"

SEEDS = [42, 123, 456, 789, 1024]      # forecaster seeds
REC_SEEDS = [42, 123, 456]             # recognizer seeds per fold (stage 1)
N_FOLDS = 5
N_INNER_DEV_SOURCES = 5
REC_EPOCHS, FC_EPOCHS, PATIENCE = 60, 50, 8
REC_BATCH, FC_BATCH = 64, 32
LR, WEIGHT_DECAY = 1e-4, 1e-5
POL_WEIGHT = 0.3
CERT_WEIGHTS = {'1': 1.0, '2': 0.75, '3': 0.5}   # only used for bookkeeping here (no weighted arms)

# (name, arm, clips, trajectory source, selection protocol); clips must be a suffix of (1, 2, 3)
EXPERIMENTS = [
    ("B1",                "B1",   (1, 2, 3), None,  "inner_dev"),
    ("traj_I-III",        "traj", (1, 2, 3), "avg", "inner_dev"),
    ("traj_II-III",       "traj", (2, 3),    "avg", "inner_dev"),
    ("traj_III",          "traj", (3,),      "avg", "inner_dev"),
] + [
    (f"traj_I-III_rec{r}", "traj", (1, 2, 3), r,     "inner_dev") for r in REC_SEEDS
] + [
    ("B1@val",            "B1",   (1, 2, 3), None,  "val"),
    ("traj_I-III@val",    "traj", (1, 2, 3), "avg", "val"),
    ("traj_II-III@val",   "traj", (2, 3),    "avg", "val"),
    ("traj_III@val",      "traj", (3,),      "avg", "val"),
]
EVAL_SPLIT = "val"
UNLOCK_TEST = False
'''),
    G3[2], G3[3], G3[4],
    ("markdown", r'''
## Stage 1 — cross-fitted recognizers (5 folds × 3 seeds)
'''),
    G3[6],
    ("code", r'''
ev_ctx = sorted(set(ev[['clip1', 'clip2', 'clip3']].values.ravel()))
OOF = {r: {} for r in REC_SEEDS}                   # clip -> (pe, pp) for training episodes, out-of-fold
EVP = {r: [None] * N_FOLDS for r in REC_SEEDS}     # per fold model: (pe, pp) aligned with ev_ctx
rec_log = []
for k in range(N_FOLDS):
    held = [s for s in train_sources if FOLD_OF[s] == k]
    rest = [s for s in train_sources if FOLD_OF[s] != k]
    rdev = sorted(random.Random(100 + k).sample(rest, 4))
    fit = [s for s in rest if s not in rdev]
    assert not set(lab.index[lab.ep.isin(held)]) & set(lab.index[lab.ep.isin(fit)])
    held_clips = sorted(set(train_all[train_all.source_folder.isin(held)][['clip1', 'clip2', 'clip3']].values.ravel()))
    for r in REC_SEEDS:
        model, dev_uar = train_recognizer(fit, rdev, r + 1000 * k)
        pe, pp = rec_predict(model, held_clips)
        OOF[r].update({c: (pe[i], pp[i]) for i, c in enumerate(held_clips)})
        EVP[r][k] = rec_predict(model, ev_ctx)
        rec_log.append({'fold': k, 'rec_seed': r, 'inner_dev_UAR': dev_uar})
        print(f"fold {k} seed {r}: recognizer inner-dev UAR {dev_uar:.2f}")
        del model
        torch.cuda.empty_cache()


def build_traj(mode):
    """mode 'avg' averages the recognizer seeds' posteriors; an int uses that seed alone.
    Returns (train map, list of per-fold eval maps, feature-averaged eval map)."""
    rs = REC_SEEDS if mode == 'avg' else [mode]
    clips = sorted(OOF[rs[0]])
    pe = np.mean([np.stack([OOF[r][c][0] for c in clips]) for r in rs], 0)
    pp = np.mean([np.stack([OOF[r][c][1] for c in clips]) for r in rs], 0)
    train_map = dict(zip(clips, rec_vector(pe, pp)))
    versions = []
    for k in range(N_FOLDS):
        pe = np.mean([EVP[r][k][0] for r in rs], 0)
        pp = np.mean([EVP[r][k][1] for r in rs], 0)
        versions.append(dict(zip(ev_ctx, rec_vector(pe, pp))))
    pe = np.mean([EVP[r][k][0] for r in rs for k in range(N_FOLDS)], 0)
    pp = np.mean([EVP[r][k][1] for r in rs for k in range(N_FOLDS)], 0)
    return train_map, versions, dict(zip(ev_ctx, rec_vector(pe, pp)))


TRAJ = {m: build_traj(m) for m in ['avg'] + REC_SEEDS}
for m, (tmap, versions, favg) in TRAJ.items():
    tr_u = war_uar(np.stack([tmap[c][:7] for c in train_all.clip3]).argmax(1), train_all.yA, 7)[1]
    ev_u = np.mean([war_uar(np.stack([v[c][:7] for c in ev.clip3]).argmax(1), ev.yA, 7)[1] for v in versions])
    fa_u = war_uar(np.stack([favg[c][:7] for c in ev.clip3]).argmax(1), ev.yA, 7)[1]
    print(f"trajectory '{m}': clip-III recognition UAR  train-OOF {tr_u:.2f} | {EVAL_SPLIT} per-fold mean {ev_u:.2f} | "
          f"{EVAL_SPLIT} feature-avg {fa_u:.2f}")
pd.DataFrame(rec_log).to_csv(f"{OUT_DIR}/g3b_recognizers.csv", index=False)
np.savez(f"{OUT_DIR}/g3b_trajectories.npz", ev_ctx=np.array(ev_ctx),
         **{f"train_{m}_clips": np.array(list(TRAJ[m][0])) for m in TRAJ},
         **{f"train_{m}_vecs": np.stack(list(TRAJ[m][0].values())) for m in TRAJ},
         **{f"ev_{m}_fold{k}": np.stack([TRAJ[m][1][k][c] for c in ev_ctx]) for m in TRAJ for k in range(N_FOLDS)})
'''),
    ("markdown", r'''
## Stage 2 — forecasters under both selection protocols
'''),
    ("code", r'''
fc_train = train_all[~train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
fc_dev = train_all[train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
COL = {1: 'clip1', 2: 'clip2', 3: 'clip3'}
_T_CACHE = {}


def make_T(rows_name, d, clips, traj_map, map_key):
    key = (rows_name, clips, map_key)
    if key not in _T_CACHE:
        vals = d[[COL[k] for k in clips]].values
        idx = torch.tensor([[CIDX[c] for c in r] for r in vals], device=DEVICE)
        if traj_map is None:
            rec = torch.zeros(len(d), len(clips), N_REC, device=DEVICE)
        else:
            rec = torch.tensor(np.stack([np.stack([traj_map[c] for c in r]) for r in vals]), device=DEVICE)
        _T_CACHE[key] = (idx, rec, torch.tensor(d.yB.values, device=DEVICE))
    return _T_CACHE[key]


def fc_predict(model, T, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(T[0]), bs):
            out.append(F.softmax(model(T[0][i:i + bs], T[1][i:i + bs]), -1).cpu())
    return torch.cat(out).numpy()


def predict_avg(model, T_list):
    """One prediction per fold-model trajectory, then average the probabilities."""
    return np.mean([fc_predict(model, T) for T in T_list], 0)


def sets_for(arm, clips, mode, protocol):
    assert clips == (1, 2, 3)[3 - len(clips):], 'clips must be a suffix of (1, 2, 3)'
    tmap, versions, favg = TRAJ[mode] if arm == 'traj' else (None, [None], None)
    mk = str(mode)
    tr_rows, tr_name = (train_all, 'train_all') if protocol == 'val' else (fc_train, 'fc_train')
    T_tr = make_T(tr_name, tr_rows, clips, tmap, mk)
    T_ev = [make_T('ev', ev, clips, v, f"{mk}_fold{k}") for k, v in enumerate(versions)]
    T_sel = T_ev if protocol == 'val' else [make_T('fc_dev', fc_dev, clips, tmap, mk)]
    T_favg = make_T('ev', ev, clips, favg, f"{mk}_favg") if arm == 'traj' else None
    return T_tr, T_sel, T_ev, T_favg


def train_forecaster(arm, clips, mode, protocol, seed):
    seed_all(seed)
    T_tr, T_sel, T_ev, T_favg = sets_for(arm, clips, mode, protocol)
    model = Forecaster(use_raw=(arm == 'B1'), use_traj=(arm == 'traj')).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    idx, rec, y = T_tr
    y_sel = T_sel[0][2].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(FC_EPOCHS):
        model.train()
        perm = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(perm), FC_BATCH):
            j = perm[i:i + FC_BATCH]
            loss = F.cross_entropy(model(idx[j], rec[j]), y[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sel_uar = war_uar(predict_avg(model, T_sel).argmax(1), y_sel, 7)[1]
        if sel_uar > best:
            best, bad = sel_uar, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    p = predict_avg(model, T_ev)
    p_favg = fc_predict(model, T_favg) if T_favg is not None else None
    return p, p_favg, best


yB = ev.yB.values
src = ev.source_folder.values
cert = (ev.unc_B == '1').values
results, PROBS = [], {}
for name, arm, clips, mode, protocol in EXPERIMENTS:
    PROBS[name] = []
    for seed in SEEDS:
        p, p_favg, sel = train_forecaster(arm, clips, mode, protocol, seed)
        PROBS[name].append(p)
        w, u = war_uar(p.argmax(1), yB, 7)
        wc, uc = war_uar(p[cert].argmax(1), yB[cert], 7)
        r = {'exp': name, 'protocol': protocol, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w,
             'UAR_certain': uc, 'WAR_certain': wc}
        if p_favg is not None:
            r['WAR_featavg'], r['UAR_featavg'] = war_uar(p_favg.argmax(1), yB, 7)
        results.append(r)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})
    torch.cuda.empty_cache()

res = pd.DataFrame(results)
res.to_csv(f"{OUT_DIR}/g3b_results_per_seed.csv", index=False)
np.savez(f"{OUT_DIR}/g3b_{EVAL_SPLIT}_probs.npz", sample_id=ev.sample_id.values,
         **{n.replace('@', '_at_').replace('-', '_'): np.stack(v) for n, v in PROBS.items()})
print("\n== mean ± std over forecaster seeds ==")
print(res.drop(columns=['seed', 'protocol']).groupby('exp', sort=False).agg(['mean', 'std']).round(2).to_string())
'''),
    ("markdown", r'''
## Logistic regression on the trajectory (simplest forecaster)
'''),
    ("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

tmap, versions, _ = TRAJ['avg']
LR_PROBS = {}
for clips in [(3,), (2, 3), (1, 2, 3)]:
    Xt = np.hstack([np.stack([tmap[c] for c in train_all[COL[k]]]) for k in clips])
    yt = train_all.yB.values
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1, 3]:
        s = []
        for a, b in GroupKFold(5).split(Xt, yt, train_all.source_folder):
            pred = LogisticRegression(max_iter=3000, C=C).fit(Xt[a], yt[a]).predict(Xt[b])
            s.append(war_uar(pred, yt[b], 7)[1])
        if best is None or np.mean(s) > best[0]:
            best = (np.mean(s), C)
    clf = LogisticRegression(max_iter=3000, C=best[1]).fit(Xt, yt)
    p = np.mean([clf.predict_proba(np.hstack([np.stack([v[c] for c in ev[COL[k]]]) for k in clips])) for v in versions], 0)
    name = "LR_traj_" + {(3,): "III", (2, 3): "II-III", (1, 2, 3): "I-III"}[clips]
    LR_PROBS[name] = p
    report(f"{name} (C={best[1]})", p.argmax(1), yB, src)
'''),
    ("markdown", r'''
## Paired comparisons, per-episode wins and the gate
'''),
    ("code", r'''
def paired_diff_ci(pa, pb, y, src, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7)
        wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


ENS = {n: np.mean(v, 0).argmax(1) for n, v in PROBS.items()}
ENS.update({n: p.argmax(1) for n, p in LR_PROBS.items()})
print(f"== seed-ensemble, {EVAL_SPLIT} (95% episode-bootstrap CI) ==")
for n in ENS:
    report(n, ENS[n], yB, src)

PAIRS = [("traj_I-III", "B1"), ("LR_traj_I-III", "B1"), ("traj_I-III@val", "B1@val"),
         ("traj_II-III", "traj_I-III"), ("traj_III", "traj_I-III"),
         ("traj_II-III@val", "traj_I-III@val"), ("traj_III@val", "traj_I-III@val")] + \
        [(f"traj_I-III_rec{r}", "traj_I-III") for r in REC_SEEDS]
print("\n== paired differences (seed-ensemble; per-seed wins where both are seeded) ==")
verdict = {}
for a, b in PAIRS:
    lo, hi = paired_diff_ci(ENS[a], ENS[b], yB, src)
    wa, ua = war_uar(ENS[a], yB, 7); wb, ub = war_uar(ENS[b], yB, 7)
    wins = ""
    if a in PROBS and b in PROBS:
        pa = res[res.exp == a].set_index('seed'); pb = res[res.exp == b].set_index('seed')
        wins = f"({int(((pa.UAR - pb.UAR) > 0).sum())}/{len(SEEDS)} seeds UAR)"
    ep_wins = sum(war_uar(ENS[a][src == s], yB[src == s], 7)[0] > war_uar(ENS[b][src == s], yB[src == s], 7)[0]
                  for s in np.unique(src))
    verdict[(a, b)] = (ua - ub, lo[0], hi[0])
    print(f"{a:<18} - {b:<15} ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}]  ΔWAR {wa - wb:+5.2f} "
          f"[{lo[1]:+5.2f},{hi[1]:+5.2f}]  {wins}  episodes won (WAR) {ep_wins}/{len(np.unique(src))}")

print("\n== feature-quality mismatch diagnostic (trajectory experiments) ==")
diag = res.dropna(subset=['UAR_featavg']).groupby('exp', sort=False)[['UAR', 'UAR_featavg', 'WAR', 'WAR_featavg']].mean().round(2)
print(diag.to_string())

d_main = verdict[("traj_I-III", "B1")]
d_val = verdict[("traj_I-III@val", "B1@val")]
rec_spread = res[res.exp.str.startswith("traj_I-III_rec")].groupby('exp').UAR.mean()
print("\n== GATE ==")
print(f"1. traj_I-III vs B1 (inner_dev): ΔUAR {d_main[0]:+.2f}, CI lower bound {d_main[1]:+.2f}  -> "
      f"{'PASS' if d_main[1] > 0 else ('WEAK (positive, CI includes 0)' if d_main[0] > 0 else 'FAIL')}")
print(f"2. traj_I-III@val vs B1@val (report protocol): ΔUAR {d_val[0]:+.2f}  -> {'PASS' if d_val[0] >= 0 else 'FAIL'}")
print(f"3. recognizer-seed spread of traj_I-III UAR: {rec_spread.max() - rec_spread.min():.2f} points "
      f"-> {'PASS' if rec_spread.max() - rec_spread.min() <= 1.5 else 'UNSTABLE'}")
print("4. clip ablation: see traj_II-III / traj_III rows above (negative Δ = earlier clips help)")
'''),
]


# ---------------------------------------------------------------- G4: preregistered test evaluation
G4 = [
    ("markdown", r'''
# G4 — Preregistered test evaluation (run once)

**Frozen method.** `traj_I-III`: a forecaster on the soft affective trajectory of clips I–III (per clip: emotion
posterior, polarity posterior, max-prob, entropy) produced by a recognizer cross-fitted over training episodes
(5 folds × 3 seeds, seed posteriors averaged; evaluation rows are predicted once per fold model and the
predictions averaged). Early stopping on the validation split.

**Primary contrast (decided before opening test).** `traj_I-III@val` − `B1@val` on **test**, ΔUAR of the
5-seed ensemble, 95% paired bootstrap over test episodes.
*Confirmed* if ΔUAR > 0 and the CI lower bound > 0; *directional* if ΔUAR > 0 but the CI includes 0;
otherwise *not confirmed*.

**Secondary (descriptive, no selection):** ΔWAR, macro-F1, rows with a *certain* B label, per-episode wins,
inner-dev-selected pair (`traj_I-III` vs `B1`), whether A adds information (`traj_I-III@val` vs `traj_I-II@val`,
`B1@val` vs `B0@val`), clip ablation (III, II–III), logistic regression on the trajectory, deployable
recognize-then-transition, and label-only references (majority, Copy-A oracle, Markov oracle).

Nothing in this notebook is tuned on test. `UNLOCK_TEST` must be switched on by hand.
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"

SEEDS = [42, 123, 456, 789, 1024]
REC_SEEDS = [42, 123, 456]
N_FOLDS = 5
N_INNER_DEV_SOURCES = 5
REC_EPOCHS, FC_EPOCHS, PATIENCE = 60, 50, 8
REC_BATCH, FC_BATCH = 64, 32
LR, WEIGHT_DECAY = 1e-4, 1e-5
POL_WEIGHT = 0.3
CERT_WEIGHTS = {'1': 1.0, '2': 0.75, '3': 0.5}

EXPERIMENTS = [   # (name, arm, clips, trajectory, protocol)
    ("B1@val",          "B1",   (1, 2, 3), None,  "val"),        # primary baseline
    ("traj_I-III@val",  "traj", (1, 2, 3), "avg", "val"),        # primary method
    ("B0@val",          "B1",   (1, 2),    None,  "val"),
    ("traj_I-II@val",   "traj", (1, 2),    "avg", "val"),
    ("traj_II-III@val", "traj", (2, 3),    "avg", "val"),
    ("traj_III@val",    "traj", (3,),      "avg", "val"),
    ("B1",              "B1",   (1, 2, 3), None,  "inner_dev"),
    ("traj_I-III",      "traj", (1, 2, 3), "avg", "inner_dev"),
]
PRIMARY = ("traj_I-III@val", "B1@val")
SECONDARY = [("traj_I-III", "B1"), ("traj_I-III@val", "traj_I-II@val"), ("B1@val", "B0@val"),
             ("traj_I-III@val", "traj_III@val"), ("traj_I-III@val", "traj_II-III@val"),
             ("traj_I-III@val", "LR_traj_I-III"), ("traj_I-III@val", "RtT_soft")]
SEL_SPLIT = "val"
EVAL_SPLIT = "test"
UNLOCK_TEST = False        # <- set to True by hand for the single preregistered run
'''),
    G3[2],
    ("code", r'''
sv = sp[sp.split == SEL_SPLIT].reset_index(drop=True)
sv['unc_B'] = sv['clip4'].map(lambda c: str(ann.at[c, 8]))
assert set(sv.source_folder).isdisjoint(ev.source_folder) and set(train_all.source_folder).isdisjoint(ev.source_folder)
print(f"selection split '{SEL_SPLIT}': {len(sv)} MCIS / {sv.source_folder.nunique()} episodes | "
      f"evaluation split '{EVAL_SPLIT}': {len(ev)} MCIS / {ev.source_folder.nunique()} episodes")
print("PRIMARY:", PRIMARY)
T, TP = transition_tables(train_all)   # P(B | E_A) from training labels, for the reference rows
'''),
    G3[3], G3[4], G3[6],
    ("markdown", r'''
## Stage 1 — cross-fitted recognizers (trajectories for train OOF, selection and evaluation rows)
'''),
    ("code", r'''
ctx = sorted(set(sv[['clip1', 'clip2', 'clip3']].values.ravel()) | set(ev[['clip1', 'clip2', 'clip3']].values.ravel()))
OOF = {r: {} for r in REC_SEEDS}
EVP = {r: [None] * N_FOLDS for r in REC_SEEDS}
for k in range(N_FOLDS):
    held = [s for s in train_sources if FOLD_OF[s] == k]
    rest = [s for s in train_sources if FOLD_OF[s] != k]
    rdev = sorted(random.Random(100 + k).sample(rest, 4))
    fit = [s for s in rest if s not in rdev]
    held_clips = sorted(set(train_all[train_all.source_folder.isin(held)][['clip1', 'clip2', 'clip3']].values.ravel()))
    for r in REC_SEEDS:
        model, dev_uar = train_recognizer(fit, rdev, r + 1000 * k)
        pe, pp = rec_predict(model, held_clips)
        OOF[r].update({c: (pe[i], pp[i]) for i, c in enumerate(held_clips)})
        EVP[r][k] = rec_predict(model, ctx)
        print(f"fold {k} seed {r}: recognizer inner-dev UAR {dev_uar:.2f}")
        del model
        torch.cuda.empty_cache()

clips_tr = sorted(OOF[REC_SEEDS[0]])
pe = np.mean([np.stack([OOF[r][c][0] for c in clips_tr]) for r in REC_SEEDS], 0)
pp = np.mean([np.stack([OOF[r][c][1] for c in clips_tr]) for r in REC_SEEDS], 0)
TRAIN_MAP = dict(zip(clips_tr, rec_vector(pe, pp)))
VERSIONS = [dict(zip(ctx, rec_vector(np.mean([EVP[r][k][0] for r in REC_SEEDS], 0),
                                     np.mean([EVP[r][k][1] for r in REC_SEEDS], 0)))) for k in range(N_FOLDS)]
for name, d in [('train OOF', train_all), (SEL_SPLIT, sv), (EVAL_SPLIT, ev)]:
    if name == 'train OOF':
        u = war_uar(np.stack([TRAIN_MAP[c][:7] for c in d.clip3]).argmax(1), d.yA, 7)[1]
    else:
        u = war_uar(np.mean([np.stack([v[c][:7] for c in d.clip3]) for v in VERSIONS], 0).argmax(1), d.yA, 7)[1]
    print(f"clip-III recognition UAR on {name}: {u:.2f}")
'''),
    ("markdown", r'''
## Stage 2 — forecasters (selection on val, evaluation on test)
'''),
    ("code", r'''
fc_train = train_all[~train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
fc_dev = train_all[train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
COL = {1: 'clip1', 2: 'clip2', 3: 'clip3'}
_T = {}


def make_T(rows_name, d, clips, tmap, key):
    # the cache key must say whether a trajectory is attached: raw arms store an all-zero trajectory
    k = (rows_name, clips, key, tmap is not None)
    if k not in _T:
        vals = d[[COL[c] for c in clips]].values
        idx = torch.tensor([[CIDX[c] for c in r] for r in vals], device=DEVICE)
        rec = (torch.zeros(len(d), len(clips), N_REC, device=DEVICE) if tmap is None else
               torch.tensor(np.stack([np.stack([tmap[c] for c in r]) for r in vals]), device=DEVICE))
        _T[k] = (idx, rec, torch.tensor(d.yB.values, device=DEVICE))
    out = _T[k]
    assert (tmap is None) == bool((out[1] == 0).all()), f"trajectory tensor mismatch for {k}"
    return out


def fc_predict(model, T, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(T[0]), bs):
            out.append(F.softmax(model(T[0][i:i + bs], T[1][i:i + bs]), -1).cpu())
    return torch.cat(out).numpy()


def predict_avg(model, T_list):
    return np.mean([fc_predict(model, T) for T in T_list], 0)


def train_forecaster(arm, clips, protocol, seed):
    seed_all(seed)
    traj = arm == 'traj'
    tr_rows, tr_name = (train_all, 'train_all') if protocol == 'val' else (fc_train, 'fc_train')
    T_tr = make_T(tr_name, tr_rows, clips, TRAIN_MAP if traj else None, 'train')
    evl = lambda name, d: [make_T(name, d, clips, v if traj else None, f'fold{k}' if traj else 'raw')
                           for k, v in enumerate(VERSIONS if traj else [None])]
    T_sv, T_ev = evl('sv', sv), evl('ev', ev)
    T_sel = T_sv if protocol == 'val' else [make_T('fc_dev', fc_dev, clips, TRAIN_MAP if traj else None, 'train')]
    model = Forecaster(use_raw=not traj, use_traj=traj, positions=tuple(c - 1 for c in clips)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    idx, rec, y = T_tr
    y_sel = T_sel[0][2].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(FC_EPOCHS):
        model.train()
        perm = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(perm), FC_BATCH):
            j = perm[i:i + FC_BATCH]
            loss = F.cross_entropy(model(idx[j], rec[j]), y[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sel_uar = war_uar(predict_avg(model, T_sel).argmax(1), y_sel, 7)[1]
        if sel_uar > best:
            best, bad = sel_uar, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return predict_avg(model, T_ev), predict_avg(model, T_sv), best


yT, srcT, certT = ev.yB.values, ev.source_folder.values, (ev.unc_B == '1').values
yV = sv.yB.values
runs, PT, PV = [], {}, {}
for name, arm, clips, _, protocol in EXPERIMENTS:
    PT[name], PV[name] = [], []
    for seed in SEEDS:
        pt, pv, sel = train_forecaster(arm, clips, protocol, seed)
        PT[name].append(pt); PV[name].append(pv)
        wt, ut = war_uar(pt.argmax(1), yT, 7)
        wv, uv = war_uar(pv.argmax(1), yV, 7)
        runs.append({'exp': name, 'seed': seed, 'sel_UAR': sel, 'test_UAR': ut, 'test_WAR': wt, 'val_UAR': uv, 'val_WAR': wv})
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in runs[-1].items()})
    torch.cuda.empty_cache()
runs = pd.DataFrame(runs)
runs.to_csv(f"{OUT_DIR}/g4_runs_per_seed.csv", index=False)
print(runs.drop(columns='seed').groupby('exp', sort=False).agg(['mean', 'std']).round(2).to_string())
'''),
    ("markdown", r'''
## Label-only references, deployable recognize-then-transition, logistic regression
'''),
    ("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PRED = {n: np.mean(v, 0).argmax(1) for n, v in PT.items()}          # seed-ensemble predictions on test
PRED['Majority'] = np.full(len(ev), np.bincount(train_all.yB, minlength=7).argmax())
PRED['[oracle] Copy-A'] = ev.yA.values
PRED['[oracle] Markov P(B|E_A)'] = rtt_forecast(np.eye(7)[ev.yA.values], T)[0]
pA_test = np.mean([np.stack([v[c][:7] for c in ev.clip3]) for v in VERSIONS], 0)
PRED['RtT_hard'], PRED['RtT_soft'], _ = rtt_forecast(pA_test, T)

Xtr = np.hstack([np.stack([TRAIN_MAP[c] for c in train_all[COL[k]]]) for k in (1, 2, 3)])
best = None
for C in [0.01, 0.03, 0.1, 0.3, 1, 3]:
    s = [war_uar(LogisticRegression(max_iter=3000, C=C).fit(Xtr[a], train_all.yB.values[a]).predict(Xtr[b]),
                 train_all.yB.values[b], 7)[1]
         for a, b in GroupKFold(5).split(Xtr, train_all.yB, train_all.source_folder)]
    if best is None or np.mean(s) > best[0]:
        best = (np.mean(s), C)
clf = LogisticRegression(max_iter=3000, C=best[1]).fit(Xtr, train_all.yB.values)
PRED['LR_traj_I-III'] = np.mean([clf.predict_proba(np.hstack([np.stack([v[c] for c in ev[COL[k]]]) for k in (1, 2, 3)]))
                                 for v in VERSIONS], 0).argmax(1)
print(f"LR C={best[1]}")
'''),
    ("markdown", r'''
## Results
'''),
    ("code", r'''
from sklearn.metrics import f1_score


def paired(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(srcT == s)[0] for s in np.unique(srcT)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(PRED[a][idx], yT[idx], 7); wb, ub = war_uar(PRED[b][idx], yT[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


rows = []
print(f"== {EVAL_SPLIT}: seed-ensemble (trained models) / deterministic references, 95% episode-bootstrap CI ==")
for n, p in PRED.items():
    r = report(n, p, yT, srcT)
    r['macroF1'] = f1_score(yT, p, average='macro', labels=list(range(7)), zero_division=0) * 100
    r['WAR_certain'], r['UAR_certain'] = war_uar(p[certT], yT[certT], 7)
    if n in PT:
        per = runs[runs.exp == n]
        r['UAR_seed_mean'], r['UAR_seed_std'] = per.test_UAR.mean(), per.test_UAR.std()
        r['val_UAR_seed_mean'] = per.val_UAR.mean()
    rows.append(r)
summary = pd.DataFrame(rows).set_index('name').round(2)
summary.to_csv(f"{OUT_DIR}/g4_test_summary.csv")
print(summary[['UAR', 'WAR', 'macroF1', 'UAR_certain', 'WAR_certain', 'UAR_seed_mean', 'UAR_seed_std', 'val_UAR_seed_mean']].to_string())

print("\n== per-episode WAR (test) ==")
print(pd.DataFrame({n: [war_uar(PRED[n][srcT == s], yT[srcT == s], 7)[0] for s in np.unique(srcT)]
                    for n in [PRIMARY[0], PRIMARY[1], '[oracle] Markov P(B|E_A)']},
                   index=np.unique(srcT)).round(1).to_string())


def contrast(a, b):
    lo, hi = paired(a, b)
    wa, ua = war_uar(PRED[a], yT, 7); wb, ub = war_uar(PRED[b], yT, 7)
    seeds = ""
    if a in PT and b in PT:
        pa = runs[runs.exp == a].set_index('seed').test_UAR; pb = runs[runs.exp == b].set_index('seed').test_UAR
        seeds = f"seeds {int(((pa - pb) > 0).sum())}/{len(SEEDS)}"
    eps = sum(war_uar(PRED[a][srcT == s], yT[srcT == s], 7)[0] > war_uar(PRED[b][srcT == s], yT[srcT == s], 7)[0]
              for s in np.unique(srcT))
    print(f"{a:<16} - {b:<16} ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}]  ΔWAR {wa - wb:+5.2f} "
          f"[{lo[1]:+5.2f},{hi[1]:+5.2f}]  {seeds}  episodes {eps}/{len(np.unique(srcT))}")
    return ua - ub, lo[0]


print("\n== PRIMARY ==")
d, lo = contrast(*PRIMARY)
print("VERDICT:", "CONFIRMED" if d > 0 and lo > 0 else ("DIRECTIONAL (CI includes 0)" if d > 0 else "NOT CONFIRMED"))
print("\n== SECONDARY (descriptive) ==")
for a, b in SECONDARY:
    contrast(a, b)
np.savez(f"{OUT_DIR}/g4_{EVAL_SPLIT}_probs.npz", sample_id=ev.sample_id.values,
         **{n.replace('@', '_at_').replace('-', '_'): np.stack(v) for n, v in PT.items()})
'''),
]


# ---------------------------------------------------------------- G5: 53-episode cross-validation
G5 = [
    ("markdown", r'''
# G5 — Episode-level cross-validation over all 53 episodes (preregistered secondary analysis)

The locked test split has only 8 episodes, so G4's estimate is noisy. This notebook repeats the **primary
contrast** (`traj_I-III` vs `B1`, both early-stopped on a selection split) under 5-fold cross-validation over
**all 53 episodes**, so that every MCIS is predicted exactly once by models that never saw its episode.

Per outer fold: test = ~1/5 of the episodes (folds balanced by MCIS count), selection = 8 other episodes,
training = the rest. The trajectory recognizer is cross-fitted *inside* the outer training episodes only.

Reported: pooled out-of-fold UAR/WAR with 95% bootstrap over the 53 episodes, the paired ΔUAR/ΔWAR, the per-fold
Δ, seed wins, and the same numbers restricted to the 45 non-test episodes and to the 8 locked-test episodes.

**Run this only after G4.** The folds evaluate on the locked-test episodes too, so running G5 first would
break the blind status of G4. Both flags below must be set by hand.
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"

N_OUTER = 5                 # outer folds over all 53 episodes
N_SEL_SOURCES = 8           # selection (early-stopping) episodes per outer fold
N_FOLDS = 5                 # inner cross-fitting folds for the recognizer
CV_REC_SEEDS = [42]         # recognizer seeds per inner fold (G4 uses 3; 1 keeps G5 affordable)
SEEDS = [42, 123, 456]      # forecaster seeds per outer fold
ARMS = [("B1", "B1", (1, 2, 3)), ("traj_I-III", "traj", (1, 2, 3))]
REC_EPOCHS, FC_EPOCHS, PATIENCE = 60, 50, 8
REC_BATCH, FC_BATCH = 64, 32
LR, WEIGHT_DECAY = 1e-4, 1e-5
POL_WEIGHT = 0.3
G4_DONE = False             # <- set True only after the single G4 test run has finished
UNLOCK_TEST = False         # <- and this, since the folds evaluate on locked-test episodes
'''),
    ("code", COMMON + r'''
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm

if not (G4_DONE and UNLOCK_TEST):
    raise RuntimeError("G5 evaluates on locked-test episodes. Run G4 first, then set G4_DONE = UNLOCK_TEST = True.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOT_CSV = os.path.join(DATASET_DIR, "Hi-EF-20260829T071606Z-1-001", "Hi-EF", "annotation.csv")
ann, sp = load_tables(ANNOT_CSV, SPLIT_CSV)
sp['unc_B'] = sp['clip4'].map(lambda c: str(ann.at[c, 8]))
LOCKED_TEST_EPS = set(sp[sp.split == 'test'].source_folder)

lab = ann[ann[7].notna()].copy()
lab['ep'] = [c.split('/')[0] for c in lab.index]
lab['y_e'] = lab[7].map(E2I)
lab['y_p'] = lab[5].map(lambda p: P2I.get(p, -1))
lab = lab[lab.y_e.notna()]

# outer folds balanced by MCIS count (greedy, deterministic)
sizes = sp.groupby('source_folder').size().sort_values(ascending=False)
OUTER = [[] for _ in range(N_OUTER)]
load = [0] * N_OUTER
for ep_, n in sizes.items():
    f = int(np.argmin(load)); OUTER[f].append(ep_); load[f] += n
for f in range(N_OUTER):
    print(f"outer fold {f}: {len(OUTER[f])} episodes, {load[f]} MCIS, locked-test episodes inside: "
          f"{sorted(set(OUTER[f]) & LOCKED_TEST_EPS)}")

# the feature-loading cell below checks these frames
train_all, ev = sp, sp
'''),
    G3[3], G3[4], G3[6],
    ("code", r'''
COL = {1: 'clip1', 2: 'clip2', 3: 'clip3'}


def fold_trajectories(tr_eps, ctx):
    """Cross-fit the recognizer inside tr_eps. Returns (OOF map for tr_eps clips, per-inner-fold maps for ctx)."""
    global train_all
    srcs = sorted(tr_eps)
    shuffled = srcs[:]
    random.Random(0).shuffle(shuffled)
    fold_of = {s: i % N_FOLDS for i, s in enumerate(shuffled)}
    oof = {r: {} for r in CV_REC_SEEDS}
    evp = {r: [None] * N_FOLDS for r in CV_REC_SEEDS}
    for k in range(N_FOLDS):
        held = [s for s in srcs if fold_of[s] == k]
        rest = [s for s in srcs if fold_of[s] != k]
        rdev = sorted(random.Random(100 + k).sample(rest, 4))
        fit = [s for s in rest if s not in rdev]
        held_clips = sorted(set(train_all[train_all.source_folder.isin(held)][['clip1', 'clip2', 'clip3']].values.ravel()))
        for r in CV_REC_SEEDS:
            model, _ = train_recognizer(fit, rdev, r + 1000 * k)
            pe, pp = rec_predict(model, held_clips)
            oof[r].update({c: (pe[i], pp[i]) for i, c in enumerate(held_clips)})
            evp[r][k] = rec_predict(model, ctx)
            del model
            torch.cuda.empty_cache()
    clips = sorted(oof[CV_REC_SEEDS[0]])
    tmap = dict(zip(clips, rec_vector(np.mean([np.stack([oof[r][c][0] for c in clips]) for r in CV_REC_SEEDS], 0),
                                      np.mean([np.stack([oof[r][c][1] for c in clips]) for r in CV_REC_SEEDS], 0))))
    versions = [dict(zip(ctx, rec_vector(np.mean([evp[r][k][0] for r in CV_REC_SEEDS], 0),
                                         np.mean([evp[r][k][1] for r in CV_REC_SEEDS], 0)))) for k in range(N_FOLDS)]
    return tmap, versions


def tensors(d, clips, tmap):
    vals = d[[COL[c] for c in clips]].values
    idx = torch.tensor([[CIDX[c] for c in r] for r in vals], device=DEVICE)
    rec = (torch.zeros(len(d), len(clips), N_REC, device=DEVICE) if tmap is None else
           torch.tensor(np.stack([np.stack([tmap[c] for c in r]) for r in vals]), device=DEVICE))
    return idx, rec, torch.tensor(d.yB.values, device=DEVICE)


def fc_predict(model, T, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(T[0]), bs):
            out.append(F.softmax(model(T[0][i:i + bs], T[1][i:i + bs]), -1).cpu())
    return torch.cat(out).numpy()


def predict_avg(model, T_list):
    return np.mean([fc_predict(model, T) for T in T_list], 0)


def train_eval(arm, clips, seed, tr, sel, te, tmap, versions):
    seed_all(seed)
    traj = arm == 'traj'
    T_tr = tensors(tr, clips, tmap if traj else None)
    T_sel = [tensors(sel, clips, v if traj else None) for v in (versions if traj else [None])]
    T_te = [tensors(te, clips, v if traj else None) for v in (versions if traj else [None])]
    model = Forecaster(use_raw=not traj, use_traj=traj, positions=tuple(c - 1 for c in clips)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    idx, rec, y = T_tr
    y_sel = sel.yB.values
    best, best_state, bad = -1, None, 0
    for ep in range(FC_EPOCHS):
        model.train()
        perm = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(perm), FC_BATCH):
            j = perm[i:i + FC_BATCH]
            loss = F.cross_entropy(model(idx[j], rec[j]), y[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        u = war_uar(predict_avg(model, T_sel).argmax(1), y_sel, 7)[1]
        if u > best:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return predict_avg(model, T_te)
'''),
    ("markdown", r'''
## Outer loop
'''),
    ("code", r'''
OOF_P = {name: np.zeros((len(SEEDS), len(sp), 7), dtype=np.float32) for name, _, _ in ARMS}
fold_rows = []
row_of = {s: i for i, s in enumerate(sp.sample_id)}
for f in range(N_OUTER):
    test_eps = OUTER[f]
    rest = sorted(set(sp.source_folder) - set(test_eps))
    sel_eps = sorted(random.Random(10 + f).sample(rest, N_SEL_SOURCES))
    tr_eps = [s for s in rest if s not in sel_eps]
    train_all = sp[sp.source_folder.isin(tr_eps)].reset_index(drop=True)
    sel = sp[sp.source_folder.isin(sel_eps)].reset_index(drop=True)
    te = sp[sp.source_folder.isin(test_eps)].reset_index(drop=True)
    assert not (set(train_all.source_folder) & set(te.source_folder)) and not (set(sel.source_folder) & set(te.source_folder))
    ctx = sorted(set(sel[['clip1', 'clip2', 'clip3']].values.ravel()) | set(te[['clip1', 'clip2', 'clip3']].values.ravel()))
    tmap, versions = fold_trajectories(tr_eps, ctx)
    rows = [row_of[s] for s in te.sample_id]
    for name, arm, clips in ARMS:
        for si, seed in enumerate(SEEDS):
            p = train_eval(arm, clips, seed, train_all, sel, te, tmap, versions)
            OOF_P[name][si, rows] = p
            w, u = war_uar(p.argmax(1), te.yB.values, 7)
            fold_rows.append({'fold': f, 'arm': name, 'seed': seed, 'UAR': u, 'WAR': w, 'n': len(te)})
            print(fold_rows[-1])
    torch.cuda.empty_cache()
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(f"{OUT_DIR}/g5_fold_results.csv", index=False)
np.savez(f"{OUT_DIR}/g5_oof_probs.npz", sample_id=sp.sample_id.values, **{k.replace('-', '_'): v for k, v in OOF_P.items()})
'''),
    ("markdown", r'''
## Pooled results
'''),
    ("code", r'''
y_all, src_all = sp.yB.values, sp.source_folder.values
PRED = {n: OOF_P[n].mean(0).argmax(1) for n in OOF_P}
a, b = "traj_I-III", "B1"


def pooled(mask, label):
    y, src = y_all[mask], src_all[mask]
    print(f"\n== {label}: {mask.sum()} MCIS, {len(np.unique(src))} episodes ==")
    for n in PRED:
        report(n, PRED[n][mask], y, src)
    rng = np.random.default_rng(0)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    d = []
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(PRED[a][mask][idx], y[idx], 7); wb, ub = war_uar(PRED[b][mask][idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    lo, hi = np.percentile(np.array(d), [2.5, 97.5], axis=0)
    wa, ua = war_uar(PRED[a][mask], y, 7); wb, ub = war_uar(PRED[b][mask], y, 7)
    eps = sum(war_uar(PRED[a][mask][src == s], y[src == s], 7)[0] > war_uar(PRED[b][mask][src == s], y[src == s], 7)[0]
              for s in np.unique(src))
    print(f"{a} - {b}: ΔUAR {ua - ub:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]  ΔWAR {wa - wb:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]  "
          f"episodes won (WAR) {eps}/{len(np.unique(src))}")
    return ua - ub, lo[0]


d_all, lo_all = pooled(np.ones(len(sp), bool), "all 53 episodes (primary CV readout)")
pooled(~sp.source_folder.isin(LOCKED_TEST_EPS).values, "45 non-test episodes")
pooled(sp.source_folder.isin(LOCKED_TEST_EPS).values, "8 locked-test episodes (compare with G4)")

per_fold = fold_df.groupby(['fold', 'arm']).UAR.mean().unstack()
per_fold['Δ'] = per_fold[a] - per_fold[b]
print("\n== per outer fold (seed-mean UAR) ==")
print(per_fold.round(2).to_string())
seed_wins = (fold_df[fold_df.arm == a].set_index(['fold', 'seed']).UAR >
             fold_df[fold_df.arm == b].set_index(['fold', 'seed']).UAR).sum()
print(f"(fold, seed) pairs won by {a}: {seed_wins}/{len(SEEDS) * N_OUTER}")
print("\nCV VERDICT:", "CONFIRMED" if d_all > 0 and lo_all > 0 else ("DIRECTIONAL (CI includes 0)" if d_all > 0 else "NOT CONFIRMED"))
'''),
]


# ---------------------------------------------------------------- G6a: who is on screen?
G6A = [
    ("markdown", r'''
# G6a — Is the listener (B) observable before B speaks?

Gate for the *listener-aware* formulation. Hi-EF forecasts B's emotion in clip IV from clips I–III, but models only
the speaker. This notebook measures, on **train + val only** (test untouched), how often B can actually be seen:

* **listening:** B's face appears in clip III while A is talking (same frame or a cut-away reaction shot);
* **previous turn:** B spoke in clip II or clip I.

Faces are unreliable for "who spoke" (speakers often turn away, profile faces break identity embeddings) and scenes
may contain a third person, so the notebook uses **two identity channels**:

* **voice** (ECAPA speaker embeddings of each clip's audio) for turn identity: is the voice of clip II / I the same as
  clip IV's (B's) voice? It also measures how often clip III and IV have the *same* voice, i.e. MCIS that violate the
  dataset rule A ≠ B;
* **faces** for listening: B's face in clip III, counted as *usable* only when roughly frontal (|yaw| ≤ MAX_YAW).

Face method, per MCIS: sample frames from clips I–IV, detect faces and compute ArcFace embeddings (InsightFace),
cluster all faces of the MCIS into persons, take the dominant person of each clip as its speaker proxy.
A = dominant person of III, B_true = dominant person of IV (**analysis only**). It also scores an inference-time
rule that picks B **without** clip IV, and writes annotated frame montages for visual checking.

Gate: B observable (usable listening face or previous-turn voice) in ≥ 40–50% of MCIS, and the no-clip-IV rules
(face and voice) are right in ≥ 80% of the cases where they fire. Results are split into two-person and multi-person scenes.
'''),
    ("code", r'''
# insightface can pull the CPU build of onnxruntime, which overwrites onnxruntime-gpu (same package dir).
# Install it first, then remove every onnxruntime build, then install the GPU build last.
!pip install -q insightface speechbrain
!pip uninstall -y -q onnxruntime onnxruntime-gpu
!pip install -q "onnxruntime-gpu==1.22.0"   # CUDA 12 build (1.30+ targets CUDA 13, which Kaggle lacks)
!pip list 2>/dev/null | grep -i onnxruntime
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"

SPLITS = ["train", "val"]    # test stays untouched
N_MCIS = 600                 # random MCIS to analyse (None = all train+val; ~4x slower)
SAMPLE_FPS, MAX_FRAMES = 3, 24
DET_SIZE = (640, 640)
MIN_DET_SCORE, MIN_FACE_PX = 0.6, 24
SAME_PERSON_COS = 0.45       # cosine similarity above which two faces are the same person
DOMINANT_MIN_FRAC = 0.25     # a clip's dominant person must be in >= this fraction of its sampled frames
N_MONTAGES = 24
MAX_YAW = 45                 # degrees; listener faces beyond this are counted as not usable for expression
VOICE_SAME_COS = 0.35        # ECAPA cosine above which two clips are taken as the same speaker
AUDIO_SR = 16000
SEED = 0
'''),
    ("code", r'''
import os, glob, random, json
import numpy as np, pandas as pd, cv2
from tqdm.auto import tqdm
from sklearn.cluster import AgglomerativeClustering

roots = sorted(glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF")))
VIDEO_ROOTS = [os.path.join(r, "video") for r in roots if os.path.isdir(os.path.join(r, "video"))]
AUDIO_ROOTS = [os.path.join(r, "audio") for r in roots if os.path.isdir(os.path.join(r, "audio"))]
ANNOT_CSV = [os.path.join(r, "annotation.csv") for r in roots if os.path.exists(os.path.join(r, "annotation.csv"))][0]
print("video roots:", VIDEO_ROOTS, "| audio roots:", AUDIO_ROOTS)


def audio_path(clip):
    ep, num = clip.split('/')
    for root in AUDIO_ROOTS:
        for ext in ('.mp3', '.wav', '.flac', '.m4a'):
            p = os.path.join(root, ep, num + ext)
            if os.path.exists(p):
                return p
    return None


def video_path(clip):
    ep, num = clip.split('/')
    for root in VIDEO_ROOTS:
        for ext in ('.mp4', '.avi', '.mkv', '.mov'):
            p = os.path.join(root, ep, num + ext)
            if os.path.exists(p):
                return p
    return None


ann = pd.read_csv(ANNOT_CSV, header=None, dtype=str).set_index(0)
sp = pd.read_csv(SPLIT_CSV, dtype=str)
sp = sp[sp.split.isin(SPLITS)].reset_index(drop=True)
assert 'test' not in set(sp.split)
if N_MCIS is not None and N_MCIS < len(sp):
    sp = sp.sample(n=N_MCIS, random_state=SEED).reset_index(drop=True)
clips = sorted(set(sp[['clip1', 'clip2', 'clip3', 'clip4']].values.ravel()))
missing = [c for c in clips if video_path(c) is None]
print(f"MCIS {len(sp)} | unique clips {len(clips)} | clips without video {len(missing)}", missing[:5])
'''),
    ("code", r'''
import torch  # loads the CUDA/cuDNN libraries that onnxruntime-gpu can reuse
import onnxruntime as ort
if hasattr(ort, 'preload_dlls'):
    try:
        ort.preload_dlls()
    except Exception as e:
        print("preload_dlls:", e)
print("onnxruntime", ort.__version__, "declared providers:", ort.get_available_providers())

from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark_3d_68'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=DET_SIZE)
USED = {k: m.session.get_providers() for k, m in app.models.items()}
ON_GPU = all('CUDAExecutionProvider' in v for v in USED.values())
print("providers actually used:", USED)
if not ON_GPU:
    # CPU fallback, decided on the providers the sessions really use (a declared CUDA provider can still fail to load)
    print("WARNING: models run on CPU -> CPU mode: fewer MCIS/frames, smaller detector input")
    N_CPU_MCIS = 150
    if len(sp) > N_CPU_MCIS:
        sp = sp.sample(n=N_CPU_MCIS, random_state=SEED).reset_index(drop=True)
        clips = sorted(set(sp[['clip1', 'clip2', 'clip3', 'clip4']].values.ravel()))
    MAX_FRAMES, DET_SIZE = 12, (480, 480)
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    print(f"CPU mode: {len(sp)} MCIS, {len(clips)} clips, <= {MAX_FRAMES} frames/clip")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = max(1, min(MAX_FRAMES, int(round(n / fps * SAMPLE_FPS)))) if n else MAX_FRAMES
    frames = []
    for i in sorted(set(np.linspace(0, max(n - 1, 0), want).astype(int).tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)   # seek instead of decoding the whole clip
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


FACES = {}      # clip -> list of dicts(frame, bbox, score, emb)
NFRAMES = {}
KEEP_FRAMES = {}  # a few frames per clip for montages
montage_mcis = set(sp.sample(n=min(N_MONTAGES, len(sp)), random_state=SEED + 1).sample_id)
montage_clips = set(sp[sp.sample_id.isin(montage_mcis)][['clip1', 'clip2', 'clip3', 'clip4']].values.ravel())
import time as _time
_t0 = _time.time()
for ci, c in enumerate(clips):
    if ci % 100 == 0:
        print(f"faces: {ci}/{len(clips)} clips, {(_time.time() - _t0) / 60:.1f} min", flush=True)
    p = video_path(c)
    frames = read_frames(p) if p else []
    NFRAMES[c] = len(frames)
    out = []
    for fi, fr in enumerate(frames):
        for f in app.get(fr):
            x1, y1, x2, y2 = f.bbox
            if f.det_score >= MIN_DET_SCORE and min(x2 - x1, y2 - y1) >= MIN_FACE_PX:
                pose = getattr(f, 'pose', None)
                out.append({'frame': fi, 'bbox': f.bbox.astype(int).tolist(), 'score': float(f.det_score),
                            'emb': f.normed_embedding.astype(np.float32),
                            'yaw': float(pose[1]) if pose is not None else 0.0})
    FACES[c] = out
    if c in montage_clips and frames:
        pick = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
        KEEP_FRAMES[c] = [(int(i), frames[i]) for i in pick]
print("clips with >=1 face:", sum(bool(v) for v in FACES.values()), "/", len(FACES))
'''),
    ("code", r'''
def analyse(row):
    cl = [row['clip1'], row['clip2'], row['clip3'], row['clip4']]
    faces = [(k, f) for k, c in enumerate(cl) for f in FACES.get(c, [])]
    res = {'sample_id': row['sample_id'], 'split': row['split'], 'source_folder': row['source_folder'],
           'emo_A': row['clip3_emotion'], 'emo_B': row['clip4_emotion'],
           'unc_B': str(ann.at[row['clip4'], 8]) if row['clip4'] in ann.index else 'NA'}
    if len(faces) == 0:
        return res | {'persons': 0}, {}
    E = np.stack([f['emb'] for _, f in faces])
    lab_ = (np.zeros(1, int) if len(E) == 1 else
            AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                    distance_threshold=1 - SAME_PERSON_COS).fit_predict(E))
    # frames per (clip, person)
    pres = {}
    for (k, f), p in zip(faces, lab_):
        pres.setdefault((k, int(p)), set()).add(f['frame'])

    def dominant(k):
        n = NFRAMES.get(cl[k], 0)
        cand = [(len(fr), p) for (kk, p), fr in pres.items() if kk == k]
        if not cand or n == 0:
            return None
        cnt, p = max(cand)
        return p if cnt / n >= DOMINANT_MIN_FRAC else None

    dom = [dominant(k) for k in range(4)]
    A, Bt = dom[2], dom[3]
    in3 = {p for (k, p) in pres if k == 2}
    others3 = sorted(in3 - {A}, key=lambda p: -len(pres[(2, p)]))
    # inference-time rule (no clip IV): a non-A person of clip III, preferring one who spoke in II or I
    spoke = [p for p in (dom[1], dom[0]) if p is not None and p != A]
    if others3:
        pref = [p for p in others3 if p in spoke]
        Bp = pref[0] if pref else others3[0]
        src = 'III_listening'
    elif spoke:
        Bp, src = spoke[0], 'previous_turn'
    else:
        Bp, src = None, 'none'
    n3 = max(NFRAMES.get(cl[2], 0), 1)
    frontal_B3 = {f['frame'] for (k, f), p in zip(faces, lab_) if k == 2 and Bt is not None and Bt != A and p == Bt
                  and abs(f['yaw']) <= MAX_YAW}
    ids3 = [frozenset(p for (k, p), fr in pres.items() if k == 2 and fi in fr) for fi in range(n3)]
    res |= {
        'persons': len(set(int(p) for p in lab_)), 'persons_III': len(in3), 'A_found': A is not None,
        'Btrue_found': Bt is not None, 'A_eq_Btrue': A is not None and A == Bt,
        'B_listening_III': Bt is not None and Bt != A and Bt in in3,
        'B_frac_III': len(pres.get((2, Bt), ())) / n3 if Bt is not None and Bt != A else 0.0,
        'B_same_frame_as_A': Bt is not None and A is not None and Bt != A and
                             bool(pres.get((2, Bt), set()) & pres.get((2, A), set())),
        'B_listening_III_frontal': len(frontal_B3) > 0,
        'B_spoke_II': Bt is not None and Bt != A and dom[1] == Bt,
        'B_spoke_I': Bt is not None and Bt != A and dom[0] == Bt,
        'Bpred_source': src, 'Bpred_correct': Bp is not None and Bt is not None and Bt != A and Bp == Bt,
        'Bpred_made': Bp is not None,
        'cuts_III': sum(ids3[i] != ids3[i - 1] for i in range(1, len(ids3))),
    }
    res['B_observable'] = res['B_listening_III'] or res['B_spoke_II'] or res['B_spoke_I']
    return res, {'faces': faces, 'labels': lab_, 'A': A, 'Bt': Bt, 'Bp': Bp}


rows, DETAIL = [], {}
for r in tqdm(sp.to_dict('records'), desc='MCIS'):
    res, det = analyse(r)
    rows.append(res)
    if r['sample_id'] in montage_mcis:
        DETAIL[r['sample_id']] = (r, det)
df = pd.DataFrame(rows)
BOOL = ['A_found', 'Btrue_found', 'A_eq_Btrue', 'B_listening_III', 'B_listening_III_frontal', 'B_same_frame_as_A',
        'B_spoke_II', 'B_spoke_I',
        'Bpred_correct', 'Bpred_made', 'B_observable']
for c in BOOL:
    df[c] = df[c].fillna(False).astype(bool) if c in df else False
for c in ['persons_III', 'B_frac_III', 'cuts_III']:
    df[c] = df[c].fillna(0) if c in df else 0
df['Bpred_source'] = df['Bpred_source'].fillna('none') if 'Bpred_source' in df else 'none'
df.to_csv(f"{OUT_DIR}/g6a_listener_visibility.csv", index=False)
'''),
    ("code", r'''
ok = df[df.A_found & df.Btrue_found & ~df.A_eq_Btrue]
print(f"MCIS analysed: {len(df)}")
print(f"  faces found at all: {(df.persons > 0).mean() * 100:.1f}%   A (clip III dominant) found: {df.A_found.mean() * 100:.1f}%   "
      f"B_true (clip IV dominant) found: {df.Btrue_found.mean() * 100:.1f}%")
print(f"  sanity: dominant(III) == dominant(IV) in {df.A_eq_Btrue.mean() * 100:.1f}% (dataset rule says A != B; high = proxy fails)")
print(f"\nAmong {len(ok)} MCIS where A and B_true are identified and distinct:")
for col, name in [('B_listening_III', 'B visible in clip III (listening/reaction)'),
                  ('B_listening_III_frontal', '  ... with a usable (near-frontal) face'),
                  ('B_same_frame_as_A', '  ... in the same frame as A'),
                  ('B_spoke_II', 'B was dominant in clip II'), ('B_spoke_I', 'B was dominant in clip I'),
                  ('B_observable', 'B observable (III listening or I/II turn)')]:
    print(f"  {name:<46} {ok[col].mean() * 100:5.1f}%")
print(f"  mean fraction of clip-III frames showing B: {ok.B_frac_III.mean():.2f}  | mean identity changes in III: {ok.cuts_III.mean():.1f}")
print(f"\nB_observable over ALL analysed MCIS (unidentified counted as not observable): "
      f"{df.B_observable.mean() * 100:.1f}%")
made = ok[ok.Bpred_made]
print(f"\nNo-clip-IV rule: proposes someone in {ok.Bpred_made.mean() * 100:.1f}% of identified MCIS; "
      f"precision {made.Bpred_correct.mean() * 100:.1f}%")
print(made.groupby('Bpred_source').Bpred_correct.agg(['size', 'mean']).rename(columns={'mean': 'precision'}).round(3).to_string())
print("\nB visible while listening, by split / by B emotion:")
print(ok.groupby('split').B_listening_III.mean().round(3).to_string())
print(ok.groupby('emo_B').B_listening_III.agg(['size', 'mean']).round(3).to_string())
'''),
    ("markdown", r'''
## Voice channel: who spoke in clips I–IV (robust to profile faces)
'''),
    ("code", r'''
import librosa, torch
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{OUT_DIR}/ecapa",
                                     run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
VOICE = {}
for c in tqdm(clips, desc='voice'):
    p = audio_path(c)
    if p is None:
        continue
    try:
        wav, _ = librosa.load(p, sr=AUDIO_SR, mono=True)
    except Exception:
        continue
    if len(wav) < AUDIO_SR // 2:
        continue
    with torch.no_grad():
        e = spk.encode_batch(torch.tensor(wav, dtype=torch.float32).unsqueeze(0)).reshape(-1).cpu().numpy()
    VOICE[c] = e / (np.linalg.norm(e) + 1e-9)
print(f"voice embeddings: {len(VOICE)}/{len(clips)} clips", flush=True)


def vcos(a, b):
    return float(VOICE[a] @ VOICE[b]) if a in VOICE and b in VOICE else np.nan


cl_of = sp.set_index('sample_id')[['clip1', 'clip2', 'clip3', 'clip4']]
for x, y in [(3, 4), (2, 4), (1, 4), (2, 3), (1, 3)]:
    df[f'vcos_{x}{y}'] = [vcos(cl_of.at[s_, f'clip{x}'], cl_of.at[s_, f'clip{y}']) for s_ in df.sample_id]
df['multi_party'] = df.persons >= 3
df.to_csv(f"{OUT_DIR}/g6a_listener_visibility.csv", index=False)
print("cosine quantiles (10/25/50/75/90%):")
for col in ['vcos_34', 'vcos_24', 'vcos_14', 'vcos_23']:
    print(f"  {col}: {np.nanpercentile(df[col], [10, 25, 50, 75, 90]).round(2)}")
'''),
    ("code", r'''
T_ = VOICE_SAME_COS
has = df.vcos_34.notna()
same34 = has & (df.vcos_34 > T_)
print(f"MCIS with clip III and IV voices: {has.sum()}")
print(f"  clip III and IV sound like the SAME speaker (violates A != B): {same34[has].mean() * 100:.1f}%")
valid = df[has & ~same34].copy()
valid['B_voice_II'] = valid.vcos_24 > T_
valid['B_voice_I'] = valid.vcos_14 > T_
valid['II_not_A'] = valid.vcos_23 <= T_          # inference-time rule: clip II is someone other than A
valid['I_not_A'] = valid.vcos_13 <= T_
valid['B_observable_v2'] = valid.B_listening_III_frontal | valid.B_voice_II | valid.B_voice_I


def block(d, name):
    print(f"\n== {name}: {len(d)} MCIS (A != B by voice) ==")
    print(f"  B spoke in clip II (voice)                 {d.B_voice_II.mean() * 100:5.1f}%")
    print(f"  B spoke in clip I  (voice)                 {d.B_voice_I.mean() * 100:5.1f}%")
    print(f"  B visible in III with usable face          {d.B_listening_III_frontal.mean() * 100:5.1f}%")
    print(f"  B observable (usable face OR voice turn)   {d.B_observable_v2.mean() * 100:5.1f}%")
    for k, name_ in [('2', 'II'), ('1', 'I')]:
        third = (d[f'vcos_{k}3'] <= T_) & (d[f'vcos_{k}4'] <= T_)
        print(f"  clip {name_:<2} speaker is a THIRD person (neither A nor B)  {third.mean() * 100:5.1f}%")
    for rule, truth in [('II_not_A', 'B_voice_II'), ('I_not_A', 'B_voice_I')]:
        fired = d[d[rule]]
        print(f"  rule '{rule}' fires {d[rule].mean() * 100:5.1f}% | precision {fired[truth].mean() * 100 if len(fired) else float('nan'):5.1f}%"
              f" | recall {d[d[truth]][rule].mean() * 100 if d[truth].any() else float('nan'):5.1f}%")


block(valid, "all")
block(valid[~valid.multi_party], "two-person scenes (<= 2 face identities)")
block(valid[valid.multi_party], "multi-person scenes (>= 3 face identities)")
print(f"\nmulti-person share: {df.multi_party.mean() * 100:.1f}% of analysed MCIS")
print("\nGATE: B observable >= 40-50%, rule precision >= 80%, and a small A==B (same voice) share.")
valid.to_csv(f"{OUT_DIR}/g6a_voice_valid.csv", index=False)
'''),
    ("markdown", r'''
## Montages (send a few of these back for a visual check)
Rows = clips I–IV, columns = sampled frames. Box colours: **red** A (dominant in III), **green** B_true
(dominant in IV), **blue** the no-clip-IV prediction when it differs from B_true, **grey** other people.
'''),
    ("code", r'''
os.makedirs(f"{OUT_DIR}/g6a_montages", exist_ok=True)
for sid, (r, det) in DETAIL.items():
    if not det:
        continue
    cl = [r['clip1'], r['clip2'], r['clip3'], r['clip4']]
    person_at = {}
    for (k, f), p in zip(det['faces'], det['labels']):
        person_at.setdefault((cl[k], f['frame']), []).append((f['bbox'], int(p)))
    tiles_rows = []
    for k, c in enumerate(cl):
        tiles = []
        for fi, fr in KEEP_FRAMES.get(c, []):
            im = fr.copy()
            for (x1, y1, x2, y2), p in person_at.get((c, fi), []):
                col = ((0, 0, 255) if p == det['A'] else (0, 200, 0) if p == det['Bt'] else
                       (255, 120, 0) if p == det['Bp'] else (160, 160, 160))
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
                cv2.putText(im, str(p), (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            tiles.append(cv2.resize(im, (256, 144)))
        while len(tiles) < 6:
            tiles.append(np.zeros((144, 256, 3), np.uint8))
        row_img = np.hstack(tiles[:6])
        cv2.putText(row_img, ['I', 'II', 'III (A speaks)', 'IV (B speaks)'][k], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles_rows.append(row_img)
    cv2.imwrite(f"{OUT_DIR}/g6a_montages/{sid}.jpg", np.vstack(tiles_rows))
print(len(os.listdir(f"{OUT_DIR}/g6a_montages")), "montages written")
'''),
]


# ---------------------------------------------------------------- G6b: does the listener's face forecast B?
G6B = [
    ("markdown", r'''
# G6b — Does the listener's face in clip III forecast B's emotion in clip IV?

G6a showed B is on screen while A speaks (cut-away reaction shots) in about half of the MCIS. This notebook asks
whether that reaction carries information about B's *next* emotion, beyond what A's face carries.

* Faces are detected on clips **I–III only**; clip IV is used only for its label.
* Roles per MCIS: **A** = dominant person of clip III; **listener** = the most frequent *other* person in clip III
  (the deployable rule from G6a, ~81% correct); **listener_ctx** = that same listener identity found in clips I/II
  (faces are clustered jointly over I–III); **ctx_II / ctx_I** = dominant person of clips II / I. G6a's voice pass
  found the clip I/II speaker is a third person 37–41% of the time, so "dominant context person" is not B.
* Each face gets an expression vector from **HSEmotion** (EfficientNet-B0 trained on AffectNet: 8 emotion
  probabilities + valence + arousal). Role features = mean over that role's frames + presence + frame share.
* Boundary check: listener frames in the last 20% of clip III might already show B starting to speak, so a
  second listener feature uses only frames in the first 80% of the clip.

Logistic regressions (C by episode-grouped CV on train) are scored on val with episode-bootstrap CIs, overall and on
MCIS where a listener is visible. **Test is untouched.**

Gate: `A + listener` beats `A only` on the listener-visible subset by ≥ 2–3 UAR with a paired CI above 0, and the
early-frames-only listener keeps most of that gain.
'''),
    ("code", r'''
!pip install -q insightface
!pip uninstall -y -q onnxruntime onnxruntime-gpu
!pip install -q "onnxruntime-gpu==1.22.0"
'''),
    ("code", r'''
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"
CACHE = f"{OUT_DIR}/g6b_face_cache.pkl"   # per-clip detections + expressions, reused on re-runs

SPLITS = ["train", "val"]    # test stays untouched
N_MCIS = None                # None = all train+val MCIS
SAMPLE_FPS, MAX_FRAMES = 3, 24
DET_SIZE = (640, 640)
MIN_DET_SCORE, MIN_FACE_PX = 0.6, 32
SAME_PERSON_COS = 0.45
DOMINANT_MIN_FRAC = 0.25
MAX_YAW = 45
EARLY_FRAC = 0.8             # "early" listener frames: relative position < EARLY_FRAC within clip III
FER_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/"
           "enet_b0_8_va_mtl.onnx")
SEED = 0
'''),
    ("code", r'''
import os, glob, random, pickle, time, urllib.request
import numpy as np, pandas as pd, cv2
from sklearn.cluster import AgglomerativeClustering
import torch
import onnxruntime as ort
if hasattr(ort, 'preload_dlls'):
    try:
        ort.preload_dlls()
    except Exception as e:
        print("preload_dlls:", e)
PROV = ['CUDAExecutionProvider', 'CPUExecutionProvider']

roots = sorted(glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF")))
VIDEO_ROOTS = [os.path.join(r, "video") for r in roots if os.path.isdir(os.path.join(r, "video"))]
ANNOT_CSV = [os.path.join(r, "annotation.csv") for r in roots if os.path.exists(os.path.join(r, "annotation.csv"))][0]


def video_path(clip):
    ep, num = clip.split('/')
    for root in VIDEO_ROOTS:
        p = os.path.join(root, ep, num + '.mp4')
        if os.path.exists(p):
            return p
    return None


EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
ann = pd.read_csv(ANNOT_CSV, header=None, dtype=str).set_index(0)
sp = pd.read_csv(SPLIT_CSV, dtype=str)
sp = sp[sp.split.isin(SPLITS)].reset_index(drop=True)
assert 'test' not in set(sp.split)
if N_MCIS is not None and N_MCIS < len(sp):
    sp = sp.sample(n=N_MCIS, random_state=SEED).reset_index(drop=True)
sp['yA'] = sp.clip3_emotion.map(E2I)
sp['yB'] = sp.clip4_emotion.map(E2I)
clips = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))   # no clip IV frames are read
print(f"MCIS {len(sp)} | context/A clips {len(clips)} | missing video {sum(video_path(c) is None for c in clips)}")
'''),
    ("code", r'''
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark_3d_68'], providers=PROV)
app.prepare(ctx_id=0, det_size=DET_SIZE)
print("insightface providers:", {k: m.session.get_providers() for k, m in app.models.items()})

fer_path = f"{OUT_DIR}/enet_b0_8_va_mtl.onnx"
if not os.path.exists(fer_path):
    urllib.request.urlretrieve(FER_URL, fer_path)
fer = ort.InferenceSession(fer_path, providers=PROV)
FER_IN = fer.get_inputs()[0].name
print("FER providers:", fer.get_providers(), "| input", fer.get_inputs()[0].shape, "| output", fer.get_outputs()[0].shape)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
FER8 = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']   # HSEmotion 8-class order


def fer_batch(crops_bgr):
    if not crops_bgr:
        return np.zeros((0, 10), np.float32)
    x = np.stack([(cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                  for c in crops_bgr]).transpose(0, 3, 1, 2)
    out = fer.run(None, {FER_IN: x.astype(np.float32)})[0]
    logits = out[:, :8]
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    va = out[:, 8:10] if out.shape[1] >= 10 else np.zeros((len(out), 2), np.float32)
    return np.concatenate([p, va], 1).astype(np.float32)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = max(1, min(MAX_FRAMES, int(round(n / fps * SAMPLE_FPS)))) if n else MAX_FRAMES
    frames = []
    for i in sorted(set(np.linspace(0, max(n - 1, 0), want).astype(int).tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


FACES = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
todo = [c for c in clips if c not in FACES]
print(f"cached clips {len(FACES)} | to process {len(todo)}")
t0 = time.time()
for ci, c in enumerate(todo):
    if ci % 200 == 0:
        print(f"clips {ci}/{len(todo)}, {(time.time() - t0) / 60:.1f} min", flush=True)
        if ci:
            pickle.dump(FACES, open(CACHE, 'wb'))
    p = video_path(c)
    frames = read_frames(p) if p else []
    dets, crops = [], []
    for fi, fr in enumerate(frames):
        H, W = fr.shape[:2]
        for f in app.get(fr):
            x1, y1, x2, y2 = f.bbox
            if f.det_score < MIN_DET_SCORE or min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                continue
            m = 0.1 * max(x2 - x1, y2 - y1)
            cx1, cy1, cx2, cy2 = int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(W, x2 + m)), int(min(H, y2 + m))
            pose = getattr(f, 'pose', None)
            dets.append({'frame': fi, 'emb': f.normed_embedding.astype(np.float32),
                         'yaw': float(pose[1]) if pose is not None else 0.0, 'area': float((x2 - x1) * (y2 - y1))})
            crops.append(fr[cy1:cy2, cx1:cx2])
    ex = fer_batch(crops)
    for d, e in zip(dets, ex):
        d['fer'] = e
    FACES[c] = {'n': len(frames), 'faces': dets}
pickle.dump(FACES, open(CACHE, 'wb'))
print(f"done in {(time.time() - t0) / 60:.1f} min")
'''),
    ("markdown", r'''
## Sanity check: does HSEmotion read Hi-EF faces at all? (A's face vs A's gold label)
'''),
    ("code", r'''
AFF2HI = {'anger': 'angry', 'contempt': 'disgust', 'disgust': 'disgust', 'fear': 'fear', 'happiness': 'happy',
          'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprise'}
M = np.zeros((8, 7), np.float32)
for i, a in enumerate(FER8):
    M[i, E2I[AFF2HI[a]]] = 1


def war_uar(pred, y, k=7):
    pred, y = np.asarray(pred), np.asarray(y)
    return (pred == y).mean() * 100, np.mean([(pred[y == c] == c).mean() * 100 for c in range(k) if (y == c).any()])


def roles(row):
    cl = [row['clip1'], row['clip2'], row['clip3']]
    faces = [(k, f) for k, c in enumerate(cl) for f in FACES.get(c, {'faces': []})['faces']]
    out = {}
    if not faces:
        return out
    E = np.stack([f['emb'] for _, f in faces])
    lab_ = (np.zeros(1, int) if len(E) == 1 else
            AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                    distance_threshold=1 - SAME_PERSON_COS).fit_predict(E))
    by = {}
    for (k, f), p in zip(faces, lab_):
        by.setdefault((k, int(p)), []).append(f)

    def dominant(k):
        n = FACES.get(cl[k], {'n': 0})['n']
        cand = [(len({f['frame'] for f in v}), p) for (kk, p), v in by.items() if kk == k]
        if not cand or n == 0:
            return None
        cnt, p = max(cand)
        return p if cnt / n >= DOMINANT_MIN_FRAC else None

    A = dominant(2)
    n3 = max(FACES.get(cl[2], {'n': 1})['n'], 1)
    others = sorted({p for (k, p) in by if k == 2} - {A}, key=lambda p: -len(by[(2, p)]))
    L = others[0] if others else None
    out['A'] = by.get((2, A), []) if A is not None else []
    out['listener'] = [f for f in by.get((2, L), []) if abs(f['yaw']) <= MAX_YAW] if L is not None else []
    out['listener_early'] = [f for f in out['listener'] if f['frame'] / max(n3 - 1, 1) < EARLY_FRAC]
    out['listener_pos'] = [f['frame'] / max(n3 - 1, 1) for f in by.get((2, L), [])] if L is not None else []
    # the same listener identity (clustered jointly over I-III) seen in the context clips; no clip IV needed
    out['listener_ctx'] = ([f for k in (0, 1) for f in by.get((k, L), []) if abs(f['yaw']) <= MAX_YAW]
                           if L is not None else [])
    for k, name in [(1, 'ctx_II'), (0, 'ctx_I')]:
        d = dominant(k)
        out[name] = by.get((k, d), []) if d is not None else []
    out['n3'] = n3
    return out


ROLE_NAMES = ['A', 'listener', 'listener_early', 'listener_ctx', 'ctx_II', 'ctx_I']
R = {r['sample_id']: roles(r) for r in sp.to_dict('records')}


def feat(fs, n):
    if not fs:
        return np.zeros(12, np.float32)
    v = np.stack([f['fer'] for f in fs]).mean(0)
    return np.concatenate([v, [1.0, len({f['frame'] for f in fs}) / max(n, 1)]]).astype(np.float32)


F = {name: np.stack([feat(R[s].get(name, []), R[s].get('n3', 1)) for s in sp.sample_id]) for name in ROLE_NAMES}
sp['has_A'] = F['A'][:, 10] > 0
sp['has_listener'] = F['listener'][:, 10] > 0
sp['has_listener_early'] = F['listener_early'][:, 10] > 0
print(f"A found {sp.has_A.mean() * 100:.1f}% | usable listener {sp.has_listener.mean() * 100:.1f}% | "
      f"usable listener in first {int(EARLY_FRAC * 100)}% of clip III {sp.has_listener_early.mean() * 100:.1f}% | "
      f"listener also seen in clip I/II {(F['listener_ctx'][:, 10] > 0).mean() * 100:.1f}%")
pos = np.concatenate([R[s].get('listener_pos', []) for s in sp.sample_id]) if len(sp) else np.array([])
if len(pos):
    print("listener frame position in clip III (0=start, 1=end), quantiles 10/25/50/75/90%:",
          np.percentile(pos, [10, 25, 50, 75, 90]).round(2), f"| share in last 20%: {(pos >= 0.8).mean() * 100:.1f}%")

for role, target, desc in [('A', 'yA', "A's face vs A's gold label"), ('listener', 'yB', "LISTENER's face vs B's NEXT label")]:
    m = F[role][:, 10] > 0
    if m.sum() == 0:
        print(f"Zero-shot HSEmotion on {desc}: no faces")
        continue
    w, u = war_uar((F[role][m, :8] @ M).argmax(1), sp[target].values[m])
    print(f"Zero-shot HSEmotion on {desc} (n={m.sum()}): WAR {w:.1f}  UAR {u:.1f}  (chance UAR 14.3)")
pd.DataFrame({'sample_id': sp.sample_id, **{f"{n}_{i}": F[n][:, i] for n in ROLE_NAMES for i in range(12)}}).to_csv(
    f"{OUT_DIR}/g6b_role_features.csv", index=False)
'''),
    ("markdown", r'''
## Forecasting B from role features (train → val)
'''),
    ("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

tr, va = (sp.split == 'train').values, (sp.split == 'val').values
src_va = sp.source_folder.values[va]
yB = sp.yB.values
SETS = {
    'A': ['A'],
    'listener': ['listener'],
    'A+listener': ['A', 'listener'],
    'A+listener_early': ['A', 'listener_early'],
    'A+ctx': ['A', 'ctx_II', 'ctx_I'],
    'A+listener+ctx': ['A', 'listener', 'ctx_II', 'ctx_I'],
    'A+listener+listener_ctx': ['A', 'listener', 'listener_ctx'],
}


def fit_predict(names, cw=None):
    X = np.hstack([F[n] for n in names])
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1]:
        s = []
        for a, b in GroupKFold(5).split(Xs[tr], yB[tr], sp.source_folder.values[tr]):
            p = LogisticRegression(max_iter=3000, C=C, class_weight=cw).fit(Xs[tr][a], yB[tr][a]).predict(Xs[tr][b])
            s.append(war_uar(p, yB[tr][b])[1])
        if best is None or np.mean(s) > best[0]:
            best = (np.mean(s), C)
    return LogisticRegression(max_iter=3000, C=best[1], class_weight=cw).fit(Xs[tr], yB[tr]).predict(Xs[va])


rng = np.random.default_rng(0)


def boot(fn, mask):
    src = src_va[mask]
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    out = []
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        out.append(fn(idx))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


PRED = {k: fit_predict(v) for k, v in SETS.items()}
subsets = {'all val': np.ones(va.sum(), bool), 'listener visible': sp.has_listener.values[va],
           'listener not visible': ~sp.has_listener.values[va]}
for sname, msk in subsets.items():
    y = yB[va][msk]
    print(f"\n== {sname}: n={msk.sum()} ==")
    if msk.sum() < 20:
        print("  too few MCIS, skipped")
        continue
    for k, p in PRED.items():
        w, u = war_uar(p[msk], y)
        lo, hi = boot(lambda idx, p=p[msk]: war_uar(p[idx], y[idx]), msk)
        print(f"  {k:<18} UAR {u:5.2f} [{lo[1]:5.1f},{hi[1]:5.1f}]  WAR {w:5.2f} [{lo[0]:5.1f},{hi[0]:5.1f}]")
    for a, b in [('A+listener', 'A'), ('A+listener_early', 'A'), ('A+listener+ctx', 'A+ctx'),
                 ('A+listener+listener_ctx', 'A+listener')]:
        pa, pb = PRED[a][msk], PRED[b][msk]
        d = boot(lambda idx: np.subtract(war_uar(pa[idx], y[idx]), war_uar(pb[idx], y[idx])), msk)
        wa, ua = war_uar(pa, y); wb, ub = war_uar(pb, y)
        print(f"  Δ {a} − {b}: UAR {ua - ub:+.2f} [{d[0][1]:+.2f},{d[1][1]:+.2f}]  WAR {wa - wb:+.2f} [{d[0][0]:+.2f},{d[1][0]:+.2f}]")
print("\nGATE: on 'listener visible', A+listener − A ≥ +2–3 UAR with CI above 0, and A+listener_early keeps most of it.")
'''),
]



# ---------------------------------------------------------------- G7b: B1 + role-grounded face features
G7B = [
    ("markdown", r"""
# G7b — Does B1 improve when it is told *whose* face is whose?

G6b/G6c (train+val, out-of-fold over 45 episodes) showed that reading the **listener's** face in clip III forecasts
B better than reading A's face (zero-shot ΔUAR +3.72 [+0.60, +7.24]), that the gain comes from the expression and
not from whether a reaction shot exists, and that B's face is often already present in clips I/II.
B1 sees whole frames and has no notion of who is who. This notebook adds the G6b **role-grounded face features**
(HSEmotion 8 probabilities + valence/arousal + presence + frame share, per role) to B1.

| Arm | Face input (per role: A, listener in III, listener seen in I/II, dominant face of II, of I) |
|---|---|
| `B1` | none, identical to G3b's B1 (reproduction check) |
| `B1+faces` | all 5 roles × 12 = 60 numbers |
| `B1+faces_early` | same, but the listener uses only frames in the first 80% of clip III (secondary) |
| `B1+presence` | control: only presence + frame share per role, no expression |

Fusion: the face vector (z-scored with training statistics) goes through a small MLP whose **last layer starts at
zero** and is added to B1's pooled representation, so every face arm starts exactly as B1.
Both selection protocols, 5 seeds, paired episode bootstrap against B1. **Test stays locked** (the face CSV has
train+val only).

Gate: `B1+faces` − `B1` (inner-dev selection) ΔUAR > 0 with CI lower bound > 0 and ≥ 3/5 seed wins; under
val selection `B1+faces@val` ≥ `B1@val`; `B1+presence` does not explain the gain.
"""),
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
ROLE_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/role-features/g6b_role_features.csv",
                          "/kaggle/input/role-features/g6b_role_features.csv")
OUT_DIR = "/kaggle/working"

SEEDS = [42, 123, 456, 789, 1024]
N_FOLDS = 5
N_INNER_DEV_SOURCES = 5
REC_EPOCHS, FC_EPOCHS, PATIENCE = 60, 50, 8
REC_BATCH, FC_BATCH = 64, 32
LR, WEIGHT_DECAY = 1e-4, 1e-5
POL_WEIGHT = 0.3
CERT_WEIGHTS = {'1': 1.0, '2': 0.75, '3': 0.5}   # bookkeeping only
REC_SEED = 42

# (name, face set, selection protocol)
EXPERIMENTS = [
    ("B1",                 None,       "inner_dev"),
    ("B1+faces",           "full",     "inner_dev"),
    ("B1+faces_early",     "early",    "inner_dev"),
    ("B1+presence",        "presence", "inner_dev"),
    ("B1@val",             None,       "val"),
    ("B1+faces@val",       "full",     "val"),
    ("B1+faces_early@val", "early",    "val"),
    ("B1+presence@val",    "presence", "val"),
]
G3B_B1_SEED_MEAN = {"inner_dev": 21.65, "val": 25.77}   # G3b per-seed mean UAR of B1, for the reproduction check
EVAL_SPLIT = "val"
UNLOCK_TEST = False
"""),
    G3[2], G3[3], G3[4],
    ("markdown", r"""
## Role-grounded face features (from G6b, clips I–III only)
"""),
    ("code", r"""
ROLE = pd.read_csv(ROLE_CSV).set_index('sample_id')
need = set(train_all.sample_id) | set(ev.sample_id)
miss = need - set(ROLE.index)
assert not miss, f"{len(miss)} MCIS without face features, e.g. {sorted(miss)[:3]}"
FACE_ROLES = {'full': ['A', 'listener', 'listener_ctx', 'ctx_II', 'ctx_I'],
              'early': ['A', 'listener_early', 'listener_ctx', 'ctx_II', 'ctx_I']}
FACE_ROLES['presence'] = FACE_ROLES['full']


def face_cols(fset):
    dims = [10, 11] if fset == 'presence' else range(12)
    return [f"{r}_{i}" for r in FACE_ROLES[fset] for i in dims]


def face_matrix(d, fset):
    return ROLE.loc[d.sample_id, face_cols(fset)].values.astype(np.float32)


# z-score with statistics of the training episodes only
FSTAT = {}
for fset in FACE_ROLES:
    X = face_matrix(train_all, fset)
    FSTAT[fset] = (X.mean(0), X.std(0) + 1e-6)

vis = ROLE.loc[ev.sample_id, 'listener_10'].values > 0
early = ROLE.loc[ev.sample_id, 'listener_early_10'].values > 0
bctx = ROLE.loc[ev.sample_id, 'listener_ctx_10'].values > 0
print(f"{EVAL_SPLIT}: listener visible {vis.mean() * 100:.1f}% | in first 80% of III {early.mean() * 100:.1f}% | "
      f"listener also seen in I/II {bctx.mean() * 100:.1f}%")


class FaceForecaster(nn.Module):
    # B1 (raw clip encoder over I-III) + a role-face vector added to the pooled representation.
    # The face MLP's last layer is zero-initialised, so the model starts exactly as B1.

    def __init__(self, n_face, d=512):
        super().__init__()
        self.base = Forecaster(use_raw=True, use_traj=False, d=d)
        self.face = nn.Sequential(nn.Linear(n_face, d // 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d // 2, d))
        nn.init.zeros_(self.face[-1].weight); nn.init.zeros_(self.face[-1].bias)

    def forward(self, clip_idx, rec, face):
        b = self.base
        B, n = clip_idx.shape
        feats = gather(clip_idx)
        flat = {k: v.reshape(B * n, *v.shape[2:]) for k, v in feats.items()}
        tok = b.enc(flat).reshape(B, n, -1)
        h = b.inter(tok + b.clip_pos[:, 3 - n:])
        return b.head(h.mean(1) + self.face(face))
"""),
    ("markdown", r"""
## Forecasters under both selection protocols
"""),
    ("code", r"""
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


fc_train = train_all[~train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
fc_dev = train_all[train_all.source_folder.isin(inner_dev_sources)].reset_index(drop=True)
_T = {}


def make_T(rows_name, d, fset):
    key = (rows_name, fset)
    if key not in _T:
        idx = torch.tensor([[CIDX[c] for c in r] for r in d[['clip1', 'clip2', 'clip3']].values], device=DEVICE)
        rec = torch.zeros(len(d), 3, N_REC, device=DEVICE)
        if fset is None:
            face = None
        else:
            mu, sd = FSTAT[fset]
            face = torch.tensor((face_matrix(d, fset) - mu) / sd, device=DEVICE)
        _T[key] = (idx, rec, torch.tensor(d.yB.values, device=DEVICE), face)
    return _T[key]


def run(model, T, i, j):
    return model(T[0][i:j], T[1][i:j]) if T[3] is None else model(T[0][i:j], T[1][i:j], T[3][i:j])


def fc_predict(model, T, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(T[0]), bs):
            out.append(F.softmax(run(model, T, i, i + bs), -1).cpu())
    return torch.cat(out).numpy()


def train_forecaster(fset, protocol, seed):
    seed_all(seed)
    tr_rows, tr_name = (train_all, 'train_all') if protocol == 'val' else (fc_train, 'fc_train')
    T_tr = make_T(tr_name, tr_rows, fset)
    T_ev = make_T('ev', ev, fset)
    T_sel = T_ev if protocol == 'val' else make_T('fc_dev', fc_dev, fset)
    model = (Forecaster(use_raw=True, use_traj=False) if fset is None
             else FaceForecaster(T_tr[3].shape[1])).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y = T_tr[2]
    y_sel = T_sel[2].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(FC_EPOCHS):
        model.train()
        perm = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(perm), FC_BATCH):
            j = perm[i:i + FC_BATCH]
            Tj = tuple(t[j] if t is not None else None for t in T_tr)
            loss = F.cross_entropy(run(model, Tj, 0, len(j)), y[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sel_uar = war_uar(fc_predict(model, T_sel).argmax(1), y_sel, 7)[1]
        if sel_uar > best:
            best, bad = sel_uar, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return fc_predict(model, T_ev), best


yB = ev.yB.values
src = ev.source_folder.values
results, PROBS = [], {}
for name, fset, protocol in EXPERIMENTS:
    PROBS[name] = []
    for seed in SEEDS:
        p, sel = train_forecaster(fset, protocol, seed)
        PROBS[name].append(p)
        w, u = war_uar(p.argmax(1), yB, 7)
        wv, uv = war_uar(p[vis].argmax(1), yB[vis], 7)
        results.append({'exp': name, 'protocol': protocol, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w,
                        'UAR_listener_visible': uv, 'WAR_listener_visible': wv})
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in results[-1].items()}, flush=True)
    torch.cuda.empty_cache()

res = pd.DataFrame(results)
res.to_csv(f"{OUT_DIR}/g7b_results_per_seed.csv", index=False)
np.savez(f"{OUT_DIR}/g7b_{EVAL_SPLIT}_probs.npz", sample_id=ev.sample_id.values,
         **{n.replace('@', '_at_').replace('+', '_'): np.stack(v) for n, v in PROBS.items()})
print("\n== mean ± std over seeds ==")
print(res.drop(columns=['seed', 'protocol']).groupby('exp', sort=False).agg(['mean', 'std']).round(2).to_string())
for prot, ref in G3B_B1_SEED_MEAN.items():
    got = res[(res.exp == ('B1' if prot == 'inner_dev' else 'B1@val'))].UAR.mean()
    print(f"reproduction check, B1 ({prot}): seed-mean UAR {got:.2f} vs G3b {ref:.2f}")
"""),
    ("markdown", r"""
## Paired comparisons against B1, subsets and the gate
"""),
    ("code", r"""
def paired_diff_ci(pa, pb, y, src, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7)
        wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


ENS = {n: np.mean(v, 0).argmax(1) for n, v in PROBS.items()}
SUBSETS = {'all': np.ones(len(yB), bool), 'listener visible': vis, 'listener not visible': ~vis,
           'listener in first 80% of III': early, 'listener also seen in I/II': bctx}
PAIRS = [("B1+faces", "B1"), ("B1+faces_early", "B1"), ("B1+presence", "B1"),
         ("B1+faces@val", "B1@val"), ("B1+faces_early@val", "B1@val"), ("B1+presence@val", "B1@val")]
verdict = {}
for sname, m in SUBSETS.items():
    print(f"\n== {sname}: n={m.sum()} ==")
    if m.sum() < 30:
        print("  too few MCIS, skipped")
        continue
    for n in ENS:
        report(f"  {n}", ENS[n][m], yB[m], src[m])
    for a, b in PAIRS:
        lo, hi = paired_diff_ci(ENS[a][m], ENS[b][m], yB[m], src[m])
        wa, ua = war_uar(ENS[a][m], yB[m], 7); wb, ub = war_uar(ENS[b][m], yB[m], 7)
        col = 'UAR' if sname == 'all' else ('UAR_listener_visible' if sname == 'listener visible' else None)
        wins = ""
        if col:
            pa = res[res.exp == a].set_index('seed')[col]; pb = res[res.exp == b].set_index('seed')[col]
            wins = f"({int(((pa - pb) > 0).sum())}/{len(SEEDS)} seeds)"
            verdict[(sname, a)] = (ua - ub, lo[0], hi[0], int(((pa - pb) > 0).sum()))
        print(f"  {a:<20} - {b:<7} ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}] {wins}  "
              f"ΔWAR {wa - wb:+5.2f} [{lo[1]:+5.2f},{hi[1]:+5.2f}]")

d1 = verdict[('all', 'B1+faces')]
d2 = verdict[('all', 'B1+faces@val')]
dp = verdict[('all', 'B1+presence')]
print("\n== GATE ==")
print(f"1. B1+faces vs B1 (inner_dev): ΔUAR {d1[0]:+.2f} [{d1[1]:+.2f},{d1[2]:+.2f}], {d1[3]}/{len(SEEDS)} seed wins -> "
      f"{'PASS' if d1[1] > 0 and d1[3] >= 3 else ('WEAK (positive, CI includes 0)' if d1[0] > 0 else 'FAIL')}")
print(f"2. B1+faces@val vs B1@val: ΔUAR {d2[0]:+.2f} -> {'PASS' if d2[0] >= 0 else 'FAIL'}")
print(f"3. presence-only control (inner_dev): ΔUAR {dp[0]:+.2f} -> "
      f"{'OK (below faces)' if dp[0] < d1[0] else 'CAUTION: presence alone explains the gain'}")
print("4. listener-visible subset and early-frame arm: see tables above (descriptive)")
"""),
]



# ---------------------------------------------------------------- G8a: role-agnostic face / voice extraction
G8A = [
    ("markdown", r"""
# G8a — Rich per-face and per-voice features for role-grounded forecasting

One extraction pass that every later role-grounded model reads. It stores **what is seen and heard**, not who is who;
roles (A, listener, others, speaker of I/II) are assigned later from these features, so the role rules can change
without re-extracting.

* Clips: every clip used as clip **I, II or III** by any MCIS in train / val / **test**. Clip IV is never read.
  No label is read (the split file is used only for clip ids), so extracting test inputs does not unlock the test.
* Per sampled frame (4 fps, ≤ 32 frames) and per detected face: box, detection score, head pose, ArcFace identity
  (512, fp16), HSEmotion 8 emotion logits + valence/arousal, the HSEmotion **penultimate embedding** (fp16),
  mouth opening from the 68 3-D landmarks, and 68 landmarks (fp16).
* Per clip audio: ECAPA speaker embedding of the whole clip and of 1.5 s windows (hop 0.75 s), plus an RMS energy
  envelope at 10 Hz (used later to match mouth movement to speech → who is speaking).

Output: `/kaggle/working/g8a/shard_*.pkl` (resumable). Expected run time ≈ 3–4 h on one T4 — use *Save & Run All*,
then turn the notebook output into a dataset (e.g. `g8a-features`).
"""),
    ("code", r"""
!pip install -q insightface speechbrain onnx
!pip uninstall -y -q onnxruntime onnxruntime-gpu
!pip install -q "onnxruntime-gpu==1.22.0"
"""),
    ("code", r"""
# ======== CONFIG ========
DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
OUT_DIR = "/kaggle/working"
SHARD_DIR = f"{OUT_DIR}/g8a"
SHARD_SIZE = 250

SAMPLE_FPS, MAX_FRAMES = 4, 32
DET_SIZE = (640, 640)
MIN_DET_SCORE, MIN_FACE_PX = 0.5, 24
AUDIO_SR = 16000
WIN_S, HOP_S = 1.5, 0.75           # windowed ECAPA
ENV_HZ = 10                        # RMS energy envelope rate
N_DECODE_THREADS = 4
FER_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/"
           "enet_b0_8_va_mtl.onnx")
"""),
    ("code", r"""
import os, glob, pickle, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2
import torch
import onnxruntime as ort
if hasattr(ort, 'preload_dlls'):
    try:
        ort.preload_dlls()
    except Exception as e:
        print("preload_dlls:", e)
PROV = ['CUDAExecutionProvider', 'CPUExecutionProvider']

roots = sorted(glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF")))
VIDEO_ROOTS = [os.path.join(r, "video") for r in roots if os.path.isdir(os.path.join(r, "video"))]
AUDIO_ROOTS = [os.path.join(r, "audio") for r in roots if os.path.isdir(os.path.join(r, "audio"))]


def find_media(roots_, clip, exts):
    ep, num = clip.split('/')
    for root in roots_:
        for ext in exts:
            p = os.path.join(root, ep, num + ext)
            if os.path.exists(p):
                return p
    return None


sp = pd.read_csv(SPLIT_CSV, dtype=str, usecols=['sample_id', 'split', 'clip1', 'clip2', 'clip3'])   # no label columns
clips = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))
print(f"MCIS {len(sp)} {sp.split.value_counts().to_dict()} | clips I-III {len(clips)} | "
      f"missing video {sum(find_media(VIDEO_ROOTS, c, ('.mp4',)) is None for c in clips)} | "
      f"missing audio {sum(find_media(AUDIO_ROOTS, c, ('.mp3', '.wav')) is None for c in clips)}")
"""),
    ("code", r"""
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark_3d_68'], providers=PROV)
app.prepare(ctx_id=0, det_size=DET_SIZE)
print("insightface providers:", {k: m.session.get_providers() for k, m in app.models.items()})

fer_path = f"{OUT_DIR}/enet_b0_8_va_mtl.onnx"
if not os.path.exists(fer_path) or os.path.getsize(fer_path) < 1_000_000:   # re-fetch missing / truncated files
    urllib.request.urlretrieve(FER_URL, fer_path)
assert os.path.getsize(fer_path) > 1_000_000, f"FER model download looks broken ({os.path.getsize(fer_path)} bytes)"
# expose the penultimate (pooled) features as a second output: input of the last Gemm/MatMul
fer_emb_path = f"{OUT_DIR}/enet_b0_8_va_mtl_emb.onnx"
try:
    import onnx
    m = onnx.load(fer_path)
    last = [n for n in m.graph.node if n.op_type in ('Gemm', 'MatMul')][-1]
    m.graph.output.append(onnx.helper.make_tensor_value_info(last.input[0], onnx.TensorProto.FLOAT, None))
    onnx.save(m, fer_emb_path)
    fer = ort.InferenceSession(fer_emb_path, providers=PROV)
except Exception as e:
    print("could not expose the FER embedding, logits only:", repr(e))
    fer = ort.InferenceSession(fer_path, providers=PROV)
FER_IN = fer.get_inputs()[0].name
print("FER providers:", fer.get_providers(), "| outputs", [(o.name, o.shape) for o in fer.get_outputs()])
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def fer_batch(crops):
    if not crops:
        return np.zeros((0, 10), np.float32), None
    x = np.stack([(cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                  for c in crops]).transpose(0, 3, 1, 2).astype(np.float32)
    outs = fer.run(None, {FER_IN: x})
    head = outs[0][:, :10] if outs[0].shape[1] >= 10 else np.pad(outs[0], ((0, 0), (0, 10 - outs[0].shape[1])))
    emb = outs[1].reshape(len(x), -1).astype(np.float16) if len(outs) > 1 else None
    return head.astype(np.float32), emb


def read_frames(path):
    # sequential decode, keep ~SAMPLE_FPS frames per second (<= MAX_FRAMES, evenly spread)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = max(1, min(MAX_FRAMES, int(round(n / fps * SAMPLE_FPS)))) if n else MAX_FRAMES
    keep = set(np.linspace(0, max(n - 1, 0), want).astype(int).tolist())
    frames, times, i = [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i in keep:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(fr); times.append(i / fps)
        i += 1
        if i > max(keep, default=0):
            break
    cap.release()
    return frames, times, {'n_video_frames': n, 'fps': fps, 'duration': n / fps if fps else 0.0}


def analyse_frames(frames, times):
    dets, crops = [], []
    for fi, fr in enumerate(frames):
        H, W = fr.shape[:2]
        for f in app.get(fr):
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            if f.det_score < MIN_DET_SCORE or min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                continue
            m = 0.1 * max(x2 - x1, y2 - y1)
            crops.append(fr[int(max(0, y1 - m)):int(min(H, y2 + m)), int(max(0, x1 - m)):int(min(W, x2 + m))])
            lm = getattr(f, 'landmark_3d_68', None)
            pose = getattr(f, 'pose', None)
            d = {'frame': fi, 't': times[fi], 'box': np.array([x1 / W, y1 / H, x2 / W, y2 / H], np.float32),
                 'score': float(f.det_score), 'pose': np.asarray(pose if pose is not None else [0, 0, 0], np.float32),
                 'arc': f.normed_embedding.astype(np.float16)}
            if lm is not None:
                bh = max(y2 - y1, 1.0)
                d['mouth'] = float(np.linalg.norm(lm[66, :2] - lm[62, :2]) / bh)
                d['lm'] = np.stack([(lm[:, 0] - x1) / max(x2 - x1, 1.0), (lm[:, 1] - y1) / bh], 1).astype(np.float16)
            else:
                d['mouth'] = np.nan
            dets.append(d)
    head, emb = fer_batch(crops)
    for k, d in enumerate(dets):
        d['fer'] = head[k]
        if emb is not None:
            d['fer_emb'] = emb[k]
    return dets
"""),
    ("code", r"""
import librosa
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{OUT_DIR}/ecapa",
                                     run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})


def ecapa(wav):
    with torch.no_grad():
        e = spk.encode_batch(torch.tensor(wav, dtype=torch.float32).unsqueeze(0)).reshape(-1).cpu().numpy()
    return (e / (np.linalg.norm(e) + 1e-9)).astype(np.float16)


def analyse_audio(path):
    if path is None:
        return None
    try:
        wav, _ = librosa.load(path, sr=AUDIO_SR, mono=True)
    except Exception:
        return None
    out = {'duration': len(wav) / AUDIO_SR}
    hop = AUDIO_SR // ENV_HZ
    out['env'] = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2)) if len(wav[i:i + hop]) else 0.0
                           for i in range(0, len(wav), hop)], np.float32)
    out['ecapa'] = ecapa(wav) if len(wav) >= AUDIO_SR // 2 else None
    win, step = int(WIN_S * AUDIO_SR), int(HOP_S * AUDIO_SR)
    starts = list(range(0, max(len(wav) - win, 0) + 1, step)) if len(wav) >= win else []
    out['win_t'] = np.array([s / AUDIO_SR for s in starts], np.float32)
    out['win_ecapa'] = np.stack([ecapa(wav[s:s + win]) for s in starts]) if starts else np.zeros((0, 192), np.float16)
    return out
"""),
    ("code", r"""
os.makedirs(SHARD_DIR, exist_ok=True)
done = set()
for f in sorted(glob.glob(f"{SHARD_DIR}/shard_*.pkl")):
    done |= set(pickle.load(open(f, 'rb')))
todo = [c for c in clips if c not in done]
n_shard = len(glob.glob(f"{SHARD_DIR}/shard_*.pkl"))
print(f"already extracted {len(done)} | to do {len(todo)}")


def load(c):
    p = find_media(VIDEO_ROOTS, c, ('.mp4', '.avi', '.mkv', '.mov'))
    return c, (read_frames(p) if p else ([], [], {'n_video_frames': 0, 'fps': 0.0, 'duration': 0.0}))


from collections import deque
from itertools import islice
PREFETCH = 8                        # decoded clips held in memory at most
t0, buf, n_done = time.time(), {}, 0


def flush():
    global n_shard, buf
    if buf:
        pickle.dump(buf, open(f"{SHARD_DIR}/shard_{n_shard:03d}.pkl", 'wb'))
        n_shard += 1
        buf = {}
    el = (time.time() - t0) / 60
    print(f"clips {n_done}/{len(todo)} | {el:.1f} min | ETA {el / max(n_done, 1) * (len(todo) - n_done):.0f} min", flush=True)


with ThreadPoolExecutor(N_DECODE_THREADS) as pool:
    it = iter(todo)
    q = deque(pool.submit(load, c) for c in islice(it, PREFETCH))
    while q:
        c, (frames, times, meta) = q.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            q.append(pool.submit(load, nxt))
        buf[c] = {'meta': {**meta, 'n_sampled': len(frames)}, 'faces': analyse_frames(frames, times),
                  'audio': analyse_audio(find_media(AUDIO_ROOTS, c, ('.mp3', '.wav', '.flac', '.m4a')))}
        n_done += 1
        if len(buf) >= SHARD_SIZE:
            flush()
flush()
assert n_done == len(todo), (n_done, len(todo))
print("extraction done")
"""),
    ("markdown", r"""
## Sanity checks (no labels)
"""),
    ("code", r"""
DATA = {}
for f in sorted(glob.glob(f"{SHARD_DIR}/shard_*.pkl")):
    DATA.update(pickle.load(open(f, 'rb')))
assert set(clips) <= set(DATA), f"{len(set(clips) - set(DATA))} clips missing"
nf = np.array([len(DATA[c]['faces']) for c in clips])
ns = np.array([DATA[c]['meta']['n_sampled'] for c in clips])
print(f"clips {len(clips)} | sampled frames mean {ns.mean():.1f} | faces per clip mean {nf.mean():.1f} | "
      f"clips with no face {(nf == 0).mean() * 100:.1f}% | audio found "
      f"{np.mean([DATA[c]['audio'] is not None for c in clips]) * 100:.1f}%")
ex = next((d for c in clips for d in DATA[c]['faces']), None)
if ex is not None:
    print("face record:", {k: (v.shape, v.dtype) if hasattr(v, 'shape') else type(v).__name__ for k, v in ex.items()})


def tracks(faces, thr=0.45):
    # greedy identity grouping inside one clip by ArcFace cosine
    reps, lab = [], []
    for d in faces:
        e = d['arc'].astype(np.float32)
        sims = [float(e @ r) for r in reps]
        if sims and max(sims) >= thr:
            lab.append(int(np.argmax(sims)))
        else:
            reps.append(e); lab.append(len(reps) - 1)
    return np.array(lab)


# active-speaker sanity on clip III (A speaks there): mouth movement of the most frequent face should follow the
# audio energy more closely than other faces do
c3 = sorted(set(sp.clip3))
cors = {'dominant': [], 'other': []}
for c in c3:
    D = DATA[c]
    if D['audio'] is None or len(D['faces']) < 4:
        continue
    lab = tracks(D['faces'])
    env = D['audio']['env']
    counts = np.bincount(lab)
    for k in np.unique(lab):
        fs = [d for d, l in zip(D['faces'], lab) if l == k and np.isfinite(d['mouth'])]
        if len(fs) < 4:
            continue
        m = np.array([d['mouth'] for d in fs])
        e = np.array([env[min(int(d['t'] * ENV_HZ), len(env) - 1)] for d in fs])
        if m.std() < 1e-6 or e.std() < 1e-9:
            continue
        cors['dominant' if k == counts.argmax() else 'other'].append(np.corrcoef(m, e)[0, 1])
print({k: (len(v), round(float(np.mean(v)), 3) if v else None) for k, v in cors.items()},
      "<- mean corr(mouth opening, audio energy); expect dominant > other")
"""),
]



# ---------------------------------------------------------------- G8b: role-grounded forecaster, episode CV
G8B = [
    ("markdown", r"""
# G8b — Role-grounded forecasting of B's emotion (episode-level cross-validation)

**Idea.** B1 encodes whole frames and never knows *who* is on screen. RoleNet turns the context into tokens tagged
with **who** they are about. Roles are assigned automatically from the G8a features, with no manual annotation and
no clip IV input:

* **A** = the most frequent identity in clip III (the speaker).
* **L** = the most frequent *other* identity in clip III (the listener; this is B in about 81% of MCIS, per G6a).
* **O** = everyone else.
* Identities are clustered jointly over clips I–III, so A and L are also found in clips I and II.

RoleNet has three token types plus a query and an encoder:

* **Face tokens (3 roles × 3 clips).** Attention-pooled per-frame features:
  * the HSEmotion embedding reduced to 128 dims by PCA, fitted **inside each training fold**;
  * emotion probabilities, valence/arousal;
  * head pose, box, mouth opening, time.

  A role missing from a clip gets a learned **absent** token.
* **Speech tokens (one per clip).** The benchmark's text and audio features, plus who-speaks cues.
* **Scene tokens (one per clip).** Means of the whole-frame and face-crop CLIP features.
* **Encoder.** A 2-layer Transformer with a query token.
* **Balance aids.** Unimodal auxiliary heads, an auxiliary head for A's emotion, and modality dropout.

**Arms (same folds, same seeds):**

| Arm | What it is |
|---|---|
| `B1` | G3b's model |
| `B1+faces-joint` | B1 plus G6b-style face features in a jointly trained, zero-initialised branch (the G7b design) |
| `LateFusion` | B1 ⊕ **unbalanced** face LR, equal log-space weights |
| `LateFusion-balancedLR` | The old G7 variant, kept for reference |
| `RoleNet` | The full model above |
| RoleNet ablations | noRole, noBalance, facesOnly, ctxOnly |

**Prior handling (same for every arm, fixed in advance).**

* Every model is trained with plain cross-entropy.
* The seed-averaged probabilities are scored twice:
  * *plain*: argmax;
  * *LA*: argmax of log p − 1·log π, where π is the training-fold class prior.

  This logit adjustment is applied exactly once.

**Protocol (fixed before running).** 5-fold CV over the 45 train+val episodes, with folds balanced by MCIS count.
Inside each outer training set, 5 episodes are held out for early stopping. 3 seeds per arm. Hyper-parameters are
set a priori. The test split stays untouched.

**Primary contrast:** `RoleNet` − `B1` under LA. This is the pooled out-of-fold ΔUAR of the seed ensemble, with a 95%
bootstrap over the 45 episodes.

**Diagnostics.** These test hypotheses raised in the design debate:

1. Is one person split into two identities?
2. Does mouth–audio synchrony point to the speaker?
3. Is emotional inertia specific to B? Clip IV audio is read **only** for this analysis: it answers whether B spoke
   in clip I or II. It is never a model input.
"""),
    ("code", r"""
!pip install -q speechbrain
"""),
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
G8A_DIR = first_existing("/kaggle/input/datasets/ptrnghieu/g8a-features", "/kaggle/input/g8a-features")
OUT_DIR = "/kaggle/working"

N_OUTER, N_INNER_DEV = 5, 5
SEEDS = [42, 123, 456]
# B1, exactly as G3b
LR, WEIGHT_DECAY = 1e-4, 1e-5
FC_EPOCHS, PATIENCE, FC_BATCH = 50, 8, 32
# RoleNet, set a priori (not tuned)
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
LATE_W = 0.5
LA_TAU = 1.0                 # post-hoc logit adjustment, applied once to seed-averaged probabilities
VOICE_SAME_COS = 0.35        # ECAPA cosine taken as "same speaker" (as in G6a)
RUN_PERSISTENCE_DIAG = True  # reads clip-IV audio for an analysis only (never a model input)
DEBUG_PER_EPISODE = None     # e.g. 6 for a quick smoke test

FULL = dict(role=True, faces=True, ctx=True, aux=True, mdrop=True)
EXPERIMENTS = [
    ("B1",                'b1',     None),
    ("B1+faces-joint",    'b1face', None),
    ("RoleNet",           'role',   FULL),
    ("RoleNet-noRole",    'role',   {**FULL, 'role': False}),
    ("RoleNet-noBalance", 'role',   {**FULL, 'aux': False, 'mdrop': False}),
    ("RoleNet-facesOnly", 'role',   {**FULL, 'ctx': False, 'mdrop': False}),
    ("RoleNet-ctxOnly",   'role',   {**FULL, 'faces': False, 'mdrop': False}),
]
"""),
    ("code", COMMON + r"""
import glob, pickle
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOT_CSV = glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF", "annotation.csv"))[0]
ann, sp = load_tables(ANNOT_CSV, SPLIT_CSV)
sp = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)      # test rows are dropped here
assert 'test' not in set(sp.split)
if DEBUG_PER_EPISODE:
    sp = sp.groupby('source_folder', group_keys=False).head(DEBUG_PER_EPISODE).reset_index(drop=True)
DEV = sp
train_all = ev = DEV          # names used by the shared feature-loading cell
N = len(DEV)
EPS = np.array(sorted(DEV.source_folder.unique()))
print(f"development MCIS {N} | episodes {len(EPS)}")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
"""),
    G3[3], G3[4],
    ("markdown", r"""
## From G8a features to role-tagged slots
"""),
    ("code", r"""
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

G8 = {}
for f in sorted(glob.glob(os.path.join(G8A_DIR, '**', 'shard_*.pkl'), recursive=True)):
    G8.update(pickle.load(open(f, 'rb')))
need = sorted(set(DEV[['clip1', 'clip2', 'clip3']].values.ravel()))
miss = [c for c in need if c not in G8]
assert not miss, f"{len(miss)} clips missing from G8a, e.g. {miss[:3]}"


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# per-clip frame features except the PCA part (18 dims), and the raw HSEmotion embeddings
FB, EMB = {}, {}
for c in need:
    fs = G8[c]['faces']
    if not fs:
        FB[c], EMB[c] = np.zeros((0, 18), np.float32), None
        continue
    dur = max(G8[c]['meta'].get('duration') or 0.0, 1e-3)
    fer = np.stack([d['fer'] for d in fs]).astype(np.float32)
    box = np.stack([d['box'] for d in fs])
    FB[c] = np.concatenate([
        softmax(fer[:, :8]), fer[:, 8:10], np.stack([d['pose'] for d in fs]) / 90.0,
        np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2,
                  np.sqrt(np.clip((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 0, None))], 1),
        np.nan_to_num(np.array([[d['mouth'] * 10] for d in fs], np.float32)),
        np.array([[min(d['t'] / dur, 1.0)] for d in fs], np.float32)], 1).astype(np.float32)
    EMB[c] = np.stack([d['fer_emb'] for d in fs]) if all('fer_emb' in d for d in fs) else None
HAS_EMB = all(EMB[c] is not None for c in need if len(G8[c]['faces']))
FDIM = (PCA_DIM if HAS_EMB else 0) + 18
print(f"HSEmotion embedding available: {HAS_EMB} | frame feature dim {FDIM}")


def pick_frames(idx, cap):
    return idx if len(idx) <= cap else [idx[i] for i in np.linspace(0, len(idx) - 1, cap).astype(int)]


def sync(c, face_ids):
    # correlation of mouth opening with the audio energy envelope, and mouth variability, for a set of faces
    A = G8[c]['audio']
    fs = [G8[c]['faces'][j] for j in face_ids]
    fs = [d for d in fs if np.isfinite(d['mouth'])]
    if A is None or len(fs) < 4:
        return 0.0, 0.0
    m = np.array([d['mouth'] for d in fs])
    env = A['env']
    e = np.array([env[min(int(d['t'] * 10), len(env) - 1)] for d in fs]) if len(env) else np.zeros(len(fs))
    r = float(np.corrcoef(m, e)[0, 1]) if m.std() > 1e-6 and e.std() > 1e-9 else 0.0
    return r, float(m.std() * 10)


def mean12(c_faces, n_sampled):
    if not c_faces:
        return np.zeros(12, np.float32)
    fer = np.stack([d['fer'] for d in c_faces]).astype(np.float32)
    v = np.concatenate([softmax(fer[:, :8]), fer[:, 8:10]], 1).mean(0)
    return np.concatenate([v, [1.0, len({d['frame'] for d in c_faces}) / max(n_sampled, 1)]]).astype(np.float32)


NVOICE = 3 * 2 + 3
SLOT, PSLOT = {}, {}                      # (n, role, clip) / (n, clip) -> (clip id, face indices)
FMASK = np.zeros((N, 3, 3, MAXF), bool); PMASK = np.zeros((N, 1, 3, MAXF_POOL), bool)
VOI = np.zeros((N, 3, NVOICE), np.float32)
LRF = np.zeros((N, 60), np.float32)       # G6b-style role means (late fusion and the joint-branch arm)
CENT_COS = np.full(N, np.nan, np.float32)
for n, row in enumerate(tqdm(DEV.itertuples(), total=N, desc='roles')):
    cl = [row.clip1, row.clip2, row.clip3]
    items = [(k, j) for k, c in enumerate(cl) for j in range(len(G8[c]['faces']))]
    lab = np.zeros(len(items), int)
    E = np.stack([G8[cl[k]]['faces'][j]['arc'] for k, j in items]).astype(np.float32) if items else None
    if len(items) > 1:
        lab = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                      distance_threshold=1 - SAME_PERSON_COS).fit_predict(E)
    frames = defaultdict(set)
    for (k, j), p in zip(items, lab):
        frames[(k, p)].add(G8[cl[k]]['faces'][j]['frame'])
    ids3 = sorted({p for (k, p) in frames if k == 2}, key=lambda p: -len(frames[(2, p)]))
    A = ids3[0] if ids3 else None
    L = ids3[1] if len(ids3) > 1 else None
    if A is not None and L is not None:
        ca, cb = E[lab == A].mean(0), E[lab == L].mean(0)
        CENT_COS[n] = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9))
    role = lambda p: 0 if p == A else (1 if p == L else 2)
    by = defaultdict(list)
    for (k, j), p in zip(items, lab):
        by[(role(p), k)].append(j)
        by[('pool', k)].append(j)
    for k, c in enumerate(cl):
        for r in range(3):
            idx = pick_frames(sorted(by[(r, k)], key=lambda j: G8[c]['faces'][j]['t']), MAXF)
            SLOT[(n, r, k)] = (c, idx); FMASK[n, r, k, :len(idx)] = True
        idx = pick_frames(sorted(by[('pool', k)], key=lambda j: G8[c]['faces'][j]['t']), MAXF_POOL)
        PSLOT[(n, k)] = (c, idx); PMASK[n, 0, k, :len(idx)] = True
        v = [x for r in range(3) for x in sync(c, by[(r, k)])]
        a3, ak = G8[cl[2]]['audio'], G8[c]['audio']
        ok = a3 is not None and ak is not None and a3.get('ecapa') is not None and ak.get('ecapa') is not None
        vcos = float(a3['ecapa'].astype(np.float32) @ ak['ecapa'].astype(np.float32)) if ok else 0.0
        VOI[n, k] = v + [vcos, float(ok), len(set(lab)) / 5.0]

    def faces_of(k, p):
        return [G8[cl[k]]['faces'][j] for (kk, j), q in zip(items, lab) if kk == k and q == p]

    def dominant(k):
        ns = G8[cl[k]]['meta']['n_sampled']
        cand = sorted({q for (kk, q) in frames if kk == k}, key=lambda q: -len(frames[(k, q)]))
        return cand[0] if cand and len(frames[(k, cand[0])]) / max(ns, 1) >= DOMINANT_MIN_FRAC else None

    n3 = G8[cl[2]]['meta']['n_sampled']
    blocks = [mean12(faces_of(2, A) if A is not None else [], n3), mean12(faces_of(2, L) if L is not None else [], n3),
              mean12((faces_of(0, L) + faces_of(1, L)) if L is not None else [], n3)]
    for k in (1, 0):
        d = dominant(k)
        blocks.append(mean12(faces_of(k, d) if d is not None else [], G8[cl[k]]['meta']['n_sampled']))
    LRF[n] = np.concatenate(blocks)

VIS = FMASK[:, 1, 2].any(-1)
print(f"A found in III {FMASK[:, 0, 2].any(-1).mean() * 100:.1f}% | listener visible in III {VIS.mean() * 100:.1f}% | "
      f"listener also in I/II {FMASK[:, 1, :2].any((-1, -2)).mean() * 100:.1f}%")


def build_face_tensors(fit_clips):
    # PCA of the HSEmotion embedding fitted on faces of the training clips of the current fold only
    pca, var = None, None
    if HAS_EMB:
        pool = np.concatenate([EMB[c] for c in fit_clips if EMB.get(c) is not None])
        pick = np.random.default_rng(0).choice(len(pool), min(60000, len(pool)), replace=False)
        pca = PCA(PCA_DIM, random_state=0).fit(pool[pick].astype(np.float32))
        var = float(pca.explained_variance_ratio_.sum())
    FV = {}
    for c in need:
        if len(FB[c]) == 0:
            FV[c] = np.zeros((0, FDIM), np.float32)
        else:
            FV[c] = np.concatenate([pca.transform(EMB[c].astype(np.float32)), FB[c]], 1) if HAS_EMB else FB[c]
    Fa = np.zeros((N, 3, 3, MAXF, FDIM), np.float16)
    Pa = np.zeros((N, 1, 3, MAXF_POOL, FDIM), np.float16)
    for (n, r, k), (c, idx) in SLOT.items():
        if idx:
            Fa[n, r, k, :len(idx)] = FV[c][idx]
    for (n, k), (c, idx) in PSLOT.items():
        if idx:
            Pa[n, 0, k, :len(idx)] = FV[c][idx]
    return torch.tensor(Fa, device=DEVICE), torch.tensor(Pa, device=DEVICE), var
"""),
    ("markdown", r"""
## Diagnostics for hypotheses raised in the design debate (no model involved)
"""),
    ("code", r"""
from sklearn.metrics import roc_auc_score

# (1) is one person split into two identities? cosine between the mean ArcFace embeddings of A and L
cc = CENT_COS[np.isfinite(CENT_COS)]
if len(cc):
    print(f"(1) A-vs-L centroid cosine, quantiles 10/25/50/75/90%: {np.percentile(cc, [10, 25, 50, 75, 90]).round(2)} | "
          f"share in [0.30, 0.45) (possible split person): {((cc >= 0.30) & (cc < 0.45)).mean() * 100:.1f}%")

# (2) does mouth-audio synchrony point to the speaker? In clips I/II the voice says whether A speaks (ECAPA vs clip III).
xs, ys = [], []
for n in range(N):
    for k in (0, 1):
        v = VOI[n, k]
        others = [v[2] if FMASK[n, 1, k].any() else None, v[4] if FMASK[n, 2, k].any() else None]
        others = [o for o in others if o is not None]
        if v[7] < 1 or not FMASK[n, 0, k].any() or not others:
            continue
        xs.append(v[0] - max(others)); ys.append(v[6] >= VOICE_SAME_COS)
if len(set(ys)) == 2:
    print(f"(2) sync margin (A minus others) separates 'A speaks' from 'someone else speaks': AUC {roc_auc_score(ys, xs):.3f} "
          f"on {len(ys)} context clips (0.5 = noise; >= 0.7 = usable)")
else:
    print("(2) not enough context clips with both A and another face for the sync check")
"""),
    ("code", r"""
# (3) is emotional inertia person-specific? Clip IV voice (analysis only) says whether B spoke in clip I / II.
if RUN_PERSISTENCE_DIAG:
    import librosa
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{OUT_DIR}/ecapa",
                                         run_opts={"device": DEVICE})
    AUDIO_ROOTS = [os.path.join(r, 'audio') for r in glob.glob(os.path.join(DATASET_DIR, '*', 'Hi-EF'))]

    def audio_path(c):
        ep, num = c.split('/')
        for root in AUDIO_ROOTS:
            for ext in ('.mp3', '.wav', '.flac', '.m4a'):
                p = os.path.join(root, ep, num + ext)
                if os.path.exists(p):
                    return p
        return None

    V4 = {}
    for c in tqdm(sorted(set(DEV.clip4)), desc='clip IV voice (analysis only)'):
        p = audio_path(c)
        if p is None:
            continue
        try:
            wav, _ = librosa.load(p, sr=16000, mono=True)
        except Exception:
            continue
        if len(wav) < 8000:
            continue
        with torch.no_grad():
            e = spk.encode_batch(torch.tensor(wav, dtype=torch.float32).unsqueeze(0)).reshape(-1).cpu().numpy()
        V4[c] = e / (np.linalg.norm(e) + 1e-9)
    del spk; torch.cuda.empty_cache()

    def gold(c):
        e = ann.at[c, 7] if c in ann.index else None
        return E2I.get(e, -1) if isinstance(e, str) else -1

    rows_ = []
    for n, row in enumerate(DEV.itertuples()):
        if row.clip4 not in V4:
            continue
        for k, (name, c) in enumerate((('I', row.clip1), ('II', row.clip2))):
            a, y = G8[c]['audio'], gold(c)
            if a is None or a.get('ecapa') is None or y < 0:
                continue
            isB = float(a['ecapa'].astype(np.float32) @ V4[row.clip4].astype(np.float32)) >= VOICE_SAME_COS
            rows_.append({'ep': row.source_folder, 'ctx': name, 'B_spoke': bool(isB), 'same': bool(y == row.yB),
                          'A_spoke': bool(VOI[n, k, 6] >= VOICE_SAME_COS)})
    pdf = pd.DataFrame(rows_, columns=['ep', 'ctx', 'B_spoke', 'same', 'A_spoke'])
    print(f"(3) clip IV voice found for {len(V4)}/{DEV.clip4.nunique()} clips; labelled context clips analysed: {len(pdf)}")
    rng_ = np.random.default_rng(0)
    for name in ('I', 'II'):
        d = pdf[pdf['ctx'] == name]
        if d.B_spoke.sum() < 10 or (~d.B_spoke).sum() < 10:
            print(f"  clip {name}: too few rows"); continue
        eps_ = d.ep.unique()
        diffs = []
        for _ in range(2000):
            s = pd.concat([d[d.ep == e] for e in rng_.choice(eps_, len(eps_))])
            if s.B_spoke.any() and (~s.B_spoke).any():
                diffs.append(s[s.B_spoke].same.mean() - s[~s.B_spoke].same.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5]) * 100
        third = d[~d.B_spoke & ~d.A_spoke]
        print(f"  clip {name}: P(y_IV = y_{name}) when B spoke {d[d.B_spoke].same.mean() * 100:.1f}% (n={d.B_spoke.sum()}) | "
              f"when someone else spoke {d[~d.B_spoke].same.mean() * 100:.1f}% (n={(~d.B_spoke).sum()}; third person only "
              f"{third.same.mean() * 100 if len(third) else float('nan'):.1f}%, n={len(third)}) | difference CI [{lo:+.1f}, {hi:+.1f}]")
    print("  -> a clearly positive difference supports a separate B-inertia path; ~0 supports one merged persistence path")
"""),
    ("markdown", r"""
## Models
"""),
    ("code", r"""
T = lambda a, dt=None: torch.tensor(a, device=DEVICE) if dt is None else torch.tensor(a, dtype=dt, device=DEVICE)
FMASK, PMASK, VOI = T(FMASK), T(PMASK), T(VOI)
FACE = POOL = LRFZ = None             # set per fold
CLIPIDX = T([[CIDX[c] for c in r] for r in DEV[['clip1', 'clip2', 'clip3']].values])
TXT, AUD, AFD = FEAT['text'][CLIPIDX], F.normalize(FEAT['audio'][CLIPIDX], dim=-1), FEAT['afound'][CLIPIDX].float()
fm = FEAT['fmask'][CLIPIDX].unsqueeze(-1).float()
SCN = torch.cat([FEAT['ori'][CLIPIDX].mean(2), (FEAT['face'][CLIPIDX] * fm).sum(2) / fm.sum(2).clamp(min=1)], -1)
ZREC = torch.zeros(N, 3, N_REC, device=DEVICE)
YB, YA = T(DEV.yB.values), T(DEV.yA.values)


class FramePool(nn.Module):
    def __init__(self, fin, d):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Linear(d, 1)

    def forward(self, x, m):                      # x [..., F, fin], m [..., F]
        h = self.proj(x.float())
        a = self.score(h).squeeze(-1).masked_fill(~m, -1e4)
        w = torch.softmax(a, -1) * m.float()
        return (w.unsqueeze(-1) * h).sum(-2), m.any(-1)


class RoleNet(nn.Module):
    def __init__(self, cfg, d=RN['D']):
        super().__init__()
        self.cfg, self.R = cfg, (3 if cfg['role'] else 1)
        if cfg['faces']:
            self.pool = FramePool(FDIM, d)
            self.absent = nn.Parameter(torch.randn(self.R, 3, d) * 0.02)
            self.face_role = nn.Parameter(torch.randn(self.R, d) * 0.02)
        if cfg['ctx']:
            self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
            self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
            self.voice = nn.Linear(NVOICE, d)
            self.scene = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
            self.ctx_role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.clip_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, RN['heads'], 4 * d, RN['dropout'], batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, RN['layers'], enable_nested_tensor=False)
        mk = lambda: nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))
        self.head = mk()
        self.head_face = mk() if cfg['aux'] and cfg['faces'] else None
        self.head_ctx = mk() if cfg['aux'] and cfg['ctx'] else None
        self.head_A = mk() if cfg['aux'] and cfg['faces'] and cfg['role'] else None

    def forward(self, ix, train=False):
        B, groups, aux = len(ix), [], {}
        if self.cfg['faces']:
            x, m = (FACE[ix], FMASK[ix]) if self.R == 3 else (POOL[ix], PMASK[ix])
            h, present = self.pool(x, m)                                      # [B, R, 3, d]
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            h = h + self.face_role[None, :, None] + self.clip_emb[None, None]
            ft = h.reshape(B, self.R * 3, -1)
            groups.append(ft)
            if self.head_face is not None:
                aux['face'] = (self.head_face(ft.mean(1)), YB[ix], RN['aux_w'])
            if self.head_A is not None:
                tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
                aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        if self.cfg['ctx']:
            spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.ctx_role[0]
            scn = self.scene(SCN[ix]) + self.ctx_role[1]
            ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)     # [B, 6, d]
            groups.append(ct)
            if self.head_ctx is not None:
                aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1)] + groups, 1)
        valid = torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)
        if train and self.cfg['mdrop'] and len(groups) == 2:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            nf = groups[0].shape[1]
            valid[:, 1:1 + nf] &= ~drop_face.unsqueeze(1)
            valid[:, 1 + nf:] &= ~drop_ctx.unsqueeze(1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


class B1Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = Forecaster(use_raw=True, use_traj=False)

    def forward(self, ix, train=False):
        return self.f(CLIPIDX[ix], ZREC[ix]), {}


class B1FaceWrap(nn.Module):
    # B1 + the 60-d G6b-style role-face vector through a zero-initialised branch, trained jointly (the G7b design)
    def __init__(self, n_face=60, d=512):
        super().__init__()
        self.f = Forecaster(use_raw=True, use_traj=False, d=d)
        self.face = nn.Sequential(nn.Linear(n_face, d // 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d // 2, d))
        nn.init.zeros_(self.face[-1].weight); nn.init.zeros_(self.face[-1].bias)

    def forward(self, ix, train=False):
        b, clip_idx = self.f, CLIPIDX[ix]
        B, n = clip_idx.shape
        feats = gather(clip_idx)
        tok = b.enc({k: v.reshape(B * n, *v.shape[2:]) for k, v in feats.items()}).reshape(B, n, -1)
        h = b.inter(tok + b.clip_pos[:, 3 - n:])
        return b.head(h.mean(1) + self.face(LRFZ[ix])), {}


HP = {'b1': dict(lr=LR, wd=WEIGHT_DECAY, epochs=FC_EPOCHS, patience=PATIENCE, batch=FC_BATCH),
      'role': dict(lr=RN['lr'], wd=RN['wd'], epochs=RN['epochs'], patience=RN['patience'], batch=RN['batch'])}
HP['b1face'] = HP['b1']
MAKE = {'b1': lambda cfg: B1Wrap(), 'b1face': lambda cfg: B1FaceWrap(), 'role': lambda cfg: RoleNet(cfg)}
print("parameters:", {n: f"{sum(p.numel() for p in MAKE[k](c).parameters()) / 1e6:.2f}M" for n, k, c in EXPERIMENTS})


def predict(model, ix, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            out.append(F.softmax(model(ix[i:i + bs])[0], -1).cpu())
    return torch.cat(out).numpy()


def train_eval(kind, cfg, tr, dev, te, seed):
    seed_all(seed)
    hp = HP[kind]
    model = MAKE[kind](cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    y_dev = YB[dev].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(hp['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), hp['batch']):
            j = perm[i:i + hp['batch']]
            logits, aux = model(j, train=True)
            loss = F.cross_entropy(logits, YB[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        u = war_uar(predict(model, dev).argmax(1), y_dev, 7)[1]
        if u > best:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    model.load_state_dict(best_state)
    return predict(model, te), best
"""),
    ("markdown", r"""
## 5-fold episode cross-validation (45 train+val episodes)
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)

y_all = DEV.yB.values
src = DEV.source_folder.values
OOF = {name: np.full((len(SEEDS), N, 7), np.nan, np.float32) for name, _, _ in EXPERIMENTS}
OOF_LR = {'unbalanced': np.full((N, 7), np.nan, np.float32), 'balanced': np.full((N, 7), np.nan, np.float32)}
LOGPI = np.zeros((N, 7), np.float32)       # log class prior of each row's training fold
log = []
t0 = time.time()


def face_lr(Xtr, ytr, gtr, Xte, cw):
    best = None
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        s = [war_uar(LogisticRegression(max_iter=3000, C=C, class_weight=cw).fit(Xtr[a], ytr[a]).predict(Xtr[b]),
                     ytr[b], 7)[1] for a, b in GroupKFold(5).split(Xtr, ytr, gtr)]
        if best is None or np.mean(s) > best[0]:
            best = (np.mean(s), C)
    clf = LogisticRegression(max_iter=3000, C=best[1], class_weight=cw).fit(Xtr, ytr)
    pr = np.full((len(Xte), 7), 1e-6, np.float32)
    pr[:, clf.classes_] = clf.predict_proba(Xte)      # a class absent from training keeps ~0 probability
    return pr / pr.sum(1, keepdims=True)


for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    mu, sd = LRF[trr].mean(0), LRF[trr].std(0) + 1e-6
    LRFZ = T(((LRF - mu) / sd).astype(np.float32))
    LOGPI[te_rows] = np.log((np.bincount(y_all[trr], minlength=7) + 1) / (len(trr) + 7))
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)} | PCA variance kept "
          f"{var if var is None else round(var, 3)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for name, kind, cfg in EXPERIMENTS:
        for si, seed in enumerate(SEEDS):
            p, sel = train_eval(kind, cfg, tr, dev, te, seed + 1000 * f)
            OOF[name][si, te_rows] = p
            w, u = war_uar(p.argmax(1), y_all[te_rows], 7)
            log.append({'fold': f, 'exp': name, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w})
            print(f"fold {f} {name:<18} seed {seed}: sel {sel:5.2f} | UAR {u:5.2f} WAR {w:5.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()
    sc = StandardScaler().fit(LRF[trr])
    for key, cw in (('unbalanced', None), ('balanced', 'balanced')):
        OOF_LR[key][te_rows] = face_lr(sc.transform(LRF[trr]), y_all[trr], src[trr], sc.transform(LRF[te_rows]), cw)

assert all(not np.isnan(v).any() for v in list(OOF.values()) + list(OOF_LR.values()))
fuse = lambda pb, pf: np.exp((1 - LATE_W) * np.log(pb + 1e-9) + LATE_W * np.log(pf + 1e-9))
OOF['LateFusion'] = np.stack([fuse(OOF['B1'][s], OOF_LR['unbalanced']) for s in range(len(SEEDS))])
OOF['LateFusion-balancedLR'] = np.stack([fuse(OOF['B1'][s], OOF_LR['balanced']) for s in range(len(SEEDS))])
OOF['FaceLR'] = OOF_LR['unbalanced'][None]
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g8b_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g8b_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, logpi=LOGPI,
         **{k.replace('-', '_').replace('+', '_'): v for k, v in OOF.items()})
"""),
    ("markdown", r"""
## Results: pooled out-of-fold scores (plain and logit-adjusted), paired contrasts, subsets
"""),
    ("code", r"""
def boot_delta(pa, pb, y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(s == e)[0] for e in np.unique(s)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7); wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


def recalls(p, y):
    return np.array([(p[y == c] == c).mean() * 100 if (y == c).any() else np.nan for c in range(7)])


LOGP = {k: np.log(v.mean(0) + 1e-9) for k, v in OOF.items()}
PRED = {'plain': {k: v.argmax(1) for k, v in LOGP.items()},
        'LA': {k: (v - LA_TAU * LOGPI).argmax(1) for k, v in LOGP.items()}}

print(f"== per-seed pooled out-of-fold UAR, plain ({N} MCIS, {len(EPS)} episodes) ==")
for k, v in OOF.items():
    per = [war_uar(v[s].argmax(1), y_all, 7)[1] for s in range(len(v))]
    print(f"  {k:<22} " + " ".join(f"{u:5.2f}" for u in per) + f"  (mean {np.mean(per):.2f})")
for mode in ('LA', 'plain'):
    print(f"\n== seed ensemble, {mode} (95% bootstrap over episodes) ==")
    for k in PRED[mode]:
        report(k, PRED[mode][k], y_all, src)

print("\n== per-class recall, LA, seed ensemble (6-class = without fear) ==")
print(f"  {'':<22}" + "".join(f"{e[:7]:>8}" for e in EMO) + "   6-class")
for k, p in PRED['LA'].items():
    r = recalls(p, y_all)
    print(f"  {k:<22}" + "".join(f"{x:8.1f}" for x in r) + f"   {np.nanmean(np.delete(r, E2I['fear'])):6.2f}")

CONTRASTS = [("RoleNet", "B1"), ("RoleNet", "LateFusion"), ("LateFusion", "B1"), ("B1+faces-joint", "B1"),
             ("RoleNet", "RoleNet-noRole"), ("RoleNet", "RoleNet-noBalance"), ("RoleNet-facesOnly", "RoleNet-ctxOnly")]
SUBSETS = {'all': np.ones(N, bool), 'listener visible in III': VIS, 'listener not visible': ~VIS}
for mode in ('LA', 'plain'):
    for sname, m in SUBSETS.items():
        if mode == 'plain' and sname != 'all':
            continue
        print(f"\n== paired contrasts, {mode}, {sname} (n={m.sum()}) ==")
        if m.sum() < 30:
            print("  too few MCIS, skipped")
            continue
        for a, b in CONTRASTS:
            pa, pb = PRED[mode][a][m], PRED[mode][b][m]
            lo, hi = boot_delta(pa, pb, y_all[m], src[m])
            wa, ua = war_uar(pa, y_all[m], 7); wb, ub = war_uar(pb, y_all[m], 7)
            print(f"  {a:<18} - {b:<18} ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}]  "
                  f"ΔWAR {wa - wb:+5.2f} [{lo[1]:+5.2f},{hi[1]:+5.2f}]")

print("\n== per-fold ΔUAR, RoleNet − B1 (LA, seed ensemble) ==")
for f in range(N_OUTER):
    m = fold_of_row == f
    print(f"  fold {f}: {war_uar(PRED['LA']['RoleNet'][m], y_all[m], 7)[1] - war_uar(PRED['LA']['B1'][m], y_all[m], 7)[1]:+.2f}")

pa, pb = PRED['LA']['RoleNet'], PRED['LA']['B1']
lo, hi = boot_delta(pa, pb, y_all, src)
d = war_uar(pa, y_all, 7)[1] - war_uar(pb, y_all, 7)[1]
print(f"\nPRIMARY (LA): RoleNet − B1 ΔUAR {d:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}] -> "
      f"{'CONFIRMED' if lo[0] > 0 else ('DIRECTIONAL (CI includes 0)' if d > 0 else 'NOT SUPPORTED')}")
"""),
]

# ---------------------------------------------------------------- G9: RoleNet+ (speaker roles + B's previous utterance)
G9 = [
    ("markdown", r"""
# G9 — RoleNet+: who spoke in the context, and B's previous utterance (round 1 of at most 2)

G8b showed three things:
- RoleNet beats B1 by +4.83 UAR [+2.40, +7.13] (5/5 folds) and beats LateFusion by +3.15.
- The specific value of **role assignment** is not established: RoleNet − noRole is about +1 and not significant.
- The diagnostics found that B's *own* previous utterance predicts B's next emotion best. In clip II, the label matches
  50.6% of the time when B spoke there, 45.2% when a third person did, and about 35% when A did.

RoleNet+ therefore grounds the **speech** in roles, not only the faces. Everything else is identical to RoleNet
(same features, hyper-parameters, folds, and seeds).

1. **Speaker-role tags.** Each utterance in clips I/II gets soft weights for "spoken by A / B / someone else". The
   weights are added to its speech token as a mix of learned role embeddings; clip III is tagged "A".
   * *by A*: the ECAPA voice cosine to clip III.
   * *by B*: a **B-pointer**, a logistic regression over clip I–III cues only. The cues are voice vs A, which faces
     are present, the listener's frame share, mouth–audio sync per role, the number of identities, and voice
     continuity between clips I and II.
   * The pointer's **training labels** come from clip IV's voice. This is used at training time only: clip IV is never
     an input.
   * The pointer is cross-fitted. Eval rows are scored by a model fitted on the training episodes. Training rows are
     scored out-of-fold, grouped by episode, so train and eval inputs have the same quality.
2. **B's previous-utterance token.** This is the pointer-weighted average of the clip I/II speech tokens, with a
   learned "absent" token when no context utterance looks like B's. It has an auxiliary head (weight 0.3) that
   predicts the gold emotion of the context clip B spoke in. That target is also training-only (clip IV voice).

**Arms:**

| Arm | What it is |
|---|---|
| `RoleNet` | Reference: G8b's model, re-run |
| `RoleNet+spk` | Adds speaker-role tags (component 1) |
| `RoleNet+` | Adds speaker-role tags and B's previous-utterance token (components 1 and 2) |
| `RoleNet+ -faceRoles` | `RoleNet+` without face role assignment |
| `RoleNet+ ORACLE` | Analysis only: uses the clip-IV-derived "B spoke" labels at evaluation, so it measures the headroom of a perfect pointer. **Never a result.** |

**Decision rule, fixed before running.** Adopt `RoleNet+` over `RoleNet` for the single test run only if all three
hold:
- its seed-ensemble ΔUAR under LA is > 0;
- its 6-class macro-recall Δ (without fear) is > 0;
- the 6-class Δ is positive in ≥ 4 of 5 folds.

Otherwise keep `RoleNet`. At most one more round follows.

5-fold CV over the 45 train+val episodes, with the same folds and seeds as G8b. The test split stays untouched.
"""),
    G8B[1],
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
G8A_DIR = first_existing("/kaggle/input/datasets/ptrnghieu/g8a-features", "/kaggle/input/g8a-features")
OUT_DIR = "/kaggle/working"

N_OUTER, N_INNER_DEV = 5, 5
SEEDS = [42, 123, 456]
# RoleNet, identical to G8b (set a priori, not tuned)
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)
BPREV_W = 0.3                # auxiliary weight of the B-previous-utterance head
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
LA_TAU = 1.0
VOICE_SAME_COS = 0.35
DEBUG_PER_EPISODE = None

FULL = dict(role=True, faces=True, ctx=True, aux=True, mdrop=True, spk=False, bprev=False, oracle=False)
EXPERIMENTS = [
    ("RoleNet",             FULL),
    ("RoleNet+spk",         {**FULL, 'spk': True}),
    ("RoleNet+",            {**FULL, 'spk': True, 'bprev': True}),
    ("RoleNet+ -faceRoles", {**FULL, 'spk': True, 'bprev': True, 'role': False}),
    ("RoleNet+ ORACLE",     {**FULL, 'spk': True, 'bprev': True, 'oracle': True}),   # analysis only
]
"""),
    G8B[3], G8B[4], G8B[5], G8B[6], G8B[7],
    ("markdown", r"""
## Who spoke in clips I/II: training labels from clip IV voice, inference cues from clips I–III
"""),
    ("code", r"""
import librosa
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{OUT_DIR}/ecapa",
                                           run_opts={"device": DEVICE})
AUDIO_ROOTS = [os.path.join(r, 'audio') for r in glob.glob(os.path.join(DATASET_DIR, '*', 'Hi-EF'))]


def audio_path(c):
    ep, num = c.split('/')
    for root in AUDIO_ROOTS:
        for ext in ('.mp3', '.wav', '.flac', '.m4a'):
            p = os.path.join(root, ep, num + ext)
            if os.path.exists(p):
                return p
    return None


V4 = {}          # clip IV voice: used ONLY to build training labels / targets and the ORACLE analysis arm
for c in tqdm(sorted(set(DEV.clip4)), desc='clip IV voice (training labels only)'):
    p = audio_path(c)
    if p is None:
        continue
    try:
        wav, _ = librosa.load(p, sr=16000, mono=True)
    except Exception:
        continue
    if len(wav) < 8000:
        continue
    with torch.no_grad():
        e = spk_model.encode_batch(torch.tensor(wav, dtype=torch.float32).unsqueeze(0)).reshape(-1).cpu().numpy()
    V4[c] = (e / (np.linalg.norm(e) + 1e-9)).astype(np.float32)
del spk_model; torch.cuda.empty_cache()


def gold(c):
    e = ann.at[c, 7] if c in ann.index else None
    return E2I.get(e, -1) if isinstance(e, str) else -1


def ecapa(c):
    a = G8[c]['audio']
    return a['ecapa'].astype(np.float32) if a is not None and a.get('ecapa') is not None else None


NPF = 16
ISB = np.zeros((N, 2), np.float32); ISB_OK = np.zeros((N, 2), bool)
BTGT = np.full(N, -100, np.int64)
PFEAT = np.zeros((N, 2, NPF), np.float32)
for n, row in enumerate(DEV.itertuples()):
    cl = [row.clip1, row.clip2]
    e4 = V4.get(row.clip4)
    for k, c in enumerate(cl):
        ek, eo = ecapa(c), ecapa(cl[1 - k])
        if e4 is not None and ek is not None:
            ISB[n, k] = float(ek @ e4 >= VOICE_SAME_COS); ISB_OK[n, k] = True
        v = VOI[n, k]
        PFEAT[n, k] = [v[6], v[7], FMASK[n, 1, k].sum() / MAXF, FMASK[n, 0, k].sum() / MAXF, FMASK[n, 2, k].sum() / MAXF,
                       v[0], v[2], v[4], v[1], v[3], v[5], v[8], float(k), float(FMASK[n, 1, 2].any()),
                       float(ek @ eo) if ek is not None and eo is not None else 0.0, v[2] - max(v[0], v[4])]
    for k in (1, 0):                      # prefer clip II
        if ISB[n, k] and gold(cl[k]) >= 0:
            BTGT[n] = gold(cl[k]); break
PA = np.where(VOI[:, :2, 7] > 0, 1 / (1 + np.exp(-(VOI[:, :2, 6] - VOICE_SAME_COS) * 20)), 0.0).astype(np.float32)
print(f"B spoke in clip I {ISB[ISB_OK[:, 0], 0].mean() * 100:.1f}% | clip II {ISB[ISB_OK[:, 1], 1].mean() * 100:.1f}% "
      f"(clip IV voice found {len(V4)}/{DEV.clip4.nunique()}) | rows with a B-previous-utterance target {(BTGT >= 0).sum()}")
"""),
    ("markdown", r"""
## Models
"""),
    ("code", r"""
T = lambda a, dt=None: torch.tensor(a, device=DEVICE) if dt is None else torch.tensor(a, dtype=dt, device=DEVICE)
FMASK, PMASK, VOI = T(FMASK), T(PMASK), T(VOI)
FACE = POOL = PBT = None             # set per fold
CLIPIDX = T([[CIDX[c] for c in r] for r in DEV[['clip1', 'clip2', 'clip3']].values])
TXT, AUD, AFD = FEAT['text'][CLIPIDX], F.normalize(FEAT['audio'][CLIPIDX], dim=-1), FEAT['afound'][CLIPIDX].float()
fm = FEAT['fmask'][CLIPIDX].unsqueeze(-1).float()
SCN = torch.cat([FEAT['ori'][CLIPIDX].mean(2), (FEAT['face'][CLIPIDX] * fm).sum(2) / fm.sum(2).clamp(min=1)], -1)
YB, YA, BT = T(DEV.yB.values), T(DEV.yA.values), T(BTGT)
PAT, ISBT = T(PA), T(ISB)


class FramePool(nn.Module):
    def __init__(self, fin, d):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Linear(d, 1)

    def forward(self, x, m):
        h = self.proj(x.float())
        a = self.score(h).squeeze(-1).masked_fill(~m, -1e4)
        w = torch.softmax(a, -1) * m.float()
        return (w.unsqueeze(-1) * h).sum(-2), m.any(-1)


class RoleNetPlus(nn.Module):
    # RoleNet (G8b) + optional speaker-role tags on speech tokens and a B-previous-utterance token
    def __init__(self, cfg, d=RN['D']):
        super().__init__()
        self.cfg, self.R = cfg, (3 if cfg['role'] else 1)
        self.pool = FramePool(FDIM, d)
        self.absent = nn.Parameter(torch.randn(self.R, 3, d) * 0.02)
        self.face_role = nn.Parameter(torch.randn(self.R, d) * 0.02)
        self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
        self.voice = nn.Linear(NVOICE, d)
        self.scene = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.ctx_role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.clip_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if cfg['spk']:
            self.spk_role = nn.Parameter(torch.randn(3, d) * 0.02)            # spoken by A / B / other
        if cfg['bprev']:
            self.bprev_absent = nn.Parameter(torch.randn(1, d) * 0.02)
            self.bprev_emb = nn.Parameter(torch.randn(1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, RN['heads'], 4 * d, RN['dropout'], batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, RN['layers'], enable_nested_tensor=False)
        mk = lambda: nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))
        self.head, self.head_face, self.head_ctx = mk(), mk(), mk()
        self.head_A = mk() if cfg['role'] else None
        self.head_bprev = mk() if cfg['bprev'] else None

    def forward(self, ix, train=False):
        B, aux = len(ix), {}
        x, m = (FACE[ix], FMASK[ix]) if self.R == 3 else (POOL[ix], PMASK[ix])
        h, present = self.pool(x, m)
        h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
        h = h + self.face_role[None, :, None] + self.clip_emb[None, None]
        ft = h.reshape(B, self.R * 3, -1)
        aux['face'] = (self.head_face(ft.mean(1)), YB[ix], RN['aux_w'])
        if self.head_A is not None:
            tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
            aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.ctx_role[0]
        ctx = []
        if self.cfg['spk'] or self.cfg['bprev']:
            pA = PAT[ix]
            pB = (ISBT[ix] if self.cfg['oracle'] else PBT[ix]) * (1 - pA)
            pO = (1 - pA - pB).clamp(min=0)
        if self.cfg['spk']:
            w = torch.stack([pA, pB, pO], -1)                                     # [B, 2, 3]
            w3 = torch.cat([w, torch.tensor([1.0, 0, 0], device=w.device).expand(B, 1, 3)], 1)
            spk_ = spk_ + w3 @ self.spk_role
        scn = self.scene(SCN[ix]) + self.ctx_role[1]
        ctx += [spk_ + self.clip_emb, scn + self.clip_emb]
        if self.cfg['bprev']:
            s = pB.sum(1, keepdim=True)
            hb = (pB.unsqueeze(-1) * spk_[:, :2]).sum(1) / s.clamp(min=1e-3)
            hb = torch.where(s > 0.05, hb, self.bprev_absent.expand(B, -1)) + self.bprev_emb
            ctx.append(hb.unsqueeze(1))
            aux['bprev'] = (self.head_bprev(hb), BT[ix], BPREV_W)
        ct = torch.cat(ctx, 1)
        aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1), ft, ct], 1)
        valid = torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)
        if train and self.cfg['mdrop']:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            nf = ft.shape[1]
            valid[:, 1:1 + nf] &= ~drop_face.unsqueeze(1)
            valid[:, 1 + nf:] &= ~drop_ctx.unsqueeze(1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


print("parameters:", {n: f"{sum(p.numel() for p in RoleNetPlus(c).parameters()) / 1e6:.2f}M" for n, c in EXPERIMENTS})


def predict(model, ix, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            out.append(F.softmax(model(ix[i:i + bs])[0], -1).cpu())
    return torch.cat(out).numpy()


def train_eval(cfg, tr, dev, te, seed):
    seed_all(seed)
    model = RoleNetPlus(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=RN['lr'], weight_decay=RN['wd'])
    y_dev = YB[dev].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(RN['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), RN['batch']):
            j = perm[i:i + RN['batch']]
            logits, aux = model(j, train=True)
            loss = F.cross_entropy(logits, YB[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        u = war_uar(predict(model, dev).argmax(1), y_dev, 7)[1]
        if u > best:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= RN['patience']:
                break
    model.load_state_dict(best_state)
    return predict(model, te), best
"""),
    ("markdown", r"""
## 5-fold episode cross-validation (same folds and seeds as G8b)
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_, "(must match G8b)")

y_all = DEV.yB.values
src = DEV.source_folder.values
X2, Y2, OK2, G2 = PFEAT.reshape(N * 2, -1), ISB.reshape(-1), ISB_OK.reshape(-1), np.repeat(src, 2)


def pointer_probs(tr_rows, te_rows):
    # B-pointer: eval rows scored by a model fitted on training episodes; training rows scored out-of-fold
    rows2 = lambda r: np.sort(np.concatenate([r * 2, r * 2 + 1]))
    out = np.zeros(N * 2, np.float32)
    tr2 = rows2(tr_rows)
    fit2 = tr2[OK2[tr2]]
    sc = StandardScaler().fit(X2[fit2])
    lr_ = lambda idx: LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X2[idx]), Y2[idx])
    te2 = rows2(te_rows)
    out[te2] = lr_(fit2).predict_proba(sc.transform(X2[te2]))[:, 1]
    for a, b in GroupKFold(5).split(tr2, groups=G2[tr2]):
        fa = tr2[a][OK2[tr2[a]]]
        out[tr2[b]] = lr_(fa).predict_proba(sc.transform(X2[tr2[b]]))[:, 1]
    return out.reshape(N, 2)


OOF = {name: np.full((len(SEEDS), N, 7), np.nan, np.float32) for name, _ in EXPERIMENTS}
PB_OOF = np.zeros((N, 2), np.float32)
LOGPI = np.zeros((N, 7), np.float32)
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    FACE, POOL, var = build_face_tensors(sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel())))
    pb = pointer_probs(trr, te_rows)
    PB_OOF[te_rows] = pb[te_rows]
    PBT = T(pb)
    LOGPI[te_rows] = np.log((np.bincount(y_all[trr], minlength=7) + 1) / (len(trr) + 7))
    ok = ISB_OK[te_rows]
    auc = roc_auc_score(ISB[te_rows][ok], pb[te_rows][ok]) if len(set(ISB[te_rows][ok])) == 2 else float('nan')
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)} | B-pointer AUC {auc:.3f}",
          flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for name, cfg in EXPERIMENTS:
        for si, seed in enumerate(SEEDS):
            p, sel = train_eval(cfg, tr, dev, te, seed + 1000 * f)
            OOF[name][si, te_rows] = p
            w, u = war_uar(p.argmax(1), y_all[te_rows], 7)
            log.append({'fold': f, 'exp': name, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w})
            print(f"fold {f} {name:<20} seed {seed}: sel {sel:5.2f} | UAR {u:5.2f} WAR {w:5.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()

assert all(not np.isnan(v).any() for v in OOF.values())
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g9_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g9_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, logpi=LOGPI, pb=PB_OOF,
         isb=ISB, isb_ok=ISB_OK, **{k.replace('-', '_').replace('+', 'plus').replace(' ', '_'): v for k, v in OOF.items()})
"""),
    ("markdown", r"""
## Results and the preregistered decision
"""),
    ("code", r"""
FEAR = E2I['fear']


def recalls(p, y):
    return np.array([(p[y == c] == c).mean() * 100 if (y == c).any() else np.nan for c in range(7)])


def uar7(p, y):
    return np.nanmean(recalls(p, y))


def uar6(p, y):
    return np.nanmean(np.delete(recalls(p, y), FEAR))


rng_ = np.random.default_rng(0)
GRP = [np.where(src == e)[0] for e in np.unique(src)]
BOOT = [np.concatenate([GRP[j] for j in rng_.integers(0, len(GRP), len(GRP))]) for _ in range(2000)]


def delta(pa, pb, m, metric):
    idx_m = np.where(m)[0]
    d0 = metric(pa[m], y_all[m]) - metric(pb[m], y_all[m])
    ds = []
    for b in BOOT:
        i = b[m[b]]
        ds.append(metric(pa[i], y_all[i]) - metric(pb[i], y_all[i]))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return d0, lo, hi


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) == 2 else float('nan')


ok = ISB_OK.reshape(-1)
print(f"B-pointer AUC (out-of-fold): all {safe_auc(ISB.reshape(-1)[ok], PB_OOF.reshape(-1)[ok]):.3f} | "
      f"clip I {safe_auc(ISB[ISB_OK[:, 0], 0], PB_OOF[ISB_OK[:, 0], 0]):.3f} | "
      f"clip II {safe_auc(ISB[ISB_OK[:, 1], 1], PB_OOF[ISB_OK[:, 1], 1]):.3f}")

LOGP = {k: np.log(v.mean(0) + 1e-9) for k, v in OOF.items()}
PRED = {'plain': {k: v.argmax(1) for k, v in LOGP.items()}, 'LA': {k: (v - LA_TAU * LOGPI).argmax(1) for k, v in LOGP.items()}}
print(f"\n== per-seed pooled UAR, plain ==")
for k, v in OOF.items():
    per = [uar7(v[s].argmax(1), y_all) for s in range(len(v))]
    print(f"  {k:<20} " + " ".join(f"{u:5.2f}" for u in per) + f"  (mean {np.mean(per):.2f})")
print("\n== seed ensemble: UAR LA | 6-class LA | UAR plain | WAR LA ==")
for k in PRED['LA']:
    print(f"  {k:<20} {uar7(PRED['LA'][k], y_all):6.2f} | {uar6(PRED['LA'][k], y_all):6.2f} | "
          f"{uar7(PRED['plain'][k], y_all):6.2f} | {(PRED['LA'][k] == y_all).mean() * 100:6.2f}")
print("\n== per-class recall, LA ==")
print(f"  {'':<20}" + "".join(f"{e[:7]:>8}" for e in EMO))
for k, p in PRED['LA'].items():
    print(f"  {k:<20}" + "".join(f"{x:8.1f}" for x in recalls(p, y_all)))

CONTRASTS = [("RoleNet+", "RoleNet"), ("RoleNet+spk", "RoleNet"), ("RoleNet+", "RoleNet+spk"),
             ("RoleNet+", "RoleNet+ -faceRoles"), ("RoleNet+ ORACLE", "RoleNet+")]
bspoke = (ISB * ISB_OK).max(1) > 0
SUBSETS = {'all': np.ones(N, bool), 'listener visible in III': VIS.copy() if isinstance(VIS, np.ndarray) else VIS,
           'B spoke in I/II (analysis)': bspoke, 'B did not speak in I/II (analysis)': ~bspoke}
for sname, m in SUBSETS.items():
    m = np.asarray(m, bool)
    print(f"\n== paired contrasts, {sname} (n={m.sum()}) — LA 7-class | LA 6-class | plain 7-class ==")
    if m.sum() < 30:
        print("  too few MCIS, skipped")
        continue
    for a, b in CONTRASTS:
        r = [delta(PRED[mode][a], PRED[mode][b], m, met) for mode, met in (('LA', uar7), ('LA', uar6), ('plain', uar7))]
        print(f"  {a:<20} - {b:<20} " + " | ".join(f"{d:+5.2f} [{lo:+5.2f},{hi:+5.2f}]" for d, lo, hi in r))

print("\n== per-fold 6-class Δ (LA), RoleNet+ − RoleNet ==")
fold_d = []
for f in range(N_OUTER):
    m = fold_of_row == f
    fold_d.append(uar6(PRED['LA']['RoleNet+'][m], y_all[m]) - uar6(PRED['LA']['RoleNet'][m], y_all[m]))
    print(f"  fold {f}: {fold_d[-1]:+.2f}")
d7 = uar7(PRED['LA']['RoleNet+'], y_all) - uar7(PRED['LA']['RoleNet'], y_all)
d6 = uar6(PRED['LA']['RoleNet+'], y_all) - uar6(PRED['LA']['RoleNet'], y_all)
adopt = d7 > 0 and d6 > 0 and sum(x > 0 for x in fold_d) >= 4
print(f"\nDECISION (preregistered): ΔUAR LA {d7:+.2f}, Δ6-class {d6:+.2f}, positive folds {sum(x > 0 for x in fold_d)}/5 -> "
      f"{'ADOPT RoleNet+ for the test run' if adopt else 'KEEP RoleNet'}")
"""),
]

# ---------------------------------------------------------------- G10: the single preregistered test run
G10 = [
    ("markdown", r"""
# G10 — The single preregistered test evaluation

The plan is fixed in `experiments/rtt/PREREG_G10_TEST.md`, which was committed before this notebook was run. In short:

* **Training data.** All arms train on the 45 train+val episodes. Five of those episodes, drawn with
  `random.Random(2026)`, are held out for early stopping and checkpoint selection.
* **Evaluation.** Once, on the 409 test MCIS from 8 episodes.
* **Arms:** `PaperBest` (the Hi-EF paper's best configuration, replicated), `B1`, `LateFusion`, `RoleNet`,
  `RoleNet-noRole`. Each neural arm runs 5 seeds, and the seed ensemble averages probabilities.
* **Confirmatory contrasts**, in a fixed sequence, measured as ΔUAR under a single post-hoc logit adjustment (LA) with a
  95% bootstrap over the test episodes:
  1. RoleNet − PaperBest (primary)
  2. RoleNet − B1
  3. RoleNet − LateFusion
  4. RoleNet − RoleNet-noRole

Run it **once**. Set `UNLOCK_TEST = True` in the CONFIG cell.
"""),
    G8B[1],
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
G8A_DIR = first_existing("/kaggle/input/datasets/ptrnghieu/g8a-features", "/kaggle/input/g8a-features")
OUT_DIR = "/kaggle/working"
UNLOCK_TEST = False          # set to True for the single preregistered run

N_INNER_DEV, SELECT_SEED = 5, 2026
SEEDS = [42, 123, 456, 789, 1024]
LR, WEIGHT_DECAY = 1e-4, 1e-5
FC_EPOCHS, PATIENCE, FC_BATCH = 50, 8, 32          # B1, as G3b / G8b
PAPER_EPOCHS, PAPER_BATCH = 50, 32                 # PaperBest, as the replication notebook
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
LATE_W, LA_TAU = 0.5, 1.0
VOICE_SAME_COS = 0.35
DEBUG_PER_EPISODE = None

FULL = dict(role=True, faces=True, ctx=True, aux=True, mdrop=True)
EXPERIMENTS = [
    ("B1",             'b1',   None),
    ("RoleNet",        'role', FULL),
    ("RoleNet-noRole", 'role', {**FULL, 'role': False}),
]
"""),
    ("code", COMMON + r"""
import glob, pickle
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm

if not UNLOCK_TEST:
    raise RuntimeError("Test split is locked. Set UNLOCK_TEST = True only for the single preregistered run.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOT_CSV = glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF", "annotation.csv"))[0]
ann, sp = load_tables(ANNOT_CSV, SPLIT_CSV)
if DEBUG_PER_EPISODE:
    sp = sp.groupby('source_folder', group_keys=False).head(DEBUG_PER_EPISODE)
DEV = sp.reset_index(drop=True)            # all rows: train+val are fitted on, test is evaluated once
train_all = ev = DEV
N = len(DEV)
IS_TEST = (DEV.split == 'test').values
src = DEV.source_folder.values
EPS = np.array(sorted(set(src[~IS_TEST])))
TEST_EPS = sorted(set(src[IS_TEST]))
print(f"training MCIS {(~IS_TEST).sum()} from {len(EPS)} episodes | test MCIS {IS_TEST.sum()} from {len(TEST_EPS)} episodes")
if not DEBUG_PER_EPISODE:
    assert len(EPS) == 45 and len(TEST_EPS) == 8 and IS_TEST.sum() == 409


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
"""),
    G3[3], G3[4], G8B[6], G8B[7], G8B[11], G8B[12],
    ("markdown", r"""
## PaperBest: the Hi-EF paper's best configuration (verbatim from the replication notebook)
"""),
    ("code", r"""
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


class IntraVideoFusion(nn.Module):
    def __init__(self, d_model=512, audio_dim=527):
        super().__init__()
        self.face_temporal = TemporalTransformer(d_model, n_heads=8, n_layers=2)
        self.ori_temporal = TemporalTransformer(d_model, n_heads=8, n_layers=2)
        self.type_fusion = CrossAttentionFusion(d_model, n_heads=8, n_layers=1)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.modality_fusion = CrossAttentionFusion(d_model, n_heads=8, n_layers=1)

    def forward(self, face_features, ori_features, text_feature, audio_feature):
        face_out = self.face_temporal(face_features).mean(dim=1, keepdim=True)
        ori_out = self.ori_temporal(ori_features).mean(dim=1, keepdim=True)
        video_feat = self.type_fusion(face_out, torch.cat([face_out, ori_out], dim=1))
        audio_feat = self.audio_proj(F.normalize(audio_feature, dim=-1)).unsqueeze(1)
        text_feat = text_feature.unsqueeze(1)
        clip_feat = self.modality_fusion(video_feat, torch.cat([video_feat, text_feat, audio_feat], dim=1))
        return clip_feat.squeeze(1)


class InterVideoFusion(nn.Module):
    def __init__(self, d_model=512, lstm_layers=3, transformer_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=lstm_layers, batch_first=False,
                            dropout=0.1)
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)

    def forward(self, c1, c2, c3):
        lstm_out, _ = self.lstm(torch.stack([c1, c2, c3], dim=0))
        return self.transformer(lstm_out.permute(1, 0, 2) + self.pos_encoding).mean(dim=1)


class PaperBest(nn.Module):
    def __init__(self, d_model=512, n_classes=7):
        super().__init__()
        self.intra_fusion = IntraVideoFusion(d_model)
        self.inter_fusion = InterVideoFusion(d_model)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.3), nn.Linear(d_model, d_model // 2),
                                        nn.GELU(), nn.Dropout(0.2), nn.Linear(d_model // 2, n_classes))

    def forward(self, ix, train=False):
        feats = gather(CLIPIDX[ix])
        clips = [self.intra_fusion(feats['face'][:, k], feats['ori'][:, k], feats['text'][:, k], feats['audio'][:, k])
                 for k in range(3)]
        return self.classifier(self.inter_fusion(*clips)), {}


print(f"PaperBest parameters: {sum(p.numel() for p in PaperBest().parameters()):,}")


def train_paper(tr, dev, te, seed):
    seed_all(seed)
    model = PaperBest().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    y_dev = YB[dev]
    best, best_state = -1, None
    for ep in range(PAPER_EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), PAPER_BATCH):
            j = perm[i:i + PAPER_BATCH]
            loss = F.cross_entropy(model(j)[0], YB[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        p = predict(model, dev)
        sched.step(F.nll_loss(torch.log(torch.tensor(p) + 1e-9), y_dev.cpu()).item())
        u = war_uar(p.argmax(1), y_dev.cpu().numpy(), 7)[1]
        if u > best:
            best = u
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model, te), best
"""),
    ("markdown", r"""
## Train on the 45 train+val episodes, evaluate once on test
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sel_eps = sorted(random.Random(SELECT_SEED).sample(list(EPS), N_INNER_DEV))
trr = np.where(~IS_TEST)[0]
fit_rows = np.where(~IS_TEST & ~np.isin(src, sel_eps))[0]
dev_rows = np.where(np.isin(src, sel_eps))[0]
te_rows = np.where(IS_TEST)[0]
assert not set(src[te_rows]) & set(src[trr])
print(f"fit {len(fit_rows)} | selection {len(dev_rows)} (episodes {sel_eps}) | test {len(te_rows)}")
y_all = DEV.yB.values
FACE, POOL, var = build_face_tensors(sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel())))
mu, sd = LRF[trr].mean(0), LRF[trr].std(0) + 1e-6
LRFZ = T(((LRF - mu) / sd).astype(np.float32))
LOGPI_TR = np.log((np.bincount(y_all[trr], minlength=7) + 1) / (len(trr) + 7))
tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)

PROBS, log = {}, []
t0 = time.time()
for name, kind, cfg in EXPERIMENTS + [("PaperBest", 'paper', None)]:
    PROBS[name] = []
    for seed in SEEDS:
        p, sel = train_paper(tr, dev, te, seed) if kind == 'paper' else train_eval(kind, cfg, tr, dev, te, seed)
        PROBS[name].append(p)
        w, u = war_uar(p.argmax(1), y_all[te_rows], 7)
        log.append({'exp': name, 'seed': seed, 'sel_UAR': sel, 'test_UAR': u, 'test_WAR': w})
        print(f"{name:<15} seed {seed}: sel {sel:5.2f} | test UAR {u:5.2f} WAR {w:5.2f} | {(time.time() - t0) / 60:.1f} min",
              flush=True)
        torch.cuda.empty_cache()
    PROBS[name] = np.stack(PROBS[name])

sc = StandardScaler().fit(LRF[trr])
Xtr, Xte = sc.transform(LRF[trr]), sc.transform(LRF[te_rows])
best = None
for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1]:
    s = [war_uar(LogisticRegression(max_iter=3000, C=C).fit(Xtr[a], y_all[trr][a]).predict(Xtr[b]), y_all[trr][b], 7)[1]
         for a, b in GroupKFold(5).split(Xtr, y_all[trr], src[trr])]
    if best is None or np.mean(s) > best[0]:
        best = (np.mean(s), C)
clf = LogisticRegression(max_iter=3000, C=best[1]).fit(Xtr, y_all[trr])
pf = np.full((len(te_rows), 7), 1e-6, np.float32); pf[:, clf.classes_] = clf.predict_proba(Xte); pf /= pf.sum(1, keepdims=True)
PROBS['LateFusion'] = np.stack([np.exp((1 - LATE_W) * np.log(PROBS['B1'][s] + 1e-9) + LATE_W * np.log(pf + 1e-9))
                                for s in range(len(SEEDS))])
PROBS['FaceLR'] = pf[None]
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g10_test_per_seed.csv", index=False)
np.savez(f"{OUT_DIR}/g10_test_probs.npz", sample_id=DEV.sample_id.values[te_rows], logpi=LOGPI_TR,
         **{k.replace('-', '_'): v for k, v in PROBS.items()})
"""),
    ("markdown", r"""
## Results: preregistered fixed-sequence contrasts, then descriptive analyses
"""),
    ("code", r"""
yt, st = y_all[te_rows], src[te_rows]
vis_t = VIS[te_rows]
FEAR = E2I['fear']


def recalls(p, y):
    return np.array([(p[y == c] == c).mean() * 100 if (y == c).any() else np.nan for c in range(7)])


def uar7(p, y):
    return np.nanmean(recalls(p, y))


def uar6(p, y):
    return np.nanmean(np.delete(recalls(p, y), FEAR))


LOGP = {k: np.log(v.mean(0) + 1e-9) for k, v in PROBS.items()}
PRED = {'LA': {k: (v - LA_TAU * LOGPI_TR).argmax(1) for k, v in LOGP.items()}, 'plain': {k: v.argmax(1) for k, v in LOGP.items()}}
rng = np.random.default_rng(0)
G = [np.where(st == e)[0] for e in np.unique(st)]
BOOT = [np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))]) for _ in range(2000)]


def contrast(a, b, mode='LA', metric=uar7, m=None):
    m = np.ones(len(yt), bool) if m is None else m
    pa, pb = PRED[mode][a], PRED[mode][b]
    d0 = metric(pa[m], yt[m]) - metric(pb[m], yt[m])
    ds = [metric(pa[i[m[i]]], yt[i[m[i]]]) - metric(pb[i[m[i]]], yt[i[m[i]]]) for i in BOOT]
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return d0, lo, hi


print(f"test MCIS {len(yt)} | episodes {len(G)} | class counts {dict(zip(EMO, np.bincount(yt, minlength=7)))}")
print("\n== per-seed test UAR (plain) ==")
print(pd.DataFrame(log).pivot(index='seed', columns='exp', values='test_UAR').round(2).to_string())
print("\n== seed ensemble (LA | plain), with 95% bootstrap over test episodes for LA UAR ==")
for k in PRED['LA']:
    ds = [uar7(PRED['LA'][k][i], yt[i]) for i in BOOT]
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    print(f"  {k:<15} UAR LA {uar7(PRED['LA'][k], yt):5.2f} [{lo:5.1f},{hi:5.1f}]  WAR LA {(PRED['LA'][k] == yt).mean() * 100:5.2f} | "
          f"UAR plain {uar7(PRED['plain'][k], yt):5.2f}  WAR plain {(PRED['plain'][k] == yt).mean() * 100:5.2f} | "
          f"6-class LA {uar6(PRED['LA'][k], yt):5.2f}")
print("  (paper, different split, not comparable: UAR 23.72, WAR 35.19)")

print("\n== PREREGISTERED fixed-sequence contrasts (ΔUAR, LA, 7-class, seed ensemble) ==")
alive = True
for i, (a, b) in enumerate([("RoleNet", "PaperBest"), ("RoleNet", "B1"), ("RoleNet", "LateFusion"),
                            ("RoleNet", "RoleNet-noRole")], 1):
    d, lo, hi = contrast(a, b)
    ok = lo > 0
    status = ('CONFIRMED' if ok else 'NOT CONFIRMED') if alive else 'descriptive only (sequence stopped)'
    print(f"  {i}. {a} − {b}: ΔUAR {d:+.2f} [{lo:+.2f},{hi:+.2f}] -> {status}")
    alive = alive and ok

print("\n== descriptive ==")
for a, b in [("RoleNet", "PaperBest"), ("RoleNet", "B1"), ("RoleNet", "LateFusion"), ("RoleNet", "RoleNet-noRole"),
             ("PaperBest", "B1")]:
    r = [contrast(a, b, 'LA', uar6), contrast(a, b, 'plain', uar7),
         contrast(a, b, 'LA', uar7, vis_t), contrast(a, b, 'LA', uar7, ~vis_t)]
    print(f"  {a:<9}− {b:<15} 6-class LA {r[0][0]:+5.2f} [{r[0][1]:+.1f},{r[0][2]:+.1f}] | plain {r[1][0]:+5.2f} "
          f"[{r[1][1]:+.1f},{r[1][2]:+.1f}] | listener visible {r[2][0]:+5.2f} (n={vis_t.sum()}) | not visible {r[3][0]:+5.2f} (n={(~vis_t).sum()})")
print("\n== per-class recall (LA) ==")
print(f"  {'':<15}" + "".join(f"{e[:7]:>8}" for e in EMO))
for k, p in PRED['LA'].items():
    print(f"  {k:<15}" + "".join(f"{x:8.1f}" for x in recalls(p, yt)))
print("\n== per-episode UAR (LA): RoleNet vs PaperBest vs B1 ==")
wins = 0
for e in np.unique(st):
    m = st == e
    r, pb_, b1 = uar7(PRED['LA']['RoleNet'][m], yt[m]), uar7(PRED['LA']['PaperBest'][m], yt[m]), uar7(PRED['LA']['B1'][m], yt[m])
    wins += r > pb_
    print(f"  episode {e} (n={m.sum()}): RoleNet {r:5.2f} | PaperBest {pb_:5.2f} | B1 {b1:5.2f}")
print(f"  RoleNet beats PaperBest in {wins}/{len(np.unique(st))} episodes")
"""),
]

# ---------------------------------------------------------------- G11: channel ablations (simulation / surrogation / target observation)
G11 = [
    ("markdown", r"""
# G11 — Which evidence does RoleNet use? Channel ablations in episode-level CV

**Motivation.** Gilbert et al. (Science 2009) distinguish two ways to forecast an emotional reaction:
* **simulation** — reason from the *event*;
* **surrogation** — use the observed reaction of *another person* who went through the same event.

In Hi-EF the forecast target is B, so RoleNet's face tokens mix three kinds of evidence:

| Channel | RoleNet tokens | Theory |
|---|---|---|
| **Simulation** (the event) | A's face tokens + speech and scene tokens | simulation |
| **Surrogation, strict sense** (other people's reactions) | **O** face tokens (bystanders) | surrogation |
| **Target observation** (B's own current / earlier reaction) | **L** face tokens | not surrogation in the strict sense; closer to reading B's state and its inertia |

This notebook removes channels one at a time and in combination, with everything else fixed. A removed token is
**excluded from attention** (key-padding mask) and from the auxiliary heads, at training and at evaluation.

**Arms (same folds, seeds and hyper-parameters as G8b; 3 seeds × 5 folds each):**

| Arm | Face tokens kept | Speech + scene | Question |
|---|---|---|---|
| `Full` | A, L, O | yes | Reference (re-run of G8b's RoleNet) |
| `minus-O` | A, L | yes | Does strict surrogation add information? |
| `minus-L` | A, O | yes | Does observing the target add information? |
| `minus-L3` | A, O, L in clips I/II only | yes | Is B's *current* listening reaction what matters? |
| `minus-A` | L, O | yes | Does A's face (part of the event) add information? |
| `Obs-only` | L, O | no | Observation channels alone |
| `Sim-only` | A | yes | Event (simulation) channel alone |
| `Surr-only` | O | no | Strict surrogation alone |
| `Self-only` | L | no | Target observation alone |

**Caveat.** L is the second most frequent identity in clip III, which is B in about 81% of MCIS (G6a). When the rule
misses, B can sit inside O, so "O" is strict surrogation only up to that error rate. Clip IV is not used to clean this.

**Hypotheses and reading rules (fixed before running).** Metric: pooled out-of-fold UAR of the seed ensemble, plain
argmax (the reporting standard chosen after G10), 95% bootstrap over the 45 episodes. CV here is exploratory; a
contrast "holds" if its CI lower bound is > 0.

* **H1 (strict surrogation):** `Full − minus-O` > 0 on MCIS where an O face is present.
  Secondary: `Surr-only − Sim-only` on MCIS with an O face and **no** listener face in clip III (only bystanders
  observed — Gilbert's comparison in its purest form here).
* **H2 (target observation):** `Full − minus-L` > 0, overall and where the listener is visible in clip III.
* **H3 (current reaction):** `Full − minus-L3` > 0 where the listener is visible in clip III.
* **H4 (observation beats simulation):** `Obs-only − Sim-only` > 0.

**How the result steers the paper's framing:**
* H1 holds → surrogation (strict) is a central channel; the three-channel framework stands.
* H1 fails, H2/H4 hold → the message is "observation beats simulation"; surrogation becomes a supporting idea.
* Neither → the theory framing is dropped.

The test split stays untouched.
"""),
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
G8A_DIR = first_existing("/kaggle/input/datasets/ptrnghieu/g8a-features", "/kaggle/input/g8a-features")
OUT_DIR = "/kaggle/working"

N_OUTER, N_INNER_DEV = 5, 5
SEEDS = [42, 123, 456]                       # as G8b
LR, WEIGHT_DECAY = 1e-4, 1e-5                # B1 settings (only needed by the shared model cell)
FC_EPOCHS, PATIENCE, FC_BATCH = 50, 8, 32
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)     # G8b, unchanged
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
VOICE_SAME_COS = 0.35
DEBUG_PER_EPISODE = None     # e.g. 6 for a quick smoke test

FULL = dict(role=True, faces=True, ctx=True, aux=True, mdrop=True, drop=())
NOCTX = dict(ctx=False, mdrop=False)
EXPERIMENTS = [
    ("Full",      'role', FULL),
    ("minus-O",   'role', {**FULL, 'drop': ('O',)}),
    ("minus-L",   'role', {**FULL, 'drop': ('L',)}),
    ("minus-L3",  'role', {**FULL, 'drop': ('L3',)}),
    ("minus-A",   'role', {**FULL, 'drop': ('A',)}),
    ("Obs-only",  'role', {**FULL, **NOCTX, 'drop': ('A',)}),
    ("Sim-only",  'role', {**FULL, 'drop': ('L', 'O')}),
    ("Surr-only", 'role', {**FULL, **NOCTX, 'drop': ('A', 'L')}),
    ("Self-only", 'role', {**FULL, **NOCTX, 'drop': ('A', 'O')}),
]
"""),
    G8B[3], G3[3], G3[4], G8B[6], G8B[7], G8B[11], G8B[12],
    ("markdown", r"""
## Channel masking
"""),
    ("code", r"""
ROLE_IDX = {'A': 0, 'L': 1, 'O': 2}


def keep_mask(drop):
    # [role, clip] -> is this face token visible to the model?
    K = torch.ones(3, 3, dtype=torch.bool, device=DEVICE)
    for g in drop:
        if g in ROLE_IDX:
            K[ROLE_IDX[g]] = False
        elif g == 'L3':
            K[1, 2] = False
        elif g == 'L12':
            K[1, :2] = False
        else:
            raise ValueError(g)
    return K


class RoleNetAbl(RoleNet):
    # RoleNet with some face tokens removed. Removed tokens are excluded from attention (key-padding mask),
    # from the face auxiliary head, and (if A's clip-III token is removed) the A-emotion head is switched off.
    def __init__(self, cfg, d=RN['D']):
        assert cfg['role'], "ablations need role-tagged face tokens"
        super().__init__(cfg, d)
        self.K = keep_mask(cfg.get('drop', ()))
        if self.head_A is not None and not bool(self.K[0, 2]):
            self.head_A = None

    def forward(self, ix, train=False):
        B, groups, keep, aux = len(ix), [], [], {}
        if self.cfg['faces']:
            h, present = self.pool(FACE[ix], FMASK[ix])                      # [B, 3, 3, d]
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            h = h + self.face_role[None, :, None] + self.clip_emb[None, None]
            ft = h.reshape(B, 9, -1)
            kf = self.K.reshape(9)
            groups.append(ft); keep.append(kf.unsqueeze(0).expand(B, -1))
            if self.head_face is not None:
                w = kf.float()[None, :, None]
                aux['face'] = (self.head_face((ft * w).sum(1) / w.sum(1)), YB[ix], RN['aux_w'])
            if self.head_A is not None:
                tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
                aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        if self.cfg['ctx']:
            spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.ctx_role[0]
            scn = self.scene(SCN[ix]) + self.ctx_role[1]
            ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)     # [B, 6, d]
            groups.append(ct); keep.append(torch.ones(B, 6, dtype=torch.bool, device=ct.device))
            if self.head_ctx is not None:
                aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1)] + groups, 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=toks.device)] + keep, 1)
        if train and self.cfg['mdrop'] and len(groups) == 2:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            valid[:, 1:10] &= ~drop_face.unsqueeze(1)
            valid[:, 10:] &= ~drop_ctx.unsqueeze(1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


MAKE['role'] = lambda cfg: RoleNetAbl(cfg)
print("parameters:", {n: f"{sum(p.numel() for p in MAKE[k](c).parameters()) / 1e6:.2f}M" for n, k, c in EXPERIMENTS})

# which evidence exists in each MCIS (computed from role slots only; no labels)
FM = FMASK.cpu().numpy()
HAS = {
    'A in III': FM[:, 0, 2].any(-1),
    'listener in III': FM[:, 1, 2].any(-1),
    'listener in I/II': FM[:, 1, :2].any((-1, -2)),
    'O anywhere': FM[:, 2].any((-1, -2)),
    'O in III': FM[:, 2, 2].any(-1),
}
HAS['O, no listener in III'] = HAS['O anywhere'] & ~HAS['listener in III']
for k, v in HAS.items():
    print(f"  {k:<24} {v.mean() * 100:5.1f}%  (n={v.sum()})")
"""),
    ("markdown", r"""
## 5-fold episode cross-validation (same folds and seeds as G8b)
"""),
    ("code", r"""
sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)

y_all = DEV.yB.values
src = DEV.source_folder.values
OOF = {name: np.full((len(SEEDS), N, 7), np.nan, np.float32) for name, _, _ in EXPERIMENTS}
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for name, kind, cfg in EXPERIMENTS:
        for si, seed in enumerate(SEEDS):
            p, sel = train_eval(kind, cfg, tr, dev, te, seed + 1000 * f)
            OOF[name][si, te_rows] = p
            w, u = war_uar(p.argmax(1), y_all[te_rows], 7)
            log.append({'fold': f, 'exp': name, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w})
            print(f"fold {f} {name:<10} seed {seed}: sel {sel:5.2f} | UAR {u:5.2f} WAR {w:5.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()

import re
assert all(not np.isnan(v).any() for v in OOF.values())
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g11_fold_seed_log.csv", index=False)
HAS_KEYS = {k: 'has_' + re.sub(r'[^0-9A-Za-z]+', '_', k).strip('_') for k in HAS}
assert len(set(HAS_KEYS.values())) == len(HAS_KEYS), HAS_KEYS          # 'I/II' and 'III' must not collide
np.savez(f"{OUT_DIR}/g11_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         **{HAS_KEYS[k]: v for k, v in HAS.items()},
         **{k.replace('-', '_'): v for k, v in OOF.items()})
print("saved g11_oof_probs.npz and g11_fold_seed_log.csv")
"""),
    ("markdown", r"""
## Results (plain argmax, seed ensemble, 95% bootstrap over episodes)
"""),
    ("code", r"""
def boot_delta(pa, pb, y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(s == e)[0] for e in np.unique(s)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7); wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


PRED = {k: np.log(v.mean(0) + 1e-9).argmax(1) for k, v in OOF.items()}

print(f"== per-seed pooled out-of-fold UAR ({N} MCIS, {len(EPS)} episodes) ==")
for k, v in OOF.items():
    per = [war_uar(v[s].argmax(1), y_all, 7)[1] for s in range(len(v))]
    print(f"  {k:<10} " + " ".join(f"{u:5.2f}" for u in per) + f"  (mean {np.mean(per):.2f})")
print("\n== seed ensemble, all MCIS ==")
for k in PRED:
    report(k, PRED[k], y_all, src)


def contrast(a, b, mask, label):
    m = mask
    if m.sum() < 30 or len(np.unique(src[m])) < 5:
        print(f"  {a:<10} - {b:<10} [{label}] too few MCIS (n={m.sum()})"); return None
    pa, pb = PRED[a][m], PRED[b][m]
    (lo, hi) = boot_delta(pa, pb, y_all[m], src[m])
    wa, ua = war_uar(pa, y_all[m], 7); wb, ub = war_uar(pb, y_all[m], 7)
    print(f"  {a:<10} - {b:<10} [{label}, n={m.sum()}] ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}]  "
          f"ΔWAR {wa - wb:+5.2f} [{lo[1]:+5.2f},{hi[1]:+5.2f}]")
    return ua - ub, lo[0]


ALL = np.ones(N, bool)
print("\n== hypotheses (fixed before running) ==")
R = {}
R['H1'] = contrast('Full', 'minus-O', HAS['O anywhere'], 'O present')
R['H1b'] = contrast('Surr-only', 'Sim-only', HAS['O, no listener in III'], 'O present, no listener in III')
R['H2'] = contrast('Full', 'minus-L', ALL, 'all')
contrast('Full', 'minus-L', HAS['listener in III'], 'listener in III')
R['H3'] = contrast('Full', 'minus-L3', HAS['listener in III'], 'listener in III')
R['H4'] = contrast('Obs-only', 'Sim-only', ALL, 'all')

print("\n== descriptive contrasts ==")
contrast('Full', 'minus-O', ALL, 'all')
contrast('Full', 'minus-A', ALL, 'all')
contrast('Full', 'Obs-only', ALL, 'all')
contrast('Full', 'Sim-only', ALL, 'all')
contrast('Self-only', 'Surr-only', ALL, 'all')
contrast('Self-only', 'Sim-only', ALL, 'all')
contrast('Surr-only', 'Sim-only', HAS['O anywhere'], 'O present')
contrast('minus-L3', 'minus-L', HAS['listener in I/II'], 'listener in I/II')

print("\n== verdicts ==")
for h, v in R.items():
    if v is None:
        print(f"  {h}: not evaluable"); continue
    d, lo = v
    print(f"  {h}: Δ {d:+.2f}, CI low {lo:+.2f} -> {'HOLDS' if lo > 0 else ('directional only' if d > 0 else 'not supported')}")
h1 = R['H1'] is not None and R['H1'][1] > 0
obs = any(R[k] is not None and R[k][1] > 0 for k in ('H2', 'H4'))
print("\nFraming: " + ("strict surrogation is a central channel -> keep the three-channel framework" if h1 else
                       "observation beats simulation; surrogation is a supporting idea" if obs else
                       "no channel claim is supported -> drop the theory framing"))
"""),
]


# ---------------------------------------------------------------- G12: causally structured, additive RoleNet (CS-RoleNet)
G12 = [
    ("markdown", r"""
# G12 — CS-RoleNet: temporally structured, additive paths (boost + explainability), episode-level CV

**Motivation (from G11).**
* Observing B across the context helps (+2.52 UAR), mostly through B's appearances in clips I/II.
* The event channel (speech + scene) and the observation channel are **complementary**: each alone ≈ 23 UAR,
  together 26.
* A's face and bystander faces add no measurable information.

RoleNet mixes all tokens in one bidirectional Transformer, so it cannot say *which* evidence drove a forecast.

**CS-RoleNet.** Three role-specific states are updated **in time order** (clip I → II → III), each by its own GRU cell
that only sees its own evidence:

| Path | Evidence per clip | Reading |
|---|---|---|
| **B-self** | the listener's face token (B in ~81% of MCIS), or a learned *absent* token | B's own affective trajectory (inertia / observation) |
| **Event** | the speech token (text + audio + who-speaks cues) and A's face token | what is said and by whom (includes A's clip-III turn) |
| **Scene** | the scene token (whole-frame + face CLIP means) and the bystanders' face token | the emotional climate of the scene |

The forecast is **additive in logit space**:

  logit(y_B) = b + h_B(z_B) + h_E(z_E) + h_S(z_S) [+ h_int(z_B, z_E, z_S) in the interaction arm]

so every prediction splits **exactly** into per-path contributions. The optional interaction term carries an L2
penalty on its logits (weight 0.1, set a priori) and its share is reported.

**Training aids** (kept from RoleNet in spirit):
* each path has an auxiliary head that predicts y_B on its own (weight 0.3);
* A's emotion is predicted from the clip-III event evidence (weight 0.3);
* **path dropout**: each path's contribution is zeroed with p = 0.1 during training.

This is a *structural prior* (time order + separated mechanisms); its contributions explain the **model**, not causal
effects in the data.

**Arms (same folds, seeds and early stopping as G8b/G11; 3 seeds × 5 folds):** `RoleNet` (reference), `CS-add`,
`CS-int`, and retrained path ablations of `CS-add`: `CS-add -B`, `CS-add -Event`, `CS-add -Scene`.

**Decision rule (fixed before running).** Metric: pooled out-of-fold UAR of the seed ensemble, plain argmax.
* Adopt `CS-add` if UAR(CS-add) ≥ UAR(RoleNet).
* Otherwise adopt `CS-int` if UAR(CS-int) ≥ UAR(RoleNet) **and** its interaction share ≤ 25%.
* Otherwise keep RoleNet.

A gain is called significant only if the 95% episode-bootstrap CI of ΔUAR is above 0.

**Faithfulness check.** Two views of path importance should agree:
* the mean centred contribution share of each path;
* the UAR lost when the path is removed and the model retrained.

Report their rank agreement over the three paths. The test split stays untouched.
"""),
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
G8A_DIR = first_existing("/kaggle/input/datasets/ptrnghieu/g8a-features", "/kaggle/input/g8a-features")
OUT_DIR = "/kaggle/working"

N_OUTER, N_INNER_DEV = 5, 5
SEEDS = [42, 123, 456]                       # as G8b / G11
LR, WEIGHT_DECAY = 1e-4, 1e-5                # B1 settings (only needed by the shared model cell)
FC_EPOCHS, PATIENCE, FC_BATCH = 50, 8, 32
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)     # G8b, unchanged (RoleNet and CS optimiser)
CS = dict(p_path_drop=0.1, int_l2=0.1, dropout=0.2)                  # CS-RoleNet, set a priori (not tuned)
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
VOICE_SAME_COS = 0.35
DEBUG_PER_EPISODE = None     # e.g. 6 for a quick smoke test

FULL = dict(role=True, faces=True, ctx=True, aux=True, mdrop=True)
PATHS = ('B', 'Event', 'Scene')
ARMS = [
    ("RoleNet",        'role', FULL),
    ("CS-add",         'cs',   dict(paths=PATHS, inter=False)),
    ("CS-int",         'cs',   dict(paths=PATHS, inter=True)),
    ("CS-add -B",      'cs',   dict(paths=('Event', 'Scene'), inter=False)),
    ("CS-add -Event",  'cs',   dict(paths=('B', 'Scene'), inter=False)),
    ("CS-add -Scene",  'cs',   dict(paths=('B', 'Event'), inter=False)),
]
EXPERIMENTS = [a for a in ARMS if a[1] == 'role']   # the shared model cell only knows 'role'; CS arms join after it
"""),
    G8B[3], G3[3], G3[4], G8B[6], G8B[7], G8B[11], G8B[12],
    ("markdown", r"""
## CS-RoleNet
"""),
    ("code", r"""
class CSRoleNet(nn.Module):
    # Role-specific states updated clip by clip (I -> II -> III); additive per-path logits.
    def __init__(self, cfg, d=RN['D']):
        super().__init__()
        self.paths, self.inter = tuple(cfg['paths']), cfg['inter']
        self.pool = FramePool(FDIM, d)
        self.absent = nn.Parameter(torch.randn(3, 3, d) * 0.02)          # [role, clip]
        self.face_role = nn.Parameter(torch.randn(3, d) * 0.02)
        self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
        self.voice = nn.Linear(NVOICE, d)
        self.scene = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.clip_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        # per-path input projection (2 evidence tokens -> d) and recurrent cell
        self.inp = nn.ModuleDict({p: nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
                                                   nn.Dropout(CS['dropout'])) for p in PATHS})
        self.cell = nn.ModuleDict({p: nn.GRUCell(d, d) for p in PATHS})
        self.h0 = nn.ParameterDict({p: nn.Parameter(torch.zeros(1, d)) for p in PATHS})
        mk = lambda: nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))
        self.head = nn.ModuleDict({p: mk() for p in PATHS})           # contribution heads (summed)
        self.aux = nn.ModuleDict({p: mk() for p in PATHS})            # each path alone -> y_B (training aid)
        self.bias = nn.Parameter(torch.zeros(7))
        self.head_A = mk()                                             # A's emotion from clip-III event evidence
        if self.inter:
            self.h_int = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(0.3),
                                       nn.Linear(d, 7))

    def evidence(self, ix):
        B = len(ix)
        h, present = self.pool(FACE[ix], FMASK[ix])                        # [B, 3 roles, 3 clips, d]
        h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
        h = h + self.face_role[None, :, None] + self.clip_emb[None, None]
        spk = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.clip_emb
        scn = self.scene(SCN[ix]) + self.clip_emb
        pres = present[:, 1].float().unsqueeze(-1)                          # listener present per clip
        return {'B': torch.stack([h[:, 1], pres.expand_as(h[:, 1]) * self.clip_emb.unsqueeze(0)], 2),
                'Event': torch.stack([spk, h[:, 0]], 2),
                'Scene': torch.stack([scn, h[:, 2]], 2)}                    # each [B, 3 clips, 2, d]

    def states(self, ix):
        ev, Z = self.evidence(ix), {}
        for p in self.paths:
            z = self.h0[p].expand(len(ix), -1)
            for k in range(3):                                              # clip I -> II -> III
                z = self.cell[p](self.inp[p](ev[p][:, k].flatten(1)), z)
            Z[p] = z
        return Z, ev

    def contributions(self, ix):
        Z, ev = self.states(ix)
        C = {p: self.head[p](Z[p]) for p in self.paths}
        if self.inter:
            zz = torch.cat([Z[p] if p in Z else torch.zeros_like(next(iter(Z.values()))) for p in PATHS], 1)
            C['int'] = self.h_int(zz)
        return C, Z, ev

    def forward(self, ix, train=False):
        C, Z, ev = self.contributions(ix)
        aux = {}
        logits = self.bias.expand(len(ix), -1)
        for p, c in C.items():
            if train and p != 'int':
                keep = (torch.rand(len(ix), 1, device=c.device) >= CS['p_path_drop']).float()
                c = c * keep
            logits = logits + c
        if train:
            for p in self.paths:
                aux['path_' + p] = (self.aux[p](Z[p]), YB[ix], RN['aux_w'])
            if 'Event' in self.paths:
                tA = torch.where(FMASK[ix][:, 0, 2].any(-1), YA[ix], torch.full_like(YA[ix], -100))
                aux['A'] = (self.head_A(ev['Event'][:, 2].mean(1)), tA, RN['a_w'])
            if self.inter:
                self._int_pen = CS['int_l2'] * C['int'].pow(2).mean()
        return logits, aux


MAKE['cs'] = lambda cfg: CSRoleNet(cfg)
HP['cs'] = HP['role']


def train_eval_cs(kind, cfg, tr, dev, te, seed):
    # same loop as train_eval, plus the interaction penalty; returns eval probs and per-path contributions
    seed_all(seed)
    hp = HP[kind]
    model = MAKE[kind](cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    y_dev = YB[dev].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(hp['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), hp['batch']):
            j = perm[i:i + hp['batch']]
            logits, aux = model(j, train=True)
            loss = F.cross_entropy(logits, YB[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            if getattr(model, 'inter', False):
                loss = loss + model._int_pen
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        u = war_uar(predict(model, dev).argmax(1), y_dev, 7)[1]
        if u > best:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    contrib = {}
    with torch.no_grad():
        for i in range(0, len(te), 256):
            C = model.contributions(te[i:i + 256])[0]
            for p, c in C.items():
                contrib.setdefault(p, []).append(c.cpu())
    return predict(model, te), best, {p: torch.cat(v).numpy() for p, v in contrib.items()}


EXPERIMENTS = ARMS
print("parameters:", {n: f"{sum(p.numel() for p in MAKE[k](c).parameters()) / 1e6:.2f}M" for n, k, c in EXPERIMENTS})
FM = FMASK.cpu().numpy()
HAS = {'listener in III': FM[:, 1, 2].any(-1), 'listener in I/II': FM[:, 1, :2].any((-1, -2)),
       'no listener': ~FM[:, 1].any((-1, -2))}
for k, v in HAS.items():
    print(f"  {k:<18} {v.mean() * 100:5.1f}%  (n={v.sum()})")
"""),
    ("markdown", r"""
## 5-fold episode cross-validation (same folds and seeds as G8b / G11)
"""),
    ("code", r"""
import re

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)

y_all = DEV.yB.values
src = DEV.source_folder.values
OOF = {name: np.full((len(SEEDS), N, 7), np.nan, np.float32) for name, _, _ in EXPERIMENTS}
CONTRIB = {name: {} for name, kind, _ in EXPERIMENTS if kind == 'cs'}
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for name, kind, cfg in EXPERIMENTS:
        for si, seed in enumerate(SEEDS):
            if kind == 'cs':
                p, sel, C = train_eval_cs(kind, cfg, tr, dev, te, seed + 1000 * f)
                for path, c in C.items():
                    CONTRIB[name].setdefault(path, np.full((len(SEEDS), N, 7), np.nan, np.float32))[si, te_rows] = c
            else:
                p, sel = train_eval(kind, cfg, tr, dev, te, seed + 1000 * f)
            OOF[name][si, te_rows] = p
            w, u = war_uar(p.argmax(1), y_all[te_rows], 7)
            log.append({'fold': f, 'exp': name, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w})
            print(f"fold {f} {name:<14} seed {seed}: sel {sel:5.2f} | UAR {u:5.2f} WAR {w:5.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()

assert all(not np.isnan(v).any() for v in OOF.values())
safe = lambda s: re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
keys = ['has_' + safe(k) for k in HAS] + [safe(k) for k in OOF] + \
       ['contrib_' + safe(n) + '_' + p for n, C in CONTRIB.items() for p in C]
assert len(keys) == len(set(keys)), "colliding npz keys"
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g12_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g12_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         **{'has_' + safe(k): v for k, v in HAS.items()}, **{safe(k): v for k, v in OOF.items()},
         **{'contrib_' + safe(n) + '_' + p: v for n, C in CONTRIB.items() for p, v in C.items()})
print("saved g12_oof_probs.npz and g12_fold_seed_log.csv")
"""),
    ("markdown", r"""
## Results: scores, decision, path contributions and faithfulness (plain argmax, seed ensemble)
"""),
    ("code", r"""
def boot_delta(pa, pb, y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(s == e)[0] for e in np.unique(s)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7); wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


PRED = {k: np.log(v.mean(0) + 1e-9).argmax(1) for k, v in OOF.items()}
UAR = {k: war_uar(p, y_all, 7)[1] for k, p in PRED.items()}
print(f"== per-seed pooled out-of-fold UAR ({N} MCIS, {len(EPS)} episodes) ==")
for k, v in OOF.items():
    per = [war_uar(v[s].argmax(1), y_all, 7)[1] for s in range(len(v))]
    print(f"  {k:<14} " + " ".join(f"{u:5.2f}" for u in per) + f"  (mean {np.mean(per):.2f})")
print("\n== seed ensemble ==")
for k in PRED:
    report(k, PRED[k], y_all, src)


def contrast(a, b, m=None):
    m = np.ones(N, bool) if m is None else m
    lo, hi = boot_delta(PRED[a][m], PRED[b][m], y_all[m], src[m])
    wa, ua = war_uar(PRED[a][m], y_all[m], 7); wb, ub = war_uar(PRED[b][m], y_all[m], 7)
    folds = [war_uar(PRED[a][m & (fold_of_row == f)], y_all[m & (fold_of_row == f)], 7)[1]
             - war_uar(PRED[b][m & (fold_of_row == f)], y_all[m & (fold_of_row == f)], 7)[1] for f in range(N_OUTER)]
    print(f"  {a:<14} - {b:<14} ΔUAR {ua - ub:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}]  ΔWAR {wa - wb:+5.2f}  "
          f"folds>0 {sum(x > 0 for x in folds)}/5")
    return ua - ub


print("\n== contrasts ==")
contrast('CS-add', 'RoleNet'); contrast('CS-int', 'RoleNet'); contrast('CS-int', 'CS-add')
DROP = {}
for p in PATHS:
    DROP[p] = contrast('CS-add', f'CS-add -{p}')


def shares(name):
    # mean |centred contribution| to the predicted class, per path (centring removes each path's constant offset)
    C = {p: v.mean(0) for p, v in CONTRIB[name].items()}
    yhat = PRED[name]
    s = {p: np.abs(c[np.arange(N), yhat] - c.mean(0)[yhat]) for p, c in C.items()}
    tot = sum(s.values())
    return {p: float((v / np.maximum(tot, 1e-9)).mean()) for p, v in s.items()}, s


print("\n== path contribution shares (mean over MCIS) ==")
SH = {}
for name in ('CS-add', 'CS-int'):
    SH[name], raw = shares(name)
    print(f"  {name}: " + ", ".join(f"{p} {v * 100:.1f}%" for p, v in SH[name].items()))
    for sname, m in HAS.items():
        tot = sum(raw.values())[m]
        print(f"      {sname:<18} " + ", ".join(f"{p} {(raw[p][m] / np.maximum(tot, 1e-9)).mean() * 100:.1f}%"
                                              for p in raw))
int_share = SH['CS-int'].get('int', 0.0)

from scipy.stats import spearmanr
rk = spearmanr([SH['CS-add'][p] for p in PATHS], [DROP[p] for p in PATHS]).correlation
print(f"\n== faithfulness: rank agreement (Spearman, 3 paths) between contribution share and retrain drop: {rk:+.2f}")
print("   shares:", {p: round(SH['CS-add'][p] * 100, 1) for p in PATHS}, "| retrain drops:", {p: round(DROP[p], 2) for p in PATHS})

print("\n== examples (CS-add, seed ensemble; contribution of each path to the predicted class, centred) ==")
C = {p: v.mean(0) for p, v in CONTRIB['CS-add'].items()}
rng_ = np.random.default_rng(0)
for i in rng_.choice(N, 5, replace=False):
    yh = PRED['CS-add'][i]
    print(f"  {DEV.sample_id.values[i]}: true {EMO[y_all[i]]:<8} pred {EMO[yh]:<8} | " +
          ", ".join(f"{p} {C[p][i, yh] - C[p][:, yh].mean():+.2f}" for p in PATHS))

print("\n== decision (fixed before running) ==")
if UAR['CS-add'] >= UAR['RoleNet']:
    print(f"ADOPT CS-add: UAR {UAR['CS-add']:.2f} >= RoleNet {UAR['RoleNet']:.2f}")
elif UAR['CS-int'] >= UAR['RoleNet'] and int_share <= 0.25:
    print(f"ADOPT CS-int: UAR {UAR['CS-int']:.2f} >= RoleNet {UAR['RoleNet']:.2f}, interaction share {int_share * 100:.1f}%")
else:
    print(f"KEEP RoleNet ({UAR['RoleNet']:.2f}); CS-add {UAR['CS-add']:.2f}, CS-int {UAR['CS-int']:.2f} "
          f"(interaction share {int_share * 100:.1f}%)")
"""),
]


# ---------------------------------------------------------------- G13: token-level ablations of RoleNet (incl. the query token)
G13 = [
    ("markdown", r"""
# G13 — Token-level ablations of RoleNet, including the query token (episode-level CV)

G11 removed whole face-token groups (A / L / O) and all speech+scene tokens. G13 asks finer questions about
**how RoleNet reads its tokens**. Everything else is fixed: same folds, seeds, hyper-parameters and early stopping as
G8b / G11 / G12; plain scoring; test untouched.

| Family | Arm | Change against `Full` (RoleNet) |
|---|---|---|
| Reference | `Full` | RoleNet as in G8b |
| Noise floor | `Full-reseed` | Identical model, different seeds (seed + 7): run-to-run noise of a seed ensemble |
| Query / readout | `Q-meanpool` | No query token; forecast from the mean of all valid output tokens |
| | `Q-readL3` | No query token; forecast from the output of the *listener, clip III* face token |
| | `Q-readonly` | Query kept, but data tokens cannot attend to it (it only reads) |
| | `Q-3queries` | Three learned query tokens; forecast from the mean of their outputs |
| Embeddings | `E-noRoleEmb` | Face tokens keep their slots but lose the A/L/O role embedding |
| | `E-noClipEmb` | No clip embedding on any token (no clip-order information) |
| | `E-absentMask` | A missing (role, clip) face is masked out instead of using a learned *absent* token |
| Speech / scene | `C-noSpeech` | Speech tokens (all clips) removed |
| | `C-noScene` | Scene tokens (all clips) removed |
| | `C-noSpeechIII` | Only the clip-III speech token (A's turn) removed |
| | `C-noText` / `C-noAudio` / `C-noVoice` | One component removed inside every speech token |
| Time horizon | `T-clipIIIonly` | Only clip-III tokens kept (= removing clips I and II) |
| | `T-noClipI` | Clip-I tokens removed |

Removed tokens are excluded from attention (key-padding mask) and from the auxiliary heads, at training and at
evaluation. Component removals (`C-noText`, …) drop one term from the speech-token sum.

**Reading rules (fixed before running).** Metric: pooled out-of-fold UAR of the seed ensemble; ΔUAR = Full − arm
with a 95% bootstrap over the 45 episodes.
* An element **matters** if the CI of Full − arm is above 0 **and** |Δ| exceeds |Full − Full-reseed|.
* It is **replaceable** if the CI includes 0.
* It **hurts** (a simplification is better) if the CI of arm − Full is above 0.
* G11 showed that retraining without a token group has a generic cost of about 1 UAR, so single contrasts near that
  size are read against the noise-floor arm.
"""),
    ("code", r"""
# ======== CONFIG ========
import os


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths}")


DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
FEATURES_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-features-v2"
SPLIT_CSV = first_existing("/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv",
                           "/kaggle/input/hi-ef-split/source_folder_split_seed42.csv")
G8A_DIR = first_existing("/kaggle/input/datasets/ptrnghieu/g8a-features", "/kaggle/input/g8a-features")
OUT_DIR = "/kaggle/working"

N_OUTER, N_INNER_DEV = 5, 5
SEEDS = [42, 123, 456]                       # as G8b / G11 / G12
LR, WEIGHT_DECAY = 1e-4, 1e-5                # B1 settings (only needed by the shared model cell)
FC_EPOCHS, PATIENCE, FC_BATCH = 50, 8, 32
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)     # G8b, unchanged
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
VOICE_SAME_COS = 0.35
DEBUG_PER_EPISODE = None     # e.g. 6 for a quick smoke test

FULL = dict(role=True, faces=True, ctx=True, aux=True, mdrop=True)
BASE = dict(readout='query', nq=1, q_readonly=False, role_emb=True, clip_emb=True, absent='token',
            drop=(), speech_parts=('text', 'audio', 'voice'), clips=(0, 1, 2), seed_offset=0)
ARMS = [
    ("Full",           'tok', BASE),
    ("Full-reseed",    'tok', {**BASE, 'seed_offset': 7}),
    ("Q-meanpool",     'tok', {**BASE, 'readout': 'mean'}),
    ("Q-readL3",       'tok', {**BASE, 'readout': 'L3'}),
    ("Q-readonly",     'tok', {**BASE, 'q_readonly': True}),
    ("Q-3queries",     'tok', {**BASE, 'nq': 3}),
    ("E-noRoleEmb",    'tok', {**BASE, 'role_emb': False}),
    ("E-noClipEmb",    'tok', {**BASE, 'clip_emb': False}),
    ("E-absentMask",   'tok', {**BASE, 'absent': 'mask'}),
    ("C-noSpeech",     'tok', {**BASE, 'drop': ('speech',)}),
    ("C-noScene",      'tok', {**BASE, 'drop': ('scene',)}),
    ("C-noSpeechIII",  'tok', {**BASE, 'drop': ('speechIII',)}),
    ("C-noText",       'tok', {**BASE, 'speech_parts': ('audio', 'voice')}),
    ("C-noAudio",      'tok', {**BASE, 'speech_parts': ('text', 'voice')}),
    ("C-noVoice",      'tok', {**BASE, 'speech_parts': ('text', 'audio')}),
    ("T-clipIIIonly",  'tok', {**BASE, 'clips': (2,)}),
    ("T-noClipI",      'tok', {**BASE, 'clips': (1, 2)}),
]
FAMILY = {'Noise floor': ['Full-reseed'], 'Query / readout': ['Q-meanpool', 'Q-readL3', 'Q-readonly', 'Q-3queries'],
          'Embeddings': ['E-noRoleEmb', 'E-noClipEmb', 'E-absentMask'],
          'Speech / scene': ['C-noSpeech', 'C-noScene', 'C-noSpeechIII', 'C-noText', 'C-noAudio', 'C-noVoice'],
          'Time horizon': ['T-clipIIIonly', 'T-noClipI']}
EXPERIMENTS = [("RoleNet", 'role', FULL)]    # the shared model cell only knows 'role'; the token arms join after it
"""),
    G8B[3], G3[3], G3[4], G8B[6], G8B[7], G8B[11], G8B[12],
    ("markdown", r"""
## RoleNet with token-level switches
"""),
    ("code", r"""
class RoleNetTok(RoleNet):
    # RoleNet with switches for the readout, the embeddings, the absent token, and which speech/scene/clip tokens exist.
    # Token layout: [queries] + 9 face tokens (role r, clip k at 3r+k) + 3 speech tokens (clip k) + 3 scene tokens (clip k).
    def __init__(self, cfg, d=RN['D']):
        super().__init__(FULL, d)
        self.t = cfg
        self.nq = cfg['nq'] if cfg['readout'] == 'query' else 0
        if self.nq > 1:
            self.queries = nn.Parameter(torch.randn(1, self.nq, d) * 0.02)
        keep_clip = torch.zeros(3, dtype=torch.bool, device=DEVICE)
        keep_clip[list(cfg['clips'])] = True
        kf = keep_clip.repeat(3)                                              # face token 3r+k -> clip k
        if 'L' in cfg['drop']:
            kf[3:6] = False                                                   # listener tokens, all clips (G11 minus-L)
        ks, kc = keep_clip.clone(), keep_clip.clone()
        if 'speech' in cfg['drop']:
            ks[:] = False
        if 'speechIII' in cfg['drop']:
            ks[2] = False
        if 'scene' in cfg['drop']:
            kc[:] = False
        self.keep_face, self.keep_ctx = kf, torch.cat([ks, kc])
        if not bool(self.keep_ctx.any()):
            self.head_ctx = None
        if not bool(kf[2]):                                                  # A's clip-III token gone
            self.head_A = None

    def forward(self, ix, train=False):
        B, t = len(ix), self.t
        d = self.query.shape[-1]
        zero = torch.zeros(1, device=DEVICE)
        role_e = self.face_role if t['role_emb'] else torch.zeros_like(self.face_role)
        clip_e = self.clip_emb if t['clip_emb'] else torch.zeros_like(self.clip_emb)
        aux = {}
        h, present = self.pool(FACE[ix], FMASK[ix])                           # [B, 3, 3, d], [B, 3, 3]
        if t['absent'] == 'token':
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            vf = self.keep_face.unsqueeze(0).expand(B, -1)
        else:
            vf = self.keep_face.unsqueeze(0) & present.reshape(B, 9)
        h = h + role_e[None, :, None] + clip_e[None, None]
        ft = h.reshape(B, 9, -1)
        parts = {'text': self.text(TXT[ix]), 'audio': self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1), 'voice': self.voice(VOI[ix])}
        spk_ = sum(parts[p] for p in t['speech_parts']) + self.ctx_role[0] + clip_e
        scn = self.scene(SCN[ix]) + self.ctx_role[1] + clip_e
        ct = torch.cat([spk_, scn], 1)                                        # [B, 6, d]
        vc = self.keep_ctx.unsqueeze(0).expand(B, -1)
        if self.head_face is not None:
            w = vf.float().unsqueeze(-1)
            aux['face'] = (self.head_face((ft * w).sum(1) / w.sum(1).clamp(min=1)), YB[ix], RN['aux_w'])
        if self.head_A is not None:
            tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
            aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        if self.head_ctx is not None:
            w = vc.float().unsqueeze(-1)
            aux['ctx'] = (self.head_ctx((ct * w).sum(1) / w.sum(1).clamp(min=1)), YB[ix], RN['aux_w'])
        vf, vc = vf.clone(), vc.clone()
        if train and self.cfg['mdrop']:
            u = torch.rand(B, device=DEVICE)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            vf &= ~drop_face.unsqueeze(1)
            vc &= ~drop_ctx.unsqueeze(1)
        nq = self.nq
        q = [] if nq == 0 else [(self.query if nq == 1 else self.queries).expand(B, -1, -1)]
        toks = torch.cat(q + [ft, ct], 1)
        valid = torch.cat([torch.ones(B, nq, dtype=torch.bool, device=DEVICE), vf, vc], 1)
        # never leave a sample without a valid data token (an all-masked row would produce NaN through attention)
        valid[:, nq] |= ~valid[:, nq:].any(1)
        mask = None
        if nq and t['q_readonly']:
            T_ = toks.shape[1]
            mask = torch.zeros(T_, T_, dtype=torch.bool, device=DEVICE)
            mask[nq:, :nq] = True                                            # data tokens cannot attend to the query
        out = self.enc(toks, mask=mask, src_key_padding_mask=~valid)
        if t['readout'] == 'query':
            z = out[:, :nq].mean(1)
        elif t['readout'] == 'mean':
            w = valid.float().unsqueeze(-1)
            z = (out * w).sum(1) / w.sum(1).clamp(min=1)
        else:                                                                 # 'L3': listener face token of clip III
            z = out[:, nq + 3 * 1 + 2]
        return self.head(z), aux


MAKE['tok'] = lambda cfg: RoleNetTok(cfg)
HP['tok'] = HP['role']
EXPERIMENTS = ARMS
print("parameters:", {n: f"{sum(p.numel() for p in MAKE[k](c).parameters()) / 1e6:.3f}M" for n, k, c in EXPERIMENTS})
"""),
    ("markdown", r"""
## 5-fold episode cross-validation (same folds and seeds as G8b / G11 / G12)
"""),
    ("code", r"""
import re

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)

y_all = DEV.yB.values
src = DEV.source_folder.values
OOF = {name: np.full((len(SEEDS), N, 7), np.nan, np.float32) for name, _, _ in EXPERIMENTS}
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for name, kind, cfg in EXPERIMENTS:
        for si, seed in enumerate(SEEDS):
            p, sel = train_eval(kind, cfg, tr, dev, te, seed + 1000 * f + cfg.get('seed_offset', 0))
            OOF[name][si, te_rows] = p
            w, u = war_uar(p.argmax(1), y_all[te_rows], 7)
            log.append({'fold': f, 'exp': name, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w})
            print(f"fold {f} {name:<14} seed {seed}: sel {sel:5.2f} | UAR {u:5.2f} WAR {w:5.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()

assert all(not np.isnan(v).any() for v in OOF.values())
safe = lambda s: re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
assert len({safe(k) for k in OOF}) == len(OOF), "colliding npz keys"
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g13_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g13_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         listener_in_III=FMASK[:, 1, 2].any(-1).cpu().numpy(), **{safe(k): v for k, v in OOF.items()})
print("saved g13_oof_probs.npz and g13_fold_seed_log.csv")
"""),
    ("markdown", r"""
## Results (plain argmax, seed ensemble, 95% bootstrap over episodes)
"""),
    ("code", r"""
def boot_delta(pa, pb, y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(s == e)[0] for e in np.unique(s)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wa, ua = war_uar(pa[idx], y[idx], 7); wb, ub = war_uar(pb[idx], y[idx], 7)
        d.append((ua - ub, wa - wb))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


PRED = {k: np.log(v.mean(0) + 1e-9).argmax(1) for k, v in OOF.items()}
print(f"== per-seed pooled out-of-fold UAR ({N} MCIS, {len(EPS)} episodes) ==")
for k, v in OOF.items():
    per = [war_uar(v[s].argmax(1), y_all, 7)[1] for s in range(len(v))]
    print(f"  {k:<14} " + " ".join(f"{u:5.2f}" for u in per) + f"  (mean {np.mean(per):.2f})")
print("\n== seed ensemble ==")
for k in PRED:
    report(k, PRED[k], y_all, src)

uF = war_uar(PRED['Full'], y_all, 7)[1]
noise = abs(uF - war_uar(PRED['Full-reseed'], y_all, 7)[1])
print(f"\n== Full − arm (ΔUAR > 0 means the removed element helps); noise floor |Full − Full-reseed| = {noise:.2f} ==")
ROWS = []
for fam, arms in FAMILY.items():
    print(f"\n  [{fam}]")
    for a in arms:
        lo, hi = boot_delta(PRED['Full'], PRED[a], y_all, src)
        wa, ua = war_uar(PRED[a], y_all, 7)
        dlt = uF - ua
        folds = sum(war_uar(PRED['Full'][fold_of_row == f], y_all[fold_of_row == f], 7)[1]
                    > war_uar(PRED[a][fold_of_row == f], y_all[fold_of_row == f], 7)[1] for f in range(N_OUTER))
        if a == 'Full-reseed':
            verdict = 'noise floor'
        elif lo[0] > 0 and abs(dlt) > noise:
            verdict = 'MATTERS'
        elif hi[0] < 0:
            verdict = 'HURTS (simpler is better)'
        else:
            verdict = 'replaceable'
        ROWS.append({'family': fam, 'arm': a, 'UAR': ua, 'dUAR': dlt, 'lo': lo[0], 'hi': hi[0], 'folds_full_better': folds,
                     'verdict': verdict})
        print(f"    {a:<14} UAR {ua:5.2f} | Full − arm {dlt:+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}] | "
              f"Full better in {folds}/5 folds | {verdict}")
pd.DataFrame(ROWS).to_csv(f"{OUT_DIR}/g13_summary.csv", index=False)

# the readout question on its natural subset: does reading the listener token work only when the listener is visible?
vis = FMASK[:, 1, 2].any(-1).cpu().numpy()
for sname, m in (('listener in III', vis), ('no listener in III', ~vis)):
    lo, hi = boot_delta(PRED['Full'][m], PRED['Q-readL3'][m], y_all[m], src[m])
    d = war_uar(PRED['Full'][m], y_all[m], 7)[1] - war_uar(PRED['Q-readL3'][m], y_all[m], 7)[1]
    print(f"\n  Full − Q-readL3 [{sname}, n={m.sum()}] {d:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]")
"""),
]


G14 = [
    ("markdown", r"""
# G14 — Ten-seed confirmation of the effects that matter (5-fold episode CV, test untouched)

G13 showed that two 3-seed runs of the *same* RoleNet differ by about 1 UAR (25.00 vs 26.17), and that 220 three-seed
ensembles drawn from 12 same-model runs span 24.6–26.4. Contrasts of about 1 UAR are therefore not established by
3 seeds. G14 re-runs the key arms with **10 seeds** and puts the seed variation into the confidence interval.

| Arm | Change against `Full` (RoleNet) | Earlier result |
|---|---|---|
| `Full` | RoleNet (as `Full` in G13) | 25.00–26.17 over four 3-seed runs |
| `Q-meanpool` | No query token; mean of the valid output tokens | G13: within noise |
| `minus-L` | Listener face tokens (all clips) removed | G11: Full − arm +2.52 |
| `T-clipIIIonly` | Only clip-III tokens kept | G13: +1.92, below all 220 same-model ensembles |
| `noRole` | One pooled face token per clip, no A/L/O roles (G8b `RoleNet-noRole`) | G8b: +1.17, within noise |

Same folds, early-stop episodes, hyper-parameters and plain scoring as G8b / G11–G13. Seeds: the three earlier
seeds plus seven new ones (seed + 1000·fold, as before).

**Analysis (fixed before running).**
* Metric: pooled out-of-fold UAR of the 10-seed ensemble (mean of probabilities, plain argmax).
* Main interval: **two-level bootstrap** (2,000 draws). Each draw resamples the 10 seeds of each arm with replacement
  *and* the 45 episodes, then computes ΔUAR = Full − arm. The interval therefore covers both seed and episode noise.
* `minus-L`, `T-clipIIIonly`, `noRole`: the element is **confirmed** if the lower bound of Full − arm is above 0,
  otherwise **not confirmed**.
* `Q-meanpool` (simplification, non-inferiority margin 1 UAR): **simpler is adequate** if the upper bound of
  Full − Q-meanpool is below +1.0; **query needed** if the lower bound is above 0; otherwise **inconclusive**, and
  RoleNet keeps the query token for continuity.
* Also reported: per-seed UAR mean ± SD per arm, the episode-only bootstrap of the 10-seed ensembles, and fold counts.
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # the three earlier seeds + seven new ones")
        .split("ARMS = [")[0] + """ARMS = [
    ("Full",          'tok',  BASE),
    ("Q-meanpool",    'tok',  {**BASE, 'readout': 'mean'}),
    ("minus-L",       'tok',  {**BASE, 'drop': ('L',)}),
    ("T-clipIIIonly", 'tok',  {**BASE, 'clips': (2,)}),
    ("noRole",        'role', {**FULL, 'role': False}),
]
CONFIRM = ['minus-L', 'T-clipIIIonly', 'noRole']
EXPERIMENTS = [a for a in ARMS if a[1] == 'role']   # the shared model cell only knows 'role'; the token arms join after it
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8], G13[9], G13[10],
    ("markdown", r"""
## 5-fold episode cross-validation (same folds as G8b / G11–G13, 10 seeds)
"""),
    ("code", G13[12][1].replace("g13_", "g14_")),
    ("markdown", r"""
## Results (plain argmax, 10-seed ensemble; two-level bootstrap over seeds and episodes)
"""),
    ("code", r"""
def uar_of(pred, idx=None):
    return war_uar(pred if idx is None else pred[idx], y_all if idx is None else y_all[idx], 7)[1]


def two_level_boot(pa, pb, n_boot=2000, seed=0):
    # pa, pb: [n_seeds, N, 7]; resample seeds of each arm and episodes, return the 95% interval of UAR(a) − UAR(b)
    rng = np.random.default_rng(seed)
    groups = [np.where(src == e)[0] for e in np.unique(src)]
    d = []
    for _ in range(n_boot):
        ea = pa[rng.integers(0, len(pa), len(pa))].mean(0).argmax(1)
        eb = pb[rng.integers(0, len(pb), len(pb))].mean(0).argmax(1)
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d.append(uar_of(ea, idx) - uar_of(eb, idx))
    return np.percentile(d, [2.5, 97.5])


def episode_boot(pa, pb, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(src == e)[0] for e in np.unique(src)]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d.append(uar_of(pa, idx) - uar_of(pb, idx))
    return np.percentile(d, [2.5, 97.5])


PRED = {k: v.mean(0).argmax(1) for k, v in OOF.items()}
print(f"== 10-seed ensemble and per-seed UAR ({N} MCIS, {len(EPS)} episodes) ==")
for k, v in OOF.items():
    per = np.array([uar_of(v[s].argmax(1)) for s in range(len(v))])
    print(f"  {k:<14} ensemble {uar_of(PRED[k]):5.2f} | per seed {per.mean():5.2f} ± {per.std(ddof=1):.2f}")

# how much a 3-seed ensemble of the same model varies (the G13 problem), from the 10 Full seeds
from itertools import combinations
c3 = np.array([uar_of(OOF['Full'][list(c)].mean(0).argmax(1)) for c in combinations(range(len(SEEDS)), 3)])
print(f"\n  Full, all {len(c3)} three-seed sub-ensembles: {c3.mean():.2f} ± {c3.std():.2f} "
      f"(range {c3.min():.2f}–{c3.max():.2f})")

uF = uar_of(PRED['Full'])
ROWS = []
print("\n== Full − arm (ΔUAR > 0: the removed element helps) ==")
for a in [n for n, _, _ in EXPERIMENTS if n != 'Full']:
    d = uF - uar_of(PRED[a])
    lo, hi = two_level_boot(OOF['Full'], OOF[a])
    elo, ehi = episode_boot(PRED['Full'], PRED[a])
    folds = sum(uar_of(PRED['Full'], fold_of_row == f) > uar_of(PRED[a], fold_of_row == f) for f in range(N_OUTER))
    if a in CONFIRM:
        verdict = 'CONFIRMED' if lo > 0 else 'not confirmed'
    else:
        verdict = ('simpler is adequate' if hi < 1.0 else 'query needed' if lo > 0 else 'inconclusive (keep query)')
    ROWS.append({'arm': a, 'UAR': uar_of(PRED[a]), 'dUAR': d, 'lo_2level': lo, 'hi_2level': hi,
                 'lo_episode': elo, 'hi_episode': ehi, 'folds_full_better': folds, 'verdict': verdict})
    print(f"  {a:<14} Δ {d:+5.2f} | seeds+episodes [{lo:+5.2f},{hi:+5.2f}] | episodes only [{elo:+5.2f},{ehi:+5.2f}] | "
          f"Full better in {folds}/5 folds | {verdict}")
pd.DataFrame(ROWS).to_csv(f"{OUT_DIR}/g14_summary.csv", index=False)
print(f"\nsaved {OUT_DIR}/g14_summary.csv")
"""),
]


G15 = [
    ("markdown", r"""
# G15 — Gates for the latent affect-dynamics formulation (analysis only, no model training; test untouched)

**Formulation under test.** B's emotion is a latent state with inertia (DynAffect: a home base, an attractor
strength, perturbations; Kuppens, Oravecz & Tuerlinckx, 2010). B is observed only now and then (the listener's face),
and B's next emotion is forecast by propagating the last observations to clip IV. Two predictions must hold on the
45 train+val episodes before a model is built on it.

**Gate 1: information from observing B decays with time before clip IV.**
Observation of B = the listener L's face (L = B in about 81% of MCIS). Features of one observation = mean over L's
frames of HSEmotion's 8 emotion probabilities plus valence and arousal (10 numbers; no presence or count features).
Predictor = multinomial logistic regression (standardised features, C = 1), out-of-fold over the G8b episode folds.
Information gain per MCIS = log p(y_B) − log π(y_B), where π is the training-fold class prior (add-one).
* **G1a (primary), same MCIS, near vs far.** MCIS where L is seen in clip III and in clip I or II. *Near* = L's frames
  in clip III; *far* = L's frames in the latest of clips II / I where L is seen. Both use the same number of frames
  (the smaller count, evenly spaced). Δ1 = mean IG(near) − mean IG(far), 95% bootstrap over episodes.
* G1b (supporting), within clip III: MCIS with ≥ 4 L frames in clip III; *late* half vs *early* half of L's frames
  (equal counts). Prediction: late > early.
* Secondary: G1a with all frames instead of equal counts; UAR of the logit-adjusted predictions.

**Gate 2: after a longer gap, B's next emotion returns to a baseline.**
Clip IV audio is read **only** for this analysis (ECAPA voice, as in G8b): B spoke in clip k if the voice of clip k
matches clip IV (cosine ≥ 0.35). Then the gold label of clip k is B's last known emotion y_last.
* *Near* group: B spoke in clip II (gap = clip III). *Far* group: B spoke in clip I but not in clip II
  (gap = clips II + III).
* Persistence model: log p(y) = log π(y) + β·[y = y_last] − log Z. β (persistence strength) is fitted by maximum
  likelihood per group.
* **G2a (primary): decay toward the population baseline.** Δβ = β_near − β_far with a 95% episode bootstrap
  (β refitted in every draw). Also reported: β for y_II vs y_I on the same MCIS when B spoke in both.
* **G2b: return to a context-specific baseline (the episode).** Episode baseline π_ep = label distribution of the
  episode's other labelled clips, excluding clips within ±5 of clip IV (smoothed with 5 pseudo-counts of the overall
  distribution). Uses labels of other clips of the same episode, so it is an analysis device, not a model input.
  Out-of-fold IG of the persistence model (IG_last) and of the baseline model log p = log π + γ(log π_ep − log π)
  (IG_ep). D = [IG_ep − IG_last]_far − [IG_ep − IG_last]_near, 95% episode bootstrap.

**Decision rules (fixed before running).**
* Gate 1: **PASS** if the CI of Δ1 is above 0; **DIRECTIONAL** if Δ1 > 0 but the CI includes 0; **FAIL** if Δ1 ≤ 0.
  If IG(near) itself has a CI including 0, the gate is **UNINFORMATIVE** (the face carries too little to test decay).
* Gate 2: **PASS** if the CI of Δβ is above 0; **DIRECTIONAL** if Δβ > 0 with the CI including 0; **FAIL** if Δβ ≤ 0.
  G2b decides the home base of the model: CI of D above 0 → context-dependent home base; otherwise a global one.
* Both gates PASS → build the latent-dynamics model (next notebook). Either FAIL → stop this formulation; the
  problem statement stays, the solution changes. DIRECTIONAL → reported, and the decision goes to the authors.

**Known limitations (stated in advance).** L is B in about 81% of MCIS. In clips I/II, L may be speaking, while in
clip III L listens; speaking faces may be more expressive, which works *against* the near > far prediction. Clip
durations measure elapsed time; gaps between clips are unknown. The episode baseline mixes all speakers.
"""),
    G8B[1],
    ("code", G8B[2][1].split("N_OUTER, N_INNER_DEV")[0] + """N_OUTER = 5
PCA_DIM, MAXF, MAXF_POOL = 128, 24, 32
SAME_PERSON_COS, DOMINANT_MIN_FRAC = 0.45, 0.25
VOICE_SAME_COS = 0.35        # ECAPA cosine taken as "same speaker" (as in G6a / G8b)
RUN_PERSISTENCE_DIAG = True  # reads clip-IV audio for an analysis only (never a model input)
DEBUG_PER_EPISODE = None     # e.g. 6 for a quick smoke test
LR_C, N_BOOT, EP_EXCL, EP_ALPHA = 1.0, 2000, 5, 5.0
"""),
    G8B[3], G8B[6], G8B[7],
    ("markdown", r"""
## Clip IV voice (analysis only) and the G8b persistence diagnostic, re-run
"""),
    G8B[10],
    ("markdown", r"""
## Gates
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar

# folds exactly as G8b-G14
sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold = DEV.source_folder.map(FOLD).values
src = DEV.source_folder.values
y = DEV.yB.values.astype(int)
DUR = np.array([[max(G8[c]['meta'].get('duration') or 0.0, 1e-3) for c in r]
                for r in DEV[['clip1', 'clip2', 'clip3']].values])
LN2 = np.log(2)


def expr(c, idx):
    # one observation: mean over frames of the 8 HSEmotion probabilities, valence, arousal
    fer = np.stack([G8[c]['faces'][j]['fer'] for j in idx]).astype(np.float32)
    return np.concatenate([softmax(fer[:, :8]), fer[:, 8:10]], 1).mean(0)


def dist_to_iv(n, k, c, idx):
    # mean time (s) from the chosen frames to the end of clip III, i.e. to the start of clip IV
    t = np.array([G8[c]['faces'][j]['t'] for j in idx], np.float32)
    return float(np.mean(DUR[n, k] - t) + DUR[n, k + 1:3].sum())


def oof_ig(X, rows):
    # out-of-fold information gain (nats) of a logistic regression, and logit-adjusted predictions
    ig, pred = np.full(len(rows), np.nan), np.full(len(rows), -1)
    yr, fr = y[rows], fold[rows]
    for f in range(N_OUTER):
        tr, te = fr != f, fr == f
        if te.sum() == 0 or len(np.unique(yr[tr])) < 2:
            continue
        pri = np.bincount(yr[tr], minlength=7) + 1.0; pri /= pri.sum()
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=LR_C, max_iter=3000).fit(sc.transform(X[tr]), yr[tr])
        P = np.full((te.sum(), 7), 1e-6); P[:, m.classes_] = m.predict_proba(sc.transform(X[te]))
        P /= P.sum(1, keepdims=True)
        ig[te] = np.log(P[np.arange(te.sum()), yr[te]]) - np.log(pri[yr[te]])
        pred[te] = (np.log(P) - np.log(pri)).argmax(1)
    return ig, pred


def ep_boot(stat, s, n_boot=N_BOOT, seed=0):
    # stat(idx) -> number; 95% interval over episodes resampled with replacement
    rng = np.random.default_rng(seed)
    groups = [np.where(s == e)[0] for e in np.unique(s)]
    v = [stat(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)]
    v = np.array(v, float); v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else np.array([np.nan, np.nan])


def verdict(d, lo):
    return 'PASS' if lo > 0 else ('DIRECTIONAL' if d > 0 else 'FAIL')


PER = pd.DataFrame({'sample_id': DEV.sample_id.values, 'src': src, 'fold': fold, 'y': y,
                    'dur1': DUR[:, 0], 'dur2': DUR[:, 1], 'dur3': DUR[:, 2]})
for k in range(3):
    PER[f'nL{k + 1}'] = [len(SLOT[(n, 1, k)][1]) for n in range(N)]
PER['L_last_gap3'] = [DUR[n, 2] - max(G8[SLOT[(n, 1, 2)][0]]['faces'][j]['t'] for j in SLOT[(n, 1, 2)][1])
                      if SLOT[(n, 1, 2)][1] else np.nan for n in range(N)]
RES = {}

# ---------------- Gate 1a: near (clip III) vs far (latest of II / I), same MCIS ----------------
rows, Xn, Xf, Xn_all, Xf_all, dn, dfar = [], [], [], [], [], [], []
for n in range(N):
    cn, idn = SLOT[(n, 1, 2)]
    kf = next((k for k in (1, 0) if SLOT[(n, 1, k)][1]), None)
    if not idn or kf is None:
        continue
    cf, idf = SLOT[(n, 1, kf)]
    m = min(len(idn), len(idf))
    pn, pf = pick_frames(idn, m), pick_frames(idf, m)
    rows.append(n); Xn.append(expr(cn, pn)); Xf.append(expr(cf, pf)); Xn_all.append(expr(cn, idn)); Xf_all.append(expr(cf, idf))
    dn.append(dist_to_iv(n, 2, cn, pn)); dfar.append(dist_to_iv(n, kf, cf, pf))
rows = np.array(rows, int)
print(f"G1a: {len(rows)} MCIS with L in clip III and in clip I/II | median time to clip IV: near {np.median(dn):.2f}s, "
      f"far {np.median(dfar):.2f}s" if len(rows) else "G1a: no MCIS")
if len(rows) >= 50:
    ign, pn_ = oof_ig(np.array(Xn), rows); igf, pf_ = oof_ig(np.array(Xf), rows)
    ign_a, _ = oof_ig(np.array(Xn_all), rows); igf_a, _ = oof_ig(np.array(Xf_all), rows)
    ok = np.isfinite(ign) & np.isfinite(igf)
    s1, d1 = src[rows][ok], (ign - igf)[ok]
    D1 = d1.mean() / LN2
    lo1, hi1 = ep_boot(lambda i: d1[i].mean(), s1) / LN2
    lon, hin = ep_boot(lambda i: ign[ok][i].mean(), s1) / LN2
    lof, hif = ep_boot(lambda i: igf[ok][i].mean(), s1) / LN2
    da = (ign_a - igf_a)[ok]
    loa, hia = ep_boot(lambda i: da[i].mean(), s1) / LN2
    un, uf = war_uar(pn_[ok], y[rows][ok], 7)[1], war_uar(pf_[ok], y[rows][ok], 7)[1]
    print(f"  IG near {ign[ok].mean() / LN2:+.4f} bits [{lon:+.4f},{hin:+.4f}] | IG far {igf[ok].mean() / LN2:+.4f} bits "
          f"[{lof:+.4f},{hif:+.4f}]")
    print(f"  Δ1 = near − far {D1:+.4f} bits [{lo1:+.4f},{hi1:+.4f}] (equal frame counts, primary)")
    print(f"  all frames: Δ {da.mean() / LN2:+.4f} bits [{loa:+.4f},{hia:+.4f}] | UAR (logit-adjusted) near {un:.2f}, far {uf:.2f}")
    g1 = 'UNINFORMATIVE' if lon <= 0 else verdict(D1, lo1)
    RES['G1a'] = dict(n=int(ok.sum()), IG_near=ign[ok].mean() / LN2, IG_far=igf[ok].mean() / LN2, delta=D1, lo=lo1, hi=hi1,
                      delta_allframes=da.mean() / LN2, med_s_near=float(np.median(dn)), med_s_far=float(np.median(dfar)),
                      verdict=g1)
    PER.loc[rows, 'IG1_near'] = ign / LN2; PER.loc[rows, 'IG1_far'] = igf / LN2
    PER.loc[rows, 's_near'] = dn; PER.loc[rows, 's_far'] = dfar
else:
    g1 = 'UNINFORMATIVE'
    RES['G1a'] = dict(n=int(len(rows)), verdict=g1)
print(f"  GATE 1 (G1a): {g1}")

# ---------------- Gate 1b: late vs early half of L's frames in clip III ----------------
rows_b, Xe, Xl = [], [], []
for n in range(N):
    c, idx = SLOT[(n, 1, 2)]
    if len(idx) >= 4:
        h = len(idx) // 2
        rows_b.append(n); Xe.append(expr(c, idx[:h])); Xl.append(expr(c, idx[-h:]))
rows_b = np.array(rows_b, int)
if len(rows_b) >= 50:
    ige, _ = oof_ig(np.array(Xe), rows_b); igl, _ = oof_ig(np.array(Xl), rows_b)
    ok = np.isfinite(ige) & np.isfinite(igl)
    db = (igl - ige)[ok]
    lob, hib = ep_boot(lambda i: db[i].mean(), src[rows_b][ok]) / LN2
    print(f"G1b: {ok.sum()} MCIS | IG late {igl[ok].mean() / LN2:+.4f}, early {ige[ok].mean() / LN2:+.4f} bits | "
          f"late − early {db.mean() / LN2:+.4f} [{lob:+.4f},{hib:+.4f}]")
    RES['G1b'] = dict(n=int(ok.sum()), delta=db.mean() / LN2, lo=lob, hi=hib)
else:
    print(f"G1b: too few MCIS ({len(rows_b)})")

# ---------------- Gate 2: persistence of B's last labelled emotion ----------------
BSP = np.zeros((N, 2), bool); YCTX = np.full((N, 2), -1)
for n, row in enumerate(DEV.itertuples()):
    v4 = V4.get(row.clip4)
    for k, c in enumerate((row.clip1, row.clip2)):
        YCTX[n, k] = gold(c)
        a = G8[c]['audio']
        if v4 is not None and a is not None and a.get('ecapa') is not None:
            BSP[n, k] = float(a['ecapa'].astype(np.float32) @ v4.astype(np.float32)) >= VOICE_SAME_COS
PER['B_spoke_I'], PER['B_spoke_II'], PER['y_I'], PER['y_II'] = BSP[:, 0], BSP[:, 1], YCTX[:, 0], YCTX[:, 1]

# episode baseline from the episode's other labelled clips (|clip number - clip IV number| > EP_EXCL)
lab = ann[7].map(E2I)
lab = lab[lab.notna()].astype(int)
lab_ep = pd.Series([i.split('/')[0] for i in lab.index], index=lab.index)
lab_num = pd.Series([int(i.split('/')[1]) for i in lab.index], index=lab.index)
dev_eps = set(DEV.source_folder)
glob_p = np.bincount(lab[lab_ep.isin(dev_eps)].values, minlength=7) + 1.0; glob_p /= glob_p.sum()
PI_EP = np.zeros((N, 7))
for n, row in enumerate(DEV.itertuples()):
    ep, num4 = row.clip4.split('/')[0], int(row.clip4.split('/')[1])
    sel = (lab_ep.values == ep) & (np.abs(lab_num.values - num4) > EP_EXCL)
    cnt = np.bincount(lab.values[sel], minlength=7)
    PI_EP[n] = (cnt + EP_ALPHA * glob_p) / (cnt.sum() + EP_ALPHA)

near = BSP[:, 1] & (YCTX[:, 1] >= 0)
far = BSP[:, 0] & ~BSP[:, 1] & (YCTX[:, 0] >= 0)
YLAST = np.where(near, YCTX[:, 1], np.where(far, YCTX[:, 0], -1))
print(f"\nG2: B spoke in clip II (near) {near.sum()} MCIS | in clip I only (far) {far.sum()} MCIS | "
      f"median gap near {np.median(DUR[near, 2]) if near.any() else float('nan'):.2f}s, "
      f"far {np.median(DUR[far, 1:].sum(1)) if far.any() else float('nan'):.2f}s")


def fit_scalar(nll):
    return minimize_scalar(nll, bounds=(-5, 5), method='bounded').x


def persist_logp(beta, yl, pri):
    lp = np.log(pri)[None].repeat(len(yl), 0); lp[np.arange(len(yl)), yl] += beta
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def base_logp(gam, pe, pri):
    lp = np.log(pri)[None] + gam * (np.log(pe) - np.log(pri)[None])
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def fit_beta(r, yl):
    pri = np.bincount(y[r], minlength=7) + 1.0; pri /= pri.sum()
    return fit_scalar(lambda b: -persist_logp(b, yl, pri)[np.arange(len(r)), y[r]].sum())


def oof_group(r):
    # out-of-fold IG (nats) of the persistence model and of the episode-baseline model
    igl, ige = np.full(len(r), np.nan), np.full(len(r), np.nan)
    for f in range(N_OUTER):
        tr, te = fold[r] != f, fold[r] == f
        if te.sum() == 0 or tr.sum() < 10:
            continue
        a, b = r[tr], r[te]
        pri = np.bincount(y[a], minlength=7) + 1.0; pri /= pri.sum()
        bet = fit_scalar(lambda v: -persist_logp(v, YLAST[a], pri)[np.arange(len(a)), y[a]].sum())
        gam = fit_scalar(lambda v: -base_logp(v, PI_EP[a], pri)[np.arange(len(a)), y[a]].sum())
        igl[te] = persist_logp(bet, YLAST[b], pri)[np.arange(len(b)), y[b]] - np.log(pri[y[b]])
        ige[te] = base_logp(gam, PI_EP[b], pri)[np.arange(len(b)), y[b]] - np.log(pri[y[b]])
    return igl, ige


rn, rf = np.where(near)[0], np.where(far)[0]
if len(rn) >= 30 and len(rf) >= 30:
    bn, bf = fit_beta(rn, YLAST[rn]), fit_beta(rf, YLAST[rf])
    allr = np.concatenate([rn, rf]); isn = np.r_[np.ones(len(rn), bool), np.zeros(len(rf), bool)]

    def dbeta(i):
        a, b = allr[i][isn[i]], allr[i][~isn[i]]
        if len(a) < 10 or len(b) < 10:
            return np.nan
        return fit_beta(a, YLAST[a]) - fit_beta(b, YLAST[b])

    lo2, hi2 = ep_boot(dbeta, src[allr], n_boot=1000)
    g2 = verdict(bn - bf, lo2)
    print(f"  G2a persistence β: near {bn:+.3f} | far {bf:+.3f} | Δβ {bn - bf:+.3f} [{lo2:+.3f},{hi2:+.3f}]  -> GATE 2: {g2}")
    print(f"      P(y_IV = y_last): near {np.mean(y[rn] == YLAST[rn]) * 100:.1f}% | far {np.mean(y[rf] == YLAST[rf]) * 100:.1f}%")
    igl_n, ige_n = oof_group(rn); igl_f, ige_f = oof_group(rf)
    PER.loc[rn, 'IG2_last'] = igl_n / LN2; PER.loc[rn, 'IG2_ep'] = ige_n / LN2
    PER.loc[rf, 'IG2_last'] = igl_f / LN2; PER.loc[rf, 'IG2_ep'] = ige_f / LN2
    PER['G2_group'] = np.where(near, 'near', np.where(far, 'far', ''))
    vv = np.r_[ige_n - igl_n, ige_f - igl_f]
    ok = np.isfinite(vv)

    def dd(i):
        i = i[ok[i]]
        a, b = vv[i][~isn[i]], vv[i][isn[i]]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan

    Dv = dd(np.arange(len(allr)))
    lod, hid = ep_boot(dd, src[allr])
    print(f"  G2b OOF IG (bits): near last {np.nanmean(igl_n) / LN2:+.4f}, episode {np.nanmean(ige_n) / LN2:+.4f} | "
          f"far last {np.nanmean(igl_f) / LN2:+.4f}, episode {np.nanmean(ige_f) / LN2:+.4f}")
    hb = 'context-dependent' if lod > 0 else 'global'
    print(f"      D = [ep − last]_far − [ep − last]_near {Dv / LN2:+.4f} bits [{lod / LN2:+.4f},{hid / LN2:+.4f}] "
          f"-> home base: {hb}")
    RES['G2a'] = dict(n_near=len(rn), n_far=len(rf), beta_near=bn, beta_far=bf, delta=bn - bf, lo=lo2, hi=hi2, verdict=g2)
    RES['G2b'] = dict(D_bits=Dv / LN2, lo=lod / LN2, hi=hid / LN2, home_base=hb)
else:
    g2 = 'UNINFORMATIVE'
    print(f"  too few MCIS for gate 2 (near {len(rn)}, far {len(rf)})")
    RES['G2a'] = dict(n_near=len(rn), n_far=len(rf), verdict=g2)

both = BSP.all(1) & (YCTX >= 0).all(1)
if both.sum() >= 30:
    r = np.where(both)[0]
    b2, b1 = fit_beta(r, YCTX[r, 1]), fit_beta(r, YCTX[r, 0])
    lob2, hib2 = ep_boot(lambda i: fit_beta(r[i], YCTX[r[i], 1]) - fit_beta(r[i], YCTX[r[i], 0]), src[r], n_boot=1000)
    print(f"  G2a within MCIS (B spoke in I and II, n={both.sum()}): β(y_II) {b2:+.3f} vs β(y_I) {b1:+.3f}, "
          f"Δ {b2 - b1:+.3f} [{lob2:+.3f},{hib2:+.3f}]")
    RES['G2a_within'] = dict(n=int(both.sum()), beta_II=b2, beta_I=b1, delta=b2 - b1, lo=lob2, hi=hib2)

decision = ('BUILD the latent-dynamics model' if g1 == 'PASS' and g2 == 'PASS' else
            'STOP this formulation' if 'FAIL' in (g1, g2) else 'REPORT, authors decide')
print(f"\n== GATE 1: {g1} | GATE 2: {g2} | decision (fixed rule): {decision} ==")
PER.to_csv(f"{OUT_DIR}/g15_per_mcis.csv", index=False)
json.dump({k: {a: (float(b) if isinstance(b, (np.floating, float)) else b) for a, b in v.items()} for k, v in RES.items()}
          | {'decision': decision}, open(f"{OUT_DIR}/g15_gates.json", 'w'), indent=1, default=str)
print(f"saved {OUT_DIR}/g15_per_mcis.csv and g15_gates.json")
"""),
]


G16 = [
    ("markdown", r"""
# G16 — Separating elapsed time from the type of observation (analysis only; test untouched)

G15 gate 1 passed: the listener's face in clip III says about 4× more about B's next emotion than the same person's
face in clip II/I. But the near observation is a *listening reaction*, while the far one may be the person
*speaking*. G15's within-clip time contrast (late vs early half of clip III) was flat. G16 asks whether elapsed
time matters when the type of observation is held fixed. Everything (features, predictor, folds, information gain,
episode bootstrap) is as in G15, whose cells run first.

B spoke in clip k is decided by the clip-IV voice (analysis only), as in G15. Only clips where both voices exist
are used for the "B did not speak" conditions.

| Contrast | MCIS | a vs b | What it isolates |
|---|---|---|---|
| **T1 (primary)** | L seen in clips I and II; B spoke in neither | L in II vs L in I (equal frames) | Time, with the type held (both non-speaking) |
| T1-all | L seen in clips I and II | L in II vs L in I | Time, more MCIS, type not controlled |
| **T2** | L seen in III and in the latest of II/I, and B did not speak in that clip | L in III vs L far | G15 gate 1 with a non-speaking far observation |
| T3 (descriptive) | far observations of G15 G1a | IG when B spoke in the far clip vs not | Type of observation, time roughly held |

**Decision rules (fixed before running).**
* T1 CI above 0 → elapsed time matters with the type held → a continuous-time (OU-type) latent state is justified.
* T1 > 0 with the CI including 0 → directional; authors decide.
* T1 ≤ 0 → no evidence of decay beyond the clip-III reaction. The model then treats the clip-III reaction and the
  context as separate evidence (a discrete-step state), not an OU process.
* T2 is read with T1: T2 CI above 0 means near > far survives when both are non-speaking observations.
"""),
] + G15[1:] + [
    ("markdown", r"""
## G16 contrasts
"""),
    ("code", r"""
VOK = np.zeros((N, 2), bool)
for n, row in enumerate(DEV.itertuples()):
    for k, c in enumerate((row.clip1, row.clip2)):
        a = G8[c]['audio']
        VOK[n, k] = row.clip4 in V4 and a is not None and a.get('ecapa') is not None
SILENT = VOK & ~BSP                                     # voice known, and B did not speak in clip k
RES16 = {}


def contrast(name, rows, Xa, Xb, la, lb):
    rows = np.array(rows, int)
    if len(rows) < 50:
        print(f"{name}: too few MCIS ({len(rows)})"); RES16[name] = dict(n=int(len(rows))); return None
    ia, _ = oof_ig(np.array(Xa), rows); ib, _ = oof_ig(np.array(Xb), rows)
    ok = np.isfinite(ia) & np.isfinite(ib)
    d, s_ = (ia - ib)[ok], src[rows][ok]
    lo, hi = ep_boot(lambda i: d[i].mean(), s_) / LN2
    print(f"{name}: n={ok.sum()} | IG {la} {ia[ok].mean() / LN2:+.4f}, {lb} {ib[ok].mean() / LN2:+.4f} bits | "
          f"{la} − {lb} {d.mean() / LN2:+.4f} [{lo:+.4f},{hi:+.4f}]")
    RES16[name] = dict(n=int(ok.sum()), IG_a=ia[ok].mean() / LN2, IG_b=ib[ok].mean() / LN2, delta=d.mean() / LN2,
                       lo=lo, hi=hi)
    return d.mean() / LN2, lo


def pair(n, ka, kb):
    (ca, ia_), (cb, ib_) = SLOT[(n, 1, ka)], SLOT[(n, 1, kb)]
    m = min(len(ia_), len(ib_))
    return expr(ca, pick_frames(ia_, m)), expr(cb, pick_frames(ib_, m))


# T1 / T1-all: clip II vs clip I
r1, a1, b1, r1a, a1a, b1a = [], [], [], [], [], []
for n in range(N):
    if SLOT[(n, 1, 0)][1] and SLOT[(n, 1, 1)][1]:
        xa, xb = pair(n, 1, 0)
        r1a.append(n); a1a.append(xa); b1a.append(xb)
        if SILENT[n].all():
            r1.append(n); a1.append(xa); b1.append(xb)
t1 = contrast('T1 (B silent in I and II)', r1, a1, b1, 'II', 'I')
contrast('T1-all', r1a, a1a, b1a, 'II', 'I')

# T2: clip III vs the latest far clip, where B did not speak in that far clip
r2, a2, b2 = [], [], []
for n in range(N):
    kf = next((k for k in (1, 0) if SLOT[(n, 1, k)][1]), None)
    if SLOT[(n, 1, 2)][1] and kf is not None and SILENT[n, kf]:
        xa, xb = pair(n, 2, kf)
        r2.append(n); a2.append(xa); b2.append(xb)
t2 = contrast('T2 (far clip non-speaking)', r2, a2, b2, 'III', 'far')

# T3: G15's far observations, split by whether B spoke in that far clip
if 'IG1_far' in PER:
    far_k = np.array([next((k for k in (1, 0) if SLOT[(n, 1, k)][1]), -1) for n in range(N)])
    has = PER['IG1_far'].notna().values & (far_k >= 0)
    kk = np.where(has, far_k, 0)
    known = has & VOK[np.arange(N), kk]
    spoke = known & BSP[np.arange(N), kk]
    quiet = known & ~BSP[np.arange(N), kk]
    for nm, m in (('far clip, B spoke', spoke), ('far clip, B silent', quiet)):
        print(f"T3 {nm}: n={m.sum()} | IG far {PER.loc[m, 'IG1_far'].mean():+.4f} bits | "
              f"IG near {PER.loc[m, 'IG1_near'].mean():+.4f} bits")
        RES16[f'T3 {nm}'] = dict(n=int(m.sum()), IG_far=float(PER.loc[m, 'IG1_far'].mean()),
                                 IG_near=float(PER.loc[m, 'IG1_near'].mean()))

if t1 is None:
    dec = 'T1 UNINFORMATIVE (too few MCIS)'
elif t1[1] > 0:
    dec = 'time matters with the type held -> continuous-time latent state justified'
elif t1[0] > 0:
    dec = 'DIRECTIONAL -> authors decide'
else:
    dec = 'no decay beyond the clip-III reaction -> discrete-step state, not OU'
print(f"\n== G16 decision (fixed rule): {dec} ==")
PER['B_voice_known_I'], PER['B_voice_known_II'] = VOK[:, 0], VOK[:, 1]
PER.to_csv(f"{OUT_DIR}/g16_per_mcis.csv", index=False)
json.dump({k: {a: (float(b) if isinstance(b, (np.floating, float)) else b) for a, b in v.items()} for k, v in RES16.items()}
          | {'decision': dec}, open(f"{OUT_DIR}/g16_results.json", 'w'), indent=1, default=str)
print(f"saved {OUT_DIR}/g16_per_mcis.csv and g16_results.json")
"""),
]


G17 = [
    ("markdown", r"""
# G17 — Listening vs speaking observations of the same person (paired; analysis only; test untouched)

G16 suggested that the value of observing the target depends on *what the target is doing*: faces of B while B
speaks carried ≈ 0 bits about B's next emotion, faces while B is silent carried +0.046 bits. That comparison was
between different MCIS. G17 makes it **paired**: the same MCIS, the same person (the listener L of clip III), one
context clip in which B speaks and one in which B is silent. G15/G16 cells run first; features, predictor, folds,
information gain and episode bootstrap are unchanged. B's speaking is decided by the clip-IV voice (analysis only).

* **P1 (primary).** MCIS where L is seen in clips I and II, both voices are known, and B spoke in exactly one of the
  two clips. *Silent* = L's frames in the clip where B did not speak; *speaking* = L's frames in the clip where B
  spoke; equal frame counts. Δ = mean IG(silent) − mean IG(speaking), separate logistic regressions as in G15/G16.
  Reported also by order (silent clip later / earlier), since G16 found no time effect but the order is not balanced.
* **P2 (secondary).** One logistic regression fitted on *all* context observations of L (both types pooled; all
  frames), out of fold. IG of each observation; paired difference silent − speaking within the P1 MCIS, and the
  mean IG of each type over all observations.

**Decision rules (fixed before running).**
* P1 CI above 0 → constraint (iii) stays in the main problem statement.
* P1 > 0 with the CI including 0 → (iii) becomes a reported finding with this caveat, not part of the statement.
* P1 ≤ 0 → (iii) is dropped.
"""),
] + G16[1:] + [
    ("markdown", r"""
## G17 contrasts
"""),
    ("code", r"""
RES17 = {}
# P1: same MCIS, one silent and one speaking context clip
rP, xs, xk, order = [], [], [], []
for n in range(N):
    if SLOT[(n, 1, 0)][1] and SLOT[(n, 1, 1)][1] and VOK[n].all() and BSP[n].sum() == 1:
        ks, kk = int(np.where(~BSP[n])[0][0]), int(np.where(BSP[n])[0][0])
        a, b = pair(n, ks, kk)
        rP.append(n); xs.append(a); xk.append(b); order.append('silent later' if ks > kk else 'silent earlier')
p1 = contrast('P1 silent vs speaking (same MCIS)', rP, xs, xk, 'silent', 'speaking')
RES17['P1'] = RES16.get('P1 silent vs speaking (same MCIS)')
order = np.array(order)
for o in ('silent later', 'silent earlier'):
    m = order == o
    contrast(f'P1 [{o}]', np.array(rP)[m], [x for x, t in zip(xs, m) if t], [x for x, t in zip(xk, m) if t],
             'silent', 'speaking')
    RES17[f'P1 {o}'] = RES16.get(f'P1 [{o}]')

# P2: one model over all context observations of L
orow, oX, otype = [], [], []
for n in range(N):
    for k in (0, 1):
        c, idx = SLOT[(n, 1, k)]
        if idx and VOK[n, k]:
            orow.append(n); oX.append(expr(c, idx)); otype.append('speaking' if BSP[n, k] else 'silent')
orow, otype = np.array(orow, int), np.array(otype)
if len(orow) >= 50:
    ig, _ = oof_ig(np.array(oX), orow)
    ig = ig / LN2
    for t in ('silent', 'speaking'):
        m = (otype == t) & np.isfinite(ig)
        lo, hi = ep_boot(lambda i: ig[m][i].mean(), src[orow][m])
        print(f"P2 all context observations, {t}: n={m.sum()} | IG {ig[m].mean():+.4f} bits [{lo:+.4f},{hi:+.4f}]")
        RES17[f'P2 all {t}'] = dict(n=int(m.sum()), IG=float(ig[m].mean()), lo=lo, hi=hi)
    pairs = [(np.where((orow == n) & (otype == 'silent'))[0], np.where((orow == n) & (otype == 'speaking'))[0]) for n in rP]
    d2 = np.array([ig[a[0]] - ig[b[0]] for a, b in pairs if len(a) and len(b)])
    ok = np.isfinite(d2)
    s2 = np.array([n for (a, b), n in zip(pairs, rP) if len(a) and len(b)])
    if ok.sum() >= 30:
        lo, hi = ep_boot(lambda i: d2[ok][i].mean(), src[s2][ok])
        print(f"P2 paired (P1 MCIS, one model): n={ok.sum()} | silent − speaking {d2[ok].mean():+.4f} bits [{lo:+.4f},{hi:+.4f}]")
        RES17['P2 paired'] = dict(n=int(ok.sum()), delta=float(d2[ok].mean()), lo=lo, hi=hi)

if p1 is None:
    dec = 'P1 UNINFORMATIVE (too few MCIS)'
elif p1[1] > 0:
    dec = 'constraint (iii) stays in the problem statement'
elif p1[0] > 0:
    dec = 'DIRECTIONAL -> (iii) becomes a reported finding with a caveat'
else:
    dec = 'constraint (iii) dropped'
print(f"\n== G17 decision (fixed rule): {dec} ==")
json.dump({k: ({a: (float(b) if isinstance(b, (np.floating, float)) else b) for a, b in v.items()} if v else v)
           for k, v in RES17.items()} | {'decision': dec}, open(f"{OUT_DIR}/g17_results.json", 'w'), indent=1, default=str)
print(f"saved {OUT_DIR}/g17_results.json")
"""),
]


G19 = [
    ("markdown", r"""
# G19 — Which emotion distinctions are forecastable, and which ones clip III adds (5-fold CV, train+val; test untouched)

**Question.** A 7-class label hides 21 pairwise distinctions that may differ in how well they can be forecast from
the observed window, and in how much clip III adds. For each pair (a, b) we measure how well a predictor separates
MCIS whose clip-IV label is a from those labelled b, with clips I–II only and with clips I–III.

**Predictors (two families).**
* RoleNet (the G14 `Full` model) restricted by the token mask to the clips of the information set; 10 seeds; the same
  folds, early stopping and hyper-parameters as G14.
* Multinomial logistic regression on pooled per-clip features: mean HSEmotion probabilities + valence/arousal over all
  faces in the clip, a face-present flag, and text, audio and scene features reduced to 32 dims each by PCA fitted in
  the training fold. Standardised, C = 1. No tuning. Trained on all training-fold rows.

**Information sets:** I–II, I–III, and III only (reported, not used in the rules).

**Measure.** Pair AUC of log p(a|X) − log p(b|X) on the out-of-fold MCIS labelled a or b (invariant to class priors).
Pairs where a class has fewer than 50 MCIS in train+val are reported as "insufficient data" and excluded from the
rules (this removes the 6 pairs with fear). 95% intervals: RoleNet by a two-level bootstrap over seeds and
source folders; the logistic regression by a bootstrap over source folders. No multiplicity correction (pilot).

**Decision rules (fixed before running).**
1. *Stable structure*: Spearman correlation between the RoleNet and the logistic-regression pair-AUC vectors at
   I–III, over the eligible pairs, is ≥ 0.7.
2. *Information-dependent structure* (RoleNet): adding clip III to I–II gives Δ = AUC(I–III) − AUC(I–II) with
   CI > 0 for at least 2 eligible pairs, **and** at least 2 eligible pairs have AUC(I–II) with CI lower bound > 0.5
   while their Δ CI includes 0.
3. **CONTINUE** if 1 and 2 hold; otherwise **STOP** this direction (structure unstable across predictors, or the gain
   from clip III is close to uniform).
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # as G14")
        .split("ARMS = [")[0] + """ARMS = [
    ("I-II",  'tok', {**BASE, 'clips': (0, 1)}),
    ("I-III", 'tok', BASE),
    ("III",   'tok', {**BASE, 'clips': (2,)}),
]
INFOSETS = {'I-II': (0, 1), 'I-III': (0, 1, 2), 'III': (2,)}
MIN_CLASS, LR_C, PCA_K, N_BOOT = 50, 1.0, 32, 1000
EXPERIMENTS = [a for a in ARMS if a[1] == 'role']
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8], G13[9], G13[10],
    ("markdown", r"""
## RoleNet: 5-fold CV over the information sets (10 seeds)
"""),
    ("code", G13[12][1].replace("g13_", "g19_")),
    ("markdown", r"""
## Logistic regression on pooled per-clip features (same folds)
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA

EXPR = np.zeros((N, 3, 11), np.float32)
for n, row in enumerate(DEV.itertuples()):
    for k, c in enumerate((row.clip1, row.clip2, row.clip3)):
        if len(FB[c]):
            EXPR[n, k, :10] = FB[c][:, :10].mean(0); EXPR[n, k, 10] = 1.0
TXTn, AUDn, SCNn = TXT.cpu().numpy(), AUD.cpu().numpy(), SCN.cpu().numpy()
LRP = {k: np.full((N, 7), np.nan, np.float32) for k in INFOSETS}
for f in range(N_OUTER):
    tr, te = fold_of_row != f, fold_of_row == f
    for name, clips in INFOSETS.items():
        blocks_tr, blocks_te = [], []
        for k in clips:
            blocks_tr.append(EXPR[tr, k]); blocks_te.append(EXPR[te, k])
            for M in (TXTn, AUDn, SCNn):
                pca = _PCA(PCA_K, random_state=0).fit(M[tr, k])
                blocks_tr.append(pca.transform(M[tr, k])); blocks_te.append(pca.transform(M[te, k]))
        Xtr, Xte = np.concatenate(blocks_tr, 1), np.concatenate(blocks_te, 1)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=LR_C, max_iter=5000).fit(sc.transform(Xtr), y_all[tr])
        P = np.full((te.sum(), 7), 1e-6); P[:, m.classes_] = m.predict_proba(sc.transform(Xte))
        LRP[name][te] = P / P.sum(1, keepdims=True)
    print(f"fold {f}: logistic regression done", flush=True)
assert all(not np.isnan(v).any() for v in LRP.values())
np.savez(f"{OUT_DIR}/g19_lr_oof_probs.npz", **{k.replace('-', '_'): v for k, v in LRP.items()})
"""),
    ("markdown", r"""
## Pair AUCs and the fixed decision rules
"""),
    ("code", r"""
import itertools
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

cnt = np.bincount(y_all, minlength=7)
PAIRS = list(itertools.combinations(range(7), 2))
ELIG = [(a, b) for a, b in PAIRS if cnt[a] >= MIN_CLASS and cnt[b] >= MIN_CLASS]
groups = [np.where(src == e)[0] for e in np.unique(src)]
rng = np.random.default_rng(0)


def pair_auc(P, a, b, idx):
    m = idx[np.isin(y_all[idx], [a, b])]
    if len(np.unique(y_all[m])) < 2:
        return np.nan
    return roc_auc_score(y_all[m] == a, np.log(P[m, a] + 1e-9) - np.log(P[m, b] + 1e-9))


def draws():
    for _ in range(N_BOOT):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), \
              rng.integers(0, len(SEEDS), len(SEEDS))


ALL = np.arange(N)
ENS = {k: OOF[k].mean(0) for k in INFOSETS}
B = list(draws())
ROWS = []
for a, b in PAIRS:
    r = {'pair': f'{EMO[a]}/{EMO[b]}', 'n_a': int(cnt[a]), 'n_b': int(cnt[b]), 'eligible': (a, b) in ELIG}
    for k in INFOSETS:
        r[f'rn_{k}'] = pair_auc(ENS[k], a, b, ALL)
        r[f'lr_{k}'] = pair_auc(LRP[k], a, b, ALL)
    bd, b12, blr = [], [], []
    for idx, sd in B:
        p12, p13 = OOF['I-II'][sd].mean(0), OOF['I-III'][sd].mean(0)
        u12 = pair_auc(p12, a, b, idx)
        bd.append(pair_auc(p13, a, b, idx) - u12); b12.append(u12)
        blr.append(pair_auc(LRP['I-III'], a, b, idx) - pair_auc(LRP['I-II'], a, b, idx))
    r['rn_delta'] = r['rn_I-III'] - r['rn_I-II']
    r['rn_delta_lo'], r['rn_delta_hi'] = np.nanpercentile(bd, [2.5, 97.5])
    r['rn_I-II_lo'] = np.nanpercentile(b12, 2.5)
    r['lr_delta'] = r['lr_I-III'] - r['lr_I-II']
    r['lr_delta_lo'], r['lr_delta_hi'] = np.nanpercentile(blr, [2.5, 97.5])
    ROWS.append(r)
T = pd.DataFrame(ROWS).sort_values('rn_I-III', ascending=False)
T.to_csv(f"{OUT_DIR}/g19_pair_auc.csv", index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    print(T[['pair', 'n_a', 'n_b', 'eligible', 'rn_III', 'rn_I-II', 'rn_I-III', 'rn_delta', 'rn_delta_lo', 'rn_delta_hi',
             'lr_I-II', 'lr_I-III', 'lr_delta', 'lr_delta_lo', 'lr_delta_hi']].round(3).to_string(index=False))

E_ = T[T.eligible]
rho = spearmanr(E_['rn_I-III'], E_['lr_I-III']).correlation
rho12 = spearmanr(E_['rn_I-II'], E_['lr_I-II']).correlation
up = int((E_['rn_delta_lo'] > 0).sum())
flat = int(((E_['rn_I-II_lo'] > 0.5) & (E_['rn_delta_lo'] <= 0) & (E_['rn_delta_hi'] >= 0)).sum())
c1, c2 = rho >= 0.7, (up >= 2 and flat >= 2)
dec = 'CONTINUE' if c1 and c2 else 'STOP'
print(f"\neligible pairs: {len(E_)} | rule 1: Spearman(RoleNet, LR) at I-III = {rho:.2f} (I-II: {rho12:.2f}) -> {c1}")
print(f"rule 2: pairs with Δ CI > 0: {up} | pairs above chance at I-II with Δ CI incl. 0: {flat} -> {c2}")
print(f"== G19 decision (fixed rule): {dec} ==")
json.dump({'spearman_I_III': float(rho), 'spearman_I_II': float(rho12), 'pairs_up': up, 'pairs_flat': flat,
           'rule1': bool(c1), 'rule2': bool(c2), 'decision': dec}, open(f"{OUT_DIR}/g19_decision.json", 'w'), indent=1)
"""),
]


G20 = [
    ("markdown", r"""
# G20 — Is there forecast value in text × audio-visual non-additivity? (5-fold CV, train+val; test untouched)

**Question.** Does B's clip-IV emotion depend on how what is said (X) combines with how it is expressed (Z), beyond
what each source gives on its own? This is a gate before any method work. It measures the **forecast value of
non-additivity in the logits**. It is not a PID synergy estimate, and passing it would not show that a model has
learned pragmatics.

* X = text features of clips I–III (taken before the sum that forms the speech token).
* Z = audio, voice (who-speaks cues), faces and scene of clips I–III.

**Models (two families, the same 5 episode folds as G8b–G19).**

| Arm | Logits | Role |
|---|---|---|
| `Additive` | a_y(X) + b_y(Z), both branches trained **jointly** by CE on the sum, no exchange of information | main control |
| `Local` | Additive + Σ_t r_y(x_t, a_t): text × audio of the **same clip**, r shared over clips | **main test** |
| `Window` | any X–Z interaction in the window | secondary |
| `LateFusion` | sum of the log-probabilities of two separately trained single-source models | control, reported only |

* Neural family: the X branch is a small Transformer over the 3 text tokens. The Z branch is RoleNet without the text
  term in its speech tokens (aux heads and modality dropout kept; they see Z only). r is a pure bilinear form
  (x̃_t ⊙ ã_t → 7 logits, no bias terms, zero at init; set to 0 for clips without audio). `Window` = full RoleNet
  (G14 `Full`). The same optimiser, early stopping (dev UAR on 5 held-out training episodes) and seeds are used for
  every arm. Each seed's logits are temperature-scaled on the early-stopping episodes; the ensemble is the mean of the
  calibrated probabilities of the 10 seeds.
* Logistic family: multinomial LR on the concatenated features (X: text PCA-32 per clip; Z: audio PCA-32, scene
  PCA-32, mean HSEmotion + face flag, voice cues, audio-found flag per clip). `Local` adds Σ_t vec(x̃_t ã_tᵀ) with
  x̃, ã = PCA-8 of text and audio shared over clips. `Window` adds the outer product of PCA-8 of the whole X block and
  PCA-8 of the whole Z block. C is chosen per arm by inner 5-fold source-grouped CV on NLL; the temperature is fitted
  on the same inner out-of-fold logits. All PCA, scaling, C choice, early stopping and calibration stay inside the
  training part of each outer fold.

**Decision (fixed before running).** Δ_NLL = NLL(Additive) − NLL(Local) on the pooled out-of-fold predictions;
positive means the interaction model is better. 95% CI: neural by a two-level bootstrap over seeds and source folders;
LR by a bootstrap over source folders; paired draws.

| Main test (Local variant only) | Decision |
|---|---|
| CI of Δ_NLL above 0 in **both** families | a method pilot is allowed |
| only one family | stop developing this direction on Hi-EF |
| neither family | stop developing this direction on Hi-EF |

A failed test means *no sufficiently strong evidence with these data and model classes*; it does **not** show that
the true distribution has no interaction. The `Window` variant, UAR and `LateFusion` are reported but cannot replace
the main criterion; the two families are compared only on the same variant.

**T2 (EMAP, Hessel & Lee 2020) on the logits of full RoleNet.** The projection
f̂(x_i, z_i) = mean_j f(x_i, z_j) + mean_j f(x_j, z_i) − mean_jk f(x_j, z_k) is computed over the cross-pairs inside
each evaluation fold (and inside the early-stopping episodes, for its own temperature). The cross-pairs only serve the
projection of the model; they are not labelled counterfactual conversations. If EMAP barely lowers the score, the
reading is: *no added forecast value of RoleNet's non-additive part has been seen*. If the main test passes but EMAP
shows no drop, the pilot is still allowed, but the gains of full RoleNet are not attributed to interaction.
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # as G14 / G19")
        .split("ARMS = [")[0] + """ARMS = [("Full", 'tok', BASE)]          # only so that the shared cells run; the G20 arms are defined below
ZCFG = {**BASE, 'speech_parts': ('audio', 'voice')}      # RoleNet without the text term = the Z branch
N_BOOT, PCA_K, PROD_K, R_DIM = 1000, 32, 8, 32
LR_CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
EXPERIMENTS = [a for a in ARMS if a[1] == 'role']
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8], G13[9], G13[10],
    ("markdown", r"""
## G20 models: additive, local interaction, single-source branches; EMAP
"""),
    ("code", r"""
from scipy.optimize import minimize_scalar


class XBranch(nn.Module):
    # text only: query + 3 text tokens (one per clip) -> the RoleNet Transformer -> 7 logits
    def __init__(self, d=RN['D']):
        super().__init__()
        self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.clip_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, RN['heads'], 4 * d, RN['dropout'], batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, RN['layers'], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))

    def forward(self, ix, train=False):
        toks = torch.cat([self.query.expand(len(ix), -1, -1), self.text(TXT[ix]) + self.clip_emb], 1)
        return self.head(self.enc(toks)[:, 0]), {}


class LocalR(nn.Module):
    # r_y(x_t, a_t) = W_y (P x_t ⊙ Q a_t): pure bilinear text x audio term of one clip, shared over clips.
    # No affine LayerNorm and no biases, so r contains no additive part; zero at init; 0 for clips without audio.
    def __init__(self, m=R_DIM):
        super().__init__()
        self.px = nn.Sequential(nn.LayerNorm(512, elementwise_affine=False), nn.Linear(512, m, bias=False))
        self.pa = nn.Sequential(nn.LayerNorm(527, elementwise_affine=False), nn.Linear(527, m, bias=False))
        self.drop, self.out = nn.Dropout(0.3), nn.Linear(m, 7, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, ix):
        r = self.out(self.drop(self.px(TXT[ix]) * self.pa(AUD[ix])))              # [B, 3, 7]
        return (r * AFD[ix].unsqueeze(-1)).sum(1)


class AdditiveNet(nn.Module):
    # a_y(X) + b_y(Z) [+ sum_t r_y(x_t, a_t)]; the branches exchange no information and are trained jointly on the sum
    def __init__(self, local=False):
        super().__init__()
        self.bx, self.bz = XBranch(), RoleNetTok(ZCFG)
        self.r = LocalR() if local else None

    def forward(self, ix, train=False):
        lx, _ = self.bx(ix, train)
        lz, aux = self.bz(ix, train)                                              # aux heads of the Z branch see Z only
        l = lx + lz
        if self.r is not None:
            l = l + self.r(ix)
        return l, aux


MAKE.update({'add': lambda cfg: AdditiveNet(False), 'local': lambda cfg: AdditiveNet(True),
             'xonly': lambda cfg: XBranch(), 'zonly': lambda cfg: RoleNetTok(ZCFG)})
for k in ('add', 'local', 'xonly', 'zonly'):
    HP[k] = HP['role']
G20_ARMS = [("Additive", 'add', None), ("Local", 'local', None), ("Window", 'tok', BASE),
            ("X-only", 'xonly', None), ("Z-only", 'zonly', None)]
print("parameters:", {n: f"{sum(p.numel() for p in MAKE[k](c).parameters() if p.requires_grad) / 1e6:.3f}M"
                      for n, k, c in G20_ARMS})


def fit_model(kind, cfg, tr, dev, seed):
    # identical to train_eval (optimiser, aux losses, early stopping on dev UAR) but returns the trained model
    seed_all(seed)
    hp = HP[kind]
    model = MAKE[kind](cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    y_dev = YB[dev].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(hp['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), hp['batch']):
            j = perm[i:i + hp['batch']]
            logits, aux = model(j, train=True)
            loss = F.cross_entropy(logits, YB[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        u = war_uar(predict(model, dev).argmax(1), y_dev, 7)[1]
        if u > best:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def logits_of(model, ix, bs=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(ix[i:i + bs])[0] for i in range(0, len(ix), bs)]).cpu().numpy().astype(np.float64)


def emap(model, ix, bs=512):
    # EMAP on logits: mean_j f(x_i, z_j) + mean_j f(x_j, z_i) - mean_jk f(x_j, z_k) over the cross-pairs of ix.
    # X (text) is swapped by pointing the global TXT to a copy whose rows ix carry the text of rows ix[(a + s) % n].
    global TXT
    n = len(ix)
    Mx, Mz = torch.zeros(n, 7, device=DEVICE), torch.zeros(n, 7, device=DEVICE)
    TXT0, TXT2 = TXT, TXT.clone()
    model.eval()
    try:
        TXT = TXT2
        with torch.no_grad():
            for s in range(n):
                sh = (torch.arange(n, device=DEVICE) + s) % n
                TXT2[ix] = TXT0[ix[sh]]
                L = torch.cat([model(ix[i:i + bs])[0] for i in range(0, n, bs)])   # row a: f(x_{sh[a]}, z_a)
                Mz += L
                Mx.index_add_(0, sh, L)
    finally:
        TXT = TXT0
    tot = Mz.sum(0, keepdim=True) / n ** 2
    return ((Mx + Mz) / n - tot).cpu().numpy().astype(np.float64)


def softmax_np(L):
    L = L - L.max(1, keepdims=True)
    E_ = np.exp(L)
    return E_ / E_.sum(1, keepdims=True)


def nll(P, y):
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def fit_T(L, y):
    return minimize_scalar(lambda t: nll(softmax_np(L / t), y), bounds=(0.05, 20), method='bounded').x


def log_softmax_np(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))
"""),
    ("markdown", r"""
## Neural family: 5-fold episode CV (same folds, early-stopping episodes and seeds as G8b–G19)
"""),
    ("code", r"""
import re

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)

y_all = DEV.yB.values
src = DEV.source_folder.values
NAMES = [a[0] for a in G20_ARMS] + ['LateFusion', 'Window-EMAP']
OOF = {k: np.full((len(SEEDS), N, 7), np.nan, np.float32) for k in NAMES}     # calibrated probabilities
log, temps = [], []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    y_dev, y_te = y_all[dev_rows], y_all[te_rows]
    for si, seed in enumerate(SEEDS):
        LG = {}
        for name, kind, cfg in G20_ARMS:
            model, sel = fit_model(kind, cfg, tr, dev, seed + 1000 * f)
            LG[name] = (logits_of(model, dev), logits_of(model, te))
            if name == 'Window':
                LG['Window-EMAP'] = (emap(model, dev), emap(model, te))
            if name == 'Additive' and si == 0:
                err = np.abs(emap(model, te) - LG[name][1]).max()
                print(f"  EMAP sanity check on the additive model (should be ~0): max |EMAP - f| = {err:.2e}")
                assert err < 1e-2, "EMAP does not reproduce an additive model"
            log.append({'fold': f, 'seed': seed, 'arm': name, 'sel_UAR': sel})
            del model
            torch.cuda.empty_cache()
        LG['LateFusion'] = tuple(log_softmax_np(LG['X-only'][i]) + log_softmax_np(LG['Z-only'][i]) for i in (0, 1))
        msg = []
        for k in NAMES:
            Ld, Lt = LG[k]
            t_ = fit_T(Ld, y_dev)
            OOF[k][si, te_rows] = softmax_np(Lt / t_)
            temps.append({'fold': f, 'seed': seed, 'arm': k, 'T': t_})
            msg.append(f"{k} {nll(OOF[k][si, te_rows], y_te):.3f}")
        print(f"fold {f} seed {seed}: NLL " + " | ".join(msg) + f" | {(time.time() - t0) / 60:.1f} min", flush=True)

assert all(not np.isnan(v).any() for v in OOF.values())
safe = lambda s: re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g20_fold_seed_log.csv", index=False)
pd.DataFrame(temps).to_csv(f"{OUT_DIR}/g20_temperatures.csv", index=False)
np.savez(f"{OUT_DIR}/g20_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         **{safe(k): v for k, v in OOF.items()})
print("saved g20_oof_probs.npz, g20_fold_seed_log.csv, g20_temperatures.csv")
"""),
    ("markdown", r"""
## Logistic family (same outer folds; C and temperature chosen inside each training part)
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA

EXPR = np.zeros((N, 3, 11), np.float32)
for n, row in enumerate(DEV.itertuples()):
    for k, c in enumerate((row.clip1, row.clip2, row.clip3)):
        if len(FB[c]):
            EXPR[n, k, :10] = FB[c][:, :10].mean(0); EXPR[n, k, 10] = 1.0
TXTn, AUDn, SCNn = TXT.cpu().numpy(), AUD.cpu().numpy(), SCN.cpu().numpy()
VOIn, AFDn = VOI.cpu().numpy(), AFD.cpu().numpy()


def pca_scores(M, tr, te, k, fit_mask=None):
    rows = tr if fit_mask is None else tr[fit_mask[tr]]
    p = _PCA(min(k, len(rows) - 1, M.shape[1]), random_state=0).fit(M[rows])
    return p.transform(M[tr]), p.transform(M[te])


def design(tr, te):
    # all transforms fitted on rows tr only; returns {block: (train matrix, eval matrix)}
    X_tr, X_te, Z_tr, Z_te = [], [], [], []
    for k in range(3):
        a, b = pca_scores(TXTn[:, k], tr, te, PCA_K); X_tr.append(a); X_te.append(b)
        a, b = pca_scores(AUDn[:, k], tr, te, PCA_K, AFDn[:, k] > 0)
        Z_tr.append(a * AFDn[tr, k:k + 1]); Z_te.append(b * AFDn[te, k:k + 1])
        a, b = pca_scores(SCNn[:, k], tr, te, PCA_K); Z_tr.append(a); Z_te.append(b)
        for M in (EXPR[:, k], VOIn[:, k], AFDn[:, k:k + 1]):
            Z_tr.append(M[tr]); Z_te.append(M[te])
    D = {'X': (np.concatenate(X_tr, 1), np.concatenate(X_te, 1)), 'Z': (np.concatenate(Z_tr, 1), np.concatenate(Z_te, 1))}
    # local: sum_t vec(x~_t a~_t^T), x~ / a~ = PCA of text / audio fitted on the clips of the training rows (shared)
    tx = _PCA(PROD_K, random_state=0).fit(TXTn[tr].reshape(-1, TXTn.shape[-1]))
    am = AFDn[tr].reshape(-1) > 0
    ta = _PCA(PROD_K, random_state=0).fit(AUDn[tr].reshape(-1, AUDn.shape[-1])[am])

    def local(rows):
        out = np.zeros((len(rows), PROD_K * PROD_K))
        for k in range(3):
            xs, as_ = tx.transform(TXTn[rows, k]), ta.transform(AUDn[rows, k]) * AFDn[rows, k:k + 1]
            out += np.einsum('ni,nj->nij', xs, as_).reshape(len(rows), -1)
        return out
    D['local'] = (local(tr), local(te))
    # window: outer product of PCA-8 of the whole (standardised) X block and of the whole Z block
    W = []
    for blk in ('X', 'Z'):
        s = StandardScaler().fit(D[blk][0])
        p = _PCA(PROD_K, random_state=0).fit(s.transform(D[blk][0]))
        W.append((p.transform(s.transform(D[blk][0])), p.transform(s.transform(D[blk][1]))))
    D['window'] = tuple(np.einsum('ni,nj->nij', W[0][i], W[1][i]).reshape(len(W[0][i]), -1) for i in (0, 1))
    return D


LR_ARMS = {'Additive': ('X', 'Z'), 'Local': ('X', 'Z', 'local'), 'Window': ('X', 'Z', 'window')}


def lr_logits(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    L = np.full((len(Xte), 7), np.nan)
    L[:, m.classes_] = m.decision_function(sc.transform(Xte))
    return np.where(np.isnan(L), np.nanmin(L, 1, keepdims=True) - 10, L)     # a class absent from training: very unlikely


LRP = {k: np.full((N, 7), np.nan) for k in LR_ARMS}
lr_log = []
for f in range(N_OUTER):
    tr, te = np.where(fold_of_row != f)[0], np.where(fold_of_row == f)[0]
    inner = {a: {C: np.zeros((len(tr), 7)) for C in LR_CS} for a in LR_ARMS}
    for ia, ib in GroupKFold(5).split(tr, groups=src[tr]):
        D = design(tr[ia], tr[ib])
        for a, blocks in LR_ARMS.items():
            Xa, Xb = (np.concatenate([D[b][i] for b in blocks], 1) for i in (0, 1))
            for C in LR_CS:
                inner[a][C][ib] = lr_logits(Xa, y_all[tr[ia]], Xb, C)
    D = design(tr, te)
    for a, blocks in LR_ARMS.items():
        scores = {C: nll(softmax_np(inner[a][C]), y_all[tr]) for C in LR_CS}
        C = min(scores, key=scores.get)
        t_ = fit_T(inner[a][C], y_all[tr])
        Xa, Xb = (np.concatenate([D[b][i] for b in blocks], 1) for i in (0, 1))
        LRP[a][te] = softmax_np(lr_logits(Xa, y_all[tr], Xb, C) / t_)
        lr_log.append({'fold': f, 'arm': a, 'C': C, 'T': t_, 'n_features': Xa.shape[1], 'inner_nll': scores[C]})
        print(f"fold {f} LR {a:<8}: C {C} | T {t_:.2f} | {Xa.shape[1]} features | NLL {nll(LRP[a][te], y_all[te]):.3f}",
              flush=True)
assert all(not np.isnan(v).any() for v in LRP.values())
pd.DataFrame(lr_log).to_csv(f"{OUT_DIR}/g20_lr_log.csv", index=False)
np.savez(f"{OUT_DIR}/g20_lr_oof_probs.npz", **LRP)
"""),
    ("markdown", r"""
## Results and the fixed decision
"""),
    ("code", r"""
groups = [np.where(src == e)[0] for e in np.unique(src)]
rng = np.random.default_rng(0)
S_ = len(SEEDS)
DRAWS = [(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), rng.integers(0, S_, S_))
         for _ in range(N_BOOT)]
ALL = np.arange(N)


def uar(P, idx):
    return war_uar(P[idx].argmax(1), y_all[idx], 7)[1]


def nn_stat(arm, idx, sd, fn):
    return fn(OOF[arm][sd].mean(0), idx) if fn is uar else nll(OOF[arm][sd].mean(0)[idx], y_all[idx])


def delta(fam, a, b, metric):
    # metric(a) - metric(b) on the pooled OOF predictions, with a paired bootstrap (two-level for the neural family)
    if fam == 'neural':
        f = lambda arm, idx, sd: nn_stat(arm, idx, sd, uar if metric == 'UAR' else None)
        pt = f(a, ALL, np.arange(S_)) - f(b, ALL, np.arange(S_))
        bs = [f(a, idx, sd) - f(b, idx, sd) for idx, sd in DRAWS]
    else:
        f = (lambda arm, idx: uar(LRP[arm], idx)) if metric == 'UAR' else (lambda arm, idx: nll(LRP[arm][idx], y_all[idx]))
        pt = f(a, ALL) - f(b, ALL)
        bs = [f(a, idx) - f(b, idx) for idx, _ in DRAWS]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi


print(f"== pooled out-of-fold scores ({N} MCIS, {len(EPS)} episodes; neural = mean of {S_} calibrated seeds) ==")
for k in NAMES:
    P = OOF[k].mean(0)
    print(f"  neural {k:<12} NLL {nll(P, y_all):.4f} | UAR {uar(P, ALL):5.2f}")
for k in LR_ARMS:
    print(f"  LR     {k:<12} NLL {nll(LRP[k], y_all):.4f} | UAR {uar(LRP[k], ALL):5.2f}")

ROWS = []
def row(test, fam, a, b, note):
    for metric in ('NLL', 'UAR'):
        pt, lo, hi = delta(fam, a, b, metric)
        if metric == 'UAR':                         # report UAR as b - a so that > 0 also means "b is better"
            pt, lo, hi = -pt, -hi, -lo
        ROWS.append({'test': test, 'family': fam, 'contrast': f"{a} vs {b}", 'metric': metric,
                     'delta': pt, 'lo': lo, 'hi': hi, 'note': note})
        print(f"  [{test}] {fam:<6} {metric}: Δ = {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({note})")


print("\n== main test: Δ_NLL = NLL(Additive) − NLL(Local); > 0 means the local interaction model is better ==")
for fam in ('neural', 'LR'):
    row('main', fam, 'Additive', 'Local', 'decision metric: NLL; UAR secondary')
print("\n== secondary: window variant ==")
for fam in ('neural', 'LR'):
    row('window', fam, 'Additive', 'Window', 'secondary')
print("\n== control: late fusion of separately trained single-source models (neural) ==")
row('control', 'neural', 'LateFusion', 'Additive', 'Δ > 0: joint additive training beats late fusion')
print("\n== T2: EMAP projection of full RoleNet; Δ = score(EMAP) − score(RoleNet) as NLL(EMAP) − NLL(RoleNet) ==")
row('EMAP', 'neural', 'Window-EMAP', 'Window', 'Δ > 0: the non-additive part of RoleNet adds forecast value')

R = pd.DataFrame(ROWS)
R.to_csv(f"{OUT_DIR}/g20_summary.csv", index=False)
main = R[(R.test == 'main') & (R.metric == 'NLL')].set_index('family')
passed = {fam: bool(main.loc[fam, 'lo'] > 0) for fam in ('neural', 'LR')}
n_pass = sum(passed.values())
dec = 'PILOT ALLOWED' if n_pass == 2 else 'STOP'
emap_row = R[(R.test == 'EMAP') & (R.metric == 'NLL')].iloc[0]
emap_drop = bool(emap_row.lo > 0)
print(f"\nmain test passed: neural {passed['neural']} | LR {passed['LR']}")
print(f"== G20 decision (fixed rule): {dec} ==")
if dec == 'STOP':
    print("   reading: no sufficiently strong evidence with these data and model classes (this does not show r = 0).")
if not emap_drop:
    print("   EMAP: no added forecast value of RoleNet's non-additive part has been seen"
          + (" -> gains of full RoleNet are not attributed to interaction." if dec != 'STOP' else "."))
json.dump({'decision': dec, 'main_pass': passed,
           'main': {fam: {k: float(main.loc[fam, k]) for k in ('delta', 'lo', 'hi')} for fam in ('neural', 'LR')},
           'emap_nll_delta': [float(emap_row.delta), float(emap_row.lo), float(emap_row.hi)], 'emap_drop': emap_drop},
          open(f"{OUT_DIR}/g20_decision.json", 'w'), indent=1)
"""),
]


G23 = [
    ("markdown", r"""
# G23 — Are the negative emotions separable at all? Recognition vs forecasting (5-fold CV, train+val; test untouched)

**Why.** The only failure that survives every model and check so far: B's clip-IV negative emotions are hardly
separable by any forecaster (pair AUC angry/sad, angry/disgust, disgust/sad ≈ 0.56–0.62 for RoleNet and LR, with or
without clip III, also on certain labels only). Before treating this as a forecasting problem, we must know whether
the same features can separate these emotions when the moment itself is observed. If they cannot, the failure is a
representation limit (an engineering issue), not a forecasting gap.

**Tasks** (same 2,421 train+val MCIS rows, same 5 episode folds as G8b–G20, same model class and features):

| Task | Input | Label | Role |
|---|---|---|---|
| `FC` | clips I–III | B's emotion at IV | forecasting (main) |
| `RB` | clip IV | B's emotion at IV | recognition of the target at the target moment (diagnostic only: clip IV is an input here and nowhere else) |
| `RA` | clip III | A's emotion at III | recognition of a speaker in a context clip |
| `FC+expr`, `RA+expr` | as above + mean HSEmotion per clip (clips I–III only) | | secondary |
| `RB-face`, `RB-text`, `RB-audio`, `RB-scene` | one stream of clip IV | B at IV | descriptive: which stream separates |

Features per clip (`hi-ef-features-v2`): CLIP face (mean over valid frames), CLIP whole frame, CLIP text, AudioSet audio
(zero when no audio), each reduced to 32 dims by PCA fitted on the training rows; plus audio-found and face-present
flags. Model: multinomial logistic regression on standardised features; C chosen per task and fold by inner 5-fold
source-grouped CV on NLL.

**Measure.** Pair AUC of the logit difference (as G19). `NEG` = mean over angry/sad, angry/disgust, disgust/sad.
`VAL` (positive control) = mean over angry/happy, happy/sad, disgust/happy. 95% CIs by a bootstrap over source folders
(1,000 draws, paired across tasks).

**Reading rules (fixed before running).**
1. *Validity*: the CI lower bound of `VAL(RB)` must be > 0.75; otherwise the recognition set-up is not informative →
   **INVALID**.
2. **PERCEPTION LIMIT** if the CI upper bound of `NEG(RB)` < 0.65: even the target's own moment does not separate the
   negative emotions with these features. The forecasting failure is then a representation limit; no
   forecasting-specific gap is claimed from it.
3. **FORECASTING-SPECIFIC** if the CI lower bound of `NEG(RB)` ≥ 0.70 **and** the CI lower bound of
   Δ = `NEG(RB)` − `NEG(FC)` > 0.05: the moment separates them, the context does not.
4. Otherwise **INTERMEDIATE**: reported, no gap claimed.

`RA`, the `+expr` arms and the per-stream arms are descriptive. A FORECASTING-SPECIFIC result says that, with these
features, the distinction appears only at clip IV; it does not by itself show why (e.g. appraisal of the event).

Note: with uninformative features, out-of-fold pair AUCs can fall below 0.5 because each fold's training prior is
shifted against its evaluation rows (seen in the dry run on random features). `FC` and `RB` share labels, rows and
folds, so this shift affects both in the same way.
"""),
    ("code", G13[1][1].split("ARMS = [")[0] + """ARMS = []                       # no neural arms in G23
EXPERIMENTS = []
N_BOOT, PCA_K = 1000, 32
LR_CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
NEG_PAIRS = [('angry', 'sad'), ('angry', 'disgust'), ('disgust', 'sad')]
VAL_PAIRS = [('angry', 'happy'), ('happy', 'sad'), ('disgust', 'happy')]
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8],
    ("markdown", r"""
## Features, folds and the logistic regressions
"""),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)

has4 = DEV.clip4.isin(CIDX).values
print(f"clip IV features available for {has4.sum()} / {N} MCIS")
ROWS = np.where(has4)[0]                       # all tasks use the same rows (paired)
yB_all, yA_all, src = DEV.yB.values, DEV.yA.values, DEV.source_folder.values


def clip_blocks(clips):
    out = {k: np.zeros((len(clips), d), np.float32) for k, d in (('face', 512), ('scene', 512), ('text', 512),
                                                                 ('audio', 527), ('flags', 2))}
    ok = np.array([c in CIDX for c in clips])
    ix = torch.tensor([CIDX[c] for c in np.asarray(clips)[ok]], device=DEVICE)
    fm = FEAT['fmask'][ix].unsqueeze(-1).float()
    af = FEAT['afound'][ix].float().unsqueeze(-1)
    out['face'][ok] = ((FEAT['face'][ix] * fm).sum(1) / fm.sum(1).clamp(min=1)).cpu().numpy()
    out['scene'][ok] = FEAT['ori'][ix].mean(1).cpu().numpy()
    out['text'][ok] = FEAT['text'][ix].cpu().numpy()
    out['audio'][ok] = (F.normalize(FEAT['audio'][ix], dim=-1) * af).cpu().numpy()
    out['flags'][ok] = torch.cat([af, (fm.sum(1) > 0).float()], 1).cpu().numpy()
    return out


BLK = {k: clip_blocks(DEV[f'clip{k}'].values) for k in (1, 2, 3, 4)}
EXPR = np.zeros((N, 3, 11), np.float32)
for n, row in enumerate(DEV.itertuples()):
    for k, c in enumerate((row.clip1, row.clip2, row.clip3)):
        if len(FB[c]):
            EXPR[n, k, :10] = FB[c][:, :10].mean(0); EXPR[n, k, 10] = 1.0

V2 = ['face', 'scene', 'text', 'audio', 'flags']
TASKS = {
    'FC':       ([(k, m) for k in (1, 2, 3) for m in V2], yB_all),
    'RB':       ([(4, m) for m in V2], yB_all),
    'RA':       ([(3, m) for m in V2], yA_all),
    'FC+expr':  ([(k, m) for k in (1, 2, 3) for m in V2 + ['expr']], yB_all),
    'RA+expr':  ([(3, m) for m in V2 + ['expr']], yA_all),
    'RB-face':  ([(4, 'face'), (4, 'flags')], yB_all),
    'RB-text':  ([(4, 'text')], yB_all),
    'RB-audio': ([(4, 'audio'), (4, 'flags')], yB_all),
    'RB-scene': ([(4, 'scene')], yB_all),
}


def design(spec, tr, te):
    # all PCAs are fitted on the training rows tr only
    A_, B_ = [], []
    for pos, mod in spec:
        if mod == 'expr':
            M = EXPR[:, pos - 1]
        else:
            M = BLK[pos][mod]
        if mod in ('flags', 'expr'):
            A_.append(M[tr]); B_.append(M[te]); continue
        fit = tr[BLK[pos]['flags'][tr, 0] > 0] if mod == 'audio' else tr
        p = _PCA(min(PCA_K, len(fit) - 1, M.shape[1]), random_state=0).fit(M[fit])
        a, b = p.transform(M[tr]), p.transform(M[te])
        if mod == 'audio':
            a, b = a * BLK[pos]['flags'][tr, :1], b * BLK[pos]['flags'][te, :1]
        A_.append(a); B_.append(b)
    return np.concatenate(A_, 1), np.concatenate(B_, 1)


def softmax_np(L):
    L = L - L.max(1, keepdims=True); E_ = np.exp(L); return E_ / E_.sum(1, keepdims=True)


def nll(L, y):
    return float(-np.log(softmax_np(L)[np.arange(len(y)), y] + 1e-12).mean())


def lr_logits(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    L = np.full((len(Xte), 7), np.nan)
    L[:, m.classes_] = m.decision_function(sc.transform(Xte))
    return np.where(np.isnan(L), np.nanmin(L, 1, keepdims=True) - 10, L)


LOGIT = {t: np.full((N, 7), np.nan) for t in TASKS}
lr_log = []
t0 = time.time()
for f in range(N_OUTER):
    tr = ROWS[fold_of_row[ROWS] != f]
    te = ROWS[fold_of_row[ROWS] == f]
    for t, (spec, yy) in TASKS.items():
        inner = {C: np.zeros((len(tr), 7)) for C in LR_CS}
        for ia, ib in GroupKFold(5).split(tr, groups=src[tr]):
            Xa, Xb = design(spec, tr[ia], tr[ib])
            for C in LR_CS:
                inner[C][ib] = lr_logits(Xa, yy[tr[ia]], Xb, C)
        scores = {C: nll(inner[C], yy[tr]) for C in LR_CS}
        C = min(scores, key=scores.get)
        Xa, Xb = design(spec, tr, te)
        LOGIT[t][te] = lr_logits(Xa, yy[tr], Xb, C)
        lr_log.append({'fold': f, 'task': t, 'C': C, 'n_features': Xa.shape[1], 'inner_nll': scores[C]})
    print(f"fold {f} done | {(time.time() - t0) / 60:.1f} min", flush=True)
assert all(not np.isnan(LOGIT[t][ROWS]).any() for t in TASKS)
pd.DataFrame(lr_log).to_csv(f"{OUT_DIR}/g23_lr_log.csv", index=False)
np.savez(f"{OUT_DIR}/g23_oof_logits.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, yB=yB_all, yA=yA_all,
         src=src, rows=ROWS, **{t.replace('+', '_').replace('-', '_'): v for t, v in LOGIT.items()})
print("saved g23_oof_logits.npz and g23_lr_log.csv")
"""),
    ("markdown", r"""
## Pair AUCs and the fixed reading rules
"""),
    ("code", r"""
from sklearn.metrics import roc_auc_score

E2 = {e: i for i, e in enumerate(EMO)}


def pair_auc(t, a, b, idx):
    yy = TASKS[t][1]
    m = idx[np.isin(yy[idx], [E2[a], E2[b]])]
    if len(np.unique(yy[m])) < 2:
        return np.nan
    return roc_auc_score(yy[m] == E2[a], LOGIT[t][m, E2[a]] - LOGIT[t][m, E2[b]])


def score(t, idx):
    d = {f"{a}/{b}": pair_auc(t, a, b, idx) for a, b in NEG_PAIRS + VAL_PAIRS}
    d['NEG'] = np.nanmean([d[f"{a}/{b}"] for a, b in NEG_PAIRS])
    d['VAL'] = np.nanmean([d[f"{a}/{b}"] for a, b in VAL_PAIRS])
    return d


groups = [ROWS[src[ROWS] == e] for e in np.unique(src[ROWS])]
rng = np.random.default_rng(0)
DRAWS = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
PT = {t: score(t, ROWS) for t in TASKS}
BS = {t: [score(t, idx) for idx in DRAWS] for t in TASKS}

out = []
for t in TASKS:
    r = {'task': t, 'UAR': war_uar(LOGIT[t][ROWS].argmax(1), TASKS[t][1][ROWS], 7)[1]}
    for k in PT[t]:
        r[k] = PT[t][k]
        r[k + '_lo'], r[k + '_hi'] = np.nanpercentile([b[k] for b in BS[t]], [2.5, 97.5])
    out.append(r)
R = pd.DataFrame(out)
R.to_csv(f"{OUT_DIR}/g23_pair_auc.csv", index=False)
cols = ['task', 'UAR'] + [f"{a}/{b}" for a, b in NEG_PAIRS] + ['NEG', 'NEG_lo', 'NEG_hi'] + \
       [f"{a}/{b}" for a, b in VAL_PAIRS] + ['VAL', 'VAL_lo', 'VAL_hi']
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    print(R[cols].round(3).to_string(index=False))

d_pt = PT['RB']['NEG'] - PT['FC']['NEG']
d_lo, d_hi = np.nanpercentile([b1['NEG'] - b2['NEG'] for b1, b2 in zip(BS['RB'], BS['FC'])], [2.5, 97.5])
rb = R.set_index('task').loc['RB']
print(f"\nΔ = NEG(RB) − NEG(FC) = {d_pt:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]")
if rb.VAL_lo <= 0.75:
    dec = 'INVALID'
elif rb.NEG_hi < 0.65:
    dec = 'PERCEPTION LIMIT'
elif rb.NEG_lo >= 0.70 and d_lo > 0.05:
    dec = 'FORECASTING-SPECIFIC'
else:
    dec = 'INTERMEDIATE'
print(f"== G23 reading (fixed rule): {dec} ==")
json.dump({'decision': dec, 'NEG_RB': [float(rb.NEG), float(rb.NEG_lo), float(rb.NEG_hi)],
           'VAL_RB': [float(rb.VAL), float(rb.VAL_lo), float(rb.VAL_hi)],
           'NEG_FC': [float(PT['FC']['NEG'])], 'delta': [float(d_pt), float(d_lo), float(d_hi)],
           'n_rows': int(len(ROWS))}, open(f"{OUT_DIR}/g23_decision.json", 'w'), indent=1)
"""),
]


APP_DIMS = ['event_valence', 'caused_by_other', 'caused_by_responder', 'caused_by_circumstance', 'responder_control',
            'loss', 'norm_violation', 'unexpected', 'goal_obstruction', 'threat']

G24_LR_CELL = (G23[10][1]
    .replace("""TASKS = {
    'FC':       ([(k, m) for k in (1, 2, 3) for m in V2], yB_all),
    'RB':       ([(4, m) for m in V2], yB_all),
    'RA':       ([(3, m) for m in V2], yA_all),
    'FC+expr':  ([(k, m) for k in (1, 2, 3) for m in V2 + ['expr']], yB_all),
    'RA+expr':  ([(3, m) for m in V2 + ['expr']], yA_all),
    'RB-face':  ([(4, 'face'), (4, 'flags')], yB_all),
    'RB-text':  ([(4, 'text')], yB_all),
    'RB-audio': ([(4, 'audio'), (4, 'flags')], yB_all),
    'RB-scene': ([(4, 'scene')], yB_all),
}""", """FCSPEC = [(k, m) for k in (1, 2, 3) for m in V2]
TASKS = {
    'FC':       (FCSPEC, yB_all),
    'FC+app':   (FCSPEC + [(0, 'app')], yB_all),
    'APP':      ([(0, 'app')], yB_all),
    'FC+llm':   (FCSPEC + [(0, 'llm')], yB_all),
}""")
    .replace("""        if mod == 'expr':
            M = EXPR[:, pos - 1]
        else:
            M = BLK[pos][mod]
        if mod in ('flags', 'expr'):""", """        if mod == 'expr':
            M = EXPR[:, pos - 1]
        elif mod == 'app':
            M = APP
        elif mod == 'llm':
            M = LLMP
        else:
            M = BLK[pos][mod]
        if mod in ('flags', 'expr', 'app', 'llm'):""")
    .replace("has4 = DEV.clip4.isin(CIDX).values\nprint(f\"clip IV features available for {has4.sum()} / {N} MCIS\")\nROWS = np.where(has4)[0]                       # all tasks use the same rows (paired)",
             "ROWS = np.arange(N)                            # clip IV is not used in G24")
    .replace("BLK = {k: clip_blocks(DEV[f'clip{k}'].values) for k in (1, 2, 3, 4)}",
             "BLK = {k: clip_blocks(DEV[f'clip{k}'].values) for k in (1, 2, 3)}")
    .replace("g23_", "g24_"))
assert G24_LR_CELL.count("'FC+app'") == 1 and "elif mod == 'app'" in G24_LR_CELL and "ROWS = np.arange(N)" in G24_LR_CELL

G24 = [
    ("markdown", r"""
# G24 — Does appraisal of the conversation content forecast *which* negative emotion B reacts with? (5-fold CV, train+val; test untouched)

**Why.** Forecasters succeed mostly when B mirrors A (F14: accuracy 56.9% vs 27.7% on shifts) and hardly separate
angry / sad / disgust (G19, G23: NEG 0.56 from clips I–III vs 0.69 at clip IV, descriptive). Appraisal theory
predicts that these emotions differ in how the responder appraises the event: anger — caused by another person and
controllable; sadness — loss, low control; disgust — norm violation. That information is in *what was said*, which our
CLIP text features do not capture (G13, G20), and an LLM reading of the subtitles was never run (G1 not executed).

**Appraisal features (no labels).** An LLM reads subtitle lines I–III only (clip IV text is never sent) and rates, for
the person who will respond next, 10 appraisal dimensions in [0, 1] (valence in [−1, 1]): event valence, caused by
another person, caused by the responder, caused by circumstances, responder's control, loss, norm violation,
unexpectedness, goal obstruction, threat. A separate call asks for a direct forecast of the responder's emotion (control
arm). Responses are cached. Both calls are zero-shot and see no Hi-EF label.

**Arms** (same LR, folds, C selection and features as G23 `FC`): `FC` (CLIP/AudioSet features of I–III), `FC+app`,
`APP` (appraisal only), `FC+llm` (+ the LLM's direct 7-way forecast; control).

**Measure.** Pair AUC as G23; `NEG` = mean over angry/sad, angry/disgust, disgust/sad. **Shift subset** = MCIS whose
B label differs from A's clip-III label (gold labels used only to define the subset, never as inputs). 95% CI by a
bootstrap over source folders, paired across arms.

**Decision (fixed before running).**
* **PASS** if Δ_shift = NEG_shift(`FC+app`) − NEG_shift(`FC`) has CI lower bound > 0 **and** NEG_shift(`APP`) has CI
  lower bound > 0.5. Then appraisal read from the content carries shift-direction information that the current
  features lack → a method pilot on reaction (non-mirroring) forecasting is justified.
* Otherwise **STOP**: with this reading of the content, the direction of B's negative reaction is not forecastable
  from clips I–III.

Descriptive: NEG on all MCIS, UAR, `FC+llm` vs `FC+app` (whether a gain is specific to appraisal or to any LLM reading
of the text), and a contamination probe (does the LLM name the series?). Hi-EF comes from TV shows the LLM may know;
a PASS must be read together with that probe.
"""),
    ("code", "!pip install -q openai"),
    ("code", G23[1][1] + """
# ---- G24: LLM settings ----
MODEL = "gpt-4o-mini"          # any OpenAI chat model with JSON output
TEMPERATURE = 0.0              # None for models that reject the parameter
MAX_WORKERS = 8
CACHE_APP = f"{OUT_DIR}/g24_appraisal_{MODEL.replace('/', '_')}.jsonl"
CACHE_DIR = f"{OUT_DIR}/g24_direct_{MODEL.replace('/', '_')}.jsonl"
"""),
    G23[2], G23[3], G23[4], G23[5], G23[6], G23[7], G23[8],
    ("markdown", r"""
## LLM reading of subtitles I–III (appraisal and, separately, a direct forecast)
"""),
    ("code", r"""
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("OPENAI_API_KEY", UserSecretsClient().get_secret("OPENAI_API_KEY"))
except Exception:
    pass
client = OpenAI()
APP_DIMS = ['event_valence', 'caused_by_other', 'caused_by_responder', 'caused_by_circumstance',
            'responder_control', 'loss', 'norm_violation', 'unexpected', 'goal_obstruction', 'threat']
SYSTEM = ("You are an expert in appraisal theory of emotion, annotating TV drama dialogue. You only see subtitle "
          "text, which may contain transcription errors.")
CONTEXT = '''Three consecutive subtitle lines from a conversation in a TV drama. Lines [1] and [2] are earlier context
(speakers not given). Line [3] is spoken by person A. The next line (not shown) will be spoken by a different person,
the RESPONDER, who reacts to what has happened.
[1] {t1}
[2] {t2}
[3] (A) {t3}
'''
TASK_APP = CONTEXT + '''
Rate how the RESPONDER most likely appraises the situation at this moment. Do not name an emotion.
- event_valence: how good (+1) or bad (-1) the situation is for the responder
- caused_by_other: caused by A or another person (0-1)
- caused_by_responder: caused by the responder themselves (0-1)
- caused_by_circumstance: caused by circumstances / nobody (0-1)
- responder_control: how much the responder can still change or cope with it (0-1)
- loss: something valued is lost or irreversibly damaged (0-1)
- norm_violation: something offensive, immoral, or repulsive happened (0-1)
- unexpected: how unexpected it is for the responder (0-1)
- goal_obstruction: it blocks something the responder wants (0-1)
- threat: it is a danger or threat to the responder (0-1)
Return JSON only with these 10 keys and numeric values.'''
TASK_DIR = CONTEXT + '''
Which emotion will the RESPONDER most likely express in the next line?
Labels: angry, disgust, fear, happy, neutral, sad, surprise.
Return JSON only, a probability for every label (summing to 1): {{"angry": p, ...}}'''


def call(task, row, retries=5):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task.format(t1=row['t1'] or '(no text)', t2=row['t2'] or '(no text)',
                                                    t3=row['t3'] or '(no text)')}]
    kw = dict(model=MODEL, messages=msgs, response_format={"type": "json_object"})
    if TEMPERATURE is not None:
        kw["temperature"] = TEMPERATURE
    for attempt in range(retries):
        try:
            return json.loads(client.chat.completions.create(**kw).choices[0].message.content)
        except Exception as e:
            ERRORS.append(f"{type(e).__name__}: {e}")
            if "temperature" in str(e) and "temperature" in kw:
                kw.pop("temperature"); continue
            time.sleep(2 ** attempt)
    return None


def run_cache(task, path):
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line); cache[d['sample_id']] = d['response']
    todo = [r for r in DEV.to_dict('records') if r['sample_id'] not in cache]
    print(f"{os.path.basename(path)}: cached {len(cache)} | to query {len(todo)}", flush=True)
    with ThreadPoolExecutor(MAX_WORKERS) as pool, open(path, 'a') as f:
        futs = {pool.submit(call, task, r): r['sample_id'] for r in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=os.path.basename(path), mininterval=10):
            sid, resp = futs[fut], fut.result()
            if resp is not None:
                cache[sid] = resp; f.write(json.dumps({'sample_id': sid, 'response': resp}) + "\n")
    n_ok = sum(s in cache for s in DEV.sample_id)
    print(f"  responses {n_ok}/{N}")
    if n_ok < 0.95 * N:
        raise RuntimeError("More than 5% of items have no response; re-run this cell (cached items are skipped).")
    return cache


ERRORS = []
if call(TASK_APP, DEV.iloc[0].to_dict(), retries=2) is None:
    raise RuntimeError(f"Pre-flight call failed (Internet on? OPENAI_API_KEY secret? MODEL?). {ERRORS[-1:]}")
C_APP, C_DIR = run_cache(TASK_APP, CACHE_APP), run_cache(TASK_DIR, CACHE_DIR)
if ERRORS:
    print(f"{len(ERRORS)} API errors, e.g.", sorted(set(ERRORS))[:2])


def num(d, k, lo, hi):
    try:
        return float(np.clip(float(d[k]), lo, hi))
    except Exception:
        return np.nan


APP = np.array([[num(C_APP.get(s) or {}, k, -1 if k == 'event_valence' else 0, 1) for k in APP_DIMS]
                for s in DEV.sample_id], np.float32)
miss = np.isnan(APP)
print("missing appraisal values per dimension:", dict(zip(APP_DIMS, miss.sum(0))))
if (miss.mean(0) > 0.05).any():
    raise RuntimeError("An appraisal dimension is missing in more than 5% of responses; check the JSON keys.")
APP = np.where(miss, np.nanmean(APP, 0), APP)         # rare; filled with the column mean (no labels involved)


def to_probs(d):
    d = {str(k).strip().lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
    p = np.array([num(d, e, 0, 1) if e in d else 0.0 for e in EMO]); p = np.nan_to_num(p)
    return p / p.sum() if p.sum() > 0 else np.full(7, 1 / 7)


LLMP = np.stack([to_probs(C_DIR.get(s)) for s in DEV.sample_id]).astype(np.float32)
print(pd.DataFrame(APP, columns=APP_DIMS).describe().loc[['mean', 'std']].round(2).to_string())
"""),
    ("markdown", r"""
## Logistic regressions (same set-up as G23 `FC`)
"""),
    ("code", G24_LR_CELL),
    ("markdown", r"""
## Pair AUCs on all MCIS and on the shift subset; fixed decision
"""),
    ("code", r"""
from sklearn.metrics import roc_auc_score

E2 = {e: i for i, e in enumerate(EMO)}
SHIFT = yB_all != yA_all
print(f"shift subset: {SHIFT.sum()} / {N} MCIS")


def pair_auc(t, a, b, idx):
    m = idx[np.isin(yB_all[idx], [E2[a], E2[b]])]
    if len(np.unique(yB_all[m])) < 2:
        return np.nan
    return roc_auc_score(yB_all[m] == E2[a], LOGIT[t][m, E2[a]] - LOGIT[t][m, E2[b]])


def score(t, idx):
    d = {f"{a}/{b}": pair_auc(t, a, b, idx) for a, b in NEG_PAIRS}
    d['NEG'] = np.nanmean(list(d.values()))
    return d


groups = [np.where(src == e)[0] for e in np.unique(src)]
rng = np.random.default_rng(0)
DRAWS = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
SUB = {'all': lambda idx: idx, 'shift': lambda idx: idx[SHIFT[idx]]}
PT = {(t, s): score(t, f(np.arange(N))) for t in TASKS for s, f in SUB.items()}
BS = {(t, s): [score(t, f(idx)) for idx in DRAWS] for t in TASKS for s, f in SUB.items()}

rows = []
for (t, s), d in PT.items():
    r = {'task': t, 'subset': s, **d}
    r['NEG_lo'], r['NEG_hi'] = np.nanpercentile([b['NEG'] for b in BS[(t, s)]], [2.5, 97.5])
    m = np.arange(N) if s == 'all' else np.where(SHIFT)[0]
    r['UAR'] = war_uar(LOGIT[t][m].argmax(1), yB_all[m], 7)[1]
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(f"{OUT_DIR}/g24_pair_auc.csv", index=False)
print(R.round(3).to_string(index=False))


def delta(a, b, s):
    pt = PT[(a, s)]['NEG'] - PT[(b, s)]['NEG']
    lo, hi = np.nanpercentile([x['NEG'] - y['NEG'] for x, y in zip(BS[(a, s)], BS[(b, s)])], [2.5, 97.5])
    return pt, lo, hi


d_shift = delta('FC+app', 'FC', 'shift')
app_lo = R[(R.task == 'APP') & (R.subset == 'shift')].NEG_lo.iloc[0]
print(f"\nΔ_shift = NEG_shift(FC+app) − NEG_shift(FC) = {d_shift[0]:+.3f} [{d_shift[1]:+.3f}, {d_shift[2]:+.3f}]")
print(f"NEG_shift(APP) lower bound = {app_lo:.3f}")
for a, b, s in [('FC+app', 'FC', 'all'), ('FC+llm', 'FC', 'shift'), ('FC+app', 'FC+llm', 'shift')]:
    p, lo, hi = delta(a, b, s)
    print(f"  (descriptive) {a} − {b} [{s}]: {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")
dec = 'PASS' if d_shift[1] > 0 and app_lo > 0.5 else 'STOP'
print(f"== G24 decision (fixed rule): {dec} ==")
json.dump({'decision': dec, 'delta_shift': list(map(float, d_shift)), 'NEG_shift_APP_lo': float(app_lo),
           'model': MODEL}, open(f"{OUT_DIR}/g24_decision.json", 'w'), indent=1)
"""),
    ("markdown", r"""
## Contamination probe (for the limitations section)
"""),
    ("code", r"""
probe = DEV.sample(n=30, random_state=0)
names = []
for r in probe.to_dict('records'):
    q = (f"Which TV series are these subtitle lines from? Answer with the series name only, or 'unknown'.\n"
         f"[1] {r['t1']}\n[2] {r['t2']}\n[3] {r['t3']}")
    kw = dict(model=MODEL, messages=[{"role": "user", "content": q}])
    if TEMPERATURE is not None:
        kw["temperature"] = TEMPERATURE
    try:
        names.append(client.chat.completions.create(**kw).choices[0].message.content.strip())
    except Exception as e:
        names.append(f"error: {type(e).__name__}")
print(pd.Series(names).value_counts().head(10).to_string())
json.dump(names, open(f"{OUT_DIR}/g24_contamination_probe.json", 'w'), indent=1)
"""),
]


G25 = [
    ("markdown", r"""
# G25 — Does the listener's face show B's ongoing state or B's reaction to A? (train+val, analysis only; test untouched)

**Why.** B's own face (the listener L) is RoleNet's main extra signal (minus-L −2.04 UAR, 10 seeds), concentrated where
L is visible and on MCIS where B mirrors A (F14). Two readings:
* **H-state** — the face shows B's ongoing affective state, which persists into clip IV. Predictions: earlier
  observations of B are about as useful as the clip-III one; *changes* of B's expression add nothing beyond its level.
* **H-reaction** — the face shows B starting to react to A's turn. Prediction: the *change* of B's expression from
  clips I/II to clip III carries information about clip IV beyond the average level, above all where B does not mirror
  A (shift MCIS).
Earlier results lean to H-state (G16 T1/T2, G17, G15 G1b), but the change prediction was never tested.

**Data.** MCIS where L is visible in clip III **and** in clip I or II (same identity cluster). Per observation: mean
HSEmotion probabilities (8 classes) + valence/arousal over at most 8 frames of L (`E3` for clip III, `E12` pooled over
clips I/II). Labels are never inputs; A's clip-III label is used only to define the shift subset (B ≠ A).

**Models** (multinomial LR, standardised, C chosen by inner source-grouped CV on NLL; outer folds = the G8b–G23 folds):
`NOW` = E3, `HIST` = E12, `AVG` = (E3 + E12)/2 (one pooled reading of a state), `SEP` = [E3, E12] (allows change).

**Rules (fixed before running; 95% CI = bootstrap over source folders, paired).**
1. *Validity*: NLL(prior) − NLL(`NOW`) on S has CI lower bound > 0, else **INVALID** (the listener features carry no
   information here).
2. **REACTION SIGNAL** if Δ = NLL(`AVG`) − NLL(`SEP`) on the **shift** MCIS of S has CI lower bound > 0.
3. Otherwise **NO REACTION SIGNAL DETECTED** (consistent with H-state; the CI of Δ is reported as the resolution).

Descriptive: Δ on all of S and on mirror MCIS; NLL(`HIST`) − NLL(`NOW`) (recency); agreement of L's argmax expression in
clip III with B's clip-IV label and with A's clip-III label, by mirror / shift; and, if `g14_oof_probs.npz` is
attached, the Spearman correlation between RoleNet's per-MCIS listener gain (log p Full − log p minus-L) and the size
of L's expression change.
"""),
    ("code", G13[1][1].split("ARMS = [")[0] + """ARMS = []
EXPERIMENTS = []
N_BOOT, MAX_OBS_FRAMES = 1000, 8
LR_CS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8],
    ("markdown", "## Listener expression per observation and the four models"),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
yB_all, yA_all, src = DEV.yB.values, DEV.yA.values, DEV.source_folder.values

# HSEmotion 8 classes (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise) -> the 7 Hi-EF labels
H2E = np.zeros((8, 7), np.float32)
for h, e in enumerate([0, 1, 1, 2, 3, 4, 5, 6]):
    H2E[h, e] = 1
rs = np.random.default_rng(0)


def obs(n, clips):
    rows = []
    for k in clips:
        c, idx = SLOT[(n, 1, k)]
        rows += [FB[c][j] for j in idx]
    if not rows:
        return None
    rows = np.stack(rows)
    if len(rows) > MAX_OBS_FRAMES:
        rows = rows[rs.choice(len(rows), MAX_OBS_FRAMES, replace=False)]
    return np.concatenate([rows[:, :8].mean(0), rows[:, 8:10].mean(0)])     # 8 probs + valence, arousal


E3 = np.full((N, 10), np.nan, np.float32); E12 = np.full((N, 10), np.nan, np.float32)
for n in range(N):
    a, b = obs(n, (2,)), obs(n, (0, 1))
    if a is not None and b is not None:
        E3[n], E12[n] = a, b
S = np.where(~np.isnan(E3[:, 0]))[0]
SHIFT = yB_all != yA_all
print(f"S (listener in III and in I/II): {len(S)} MCIS | shift {SHIFT[S].sum()} | mirror {(~SHIFT[S]).sum()}")
X = {'NOW': E3, 'HIST': E12, 'AVG': (E3 + E12) / 2, 'SEP': np.concatenate([E3, E12], 1)}


def lr_logp(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    L = np.full((len(Xte), 7), np.log(1e-6))
    L[:, m.classes_] = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-6, 1))
    return L - np.log(np.exp(L).sum(1, keepdims=True))


LOGP = {k: np.full((N, 7), np.nan) for k in list(X) + ['prior']}
clog = []
for f in range(N_OUTER):
    tr, te = S[fold_of_row[S] != f], S[fold_of_row[S] == f]
    pri = (np.bincount(yB_all[tr], minlength=7) + 1) / (len(tr) + 7)
    LOGP['prior'][te] = np.log(pri)
    for k, M in X.items():
        best = None
        for C in LR_CS:
            L = np.zeros((len(tr), 7))
            for a, b in GroupKFold(5).split(tr, groups=src[tr]):
                L[b] = lr_logp(M[tr[a]], yB_all[tr[a]], M[tr[b]], C)
            s = -L[np.arange(len(tr)), yB_all[tr]].mean()
            if best is None or s < best[0]:
                best = (s, C)
        LOGP[k][te] = lr_logp(M[tr], yB_all[tr], M[te], best[1])
        clog.append({'fold': f, 'model': k, 'C': best[1]})
pd.DataFrame(clog).to_csv(f"{OUT_DIR}/g25_lr_log.csv", index=False)
np.savez(f"{OUT_DIR}/g25_listener_obs.npz", sample_id=DEV.sample_id.values, S=S, E3=E3, E12=E12,
         **{f"logp_{k}": v for k, v in LOGP.items()})
"""),
    ("markdown", "## Fixed decision and descriptive analyses"),
    ("code", r"""
from scipy.stats import spearmanr

gS = [S[src[S] == e] for e in np.unique(src[S])]
rng = np.random.default_rng(0)
DRAWS = [np.concatenate([gS[i] for i in rng.integers(0, len(gS), len(gS))]) for _ in range(N_BOOT)]
nl = lambda k, idx: -LOGP[k][idx, yB_all[idx]].mean()
SUB = {'all S': lambda idx: idx, 'shift': lambda idx: idx[SHIFT[idx]], 'mirror': lambda idx: idx[~SHIFT[idx]]}


def delta(a, b, sub):
    f = SUB[sub]
    pt = nl(a, f(S)) - nl(b, f(S))
    lo, hi = np.percentile([nl(a, f(i)) - nl(b, f(i)) for i in DRAWS], [2.5, 97.5])
    return pt, lo, hi


print("NLL on S:", {k: round(nl(k, S), 4) for k in LOGP})
valid = delta('prior', 'NOW', 'all S')
rows = [('validity: NLL(prior) - NLL(NOW)', 'all S', *valid)]
for sub in ('shift', 'all S', 'mirror'):
    rows.append(('change: NLL(AVG) - NLL(SEP)', sub, *delta('AVG', 'SEP', sub)))
for sub in ('all S', 'shift'):
    rows.append(('recency: NLL(HIST) - NLL(NOW)', sub, *delta('HIST', 'NOW', sub)))
R = pd.DataFrame(rows, columns=['contrast', 'subset', 'delta', 'lo', 'hi'])
R.to_csv(f"{OUT_DIR}/g25_contrasts.csv", index=False)
print(R.round(4).to_string(index=False))

main = R[(R.contrast.str.startswith('change')) & (R.subset == 'shift')].iloc[0]
if valid[1] <= 0:
    dec = 'INVALID'
elif main.lo > 0:
    dec = 'REACTION SIGNAL'
else:
    dec = 'NO REACTION SIGNAL DETECTED'
print(f"\n== G25 decision (fixed rule): {dec} ==  (change on shift MCIS: {main.delta:+.4f} [{main.lo:+.4f}, {main.hi:+.4f}])")

# descriptive: does L's clip-III expression already show B's clip-IV label (early recognition) or A's label (mirroring)?
e3 = (E3[S, :8] @ H2E).argmax(1)
for name, m in (('mirror', ~SHIFT[S]), ('shift', SHIFT[S])):
    print(f"  {name}: L's clip-III argmax expression = B's IV label {np.mean(e3[m] == yB_all[S][m]) * 100:.1f}% | "
          f"= A's III label {np.mean(e3[m] == yA_all[S][m]) * 100:.1f}% (n={m.sum()})")
chg = np.abs(E3[S, :8] - E12[S, :8]).sum(1)
print(f"  size of L's expression change (L1 over 8 classes): mean {chg.mean():.3f} | shift {chg[SHIFT[S]].mean():.3f} vs "
      f"mirror {chg[~SHIFT[S]].mean():.3f}")

hits = sorted(glob.glob("/kaggle/input/**/g14_oof_probs.npz", recursive=True))
if hits:
    z = np.load(hits[0], allow_pickle=True)
    pos = {s: i for i, s in enumerate(z['sample_id'])}
    ii = np.array([pos.get(s, -1) for s in DEV.sample_id.values[S]])
    ok = ii >= 0
    yy = yB_all[S][ok]
    gain = (np.log(z['Full'].mean(0)[ii[ok], yy] + 1e-9) - np.log(z['minus_L'].mean(0)[ii[ok], yy] + 1e-9))
    for name, m in (('all S', np.ones(ok.sum(), bool)), ('shift', SHIFT[S][ok]), ('mirror', ~SHIFT[S][ok])):
        print(f"  RoleNet listener gain vs expression change [{name}]: Spearman {spearmanr(gain[m], chg[ok][m]).correlation:+.3f} "
              f"(n={m.sum()}) | mean gain {gain[m].mean():+.3f}")
else:
    print("  (g14_oof_probs.npz not attached: RoleNet correlation skipped)")
json.dump({'decision': dec, 'change_shift': [float(main.delta), float(main.lo), float(main.hi)],
           'validity': list(map(float, valid)), 'n_S': int(len(S))}, open(f"{OUT_DIR}/g25_decision.json", 'w'), indent=1)
"""),
]


G26A = [
    ("markdown", r"""
# G26a — G8a face / voice features for clip IV (train + val only)

The unchanged G8a extraction, run on the **clip IV** of every train/val MCIS (test clip IV is not read). These features
are used only (a) as a training-time target for the responder pointer of G26 (who is B?) and (b) to measure the
oracle in G26. Clip IV is never an input of a forecaster. No label is read. Output: `c4shard_*.pkl` → make it a dataset.
"""),
    G8A[1],
    ("code", G8A[2][1].replace('SHARD_DIR = f"{OUT_DIR}/g8a"', 'SHARD_DIR = f"{OUT_DIR}/g8a_clip4"')),
    ("code", G8A[3][1]
        .replace("usecols=['sample_id', 'split', 'clip1', 'clip2', 'clip3'])   # no label columns",
                 "usecols=['sample_id', 'split', 'clip1', 'clip2', 'clip3', 'clip4'])   # no label columns")
        .replace("clips = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))",
                 "clips = sorted(set(sp[sp.split.isin(['train', 'val'])].clip4))      # clip IV, train + val only")
        .replace("| clips I-III {len(clips)}", "| clips IV (train+val) {len(clips)}")),
    G8A[4], G8A[5],
    ("code", G8A[6][1].replace('f"{SHARD_DIR}/shard_*.pkl"', 'f"{SHARD_DIR}/c4shard_*.pkl"')
                       .replace('f"{SHARD_DIR}/shard_{n_shard:03d}.pkl"', 'f"{SHARD_DIR}/c4shard_{n_shard:03d}.pkl"')),
    G8A[7],
    ("code", G8A[8][1].replace('f"{SHARD_DIR}/shard_*.pkl"', 'f"{SHARD_DIR}/c4shard_*.pkl"')
                       .replace("c3 = sorted(set(sp.clip3))", "c3 = sorted(clips)        # clip IV: B speaks here")),
]
assert "clip IV, train + val only" in G26A[3][1] and G26A[6][1].count("c4shard_") == 3 and "c4shard_" in G26A[8][1]

G26 = [
    ("markdown", r"""
# G26 — Responder-Pointer RoleNet: learning who will respond (5-fold CV, train+val, 10 seeds; test untouched)

**Question.** Forecasting B's emotion needs B's face (minus-L −2.04 UAR), but B's identity is not given; RoleNet picks
the listener L with a fixed rule (the second most frequent identity in clip III, B in ~81% of MCIS), and pooling all
faces without roles is about as good (noRole +0.44, n.s.). Does *knowing who the responder is* matter, and can a model
learn it?

**Responder identity (training target / oracle only).** B = the dominant identity of clip IV (G26a features), matched to
an identity cluster of clips I–III by ArcFace centroid cosine ≥ 0.45 (not A). Clip IV is never a forecaster input.

**Arms** (RoleNet G14 `Full` settings, same folds, early stopping, 10 seeds):
* `Heuristic` — RoleNet as is (L by rule).
* `Oracle` — L = the true responder cluster (empty if B is not seen in I–III); upper bound for any pointer.
* `Pointer` — the L tokens are an α-weighted mixture of up to K = 4 candidate identities (all non-A clusters, the rule's
  L first); α = softmax over candidates + a null option, computed from each candidate's pooled face tokens and
  observable cues (presence per clip, frame share and mouth-audio synchrony in clip III, face size). Trained with
  CE(emotion) + 0.5 · CE(pointer, true responder); at inference only clips I–III are used.

**Decision (fixed before running; ΔUAR plain, 10-seed ensemble, two-level bootstrap over seeds and episodes).**
1. **Stage 1 (headroom):** Δ_oracle = UAR(`Oracle`) − UAR(`Heuristic`). If its CI lower bound ≤ 0 → **STOP: knowing the
   responder does not improve the forecast on Hi-EF**; `Pointer` is not trained.
2. **Stage 2 (method):** only if stage 1 passes, Δ_pointer = UAR(`Pointer`) − UAR(`Heuristic`); CI lower bound > 0 →
   **POINTER HELPS**, else **POINTER NOT SHOWN**.
Reported: NLL differences, responder coverage, rule accuracy vs pointer accuracy on held-out folds.
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # as G14")
        .split("ARMS = [")[0] + """ARMS = [("Full", 'tok', BASE)]
EXPERIMENTS = []
K_CAND, LAMBDA_PTR, N_BOOT = 4, 0.5, 2000
# G26a output: attach the G26a notebook output directly or a dataset made from it; files are found recursively
CLIP4_SHARDS = "/kaggle/input/**/c4shard_*.pkl"
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8], G13[9], G13[10],
    ("markdown", "## True responder (clip IV, target/oracle only), oracle roles and candidate identities"),
    ("code", r"""
G84 = {}
for f in sorted(glob.glob(CLIP4_SHARDS, recursive=True)):
    G84.update(pickle.load(open(f, 'rb')))
assert G84, f"no clip-IV shards found at {CLIP4_SHARDS}: attach the G26a output"
miss4 = sorted(set(DEV.clip4) - set(G84))
print(f"clip IV records {len(G84)} | missing for {len(miss4)} MCIS clips")


def cluster_arcs(E):
    if len(E) < 2:
        return np.zeros(len(E), int)
    return AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                   distance_threshold=1 - SAME_PERSON_COS).fit_predict(E)


unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
SLOT_O, FMASK_O, VOI_O = {}, np.zeros_like(FMASK.cpu().numpy() if torch.is_tensor(FMASK) else FMASK), np.zeros_like(
    VOI.cpu().numpy() if torch.is_tensor(VOI) else VOI)
CSLOT = {}
CMASK = np.zeros((N, K_CAND, 3, MAXF), bool)
CF = np.zeros((N, K_CAND, 7), np.float32)
TGT = np.full(N, K_CAND, np.int64)                    # index of the true responder among the candidates; K = null
STAT = {'B_found_IV': 0, 'B_in_I_III': 0, 'rule_L_is_B': 0, 'rule_has_L': 0, 'B_in_candidates': 0}
for n, row in enumerate(tqdm(DEV.itertuples(), total=N, desc='responder')):
    cl = [row.clip1, row.clip2, row.clip3]
    items = [(k, j) for k, c in enumerate(cl) for j in range(len(G8[c]['faces']))]
    E = np.stack([G8[cl[k]]['faces'][j]['arc'] for k, j in items]).astype(np.float32) if items else np.zeros((0, 512))
    lab = cluster_arcs(E) if len(items) else np.zeros(0, int)        # identical to the role cell
    frames = defaultdict(set)
    for (k, j), p in zip(items, lab):
        frames[(k, p)].add(G8[cl[k]]['faces'][j]['frame'])
    ids3 = sorted({p for (k, p) in frames if k == 2}, key=lambda p: -len(frames[(2, p)]))
    A = ids3[0] if ids3 else None
    L = ids3[1] if len(ids3) > 1 else None
    tot = defaultdict(int)
    for (k, p), fr in frames.items():
        tot[p] += len(fr)
    # true responder: dominant identity of clip IV matched to a clip I-III cluster (not A)
    bt = None
    f4 = G84.get(row.clip4, {}).get('faces', [])
    if f4:
        E4 = np.stack([d['arc'] for d in f4]).astype(np.float32)
        l4 = cluster_arcs(E4)
        fr4 = defaultdict(set)
        for d, p in zip(f4, l4):
            fr4[p].add(d['frame'])
        dom = max(fr4, key=lambda p: len(fr4[p]))
        if len(fr4[dom]) / max(G84[row.clip4]['meta']['n_sampled'], 1) >= DOMINANT_MIN_FRAC:
            STAT['B_found_IV'] += 1
            c4 = unit(E4[l4 == dom].mean(0))
            best, bp = SAME_PERSON_COS, None
            for p in tot:
                s = float(unit(E[lab == p].mean(0)) @ c4)
                if s >= best:
                    best, bp = s, p
            bt = bp if bp is not None and bp != A else None
    STAT['B_in_I_III'] += bt is not None
    STAT['rule_has_L'] += L is not None
    STAT['rule_L_is_B'] += (L is not None) and (L == bt)
    # oracle roles: A as before, L = true responder, O = everyone else
    roleO = lambda p: 0 if p == A else (1 if (bt is not None and p == bt) else 2)
    by = defaultdict(list)
    for (k, j), p in zip(items, lab):
        by[(roleO(p), k)].append(j)
    for k, c in enumerate(cl):
        for r in range(3):
            idx = pick_frames(sorted(by[(r, k)], key=lambda j: G8[c]['faces'][j]['t']), MAXF)
            SLOT_O[(n, r, k)] = (c, idx); FMASK_O[n, r, k, :len(idx)] = True
        v = [x for r in range(3) for x in sync(c, by[(r, k)])]
        VOI_O[n, k] = v + list(VOI[n, k][6:].cpu().numpy() if torch.is_tensor(VOI) else VOI[n, k][6:])
    # candidates: the rule's L first, then the other non-A identities by frame count in I-III
    cand = ([L] if L is not None else []) + sorted([p for p in tot if p not in (A, L)], key=lambda p: -tot[p])
    cand = cand[:K_CAND]
    if bt is not None and bt in cand:
        TGT[n] = cand.index(bt); STAT['B_in_candidates'] += 1
    n3 = max(G8[cl[2]]['meta']['n_sampled'], 1)
    for i, p in enumerate(cand):
        for k, c in enumerate(cl):
            idx = pick_frames(sorted([j for (kk, j), q in zip(items, lab) if kk == k and q == p],
                                     key=lambda j: G8[c]['faces'][j]['t']), MAXF)
            CSLOT[(n, i, k)] = (c, idx); CMASK[n, i, k, :len(idx)] = True
        f3 = [j for (kk, j), q in zip(items, lab) if kk == 2 and q == p]
        r3, sd3 = sync(cl[2], f3)
        size3 = np.mean([np.sqrt(max((G8[cl[2]]['faces'][j]['box'][2] - G8[cl[2]]['faces'][j]['box'][0]) *
                                     (G8[cl[2]]['faces'][j]['box'][3] - G8[cl[2]]['faces'][j]['box'][1]), 0)) / 100
                         for j in f3]) if f3 else 0.0
        CF[n, i] = [float(CMASK[n, i, k].any()) for k in range(3)] + [len(frames[(2, p)]) / n3, r3, sd3, size3]
print({k: v for k, v in STAT.items()})
print(f"rule accuracy (L == true responder | responder seen in I-III and rule has L): "
      f"{STAT['rule_L_is_B'] / max(STAT['B_in_I_III'], 1):.3f}")
print("pointer targets:", np.bincount(TGT, minlength=K_CAND + 1), "(last = null)")


def build_all(fit_clips):
    pca = None
    if HAS_EMB:
        pool = np.concatenate([EMB[c] for c in fit_clips if EMB.get(c) is not None])
        pick = np.random.default_rng(0).choice(len(pool), min(60000, len(pool)), replace=False)
        pca = PCA(PCA_DIM, random_state=0).fit(pool[pick].astype(np.float32))
    FV = {c: (np.zeros((0, FDIM), np.float32) if len(FB[c]) == 0 else
              (np.concatenate([pca.transform(EMB[c].astype(np.float32)), FB[c]], 1) if HAS_EMB else FB[c])) for c in need}

    def fill(slots, shape):
        a = np.zeros(shape, np.float16)
        for (n, r, k), (c, idx) in slots.items():
            if idx:
                a[n, r, k, :len(idx)] = FV[c][idx]
        return torch.tensor(a, device=DEVICE)
    return (fill(SLOT, (N, 3, 3, MAXF, FDIM)), fill(SLOT_O, (N, 3, 3, MAXF, FDIM)),
            fill(CSLOT, (N, K_CAND, 3, MAXF, FDIM)))


FMASK_H, VOI_H = FMASK, VOI
FMASK_O, VOI_O = T(FMASK_O), T(VOI_O.astype(np.float32))
CMASK, CF, TGT_T = T(CMASK), T(CF), T(TGT)
"""),
    ("markdown", "## Pointer model"),
    ("code", r"""
class RoleNetPtr(RoleNetTok):
    # RoleNet whose listener tokens are a learned mixture over candidate identities (+ a null option)
    def __init__(self, cfg, d=RN['D']):
        super().__init__(cfg, d)
        self.cand_proj = nn.Sequential(nn.LayerNorm(d + 7), nn.Linear(d + 7, d), nn.GELU(), nn.Linear(d, 1))
        self.null_logit = nn.Parameter(torch.zeros(1))

    def forward(self, ix, train=False):
        B = len(ix)
        hc, pc = self.pool(CAND[ix], CMASK[ix])                               # [B, K, 3, d], [B, K, 3]
        w = pc.float().unsqueeze(-1)
        summ = (hc * w).sum(2) / w.sum(2).clamp(min=1)                        # [B, K, d]
        logit = self.cand_proj(torch.cat([summ, CF[ix]], -1)).squeeze(-1)     # [B, K]
        logit = logit.masked_fill(~pc.any(-1), -1e4)
        full = torch.cat([logit, self.null_logit.expand(B, 1)], 1)            # [B, K + 1]
        alpha = torch.softmax(full, 1)
        absL = self.absent[1].unsqueeze(0).unsqueeze(0)                       # [1, 1, 3, d]
        hc = torch.where(pc.unsqueeze(-1), hc, absL.expand(B, hc.shape[1], -1, -1))
        hL = (alpha[:, :-1, None, None] * hc).sum(1) + alpha[:, -1, None, None] * self.absent[1].unsqueeze(0)
        self._hL = hL                                                         # used by the patched pool below
        logits, aux = super().forward(ix, train)
        tgt = torch.where(TGT_T[ix] >= 0, TGT_T[ix], torch.full_like(TGT_T[ix], -100))
        aux['ptr'] = (full, tgt, LAMBDA_PTR)
        if not train:
            PTR_EVAL.append((ix.detach().cpu().numpy(), full.argmax(1).detach().cpu().numpy()))
        return logits, aux


PTR_EVAL = []                                                                 # (rows, pointer argmax) at evaluation
_orig_pool_forward = FramePool.forward


def _pool_with_listener(self, x, m):
    h, present = _orig_pool_forward(self, x, m)
    owner = getattr(self, '_owner', None)
    if owner is not None and getattr(owner, '_hL', None) is not None and h.dim() == 4 and h.shape[1] == 3:
        h = h.clone(); present = present.clone()
        h[:, 1] = owner._hL                                                   # listener slots := pointer mixture
        present[:, 1] = True
        owner._hL = None
    return h, present


FramePool.forward = _pool_with_listener


def make_ptr(cfg):
    m = RoleNetPtr(cfg)
    object.__setattr__(m.pool, '_owner', m)          # plain reference, not a registered submodule
    return m


MAKE['ptr'] = make_ptr
HP['ptr'] = HP['role']
print("pointer parameters:", f"{sum(p.numel() for p in make_ptr(BASE).parameters()) / 1e6:.3f}M")
"""),
    ("markdown", "## 5-fold CV: Heuristic and Oracle; Pointer only if stage 1 passes"),
    ("code", r"""
import re
sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
y_all = DEV.yB.values
src = DEV.source_folder.values
OOF, PTR_ACC, log = {}, [], []
t0 = time.time()


def run_arms(arms):
    global FACE, FMASK, VOI, CAND
    for a in arms:
        OOF[a] = np.full((len(SEEDS), N, 7), np.nan, np.float32)
    for f in range(N_OUTER):
        tr_eps = [e for e in EPS if FOLD[e] != f]
        dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
        trr = np.where(np.isin(src, tr_eps))[0]
        fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
        dev_rows = np.where(np.isin(src, dev_eps))[0]
        te_rows = np.where(fold_of_row == f)[0]
        FH, FO, FC = build_all(sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel())))
        tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
        for a in arms:
            FACE, FMASK, VOI = (FO, FMASK_O, VOI_O) if a == 'Oracle' else (FH, FMASK_H, VOI_H)
            CAND = FC
            kind = 'ptr' if a == 'Pointer' else 'tok'
            for si, seed in enumerate(SEEDS):
                PTR_EVAL.clear()
                p, sel = train_eval(kind, BASE, tr, dev, te, seed + 1000 * f)
                OOF[a][si, te_rows] = p
                if a == 'Pointer':
                    last = {}
                    for rows_, am in PTR_EVAL:
                        last.update(zip(rows_.tolist(), am.tolist()))
                    pr = np.array([last[r] for r in te_rows])
                    PTR_ACC.append({'fold': f, 'seed': seed, 'pointer_acc': float(np.mean(pr == TGT[te_rows])),
                                    'rule_acc': float(np.mean(np.where(CMASK[te_rows, 0].any((-1, -2)).cpu().numpy(), 0, K_CAND) == TGT[te_rows]))})
                u = war_uar(p.argmax(1), y_all[te_rows], 7)[1]
                log.append({'fold': f, 'arm': a, 'seed': seed, 'sel_UAR': sel, 'UAR': u})
                print(f"fold {f} {a:<9} seed {seed}: sel {sel:5.2f} | UAR {u:5.2f} | {(time.time() - t0) / 60:.1f} min",
                      flush=True)
                torch.cuda.empty_cache()
        FACE, FMASK, VOI = FH, FMASK_H, VOI_H


def uar_of(pred, idx=None):
    return war_uar(pred if idx is None else pred[idx], y_all if idx is None else y_all[idx], 7)[1]


def two_level(pa, pb, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(src == e)[0] for e in np.unique(src)]
    d, dn = [], []
    for _ in range(n_boot):
        A_, B_ = pa[rng.integers(0, len(pa), len(pa))].mean(0), pb[rng.integers(0, len(pb), len(pb))].mean(0)
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d.append(uar_of(A_.argmax(1), idx) - uar_of(B_.argmax(1), idx))
        dn.append(-np.log(B_[idx, y_all[idx]] + 1e-9).mean() + np.log(A_[idx, y_all[idx]] + 1e-9).mean())
    return np.percentile(d, [2.5, 97.5]), np.percentile(dn, [2.5, 97.5])


def report(a, b):
    pa, pb = OOF[a], OOF[b]
    d = uar_of(pa.mean(0).argmax(1)) - uar_of(pb.mean(0).argmax(1))
    dn = -np.log(pb.mean(0)[np.arange(N), y_all] + 1e-9).mean() + np.log(pa.mean(0)[np.arange(N), y_all] + 1e-9).mean()
    (lo, hi), (nlo, nhi) = two_level(pa, pb)
    print(f"  {a} − {b}: ΔUAR {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | ΔNLL (b − a) {dn:+.4f} [{nlo:+.4f}, {nhi:+.4f}]")
    return d, lo, hi


run_arms(['Heuristic', 'Oracle'])
print({k: f"{uar_of(v.mean(0).argmax(1)):.2f}" for k, v in OOF.items()})
print("\n== stage 1 ==")
d1, lo1, hi1 = report('Oracle', 'Heuristic')
seen = TGT < K_CAND
for nm, m in (('responder seen in I-III', seen), ('not seen', ~seen)):
    print(f"  [{nm}, n={m.sum()}] Oracle {uar_of(OOF['Oracle'].mean(0).argmax(1), m):.2f} | "
          f"Heuristic {uar_of(OOF['Heuristic'].mean(0).argmax(1), m):.2f}")
stage1 = lo1 > 0
dec = 'STOP: knowing the responder does not improve the forecast on Hi-EF'
d2 = lo2 = hi2 = None
if stage1:
    run_arms(['Pointer'])
    print("\n== stage 2 ==")
    d2, lo2, hi2 = report('Pointer', 'Heuristic')
    report('Oracle', 'Pointer')
    dec = 'POINTER HELPS' if lo2 > 0 else 'POINTER NOT SHOWN'
    acc = pd.DataFrame(PTR_ACC)
    acc.to_csv(f"{OUT_DIR}/g26_pointer_accuracy.csv", index=False)
    print(f"  responder identification on held-out folds (target incl. null): pointer {acc.pointer_acc.mean():.3f} "
          f"vs rule {acc.rule_acc.mean():.3f}")
print(f"\n== G26 decision (fixed rule): {dec} ==")
safe = lambda s: re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g26_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g26_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src, tgt=TGT,
         **{safe(k): v for k, v in OOF.items()})
json.dump({'decision': dec, 'stage1': [d1, lo1, hi1], 'stage2': [d2, lo2, hi2], 'stats': STAT},
          open(f"{OUT_DIR}/g26_decision.json", 'w'), indent=1, default=float)
"""),
]


G27 = [
    ("markdown", r"""
# G27 — Pilot: aggregate mention representations before or after classification? (5-fold CV, train+val, 3 seeds; test untouched)

**What this pilot tests — and what it does not.** It tests whether keeping *contextualised mention representations*
separate until after the classifier (late aggregation) forecasts B's clip-IV emotion better than merging them before
the classifier (early aggregation), with the same candidates, gate and head. It does **not** test full event-role
perspectives: there is no semantic-role parsing, no coreference, no cross-turn merging, and no perspective-specific
training (no external experiencer data). A negative result is limited to this design; it does not show that event roles
or experiencer-specific training are useless. A positive result shows a benefit of late aggregation here, not that the
model understands perspectives.

**Candidates (from clips I–III only).** Every mention span is its own candidate, tagged with its clip: pronouns by a
fixed list (1st / 2nd / 3rd person, incl. apostrophe-less forms such as "youre", "hes") and person names from a fixed
NER model (`dslim/bert-base-NER`, PER). No merging across clips or mentions ("I" in clip I and clip II stay separate;
"you" is not assumed to be B). One **whole-context candidate** (mean of all content tokens) is always present, so
windows without mentions still get a valid forecast.

**Encoder.** RoBERTa-large, frozen, no new tokens. Input `<s> I </s> II </s> III </s>` with the tokenizer's existing
separator; a mention = mean of the last-4-layer hidden states over its tokens (character offsets). Cached once.

**Audio-visual vector v** (same for all arms): per clip CLIP face / frame / AudioSet audio (PCA-32 each), mean
HSEmotion + face flag, audio flag; scaler and PCA fitted on the training part of each fold. A compact baseline, not
RoleNet.

**Arms** (same gate g over candidates, same head F; trained separately):
* `A` — whole-context candidate only: p = softmax F([z_ctx; v]).
* `B` early — p = softmax F([Σ_j w_j z_j; v]).
* `C` late — p = Σ_j w_j softmax F([z_j; v]) (shared head for all candidates).
w = softmax over candidates of g([z_j; v]); z_j includes a candidate-type and clip embedding.

**Protocol (fixed before running).** Same source folds as G8b–G26; early stopping on 5 held-out training episodes
(dev NLL); 3 seeds; OOF probabilities averaged over seeds. **Primary:** Δ = NLL(B) − NLL(C), per-MCIS loss differences
bootstrapped over source folders (seeds and mentions are not treated as independent samples). **C beats B** if the CI
lower bound is > 0. Secondary: C − A, B − A, UAR; windows with vs without mentions.
"""),
    ("code", G13[1][1].split("ARMS = [")[0] + """ARMS = []
EXPERIMENTS = []
TEXT_MODEL, NER_MODEL = "roberta-large", "dslim/bert-base-NER"
SEEDS3 = [42, 123, 456]
N_BOOT, PCA_K, MAX_CAND = 2000, 32, 16
PILOT = dict(d=128, lr=1e-3, wd=1e-2, dropout=0.3, epochs=100, patience=10, batch=64)
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8],
    ("markdown", "## Mention candidates and frozen RoBERTa-large representations (cached)"),
    ("code", r"""
import re
from transformers import AutoTokenizer, AutoModel, pipeline

PRON = {'1st': "i me my mine myself we us our ours ourselves im ive",
        '2nd': "you your yours yourself yourselves youre youve youll youd ya",
        '3rd': "he him his himself she her hers herself they them their theirs themselves hes shes theyre theyve theyll theyd"}
PRON = {w: k for k, ws in PRON.items() for w in ws.split()}
TYPES = ['context', '1st', '2nd', '3rd', 'name']
tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
enc = AutoModel.from_pretrained(TEXT_MODEL, output_hidden_states=True).to(DEVICE).eval()
ner = pipeline('ner', model=NER_MODEL, aggregation_strategy='simple', device=0 if DEVICE == 'cuda' else -1)

TXTS = [[r.t1 or '', r.t2 or '', r.t3 or ''] for r in DEV.itertuples()]
uniq = sorted({t for ts in TXTS for t in ts if t.strip()})
NERS = {}
for i in range(0, len(uniq), 64):
    for t, ents in zip(uniq[i:i + 64], ner(uniq[i:i + 64])):
        NERS[t] = [(int(e['start']), int(e['end'])) for e in ents if e['entity_group'] == 'PER']


def mentions(t):
    out = [(m.start(), m.end(), PRON[m.group(0).lower()]) for m in re.finditer(r"[A-Za-z]+", t) if m.group(0).lower() in PRON]
    out += [(s, e, 'name') for s, e in NERS.get(t, [])]
    return sorted(out)


ZC, ZT, ZK, ZM, MENT = [], [], [], [], []             # vectors, type id, clip id, mask, readable spans
D_TXT = enc.config.hidden_size
for n, ts in enumerate(tqdm(TXTS, desc='encode')):
    ids, spans = [tok.cls_token_id], []
    for k, t in enumerate(ts):
        e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        base = len(ids)
        ids += e['input_ids'] + [tok.sep_token_id]
        for s, en, ty in mentions(t):
            toks = [base + i for i, (a, b) in enumerate(e['offset_mapping']) if a < en and b > s]
            if toks:
                spans.append((toks, TYPES.index(ty), k, t[s:en]))
    with torch.no_grad():
        hs = enc(torch.tensor([ids], device=DEVICE)).hidden_states
    h = torch.stack(hs[-4:]).mean(0)[0].float().cpu().numpy()                 # [T, D]
    content = [i for i, x in enumerate(ids) if x not in (tok.cls_token_id, tok.sep_token_id)]
    vec = [h[content].mean(0) if content else np.zeros(D_TXT, np.float32)]
    ty, cl, ment = [0], [3], []
    for toks, t_, k, s in spans[:MAX_CAND - 1]:
        vec.append(h[toks].mean(0)); ty.append(t_); cl.append(k); ment.append((s, TYPES[t_], k))
    m = np.zeros(MAX_CAND, bool); m[:len(vec)] = True
    V = np.zeros((MAX_CAND, D_TXT), np.float32); V[:len(vec)] = np.stack(vec)
    T_ = np.zeros(MAX_CAND, np.int64); T_[:len(ty)] = ty
    K_ = np.full(MAX_CAND, 3, np.int64); K_[:len(cl)] = cl
    ZC.append(V); ZT.append(T_); ZK.append(K_); ZM.append(m); MENT.append(ment)
ZC, ZT, ZK, ZM = np.stack(ZC), np.stack(ZT), np.stack(ZK), np.stack(ZM)
NMENT = ZM.sum(1) - 1
print(f"mentions per window: mean {NMENT.mean():.2f} | windows with >= 1 mention {np.mean(NMENT > 0) * 100:.1f}% | "
      f"capped at {MAX_CAND - 1}: {np.mean(NMENT >= MAX_CAND - 1) * 100:.1f}%")
print("mention types:", {TYPES[t]: int(((ZT == t) & ZM).sum()) for t in range(1, 5)})
json.dump({s: m for s, m in zip(DEV.sample_id.values, MENT)}, open(f"{OUT_DIR}/g27_mentions.json", 'w'))
del enc; torch.cuda.empty_cache()
"""),
    ("markdown", "## Audio-visual vector (fitted per fold) and the three arms"),
    ("code", r"""
from sklearn.decomposition import PCA as _PCA
from sklearn.preprocessing import StandardScaler

EXPR = np.zeros((N, 3, 11), np.float32)
for n, row in enumerate(DEV.itertuples()):
    for k, c in enumerate((row.clip1, row.clip2, row.clip3)):
        if len(FB[c]):
            EXPR[n, k, :10] = FB[c][:, :10].mean(0); EXPR[n, k, 10] = 1.0
ix3 = CLIPIDX.cpu().numpy() if torch.is_tensor(CLIPIDX) else CLIPIDX
fmk = FEAT['fmask'].float().unsqueeze(-1)
FACEm = ((FEAT['face'] * fmk).sum(1) / fmk.sum(1).clamp(min=1)).cpu().numpy()
ORIm = FEAT['ori'].mean(1).cpu().numpy()
AF = FEAT['afound'].float().cpu().numpy()
AUDn = (F.normalize(FEAT['audio'], dim=-1).cpu().numpy()) * AF[:, None]


def av_vectors(tr, te):
    A_, B_ = [], []
    for k in range(3):
        cidx = ix3[:, k]
        for M, fitmask in ((FACEm, None), (ORIm, None), (AUDn, AF > 0)):
            rows = cidx[tr] if fitmask is None else cidx[tr][fitmask[cidx[tr]]]
            p = _PCA(min(PCA_K, len(rows) - 1), random_state=0).fit(M[rows])
            A_.append(p.transform(M[cidx[tr]])); B_.append(p.transform(M[cidx[te]]))
        A_.append(np.concatenate([EXPR[tr, k], AF[cidx[tr], None]], 1)); B_.append(np.concatenate([EXPR[te, k], AF[cidx[te], None]], 1))
    a, b = np.concatenate(A_, 1), np.concatenate(B_, 1)
    sc = StandardScaler().fit(a)
    return sc.transform(a).astype(np.float32), sc.transform(b).astype(np.float32)


class Pilot(nn.Module):
    def __init__(self, arm, d_av, d=PILOT['d'], p=PILOT['dropout']):
        super().__init__()
        self.arm = arm
        self.z = nn.Sequential(nn.LayerNorm(D_TXT), nn.Linear(D_TXT, d))
        self.type_emb, self.clip_emb = nn.Embedding(len(TYPES), d), nn.Embedding(4, d)
        self.v = nn.Sequential(nn.Linear(d_av, d), nn.GELU(), nn.Dropout(p))
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 7))

    def forward(self, zc, zt, zk, zm, av):                        # returns log-probabilities [B, 7]
        z = self.z(zc) + self.type_emb(zt) + self.clip_emb(zk)    # [B, M, d]
        v = self.v(av)
        if self.arm == 'A':
            return F.log_softmax(self.head(torch.cat([z[:, 0], v], -1)), -1)
        vv = v.unsqueeze(1).expand_as(z)
        g = self.gate(torch.cat([z, vv], -1)).squeeze(-1).masked_fill(~zm, -1e4)
        w = torch.softmax(g, 1)
        self.last_w = w.detach()
        if self.arm == 'B':
            return F.log_softmax(self.head(torch.cat([(w.unsqueeze(-1) * z).sum(1), v], -1)), -1)
        lq = F.log_softmax(self.head(torch.cat([z, vv], -1)), -1)           # [B, M, 7]
        return torch.logsumexp(torch.log(w.clamp(min=1e-12)).unsqueeze(-1) + lq, 1)


ZC_T, ZT_T, ZK_T, ZM_T = T(ZC), T(ZT), T(ZK), T(ZM)


def fit_pilot(arm, AV_all, tr, dev, te, seed):
    seed_all(seed)
    m = Pilot(arm, AV_all.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=PILOT['lr'], weight_decay=PILOT['wd'])
    AVt = T(AV_all)
    run = lambda ix: m(ZC_T[ix], ZT_T[ix], ZK_T[ix], ZM_T[ix], AVt[ix])
    yb = T(y_all)
    best, state, bad = 1e9, None, 0
    for ep in range(PILOT['epochs']):
        m.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), PILOT['batch']):
            j = perm[i:i + PILOT['batch']]
            loss = F.nll_loss(run(j), yb[j])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            dl = F.nll_loss(run(dev), yb[dev]).item()
        if dl < best - 1e-4:
            best, bad, state = dl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PILOT['patience']:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        lp = run(te).exp().cpu().numpy()
        w = m.last_w.cpu().numpy() if arm != 'A' else None
    return lp, w, best
"""),
    ("markdown", "## 5-fold CV (same folds as G8b–G26), 3 seeds"),
    ("code", r"""
sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
y_all = DEV.yB.values
src = DEV.source_folder.values
ARMS3 = ['A', 'B', 'C']
OOF = {a: np.full((len(SEEDS3), N, 7), np.nan, np.float32) for a in ARMS3}
GATE = {a: np.full((len(SEEDS3), N, MAX_CAND), np.nan, np.float32) for a in ('B', 'C')}
log, t0 = [], time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    a_tr, a_te = av_vectors(trr, te_rows)                   # scaler / PCA fitted on the training part only
    AV_all = np.zeros((N, a_tr.shape[1]), np.float32); AV_all[trr], AV_all[te_rows] = a_tr, a_te
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for a in ARMS3:
        for si, seed in enumerate(SEEDS3):
            p, w, sel = fit_pilot(a, AV_all, tr, dev, te, seed + 1000 * f)
            OOF[a][si, te_rows] = p
            if w is not None:
                GATE[a][si, te_rows] = w
            log.append({'fold': f, 'arm': a, 'seed': seed, 'dev_nll': sel})
            print(f"fold {f} arm {a} seed {seed}: dev NLL {sel:.4f} | eval NLL "
                  f"{-np.log(p[np.arange(len(te_rows)), y_all[te_rows]] + 1e-9).mean():.4f} | {(time.time() - t0) / 60:.1f} min",
                  flush=True)
assert all(not np.isnan(v).any() for v in OOF.values())
pd.DataFrame(log).to_csv(f"{OUT_DIR}/g27_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g27_oof.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src, n_mentions=NMENT,
         **{f"p_{a}": v for a, v in OOF.items()}, **{f"gate_{a}": v for a, v in GATE.items()})
"""),
    ("markdown", "## Primary contrast and secondary analyses"),
    ("code", r"""
P = {a: OOF[a].mean(0) for a in ARMS3}
L = {a: -np.log(P[a][np.arange(N), y_all] + 1e-9) for a in ARMS3}           # per-MCIS loss, averaged over seeds
groups = [np.where(src == e)[0] for e in np.unique(src)]
rng = np.random.default_rng(0)
DRAWS = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]


def uar(p, idx):
    return war_uar(p[idx].argmax(1), y_all[idx], 7)[1]


def contrast(a, b, mask=None):
    base = np.arange(N) if mask is None else np.where(mask)[0]
    d = (L[a] - L[b])
    pt = d[base].mean()
    bs = []
    for idx in DRAWS:
        ii = idx if mask is None else idx[mask[idx]]
        bs.append(d[ii].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    du = uar(P[b], base) - uar(P[a], base)
    return pt, lo, hi, du


print({a: f"NLL {L[a].mean():.4f} | UAR {uar(P[a], np.arange(N)):.2f}" for a in ARMS3})
rows = []
for a, b in (('B', 'C'), ('A', 'C'), ('A', 'B')):
    for name, m in (('all', None), ('with mention', NMENT > 0), ('no mention', NMENT == 0)):
        pt, lo, hi, du = contrast(a, b, m)
        rows.append({'contrast': f"NLL({a}) - NLL({b})", 'subset': name, 'delta': pt, 'lo': lo, 'hi': hi,
                     'UAR(b) - UAR(a)': du})
R = pd.DataFrame(rows)
R.to_csv(f"{OUT_DIR}/g27_contrasts.csv", index=False)
print(R.round(4).to_string(index=False))
main = R[(R.contrast == 'NLL(B) - NLL(C)') & (R.subset == 'all')].iloc[0]
dec = 'C BEATS B (late aggregation helps in this design)' if main.lo > 0 else 'NO EVIDENCE THAT LATE AGGREGATION HELPS (this design)'
print(f"\n== G27 decision (fixed rule): {dec} ==  Δ = {main.delta:+.4f} [{main.lo:+.4f}, {main.hi:+.4f}]")
for a in ('B', 'C'):
    G = np.nanmean(GATE[a], 0)
    w_ctx = G[:, 0]
    print(f"  gate {a}: mean weight on the whole-context candidate {np.nanmean(w_ctx):.3f} | on mentions (windows with mentions) "
          f"{np.nanmean(1 - w_ctx[NMENT > 0]):.3f}")
json.dump({'decision': dec, 'delta_BC': [float(main.delta), float(main.lo), float(main.hi)],
           'mention_share': float(np.mean(NMENT > 0))}, open(f"{OUT_DIR}/g27_decision.json", 'w'), indent=1)
"""),
]


# ======================= MELD in the Hi-EF format (M1 features, M2 G8a, M3 RoleNet) =======================
_REPO = HERE.parent.parent
_ESR_FILES = {f"esresnet/{n}": (_REPO / "esresnet" / n).read_text() for n in ("__init__.py", "attention.py", "base.py", "fbsp.py")}
_ESR_FILES["ignite_trainer/_interfaces.py"] = (_REPO / "ignite_trainer" / "_interfaces.py").read_text()
_ESR_FILES["ignite_trainer/__init__.py"] = "from ._interfaces import AbstractNet, AbstractTransform\n"
_t = (_REPO / "utils" / "transforms.py").read_text()
_ESR_FILES["utils/transforms.py"] = "import math\nimport numpy as np\nimport torch\n\n" + _t[_t.index("def scale"):_t.index("class ToTensor1D")]
_ESR_FILES["utils/__init__.py"] = ""

MELD_FETCH = r'''
import os, re, glob, subprocess
from concurrent.futures import ThreadPoolExecutor

MELD_URL = "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
CSV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{}_sent_emo.csv"
EMO_MAP = {'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'joy': 'happy', 'neutral': 'neutral',
           'sadness': 'sad', 'surprise': 'surprise'}
SPLITS = (('tr', 'train', 'train'), ('dv', 'dev', 'val'), ('te', 'test', 'test'))   # prefix, MELD name, Hi-EF name


MELD_HELP = ("Could not download MELD.Raw automatically. Attach a Kaggle dataset that holds MELD.Raw "
             "(Add Input -> search 'MELD') and set MELD_LOCAL to its folder, e.g. MELD_LOCAL = '/kaggle/input/<name>'. "
             "Unextracted train/dev/test .tar.gz files inside it are fine.")


def _probe_gzip(url):
    # first two bytes of the response must be the gzip magic 1f 8b; prints the real error otherwise
    pr = subprocess.Popen(['curl', '-fsSL', '--max-time', '60', url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    head = pr.stdout.read(2)
    pr.kill()
    err = pr.stderr.read().decode(errors='replace').strip()
    pr.wait()
    if head != b'\x1f\x8b':
        print(f"  {url}: {err[:300]!r}, first bytes {head!r}", flush=True)
        return False
    return True


def _unpack_inner(src_root, dst_root):
    # MELD.Raw ships train/dev/test as inner .tar.gz archives; unpack them into dst_root (writable)
    for inner in glob.glob(f"{src_root}/**/*.tar.gz", recursive=True):
        rel = os.path.dirname(os.path.relpath(inner, src_root))
        out = os.path.join(dst_root, rel)
        os.makedirs(out, exist_ok=True)
        print("  unpacking", inner, flush=True)
        subprocess.run(['tar', '-xzf', inner, '-C', out], check=True)
        if inner.startswith(dst_root):
            os.remove(inner)


def fetch_meld():
    # returns {(prefix, dialogue, utterance): mp4 path} and {prefix: csv path}
    root = MELD_LOCAL
    if root:
        if not glob.glob(f"{root}/**/*.mp4", recursive=True):
            os.makedirs(WORK, exist_ok=True)
            if not glob.glob(f"{WORK}/**/*.mp4", recursive=True):
                _unpack_inner(root, WORK)            # read-only input: unpack the inner archives into WORK
            assert glob.glob(f"{WORK}/**/*.mp4", recursive=True), f"no MELD .mp4 or .tar.gz found under {root}"
            for c in glob.glob(f"{root}/**/*_sent_emo.csv", recursive=True):
                if not os.path.exists(os.path.join(WORK, os.path.basename(c))):
                    subprocess.run(['cp', c, WORK], check=True)
            root = WORK
    else:
        root = WORK
        os.makedirs(WORK, exist_ok=True)
        if not glob.glob(f"{WORK}/**/*.mp4", recursive=True):
            print("checking the MELD.Raw download link ...", flush=True)
            if not _probe_gzip(MELD_URL):
                raise RuntimeError(MELD_HELP)
            print("downloading and unpacking MELD.Raw (about 10 GB) ...", flush=True)
            subprocess.run(f"set -o pipefail; curl -fsSL --retry 3 {MELD_URL} | tar -xz -C {WORK}", shell=True,
                           check=True, executable='/bin/bash')
            _unpack_inner(WORK, WORK)
    vids = {}
    for p in glob.glob(f"{root}/**/*.mp4", recursive=True):
        m = re.fullmatch(r'dia(\d+)_utt(\d+)\.mp4', os.path.basename(p))
        if not m:
            continue                                   # skips macOS '._' files and other names
        low = os.path.relpath(p, root).lower()
        s = 'te' if 'test' in low else 'dv' if 'dev' in low else 'tr' if 'train' in low else None
        if s:
            vids[(s, int(m[1]), int(m[2]))] = p
    csvs = {}
    for s, name, _ in SPLITS:
        found = glob.glob(f"{root}/**/{name}_sent_emo.csv", recursive=True)
        if found:
            csvs[s] = found[0]
        else:
            csvs[s] = f"{WORK}/{name}_sent_emo.csv"
            os.makedirs(WORK, exist_ok=True)
            subprocess.run(['curl', '-fsSL', '--retry', '3', '-o', csvs[s], CSV_URL.format(name)], check=True)
    print("videos found per split:", {s: sum(k[0] == s for k in vids) for s, _, _ in SPLITS})
    return vids, csvs


def clip_id(s, d, u):
    return f"{s}{int(d):05d}/{int(u):03d}"


def extract_wavs(clips, vids, sr):
    # mono wav per clip at <WORK>/audio/<ep>/<num>.wav (ffmpeg), skipped when present
    def one(c):
        s, d, u = c[:2], int(c[2:7]), int(c[8:])
        out = f"{WORK}/audio/{c}.wav"
        if os.path.exists(out) or (s, d, u) not in vids:
            return
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', vids[(s, d, u)], '-vn', '-ac', '1', '-ar', str(sr),
                        out], check=False)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, clips))
    print("wav files:", len(glob.glob(f"{WORK}/audio/**/*.wav", recursive=True)), flush=True)
'''

M1 = [
    ("markdown", r"""
# M1 — MELD in the Hi-EF format: MCIS windows, annotation file and Hi-EF-style clip features

* **MCIS**: three consecutive utterances I, II, III of one MELD dialogue, then utterance IV spoken by a **different**
  speaker than III (Hi-EF's interaction rule). Label = MELD emotion of IV, mapped to the Hi-EF names
  (anger→angry, joy→happy, sadness→sad). Splits: MELD train → `train`, dev → `val` (early stopping), test → `test`.
  `source_folder` = dialogue. Speaker names are **not** written anywhere the models read.
* **Annotation file** in the Hi-EF layout (`MELD/Hi-EF/annotation.csv`, no header: 0 clip id, 1 text, 5 polarity =
  MELD sentiment, 7 emotion, 8 uncertainty = 1). Clip id = `<tr|dv|te><dialogue:05d>/<utterance:03d>`.
* **Features per clip I–III** in the `hi-ef-features-v2` format (re-implemented; the original Hi-EF extraction script
  is not available): 16 evenly spaced frames; CLIP ViT-B/32 image features of the largest detected face per frame
  (`face_features`, `face_valid_mask`) and of the whole frame (`ori_features`); CLIP text features of the transcript;
  AudioCLIP ESResNeXt-FBSP AudioSet outputs (527) on 3 s of de-silenced audio at 22.05 kHz, as in the Hi-EF code.
* Clip IV is never processed. Output: `meld_split.csv`, `meld_hief/`, `meld_features/` → make it a dataset for M2/M3.

Needs Internet (MELD.Raw ≈ 10 GB, unpacked in `/tmp`) or an attached copy of MELD.Raw (`MELD_LOCAL`).
"""),
    ("code", "!pip install -q insightface onnx termcolor\n!pip uninstall -y -q onnxruntime onnxruntime-gpu\n!pip install -q \"onnxruntime-gpu==1.22.0\""),
    ("code", r"""
# ======== CONFIG ========
OUT_DIR = "/kaggle/working"
WORK = "/tmp/meld"                  # large scratch space for the raw videos
MELD_LOCAL = None                   # or the path of an attached dataset holding the MELD.Raw videos and CSVs
ESR_URL = "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/ESRNXFBSP.pt"
N_FRAMES, DET_SIZE, MIN_DET_SCORE, MIN_FACE_PX, FACE_MARGIN = 16, (640, 640), 0.5, 24, 0.2
AUDIO_SR = 22050                    # AudioCLIP sample rate (the Hi-EF preprocessing)
DEBUG_N = None                      # e.g. 40 windows for a smoke test
"""),
    ("code", MELD_FETCH + r"""
import numpy as np, pandas as pd
vids, csvs = fetch_meld()
"""),
    ("markdown", "## MCIS windows, split file and annotation file"),
    ("code", r"""
U = []
for s, name, hname in SPLITS:
    d = pd.read_csv(csvs[s])
    d['s'], d['hsplit'] = s, hname
    U.append(d)
U = pd.concat(U, ignore_index=True)
U['cid'] = [clip_id(s, d, u) for s, d, u in zip(U.s, U.Dialogue_ID, U.Utterance_ID)]
U['emo'] = U.Emotion.map(EMO_MAP)
assert U.emo.notna().all()
U['text'] = U.Utterance.astype(str).str.replace('\x92', "'").str.replace('\x91', "'").str.replace('\x93', '"').str.replace('\x94', '"')
rows = []
for (s, d), g in U.sort_values(['s', 'Dialogue_ID', 'Utterance_ID']).groupby(['s', 'Dialogue_ID'], sort=False):
    c, sp_, em = g.cid.tolist(), g.Speaker.tolist(), g.emo.tolist()
    for i in range(3, len(g)):
        if sp_[i] == sp_[i - 1]:
            continue                                   # Hi-EF: the last two clips feature different people
        rows.append({'split': g.hsplit.iloc[0], 'source_folder': f"{s}{int(d):05d}", 'clip1': c[i - 3],
                     'clip2': c[i - 2], 'clip3': c[i - 1], 'clip4': c[i], 'clip3_emotion': em[i - 1],
                     'clip4_emotion': em[i]})
SP = pd.DataFrame(rows)
if DEBUG_N:
    SP = SP.groupby('split', group_keys=False).head(DEBUG_N)
SP.insert(0, 'sample_id', [f"meld{n:05d}" for n in range(len(SP))])
SP.to_csv(f"{OUT_DIR}/meld_split.csv", index=False)
ann_dir = f"{OUT_DIR}/meld_hief/MELD/Hi-EF"
os.makedirs(ann_dir, exist_ok=True)
A = pd.DataFrame({0: U.cid, 1: U.text, 2: '', 3: '', 4: '', 5: U.Sentiment.str.lower(), 6: '', 7: U.emo, 8: '1'})
A.to_csv(f"{ann_dir}/annotation.csv", header=False, index=False)
print("MCIS per split:", SP.split.value_counts().to_dict())
print("label share:", SP.clip4_emotion.value_counts(normalize=True).round(3).to_dict())
print(f"P(clip IV emotion == clip III emotion) = {(SP.clip4_emotion == SP.clip3_emotion).mean():.3f}")
CLIPS = sorted(set(SP[['clip1', 'clip2', 'clip3']].values.ravel()))
miss = [c for c in CLIPS if (c[:2], int(c[2:7]), int(c[8:])) not in vids]
print(f"clips I-III {len(CLIPS)} | without video {len(miss)} (features stay empty for those)")
"""),
    ("markdown", "## Audio, models and per-clip features"),
    ("code", r"""
import torch, cv2, sys, urllib.request
from PIL import Image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
extract_wavs(CLIPS, vids, AUDIO_SR)

# AudioCLIP's ESResNeXt-FBSP (code as in this repository's esresnet/, with minimal stubs for its imports)
PKG = f"{WORK}/esr_pkg"
for rel, src in """ + repr(_ESR_FILES) + r""".items():
    os.makedirs(os.path.dirname(f"{PKG}/{rel}"), exist_ok=True)
    open(f"{PKG}/{rel}", 'w').write(src)
sys.path.insert(0, PKG)
from esresnet import ESResNeXtFBSP
esr_path = f"{WORK}/ESRNXFBSP.pt"
if not os.path.exists(esr_path):
    urllib.request.urlretrieve(ESR_URL, esr_path)
esr = ESResNeXtFBSP(n_fft=2048, hop_length=561, win_length=1654, window='blackmanharris', normalized=True,
                    onesided=True, spec_height=-1, spec_width=-1, num_classes=527, apply_attention=True, pretrained=False)
r = esr.load_state_dict(torch.load(esr_path, map_location='cpu'), strict=False)
assert not r.missing_keys, r.missing_keys[:5]
esr = esr.to(DEVICE).eval()

from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
feat = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float().cpu()

from insightface.app import FaceAnalysis
det = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
det.prepare(ctx_id=0, det_size=DET_SIZE)
import librosa
"""),
    ("code", r"""
FEAT_DIR = f"{OUT_DIR}/meld_features"
os.makedirs(FEAT_DIR, exist_ok=True)
TEXT = dict(zip(U.cid, U.text))


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


def largest_face(fr):
    faces = [f for f in det.get(fr) if f.det_score >= MIN_DET_SCORE]
    if not faces:
        return None
    x1, y1, x2, y2 = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).bbox
    w, h = x2 - x1, y2 - y1
    if min(w, h) < MIN_FACE_PX:
        return None
    H, W = fr.shape[:2]
    x1, y1 = int(max(0, x1 - FACE_MARGIN * w)), int(max(0, y1 - FACE_MARGIN * h))
    x2, y2 = int(min(W, x2 + FACE_MARGIN * w)), int(min(H, y2 + FACE_MARGIN * h))
    return fr[y1:y2, x1:x2]


def audio_527(c):
    p = f"{WORK}/audio/{c}.wav"
    if not os.path.exists(p):
        return torch.zeros(527), False
    try:
        wav, _ = librosa.load(p, sr=AUDIO_SR, mono=True)
    except Exception:
        return torch.zeros(527), False
    if len(wav) < AUDIO_SR // 10 or np.abs(wav).max() < 1e-4:
        return torch.zeros(527), False
    L = 3 * AUDIO_SR                                   # the Hi-EF preprocessing: trim silence, centre 3 s
    wav, idx = librosa.effects.trim(wav, top_db=20)
    if len(wav) < L:
        wav = np.pad(wav, (0, L - len(wav)))
    elif len(wav) > L:
        mid = len(wav) // 2
        wav = wav[mid - L // 2: mid - L // 2 + L]
    x = torch.tensor(wav[None, None] * 32768.0, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return esr(x)[0].float().cpu(), True


todo = [c for c in CLIPS if not os.path.exists(f"{FEAT_DIR}/{c.replace('/', '_')}.pt")]
print(f"features to extract: {len(todo)} / {len(CLIPS)}", flush=True)
stats = {'no_video': 0, 'face_frames': 0, 'frames': 0, 'audio': 0}
t0 = time.time() if 'time' in dir() else None
import time
t0 = time.time()
for n, c in enumerate(todo):
    key = (c[:2], int(c[2:7]), int(c[8:]))
    frs = frames_of(vids[key]) if key in vids else []
    stats['no_video'] += not frs
    face = torch.zeros(N_FRAMES, 512); ori = torch.zeros(N_FRAMES, 512); fmask = torch.zeros(N_FRAMES, dtype=torch.bool)
    if frs:
        crops = [largest_face(f) for f in frs]
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frs]
        cimg = [Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)) for cr in crops if cr is not None]
        with torch.no_grad():
            o = feat(clip.get_image_features(pixel_values=proc(images=imgs + cimg, return_tensors='pt').pixel_values.to(DEVICE)))
        ori[:len(frs)] = o[:len(frs)]
        j = len(frs)
        for k, cr in enumerate(crops):
            if cr is not None:
                face[k], fmask[k] = o[j], True
                j += 1
        stats['frames'] += len(frs); stats['face_frames'] += int(fmask.sum())
    with torch.no_grad():
        tok = proc.tokenizer([TEXT.get(c, '') or ' '], truncation=True, max_length=77, padding=True, return_tensors='pt')
        txt = feat(clip.get_text_features(**{k: v.to(DEVICE) for k, v in tok.items()}))[0]
    aud, found = audio_527(c)
    stats['audio'] += found
    torch.save({'face_features': face.half(), 'ori_features': ori.half(), 'face_valid_mask': fmask,
                'text_feature': txt.half(), 'audio_feature': aud, 'audio_found': found},
               f"{FEAT_DIR}/{c.replace('/', '_')}.pt")
    if (n + 1) % 500 == 0:
        el = (time.time() - t0) / 60
        print(f"{n + 1}/{len(todo)} | {el:.1f} min | ETA {el / (n + 1) * (len(todo) - n - 1):.0f} min | {stats}", flush=True)
print("done", stats)
"""),
    ("markdown", "## Sanity checks"),
    ("code", r"""
have = {f[:-3].replace('_', '/', 1) for f in os.listdir(FEAT_DIR) if f.endswith('.pt')}
assert set(CLIPS) <= have, f"{len(set(CLIPS) - have)} clips without features"
sample = [torch.load(f"{FEAT_DIR}/{c.replace('/', '_')}.pt") for c in CLIPS[:: max(1, len(CLIPS) // 300)]]
print(f"face-valid frame share {np.mean([s['face_valid_mask'].float().mean().item() for s in sample]):.2f} | "
      f"audio found {np.mean([s['audio_found'] for s in sample]):.2f} | "
      f"text norm {np.mean([s['text_feature'].float().norm().item() for s in sample]):.2f}")
print("outputs:", os.listdir(OUT_DIR))
"""),
]

_G8A_CFG_OLD = G8A[2][1]
M2 = [
    ("markdown", r"""
# M2 — G8a face / voice features for MELD (clips I–III), in chunks

Runs the unchanged G8a extraction on the MELD clips laid out like Hi-EF (`<root>/MELD/Hi-EF/video|audio/<ep>/<num>`).
Attach the M1 output (for `meld_split.csv`). The work is split into `N_CHUNKS` parts so that each Kaggle session
stays within its time limit: run this notebook once per `CHUNK` (0 … N_CHUNKS−1) and attach every output to M3.
"""),
    G8A[1],
    ("code", r"""
# ======== CONFIG ========
import glob
WORK = "/tmp/meld"
MELD_LOCAL = None                    # or the path of an attached MELD.Raw copy
N_CHUNKS, CHUNK = 3, 0               # run once per CHUNK
SPLIT_CSV = sorted(glob.glob("/kaggle/input/**/meld_split.csv", recursive=True))[0]
DATASET_DIR = "/tmp/meldhief"
OUT_DIR = "/kaggle/working"
SHARD_DIR = f"{OUT_DIR}/g8a"
SHARD_SIZE = 250
""" + _G8A_CFG_OLD[_G8A_CFG_OLD.index("SAMPLE_FPS"):]),
    ("code", MELD_FETCH + r"""
import pandas as pd
vids, csvs = fetch_meld()
_sp = pd.read_csv(SPLIT_CSV, dtype=str)
_clips = sorted(set(_sp[['clip1', 'clip2', 'clip3']].values.ravel()))[CHUNK::N_CHUNKS]
vroot = f"{DATASET_DIR}/MELD/Hi-EF/video"
for c in _clips:
    key = (c[:2], int(c[2:7]), int(c[8:]))
    if key in vids:
        os.makedirs(f"{vroot}/{c.split('/')[0]}", exist_ok=True)
        if not os.path.exists(f"{vroot}/{c}.mp4"):
            os.symlink(vids[key], f"{vroot}/{c}.mp4")
extract_wavs(_clips, vids, 16000)
aroot = f"{DATASET_DIR}/MELD/Hi-EF/audio"
if not os.path.exists(aroot):
    os.symlink(f"{WORK}/audio", aroot)
"""),
    ("code", G8A[3][1].replace(
        "clips = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))",
        "clips = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))[CHUNK::N_CHUNKS]   # this chunk only")),
    G8A[4], G8A[5],
    ("code", G8A[6][1].replace('f"{SHARD_DIR}/shard_{n_shard:03d}.pkl"', 'f"{SHARD_DIR}/shard_c{CHUNK}_{n_shard:03d}.pkl"')),
    G8A[7],
    ("code", G8A[8][1].replace("c3 = sorted(set(sp.clip3))", "c3 = sorted(set(sp.clip3) & set(clips))")),
]
assert "shard_c{CHUNK}" in M2[7][1] and "[CHUNK::N_CHUNKS]" in M2[4][1] and "set(clips))" in M2[9][1]

_M3_CFG = G10[2][1]
_M3_CFG = _M3_CFG[:_M3_CFG.index('DATASET_DIR = ')] + r'''import glob


def find_one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"attach the dataset holding {pattern}")
    return hits[0]


DATASET_DIR = find_one("meld_hief")
FEATURES_DIR = find_one("meld_features")
SPLIT_CSV = find_one("meld_split.csv")
G8A_DIR = "/kaggle/input"            # every attached M2 output (shard_c*_*.pkl) is read
OUT_DIR = "/kaggle/working"
UNLOCK_TEST = True                   # MELD test is read once, by this notebook
''' + _M3_CFG[_M3_CFG.index('\nN_INNER_DEV'):]
_M3_CFG = _M3_CFG.replace("SEEDS = [42, 123, 456, 789, 1024]", "SEEDS = [42, 123, 456]")
assert "SEEDS = [42, 123, 456]\n" in _M3_CFG

_M3_RES = (G10[15][1]
    .replace("('CONFIRMED' if ok else 'NOT CONFIRMED')", "('CI > 0' if ok else 'CI includes 0')")
    .replace("'descriptive only (sequence stopped)'", "('CI > 0' if ok else 'CI includes 0') + ' (descriptive)'")
    .replace("PREREGISTERED fixed-sequence contrasts", "Hi-EF G10 contrasts, descriptive on MELD"))
assert 'print("\\n== per-episode' in _M3_RES

M3 = [
    ("markdown", r"""
# M3 — RoleNet and the Hi-EF baselines on MELD in the Hi-EF setting

Same code as G10 (RoleNet, RoleNet-noRole, B1, LateFusion, PaperBest), with MELD data from M1 (windows, annotation,
CLIP/AudioCLIP features) and M2 (G8a faces and voices). Differences from G10, all fixed before running:
* training rows = MELD train MCIS, early stopping / selection rows = MELD dev MCIS, evaluation = MELD test MCIS (read
  once, by this notebook); 3 seeds per neural arm;
* speaker names are never an input: roles come from face clustering exactly as in Hi-EF; clip IV is never read;
* the Hi-EF fixed-sequence contrasts are reported as descriptive (CI > 0 / CI includes 0); the per-episode table is
  dropped (MELD test has ~280 dialogues);
* extra analysis: plain-scored accuracy on MCIS where B's label equals A's (mirroring) vs not (shift), as Hi-EF F14,
  and the gold-label references (copy-A, P(B | A)).
"""),
    G10[1],
    ("code", _M3_CFG),
    ("code", G10[3][1].replace(
        "if not DEBUG_PER_EPISODE:\n    assert len(EPS) == 45 and len(TEST_EPS) == 8 and IS_TEST.sum() == 409\n", "")),
    G10[4], G10[5], G10[6],
    ("code", G10[7][1].replace("os.path.join(G8A_DIR, '**', 'shard_*.pkl')", "os.path.join(G8A_DIR, '**', 'shard_c*_*.pkl')")),
    G10[8], G10[9], G10[10], G10[11],
    ("markdown", "## Train on MELD train (early stopping on MELD dev), evaluate once on MELD test"),
    ("code", G10[13][1].replace(
        """sel_eps = sorted(random.Random(SELECT_SEED).sample(list(EPS), N_INNER_DEV))
trr = np.where(~IS_TEST)[0]
fit_rows = np.where(~IS_TEST & ~np.isin(src, sel_eps))[0]
dev_rows = np.where(np.isin(src, sel_eps))[0]""",
        """sel_eps = 'MELD dev'
trr = np.where(~IS_TEST)[0]
fit_rows = np.where((DEV.split == 'train').values)[0]
dev_rows = np.where((DEV.split == 'val').values)[0]""").replace("g10_", "m3_")),
    G10[14],
    ("code", _M3_RES[:_M3_RES.index('print("\\n== per-episode')]),
    ("markdown", "## Mirroring vs shift (as Hi-EF F14) and gold-label references"),
    ("code", r"""
yA_t = DEV.yA.values[te_rows]
same = yt == yA_t
print(f"P(yB == yA) on MELD test MCIS: {same.mean():.3f} (Hi-EF train+val 0.355)")
for k in PRED['plain']:
    p = PRED['plain'][k]
    print(f"  {k:<15} plain acc mirror {100 * np.mean(p[same] == yt[same]):5.1f} | shift {100 * np.mean(p[~same] == yt[~same]):5.1f} "
          f"| UAR all {uar7(p, yt):5.2f} | predicts A's label on shifts {100 * np.mean(p[~same] == yA_t[~same]):4.1f}%")
Ttr = np.ones((7, 7))
for a, b in zip(DEV.yA.values[trr], y_all[trr]):
    Ttr[a, b] += 1
Ttr /= Ttr.sum(1, keepdims=True)
print(f"[gold reference] copy A's label: UAR {uar7(yA_t, yt):.2f} | P(B | A_gold) plain UAR {uar7(Ttr[yA_t].argmax(1), yt):.2f}, "
      f"LA UAR {uar7((np.log(Ttr[yA_t]) - LA_TAU * LOGPI_TR).argmax(1), yt):.2f}")
"""),
]
assert "fit_rows = np.where((DEV.split == 'train').values)[0]" in M3[13][1]
assert "per-episode" not in M3[15][1] and "CI includes 0" in M3[15][1]

# ---------------------------------------------------------------- G28: does training with L masked help under the same masking?
G28 = [
    ("markdown", r"""
# G28 — Training RoleNet with masked listener features: adaptation and cost (5-fold CV, train+val, 10 seeds; test untouched)

**Question.** If the listener's (L) face features are missing, does training RoleNet with the *same* kind of masking
improve the forecast under that masking, and what does it cost on the data as they are? The architecture is unchanged;
only the training policy differs. This is not a test of a new architecture.

**Arms** (RoleNet `Full` of G13/G14, same folds, inner early-stop episodes and hyper-parameters; 10 seeds, same seed
per fold for both arms):
* `R-std` — RoleNet trained as before.
* `R-mask` — **training matched to the evaluation masking**: for each training MCIS, with probability 0.5 one of the
  conditions C1–C3 that *changes that MCIS* is drawn uniformly and applied; if none applies, the MCIS is left as is.
  The actual masked share and the frequency of each condition are reported (the share is below 50% because some MCIS
  have no L). A policy that does not help does not show that other policies cannot.

**Masking conditions** (applied after role assignment, to the L face-token branch only):

| Condition | What is removed | MCIS on which it is evaluated |
|---|---|---|
| C0 | nothing (data as they are) | all, and for each Ck also on exactly the Ck set |
| C1 | all L face observations (clips I–III) | at least one L observation |
| C2 | L in clip III, L history (I/II) kept | L in III **and** L in I or II |
| C3 | L history (I/II), L in clip III kept | L in III **and** L in I or II |

* The mask is applied to the frame mask **before** the face tokens are built, so the masked cells become the learned
  *absent* token exactly like naturally missing cells, and the face auxiliary head sees the same masked tokens as the
  main head.
* Speech tokens, scene tokens (whole-frame and face CLIP features) and voice cues (which include per-role mouth–audio
  synchrony) are **kept**. The intervention is therefore "missing role-specific face features of L", not "L was never
  observed".
* C1 and C3 give **presence patterns** that the role rule can also produce naturally (L exists only when clip III has at
  least two identities, so L absent in III implies L absent everywhere). The masked MCIS are still artificial: e.g. after
  C1 the voice cues may still carry L information, which a natural MCIS without L does not have. C2 creates a pattern
  the role rule never produces (L in I/II without L in III).
* Modality dropout (whole face branch / whole context branch, as in G8b) is kept identical in both arms. The share of
  L-masked training MCIS that also had the whole face branch dropped (so the L mask had no effect on the main head) is
  reported.

**Checkpoint selection (differs from G13).** Both arms select the epoch by the same inner-dev criterion
J_dev = 0.5·NLL(C0) + 0.5·mean_k NLL(Ck), each NLL(Ck) on the dev MCIS eligible for Ck (G13 used dev UAR on C0).
`R-std` keeps its training, only the selection rule is G28's.

**Estimand and interval (fixed before running).** NLL is the only primary metric.
* NLL_{s,k}: mean NLL of seed s's probabilities on the Ck set under Ck.
* **Δ_rec = (1/3S) Σ_k Σ_s [NLL^std_{s,k} − NLL^mask_{s,k}]** — mean of per-seed NLLs, not the NLL of the ensemble
  (ensemble NLL is reported separately). Δ_rec > 0 means `R-mask` forecasts better under masking.
* Interval: 2,000 bootstrap draws resampling the 10 seeds and the 45 episodes, with the **same** draws for both arms and
  all three conditions; every quantity is recomputed per draw (C1–C3 are not treated as independent).
* Secondary: Δ_clean = NLL^mask(C0) − NLL^std(C0) on all MCIS (> 0 = cost of masked training on the data as they are);
  sensitivity_k = NLL^std(Ck) − NLL^std(C0) on the same Ck set; per-condition Δ_rec,k; UAR (descriptive).
* Recovery share Δ_rec,k / sensitivity_k is reported only when the sensitivity's CI lower bound is > 0; it is not a share
  of "recovered information".

**Reading rules (fixed before running).**

| Result | Allowed conclusion |
|---|---|
| CI of Δ_rec entirely > 0 | masked training helps adaptation to the interventions tried |
| CI of Δ_rec contains 0 | not shown that this policy helps; **cannot tell** whether the drop under masking is mostly lost evidence or lack of adaptation |
| CI of Δ_rec entirely < 0 | this policy makes the forecast under masking worse |

A positive Δ_rec together with a Δ_clean whose CI is above 0 is a trade-off, not robustness without cost.

**Auxiliary analysis (not a gate).** Do the 9 presence flags (A/L/O × I/II/III, after role assignment) forecast B's
emotion better than the training-fold prior? Regularised logistic regression (C chosen inside the training fold by
grouped inner CV) and the same with pairwise flag interactions; out-of-fold by episode; ΔNLL vs prior with an episode
bootstrap. A gain only shows an association of the presence pattern with the label, not that it adds information
beyond the observed content.
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # as G14")
        .split("ARMS = [")[0] + """ARMS = ['R-std', 'R-mask']
EXPERIMENTS = [("RoleNet", 'role', FULL)]    # only used by the shared model cell's parameter print
P_MASK, N_BOOT = 0.5, 2000
LR_GRID = [0.01, 0.1, 1.0, 10.0]             # auxiliary presence analysis
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8],
    ("markdown", "## Presence patterns and the MCIS sets of C1–C3"),
    ("code", r"""
PRES = FMASK.any(-1).cpu().numpy()                    # [N, role A/L/O, clip I/II/III]
L_any, L3, L12 = PRES[:, 1].any(-1), PRES[:, 1, 2], PRES[:, 1, :2].any(-1)
ELIG = np.stack([L_any, L3 & L12, L3 & L12], 1)      # C1, C2, C3
COND = ['C1 no L', 'C2 no L-III (history kept)', 'C3 no L history (L-III kept)']
for k in range(3):
    print(f"{COND[k]:<32} eligible MCIS {ELIG[:, k].sum():5d} ({ELIG[:, k].mean() * 100:.1f}%)")
print("role-rule checks (expected 0): L in I/II without L in III", int((L12 & ~L3).sum()),
      "| O in III without L in III", int((PRES[:, 2, 2] & ~L3).sum()))
pat = pd.Series([' '.join(r + ':' + ''.join(str(int(x)) for x in PRES[i, j]) for j, r in enumerate('ALO'))
                 for i in range(N)])
vc = pat.value_counts()
print(f"\n{len(vc)} distinct presence patterns (A/L/O, clips I II III); most frequent:")
print((vc.head(15) / N * 100).round(1).to_string())
ELIG_T = T(ELIG)
"""),
    ("markdown", "## Auxiliary analysis: do the presence flags forecast B's emotion? (logistic, out-of-fold)"),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
y_all = DEV.yB.values
src = DEV.source_folder.values
print("fold sizes (MCIS):", load_)

X9 = PRES.reshape(N, 9).astype(float)
X_int = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(X9)


def lr_proba(Xtr, ytr, Xte, C):
    m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
    out = np.full((len(Xte), 7), 1e-6)
    out[:, m.classes_] = m.predict_proba(Xte)
    return out / out.sum(1, keepdims=True)


def nll_rows(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, None))


PRES_OOF = {k: np.zeros((N, 7)) for k in ('prior', 'flags', 'flags+pairs')}
CHOSEN = {'flags': [], 'flags+pairs': []}
for f in range(N_OUTER):
    tr, te = fold_of_row != f, fold_of_row == f
    cnt = np.bincount(y_all[tr], minlength=7) + 1.0
    PRES_OOF['prior'][te] = cnt / cnt.sum()
    for name, X in (('flags', X9), ('flags+pairs', X_int)):
        gkf = GroupKFold(5)
        score = {}
        for C in LR_GRID:
            s = []
            for a, b in gkf.split(X[tr], y_all[tr], src[tr]):
                s.append(nll_rows(lr_proba(X[tr][a], y_all[tr][a], X[tr][b], C), y_all[tr][b]).mean())
            score[C] = np.mean(s)
        C = min(score, key=score.get)
        CHOSEN[name].append(C)
        PRES_OOF[name][te] = lr_proba(X[tr], y_all[tr], X[te], C)
print("C chosen per fold:", CHOSEN)

NLLP = {k: nll_rows(v, y_all) for k, v in PRES_OOF.items()}
groups = [np.where(src == e)[0] for e in np.unique(src)]
rng = np.random.default_rng(0)
for name in ('flags', 'flags+pairs'):
    d = NLLP[name] - NLLP['prior']
    bs = [d[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {name:<12} NLL {NLLP[name].mean():.4f} vs prior {NLLP['prior'].mean():.4f} | "
          f"ΔNLL (model − prior) {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
          + ("  -> presence pattern is associated with the label" if hi < 0 else ""))
"""),
    ("markdown", "## RoleNet with L masking applied before the face tokens"),
    ("code", r"""
def apply_cond(m, cond):
    # m: [B, 3, 3, F] frame mask (role, clip, frame); cond: [B] in {0 none, 1 C1, 2 C2, 3 C3}
    m = m.clone()
    m[cond == 1, 1] = False                  # C1: L, all clips
    m[cond == 2, 1, 2] = False               # C2: L in clip III
    m[cond == 3, 1, :2] = False              # C3: L in clips I/II
    return m


class RoleNetMask(RoleNet):
    # RoleNet (G8b, all switches on) whose L frame mask can be changed per sample before pooling; the masked cells
    # become the absent token, and the face auxiliary head sees the same tokens as the main head.
    def forward(self, ix, train=False, cond=None):
        B, aux = len(ix), {}
        m = FMASK[ix] if cond is None else apply_cond(FMASK[ix], cond)
        h, present = self.pool(FACE[ix], m)                                   # [B, 3, 3, d]
        h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
        h = h + self.face_role[None, :, None] + self.clip_emb[None, None]
        ft = h.reshape(B, 9, -1)
        aux['face'] = (self.head_face(ft.mean(1)), YB[ix], RN['aux_w'])
        tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
        aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.ctx_role[0]
        scn = self.scene(SCN[ix]) + self.ctx_role[1]
        ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)         # [B, 6, d]
        aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1), ft, ct], 1)
        valid = torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)
        self.last_drop_face = None
        if train:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            valid[:, 1:10] &= ~drop_face.unsqueeze(1)
            valid[:, 10:] &= ~drop_ctx.unsqueeze(1)
            self.last_drop_face = drop_face
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


def draw_cond(j):
    # R-mask: with prob. P_MASK pick uniformly one condition that changes this MCIS; none applicable -> unchanged
    el = ELIG_T[j]
    do = (torch.rand(len(j), device=DEVICE) < P_MASK) & el.any(1)
    w = el.float() + (~el.any(1, keepdim=True)).float()
    pick = torch.multinomial(w, 1).squeeze(1) + 1
    return torch.where(do, pick, torch.zeros_like(pick))


def predict_c(model, ix, k, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            j = ix[i:i + bs]
            out.append(F.softmax(model(j, cond=torch.full((len(j),), k, dtype=torch.long, device=DEVICE))[0], -1).cpu())
    return torch.cat(out).numpy()


def nll_mean(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, None)).mean())


def train_eval_g28(arm, tr, dev, te, seed):
    seed_all(seed)
    hp = HP['role']
    model = RoleNetMask(FULL).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    y_dev = YB[dev].cpu().numpy()
    el_dev = ELIG[dev.cpu().numpy()]
    st = {'n': 0, 'masked': 0, 'C1': 0, 'C2': 0, 'C3': 0, 'masked_and_facedrop': 0}
    best, best_state, bad, best_ep = np.inf, None, 0, -1
    for ep in range(hp['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), hp['batch']):
            j = perm[i:i + hp['batch']]
            cond = draw_cond(j)                       # drawn in both arms so that their random streams stay aligned
            if arm == 'R-std':
                cond = torch.zeros_like(cond)
            logits, aux = model(j, train=True, cond=cond)
            st['n'] += len(j); st['masked'] += int((cond > 0).sum())
            for k in (1, 2, 3):
                st[f'C{k}'] += int((cond == k).sum())
            st['masked_and_facedrop'] += int(((cond > 0) & model.last_drop_face).sum())
            loss = F.cross_entropy(logits, YB[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        terms = [nll_mean(predict_c(model, dev[torch.tensor(el_dev[:, k - 1], device=DEVICE)], k), y_dev[el_dev[:, k - 1]])
                 for k in (1, 2, 3) if el_dev[:, k - 1].any()]
        J = 0.5 * nll_mean(predict_c(model, dev, 0), y_dev) + 0.5 * (np.mean(terms) if terms else 0.0)
        if J < best:
            best, bad, best_ep = J, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    model.load_state_dict(best_state)
    probs = np.stack([predict_c(model, te, k) for k in range(4)])         # [4 conditions, n_te, 7]
    return probs, best, best_ep, st, int(el_dev.any(0).sum())


print("parameters:", f"{sum(p.numel() for p in RoleNetMask(FULL).parameters()) / 1e6:.3f}M")
"""),
    ("markdown", "## 5-fold episode cross-validation (same folds, early-stop episodes and seeds as G14)"),
    ("code", r"""
import re

OOF = {a: np.full((len(SEEDS), 4, N, 7), np.nan, np.float32) for a in ARMS}
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)} | dev MCIS eligible for "
          f"C1/C2/C3 {ELIG[dev_rows].sum(0).tolist()}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for arm in ARMS:
        for si, seed in enumerate(SEEDS):
            p, J, ep, st, n_terms = train_eval_g28(arm, tr, dev, te, seed + 1000 * f)
            OOF[arm][si][:, te_rows] = p
            yt = y_all[te_rows]
            log.append({'fold': f, 'arm': arm, 'seed': seed, 'J_dev': J, 'best_epoch': ep, 'dev_cond_terms': n_terms,
                        **{f'NLL_C{k}': nll_mean(p[k], yt) for k in range(4)}, **st})
            print(f"fold {f} {arm:<6} seed {seed:>3}: J_dev {J:.4f} (epoch {ep}) | NLL C0 {nll_mean(p[0], yt):.4f} | "
                  f"masked {st['masked'] / max(st['n'], 1) * 100:4.1f}% of training draws | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()

assert all(not np.isnan(v).any() for v in OOF.values())
LOG = pd.DataFrame(log)
LOG.to_csv(f"{OUT_DIR}/g28_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g28_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         presence=PRES, eligible=ELIG, presence_oof_prior=PRES_OOF['prior'], presence_oof_flags=PRES_OOF['flags'],
         presence_oof_pairs=PRES_OOF['flags+pairs'], **{re.sub(r'[^0-9A-Za-z]+', '_', a): v for a, v in OOF.items()})
print("saved g28_oof_probs.npz (arm arrays: [seed, condition C0..C3, MCIS, 7]) and g28_fold_seed_log.csv")
"""),
    ("markdown", "## Masking actually applied during training"),
    ("code", r"""
m = LOG[LOG.arm == 'R-mask']
tot = m[['n', 'masked', 'C1', 'C2', 'C3', 'masked_and_facedrop']].sum()
print(f"R-mask: masked share of training draws {tot.masked / tot.n * 100:.1f}% (nominal {P_MASK * 100:.0f}%, lower because "
      f"some MCIS have no applicable condition)")
print("  condition frequency among masked draws:", {k: f"{tot[k] / max(tot.masked, 1) * 100:.1f}%" for k in ('C1', 'C2', 'C3')})
print(f"  masked draws whose whole face branch was also dropped by modality dropout: "
      f"{tot.masked_and_facedrop / max(tot.masked, 1) * 100:.1f}% (the L mask had no effect on the main head there)")
print("  R-std masked draws:", int(LOG[LOG.arm == 'R-std'].masked.sum()), "(expected 0)")
print("  best epoch per arm (mean):", LOG.groupby('arm').best_epoch.mean().round(1).to_dict())
"""),
    ("markdown", "## Results (fixed estimand: mean of per-seed NLL; two-level bootstrap with shared draws)"),
    ("code", r"""
S = len(SEEDS)
LOSS = {a: -np.log(np.clip(np.take_along_axis(OOF[a], y_all[None, None, :, None], -1)[..., 0], 1e-7, None))
        for a in ARMS}                                                   # [S, 4, N]
SETS = [np.ones(N, bool), ELIG[:, 0], ELIG[:, 1], ELIG[:, 2]]          # evaluation set of C0 (all), C1, C2, C3


def quantities(sidx, w):
    # every reported quantity for one draw: seeds sidx (indices into SEEDS), row weights w (episode multiplicities)
    M = {a: LOSS[a][sidx].mean(0) for a in ARMS}                        # mean over drawn seeds, [4, N]
    nll = lambda a, k, s: float((M[a][k] * w * s).sum() / max((w * s).sum(), 1e-12))
    q = {}
    for k in (1, 2, 3):
        s = SETS[k]
        q[f'drec_C{k}'] = nll('R-std', k, s) - nll('R-mask', k, s)
        q[f'sens_C{k}'] = nll('R-std', k, s) - nll('R-std', 0, s)
        q[f'sens_mask_C{k}'] = nll('R-mask', k, s) - nll('R-mask', 0, s)
        q[f'clean_on_C{k}'] = nll('R-mask', 0, s) - nll('R-std', 0, s)
        q[f'std_C{k}'], q[f'mask_C{k}'] = nll('R-std', k, s), nll('R-mask', k, s)
    q['drec'] = np.mean([q[f'drec_C{k}'] for k in (1, 2, 3)])
    q['dclean'] = nll('R-mask', 0, SETS[0]) - nll('R-std', 0, SETS[0])
    q['std_C0'], q['mask_C0'] = nll('R-std', 0, SETS[0]), nll('R-mask', 0, SETS[0])
    return q


point = quantities(np.arange(S), np.ones(N))
rng = np.random.default_rng(0)
gidx = [np.where(src == e)[0] for e in np.unique(src)]
draws = []
for _ in range(N_BOOT):
    w = np.zeros(N)
    for i in rng.integers(0, len(gidx), len(gidx)):
        w[gidx[i]] += 1
    draws.append(quantities(rng.integers(0, S, S), w))
D = pd.DataFrame(draws)
CI = {k: np.percentile(D[k], [2.5, 97.5]) for k in D}


def show(k, label):
    lo, hi = CI[k]
    print(f"  {label:<58} {point[k]:+.4f} [{lo:+.4f}, {hi:+.4f}]")


print(f"NLL (mean of per-seed NLL), C0 all MCIS: R-std {point['std_C0']:.4f} | R-mask {point['mask_C0']:.4f}")
for k in (1, 2, 3):
    print(f"  {COND[k - 1]:<32} n={SETS[k].sum():4d}: R-std {point[f'std_C{k}']:.4f} | R-mask {point[f'mask_C{k}']:.4f}")
print("\n== primary ==")
show('drec', "Δ_rec = mean_k,s [NLL_std − NLL_mask] under C1–C3")
print("\n== secondary ==")
show('dclean', "Δ_clean = NLL_mask − NLL_std on C0, all MCIS (> 0 = cost)")
for k in (1, 2, 3):
    show(f'drec_C{k}', f"Δ_rec,{k} ({COND[k - 1]})")
for k in (1, 2, 3):
    show(f'sens_C{k}', f"sensitivity R-std, C{k} − C0 on the C{k} set")
    show(f'sens_mask_C{k}', f"sensitivity R-mask, C{k} − C0 on the C{k} set")
    show(f'clean_on_C{k}', f"NLL_mask − NLL_std under C0 on the C{k} set")
print("\n== recovery share (only where the sensitivity CI lower bound > 0) ==")
for k in (1, 2, 3):
    if CI[f'sens_C{k}'][0] > 0:
        r = D[f'drec_C{k}'] / D[f'sens_C{k}']
        lo, hi = np.percentile(r, [2.5, 97.5])
        print(f"  C{k}: {point[f'drec_C{k}'] / point[f'sens_C{k}']:.2f} [{lo:.2f}, {hi:.2f}]  (not a share of recovered information)")
    else:
        print(f"  C{k}: not reported (sensitivity CI includes 0 or is negative); see the two NLL differences above")

lo, hi = CI['drec']
verdict = ('MASKED TRAINING HELPS ADAPTATION to the interventions tried' if lo > 0 else
           'MASKED TRAINING MAKES THE FORECAST UNDER MASKING WORSE' if hi < 0 else
           'NOT SHOWN that this policy helps; cannot tell lost evidence from lack of adaptation')
print(f"\n== G28 reading (fixed rule): {verdict} ==")
if lo > 0 and CI['dclean'][0] > 0:
    print("   with a cost on the data as they are (Δ_clean CI above 0): a trade-off, not robustness without cost")

print("\n== descriptive: UAR (per-seed mean / 10-seed ensemble) and ensemble NLL ==")
rows = []
for a in ARMS:
    for k in range(4):
        s = SETS[k]
        per = [war_uar(OOF[a][si, k, s].argmax(1), y_all[s], 7)[1] for si in range(S)]
        ens = OOF[a][:, k, s].mean(0)
        rows.append({'arm': a, 'condition': ['C0', 'C1', 'C2', 'C3'][k],
                     'n': int(s.sum()), 'UAR_seed_mean': np.mean(per), 'UAR_ensemble': war_uar(ens.argmax(1), y_all[s], 7)[1],
                     'NLL_seed_mean': LOSS[a][:, k, s].mean(), 'NLL_ensemble': nll_mean(ens, y_all[s])})
SUM = pd.DataFrame(rows)
print(SUM.round(4).to_string(index=False))
SUM.to_csv(f"{OUT_DIR}/g28_summary.csv", index=False)
pd.DataFrame({k: [point[k], CI[k][0], CI[k][1]] for k in point}, index=['point', 'lo', 'hi']).T.to_csv(
    f"{OUT_DIR}/g28_estimands.csv")
print("saved g28_summary.csv and g28_estimands.csv")
"""),
]
assert "train_eval(" not in G28[-3][1] and "G13" in G28[0][1]


# ---------------------------------------------------------------- G29: context-conditioned evidence pooling before compression
G29 = [
    ("markdown", r"""
# G29 — Context-conditioned face-evidence pooling before compression (5-fold CV, train+val, 10 seeds; test untouched)

**Question.** RoleNet compresses the frames of each (role, clip) cell into one token with FramePool, whose frame scores
depend on the frame alone: h_t = φ(x_t), a_t = softmax_t(wᵀh_t), f = Σ a_t h_t. The Transformer sees context only after
this choice. Does letting the context choose the frames (before compression) help the forecast, compared with a
stronger context-free pooling and with adding the same context after pooling?

**Arms** (same folds, inner early-stop episodes, hyper-parameters, frame budget MAXF and 9 face tokens as G14; the
same φ = FramePool's projection; absent cells use the learned absent token in every arm):

| Arm | Cell token f_{r,k} | Role |
|---|---|---|
| `P0` | FramePool: softmax_t(wᵀh_t) | current RoleNet |
| `P1` | mean_t h_t (same φ) | does the current frame selection matter? |
| `P2` | softmax_t(MLP(h_t)), MLP = d→d_a→d_a→1 with tanh, d_a = d | stronger context-free scorer, scorer size matched to P4 |
| `P3` | softmax_t(vᵀtanh(W_h h_t + b)) pooled, **then** f + U c_{r,k} on observed cells | context after pooling |
| `P4` | softmax_t(vᵀtanh(W_h h_t + W_c c_{r,k} + b)) pooled; values stay h_t | **context in the frame scores** (proposed) |
| `R` | no cell pooling: every frame token h_t + e_r + e_k (and one absent token per empty cell) enters the Transformer | reference "no compression" |

**Context code c_{r,k} (P3 and P4, identical module g_θ).** Preliminary cell summaries h̄_{r,k} = mean_t h_t (absent token
if empty) + e_r + e_k, together with the 6 speech/scene tokens of clips I–III, go through **one** pre-norm Transformer
layer (d = 128, 4 heads, FFN 2d); c_{r,k} is its output at cell (r,k). Modality dropout is drawn **before** g_θ, and
g_θ does not read a dropped branch. W_c (P4) and U (P3) are zero-initialised and the scorer (W_h, b, v) is shared in
form, so P3 and P4 start as the same function and differ only in where the context enters. The post-pooling map in P3
(U, d×d) and the score map in P4 (W_c, d×d_a) have the same size; this is a controlled comparison, not a perfect
isolation, because a context term added to the token can also change content while P4 only reweights frames.

* **Auxiliary heads** (face, A) read the pooled cell token **before** any post-pooling context term (P3: f, not
  f + U c). In P4 that token is a context-weighted average of the same frame values. In R they read the cell means h̄.
* A cell with a single observation gives the same token under any normalised pooling, so P4 can only change cells
  with ≥ 2 frames; their share is reported in step 0.
* R uses exactly the frames of P0–P4 (MAXF per cell), the same φ, role/clip embeddings and absent handling.

**Checkpoint selection:** the G14 rule for every arm (UAR on the inner-dev episodes, same patience and maximum epochs).

**Estimand and interval (fixed before running).** NLL is the only deciding metric: NLL_{s,a} = mean NLL of seed s's
out-of-fold probabilities of arm a over all 2,421 MCIS; quantities are means of per-seed NLL (not the ensemble NLL).
* **Δ_43 = NLL(P3) − NLL(P4)** and **Δ_42 = NLL(P2) − NLL(P4)** (positive = P4 better).
* Interval: 2,000 bootstrap draws resampling the 10 seeds and the 45 episodes, the same draws for all arms.
* **Main hypothesis confirmed only if both CIs are entirely above 0.**
* Reported in every case (no further winning rule): NLL(P0) − NLL(P4), NLL(R) − NLL(P4), NLL(P1) − NLL(P0),
  per-arm UAR (descriptive), training time and peak GPU memory per run. If the main hypothesis is confirmed, the
  report says whether P4 is also better than, indistinguishable from, or worse than the current RoleNet (P0) by the CI
  of NLL(P0) − NLL(P4); only "better" allows calling P4 an improvement of RoleNet.

**Step 0 (descriptive only, not a gate).** Observations per cell by role (0 / 1 / 2 / > 2), share of cells P4 can
reweight, and the within-cell share of variance (within / (within + between cells), per dimension, averaged) separately
for the HSEmotion PCA block and the 18 geometric features. O can hold several people, so its within-cell variance is not
purely temporal.
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # as G14")
        .split("ARMS = [")[0] + """ARMS = ['P0', 'P1', 'P2', 'P3', 'P4', 'R']
EXPERIMENTS = [("RoleNet", 'role', FULL)]    # only used by the shared model cell's parameter print
N_BOOT = 2000
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8],
    ("markdown", "## Step 0 — observations per cell (descriptive)"),
    ("code", r"""
NOBS = FMASK.sum(-1).cpu().numpy()                     # [N, role, clip] frames kept per cell
rows = []
for r, name in enumerate('ALO'):
    v = NOBS[:, r].ravel()
    rows.append({'role': name, '0': (v == 0).mean(), '1': (v == 1).mean(), '2': (v == 2).mean(), '>2': (v > 2).mean(),
                 'mean frames | observed': v[v > 0].mean() if (v > 0).any() else np.nan})
print("share of (MCIS, clip) cells by number of kept frames:")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"cells P4 can reweight (>= 2 frames): {(NOBS >= 2).mean() * 100:.1f}% of all cells, "
      f"{(NOBS >= 2).sum() / max((NOBS >= 1).sum(), 1) * 100:.1f}% of observed cells; "
      f"MCIS with no such cell: {((NOBS >= 2).sum((1, 2)) == 0).mean() * 100:.1f}%")
"""),
    ("markdown", "## Pooling variants"),
    ("code", r"""
class RoleNetPool(RoleNet):
    # RoleNet (G8b, all switches on) with the cell pooling chosen by `arm`; the rest of the model is unchanged.
    def __init__(self, arm, d=RN['D']):
        super().__init__(FULL, d)
        self.arm, da = arm, d
        self.score2 = nn.Sequential(nn.Linear(d, da), nn.Tanh(), nn.Linear(da, da), nn.Tanh(), nn.Linear(da, 1))   # P2
        self.Wh, self.v = nn.Linear(d, da), nn.Linear(da, 1, bias=False)                                            # P3, P4
        self.Wc = nn.Linear(d, da, bias=False)                                                                      # P4
        self.U = nn.Linear(d, d, bias=False)                                                                        # P3
        nn.init.zeros_(self.Wc.weight); nn.init.zeros_(self.U.weight)
        layer = nn.TransformerEncoderLayer(d, RN['heads'], 2 * d, RN['dropout'], batch_first=True, norm_first=True)
        self.g = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)                                       # P3, P4
        used = {'P0': ['pool.score'], 'P1': [], 'P2': ['score2'], 'P3': ['Wh', 'v', 'U', 'g'],
                'P4': ['Wh', 'v', 'Wc', 'g'], 'R': []}[arm]
        extra = ['pool.score', 'score2', 'Wh', 'v', 'Wc', 'U', 'g']
        self.unused = [e for e in extra if e not in used]

    def active_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not any(n.startswith(u + '.') for u in self.unused))

    @staticmethod
    def attend(s, h, m):
        a = s.squeeze(-1).masked_fill(~m, -1e4)
        w = torch.softmax(a, -1) * m.float()
        return (w.unsqueeze(-1) * h).sum(-2)

    def forward(self, ix, train=False):
        B, arm, aux = len(ix), self.arm, {}
        x, m = FACE[ix], FMASK[ix]                                            # [B, 3, 3, F, fin], [B, 3, 3, F]
        h = self.pool.proj(x.float())                                        # shared φ, [B, 3, 3, F, d]
        present = m.any(-1)
        mf = m.float().unsqueeze(-1)
        hbar = (h * mf).sum(-2) / mf.sum(-2).clamp(min=1)                    # [B, 3, 3, d]
        absent = self.absent.unsqueeze(0).expand(B, -1, -1, -1)
        emb = self.face_role[None, :, None] + self.clip_emb[None, None]
        spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.ctx_role[0]
        scn = self.scene(SCN[ix]) + self.ctx_role[1]
        ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)        # [B, 6, d]
        drop_face = drop_ctx = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        if train:                                                             # modality dropout, drawn before g_θ
            u = torch.rand(B, device=DEVICE)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
        prelim = (torch.where(present.unsqueeze(-1), hbar, absent) + emb).reshape(B, 9, -1)
        if arm in ('P3', 'P4'):
            kpm = torch.cat([drop_face.unsqueeze(1).expand(-1, 9), drop_ctx.unsqueeze(1).expand(-1, 6)], 1)
            c = self.g(torch.cat([prelim, ct], 1), src_key_padding_mask=kpm)[:, :9].reshape(B, 3, 3, -1)
        if arm == 'P0':
            f = self.attend(self.pool.score(h), h, m)
        elif arm in ('P1', 'R'):
            f = hbar
        elif arm == 'P2':
            f = self.attend(self.score2(h), h, m)
        elif arm == 'P3':
            f = self.attend(self.v(torch.tanh(self.Wh(h))), h, m)
        else:
            f = self.attend(self.v(torch.tanh(self.Wh(h) + self.Wc(c).unsqueeze(-2))), h, m)
        cell = torch.where(present.unsqueeze(-1), f, absent) + emb            # pooled token before any context term
        aux['face'] = (self.head_face(cell.reshape(B, 9, -1).mean(1)), YB[ix], RN['aux_w'])
        tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
        aux['A'] = (self.head_A(cell[:, 0, 2]), tA, RN['a_w'])
        aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        if arm == 'P3':
            cell = cell + torch.where(present.unsqueeze(-1), self.U(c), torch.zeros_like(c))
        if arm == 'R':
            fr = (h + emb.unsqueeze(-2)).reshape(B, -1, h.shape[-1])          # every kept frame is a token
            ab = (absent + emb).reshape(B, 9, -1)
            face_toks = torch.cat([fr, ab], 1)
            vf = torch.cat([m.reshape(B, -1), ~present.reshape(B, 9)], 1)
        else:
            face_toks, vf = cell.reshape(B, 9, -1), torch.ones(B, 9, dtype=torch.bool, device=DEVICE)
        vf = vf & ~drop_face.unsqueeze(1)
        vc = torch.ones(B, 6, dtype=torch.bool, device=DEVICE) & ~drop_ctx.unsqueeze(1)
        toks = torch.cat([self.query.expand(B, -1, -1), face_toks, ct], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=DEVICE), vf, vc], 1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


for a in ARMS:
    MAKE[a] = (lambda arm: (lambda cfg: RoleNetPool(arm)))(a)
    HP[a] = HP['role']
print("active parameters:", {a: f"{RoleNetPool(a).active_params() / 1e6:.3f}M" for a in ARMS})
_m = RoleNetPool('P4')
print("P4 scorer:", sum(p.numel() for n in ('Wh', 'v', 'Wc') for p in getattr(_m, n).parameters()),
      "| P2 scorer:", sum(p.numel() for p in _m.score2.parameters()),
      "| P3 scorer + U:", sum(p.numel() for n in ('Wh', 'v', 'U') for p in getattr(_m, n).parameters()),
      "| g_θ:", sum(p.numel() for p in _m.g.parameters()))
"""),
    ("markdown", "## 5-fold episode cross-validation (same folds, early-stop episodes and seeds as G14)"),
    ("code", r"""
import re

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
print("fold sizes (MCIS):", load_)
y_all = DEV.yB.values
src = DEV.source_folder.values


def variance_share(Fn, M, dims):
    # within-cell share of variance per dimension, averaged over dimensions; cells with >= 2 frames for 'within'
    out = {}
    for r, name in enumerate('ALO'):
        X, mk = Fn[:, r][..., dims].reshape(-1, Fn.shape[3], len(dims)), M[:, r].reshape(-1, Fn.shape[3])
        cnt = mk.sum(1)
        obs = cnt >= 1
        mu = (X * mk[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]
        multi = cnt >= 2
        if multi.sum() < 10:
            out[name] = np.nan; continue
        dev_ = ((X - mu[:, None]) ** 2 * mk[..., None]).sum(1)
        within = (dev_[multi] / (cnt[multi, None] - 1)).mean(0)
        between = mu[obs].var(0)
        out[name] = float(np.mean(within / np.maximum(within + between, 1e-12)))
    return out


OOF = {a: np.full((len(SEEDS), N, 7), np.nan, np.float32) for a in ARMS}
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    if f == 0:                                                            # step 0, descriptive (fold-0 PCA)
        Fn, Mn = FACE.float().cpu().numpy(), FMASK.cpu().numpy()
        emo = list(range(PCA_DIM)) if HAS_EMB else []
        geo = list(range(FDIM - 18, FDIM))
        if emo:
            print("step 0, within-cell share of variance, HSEmotion PCA block:", variance_share(Fn, Mn, emo))
        print("step 0, within-cell share of variance, geometric block:     ", variance_share(Fn, Mn, geo))
        del Fn
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for a in ARMS:
        for si, seed in enumerate(SEEDS):
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            p, sel = train_eval(a, FULL, tr, dev, te, seed + 1000 * f)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            OOF[a][si, te_rows] = p
            yt = y_all[te_rows]
            nll = float(-np.log(np.clip(p[np.arange(len(yt)), yt], 1e-7, None)).mean())
            w, u = war_uar(p.argmax(1), yt, 7)
            log.append({'fold': f, 'arm': a, 'seed': seed, 'sel_UAR': sel, 'UAR': u, 'WAR': w, 'NLL': nll,
                        'train_s': time.time() - t1,
                        'peak_MB': torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else np.nan})
            print(f"fold {f} {a:<3} seed {seed:>3}: sel UAR {sel:5.2f} | NLL {nll:.4f} UAR {u:5.2f} | "
                  f"{log[-1]['train_s']:.0f} s | {(time.time() - t0) / 60:.1f} min", flush=True)
            torch.cuda.empty_cache()

assert all(not np.isnan(v).any() for v in OOF.values())
LOG = pd.DataFrame(log)
LOG.to_csv(f"{OUT_DIR}/g29_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g29_oof_probs.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         n_obs=NOBS, **OOF)
print("saved g29_oof_probs.npz (arm arrays: [seed, MCIS, 7]) and g29_fold_seed_log.csv")
"""),
    ("markdown", "## Results (fixed estimand: mean of per-seed NLL; two-level bootstrap with shared draws)"),
    ("code", r"""
S = len(SEEDS)
LOSS = {a: -np.log(np.clip(np.take_along_axis(OOF[a], y_all[None, :, None], -1)[..., 0], 1e-7, None)) for a in ARMS}
PAIRS = {'D43 = NLL(P3) - NLL(P4)': ('P3', 'P4'), 'D42 = NLL(P2) - NLL(P4)': ('P2', 'P4'),
         'NLL(P0) - NLL(P4)': ('P0', 'P4'), 'NLL(R) - NLL(P4)': ('R', 'P4'), 'NLL(P1) - NLL(P0)': ('P1', 'P0'),
         'NLL(P2) - NLL(P0)': ('P2', 'P0'), 'NLL(P3) - NLL(P0)': ('P3', 'P0'), 'NLL(R) - NLL(P0)': ('R', 'P0')}


def quantities(sidx, w):
    nl = {a: float((LOSS[a][sidx].mean(0) * w).sum() / w.sum()) for a in ARMS}
    q = {f'NLL {a}': nl[a] for a in ARMS}
    q.update({k: nl[x] - nl[y] for k, (x, y) in PAIRS.items()})
    return q


point = quantities(np.arange(S), np.ones(N))
rng = np.random.default_rng(0)
gidx = [np.where(src == e)[0] for e in np.unique(src)]
draws = []
for _ in range(N_BOOT):
    w = np.zeros(N)
    for i in rng.integers(0, len(gidx), len(gidx)):
        w[gidx[i]] += 1
    draws.append(quantities(rng.integers(0, S, S), w))
D = pd.DataFrame(draws)
CI = {k: np.percentile(D[k], [2.5, 97.5]) for k in D}

print("mean per-seed NLL (all MCIS):", {a: round(point[f'NLL {a}'], 4) for a in ARMS})
print("\n== primary (positive = P4 better) ==")
for k in list(PAIRS)[:2]:
    print(f"  {k:<26} {point[k]:+.4f} [{CI[k][0]:+.4f}, {CI[k][1]:+.4f}]")
print("\n== reported in every case ==")
for k in list(PAIRS)[2:]:
    print(f"  {k:<26} {point[k]:+.4f} [{CI[k][0]:+.4f}, {CI[k][1]:+.4f}]")

ok = CI['D43 = NLL(P3) - NLL(P4)'][0] > 0 and CI['D42 = NLL(P2) - NLL(P4)'][0] > 0
print("\n== G29 decision (fixed rule): " + ("CONFIRMED — context in the frame scores beats both controls" if ok else
      "NOT CONFIRMED — P4 does not beat both P3 and P2") + " ==")
if ok:
    lo, hi = CI['NLL(P0) - NLL(P4)']
    print("   relative to the current RoleNet (P0): " + ("better → P4 may be called an improvement of RoleNet" if lo > 0 else
          "worse → supports the conditional design within the controls only, not an improvement of RoleNet" if hi < 0
          else "not distinguishable → not an improvement of RoleNet"))

rows = []
for a in ARMS:
    per = [war_uar(OOF[a][s].argmax(1), y_all, 7)[1] for s in range(S)]
    la = LOG[LOG.arm == a]
    rows.append({'arm': a, 'NLL_seed_mean': point[f'NLL {a}'], 'NLL_ensemble': float(-np.log(np.clip(
        OOF[a].mean(0)[np.arange(N), y_all], 1e-7, None)).mean()), 'UAR_seed_mean': np.mean(per), 'UAR_seed_sd': np.std(per),
        'UAR_ensemble': war_uar(OOF[a].mean(0).argmax(1), y_all, 7)[1], 'train_s_mean': la.train_s.mean(),
        'peak_MB_mean': la.peak_MB.mean(), 'active_params': RoleNetPool(a).active_params()})
SUM = pd.DataFrame(rows)
print("\n== per arm (UAR, time, memory descriptive) ==")
print(SUM.round(4).to_string(index=False))
SUM.to_csv(f"{OUT_DIR}/g29_summary.csv", index=False)
pd.DataFrame({k: [point[k], CI[k][0], CI[k][1]] for k in point}, index=['point', 'lo', 'hi']).T.to_csv(
    f"{OUT_DIR}/g29_estimands.csv")
print("saved g29_summary.csv and g29_estimands.csv")
"""),
]


# ---------------------------------------------------------------- G30: context-conditioned auxiliary supervision with gradient routing
G30 = [
    ("markdown", r"""
# G30 — Auxiliary face supervision of RoleNet: whole-face head vs L-only vs L + context, with and without gradient routing (5-fold CV, train+val, 10 seeds; test untouched)

**Question.** RoleNet's face auxiliary head predicts B from the mean of all 9 face tokens (absent tokens included), so
the auxiliary loss sends the same token-level gradient to every cell. Does supervising the listener representation
**in the presence of the context** forecast better than supervising the mean of all faces, and does it matter whether
the auxiliary loss may also update the context path?

Hypothesis: supervising the L representation conditioned on the context forecasts better than supervising the mean of
all face tokens. This is context-conditioned auxiliary supervision with gradient routing. It is **not** claimed to
learn "L's residual information" or to reduce mirroring bias; the MLP head has no additive (product-of-experts)
decomposition, and the context path keeps changing through the main and other losses.

**Arms** (only the face auxiliary head changes; the A head (Y_A) and the context head stay as in RoleNet; weight 0.3):

| Arm | Face auxiliary head |
|---|---|
| `S0` | original: CE(h_face(mean of 9 face tokens), Y_B), all MCIS |
| `S1` | none |
| `S2` | L only: h_ψ(z_L, 0, 0, 0, 0) |
| `S3` | L + context: h_ψ(z_L, z_AO, z_speech, z_scene, m_AO) |
| `S4` | L + stop-gradient(context): h_ψ(z_L, sg(z_AO), sg(z_speech), sg(z_scene), m_AO) |

* z_L = mean of the **observed** L face tokens (clips I–III, before the Transformer, absent tokens excluded). MCIS
  without any L observation get no auxiliary loss in S2–S4 (target ignored); S2–S4 use the same MCIS set and the same
  normalisation (mean over valid MCIS of the batch). S0 keeps its original behaviour on all MCIS.
* z_AO = mean of the observed A/O face tokens (zero vector and m_AO = 0 if none); z_speech, z_scene = means of the three
  speech / scene tokens. z_C is **not** "everything except L": scene and voice cues may still carry L information.
* h_ψ: per-group LayerNorm, concatenation with m_AO, Linear(4d + 1 → 128), GELU, Linear(128 → 7). S2–S4 use the same
  module; S2 feeds zeros (and m_AO = 0) instead of the context groups; S4 detaches the context vectors **before** the
  head's LayerNorms (so the head's own LayerNorm parameters still learn).
* The auxiliary heads read the tokens **before** modality dropout (as in RoleNet); modality dropout only masks the main
  Transformer's input. Same rule in every arm.
* All modules are created in every arm (same initialisation per seed); active parameter counts are reported.

**Checkpoint selection and estimand (fixed before running).** Every arm selects the epoch by **NLL on the inner-dev
episodes** (same patience and maximum epochs as G14). Deciding metric: NLL_{s,a} = mean NLL of seed s's out-of-fold
probabilities of arm a over all 2,421 MCIS; quantities are means of per-seed NLL. Interval: 2,000 bootstrap draws over
the 10 seeds and the 45 episodes, the same draws for all arms.

**Reading rules (fixed before running).**
* **Method (gradient routing) supported** iff the CI of NLL(S3) − NLL(S4) is entirely above 0.
* **Worth adding to RoleNet** iff, in addition, the CIs of NLL(S0) − NLL(S4) **and** NLL(S1) − NLL(S4) are entirely above 0.
  Beating S0 but not S1 only shows that fixing or removing the old head helps.
* Always reported: NLL(S2) − NLL(S4) (context in the routed design), NLL(S1) − NLL(S0), per-arm UAR and ensemble NLL
  (descriptive). S4 − S0 is not an isolation of gradient routing (input sources and the set of supervised MCIS differ).

**Mechanism diagnostic (not a gate).** For each frozen outer model: probes on its own pre-Transformer embeddings,
fitted (with C chosen by grouped inner CV) on the outer-fold training rows with an L observation and scored on the
outer fold: q_C reads z_C = [z_AO, z_speech, z_scene, m_AO], q_LC reads [z_L, z_C]. Δ_probe = NLL(q_C) − NLL(q_LC) is
the probe's conditional predictive gain from z_L; it depends on the probe family and is not I(Y_B; z_L | z_C). Probes
are never fitted across encoders.

Saved for later analyses: every selected checkpoint, inner-dev and outer logits (for e.g. temperature scaling).
"""),
    ("code", G13[1][1]
        .replace("SEEDS = [42, 123, 456]                       # as G8b / G11 / G12",
                 "SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]    # as G14")
        .split("ARMS = [")[0] + """ARMS = ['S0', 'S1', 'S2', 'S3', 'S4']
EXPERIMENTS = [("RoleNet", 'role', FULL)]    # only used by the shared model cell's parameter print
N_BOOT = 2000
AUX_HIDDEN = 128
RUN_PROBE = True
PROBE_C = [0.01, 0.1, 1.0]
"""),
    G13[2], G13[3], G13[4], G13[5], G13[6], G13[7], G13[8],
    ("markdown", "## RoleNet with the face auxiliary head chosen by arm"),
    ("code", r"""
class RoleNetAux(RoleNet):
    def __init__(self, arm, d=RN['D']):
        super().__init__(FULL, d)
        self.arm = arm
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])          # z_L, z_AO, z_speech, z_scene
        self.aux_mlp = nn.Sequential(nn.Linear(4 * d + 1, AUX_HIDDEN), nn.GELU(), nn.Linear(AUX_HIDDEN, 7))
        self.unused = {'S0': ['ln', 'aux_mlp'], 'S1': ['ln', 'aux_mlp', 'head_face'],
                       'S2': ['head_face'], 'S3': ['head_face'], 'S4': ['head_face']}[arm]

    def active_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not any(n.startswith(u + '.') for u in self.unused))

    def tokens(self, ix):
        B = len(ix)
        h, present = self.pool(FACE[ix], FMASK[ix])                           # [B, 3, 3, d], [B, 3, 3]
        h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
        h = h + self.face_role[None, :, None] + self.clip_emb[None, None]
        spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.ctx_role[0]
        scn = self.scene(SCN[ix]) + self.ctx_role[1]
        ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)         # [B, 6, d]
        return h, present, ct

    @staticmethod
    def groups(h, present, ct):
        mL = present[:, 1].float().unsqueeze(-1)                              # [B, 3, 1]
        zL = (h[:, 1] * mL).sum(1) / mL.sum(1).clamp(min=1)
        validL = present[:, 1].any(-1)
        hAO, mAO = h[:, [0, 2]].reshape(len(h), 6, -1), present[:, [0, 2]].reshape(len(h), 6).float().unsqueeze(-1)
        zAO = (hAO * mAO).sum(1) / mAO.sum(1).clamp(min=1)
        fAO = (mAO.sum(1) > 0).float()                                         # [B, 1]
        zAO = zAO * fAO
        return zL, validL, zAO, fAO, ct[:, :3].mean(1), ct[:, 3:].mean(1)

    def forward(self, ix, train=False):
        B, arm, aux = len(ix), self.arm, {}
        h, present, ct = self.tokens(ix)
        ft = h.reshape(B, 9, -1)
        if arm == 'S0':
            aux['face'] = (self.head_face(ft.mean(1)), YB[ix], RN['aux_w'])
        elif arm in ('S2', 'S3', 'S4'):
            zL, validL, zAO, fAO, zsp, zsc = self.groups(h, present, ct)
            ctx = [zAO, zsp, zsc]
            if arm == 'S2':
                ctx, fAO = [torch.zeros_like(c) for c in ctx], torch.zeros_like(fAO)
            elif arm == 'S4':
                ctx = [c.detach() for c in ctx]                               # stop-gradient before the head's LayerNorm
            inp = torch.cat([self.ln[0](zL)] + [self.ln[i + 1](c) for i, c in enumerate(ctx)] + [fAO], -1)
            aux['face'] = (self.aux_mlp(inp), torch.where(validL, YB[ix], torch.full_like(YB[ix], -100)), RN['aux_w'])
        tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
        aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1), ft, ct], 1)
        valid = torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)
        if train:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            valid[:, 1:10] &= ~drop_face.unsqueeze(1)
            valid[:, 10:] &= ~drop_ctx.unsqueeze(1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux

    @torch.no_grad()
    def embed(self, ix, bs=512):
        self.eval()
        out = []
        for i in range(0, len(ix), bs):
            zL, validL, zAO, fAO, zsp, zsc = self.groups(*self.tokens(ix[i:i + bs]))
            out.append(torch.cat([zL, zAO, zsp, zsc, fAO, validL.float().unsqueeze(-1)], -1).cpu())
        return torch.cat(out).numpy()                                         # [n, 4d + 2]: zL | zC (zAO, sp, sc, fAO) | validL


def logits_of(model, ix, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            out.append(model(ix[i:i + bs])[0].float().cpu())
    return torch.cat(out).numpy()


def nll_logits(z, y):
    z = z - z.max(1, keepdims=True)
    return float((np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]).mean())


def train_eval_g30(arm, tr, dev, te, seed, ckpt_path):
    seed_all(seed)
    hp = HP['role']
    model = RoleNetAux(arm).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    y_dev = YB[dev].cpu().numpy()
    best, best_state, bad, best_ep = np.inf, None, 0, -1
    for ep in range(hp['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(perm), hp['batch']):
            j = perm[i:i + hp['batch']]
            logits, aux = model(j, train=True)
            loss = F.cross_entropy(logits, YB[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        J = nll_logits(logits_of(model, dev), y_dev)
        if J < best:
            best, bad, best_ep = J, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    return model, logits_of(model, dev), logits_of(model, te), best, best_ep


for a in ARMS:
    print(a, "active parameters:", f"{RoleNetAux(a).active_params() / 1e6:.4f}M")
"""),
    ("markdown", "## Probe helper (mechanism diagnostic)"),
    ("code", r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

D_ = RN['D']


def probe_losses(E_tr, y_tr, g_tr, E_te, y_te):
    # per-row NLL on the evaluation rows for q_C (z_C) and q_LC ([z_L, z_C]); C by grouped inner CV on training rows
    out = []
    for cols in (slice(D_, 4 * D_ + 1), slice(0, 4 * D_ + 1)):
        Xtr, Xte = E_tr[:, cols], E_te[:, cols]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        score = {}
        for C in PROBE_C:
            s = []
            for a, b in GroupKFold(3).split(Xtr, y_tr, g_tr):
                m = LogisticRegression(C=C, max_iter=500).fit(Xtr[a], y_tr[a])
                p = np.full((len(b), 7), 1e-6); p[:, m.classes_] = m.predict_proba(Xtr[b])
                s.append(-np.log(np.clip(p[np.arange(len(b)), y_tr[b]] / p.sum(1), 1e-7, None)).mean())
            score[C] = np.mean(s)
        m = LogisticRegression(C=min(score, key=score.get), max_iter=500).fit(Xtr, y_tr)
        p = np.full((len(Xte), 7), 1e-6); p[:, m.classes_] = m.predict_proba(Xte)
        out.append(-np.log(np.clip(p[np.arange(len(y_te)), y_te] / p.sum(1), 1e-7, None)))
    return np.stack(out, 1)                                                  # [n_te, 2]: q_C, q_LC
"""),
    ("markdown", "## 5-fold episode cross-validation (same folds, early-stop episodes and seeds as G14)"),
    ("code", r"""
import re

sizes = DEV.source_folder.value_counts()
order = list(sizes.index)
random.Random(0).shuffle(order)
order = sorted(order, key=lambda e: -sizes[e])
load_, FOLD = [0] * N_OUTER, {}
for e in order:
    f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
fold_of_row = DEV.source_folder.map(FOLD).values
y_all = DEV.yB.values
src = DEV.source_folder.values
print("fold sizes (MCIS):", load_, "| MCIS with an L observation:", int(FMASK[:, 1].any(-1).any(-1).sum()))

os.makedirs(f"{OUT_DIR}/g30_ckpt", exist_ok=True)
LOGIT = {a: np.full((len(SEEDS), N, 7), np.nan, np.float32) for a in ARMS}
PROBE = {a: np.full((len(SEEDS), N, 2), np.nan, np.float32) for a in ARMS}
DEVLOG, DEVROWS = {}, {}
log = []
t0 = time.time()
for f in range(N_OUTER):
    tr_eps = [e for e in EPS if FOLD[e] != f]
    dev_eps = sorted(random.Random(100 + f).sample(tr_eps, N_INNER_DEV))
    trr = np.where(np.isin(src, tr_eps))[0]
    fit_rows = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev_rows = np.where(np.isin(src, dev_eps))[0]
    te_rows = np.where(fold_of_row == f)[0]
    DEVROWS[f] = dev_rows
    fit_clips = sorted(set(DEV.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel()))
    FACE, POOL, var = build_face_tensors(fit_clips)
    print(f"fold {f}: train {len(fit_rows)} | early-stop {len(dev_rows)} | eval {len(te_rows)}", flush=True)
    tr, dev, te = T(fit_rows), T(dev_rows), T(te_rows)
    for a in ARMS:
        DEVLOG[(a, f)] = np.zeros((len(SEEDS), len(dev_rows), 7), np.float32)
        for si, seed in enumerate(SEEDS):
            t1 = time.time()
            model, zd, zt, J, ep = train_eval_g30(a, tr, dev, te, seed + 1000 * f,
                                                  f"{OUT_DIR}/g30_ckpt/{a}_fold{f}_seed{seed}.pt")
            LOGIT[a][si, te_rows] = zt
            DEVLOG[(a, f)][si] = zd
            yt = y_all[te_rows]
            p = np.exp(zt - zt.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            w, u = war_uar(p.argmax(1), yt, 7)
            rec = {'fold': f, 'arm': a, 'seed': seed, 'dev_NLL': J, 'best_epoch': ep, 'NLL': nll_logits(zt, yt),
                   'UAR': u, 'WAR': w, 'train_s': time.time() - t1}
            if RUN_PROBE:
                t2 = time.time()
                E_tr, E_te = model.embed(T(trr)), model.embed(te)
                ktr, kte = E_tr[:, -1] > 0, E_te[:, -1] > 0
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    pl = probe_losses(E_tr[ktr, :-1], y_all[trr][ktr], src[trr][ktr], E_te[kte, :-1], yt[kte])
                PROBE[a][si, te_rows[kte]] = pl
                rec.update({'probe_qC': pl[:, 0].mean(), 'probe_qLC': pl[:, 1].mean(), 'probe_s': time.time() - t2})
            log.append(rec)
            print(f"fold {f} {a} seed {seed:>3}: dev NLL {J:.4f} (epoch {ep}) | NLL {rec['NLL']:.4f} UAR {u:5.2f}"
                  + (f" | probe Δ {rec['probe_qC'] - rec['probe_qLC']:+.4f}" if RUN_PROBE else "")
                  + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
            del model
            torch.cuda.empty_cache()

assert all(not np.isnan(v).any() for v in LOGIT.values())
LOG = pd.DataFrame(log)
LOG.to_csv(f"{OUT_DIR}/g30_fold_seed_log.csv", index=False)
np.savez(f"{OUT_DIR}/g30_logits.npz", sample_id=DEV.sample_id.values, fold=fold_of_row, y=y_all, src=src,
         validL=FMASK[:, 1].any(-1).any(-1).cpu().numpy(),
         **{f"outer_{a}": v for a, v in LOGIT.items()}, **{f"probe_{a}": v for a, v in PROBE.items()},
         **{f"dev_{a}_fold{f}": v for (a, f), v in DEVLOG.items()}, **{f"devrows_fold{f}": r for f, r in DEVROWS.items()})
print("saved g30_logits.npz (outer logits [seed, MCIS, 7], inner-dev logits per fold, probe losses), "
      "g30_fold_seed_log.csv and the checkpoints in g30_ckpt/")
"""),
    ("markdown", "## Results (fixed estimand: mean of per-seed NLL; two-level bootstrap with shared draws)"),
    ("code", r"""
S = len(SEEDS)


def row_nll(z):
    z = z - z.max(-1, keepdims=True)
    return np.log(np.exp(z).sum(-1)) - np.take_along_axis(z, y_all[None, :, None], -1)[..., 0]


LOSS = {a: row_nll(LOGIT[a]) for a in ARMS}                                # [S, N]
PAIRS = {'NLL(S3) - NLL(S4)': ('S3', 'S4'), 'NLL(S0) - NLL(S4)': ('S0', 'S4'), 'NLL(S1) - NLL(S4)': ('S1', 'S4'),
         'NLL(S2) - NLL(S4)': ('S2', 'S4'), 'NLL(S1) - NLL(S0)': ('S1', 'S0')}
VL = FMASK[:, 1].any(-1).any(-1).cpu().numpy()


def quantities(sidx, w):
    nl = {a: float((LOSS[a][sidx].mean(0) * w).sum() / w.sum()) for a in ARMS}
    q = {f'NLL {a}': nl[a] for a in ARMS}
    q.update({k: nl[x] - nl[y] for k, (x, y) in PAIRS.items()})
    if RUN_PROBE:
        for a in ARMS:
            P = PROBE[a][sidx].mean(0)                                         # [N, 2], NaN where no L
            wv = w * VL
            q[f'probe gain {a}'] = float(((P[:, 0] - P[:, 1]) * wv)[VL].sum() / wv.sum())
        q['probe gain S4 - S3'] = q['probe gain S4'] - q['probe gain S3']
        q['probe gain S4 - S0'] = q['probe gain S4'] - q['probe gain S0']
    return q


point = quantities(np.arange(S), np.ones(N))
rng = np.random.default_rng(0)
gidx = [np.where(src == e)[0] for e in np.unique(src)]
draws = []
for _ in range(N_BOOT):
    w = np.zeros(N)
    for i in rng.integers(0, len(gidx), len(gidx)):
        w[gidx[i]] += 1
    draws.append(quantities(rng.integers(0, S, S), w))
D = pd.DataFrame(draws)
CI = {k: np.percentile(D[k], [2.5, 97.5]) for k in D}
show = lambda k: print(f"  {k:<24} {point[k]:+.4f} [{CI[k][0]:+.4f}, {CI[k][1]:+.4f}]")

print("mean per-seed NLL (all MCIS):", {a: round(point[f'NLL {a}'], 4) for a in ARMS})
print("best epoch (mean):", LOG.groupby('arm').best_epoch.mean().round(1).to_dict())
print("\n== primary: gradient routing ==")
show('NLL(S3) - NLL(S4)')
print("\n== practical value (with the primary) ==")
show('NLL(S0) - NLL(S4)'); show('NLL(S1) - NLL(S4)')
print("\n== always reported ==")
show('NLL(S2) - NLL(S4)'); show('NLL(S1) - NLL(S0)')

lo = {k: CI[k][0] for k in PAIRS}
method = lo['NLL(S3) - NLL(S4)'] > 0
print("\n== G30 reading (fixed rules) ==")
print("  method (gradient routing): " + ("SUPPORTED" if method else "NOT SUPPORTED (CI of NLL(S3) − NLL(S4) not above 0)"))
if method and lo['NLL(S0) - NLL(S4)'] > 0 and lo['NLL(S1) - NLL(S4)'] > 0:
    print("  worth adding to RoleNet: YES (S4 also beats S0 and S1)")
else:
    print("  worth adding to RoleNet: NO")
if lo['NLL(S0) - NLL(S4)'] > 0 and not lo['NLL(S1) - NLL(S4)'] > 0:
    print("  note: S4 beats S0 but not S1 → fixing or removing the old head helps; no evidence for the new head itself")

if RUN_PROBE:
    print("\n== mechanism diagnostic (probe conditional gain from z_L, NLL(q_C) − NLL(q_LC); not a gate) ==")
    for a in ARMS:
        show(f'probe gain {a}')
    show('probe gain S4 - S3'); show('probe gain S4 - S0')

rows = []
for a in ARMS:
    P = np.exp(LOGIT[a] - LOGIT[a].max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    per = [war_uar(P[s].argmax(1), y_all, 7)[1] for s in range(S)]
    ens = P.mean(0)
    rows.append({'arm': a, 'NLL_seed_mean': point[f'NLL {a}'],
                 'NLL_ensemble': float(-np.log(np.clip(ens[np.arange(N), y_all], 1e-7, None)).mean()),
                 'UAR_seed_mean': np.mean(per), 'UAR_seed_sd': np.std(per), 'UAR_ensemble': war_uar(ens.argmax(1), y_all, 7)[1],
                 'best_epoch_mean': LOG[LOG.arm == a].best_epoch.mean(), 'active_params': RoleNetAux(a).active_params()})
SUM = pd.DataFrame(rows)
print("\n== per arm (UAR descriptive) ==")
print(SUM.round(4).to_string(index=False))
SUM.to_csv(f"{OUT_DIR}/g30_summary.csv", index=False)
pd.DataFrame({k: [point[k], CI[k][0], CI[k][1]] for k in point}, index=['point', 'lo', 'hi']).T.to_csv(
    f"{OUT_DIR}/g30_estimands.csv")
print("saved g30_summary.csv and g30_estimands.csv")
"""),
]


if __name__ == "__main__":
    for name, cells in [("g1_llm_recognition.ipynb", G1), ("g2_recognizer_all_labels.ipynb", G2),
                        ("g3_trajectory_forecaster.ipynb", G3), ("g3b_robustness.ipynb", G3B),
                        ("g4_test_preregistered.ipynb", G4), ("g5_episode_cv.ipynb", G5),
                        ("g6a_listener_visibility.ipynb", G6A), ("g6b_listener_expression.ipynb", G6B),
                        ("g7b_faces_into_b1.ipynb", G7B), ("g8a_role_features.ipynb", G8A),
                        ("g8b_rolenet_cv.ipynb", G8B),
                        ("g9_rolenet_plus_cv.ipynb", G9),
                        ("g10_test_preregistered.ipynb", G10),
                        ("g11_channel_ablations_cv.ipynb", G11),
                        ("g12_cs_rolenet_cv.ipynb", G12),
                        ("g13_token_ablations_cv.ipynb", G13),
                        ("g14_ten_seed_confirmation_cv.ipynb", G14),
                        ("g15_dynamics_gates.ipynb", G15),
                        ("g16_time_vs_type.ipynb", G16),
                        ("g17_listening_vs_speaking.ipynb", G17),
                        ("g19_forecastable_distinctions_cv.ipynb", G19),
                        ("g20_interaction_gate_cv.ipynb", G20),
                        ("g23_negative_separability_cv.ipynb", G23),
                        ("g24_appraisal_reaction_cv.ipynb", G24),
                        ("g25_listener_state_vs_reaction.ipynb", G25),
                        ("g26a_clip4_faces.ipynb", G26A),
                        ("g26_responder_pointer_cv.ipynb", G26),
                        ("g27_mention_aggregation_pilot.ipynb", G27),
                        ("g28_masked_listener_training_cv.ipynb", G28),
                        ("g29_context_pooling_cv.ipynb", G29),
                        ("g30_aux_supervision_cv.ipynb", G30),
                        ("m1_meld_prepare_features.ipynb", M1),
                        ("m2_meld_g8a_features.ipynb", M2),
                        ("m3_meld_rolenet.ipynb", M3)]:
        (HERE / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False))
        print("wrote", HERE / name)
