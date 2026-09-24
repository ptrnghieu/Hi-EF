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

if __name__ == "__main__":
    for name, cells in [("g1_llm_recognition.ipynb", G1), ("g2_recognizer_all_labels.ipynb", G2),
                        ("g3_trajectory_forecaster.ipynb", G3), ("g3b_robustness.ipynb", G3B),
                        ("g4_test_preregistered.ipynb", G4), ("g5_episode_cv.ipynb", G5),
                        ("g6a_listener_visibility.ipynb", G6A), ("g6b_listener_expression.ipynb", G6B),
                        ("g7b_faces_into_b1.ipynb", G7B), ("g8a_role_features.ipynb", G8A),
                        ("g8b_rolenet_cv.ipynb", G8B)]:
        (HERE / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False))
        print("wrote", HERE / name)
