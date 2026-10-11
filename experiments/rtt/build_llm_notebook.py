"""Build l1_qwen_vl_zero_few_shot.ipynb: Qwen2.5-VL-7B-Instruct zero-shot and few-shot emotion forecasting on Hi-EF or MELD."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    return {"cells": [{"cell_type": k, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if k == "code" else {})} for k, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


m1 = json.load(open(HERE / "m1_meld_prepare_features.ipynb"))
M1_FETCH = "".join(m1["cells"][3]["source"])
M1_FETCH = M1_FETCH[:M1_FETCH.index("vids, csvs = fetch_meld()")]

CELLS = [
("markdown", """
# L1 — Qwen2.5-VL-7B-Instruct, zero-shot and few-shot, on Hi-EF or MELD

A multimodal LLM baseline for emotion forecasting. Input for each MCIS: clips I–III as **video** (8 evenly spaced frames
per clip, 336×252) with the subtitle of each clip; **no audio** (Qwen2.5-VL has no audio input). Clip IV and speaker
names are never shown. The model is asked which of the seven emotions the *next speaker* will express. The answer is
read **without generation**: the logits of the first token of each of the seven label words at the answer position
give a 7-class distribution (softmax over the seven logits), from which UAR / WAR (plain argmax) are computed.

* **Zero-shot**: instruction + the three clips.
* **Few-shot**: 7 worked examples, one per class, drawn from the training rows of the same dataset (Hi-EF: train+val
  episodes; MELD: train), each shown as subtitles + the middle frame of each clip (224×168) and the true label of the next
  speaker; the example classes are shuffled with a fixed seed. `SHOT_SEEDS` gives independent example sets
  (results are reported per set and as mean).
* Evaluation on the **test split** only (Hi-EF: 409 MCIS; MELD: 1,341 windows), read once by this notebook (an
  additional test access to list in the paper's appendix). The prompt below is fixed before the test run; the smoke
  test uses validation rows only.
* Needs a GPU with about 16 GB per device (two T4 in fp16, or one A100/L4), internet on for the weights.
"""),
("code", "!pip install -q -U transformers accelerate"),
("code", '''
# ======== CONFIG ========
import os, glob
SOURCE = "hief"                         # "hief" or "meld"
OUT_DIR = "/kaggle/working"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DTYPE = "float16"                       # T4 has no bfloat16; use "bfloat16" on A100/L4
N_FRAMES = 8                            # frames per clip for the video input
FRAME_WH = (336, 252)                   # multiples of 28, as required by the vision tower
FRAME_WH_SHOT = (224, 168)              # few-shot example stills (smaller, to fit the memory of a 15 GB GPU)
RESUME_NPZ = None                       # path of a previous l1_<SOURCE>_probs.npz: finished settings are loaded and skipped
SHOT_SEEDS = [0, 1, 2]                  # independent few-shot example sets (set [] to skip few-shot)
RUN_ZERO_SHOT = True
SMOKE_N = 12                            # validation rows printed before the test run
DEBUG_N = None                          # e.g. 20: evaluate only the first 20 test rows

if SOURCE == "hief":
    DATASET_DIR = "/kaggle/input/datasets/ptrnghieu/hi-ef-dataset"
    SPLIT_CSV = "/kaggle/input/datasets/ptrnghieu/hi-ef-split/source_folder_split_seed42.csv"
else:
    MELD_LOCAL = None                   # MELD.Raw folder; None = auto-detect an attached MELD dataset
    WORK = "/tmp/meld"


def find_one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"attach the dataset holding {pattern}")
    return hits[0]


def find_meld_raw():
    out = []
    for base in ["/kaggle/input"] + sorted(glob.glob("/kaggle/input/datasets/*")):
        for d in sorted(glob.glob(base + "/*")):
            n = os.path.basename(d).lower()
            if os.path.isdir(d) and "meld" in n and "m1" not in n and "hief" not in n and "features" not in n:
                out.append(d)
    return out


if SOURCE == "meld":
    DATASET_DIR = find_one("meld_hief")
    SPLIT_CSV = find_one("meld_split.csv")
    if MELD_LOCAL is None:
        _c = find_meld_raw()
        assert _c, "attach the MELD.Raw dataset or set MELD_LOCAL"
        MELD_LOCAL = _c[0]
        print("MELD_LOCAL =", MELD_LOCAL)
'''),
("code", "import numpy as np, pandas as pd\n" + M1_FETCH),
("code", '''
import random, time, cv2
import torch
from PIL import Image

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
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
else:
    vids, csvs = fetch_meld()

    def video_of(c):
        return vids.get((c[:2], int(c[2:7]), int(c[8:])))

ann = pd.read_csv(glob.glob(os.path.join(DATASET_DIR, "*", "Hi-EF", "annotation.csv"))[0], header=None, dtype=str).set_index(0)
SUB = {c: (t if isinstance(t, str) else '') for c, t in ann[1].items()}
sp = pd.read_csv(SPLIT_CSV, dtype=str)
sp['yB'] = sp['clip4_emotion'].map(E2I)
assert sp['yB'].notna().all()
print(sp.split.value_counts().to_dict())
'''),
("code", '''
# ---- frames
def frames_of(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    keep = sorted(set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())) if total else []
    out, i = [], 0
    while len(out) < len(keep):
        ok, fr = cap.read()
        if not ok:
            break
        if i in keep:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize(FRAME_WH))
        i += 1
    cap.release()
    while out and len(out) < n:                      # pad short or truncated clips by repeating the last frame
        out.append(out[-1])
    return out


def clip_video(c):
    p = video_of(c)
    fr = frames_of(p, N_FRAMES) if p else []
    return fr or [Image.new('RGB', FRAME_WH)] * N_FRAMES          # a blank clip when the video is missing


def clip_still(c):
    return clip_video(c)[N_FRAMES // 2].resize(FRAME_WH_SHOT)


# ---- the fixed prompt
INSTR = ("You are shown three consecutive clips (I, II, III) of a conversation from a TV series, each with its subtitle. "
         "A different person than the speaker of clip III will speak in the next clip, which is not shown. "
         "Predict which emotion that next speaker will express. Answer with exactly one word from: "
         + ", ".join(EMO) + ".")
ROM = ['I', 'II', 'III']


def sub_line(c):
    return SUB.get(c, '').strip() or '(no subtitle)'


def example_messages(r):
    """a few-shot example: subtitles + the middle frame of each clip, then the true label as the assistant turn"""
    content, imgs = [{"type": "text", "text": INSTR}], []
    for k in range(3):
        c = r[f'clip{k + 1}']
        content += [{"type": "text", "text": f"\\nClip {ROM[k]} subtitle: {sub_line(c)}\\nClip {ROM[k]} frame:"}, {"type": "image"}]
        imgs.append(clip_still(c))
    content.append({"type": "text", "text": "\\nNext speaker's emotion:"})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": EMO[int(r['yB'])]}]}], imgs


def target_messages(r):
    content, vids_ = [{"type": "text", "text": INSTR}], []
    for k in range(3):
        c = r[f'clip{k + 1}']
        content += [{"type": "text", "text": f"\\nClip {ROM[k]} subtitle: {sub_line(c)}\\nClip {ROM[k]} video:"}, {"type": "video"}]
        vids_.append(clip_video(c))
    content.append({"type": "text", "text": "\\nNext speaker's emotion:"})
    return [{"role": "user", "content": content}], vids_


def pick_shots(seed, rows):
    """one training example per class, fixed by the seed; classes in a seeded random order"""
    pool = rows[rows.split != 'test']
    order = list(range(7)); random.Random(seed).shuffle(order)
    out = []
    for c in order:
        cand = pool[(pool.yB == c)].copy()
        cand = cand[[all(video_of(cl) is not None and SUB.get(cl, '').strip() for cl in (a, b, d))
                     for a, b, d in zip(cand.clip1, cand.clip2, cand.clip3)]]
        out.append(cand.sample(1, random_state=seed * 100 + c).iloc[0])
    return out
'''),
("code", '''
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
DT = getattr(torch, DTYPE)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DT, device_map="balanced",
                                                           attn_implementation="sdpa").eval()
proc = AutoProcessor.from_pretrained(MODEL_ID)
tok = proc.tokenizer
LABEL_ID = [tok.encode(e, add_special_tokens=False)[0] for e in EMO]
print("first tokens of the labels:", [tok.decode([i]) for i in LABEL_ID])
assert len(set(LABEL_ID)) == 7, "label first tokens are not unique"


@torch.no_grad()
def score(shots, r):
    """7-class distribution for one MCIS; shots = list of (messages, images) few-shot examples (may be empty)"""
    msgs, imgs = [], []
    for m_, i_ in shots:
        msgs += m_; imgs += i_
    tm, v_ = target_messages(r)
    text = proc.apply_chat_template(msgs + tm, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=imgs or None, videos=v_, return_tensors="pt").to(model.device)
    logits = model(**inp, use_cache=False).logits[0, -1].float()
    return torch.softmax(logits[LABEL_ID], -1).cpu().numpy()
'''),
("markdown", "## Smoke test on validation rows (test is not touched)"),
("code", '''
val = sp[sp.split == 'val'].head(SMOKE_N) if SMOKE_N else sp.iloc[:0]
t0 = time.time()
for _, r in val.iterrows():
    p = score([], r)
    print(f"{r.sample_id}: true {EMO[int(r.yB)]:<8} pred {EMO[p.argmax()]:<8} p={np.round(p, 2).tolist()}")
if len(val):
    print(f"{(time.time() - t0) / len(val):.1f} s per MCIS (zero-shot)")
'''),
("markdown", "## Test: zero-shot and few-shot"),
("code", '''
te = sp[sp.split == 'test'].reset_index(drop=True)
if DEBUG_N:
    te = te.head(DEBUG_N)
yt = te.yB.values
PROBS = {}
if RESUME_NPZ and os.path.exists(RESUME_NPZ):
    _z = np.load(RESUME_NPZ, allow_pickle=True)
    assert (_z['sample_id'] == te.sample_id.values).all(), "the resumed file belongs to another sample list"
    PROBS = {k.replace('_', '-') if k.startswith(('zero', 'few')) else k: _z[k] for k in _z.files if k not in ('sample_id', 'y')}
    print("resumed:", list(PROBS))


def run(name, shots):
    if name in PROBS:
        print("skip", name, "(resumed)"); return
    out, t0 = [], time.time()
    for n, (_, r) in enumerate(te.iterrows()):
        out.append(score(shots, r))
        if (n + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (n + 1) % 100 == 0:
            el = (time.time() - t0) / 60
            print(f"{name}: {n + 1}/{len(te)} | {el:.1f} min | ETA {el / (n + 1) * (len(te) - n - 1):.0f} min", flush=True)
    PROBS[name] = np.stack(out)
    np.savez(f"{OUT_DIR}/l1_{SOURCE}_probs.npz", sample_id=te.sample_id.values, y=yt, **{k.replace('-', '_'): v for k, v in PROBS.items()})


if RUN_ZERO_SHOT:
    run("zero-shot", [])
for s in SHOT_SEEDS:
    ex = pick_shots(s, sp)
    print(f"few-shot set {s}:", [(r.sample_id, EMO[int(r.yB)]) for r in ex], flush=True)
    run(f"few-shot-{s}", [example_messages(r) for r in ex])
'''),
("markdown", "## Summary (plain argmax; per-class recall; a train-prior logit adjustment as a descriptive extra)"),
("code", '''
def war_uar(pred, y):
    return (pred == y).mean() * 100, np.mean([(pred[y == c] == c).mean() * 100 for c in range(7) if (y == c).any()])


ytr = sp[sp.split != 'test'].yB.values
logpi = np.log((np.bincount(ytr, minlength=7) + 1) / (len(ytr) + 7))
rows = []
for k, P in PROBS.items():
    w, u = war_uar(P.argmax(1), yt)
    wa, ua = war_uar((np.log(P + 1e-9) - logpi).argmax(1), yt)
    rec = [(P.argmax(1)[yt == c] == c).mean() * 100 if (yt == c).any() else np.nan for c in range(7)]
    rows.append({'setting': k, 'UAR': u, 'WAR': w, 'UAR_LA': ua, 'WAR_LA': wa, **{f"R_{e}": x for e, x in zip(EMO, rec)},
                 'pred_neutral_%': (P.argmax(1) == 4).mean() * 100})
S = pd.DataFrame(rows)
fs = S[S.setting.str.startswith('few-shot')]
if len(fs) > 1:
    S = pd.concat([S, pd.DataFrame([{'setting': 'few-shot mean', **fs.drop(columns='setting').mean().to_dict()}])], ignore_index=True)
S.to_csv(f"{OUT_DIR}/l1_{SOURCE}_summary.csv", index=False)
print(S.round(2).to_string(index=False))
'''),
]

(HERE / "l1_qwen_vl_zero_few_shot.ipynb").write_text(json.dumps(nb(CELLS), indent=1))
print("written")
