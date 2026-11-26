"""Writes k1_partiset_controls.ipynb: PartiSet under input controls (M2, M3) and the fusion cost (R4), for Kaggle."""
import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip('\n').splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip('\n').splitlines(True)}

cells = [md("""
# PartiSet under input controls (cross-turn continuity, listener evidence) and the cost of the fusion stage

Runs `scripts/run_partiset_controls.py` of the code package: development cross-validation (5 episode-disjoint folds, the same folds,
early-stopping episodes and seeds as the main experiments), the **test split is never read**. Arms: `Full`; the listener slot emptied;
the listener slot replaced by the other-participants slot; the listener slot taken from another sample; slots of clips I/II permuted
independently of clip III (continuity broken); one permutation for all turns (continuity kept, slot meaning changed).

**Inputs to attach:** the code dataset (the release zip), the Hi-EF dataset (video + annotation), the pre-extracted CLIP/AudioCLIP features,
the split table `source_folder_split_seed42.csv`, and the folder with the participant-feature shards (`shard_*.pkl`). GPU on.

Set `QUICK = True` for a first check (one seed, two arms), then `False`. To fit the session limit, run the arms in several sessions with
`ARMS` and `TAG`; the last cell merges every `partiset_controls_oof*.npz` it finds in `/kaggle/working/out`.
"""),
code("""
# ======== SETUP ========
import glob, os, shutil, subprocess, sys
CODE_SRC = next(p for p in glob.glob('/kaggle/input/**/partiset', recursive=True) if os.path.isdir(p))
CODE_ROOT = os.path.dirname(CODE_SRC)
shutil.copytree(CODE_ROOT, '/kaggle/working/code', dirs_exist_ok=True)
os.chdir('/kaggle/working/code')
print('code from', CODE_ROOT)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], capture_output=True, text=True)
print(r.stdout[-400:], r.stderr[-300:])
"""),
code("""
# ======== CONFIG ========
QUICK = True                      # True: one seed, two arms (check the pipeline); False: the full run
ALL_ARMS = ['Full', 'listener-off (masked)', 'listener-from-others', 'listener-from-other-sample', 'continuity-broken', 'global-permutation']
ARMS = ALL_ARMS                   # e.g. ALL_ARMS[:3] in one session and ALL_ARMS[3:] in the next
TAG = 'a'                         # suffix of this session's output files
SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]
OUT_DIR = '/kaggle/working/out'

def find(pattern, d=False):
    hits = [p for p in sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)) if (os.path.isdir(p) if d else True)]
    assert hits, f'attach the dataset holding {pattern}'
    return hits[0]

DATASET_DIR = find('hi-ef-dataset', d=True)
FEATURES_DIR = find('hi-ef-features-v2', d=True)
SPLIT_CSV = find('source_folder_split_seed42.csv')
SHARD_DIR = os.path.dirname(find('shard_*.pkl'))
if QUICK:
    SEEDS, ARMS = SEEDS[:1], ARMS[:2]
for k in ('DATASET_DIR', 'FEATURES_DIR', 'SPLIT_CSV', 'SHARD_DIR'):
    print(k, '=', globals()[k])
print('arms', ARMS, '| seeds', SEEDS)
"""),
code("""
# ======== RUN ========
import shlex
cmd = [sys.executable, 'scripts/run_partiset_controls.py', '--dataset', DATASET_DIR, '--features', FEATURES_DIR, '--split-csv', SPLIT_CSV,
       '--participant-features', SHARD_DIR, '--out', OUT_DIR, '--tag', TAG + ('_quick' if QUICK else ''),
       '--seeds', *map(str, SEEDS), '--arms', *ARMS]
print(shlex.join(cmd))
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='', flush=True)
assert p.wait() == 0, 'the run failed'
"""),
md("## Cost of the fusion stage (parameters, time per sample, GPU memory)"),
code("""
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
r = subprocess.run([sys.executable, 'scripts/profile_fusion.py'], capture_output=True, text=True)
print(r.stdout, r.stderr[-500:])
"""),
md("## Results (seed ensembles; paired differences with the Full model, resampling whole episodes)"),
code("""
import numpy as np, pandas as pd
files = sorted(glob.glob(f'{OUT_DIR}/partiset_controls_oof*.npz'))
files = [f for f in files if ('_quick' in f) == QUICK] or files
print(files)
Z = [np.load(f, allow_pickle=True) for f in files]
y, src = Z[0]['y'].astype(int), Z[0]['src']
for z in Z: assert (z['sample_id'] == Z[0]['sample_id']).all()
arms = {k: z[k] for z in Z for k in z.files if k not in ('sample_id', 'fold', 'y', 'src')}
safe = lambda s: __import__('re').sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
eps = np.unique(src); idx = [np.where(src == e)[0] for e in eps]
def uar_war(pred, w):
    war = (w * (pred == y)).sum() / w.sum() * 100
    rec = [(w * (y == c) * (pred == y)).sum() / (w * (y == c)).sum() * 100 for c in range(7) if (w * (y == c)).sum() > 0]
    return np.mean(rec), war
def weights(counts):
    w = np.zeros(len(y))
    for c, ix in zip(counts, idx): w[ix] = c
    return w
rng = np.random.default_rng(0)
BOOT = [weights(np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps))) for _ in range(2000)]
ens = {k: v.mean(0).argmax(1) for k, v in arms.items()}
ones = np.ones(len(y)); full = ens.get('Full')
rows = []
for k, v in arms.items():
    u, w = uar_war(ens[k], ones)
    per = np.array([uar_war(p.argmax(1), ones) for p in v])
    r = dict(arm=k, seeds=len(v), UAR=u, WAR=w, UAR_seed_sd=per[:, 0].std(ddof=1) if len(v) > 1 else np.nan)
    if full is not None and k != 'Full':
        d = u - uar_war(full, ones)[0]
        bs = [uar_war(ens[k], b)[0] - uar_war(full, b)[0] for b in BOOT]
        r.update(dUAR_vs_Full=d, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                 episodes_better=sum(uar_war(ens[k], weights((np.arange(len(eps)) == i).astype(float)))[0] > uar_war(full, weights((np.arange(len(eps)) == i).astype(float)))[0] for i in range(len(eps))))
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'{OUT_DIR}/partiset_controls_summary.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(2).to_string(index=False))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('k1_partiset_controls.ipynb', 'w'), indent=1)
print('written k1_partiset_controls.ipynb')
