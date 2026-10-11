# B's own emotion change (development CV, analysis only, no training).
# B's previous emotion = gold label of B's own latest earlier turn (clip II, else clip I) where B spoke there (voice
# match to clip IV, G15/G16; analysis only) and that clip is labelled. Groups: B keeps vs B changes emotion, crossed
# with whether B mirrors the clip-III speaker A. Models: G41 out-of-fold predictions (RoleNet, PaperBest, PaperBest-FM).
# Usage: OOF=<unpacked oof-probs dir> G16=<g16_per_mcis.csv> SPLIT=<split csv> OUT=<dir> python b_self_shift.py
import os
import numpy as np, pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
D = os.environ['OOF']
sp = pd.read_csv(os.environ['SPLIT'], dtype=str)
dev = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)
y = dev.clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
yA = dev.clip3_emotion.map({e: i for i, e in enumerate(EMO)}).values
src = dev.source_folder.values
g = pd.read_csv(os.environ['G16'], dtype={'sample_id': str}).set_index('sample_id').loc[dev.sample_id.values]
assert (g.y.values == y).all(), "G16 table and split disagree on labels"
near = g.B_spoke_II.values.astype(bool) & (g.y_II.values >= 0)
far = g.B_spoke_I.values.astype(bool) & (g.y_I.values >= 0) & ~near
yprev = np.where(near, g.y_II.values, np.where(far, g.y_I.values, -1)).astype(int)
known = yprev >= 0
keep, change = known & (yprev == y), known & (yprev != y)
mirror = yA == y

M = {k: np.load(os.path.join(D, 'g41_oof_probs', f'{k}.npy'), allow_pickle=False) for k in ('RoleNet', 'PaperBest', 'PaperBest_FM')}
assert (np.load(os.path.join(D, 'g41_oof_probs', 'y.npy'), allow_pickle=False) == y).all()
PRED = {k: v.mean(0).argmax(1) for k, v in M.items()}
EP = [np.where(src == e)[0] for e in np.unique(src)]


def uar(p, t):
    return np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()])


def boot_acc(a, b, m, n=2000, seed=0):
    # accuracy difference on subset m; resample seeds of both models and episodes
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        pa = M[a][rng.integers(0, 5, 5)].mean(0).argmax(1); pb = M[b][rng.integers(0, 5, 5)].mean(0).argmax(1)
        idx = np.concatenate([EP[i] for i in rng.integers(0, len(EP), len(EP))]); idx = idx[m[idx]]
        d.append((pa[idx] == y[idx]).mean() * 100 - (pb[idx] == y[idx]).mean() * 100)
    return np.percentile(d, [2.5, 97.5])


print(f"MCIS {len(y)} | B's previous emotion known {known.sum()} ({known.mean() * 100:.1f}%) | "
      f"keeps {keep.sum()} | changes {change.sum()} | keep rate {keep.sum() / max(known.sum(), 1) * 100:.1f}%")
print(f"copy B's previous emotion on the known subset: acc {(yprev[known] == y[known]).mean() * 100:.1f}, "
      f"UAR {uar(yprev[known], y[known]):.2f}")
groups = [('all', np.ones(len(y), bool)), ('B prev known', known), ('B keeps', keep), ('B changes', change),
          ('B keeps & mirrors A', keep & mirror), ('B keeps & not mirror A', keep & ~mirror),
          ('B changes & mirrors A', change & mirror), ('B changes & not mirror A', change & ~mirror),
          ('B prev unknown', ~known)]
rows = []
for name, m in groups:
    r = {'group': name, 'n': int(m.sum())}
    for k, p in PRED.items():
        r[f'ACC_{k}'] = (p[m] == y[m]).mean() * 100
        r[f'UAR_{k}'] = uar(p[m], y[m])
    for b in ('PaperBest', 'PaperBest_FM'):
        lo, hi = boot_acc('RoleNet', b, m)
        r[f'dACC_vs_{b}'] = r['ACC_RoleNet'] - r[f'ACC_{b}']; r[f'dACC_vs_{b}_lo'] = lo; r[f'dACC_vs_{b}_hi'] = hi
    r['RoleNet_predicts_Bprev_%'] = (PRED['RoleNet'][m & known] == yprev[m & known]).mean() * 100 if (m & known).any() else np.nan
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
cols = ['group', 'n', 'ACC_PaperBest', 'ACC_PaperBest_FM', 'ACC_RoleNet', 'dACC_vs_PaperBest', 'dACC_vs_PaperBest_lo',
        'dACC_vs_PaperBest_hi', 'dACC_vs_PaperBest_FM', 'dACC_vs_PaperBest_FM_lo', 'dACC_vs_PaperBest_FM_hi',
        'UAR_PaperBest', 'UAR_RoleNet', 'RoleNet_predicts_Bprev_%']
print(T[cols].round(1).to_string(index=False))
T.to_csv(os.path.join(os.environ.get('OUT', '.'), 'b_self_shift.csv'), index=False)
