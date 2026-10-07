# Development-CV comparison of RoleNet variants (G40 truncation, G43 listener tokens, G44 oracle truncation) with the
# baseline (G41 PaperBest and PaperBest-FM out-of-fold predictions; same folds and seeds, not retrained).
# Inputs: OOF=<dir with the unpacked g4x_oof_probs* folders of .npy files>, SPLIT=<split csv>.
# sample_id / src are not unpickled: the MCIS order is rebuilt from the split file (train+val rows, file order, as in
# the notebooks) and checked against y and fold of every file.
import os
import numpy as np, pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
D, OUT = os.environ['OOF'], os.environ.get('OUT', '.')
sp = pd.read_csv(os.environ['SPLIT'], dtype=str)
dev = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)
y = dev.clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
src = dev.source_folder.values


def load(folder):
    p = os.path.join(D, folder)
    a = {f[:-4]: np.load(os.path.join(p, f), allow_pickle=False) for f in os.listdir(p)
         if f.endswith('.npy') and f[:-4] not in ('sample_id', 'src')}
    assert (a.pop('y') == y).all(), f"{folder}: labels / MCIS order differ from the split file"
    return a


R = {k: load(k) for k in sorted(os.listdir(D)) if os.path.isdir(os.path.join(D, k))}
fold = R['g41_oof_probs']['fold']
for k, a in R.items():
    assert (a['fold'] == fold).all(), f"{k}: different folds"
G = [np.where(src == e)[0] for e in np.unique(src)]


def uar(p, idx=None):
    t = y if idx is None else y[idx]
    p = p if idx is None else p[idx]
    return np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()])


def boot(pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        ea = pa[rng.integers(0, len(pa), len(pa))].mean(0).argmax(1)
        eb = pb[rng.integers(0, len(pb), len(pb))].mean(0).argmax(1)
        idx = np.concatenate([G[i] for i in rng.integers(0, len(G), len(G))])
        d.append(uar(ea, idx) - uar(eb, idx))
    return np.percentile(d, [2.5, 97.5])


PB, FM = R['g41_oof_probs']['PaperBest'], R['g41_oof_probs']['PaperBest_FM']
ROWS = [('Baseline', 'PaperBest (original features)', PB), ('Baseline', 'PaperBest-FM (RoleNet features)', FM),
        ('G41', 'RoleNet', R['g41_oof_probs']['RoleNet'])]
for k, lab in [('L_both', 'L-both (= RoleNet)'), ('L_history_only', 'L-history only'),
               ('L_current_only', 'L-current only'), ('L_none', 'L-none')]:
    ROWS.append(('G43 listener tokens', lab, R['g43_oof_probs'][k]))
for g, name in [('g40', 'G40 truncation (proxy listener)'), ('g44', 'G44 truncation (oracle listener)')]:
    for p in ('100', '080', '060'):
        for arm in ('Full', 'minus_L'):
            ROWS.append((name, f"clip III {int(p)}% {'RoleNet' if arm == 'Full' else 'minus-L'}", R[f'{g}_oof_probs_p{p}'][arm]))
out = []
for grp, lab, P in ROWS:
    e = P.mean(0).argmax(1)
    r = {'group': grp, 'variant': lab, 'UAR': uar(e), 'WAR': (e == y).mean() * 100}
    if grp != 'Baseline':
        for bn, B in (('PaperBest', PB), ('PaperBest_FM', FM)):
            lo, hi = boot(P, B)
            r[f'd_{bn}'] = r['UAR'] - uar(B.mean(0).argmax(1)); r[f'{bn}_lo'] = lo; r[f'{bn}_hi'] = hi
            r[f'folds_vs_{bn}'] = sum(uar(e, np.where(fold == f)[0]) > uar(B.mean(0).argmax(1), np.where(fold == f)[0])
                                      for f in range(5))
    out.append(r)
T = pd.DataFrame(out)
pd.set_option('display.width', 250)
print(T.round(2).to_string(index=False))
T.to_csv(os.path.join(OUT, 'cv_vs_baseline.csv'), index=False)
print('saved', os.path.join(OUT, 'cv_vs_baseline.csv'))
