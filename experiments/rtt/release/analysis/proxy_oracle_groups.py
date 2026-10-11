# Proxy vs true responder by G48 group (development CV, analysis only; no training).
# Proxy = G41 RoleNet, oracle listener = G44 RoleNet at 100% (L = true responder by the G26/G44 rule), no listener =
# G43 L-none, baseline = G41 PaperBest (same folds and seeds). Groups come from G48 (g48_groups.csv).
# Usage: OOF=<dir with g41_oof_probs/, g43_oof_probs/, g44_oof_probs_p100/> SPLIT=<split csv> GROUPS_CSV=<g48_groups.csv>
#        OUT=<dir> python proxy_oracle_groups.py
import os
import numpy as np, pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
D, OUT = os.environ['OOF'], os.environ.get('OUT', '.')
sp = pd.read_csv(os.environ['SPLIT'], dtype=str)
dev = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)
y = dev.clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
src = dev.source_folder.values


def load(folder, key):
    p = os.path.join(D, folder)
    assert (np.load(os.path.join(p, 'y.npy'), allow_pickle=False) == y).all(), f"{folder}: MCIS order differs"
    return np.load(os.path.join(p, key + '.npy'), allow_pickle=False)


M = {'proxy (RoleNet)': load('g41_oof_probs', 'RoleNet'), 'oracle listener': load('g44_oof_probs_p100', 'Full'),
     'no listener': load('g43_oof_probs', 'L_none'), 'baseline': load('g41_oof_probs', 'PaperBest')}
G = pd.read_csv(os.environ['GROUPS_CSV'], dtype={'sample_id': str}).set_index('sample_id').reindex(dev.sample_id.values)
assert G.group.notna().all(), "groups missing for some MCIS"
grp = G.group.values
seen_diff = grp == 'B seen ≠ proxy'
SUBS = {'all MCIS': np.ones(len(y), bool)}
for g_ in ['B seen = proxy', 'B seen ≠ proxy', 'B not seen', 'uncertain']:
    SUBS[g_] = grp == g_
SUBS['B seen ≠ proxy, B visible in clip III'] = seen_diff & (G.B_frames_III.values > 0)
SUBS['B seen ≠ proxy, B only in clips I-II'] = seen_diff & (G.B_frames_III.values == 0)
EP = [np.where(src == e)[0] for e in np.unique(src)]


def uar(p, t):
    return np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()]) if len(t) else np.nan


def boot(a, b, m, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        pa = M[a][rng.integers(0, len(M[a]), len(M[a]))].mean(0).argmax(1)
        pb = M[b][rng.integers(0, len(M[b]), len(M[b]))].mean(0).argmax(1)
        idx = np.concatenate([EP[i] for i in rng.integers(0, len(EP), len(EP))]); idx = idx[m[idx]]
        if len(idx):
            d.append(uar(pa[idx], y[idx]) - uar(pb[idx], y[idx]))
    return np.percentile(d, [2.5, 97.5])


P = {k: v.mean(0).argmax(1) for k, v in M.items()}
rows = []
for s, m in SUBS.items():
    r = {'subset': s, 'n': int(m.sum())}
    for k, p in P.items():
        r[f'UAR {k}'] = uar(p[m], y[m]); r[f'WAR {k}'] = (p[m] == y[m]).mean() * 100 if m.any() else np.nan
    for a, b in [('oracle listener', 'proxy (RoleNet)'), ('proxy (RoleNet)', 'no listener'), ('proxy (RoleNet)', 'baseline')]:
        r[f'{a} − {b}'] = r[f'UAR {a}'] - r[f'UAR {b}']
        r[f'{a} − {b} lo'], r[f'{a} − {b} hi'] = boot(a, b, m) if m.sum() >= 20 else (np.nan, np.nan)
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(T.round(2).to_string(index=False))
os.makedirs(OUT, exist_ok=True)
T.to_csv(os.path.join(OUT, 'proxy_oracle_groups.csv'), index=False)
