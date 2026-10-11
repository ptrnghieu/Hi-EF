# RoleNet vs the baseline on the test split by where the true responder B is visible (descriptive; saved G10
# predictions, no new test read). Groups come from G42 (B = dominant clip-IV face matched to a clip I-III identity
# other than A; clip IV used for grouping only).
# Usage: G42_CSV=g42_test_responder_groups.csv G10_NPZ=g10_test_probs.npz SPLIT=<split csv> python responder_groups_test.py
import os
import numpy as np, pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
GROUPS = ['history only', 'current only', 'both', 'none', 'B not found in IV']
rg = pd.read_csv(os.environ['G42_CSV'], dtype={'sample_id': str}).set_index('sample_id')
z = np.load(os.environ['G10_NPZ'], allow_pickle=True)
sp = pd.read_csv(os.environ['SPLIT'], dtype=str).set_index('sample_id')
sid = z['sample_id']
assert set(sid) == set(rg.index), "G42 groups and G10 predictions cover different MCIS"
y = sp.loc[sid].clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
grp = rg.loc[sid].group.values
P = {'Baseline': z['PaperBest'].mean(0).argmax(1), 'RoleNet': z['RoleNet'].mean(0).argmax(1)}


def uar(p, t):
    return np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()])


rows = []
for g in GROUPS + ['all']:
    m = np.ones(len(y), bool) if g == 'all' else grp == g
    if not m.any():
        continue
    r = {'group': g, 'n': int(m.sum()), 'share_%': m.mean() * 100}
    for k, p in P.items():
        r[f'UAR_{k}'] = uar(p[m], y[m]); r[f'ACC_{k}'] = (p[m] == y[m]).mean() * 100
    r['dUAR'] = r['UAR_RoleNet'] - r['UAR_Baseline']; r['dACC'] = r['ACC_RoleNet'] - r['ACC_Baseline']
    r['classes_present'] = int(len(np.unique(y[m])))
    rows.append(r)
T = pd.DataFrame(rows)
print(T.round(2).to_string(index=False))
out = os.environ.get('OUT', '.')
T.to_csv(os.path.join(out, 'responder_groups_test.csv'), index=False)
print('saved', os.path.join(out, 'responder_groups_test.csv'))
