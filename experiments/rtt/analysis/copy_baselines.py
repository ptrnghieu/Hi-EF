# Diagnostic baselines on development CV (G41 run; analysis only, no training): majority class of the training folds,
# copy-A (oracle: A's gold clip-III label), copy-previous-B (oracle: gold label of B's own latest earlier turn, where B
# spoke in clip I/II; B found by clip-IV voice, G16), next to the baseline, the feature-matched baseline and RoleNet.
# Usage: OOF=<dir with g41_oof_probs/> SPLIT=<split csv> G16=<g16_per_mcis.csv> OUT=<dir> python copy_baselines.py
import os
import numpy as np, pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
sp = pd.read_csv(os.environ['SPLIT'], dtype=str)
dev = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)
y = dev.clip4_emotion.map(E2I).values
yA = dev.clip3_emotion.map(E2I).values
D = os.path.join(os.environ['OOF'], 'g41_oof_probs')
assert (np.load(os.path.join(D, 'y.npy'), allow_pickle=False) == y).all(), "MCIS order differs from the split file"
M = {k: np.load(os.path.join(D, k + '.npy'), allow_pickle=False).mean(0).argmax(1) for k in ('PaperBest', 'PaperBest_FM', 'RoleNet')}
g = pd.read_csv(os.environ['G16'], dtype={'sample_id': str}).set_index('sample_id').loc[dev.sample_id.values]
assert (g.y.values == y).all()
fold = g.fold.values
near = g.B_spoke_II.values.astype(bool) & (g.y_II.values >= 0)
far = g.B_spoke_I.values.astype(bool) & (g.y_I.values >= 0) & ~near
yp = np.where(near, g.y_II.values, np.where(far, g.y_I.values, -1))
maj = np.zeros_like(y)
for f in np.unique(fold):
    maj[fold == f] = np.bincount(y[fold != f], minlength=7).argmax()


def uar(p, t):
    return np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()])


rows = []
for sname, m in [('all MCIS', np.ones(len(y), bool)), ("B's previous emotion known", yp >= 0)]:
    P = {'majority (training folds)': maj, 'copy A (oracle)': yA}
    if sname != 'all MCIS':
        P['copy previous B (oracle)'] = yp
    P.update({'Baseline': M['PaperBest'], 'Baseline, RoleNet features': M['PaperBest_FM'], 'RoleNet': M['RoleNet']})
    for k, p in P.items():
        rows.append({'subset': sname, 'n': int(m.sum()), 'predictor': k, 'UAR': uar(p[m], y[m]), 'WAR': (p[m] == y[m]).mean() * 100})
T = pd.DataFrame(rows)
print(T.round(2).to_string(index=False))
print('majority class per fold:', {int(f): EMO[np.bincount(y[fold != f], minlength=7).argmax()] for f in np.unique(fold)})
os.makedirs(os.environ.get('OUT', '.'), exist_ok=True)
T.to_csv(os.path.join(os.environ.get('OUT', '.'), 'copy_baselines.csv'), index=False)
