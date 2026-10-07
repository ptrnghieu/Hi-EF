# Would a skip connection that feeds B's previous emotion to the prediction help? (development CV, analysis only)
# On the MCIS where B's previous emotion Z is known (B spoke in clip I/II, labelled; G16), combine RoleNet's OOF
# log-probabilities with a one-hot Z of given accuracy (wrong values drawn from the Z prior) in an episode-fold LR, and
# compare with LR on RoleNet alone. Usage: OOF=<oof dir> G16=<g16_per_mcis.csv> SPLIT=<split csv> python skip_prevB_sim.py
import os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
sp = pd.read_csv(os.environ['SPLIT'], dtype=str)
dev = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)
y = dev.clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
src = dev.source_folder.values
g = pd.read_csv(os.environ['G16'], dtype={'sample_id': str}).set_index('sample_id').loc[dev.sample_id.values]
near = g.B_spoke_II.values.astype(bool) & (g.y_II.values >= 0)
far = g.B_spoke_I.values.astype(bool) & (g.y_I.values >= 0) & ~near
Z = np.where(near, g.y_II.values, np.where(far, g.y_I.values, -1)).astype(int)
P = np.load(os.path.join(os.environ['OOF'], 'g43_oof_probs', 'L_both.npy'), allow_pickle=False).mean(0)
uar = lambda p, t: np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()])
idx = np.where(Z >= 0)[0]
X, yy, ee, Zg = np.log(P[idx] + 1e-6), y[idx], src[idx], Z[idx]
prior = np.bincount(Zg, minlength=7) / len(Zg)
perm = np.random.default_rng(0).permutation(np.unique(ee))
f = np.array([{e: i % 5 for i, e in enumerate(perm)}[e] for e in ee])


def cv(F):
    pr = np.zeros(len(yy), int)
    for i in range(5):
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(F[f != i], yy[f != i])
        pr[f == i] = m.predict(F[f == i])
    return uar(pr, yy)


base = cv(X)
print(f"subset n={len(idx)} | LR(RoleNet) UAR {base:.2f}")
for acc in (1.0, 0.6, 0.45, 0.31):
    res = []
    for rep in range(10):
        r = np.random.default_rng(rep); Zn = Zg.copy()
        for i in np.where(r.random(len(Zg)) > acc)[0]:
            p = prior.copy(); p[Zg[i]] = 0; Zn[i] = r.choice(7, p=p / p.sum())
        res.append(cv(np.hstack([X, np.eye(7)[Zn]])) - base)
    print(f"Z accuracy {acc:.2f}: gain on subset {np.mean(res):+.2f} ± {np.std(res):.2f} UAR")
