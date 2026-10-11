# Headroom of shift-aware prediction on development CV (analysis only, no training).
# On shifts (B's label != A's clip-III label) RoleNet still predicts A's label ~29% of the time. Upper bound: exclude A's
# (gold) label wherever a shift is known; realistic-ish bound: a shift detector of given AUC (binormal simulation),
# threshold tuned on the same data (optimistic), A's label still gold.
# Usage: OOF=<unpacked oof-probs dir> SPLIT=<split csv> python shift_headroom.py
import os
import numpy as np, pandas as pd
from scipy.stats import norm

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
D = os.environ['OOF']
sp = pd.read_csv(os.environ['SPLIT'], dtype=str)
dev = sp[sp.split.isin(['train', 'val'])].reset_index(drop=True)
y = dev.clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
yA = dev.clip3_emotion.map({e: i for i, e in enumerate(EMO)}).values
assert (np.load(os.path.join(D, 'g43_oof_probs', 'y.npy'), allow_pickle=False) == y).all()
P = np.load(os.path.join(D, 'g43_oof_probs', 'L_both.npy'), allow_pickle=False).mean(0)
sh = yA != y
uar = lambda p, t: np.mean([(p[t == c] == c).mean() * 100 for c in range(7) if (t == c).any()])
base = uar(P.argmax(1), y)
Q = P.copy(); Q[sh, yA[sh]] = -1
print(f"base UAR {base:.2f} | oracle (known shift + gold A label) {uar(Q.argmax(1), y):.2f}")
for auc in (0.70, 0.80, 0.90):
    d = np.sqrt(2) * norm.ppf(auc)
    best = []
    for rep in range(20):
        s = np.random.default_rng(rep).normal(size=len(y)) + d * sh
        res = []
        for q in np.linspace(0.05, 0.95, 19):
            flag = s > np.quantile(s, q)
            Q = P.copy(); Q[flag, yA[flag]] = -1
            res.append(uar(Q.argmax(1), y))
        best.append(max(res))
    print(f"shift detector AUC {auc:.2f}: best UAR {np.mean(best):.2f} ({np.mean(best) - base:+.2f})")
