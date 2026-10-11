# Exploratory (train+val OOF only; test untouched): RoleNet (G14 Full, 10 seeds) on samples where B's clip-IV label
# equals A's clip-III label ("mirror") vs differs ("shift"), and where the listener-token gain comes from.
# usage: python mirror_vs_shift.py <g14 results dir> <annotation.csv> <source_folder_split_seed42.csv>
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
RES, ANN, SPLIT = sys.argv[1:4]
E = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
z = np.load(os.path.join(RES, 'g14_oof_probs.npz'), allow_pickle=True)
sp = pd.read_csv(SPLIT, dtype=str).set_index('sample_id').loc[z['sample_id']]
y, L = z['y'], z['listener_in_III'].astype(bool)
yA = sp.clip3_emotion.map(E.index).values
P, PL = z['Full'].mean(0), z['minus_L'].mean(0)
same = y == yA
acc = lambda p, m: 100 * np.mean(p[m].argmax(1) == y[m])
lp = lambda p, m: np.log(p[m][np.arange(m.sum()), y[m]] + 1e-12)
print(f"P(yB == yA) = {same.mean():.3f}")
for name, m in [('mirror', same), ('shift', ~same), ('mirror & L', same & L), ('shift & L', ~same & L),
                ('mirror & no L', same & ~L), ('shift & no L', ~same & ~L)]:
    print(f"{name:14s} n {m.sum():4d} | acc Full {acc(P, m):5.1f} minus-L {acc(PL, m):5.1f} | "
          f"mean log p gain from L {np.mean(lp(P, m) - lp(PL, m)):+.3f}")
print(f"AUC of p(yB = yA) for mirror vs shift: {roc_auc_score(same, P[np.arange(len(y)), yA]):.3f}")
pr = P.argmax(1)
print(f"on shift samples Full predicts A's label {np.mean(pr[~same] == yA[~same]):.3f}")
for c in range(7):
    ms, mm = (~same) & (y == c), same & (y == c)
    print(f"  {E[c]:8s} recall shift {100 * np.mean(pr[ms] == c):5.1f} (n {ms.sum()}) | mirror {100 * np.mean(pr[mm] == c) if mm.any() else float('nan'):5.1f} (n {mm.sum()})")
T = pd.crosstab(pd.Series(yA).map(dict(enumerate(E))), pd.Series(y).map(dict(enumerate(E))))
base = pd.Series(y).map(dict(enumerate(E))).value_counts(normalize=True)
print("lift of P(B | A) over the base rate:"); print((T.div(T.sum(axis=1), axis=0) / base).round(2))
