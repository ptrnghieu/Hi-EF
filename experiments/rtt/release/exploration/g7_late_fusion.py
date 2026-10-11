"""Late fusion of B1 with a face-only model (validation only; test untouched).

B1 probabilities come from G3b (`g3b_val_probs.npz`, key `B1`, inner-dev selection, 5 seeds; identical to G7b's B1).
The face model is a class-balanced logistic regression on the G6b role features (A, listener, listener in I/II,
dominant faces of I/II), trained on the train split with C chosen by episode-grouped CV on train.
Fusion: argmax of (1 - w) log p_B1 + w log p_face with w = 0.5 (equal weight, no tuning), plus a presence-only control.

Usage: python g7_late_fusion.py --features g6b_role_features.csv --split source_folder_split_seed42.csv \
           --b1 g3b_val_probs.npz
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
ROLES = ['A', 'listener', 'listener_ctx', 'ctx_II', 'ctx_I']


def war_uar(p, y):
    return (p == y).mean() * 100, np.mean([(p[y == c] == c).mean() * 100 for c in range(7) if (y == c).any()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--b1', required=True)
    ap.add_argument('--w', type=float, default=0.5)
    a = ap.parse_args()

    sp = pd.read_csv(a.split, dtype=str)
    role = pd.read_csv(a.features).set_index('sample_id')
    pz = np.load(a.b1, allow_pickle=True)
    ev = sp.set_index('sample_id').loc[pz['sample_id']].reset_index()
    assert set(ev.split) == {'val'}
    tr = sp[sp.split == 'train'].reset_index(drop=True)
    yv, yt = ev.clip4_emotion.map(E2I).values, tr.clip4_emotion.map(E2I).values
    src = ev.source_folder.values

    def face_lr(presence_only):
        cols = [f'{r}_{i}' for r in ROLES for i in ([10, 11] if presence_only else range(12))]
        sc = StandardScaler().fit(role.loc[tr.sample_id, cols].values)
        Xt, Xv = sc.transform(role.loc[tr.sample_id, cols].values), sc.transform(role.loc[ev.sample_id, cols].values)
        best = None
        for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1]:
            s = [war_uar(LogisticRegression(max_iter=3000, C=C, class_weight='balanced').fit(Xt[i], yt[i]).predict(Xt[j]),
                         yt[j])[1] for i, j in GroupKFold(5).split(Xt, yt, tr.source_folder)]
            if best is None or np.mean(s) > best[0]:
                best = (np.mean(s), C)
        return LogisticRegression(max_iter=3000, C=best[1], class_weight='balanced').fit(Xt, yt).predict_proba(Xv)

    PF = {'faces': face_lr(False), 'presence': face_lr(True)}
    for k, p in PF.items():
        print(f"face-only LR [{k}] on val: UAR {war_uar(p.argmax(1), yv)[1]:.2f}")

    def fuse(pb, pf):
        return ((1 - a.w) * np.log(pb + 1e-9) + a.w * np.log(pf + 1e-9)).argmax(1)

    def boot(pa, pb, m, n=2000):
        rng = np.random.default_rng(0)
        g = [np.where(src[m] == s)[0] for s in np.unique(src[m])]
        y, pa, pb = yv[m], pa[m], pb[m]
        d = []
        for _ in range(n):
            i = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))])
            d.append(war_uar(pa[i], y[i])[1] - war_uar(pb[i], y[i])[1])
        return np.percentile(d, [2.5, 97.5])

    B = pz['B1']
    print(f"\nper-seed ΔUAR vs B1 at w={a.w}:")
    for s in range(len(B)):
        u0 = war_uar(B[s].argmax(1), yv)[1]
        print(f"  seed {s}: B1 {u0:.2f}  " + "  ".join(
            f"{k} {war_uar(fuse(B[s], p), yv)[1] - u0:+.2f}" for k, p in PF.items()))
    Bm = B.mean(0)
    b = Bm.argmax(1)
    vis = role.loc[ev.sample_id, 'listener_10'].values > 0
    for name, m in [('all val', np.ones(len(yv), bool)), ('listener visible', vis), ('listener not visible', ~vis)]:
        w0, u0 = war_uar(b[m], yv[m])
        print(f"\n== seed-ensemble, {name} (n={m.sum()}): B1 UAR {u0:.2f} WAR {w0:.2f}")
        for k, p in PF.items():
            f = fuse(Bm, p)
            lo, hi = boot(f, b, m)
            w1, u1 = war_uar(f[m], yv[m])
            print(f"  B1+{k:<9} UAR {u1:.2f} Δ {u1 - u0:+.2f} [{lo:+.2f},{hi:+.2f}]  WAR {w1:.2f}")


if __name__ == '__main__':
    main()
