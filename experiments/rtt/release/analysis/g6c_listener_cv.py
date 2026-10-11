"""G6c: re-analysis of G6b role features without new extraction (train+val only; test untouched).

1. Zero-shot, paired on the same MCIS: HSEmotion on the listener's face vs on A's face as a forecast of B.
2. Logistic regression with balanced class weights, out-of-fold over all 45 train+val episodes
   (episode-grouped 5-fold, repeated with 3 episode shuffles; C chosen by inner grouped CV).
   Controls: listener presence/frame-share only (does *whether* a reaction shot exists explain the gain?).
3. Boundary split: MCIS whose listener appears in the first 80% of clip III vs only in the last 20%.

Usage: python g6c_listener_cv.py --features g6b_role_features.csv --split source_folder_split_seed42.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
FER2HI = [0, 1, 1, 2, 3, 4, 5, 6]   # anger, contempt→disgust, disgust, fear, happiness, neutral, sadness, surprise
ROLES = ['A', 'listener', 'listener_early', 'listener_ctx', 'ctx_II', 'ctx_I']


def war_uar(pred, y):
    return (pred == y).mean() * 100, np.mean([(pred[y == c] == c).mean() * 100 for c in range(7) if (y == c).any()])


def zero_shot(F, role):
    M = np.zeros((8, 7))
    for i, j in enumerate(FER2HI):
        M[i, j] = 1
    return (F[role][:, :8] @ M).argmax(1)


def boot_pairs(preds, y, groups, n=2000, seed=0):
    """Episode bootstrap: CI of each model's (WAR, UAR) and of each model minus the first."""
    rng = np.random.default_rng(seed)
    eps = np.unique(groups)
    idx_of = {e: np.where(groups == e)[0] for e in eps}
    names = list(preds)
    stats = {k: [] for k in names}
    for _ in range(n):
        idx = np.concatenate([idx_of[e] for e in rng.choice(eps, len(eps))])
        for k in names:
            stats[k].append(war_uar(preds[k][idx], y[idx]))
    return {k: np.array(v) for k, v in stats.items()}


def report(title, preds, y, groups, ref):
    print(f"\n== {title}: n={len(y)} MCIS, {len(np.unique(groups))} episodes ==")
    b = boot_pairs(preds, y, groups)
    for k, p in preds.items():
        w, u = war_uar(p, y)
        lo, hi = np.percentile(b[k], [2.5, 97.5], axis=0)
        line = f"  {k:<26} UAR {u:5.2f} [{lo[1]:5.1f},{hi[1]:5.1f}]  WAR {w:5.2f} [{lo[0]:5.1f},{hi[0]:5.1f}]"
        if k != ref:
            d = b[k] - b[ref]
            dlo, dhi = np.percentile(d, [2.5, 97.5], axis=0)
            wr, ur = war_uar(preds[ref], y)
            line += f"   Δ vs {ref}: UAR {u - ur:+5.2f} [{dlo[1]:+.2f},{dhi[1]:+.2f}]"
        print(line)


def episode_folds(groups, k, seed):
    eps = np.unique(groups)
    rng = np.random.default_rng(seed)
    eps = eps[rng.permutation(len(eps))]
    fold_of = {e: i % k for i, e in enumerate(eps)}
    return np.array([fold_of[g] for g in groups])


def oof_lr(X, y, groups, seeds=(0, 1, 2), k=5, Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1)):
    """Out-of-fold predictions; per outer fold, C is chosen by inner episode-grouped CV on UAR.
    Returns one prediction vector per shuffle seed."""
    outs = []
    for s in seeds:
        fold = episode_folds(groups, k, s)
        pred = np.zeros(len(y), int)
        for f in range(k):
            tr, te = fold != f, fold == f
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            inner = episode_folds(groups[tr], 4, 100 + s)
            best = None
            for C in Cs:
                us = []
                for g in range(4):
                    a, b = inner != g, inner == g
                    m = LogisticRegression(max_iter=3000, C=C, class_weight='balanced').fit(Xtr[a], y[tr][a])
                    us.append(war_uar(m.predict(Xtr[b]), y[tr][b])[1])
                if best is None or np.mean(us) > best[0]:
                    best = (np.mean(us), C)
            m = LogisticRegression(max_iter=3000, C=best[1], class_weight='balanced').fit(Xtr, y[tr])
            pred[te] = m.predict(Xte)
        outs.append(pred)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--split', required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.features).merge(pd.read_csv(a.split, dtype=str), on='sample_id')
    assert set(df.split) <= {'train', 'val'}, "test rows must not be analysed"
    F = {r: df[[f"{r}_{i}" for i in range(12)]].values.astype(float) for r in ROLES}
    yA = df.clip3_emotion.map(E2I).values
    yB = df.clip4_emotion.map(E2I).values
    groups = df.source_folder.values
    vis = F['listener'][:, 10] > 0
    early = F['listener_early'][:, 10] > 0
    print(f"MCIS {len(df)} | episodes {len(np.unique(groups))} | listener visible {vis.mean() * 100:.1f}% | "
          f"listener in first 80% of III {early.mean() * 100:.1f}% | only in last 20% {(vis & ~early).mean() * 100:.1f}%")
    print("B label distribution (%):", dict(zip(EMO, np.bincount(yB, minlength=7) / len(yB) * 100 // 0.1 / 10)))

    # ---------- 1. zero-shot, paired on the same MCIS ----------
    zsA, zsL, zsLe = zero_shot(F, 'A'), zero_shot(F, 'listener'), zero_shot(F, 'listener_early')
    m = vis
    report("ZERO-SHOT, listener visible (forecast B's clip-IV label)",
           {"A's face (copy A)": zsA[m], "listener's face": zsL[m], "gold A label (Copy-A oracle)": yA[m]},
           yB[m], groups[m], "A's face (copy A)")
    m = early
    report("ZERO-SHOT, listener seen in first 80% of III",
           {"A's face (copy A)": zsA[m], "listener, all frames": zsL[m], "listener, first 80% only": zsLe[m]},
           yB[m], groups[m], "A's face (copy A)")
    m = vis & ~early
    report("ZERO-SHOT, listener ONLY in last 20% of III",
           {"A's face (copy A)": zsA[m], "listener's face": zsL[m]}, yB[m], groups[m], "A's face (copy A)")
    print("\n  A's own face vs A's label (sanity):", "UAR %.2f" % war_uar(zsA, yA)[1])

    # ---------- 2. balanced LR, out-of-fold over 45 episodes ----------
    pres = lambda r: F[r][:, 10:12]
    SETS = {
        'A': np.hstack([F['A']]),
        'A+listener_presence': np.hstack([F['A'], pres('listener')]),
        'A+listener': np.hstack([F['A'], F['listener']]),
        'A+listener_early': np.hstack([F['A'], F['listener_early']]),
        'listener': F['listener'],
        'A+ctx': np.hstack([F['A'], F['ctx_II'], F['ctx_I']]),
        'A+ctx+listener': np.hstack([F['A'], F['ctx_II'], F['ctx_I'], F['listener']]),
        'A+listener+listener_ctx': np.hstack([F['A'], F['listener'], F['listener_ctx']]),
    }
    P = {k: oof_lr(X, yB, groups) for k, X in SETS.items()}
    print("\nPer-shuffle UAR (all MCIS):", {k: [round(war_uar(p, yB)[1], 2) for p in v] for k, v in P.items()})
    for sub, m in [("ALL MCIS", np.ones(len(yB), bool)), ("listener visible", vis),
                   ("listener in first 80%", early), ("listener only in last 20%", vis & ~early),
                   ("listener not visible", ~vis)]:
        for ref, names in [('A', ['A', 'A+listener_presence', 'A+listener', 'A+listener_early', 'listener',
                                  'A+listener+listener_ctx']),
                           ('A+ctx', ['A+ctx', 'A+ctx+listener'])]:
            report(f"LR out-of-fold (shuffle 0), {sub}", {k: P[k][0][m] for k in names}, yB[m], groups[m], ref)
        # mean over shuffles of the key delta
        d = [war_uar(P['A+listener'][s][m], yB[m])[1] - war_uar(P['A'][s][m], yB[m])[1] for s in range(3)]
        dp = [war_uar(P['A+listener_presence'][s][m], yB[m])[1] - war_uar(P['A'][s][m], yB[m])[1] for s in range(3)]
        print(f"  shuffles: Δ(A+listener − A) UAR {np.round(d, 2)} | Δ(A+presence − A) {np.round(dp, 2)}")


if __name__ == '__main__':
    main()
