"""5-fold episode cross-validation, 10 seeds on the development episodes (episodes never cross folds); the test split is never read. Arms: PartiSet and four ablations."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from partiset import BASE, FULL, HP, PartiSet, PartiSetCore, train_eval, war_uar
from partiset.cv import assign_folds, cv_fold_rows
from partiset.pipeline import prepare

SEEDS = [42, 123, 456, 7, 11, 19, 23, 31, 37, 43]
ARMS = [("Full", lambda B: PartiSet(BASE, B.FDIM, 9)),
        ("Q-meanpool", lambda B: PartiSet({**BASE, 'readout': 'mean'}, B.FDIM, 9)),
        ("minus-L", lambda B: PartiSet({**BASE, 'drop': ('L',)}, B.FDIM, 9)),
        ("T-clipIIIonly", lambda B: PartiSet({**BASE, 'clips': (2,)}, B.FDIM, 9)),
        ("noSlots", lambda B: PartiSetCore({**FULL, 'slots': False}, B.FDIM, 9))]

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--features', required=True)
ap.add_argument('--split-csv', required=True); ap.add_argument('--participant-features', required=True)
ap.add_argument('--out', default='.'); ap.add_argument('--folds', type=int, default=5)
a = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
sp, B, X = prepare(a.dataset, a.features, a.split_csv, a.participant_features, dev, rows=lambda t: t.split != 'test')
src, y = sp.source_folder.values, sp.yB.values
fold, load = assign_folds(src, a.folds)
fold_of_row = np.array([fold[e] for e in src])
print('fold sizes:', load)
T = lambda r: torch.tensor(r, device=dev)
oof = {n: np.full((len(SEEDS), len(sp), 7), np.nan, np.float32) for n, _ in ARMS}
log = []
for f in range(a.folds):
    fit, dv, te, trr = cv_fold_rows(src, fold, f)
    X.set_faces(*B.build_face_tensors(sorted(set(sp.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel())), dev)[:2])
    for name, mk in ARMS:
        for si, s in enumerate(SEEDS):
            p, sel = train_eval(lambda: mk(B), HP, X, T(fit), T(dv), T(te), s + 1000 * f, dev)
            oof[name][si, te] = p
            w, u = war_uar(p.argmax(1), y[te])
            log.append(dict(fold=f, arm=name, seed=s, sel_UAR=sel, UAR=u, WAR=w)); print(f'fold {f} {name} seed {s} done', flush=True)
os.makedirs(a.out, exist_ok=True)
pd.DataFrame(log).to_csv(os.path.join(a.out, 'partiset_cv_log.csv'), index=False)
safe = lambda s: re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
np.savez(os.path.join(a.out, 'partiset_cv_oof.npz'), sample_id=sp.sample_id.values, fold=fold_of_row, y=y, src=src,
         **{safe(k): v for k, v in oof.items()})
print("saved", os.path.join(a.out, "partiset_cv_log.csv"), "and", os.path.join(a.out, "partiset_cv_oof.npz"))
