"""Cross-validation of PartiSet under input controls (same folds, early-stopping episodes and seeds as run_partiset_cv.py).

Arms: full model, listener tokens emptied, listener replaced by the other-participants slot, listener taken from another
sample, continuity of the slots across turns broken, and the matching global-permutation control. The development
episodes only are used; the test split is never read."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from partiset import BASE, HP, PartiSet, train_eval, war_uar
from partiset.controls import ControlledInputs
from partiset.cv import assign_folds, cv_fold_rows
from partiset.pipeline import prepare

ARMS = [('Full', 'none', {}), ('listener-off (masked)', 'listener_off', {}), ('listener-from-others', 'listener_other', {}),
        ('listener-from-other-sample', 'listener_swap', {}), ('continuity-broken', 'continuity', {}), ('global-permutation', 'global_perm', {}),
        ('no-slot-embeddings', 'none', {'slot_emb': False}), ('A-L-swapped-half', 'al_swap_half', {}),
        ('replicated-pooling', 'replicated_pool', {}), ('random-grouping', 'random_grouping', {})]

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--features', required=True)
ap.add_argument('--split-csv', required=True); ap.add_argument('--participant-features', required=True)
ap.add_argument('--out', default='.'); ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 7, 11, 19, 23, 31, 37, 43])
ap.add_argument('--arms', nargs='+', default=[a[0] for a in ARMS], help='subset of arm names')
ap.add_argument('--tag', default='', help='suffix of the output files, to run the arms in several sessions')
a = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
sp, B, base = prepare(a.dataset, a.features, a.split_csv, a.participant_features, dev, rows=lambda t: t.split != 'test')
src, y = sp.source_folder.values, sp.yB.values
fold, load = assign_folds(src, a.folds)
fold_of_row = np.array([fold[e] for e in src])
arms = [x for x in ARMS if x[0] in a.arms]
T = lambda r: torch.tensor(r, device=dev)
oof = {x[0]: np.full((len(a.seeds), len(sp), 7), np.nan, np.float32) for x in arms}
log = []
for f in range(a.folds):
    fit, dv, te, trr = cv_fold_rows(src, fold, f, 5)
    base.set_faces(*B.build_face_tensors(sorted(set(sp.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel())), dev)[:2])
    for name, kind, over in arms:
        X = ControlledInputs(base, kind, seed=1000 + f)
        for si, s in enumerate(a.seeds):
            p, sel = train_eval(lambda: PartiSet({**BASE, **over}, B.FDIM, 9), HP, X, T(fit), T(dv), T(te), s + 1000 * f, dev)
            oof[name][si, te] = p
            w, u = war_uar(p.argmax(1), y[te])
            log.append(dict(fold=f, arm=name, seed=s, sel_UAR=sel, UAR=u, WAR=w)); print(f'fold {f} {name} seed {s} done', flush=True)
os.makedirs(a.out, exist_ok=True)
sfx = f'_{a.tag}' if a.tag else ''
pd.DataFrame(log).to_csv(os.path.join(a.out, f'partiset_controls_log{sfx}.csv'), index=False)
safe = lambda s: re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_')
np.savez(os.path.join(a.out, f'partiset_controls_oof{sfx}.npz'), sample_id=sp.sample_id.values, fold=fold_of_row, y=y, src=src,
         **{safe(k): v for k, v in oof.items()})
print('saved', a.out)
