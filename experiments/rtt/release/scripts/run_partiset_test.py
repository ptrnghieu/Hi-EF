"""Preregistered single read of the test split (G10 protocol): 5 seeds, PartiSet and PartiSet without slots.
Needs --unlock-test; see docs/PREREG_G10_TEST.md and docs/test_accesses.md before running."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from partiset import FULL, HP, PartiSetCore, train_eval, war_uar, source_boot_ci
from partiset.cv import test_protocol_rows
from partiset.pipeline import prepare

SEEDS = [42, 123, 456, 789, 1024]
ARMS = [('PartiSet', FULL), ('PartiSet-noSlots', {**FULL, 'role': False})]

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--features', required=True)
ap.add_argument('--split-csv', required=True); ap.add_argument('--g8', required=True)
ap.add_argument('--out', default='.'); ap.add_argument('--unlock-test', action='store_true')
a = ap.parse_args()
if not a.unlock_test:
    sys.exit('The test split is locked. Pass --unlock-test for the single preregistered run.')

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
sp, B, X = prepare(a.dataset, a.features, a.split_csv, a.g8, dev)
src, y = sp.source_folder.values, sp.yB.values
fit, dv, te, sel = test_protocol_rows(src, (sp.split == 'test').values)
trr = np.where(sp.split.values != 'test')[0]
X.set_faces(*B.build_face_tensors(sorted(set(sp.iloc[trr][['clip1', 'clip2', 'clip3']].values.ravel())), dev)[:2])
T = lambda r: torch.tensor(r, device=dev)
probs, log = {}, []
for name, cfg in ARMS:
    probs[name] = []
    for s in SEEDS:
        p, sel_u = train_eval(lambda: PartiSetCore(cfg, B.FDIM, 9), HP['role'], X, T(fit), T(dv), T(te), s, dev)
        probs[name].append(p)
        w, u = war_uar(p.argmax(1), y[te])
        log.append(dict(arm=name, seed=s, sel_UAR=sel_u, test_UAR=u, test_WAR=w)); print(log[-1], flush=True)
    probs[name] = np.stack(probs[name])
    w, u = war_uar(probs[name].mean(0).argmax(1), y[te])
    lo, hi = source_boot_ci(probs[name].mean(0).argmax(1), y[te], src[te])
    print(f'{name}: ensemble UAR {u:.2f} [{lo[1]:.1f},{hi[1]:.1f}] WAR {w:.2f}')
os.makedirs(a.out, exist_ok=True)
pd.DataFrame(log).to_csv(os.path.join(a.out, 'partiset_test_per_seed.csv'), index=False)
np.savez(os.path.join(a.out, 'partiset_test_probs.npz'), sample_id=sp.sample_id.values[te],
         **{k.replace('-', '_'): v for k, v in probs.items()})
