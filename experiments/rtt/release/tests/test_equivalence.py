"""The package gives the same outputs as the code it was refactored from (synthetic data, CPU, no dataset needed).
reference_model.py / reference_protocol.py hold that code; run `python -m pytest tests`."""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import reference_model as ref
from reference_protocol import reference_folds, reference_two_level_boot
from partiset import BASE, FULL, HP, RN, Batch, PartiSet, PartiSetCore, train_eval, two_level_boot, war_uar
from partiset.cv import assign_folds, cv_fold_rows

N, FD, NV = 24, 146, 9


def synthetic():
    torch.manual_seed(0)
    g = dict(FACE=torch.randn(N, 3, 3, 24, FD), FMASK=torch.rand(N, 3, 3, 24) > 0.5, POOL=torch.randn(N, 1, 3, 32, FD),
             PMASK=torch.rand(N, 1, 3, 32) > 0.4, TXT=torch.randn(N, 3, 512), AUD=F.normalize(torch.randn(N, 3, 527), dim=-1),
             AFD=(torch.rand(N, 3) > 0.3).float(), SCN=torch.randn(N, 3, 1024), VOI=torch.randn(N, 3, NV),
             YB=torch.randint(0, 7, (N,)), YA=torch.randint(0, 7, (N,)))
    g['FMASK'][:, 0, 2, 0] = True
    for k, v in dict(FDIM=FD, NVOICE=NV, DEVICE='cpu', RN=RN, FULL=FULL).items():
        g[k] = v
    for k, v in g.items():                        # the reference code reads module-level tensors
        setattr(ref, k, v)
    batch = Batch(g['FACE'], g['POOL'], g['FMASK'], g['PMASK'], g['TXT'], g['AUD'], g['AFD'], g['SCN'], g['VOI'], g['YB'], g['YA'])
    return batch


def test_models_match_reference():
    b = synthetic()
    ix = torch.arange(N)
    cfgs = [('core', FULL), ('core', {**FULL, 'slots': False}), ('core', {**FULL, 'aux': False}),
            ('full', BASE), ('full', {**BASE, 'readout': 'mean'}), ('full', {**BASE, 'drop': ('L',)}),
            ('full', {**BASE, 'clips': (2,)}), ('full', {**BASE, 'absent': 'mask'}), ('full', {**BASE, 'drop': ('speech', 'scene')})]
    for kind, cfg in cfgs:
        torch.manual_seed(1)
        old = ref.ReferenceCore(cfg) if kind == 'core' else ref.ReferenceTokenModel(cfg)
        torch.manual_seed(1)
        new = PartiSetCore(cfg, FD, NV) if kind == 'core' else PartiSet(cfg, FD, NV)
        assert list(old.state_dict()) == list(new.state_dict()), cfg                  # same parameters, same order
        assert all(torch.equal(p, q) for p, q in zip(old.state_dict().values(), new.state_dict().values())), cfg   # same init
        new.load_state_dict(old.state_dict())
        old.eval(); new.eval()
        with torch.no_grad():
            lo, aux_old = old(ix)
            ln, aux_new = new(b)
        assert torch.allclose(lo, ln, atol=1e-6), cfg
        assert set(aux_old) == set(aux_new), cfg


def test_param_count():
    n = sum(p.numel() for p in PartiSet(BASE, 146, 9).parameters())
    assert 0.65e6 < n < 0.8e6, n


def test_training_is_deterministic():
    b = synthetic()

    class X:
        yb = b.yb
        index = staticmethod(lambda ix: Batch(*(t[ix] for t in b)))
    hp = dict(HP, epochs=3, patience=2, batch=8)
    tr, dv, te = torch.arange(0, 12), torch.arange(12, 18), torch.arange(18, 24)
    p1, s1 = train_eval(lambda: PartiSetCore(FULL, FD, NV), hp, X, tr, dv, te, 5, 'cpu')
    p2, s2 = train_eval(lambda: PartiSetCore(FULL, FD, NV), hp, X, tr, dv, te, 5, 'cpu')
    assert np.allclose(p1, p2) and s1 == s2 and p1.shape == (6, 7) and np.allclose(p1.sum(1), 1, atol=1e-5)


def test_folds_match_reference():
    src = np.array([f'ep{i % 13}' for i in range(260)])
    fold, load = assign_folds(src, 5)
    ref_fold, ref_load = reference_folds(pd.DataFrame({'source_folder': src}))
    assert fold == ref_fold and load == ref_load
    eps = sorted(set(src))
    for f in range(5):
        fit, dev, te, _ = cv_fold_rows(src, fold, f, 5)
        tr_eps = [e for e in eps if fold[e] != f]
        dev_eps = sorted(random.Random(100 + f).sample(tr_eps, 5))
        assert set(src[dev]) == set(dev_eps) and not set(src[fit]) & set(dev_eps) and not set(src[te]) & set(tr_eps)


def test_bootstrap_matches_reference():
    rng = np.random.default_rng(0)
    src = np.repeat(np.arange(9).astype(str), 12); y = rng.integers(0, 7, len(src))
    pa, pb = rng.dirichlet(np.ones(7), (3, len(src))), rng.dirichlet(np.ones(7), (3, len(src)))
    assert np.allclose(reference_two_level_boot(pa, pb, y, src, war_uar, n_boot=50), two_level_boot(pa, pb, y, src, n_boot=50))
