"""Equivalence of the package with the notebooks it was refactored from (synthetic data, CPU).
The class definitions are taken verbatim from notebooks/g14_ten_seed_confirmation_cv.ipynb and executed with synthetic
module-level tensors; the package models, loaded with the same weights, must give identical logits."""
import ast
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from partiset import (BASE, FULL, RN, Batch, HP, PartiSet, PartiSetCore, train_eval, two_level_boot, war_uar)
from partiset.cv import assign_folds, cv_fold_rows

NB = json.load(open(os.path.join(ROOT, 'notebooks', 'g14_ten_seed_confirmation_cv.ipynb')))
CODE = [''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code']
N, FD, NV = 24, 146, 9


def notebook_namespace():
    torch.manual_seed(0)
    g = dict(torch=torch, nn=nn, F=F, np=np, RN=RN, FULL=FULL, DEVICE='cpu', FDIM=FD, NVOICE=NV, N=N)
    g['FACE'] = torch.randn(N, 3, 3, 24, FD)
    g['FMASK'] = torch.rand(N, 3, 3, 24) > 0.5
    g['FMASK'][:, 0, 2, 0] = True
    g['POOL'] = torch.randn(N, 1, 3, 32, FD)
    g['PMASK'] = torch.rand(N, 1, 3, 32) > 0.4
    g['TXT'], g['AUD'] = torch.randn(N, 3, 512), F.normalize(torch.randn(N, 3, 527), dim=-1)
    g['AFD'], g['SCN'], g['VOI'] = (torch.rand(N, 3) > 0.3).float(), torch.randn(N, 3, 1024), torch.randn(N, 3, NV)
    g['YB'], g['YA'] = torch.randint(0, 7, (N,)), torch.randint(0, 7, (N,))
    src = '\n'.join(s for s in CODE if 'class FramePool' in s or 'class RoleNetTok' in s)
    classes = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(classes, []), 'nb', 'exec'), g)
    return g


def batch(g, role):
    ix = torch.arange(N)
    return ix, Batch(g['FACE'], g['POOL'], g['FMASK'], g['PMASK'], g['TXT'], g['AUD'], g['AFD'], g['SCN'], g['VOI'],
                     g['YB'], g['YA'])


def test_models_match_notebook():
    g = notebook_namespace()
    ix, b = batch(g, True)
    cfgs = [('role', FULL), ('role', {**FULL, 'role': False}), ('role', {**FULL, 'aux': False}),
            ('tok', BASE), ('tok', {**BASE, 'readout': 'mean'}), ('tok', {**BASE, 'drop': ('L',)}),
            ('tok', {**BASE, 'clips': (2,)}), ('tok', {**BASE, 'absent': 'mask'}), ('tok', {**BASE, 'drop': ('speech', 'scene')})]
    for kind, cfg in cfgs:
        torch.manual_seed(1)
        old = g['RoleNet'](cfg) if kind == 'role' else g['RoleNetTok'](cfg)
        torch.manual_seed(1)
        new = PartiSetCore(cfg, FD, NV) if kind == 'role' else PartiSet(cfg, FD, NV)
        assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters()), cfg
        assert list(old.state_dict()) == list(new.state_dict()), cfg      # same parameters in the same order
        assert all(torch.equal(p, q) for p, q in zip(old.state_dict().values(), new.state_dict().values())), cfg  # same init
        new.load_state_dict(old.state_dict())
        old.eval(); new.eval()
        with torch.no_grad():
            lo, la = old(ix)
            ln, an = new(b)
        assert torch.allclose(lo, ln, atol=1e-6), cfg
        assert set(la) == set(an), cfg


def test_param_count():
    n = sum(p.numel() for p in PartiSet(BASE, 146, 9).parameters())
    assert 0.65e6 < n < 0.8e6, n


def test_training_is_deterministic_and_selects_best_epoch():
    class X:                                                   # minimal Inputs-like container
        pass
    g = notebook_namespace()
    _, b = batch(g, True)
    x = X(); x.yb = b.yb; x.index = lambda ix: Batch(*(t[ix] for t in b))
    hp = dict(HP['role'], epochs=3, patience=2, batch=8)
    tr, dv, te = torch.arange(0, 12), torch.arange(12, 18), torch.arange(18, 24)
    p1, s1 = train_eval(lambda: PartiSetCore(FULL, FD, NV), hp, x, tr, dv, te, 5, 'cpu')
    p2, s2 = train_eval(lambda: PartiSetCore(FULL, FD, NV), hp, x, tr, dv, te, 5, 'cpu')
    assert np.allclose(p1, p2) and s1 == s2 and p1.shape == (6, 7) and np.allclose(p1.sum(1), 1, atol=1e-5)


def test_folds_match_notebook_cell():
    src = np.array([f'ep{i % 13}' for i in range(200)] + [f'ep{i % 13}' for i in range(60)])
    sizes_src = src
    DEV = pd.DataFrame({'source_folder': sizes_src})
    cell = [s for s in CODE if 'load_, FOLD = ' in s][0]
    ns = dict(DEV=DEV, N_OUTER=5, random=random, np=np, pd=pd, N_INNER_DEV=5, EPS=sorted(set(src)))
    pre = cell.split('y_all = DEV.yB.values')[0].replace('print("fold sizes (MCIS):", load_)', '')
    exec(pre, ns)
    fold, load = assign_folds(src, 5)
    assert fold == ns['FOLD'] and load == ns['load_']
    for f in range(5):
        fit, dev, te, trr = cv_fold_rows(src, fold, f, 5)
        tr_eps = [e for e in ns['EPS'] if ns['FOLD'][e] != f]
        dev_eps = sorted(random.Random(100 + f).sample(tr_eps, 5))
        assert set(src[dev]) == set(dev_eps) and not set(src[fit]) & set(dev_eps) and not set(src[te]) & set(tr_eps)


def test_bootstrap_matches_notebook():
    rng = np.random.default_rng(0)
    src = np.repeat(np.arange(9).astype(str), 12); y = rng.integers(0, 7, len(src))
    pa, pb = rng.dirichlet(np.ones(7), (3, len(src))), rng.dirichlet(np.ones(7), (3, len(src)))
    cell = [s for s in CODE if 'def two_level_boot' in s][0]
    ns = dict(np=np, src=src, y_all=y, war_uar=lambda p, t, k: war_uar(p, t, k))
    exec(cell.split('def episode_boot')[0], ns)
    assert np.allclose(ns['two_level_boot'](pa, pb, n_boot=50), two_level_boot(pa, pb, y, src, n_boot=50))
