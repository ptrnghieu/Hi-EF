"""Episode-level splits used in the paper. Samples of one source video (episode) never cross a split."""
import random

import numpy as np


def inner_dev_episodes(eps, n, seed):
    """Early-stopping episodes: `random.Random(seed).sample(eps, n)` (G10: seed 2026; G14: seed 100 + fold)."""
    return sorted(random.Random(seed).sample(list(eps), n))


def assign_folds(src, n_outer=5):
    """Greedy size balancing of episodes over folds (G14): episodes sorted by decreasing size are put into the lightest fold."""
    import pandas as pd
    sizes = pd.Series(src).value_counts()
    order = list(sizes.index)
    random.Random(0).shuffle(order)
    order = sorted(order, key=lambda e: -sizes[e])
    load, fold = [0] * n_outer, {}
    for e in order:
        f = int(np.argmin(load)); fold[e] = f; load[f] += sizes[e]
    return fold, load


def test_protocol_rows(src, is_test, n_inner=5, seed=2026):
    """G10: fit on training episodes minus `n_inner` early-stopping episodes, evaluate on the test rows."""
    eps = np.array(sorted(set(src[~is_test])))
    sel = inner_dev_episodes(eps, n_inner, seed)
    fit = np.where(~is_test & ~np.isin(src, sel))[0]
    dev = np.where(np.isin(src, sel))[0]
    te = np.where(is_test)[0]
    assert not set(src[te]) & set(src[~is_test])
    return fit, dev, te, sel


def cv_fold_rows(src, fold, f, n_inner=5):
    """G14: rows of fit / early-stopping / held-out fold f, and the training episodes."""
    eps = sorted(set(src))
    tr_eps = [e for e in eps if fold[e] != f]
    dev_eps = inner_dev_episodes(tr_eps, n_inner, 100 + f)
    fit = np.where(np.isin(src, tr_eps) & ~np.isin(src, dev_eps))[0]
    dev = np.where(np.isin(src, dev_eps))[0]
    te = np.where(np.array([fold[e] == f for e in src]))[0]
    trr = np.where(np.isin(src, tr_eps))[0]
    return fit, dev, te, trr
