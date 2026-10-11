"""Reference fold assignment and two-level bootstrap (copied from the first implementation)."""
import random

import numpy as np
import pandas as pd


def reference_folds(DEV, N_OUTER=5):
    sizes = DEV.source_folder.value_counts()
    order = list(sizes.index)
    random.Random(0).shuffle(order)
    order = sorted(order, key=lambda e: -sizes[e])
    load_, FOLD = [0] * N_OUTER, {}
    for e in order:
        f = int(np.argmin(load_)); FOLD[e] = f; load_[f] += sizes[e]
    return FOLD, load_


def reference_two_level_boot(pa, pb, y_all, src, war_uar, n_boot=2000, seed=0):
    uar_of = lambda pred, idx: war_uar(pred[idx], y_all[idx], 7)[1]
    rng = np.random.default_rng(seed)
    groups = [np.where(src == e)[0] for e in np.unique(src)]
    d = []
    for _ in range(n_boot):
        ea = pa[rng.integers(0, len(pa), len(pa))].mean(0).argmax(1)
        eb = pb[rng.integers(0, len(pb), len(pb))].mean(0).argmax(1)
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d.append(uar_of(ea, idx) - uar_of(eb, idx))
    return np.percentile(d, [2.5, 97.5])
