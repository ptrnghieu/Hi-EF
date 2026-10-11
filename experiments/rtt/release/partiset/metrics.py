import numpy as np


def war_uar(pred, y, k=7):
    """Weighted (accuracy) and unweighted (macro) average recall, in percent."""
    pred, y = np.asarray(pred), np.asarray(y)
    war = (pred == y).mean() * 100
    uar = np.mean([(pred[y == c] == c).mean() * 100 for c in range(k) if (y == c).any()])
    return war, uar


def source_boot_ci(pred, y, src, k=7, n_boot=2000, seed=0):
    """95% interval by resampling whole source folders (episodes) with replacement."""
    pred, y, src = np.asarray(pred), np.asarray(y), np.asarray(src)
    rng = np.random.default_rng(seed)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        stats.append(war_uar(pred[idx], y[idx], k))
    return np.percentile(np.array(stats), [2.5, 97.5], axis=0)


def two_level_boot(pa, pb, y, src, n_boot=2000, seed=0):
    """pa, pb: [n_seeds, N, 7] probabilities of two arms. Resamples the seeds of each arm and the episodes and returns the 95%
    interval of UAR(a) - UAR(b) of the seed ensembles."""
    rng = np.random.default_rng(seed)
    groups = [np.where(src == e)[0] for e in np.unique(src)]
    d = []
    for _ in range(n_boot):
        ea = pa[rng.integers(0, len(pa), len(pa))].mean(0).argmax(1)
        eb = pb[rng.integers(0, len(pb), len(pb))].mean(0).argmax(1)
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d.append(war_uar(ea[idx], y[idx])[1] - war_uar(eb[idx], y[idx])[1])
    return np.percentile(d, [2.5, 97.5])
