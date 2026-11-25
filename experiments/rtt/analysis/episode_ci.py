"""Episode-aware uncertainty for a paired model comparison (development CV or test predictions).

Input .npz: sample_id, y, src (episode of each sample), and one array [seeds, N, 7] of class probabilities per model.
For models a and b the script reports, from the seed ensembles (mean of the seed probabilities):
  * UAR / WAR of each model and the paired difference a - b on all samples;
  * seed variation: mean and SD of the per-seed differences (separate from sample uncertainty);
  * sample uncertainty, resampling whole episodes (samples of one episode are not independent):
      - cluster bootstrap over episodes (percentile interval, n_boot draws),
      - Bayesian bootstrap (Dirichlet(1) episode weights; keeps every class in every draw),
      - leave-one-episode-out range and the number of episodes in which a beats b.
Difference of UAR is computed over the classes present in the (weighted) sample."""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--oof', required=True); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
ap.add_argument('--n-boot', type=int, default=2000); ap.add_argument('--out', default='episode_ci.csv')
a = ap.parse_args()

z = np.load(a.oof, allow_pickle=True)
y, src = z['y'].astype(int), z['src']
PA, PB = z[a.a], z[a.b]
eps = np.unique(src)
ep_idx = [np.where(src == e)[0] for e in eps]
K = 7


def metrics(pred, w):
    """weighted WAR and UAR (macro recall over classes with positive weight)."""
    war = (w * (pred == y)).sum() / w.sum() * 100
    rec = []
    for c in range(K):
        m = y == c
        if (w * m).sum() > 0:
            rec.append((w * m * (pred == y)).sum() / (w * m).sum() * 100)
    return war, float(np.mean(rec))


def diff(pa, pb, w):
    wa, ua = metrics(pa, w); wb, ub = metrics(pb, w)
    return ua - ub, wa - wb


ones = np.ones(len(y))
ea, eb = PA.mean(0).argmax(1), PB.mean(0).argmax(1)
(dU, dW), (wa, ua), (wb, ub) = diff(ea, eb, ones), metrics(ea, ones), metrics(eb, ones)
per = np.array([diff(PA[s].argmax(1), PB[s].argmax(1), ones) for s in range(len(PA))])
rng = np.random.default_rng(0)


def ep_weights(counts):
    w = np.zeros(len(y))
    for c, ix in zip(counts, ep_idx):
        w[ix] = c
    return w


cl = np.array([diff(ea, eb, ep_weights(np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps)))) for _ in range(a.n_boot)])
bb = np.array([diff(ea, eb, ep_weights(rng.dirichlet(np.ones(len(eps))) * len(eps))) for _ in range(a.n_boot)])
loo = np.array([diff(ea, eb, ep_weights(np.where(np.arange(len(eps)) == i, 0, 1))) for i in range(len(eps))])
per_ep = np.array([diff(ea, eb, ep_weights((np.arange(len(eps)) == i).astype(float))) for i in range(len(eps))])
q = lambda v: np.percentile(v, [2.5, 97.5])
rows = []
for k, (i, name) in enumerate(((0, 'UAR'), (1, 'WAR'))):
    rows.append(dict(metric=name, a=(ua if i == 0 else wa), b=(ub if i == 0 else wb), delta=(dU, dW)[i],
                     seed_delta_mean=per[:, i].mean(), seed_delta_sd=per[:, i].std(ddof=1) if len(per) > 1 else np.nan,
                     boot_lo=q(cl[:, i])[0], boot_hi=q(cl[:, i])[1], bayes_lo=q(bb[:, i])[0], bayes_hi=q(bb[:, i])[1],
                     loo_min=loo[:, i].min(), loo_max=loo[:, i].max(), episodes_a_better=int((per_ep[:, i] > 0).sum()),
                     episodes=len(eps), samples=len(y)))
R = pd.DataFrame(rows)
R.to_csv(a.out, index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
print(f'{a.a} vs {a.b}'); print(R.round(2).to_string(index=False))
