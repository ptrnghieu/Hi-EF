"""PartiSet vs the feature-matched clip-pooled Baseline on samples with <=2 and >=3 detected people (development CV predictions).

Inputs (written by notebooks g41 and g39 on Kaggle):
  --oof    g41_oof_probs.npz   sample_id, y, src, RoleNet [seeds,N,7] (= PartiSet), PaperBest_FM [seeds,N,7] (= feature-matched Baseline)
  --people people_per_sample.csv (from analysis/count_people.py; or g39_per_mcis.csv)  sample_id, n_people (distinct people in clips I-III, as in the 47% statistic of the paper)
Per group: UAR / WAR of the seed ensemble (mean of the seed probabilities), per-seed mean, and the PartiSet - Baseline difference
with a 95% interval from resampling episodes (2000 draws). Also the difference of the two group differences."""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--oof', required=True); ap.add_argument('--people', required=True)
ap.add_argument('--a', default='RoleNet'); ap.add_argument('--b', default='PaperBest_FM')
ap.add_argument('--out', default='people_groups.csv'); ap.add_argument('--n-boot', type=int, default=2000)
a = ap.parse_args()


def war_uar(pred, y):
    war = (pred == y).mean() * 100
    return war, np.mean([(pred[y == c] == c).mean() * 100 for c in range(7) if (y == c).any()])


z = np.load(a.oof, allow_pickle=True)
ppl = pd.read_csv(a.people).set_index('sample_id')
sid = z['sample_id']
assert set(sid) <= set(ppl.index), 'samples of the predictions missing from the people table'
n_people = ppl.loc[sid, 'n_people'].values
y, src = z['y'], z['src']
PA, PB = z[a.a], z[a.b]
ea, eb = PA.mean(0).argmax(1), PB.mean(0).argmax(1)
groups = {'<=2 people': n_people <= 2, '>=3 people': n_people >= 3}


def diff(mask, idx_eps, metric):
    """metric 0 = WAR, 1 = UAR of (a - b) on the samples of `mask` whose episode is in idx_eps (list, with repeats)."""
    sel = np.concatenate([np.where((src == e) & mask)[0] for e in idx_eps])
    return war_uar(ea[sel], y[sel])[metric] - war_uar(eb[sel], y[sel])[metric]


rng = np.random.default_rng(0)
eps = np.unique(src)
boots = [rng.choice(eps, len(eps)) for _ in range(a.n_boot)]
rows = []
for name, m in groups.items():
    row = {'group': name, 'N': int(m.sum()), 'episodes': len(set(src[m]))}
    for tag, P, e in (('PartiSet', PA, ea), ('Baseline-FM', PB, eb)):
        row[f'{tag}_UAR'], row[f'{tag}_WAR'] = war_uar(e[m], y[m])[1], war_uar(e[m], y[m])[0]
        per = np.array([war_uar(p.argmax(1)[m], y[m]) for p in P])
        row[f'{tag}_UAR_seedmean'], row[f'{tag}_WAR_seedmean'] = per[:, 1].mean(), per[:, 0].mean()
    for k, metric in (('UAR', 1), ('WAR', 0)):
        d = diff(m, eps, metric)
        bs = [diff(m, b, metric) for b in boots]
        row[f'd{k}'], row[f'd{k}_lo'], row[f'd{k}_hi'] = d, *np.percentile(bs, [2.5, 97.5])
    rows.append(row)
R = pd.DataFrame(rows)
# difference of differences (>=3 minus <=2) on UAR
m2, m3 = groups['<=2 people'], groups['>=3 people']
dd = diff(m3, eps, 1) - diff(m2, eps, 1)
bs = [diff(m3, b, 1) - diff(m2, b, 1) for b in boots]
R.loc[len(R)] = {'group': '(>=3) - (<=2) in dUAR', 'dUAR': dd, 'dUAR_lo': np.percentile(bs, 2.5), 'dUAR_hi': np.percentile(bs, 97.5)}
R.to_csv(a.out, index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
print(R.round(2).to_string(index=False))
