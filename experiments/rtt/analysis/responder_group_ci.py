import sys, numpy as np, pandas as pd
O, G = sys.argv[1], sys.argv[2]
z = np.load(O + '/g41_oof_probs.npz', allow_pickle=True)
g = pd.read_csv(G).set_index('sample_id').loc[z['sample_id']]
y, src = z['y'].astype(int), z['src']
arms = {'PartiSet': 'RoleNet', 'Baseline-PartiSet-features': 'PaperBest_FM', 'Baseline': 'PaperBest'}
pred = {k: z[v].mean(0).argmax(1) for k, v in arms.items()}
eps = np.unique(src); idx = [np.where(src == e)[0] for e in eps]
rng = np.random.default_rng(0)
Bt = [np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps)) for _ in range(2000)]
def W(c):
    w = np.zeros(len(y))
    for k, ix in zip(c, idx): w[ix] = k
    return w
Bw = [W(c) for c in Bt]
def uar(p, w):
    r = [(w*(y==c)*(p==y)).sum()/(w*(y==c)).sum()*100 for c in range(7) if (w*(y==c)).sum() > 0]
    return np.mean(r)
def war(p, w): return (w*(p==y)).sum()/w.sum()*100
rows = []
for grp in ['all'] + list(g['group'].unique()):
    m = np.ones(len(y), bool) if grp == 'all' else (g['group'].values == grp)
    r = dict(group=grp, n=int(m.sum()), classes=len(set(y[m])))
    for k, p in pred.items(): r[f'{k}_UAR'] = uar(p, m.astype(float)); r[f'{k}_WAR'] = war(p, m.astype(float))
    for b in ('Baseline-PartiSet-features', 'Baseline'):
        d = uar(pred['PartiSet'], m.astype(float)) - uar(pred[b], m.astype(float))
        bs = [uar(pred['PartiSet'], w*m) - uar(pred[b], w*m) for w in Bw if (w*m).sum() > 0]
        r[f'dUAR_vs_{b}'] = d; r[f'lo_{b}'], r[f'hi_{b}'] = np.percentile(bs, [2.5, 97.5])
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(sys.argv[3], index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(R.round(2).to_string(index=False))
