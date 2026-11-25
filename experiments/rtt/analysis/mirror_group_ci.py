import sys, numpy as np, pandas as pd
O, SPL = sys.argv[1], sys.argv[2]
z = np.load(O + '/g41_oof_probs.npz', allow_pickle=True)
sp = pd.read_csv(SPL, dtype=str).set_index('sample_id').loc[z['sample_id']]
EMO = ['angry','disgust','fear','happy','neutral','sad','surprise']
m = {e: i for i, e in enumerate(EMO)}
y = z['y'].astype(int); yA = sp['clip3_emotion'].map(m).values.astype(int); src = z['src']
pred = {k: z[k].mean(0).argmax(1) for k in ('RoleNet', 'PaperBest_FM', 'PaperBest')}
eps = np.unique(src); idx = [np.where(src == e)[0] for e in eps]
def uar(p, w):
    r = [(w*(y==c)*(p==y)).sum()/(w*(y==c)).sum()*100 for c in range(7) if (w*(y==c)).sum() > 0]
    return np.mean(r)
def acc(p, w): return (w*(p==y)).sum()/w.sum()*100
def W(cnt):
    w = np.zeros(len(y))
    for c, ix in zip(cnt, idx): w[ix] = c
    return w
mir = (y == yA)
rng = np.random.default_rng(0)
B = [W(np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps))) for _ in range(2000)]
def d(a, b, g, f, w): return f(pred[a], w*g) - f(pred[b], w*g)
for b in ('PaperBest_FM', 'PaperBest'):
    for name, f in (('UAR', uar), ('acc', acc)):
        out = {}
        for gn, g in (('mirror', mir), ('non-mirror', ~mir)):
            pt = d('RoleNet', b, g, f, np.ones(len(y)))
            bs = [d('RoleNet', b, g, f, w) for w in B]
            out[gn] = (pt, *np.percentile(bs, [2.5, 97.5]))
        gap_pt = out['mirror'][0] - out['non-mirror'][0]
        gap = [d('RoleNet', b, mir, f, w) - d('RoleNet', b, ~mir, f, w) for w in B]
        print(f'PartiSet - {b} {name}: mirror {out["mirror"][0]:+.2f} [{out["mirror"][1]:+.2f},{out["mirror"][2]:+.2f}] | non-mirror {out["non-mirror"][0]:+.2f} [{out["non-mirror"][1]:+.2f},{out["non-mirror"][2]:+.2f}] | mirror-minus-non {gap_pt:+.2f} [{np.percentile(gap,2.5):+.2f},{np.percentile(gap,97.5):+.2f}]')
