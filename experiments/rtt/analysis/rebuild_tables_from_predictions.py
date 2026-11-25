import sys, numpy as np
O = sys.argv[1]
H = np.load(O + '/hief_test_all.npz', allow_pickle=True); M = np.load(O + '/meld_test_all.npz', allow_pickle=True)
def wu(P, y):
    p = P.mean(0).argmax(1)
    return np.mean([(p[y == c] == c).mean() * 100 for c in range(7) if (y == c).any()]), (p == y).mean() * 100
def wu_single(P, y):
    return wu(P[None], y)
res = {}
for k in [f'row{r}' for r in range(5, 12)] + ['PartiSet']:
    res[k] = (wu(H[k], H['y']), wu(M[k], M['y']))
# Qwen: Hi-EF few-shot = mean over the three example sets (single-set predictions), zero-shot = one run
zs_h = wu(H['zero_shot'], H['y'])
fs_h = tuple(np.mean([wu(H[f'few_shot_{i}'], H['y']) for i in range(3)], 0))
zs_m = wu(M['zero_shot'], M['y'])
FS_M = (24.85, 23.19)   # few-shot set 0 on MELD: from the summary CSV (probabilities were not in the uploaded file)
rows = [  # key, inputs, intra, inter, label
 ('row5', r'F,B,S', 'Trans', 'LSTM+Trans'), ('row6', r'F,B,S,A', 'Trans', 'LSTM+Trans'), ('row7', r'F,B,S,T', 'Trans', 'LSTM+Trans'),
 ('row8', r'F,B,S,T,A', 'Avg', 'Avg'), ('row9', r'F,B,S,T,A', 'Trans', 'Avg'), ('row10', r'F,B,S,T,A', 'Avg', 'LSTM+Trans'),
 ('row11', r'F,B,S,T,A', 'Trans', 'LSTM+Trans')]
allvals = {k: res[k] for k, *_ in rows}; allvals['PartiSet'] = res['PartiSet']
cols = {'hU': [res[k][0][0] for k in allvals], 'hW': [res[k][0][1] for k in allvals],
        'mU': [res[k][1][0] for k in allvals] + [zs_m[0], FS_M[0]], 'mW': [res[k][1][1] for k in allvals] + [zs_m[1], FS_M[1]]}
hU = max(cols['hU'] + [zs_h[0], fs_h[0]]); hW = max(cols['hW'] + [zs_h[1], fs_h[1]])
best = {'hU': hU, 'hW': hW, 'mU': max(cols['mU']), 'mW': max(cols['mW'])}
def f(v, key): 
    s = f'{v:.2f}'
    return rf'\textbf{{{s}}}' if round(v, 2) == round(best[key], 2) else s
print('% ---- Table 1 body (ViT rows retrained on our splits; values computed from saved test predictions) ----')
for k, inp, a, b in rows:
    (hu, hw), (mu, mw) = res[k]
    name = r'ViT (Baseline)' if k == 'row11' else 'ViT'
    print(f'{name} & {inp} & {a} & {b} & {f(hu,"hU")} & {f(hw,"hW")} & {f(mu,"mU")} & {f(mw,"mW")} \\\\')
print(r'\midrule')
print(f'Qwen2.5-VL (zero-shot) & V,T & -- & -- & {f(zs_h[0],"hU")} & {f(zs_h[1],"hW")} & {f(zs_m[0],"mU")} & {f(zs_m[1],"mW")} \\\\')
print(f'Qwen2.5-VL (few-shot) & V,T & -- & -- & {f(fs_h[0],"hU")} & {f(fs_h[1],"hW")} & {f(FS_M[0],"mU")} & {f(FS_M[1],"mW")} \\\\')
print(r'\midrule')
(hu, hw), (mu, mw) = res['PartiSet']
print(f'PartiSet (ours) & F,S,T,A & Set-based & Set-based & {f(hu,"hU")} & {f(hw,"hW")} & {f(mu,"mU")} & {f(mw,"mW")} \\\\')
print()
print('% ---- Table 9 body: published columns copied from the manuscript (please verify against the Hi-EF paper) ----')
pub = {'row5': (22.39, 31.37), 'row6': (21.27, 33.34), 'row7': (21.80, 31.03), 'row8': (18.55, 31.44), 'row9': (21.71, 33.38), 'row10': (23.19, 34.49), 'row11': (23.72, 35.19)}
res_pub = [('ResNet18','F','--','Avg',17.02,22.22),('ResNet18','F,B,S','Avg','Avg',18.58,23.15),('ResNet18','F,B,S','Trans','Avg',19.75,32.41),('ResNet18','F,B,S','Avg','Trans',21.09,28.70),
           ('I3D','F','--','Avg',21.73,28.57),('I3D','F,B,S','Avg','Avg',21.16,29.62),('I3D','F,B,S','Trans','Avg',21.19,29.63),('I3D','F,B,S','Avg','Trans',22.99,29.62)]
for bb, inp, a, b, u, w in res_pub:
    print(f'{bb} & {inp} & {a} & {b} & {u:.2f} & {w:.2f} & -- & -- & -- & -- \\\\')
print(r'\midrule')
for k, inp, a, b in rows:
    (hu, hw), (mu, mw) = res[k]; pu, pw = pub[k]
    name = 'ViT (Baseline)' if k == 'row11' else 'ViT'
    print(f'{name} & {inp} & {a} & {b} & {pu:.2f} & {pw:.2f} & {f(hu,"hU")} & {f(hw,"hW")} & {f(mu,"mU")} & {f(mw,"mW")} \\\\')
print(r'\midrule')
print(f'Qwen2.5-VL-7B, zero-shot & V,T & -- & -- & -- & -- & \\textit{{{zs_h[0]:.2f}}} & \\textit{{{zs_h[1]:.2f}}} & {f(zs_m[0],"mU")} & {f(zs_m[1],"mW")} \\\\')
print(f'Qwen2.5-VL-7B, few-shot & V,T & -- & -- & -- & -- & \\textit{{{fs_h[0]:.2f}}} & \\textit{{{fs_h[1]:.2f}}} & {f(FS_M[0],"mU")} & {f(FS_M[1],"mW")} \\\\')
print(r'\midrule')
(hu, hw), (mu, mw) = res['PartiSet']
print(f'PartiSet (ours) & F,S,T,A & Set-based & Set-based & -- & -- & {f(hu,"hU")} & {f(hw,"hW")} & {f(mu,"mU")} & {f(mw,"mW")} \\\\')
