"""Do the same samples that are hard for PartiSet also hard for the other models? (development CV predictions)

Inputs: --oof (.npz: sample_id, y, src, [seeds,N,7] probabilities per model), --split-csv (source_folder_split_seed42.csv with
clip3_emotion = emotion of the last observed speaker A and clip4_emotion = target) and optionally --groups-csv (sample_id plus
any 0/1 or string columns, e.g. label certainty or responder visibility).
Groups: mirror (target emotion equals A's emotion) / non-mirror, plus every column of --groups-csv.
For every group and model: n, accuracy, UAR (classes present), per-class support, and class-adjusted accuracy (samples
re-weighted so that the group has the class distribution of all samples). Also the 'copy A' reference (predict A's emotion) and
the overlap of errors between each pair of models on the same samples."""
import argparse

import numpy as np
import pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
ap = argparse.ArgumentParser()
ap.add_argument('--oof', required=True); ap.add_argument('--split-csv', required=True)
ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--groups-csv'); ap.add_argument('--out', default='cross_model_groups.csv')
a = ap.parse_args()

z = np.load(a.oof, allow_pickle=True)
sid, y = z['sample_id'], z['y'].astype(int)
sp = pd.read_csv(a.split_csv, dtype=str).set_index('sample_id').loc[sid]
yA = sp['clip3_emotion'].map({e: i for i, e in enumerate(EMO)}).values.astype(int)
assert (sp['clip4_emotion'].map({e: i for i, e in enumerate(EMO)}).values == y).all(), 'labels of the split and the predictions differ'
G = {'all': np.ones(len(y), bool), 'mirror (target = A emotion)': y == yA, 'non-mirror': y != yA}
if a.groups_csv:
    g = pd.read_csv(a.groups_csv).set_index('sample_id').reindex(sid)
    for c in g.columns:
        for v in g[c].dropna().unique():
            G[f'{c} = {v}'] = (g[c] == v).values
pred = {m: z[m].mean(0).argmax(1) for m in a.models}
pred['copy A'] = yA
prior = np.bincount(y, minlength=7) / len(y)


def stats(p, m):
    ys, ps = y[m], p[m]
    rec = [(ps[ys == c] == c).mean() * 100 for c in range(7) if (ys == c).any()]
    pg = np.bincount(ys, minlength=7) / max(len(ys), 1)
    w = np.where(pg[ys] > 0, prior[ys] / np.maximum(pg[ys], 1e-12), 0)
    return dict(acc=(ps == ys).mean() * 100, uar=float(np.mean(rec)), acc_class_adjusted=(w * (ps == ys)).sum() / w.sum() * 100)


rows = []
for gname, m in G.items():
    for mname, p in pred.items():
        rows.append(dict(group=gname, model=mname, n=int(m.sum()), classes_present=int(len(set(y[m]))), **stats(p, m)))
R = pd.DataFrame(rows)
R.to_csv(a.out, index=False)
pd.set_option('display.width', 200)
print(R.round(2).to_string(index=False))
print('\nerror overlap (all samples): P(both wrong) / P(either wrong) and correlation of correctness')
ms = a.models
for i in range(len(ms)):
    for j in range(i + 1, len(ms)):
        wi, wj = pred[ms[i]] != y, pred[ms[j]] != y
        print(f'{ms[i]} vs {ms[j]}: {(wi & wj).sum() / max((wi | wj).sum(), 1):.2f} | corr {np.corrcoef(wi, wj)[0, 1]:.2f}')
