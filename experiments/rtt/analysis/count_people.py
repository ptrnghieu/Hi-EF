"""Number of people per sample from the participant-feature shards alone (no clip IV, no GPU).
Same rule as the party statistics of the paper: ArcFace embeddings of all faces of clips I-III are clustered (average linkage,
cosine distance < 0.55); a person is an identity seen in >= 2 sampled frames. Writes sample_id, n_people (clips I-III) and
n_people_III (clip III)."""
import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser()
ap.add_argument('--split-csv', required=True); ap.add_argument('--shards', required=True)
ap.add_argument('--out', default='people_per_sample.csv'); ap.add_argument('--include-test', action='store_true')
a = ap.parse_args()
SAME_PERSON_COS, MIN_FRAMES = 0.45, 2

sp = pd.read_csv(a.split_csv, dtype=str)
DEV = (sp if a.include_test else sp[sp.split != 'test']).reset_index(drop=True)      # development samples only by default
need = set(DEV[['clip1', 'clip2', 'clip3']].values.ravel())
G8 = {}
for f in sorted(glob.glob(os.path.join(a.shards, '**', 'shard_*.pkl'), recursive=True)):
    G8.update({k: v for k, v in pickle.load(open(f, 'rb')).items() if k in need})
print(f'samples {len(DEV)} | clips with features {len(G8)} / {len(need)}')


def cluster(E):
    if len(E) < 2:
        return np.zeros(len(E), int)
    return AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                   distance_threshold=1 - SAME_PERSON_COS).fit_predict(E)


rows = []
for r in DEV.itertuples():
    cl = [r.clip1, r.clip2, r.clip3]
    items = [(k, j) for k, c in enumerate(cl) for j in range(len(G8.get(c, {}).get('faces', [])))]
    if not items:
        rows.append({'sample_id': r.sample_id, 'n_people': 0, 'n_people_III': 0}); continue
    E = np.stack([G8[cl[k]]['faces'][j]['arc'] for k, j in items]).astype(np.float32)
    lab = cluster(E)
    frames = defaultdict(set)
    for (k, j), p in zip(items, lab):
        frames[(k, p)].add(G8[cl[k]]['faces'][j]['frame'])
    tot = defaultdict(int)
    for (k, p), fr in frames.items():
        tot[p] += len(fr)
    ids3 = {p for (k, p) in frames if k == 2}
    rows.append({'sample_id': r.sample_id, 'n_people': sum(v >= MIN_FRAMES for v in tot.values()),
                 'n_people_III': sum(len(frames[(2, p)]) >= MIN_FRAMES for p in ids3)})
P = pd.DataFrame(rows)
P.to_csv(a.out, index=False)
print('share with >=3 people:', round(float((P.n_people >= 3).mean() * 100), 1), '%')
