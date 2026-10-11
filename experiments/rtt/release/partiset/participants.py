"""Participant grouping (Section 3.2): faces of a sample are clustered into people, the people are sorted into the slots
A (speaker of the last turn), L (second most visible) and O (the rest); frame features are assembled per (slot, turn)."""
import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from .config import Constants


def load_feature_shards(shards_dir):
    """Per-clip face and voice records produced by notebooks/shardsa_role_features.ipynb (or m2 for MELD)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(shards_dir, '**', 'shard_*.pkl'), recursive=True)):
        out.update(pickle.load(open(f, 'rb')))
    return out


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def pick_frames(idx, cap):
    return idx if len(idx) <= cap else [idx[i] for i in np.linspace(0, len(idx) - 1, cap).astype(int)]


class ParticipantBuilder:
    """Builds, for the rows of `sp` (clip1..clip3), the slot assignment of every face, the masks and the turn descriptors.

    Attributes after construction: SLOT, PSLOT, FMASK [N,3,3,maxf], PMASK [N,1,3,maxf_pool], VOI [N,3,n_voice],
    VIS (listener visible in the last turn), FDIM."""

    def __init__(self, sp, shards, const=Constants(), progress=None):
        self.sp, self.SH, self.c = sp, shards, const
        c = const
        need = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))
        miss = [x for x in need if x not in shards]
        assert not miss, f"{len(miss)} clips missing from the participant features, e.g. {miss[:3]}"
        self.need = need
        # per-clip frame features except the PCA part (18 dims), and the raw HSEmotion embeddings
        FB, EMB = {}, {}
        for cl in need:
            fs = shards[cl]['faces']
            if not fs:
                FB[cl], EMB[cl] = np.zeros((0, c.n_face_geo), np.float32), None
                continue
            dur = max(shards[cl]['meta'].get('duration') or 0.0, 1e-3)
            fer = np.stack([d['fer'] for d in fs]).astype(np.float32)
            box = np.stack([d['box'] for d in fs])
            FB[cl] = np.concatenate([
                softmax(fer[:, :8]), fer[:, 8:10], np.stack([d['pose'] for d in fs]) / 90.0,
                np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2,
                          np.sqrt(np.clip((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 0, None))], 1),
                np.nan_to_num(np.array([[d['mouth'] * 10] for d in fs], np.float32)),
                np.array([[min(d['t'] / dur, 1.0)] for d in fs], np.float32)], 1).astype(np.float32)
            EMB[cl] = np.stack([d['fer_emb'] for d in fs]) if all('fer_emb' in d for d in fs) else None
        self.FB, self.EMB = FB, EMB
        self.HAS_EMB = all(EMB[x] is not None for x in need if len(shards[x]['faces']))
        self.FDIM = (c.pca_dim if self.HAS_EMB else 0) + c.n_face_geo
        self._assign(progress)

    def _sync(self, cl, face_ids):
        # correlation of mouth opening with the audio energy envelope, and mouth variability, for a set of faces
        A = self.SH[cl]['audio']
        fs = [self.SH[cl]['faces'][j] for j in face_ids]
        fs = [d for d in fs if np.isfinite(d['mouth'])]
        if A is None or len(fs) < 4:
            return 0.0, 0.0
        m = np.array([d['mouth'] for d in fs])
        env = A['env']
        e = np.array([env[min(int(d['t'] * 10), len(env) - 1)] for d in fs]) if len(env) else np.zeros(len(fs))
        r = float(np.corrcoef(m, e)[0, 1]) if m.std() > 1e-6 and e.std() > 1e-9 else 0.0
        return r, float(m.std() * 10)

    def _assign(self, progress):
        SH, c, sp = self.SH, self.c, self.sp
        N = len(sp)
        self.SLOT, self.PSLOT = {}, {}        # (n, slot, clip) / (n, clip) -> (clip id, face indices)
        self.FMASK = np.zeros((N, 3, 3, c.maxf), bool)
        self.PMASK = np.zeros((N, 1, 3, c.maxf_pool), bool)
        self.VOI = np.zeros((N, 3, c.n_voice), np.float32)
        rows = sp.itertuples()
        if progress is not None:
            rows = progress(rows, total=N, desc='participants')
        for n, row in enumerate(rows):
            cl = [row.clip1, row.clip2, row.clip3]
            items = [(k, j) for k, x in enumerate(cl) for j in range(len(SH[x]['faces']))]
            lab = np.zeros(len(items), int)
            E = np.stack([SH[cl[k]]['faces'][j]['arc'] for k, j in items]).astype(np.float32) if items else None
            if len(items) > 1:
                lab = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                              distance_threshold=1 - c.same_person_cos).fit_predict(E)
            frames = defaultdict(set)
            for (k, j), p in zip(items, lab):
                frames[(k, p)].add(SH[cl[k]]['faces'][j]['frame'])
            ids3 = sorted({p for (k, p) in frames if k == 2}, key=lambda p: -len(frames[(2, p)]))
            A = ids3[0] if ids3 else None
            L = ids3[1] if len(ids3) > 1 else None
            slot_of = lambda p: 0 if p == A else (1 if p == L else 2)
            by = defaultdict(list)
            for (k, j), p in zip(items, lab):
                by[(slot_of(p), k)].append(j)
                by[('pool', k)].append(j)
            for k, x in enumerate(cl):
                for r in range(3):
                    idx = pick_frames(sorted(by[(r, k)], key=lambda j: SH[x]['faces'][j]['t']), c.maxf)
                    self.SLOT[(n, r, k)] = (x, idx)
                    self.FMASK[n, r, k, :len(idx)] = True
                idx = pick_frames(sorted(by[('pool', k)], key=lambda j: SH[x]['faces'][j]['t']), c.maxf_pool)
                self.PSLOT[(n, k)] = (x, idx)
                self.PMASK[n, 0, k, :len(idx)] = True
                v = [y for r in range(3) for y in self._sync(x, by[(r, k)])]
                a3, ak = SH[cl[2]]['audio'], SH[x]['audio']
                ok = a3 is not None and ak is not None and a3.get('ecapa') is not None and ak.get('ecapa') is not None
                vcos = float(a3['ecapa'].astype(np.float32) @ ak['ecapa'].astype(np.float32)) if ok else 0.0
                self.VOI[n, k] = v + [vcos, float(ok), len(set(lab)) / 5.0]
        self.VIS = self.FMASK[:, 1, 2].any(-1)

    def build_face_tensors(self, fit_clips, device):
        """Frame tensors [N,3,3,maxf,FDIM] (slots) and [N,1,3,maxf_pool,FDIM] (pooled), with the PCA of the HSEmotion
        embedding fitted on the faces of `fit_clips` (the training clips of the current fold) only."""
        c, N = self.c, len(self.sp)
        pca, var = None, None
        if self.HAS_EMB:
            pool = np.concatenate([self.EMB[x] for x in fit_clips if self.EMB.get(x) is not None])
            pick = np.random.default_rng(0).choice(len(pool), min(60000, len(pool)), replace=False)
            pca = PCA(c.pca_dim, random_state=0).fit(pool[pick].astype(np.float32))
            var = float(pca.explained_variance_ratio_.sum())
        FV = {}
        for x in self.need:
            if len(self.FB[x]) == 0:
                FV[x] = np.zeros((0, self.FDIM), np.float32)
            else:
                FV[x] = (np.concatenate([pca.transform(self.EMB[x].astype(np.float32)), self.FB[x]], 1)
                         if self.HAS_EMB else self.FB[x])
        Fa = np.zeros((N, 3, 3, c.maxf, self.FDIM), np.float16)
        Pa = np.zeros((N, 1, 3, c.maxf_pool, self.FDIM), np.float16)
        for (n, r, k), (x, idx) in self.SLOT.items():
            if idx:
                Fa[n, r, k, :len(idx)] = FV[x][idx]
        for (n, k), (x, idx) in self.PSLOT.items():
            if idx:
                Pa[n, 0, k, :len(idx)] = FV[x][idx]
        return torch.tensor(Fa, device=device), torch.tensor(Pa, device=device), var
