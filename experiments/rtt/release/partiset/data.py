"""Tables, clip features and the batch container shared by the training and evaluation code."""
import glob
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import E2I

POL = ['positive', 'neutral', 'negative']
P2I = {p: i for i, p in enumerate(POL)}


def load_split(dataset_dir, split_csv):
    """Split table with the emotion labels. annotation.csv has no header: 0 clip_id, 1 text, 5 polarity, 7 emotion."""
    annot = glob.glob(os.path.join(dataset_dir, "*", "Hi-EF", "annotation.csv"))[0]
    ann = pd.read_csv(annot, header=None, dtype=str).set_index(0)
    sp = pd.read_csv(split_csv, dtype=str)
    sp['yA'] = sp['clip3_emotion'].map(E2I)
    sp['yB'] = sp['clip4_emotion'].map(E2I)
    sp['pA'] = sp['clip3'].map(lambda c: P2I.get(ann.at[c, 5], -1))
    assert sp[['yA', 'yB']].notna().all().all(), 'missing A/B emotion labels'
    return ann, sp


def load_clip_features(features_dir, clips, device):
    """One .pt file per clip -> stacked tensors on `device`: face / whole-frame CLIP vectors, text, AudioCLIP, masks."""
    bufs = {k: [] for k in ('face', 'fmask', 'ori', 'text', 'audio', 'afound')}
    for c in clips:
        d = torch.load(os.path.join(features_dir, c.replace('/', '_') + '.pt'), map_location='cpu', weights_only=False)
        face = d['face_features'].float()
        fm = d.get('face_valid_mask')
        bufs['face'].append(face)
        bufs['fmask'].append(torch.ones(face.shape[0], dtype=torch.bool) if fm is None
                             else torch.as_tensor(fm).bool().reshape(-1))
        bufs['ori'].append(d['ori_features'].float())
        bufs['text'].append(d['text_feature'].float().reshape(-1))
        bufs['audio'].append(d.get('audio_feature', torch.zeros(527)).float().reshape(-1))
        bufs['afound'].append(torch.tensor(bool(d.get('audio_found', True))))
    return {k: torch.stack(v).to(device) for k, v in bufs.items()}


Batch = namedtuple('Batch', 'face pool fmask pmask txt aud afd scn voi yb ya')


class Inputs:
    """Everything the model reads, for the N samples of one table. `index(ix)` returns the batch of the rows `ix`.
    The per-fold face tensors (PCA fitted on the training clips of the fold) are set with `set_faces`."""

    def __init__(self, feat, clip_idx, sp, fmask, pmask, voi, device):
        T = lambda a, dt=None: torch.tensor(a, device=device) if dt is None else torch.tensor(a, dtype=dt, device=device)
        self.device, self.N = device, len(sp)
        self.fmask, self.pmask, self.voi = T(fmask), T(pmask), T(voi)
        self.txt = feat['text'][clip_idx]
        self.aud = F.normalize(feat['audio'][clip_idx], dim=-1)
        self.afd = feat['afound'][clip_idx].float()
        fm = feat['fmask'][clip_idx].unsqueeze(-1).float()
        self.scn = torch.cat([feat['ori'][clip_idx].mean(2),
                              (feat['face'][clip_idx] * fm).sum(2) / fm.sum(2).clamp(min=1)], -1)
        self.yb, self.ya = T(sp.yB.values), T(sp.yA.values)
        self.face = self.pool = None

    def set_faces(self, face, pool):
        self.face, self.pool = face, pool

    def index(self, ix):
        return Batch(self.face[ix] if self.face is not None else None, self.pool[ix] if self.pool is not None else None,
                     self.fmask[ix], self.pmask[ix], self.txt[ix], self.aud[ix], self.afd[ix], self.scn[ix], self.voi[ix],
                     self.yb[ix], self.ya[ix])
