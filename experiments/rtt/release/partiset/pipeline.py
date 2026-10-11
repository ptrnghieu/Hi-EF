"""Glue: tables + features + participant grouping -> `Inputs` ready for `train_eval`."""
import numpy as np
import torch

from .config import Constants
from .data import Inputs, load_clip_features, load_split
from .participants import ParticipantBuilder, load_feature_shards


def prepare(dataset_dir, features_dir, split_csv, features_dir_participants, device, rows=None, const=Constants(), progress=None):
    """Returns (sp, builder, inputs). `rows` optionally restricts the table (e.g. sp.split != 'test' for cross-validation)."""
    _, sp = load_split(dataset_dir, split_csv)
    if rows is not None:
        sp = sp[rows(sp)]
    sp = sp.reset_index(drop=True)
    shards = load_feature_shards(features_dir_participants)
    builder = ParticipantBuilder(sp, shards, const, progress)
    clips = sorted(set(sp[['clip1', 'clip2', 'clip3']].values.ravel()))
    cidx = {c: i for i, c in enumerate(clips)}
    feat = load_clip_features(features_dir, clips, device)
    clip_idx = torch.tensor([[cidx[c] for c in r] for r in sp[['clip1', 'clip2', 'clip3']].values], device=device)
    inputs = Inputs(feat, clip_idx, sp, builder.FMASK, builder.PMASK, builder.VOI, device)
    return sp, builder, inputs
