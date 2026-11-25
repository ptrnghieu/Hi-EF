"""The input controls change only what they claim to change (synthetic data, CPU)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from partiset import Batch
from partiset.controls import ControlledInputs, KINDS, permute_slots

N, FD, MAXF = 30, 8, 5


class FakeInputs:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.N, self.device = N, 'cpu'
        self.face = torch.randn(N, 3, 3, MAXF, FD, generator=g)
        self.fmask = torch.rand(N, 3, 3, MAXF, generator=g) > 0.4
        self.voi = torch.randn(N, 3, 9, generator=g)
        self.yb = torch.randint(0, 7, (N,), generator=g)

    def set_faces(self, face, pool):
        pass

    def index(self, ix):
        z = lambda *s: torch.zeros(len(ix), *s)
        return Batch(self.face[ix], z(1), self.fmask[ix], z(1), z(3, 4), z(3, 4), z(3), z(3, 4), self.voi[ix], self.yb[ix], self.yb[ix])


def batch_of(kind, seed=0):
    X = ControlledInputs(FakeInputs(), kind, seed)
    return X, X.index(torch.arange(N)), X.base.index(torch.arange(N))


def test_none_is_identity_and_all_kinds_keep_shapes():
    for kind in KINDS:
        _, b, o = batch_of(kind)
        for f in ('face', 'fmask', 'voi'):
            assert getattr(b, f).shape == getattr(o, f).shape, (kind, f)
    _, b, o = batch_of('none')
    assert all(torch.equal(x, y) for x, y in zip(b, o))


def test_continuity_breaks_clips_I_II_only_and_keeps_content():
    _, b, o = batch_of('continuity')
    assert torch.equal(b.face[:, :, 2], o.face[:, :, 2]) and torch.equal(b.fmask[:, :, 2], o.fmask[:, :, 2])   # clip III untouched
    assert torch.equal(b.voi[:, 2], o.voi[:, 2])
    for k in (0, 1):                                                           # same faces, different slots
        assert torch.equal(b.fmask[:, :, k].sum((1, 2)), o.fmask[:, :, k].sum((1, 2)))
        assert not torch.equal(b.face[:, :, k], o.face[:, :, k])
        assert torch.allclose(b.face[:, :, k].sum(1), o.face[:, :, k].sum(1))
    assert torch.equal(b.voi[..., 6:], o.voi[..., 6:])                          # shared turn descriptor unchanged


def test_global_permutation_is_consistent_across_turns():
    X, b, o = batch_of('global_perm')
    perm = X.perm
    assert (perm[:, 0] == perm[:, 1]).all() and (perm[:, 1] == perm[:, 2]).all()
    assert (perm != torch.arange(3)).any(-1).all()                              # never the identity
    # the slot of every person moves the same way in all turns: slot r of every turn now holds old slot perm[r]
    for i in range(N):
        for k in range(3):
            assert torch.equal(b.face[i, :, k], o.face[i, perm[i, 0], k])


def test_sync_features_move_with_their_slot():
    face, fm, voi = torch.zeros(1, 3, 3, 2, 1), torch.ones(1, 3, 3, 2, dtype=torch.bool), torch.arange(9.).reshape(1, 1, 9).repeat(1, 3, 1)
    perm = torch.tensor([[[1, 2, 0]] * 3])
    _, _, v = permute_slots(face, fm, voi, perm)
    assert v[0, 0, :6].tolist() == [2., 3., 4., 5., 0., 1.]                    # slots (1, 2, 0)
    assert v[0, 0, 6:].tolist() == [6., 7., 8.]


def test_listener_controls():
    _, b, o = batch_of('listener_off')
    assert not b.fmask[:, 1].any() and torch.equal(b.fmask[:, [0, 2]], o.fmask[:, [0, 2]]) and (b.voi[..., 2:4] == 0).all()
    _, b, o = batch_of('listener_other')
    assert torch.equal(b.face[:, 1], o.face[:, 2]) and torch.equal(b.face[:, 0], o.face[:, 0]) and torch.equal(b.voi[..., 2:4], o.voi[..., 4:6])
    X, b, o = batch_of('listener_swap')
    assert (X.swap != torch.arange(N)).all() and torch.equal(b.face[:, 1], o.face[X.swap][:, 1])
    assert torch.equal(b.face[:, 0], o.face[:, 0]) and torch.equal(b.face[:, 2], o.face[:, 2])


def test_controls_are_deterministic_and_work_on_subsets():
    X1, b1, _ = batch_of('continuity', seed=3); X2, b2, _ = batch_of('continuity', seed=3)
    assert torch.equal(b1.face, b2.face)
    ix = torch.tensor([4, 9, 17])
    assert torch.equal(X1.index(ix).face, b1.face[ix])                         # same result whether indexed alone or in a batch
