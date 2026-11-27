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
        self.pool = torch.randn(N, 1, 3, 32, FD, generator=g)
        self.pmask = torch.rand(N, 1, 3, 32, generator=g) > 0.4
        self.yb = torch.randint(0, 7, (N,), generator=g)

    def set_faces(self, face, pool):
        pass

    def index(self, ix):
        z = lambda *s: torch.zeros(len(ix), *s)
        return Batch(self.face[ix], self.pool[ix], self.fmask[ix], self.pmask[ix], z(3, 4), z(3, 4), z(3), z(3, 4), self.voi[ix], self.yb[ix], self.yb[ix])


def batch_of(kind, seed=0):
    X = ControlledInputs(FakeInputs(), kind, seed)
    return X, X.index(torch.arange(N)), X.base.index(torch.arange(N))


def test_none_is_identity_and_all_kinds_keep_shapes():
    for kind in KINDS:
        _, b, o = batch_of(kind)
        for f in ('face', 'fmask', 'voi'):
            if kind in ('replicated_pool', 'random_grouping') and f != 'voi':           # these use the 32 pooled frames per turn
                assert getattr(b, f).shape[:3] == getattr(o, f).shape[:3], (kind, f)
                continue
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


def test_al_swap_half_swaps_a_and_l_in_a_random_half_of_the_samples():
    X, b, o = batch_of('al_swap_half', seed=5)
    half = X.half
    assert 0.2 < half.float().mean() < 0.8
    assert torch.equal(b.face[~half], o.face[~half])                                  # the other half is untouched
    assert torch.equal(b.face[half][:, 0], o.face[half][:, 1]) and torch.equal(b.face[half][:, 1], o.face[half][:, 0])
    assert torch.equal(b.face[half][:, 2], o.face[half][:, 2])                        # O is never moved
    assert torch.equal(b.fmask[half][:, 0], o.fmask[half][:, 1])


def test_replicated_pooling_gives_every_slot_the_same_pooled_evidence():
    X, b, o = batch_of('replicated_pool')
    assert b.face.shape == (N, 3, 3, 32, FD) and b.fmask.shape == (N, 3, 3, 32)
    for r in range(3):
        assert torch.equal(b.face[:, r], o.pool[:, 0]) and torch.equal(b.fmask[:, r], o.pmask[:, 0])
    assert torch.equal(b.voi[..., 0:2], b.voi[..., 2:4]) and torch.equal(b.voi[..., 2:4], b.voi[..., 4:6])
    assert torch.allclose(b.voi[..., 0:2], o.voi[..., :6].reshape(N, 3, 3, 2).mean(2))
    assert torch.equal(b.voi[..., 6:], o.voi[..., 6:])


def test_random_grouping_partitions_the_pooled_frames_without_identity():
    X, b, o = batch_of('random_grouping', seed=2)
    assert b.face.shape == (N, 3, 3, 32, FD)
    valid = o.pmask[:, 0]                                                      # [N, turn, frames]
    assert torch.equal(b.fmask.sum((1, 3)), valid.sum(-1))                          # every valid frame lands in exactly one slot
    for i in range(N):
        for k in range(3):
            got = sorted(tuple(f.tolist()) for r in range(3) for f in b.face[i, r, k][b.fmask[i, r, k]])
            want = sorted(tuple(f.tolist()) for f in o.pool[i, 0, k][valid[i, k]])
            assert got == want                                                  # same frames, only redistributed
    assert (b.fmask.sum(-1) > 0).any() and not torch.equal(b.fmask.sum(-1)[:, 0], b.fmask.sum(-1)[:, 1])   # slots differ in size
    assert torch.equal(b.voi[..., 0:2], b.voi[..., 4:6])                        # no per-slot synchrony left
    X2, b2, _ = batch_of('random_grouping', seed=2)
    assert torch.equal(b.face, b2.face)                                         # deterministic
