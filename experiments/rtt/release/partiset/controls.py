"""Input controls for the analyses of the paper: they change what the model reads, not the model.

`ControlledInputs(base, kind, seed)` wraps `data.Inputs` and edits every batch returned by `index`:

  continuity    slots of clips I and II are permuted independently of clip III (a different non-identity permutation per
                sample and turn): the identities are still separate per turn, but a slot no longer follows one person
                across turns. The per-slot mouth-synchrony entries of the turn descriptor move with their slot.
  global_perm   control for `continuity`: one non-identity permutation per sample, applied to all three turns, so that
                continuity is kept and only the meaning of the slots is changed.
  al_swap_half  slots A and L are exchanged in all turns of a random half of the samples (the table of the ablations)
  listener_off  the listener slot is emptied in every turn (its tokens become the learned absent token and its synchrony
                entries are zeroed); the number of tokens is unchanged.
  listener_other the listener slot is replaced by the other-participants slot (faces of other people of the same sample).
  listener_swap the listener slot is replaced by the listener slot of another sample (a derangement of the table). This
                is an out-of-distribution placebo and must be trained with the control, not applied to a trained model.

Permutations are drawn once from `seed`, so training and evaluation see the same control."""
import itertools

import numpy as np
import torch

from .data import Batch

KINDS = ('none', 'continuity', 'global_perm', 'al_swap_half', 'listener_off', 'listener_other', 'listener_swap')
N_SYNC = 6          # VOI[..., :6] = (synchrony, mouth variability) for each of the three slots


def _non_identity_perms(n, seed):
    rng = np.random.default_rng(seed)
    cand = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    return np.array([cand[i] for i in rng.integers(0, len(cand), n)])


def _derangement(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    for i in np.where(p == np.arange(n))[0]:                 # break fixed points by swapping with a neighbour
        j = (i + 1) % n
        p[i], p[j] = p[j], p[i]
    assert not (p == np.arange(n)).any()
    return p


def permute_slots(face, fmask, voi, perm):
    """perm [B, turns, 3]: new slot r of turn k is old slot perm[b, k, r]."""
    B = face.shape[0]
    idx_m = perm.permute(0, 2, 1)                                             # [B, slot, turn]
    face_new = torch.gather(face, 1, idx_m[..., None, None].expand(-1, -1, -1, face.shape[3], face.shape[4]))
    mask_new = torch.gather(fmask, 1, idx_m[..., None].expand(-1, -1, -1, fmask.shape[3]))
    sync = voi[..., :N_SYNC].reshape(B, voi.shape[1], 3, 2)
    sync_new = torch.gather(sync, 2, perm[..., None].expand(-1, -1, -1, 2)).reshape(B, voi.shape[1], N_SYNC)
    return face_new, mask_new, torch.cat([sync_new, voi[..., N_SYNC:]], -1)


class ControlledInputs:
    def __init__(self, base, kind='none', seed=0):
        assert kind in KINDS, kind
        self.base, self.kind, self.N, self.device = base, kind, base.N, base.device
        self.yb = base.yb
        n = base.N
        if kind == 'continuity':
            P = np.tile(np.arange(3), (n, 3, 1))                              # [n, turn, slot]
            for k, s in ((0, seed), (1, seed + 1)):
                P[:, k] = _non_identity_perms(n, s)
            self.perm = torch.tensor(P, device=base.device)
        elif kind == 'global_perm':
            P = _non_identity_perms(n, seed)
            self.perm = torch.tensor(np.repeat(P[:, None], 3, axis=1), device=base.device)
        elif kind == 'al_swap_half':
            half = np.random.default_rng(seed).random(n) < 0.5
            P = np.tile(np.arange(3), (n, 3, 1)); P[half] = (1, 0, 2)
            self.perm = torch.tensor(P, device=base.device); self.half = torch.tensor(half, device=base.device)
        elif kind == 'listener_swap':
            self.swap = torch.tensor(_derangement(n, seed), device=base.device)

    def set_faces(self, face, pool):
        self.base.set_faces(face, pool)

    def index(self, ix):
        b = self.base.index(ix)
        if self.kind == 'none':
            return b
        face, fmask, voi = b.face, b.fmask, b.voi
        if self.kind in ('continuity', 'global_perm', 'al_swap_half'):
            face, fmask, voi = permute_slots(face, fmask, voi, self.perm[ix])
        elif self.kind == 'listener_off':
            fmask = fmask.clone(); fmask[:, 1] = False
            voi = voi.clone(); voi[..., 2:4] = 0
        elif self.kind == 'listener_other':
            face = face.clone(); fmask = fmask.clone(); voi = voi.clone()
            face[:, 1], fmask[:, 1], voi[..., 2:4] = face[:, 2], fmask[:, 2], voi[..., 4:6]
        elif self.kind == 'listener_swap':
            o = self.base.index(self.swap[ix])
            face = face.clone(); fmask = fmask.clone(); voi = voi.clone()
            face[:, 1], fmask[:, 1], voi[..., 2:4] = o.face[:, 1], o.fmask[:, 1], o.voi[..., 2:4]
        return b._replace(face=face, fmask=fmask, voi=voi)
