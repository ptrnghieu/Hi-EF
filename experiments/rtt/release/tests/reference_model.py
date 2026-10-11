"""Reference implementation used only by the tests: the model, the fold assignment and the bootstrap exactly as they were first
written (module-level tensors instead of a Batch; copied verbatim, with only the names changed to the ones used in the package).
Run tests/test_equivalence.py to check that partiset/ gives identical outputs."""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from partiset.config import RN, FULL           # hyper-parameters (same values as the first implementation)
FDIM = NVOICE = DEVICE = None

# the reference code reads these module-level names; tests/test_equivalence.py sets them (FACE, FMASK, POOL, PMASK, TXT, AUD,
# AFD, SCN, VOI, YB, YA, FDIM, NVOICE, DEVICE, RN, FULL)


class FramePool(nn.Module):
    def __init__(self, fin, d):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Linear(d, 1)

    def forward(self, x, m):                      # x [..., F, fin], m [..., F]
        h = self.proj(x.float())
        a = self.score(h).squeeze(-1).masked_fill(~m, -1e4)
        w = torch.softmax(a, -1) * m.float()
        return (w.unsqueeze(-1) * h).sum(-2), m.any(-1)


class ReferenceCore(nn.Module):
    def __init__(self, cfg, d=RN['D']):
        super().__init__()
        self.cfg, self.R = cfg, (3 if cfg['slots'] else 1)
        if cfg['faces']:
            self.pool = FramePool(FDIM, d)
            self.absent = nn.Parameter(torch.randn(self.R, 3, d) * 0.02)
            self.slot_emb_p = nn.Parameter(torch.randn(self.R, d) * 0.02)
        if cfg['ctx']:
            self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
            self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
            self.voice = nn.Linear(NVOICE, d)
            self.scene = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
            self.stream_emb = nn.Parameter(torch.randn(2, d) * 0.02)
        self.clip_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, RN['heads'], 4 * d, RN['dropout'], batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, RN['layers'], enable_nested_tensor=False)
        mk = lambda: nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))
        self.head = mk()
        self.head_face = mk() if cfg['aux'] and cfg['faces'] else None
        self.head_ctx = mk() if cfg['aux'] and cfg['ctx'] else None
        self.head_A = mk() if cfg['aux'] and cfg['faces'] and cfg['slots'] else None

    def forward(self, ix, train=False):
        B, groups, aux = len(ix), [], {}
        if self.cfg['faces']:
            x, m = (FACE[ix], FMASK[ix]) if self.R == 3 else (POOL[ix], PMASK[ix])
            h, present = self.pool(x, m)                                      # [B, R, 3, d]
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            h = h + self.slot_emb_p[None, :, None] + self.clip_emb[None, None]
            ft = h.reshape(B, self.R * 3, -1)
            groups.append(ft)
            if self.head_face is not None:
                aux['face'] = (self.head_face(ft.mean(1)), YB[ix], RN['aux_w'])
            if self.head_A is not None:
                tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
                aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        if self.cfg['ctx']:
            spk_ = self.text(TXT[ix]) + self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1) + self.voice(VOI[ix]) + self.stream_emb[0]
            scn = self.scene(SCN[ix]) + self.stream_emb[1]
            ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)     # [B, 6, d]
            groups.append(ct)
            if self.head_ctx is not None:
                aux['ctx'] = (self.head_ctx(ct.mean(1)), YB[ix], RN['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1)] + groups, 1)
        valid = torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)
        if train and self.cfg['mdrop'] and len(groups) == 2:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            nf = groups[0].shape[1]
            valid[:, 1:1 + nf] &= ~drop_face.unsqueeze(1)
            valid[:, 1 + nf:] &= ~drop_ctx.unsqueeze(1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


class B1Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = Forecaster(use_raw=True, use_traj=False)

    def forward(self, ix, train=False):
        return self.f(CLIPIDX[ix], ZREC[ix]), {}


class B1FaceWrap(nn.Module):
    # (reference code, see header)
    def __init__(self, n_face=60, d=512):
        super().__init__()
        self.f = Forecaster(use_raw=True, use_traj=False, d=d)
        self.face = nn.Sequential(nn.Linear(n_face, d // 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d // 2, d))
        nn.init.zeros_(self.face[-1].weight); nn.init.zeros_(self.face[-1].bias)

    def forward(self, ix, train=False):
        b, clip_idx = self.f, CLIPIDX[ix]
        B, n = clip_idx.shape
        feats = gather(clip_idx)
        tok = b.enc({k: v.reshape(B * n, *v.shape[2:]) for k, v in feats.items()}).reshape(B, n, -1)
        h = b.inter(tok + b.clip_pos[:, 3 - n:])
        return b.head(h.mean(1) + self.face(LRFZ[ix])), {}


class ReferenceTokenModel(ReferenceCore):
    # ReferenceCore with switches for the readout, the embeddings, the absent token, and which speech/scene/clip tokens exist.
    # Token layout: [queries] + 9 face tokens (role r, clip k at 3r+k) + 3 speech tokens (clip k) + 3 scene tokens (clip k).
    def __init__(self, cfg, d=RN['D']):
        super().__init__(FULL, d)
        self.t = cfg
        self.nq = cfg['nq'] if cfg['readout'] == 'query' else 0
        if self.nq > 1:
            self.queries = nn.Parameter(torch.randn(1, self.nq, d) * 0.02)
        keep_clip = torch.zeros(3, dtype=torch.bool, device=DEVICE)
        keep_clip[list(cfg['clips'])] = True
        kf = keep_clip.repeat(3)                                              # face token 3r+k -> clip k
        if 'L' in cfg['drop']:
            kf[3:6] = False                                                   # listener tokens, all clips
        ks, kc = keep_clip.clone(), keep_clip.clone()
        if 'speech' in cfg['drop']:
            ks[:] = False
        if 'speechIII' in cfg['drop']:
            ks[2] = False
        if 'scene' in cfg['drop']:
            kc[:] = False
        self.keep_face, self.keep_ctx = kf, torch.cat([ks, kc])
        if not bool(self.keep_ctx.any()):
            self.head_ctx = None
        if not bool(kf[2]):                                                  # A's clip-III token gone
            self.head_A = None

    def forward(self, ix, train=False):
        B, t = len(ix), self.t
        d = self.query.shape[-1]
        zero = torch.zeros(1, device=DEVICE)
        slot_e = self.slot_emb_p if t['slot_emb'] else torch.zeros_like(self.slot_emb_p)
        clip_e = self.clip_emb if t['clip_emb'] else torch.zeros_like(self.clip_emb)
        aux = {}
        h, present = self.pool(FACE[ix], FMASK[ix])                           # [B, 3, 3, d], [B, 3, 3]
        if t['absent'] == 'token':
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            vf = self.keep_face.unsqueeze(0).expand(B, -1)
        else:
            vf = self.keep_face.unsqueeze(0) & present.reshape(B, 9)
        h = h + slot_e[None, :, None] + clip_e[None, None]
        ft = h.reshape(B, 9, -1)
        parts = {'text': self.text(TXT[ix]), 'audio': self.audio(AUD[ix]) * AFD[ix].unsqueeze(-1), 'voice': self.voice(VOI[ix])}
        spk_ = sum(parts[p] for p in t['speech_parts']) + self.stream_emb[0] + clip_e
        scn = self.scene(SCN[ix]) + self.stream_emb[1] + clip_e
        ct = torch.cat([spk_, scn], 1)                                        # [B, 6, d]
        vc = self.keep_ctx.unsqueeze(0).expand(B, -1)
        if self.head_face is not None:
            w = vf.float().unsqueeze(-1)
            aux['face'] = (self.head_face((ft * w).sum(1) / w.sum(1).clamp(min=1)), YB[ix], RN['aux_w'])
        if self.head_A is not None:
            tA = torch.where(present[:, 0, 2], YA[ix], torch.full_like(YA[ix], -100))
            aux['A'] = (self.head_A(h[:, 0, 2]), tA, RN['a_w'])
        if self.head_ctx is not None:
            w = vc.float().unsqueeze(-1)
            aux['ctx'] = (self.head_ctx((ct * w).sum(1) / w.sum(1).clamp(min=1)), YB[ix], RN['aux_w'])
        vf, vc = vf.clone(), vc.clone()
        if train and self.cfg['mdrop']:
            u = torch.rand(B, device=DEVICE)
            drop_ctx = u < RN['p_drop_ctx']
            drop_face = (u >= RN['p_drop_ctx']) & (u < RN['p_drop_ctx'] + RN['p_drop_face'])
            vf &= ~drop_face.unsqueeze(1)
            vc &= ~drop_ctx.unsqueeze(1)
        nq = self.nq
        q = [] if nq == 0 else [(self.query if nq == 1 else self.queries).expand(B, -1, -1)]
        toks = torch.cat(q + [ft, ct], 1)
        valid = torch.cat([torch.ones(B, nq, dtype=torch.bool, device=DEVICE), vf, vc], 1)
        # never leave a sample without a valid data token (an all-masked row would produce NaN through attention)
        valid[:, nq] |= ~valid[:, nq:].any(1)
        mask = None
        if nq and t['q_readonly']:
            T_ = toks.shape[1]
            mask = torch.zeros(T_, T_, dtype=torch.bool, device=DEVICE)
            mask[nq:, :nq] = True                                            # data tokens cannot attend to the query
        out = self.enc(toks, mask=mask, src_key_padding_mask=~valid)
        if t['readout'] == 'query':
            z = out[:, :nq].mean(1)
        elif t['readout'] == 'mean':
            w = valid.float().unsqueeze(-1)
            z = (out * w).sum(1) / w.sum(1).clamp(min=1)
        else:                                                                 # 'L3': listener face token of clip III
            z = out[:, nq + 3 * 1 + 2]
        return self.head(z), aux
