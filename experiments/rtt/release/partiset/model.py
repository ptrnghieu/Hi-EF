"""PartiSet: one attention-pooled token per (slot, turn) face cell, one speech and one scene token per turn, read jointly by a
small pre-norm transformer with a learned query. `PartiSetCore` is the model with switches for the slots, the faces, the
context tokens, the auxiliary heads and the modality dropout (the ablations of the paper); `PartiSet` adds the token-level
switches (readout, embeddings, absent-participant token, dropped token groups). Batches are `data.Batch` tuples."""
import torch
import torch.nn as nn

from .config import RN


class FramePool(nn.Module):
    """Attention pooling of the frames of one cell (Section 3.3)."""

    def __init__(self, fin, d):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Linear(d, 1)

    def forward(self, x, m):                      # x [..., F, fin], m [..., F]
        h = self.proj(x.float())
        a = self.score(h).squeeze(-1).masked_fill(~m, -1e4)
        w = torch.softmax(a, -1) * m.float()
        return (w.unsqueeze(-1) * h).sum(-2), m.any(-1)


class PartiSetCore(nn.Module):
    def __init__(self, cfg, fdim, nvoice, d=RN['D'], rn=RN):
        super().__init__()
        self.cfg, self.R, self.rn = cfg, (3 if cfg['slots'] else 1), rn
        if cfg['faces']:
            self.pool = FramePool(fdim, d)
            self.absent = nn.Parameter(torch.randn(self.R, 3, d) * 0.02)
            self.slot_emb_p = nn.Parameter(torch.randn(self.R, d) * 0.02)
        if cfg['ctx']:
            self.text = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
            self.audio = nn.Sequential(nn.LayerNorm(527), nn.Linear(527, d))
            self.voice = nn.Linear(nvoice, d)
            self.scene = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
            self.stream_emb = nn.Parameter(torch.randn(2, d) * 0.02)
        self.clip_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, rn['heads'], 4 * d, rn['dropout'], batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, rn['layers'], enable_nested_tensor=False)
        mk = lambda: nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))
        self.head = mk()
        self.head_face = mk() if cfg['aux'] and cfg['faces'] else None
        self.head_ctx = mk() if cfg['aux'] and cfg['ctx'] else None
        self.head_A = mk() if cfg['aux'] and cfg['faces'] and cfg['slots'] else None

    def forward(self, b, train=False):
        rn = self.rn
        B, groups, aux = b.yb.shape[0], [], {}
        if self.cfg['faces']:
            x, m = (b.face, b.fmask) if self.R == 3 else (b.pool, b.pmask)
            h, present = self.pool(x, m)                                      # [B, R, 3, d]
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            h = h + self.slot_emb_p[None, :, None] + self.clip_emb[None, None]
            ft = h.reshape(B, self.R * 3, -1)
            groups.append(ft)
            if self.head_face is not None:
                aux['face'] = (self.head_face(ft.mean(1)), b.yb, rn['aux_w'])
            if self.head_A is not None:
                tA = torch.where(present[:, 0, 2], b.ya, torch.full_like(b.ya, -100))
                aux['A'] = (self.head_A(h[:, 0, 2]), tA, rn['a_w'])
        if self.cfg['ctx']:
            spk_ = self.text(b.txt) + self.audio(b.aud) * b.afd.unsqueeze(-1) + self.voice(b.voi) + self.stream_emb[0]
            scn = self.scene(b.scn) + self.stream_emb[1]
            ct = torch.cat([spk_ + self.clip_emb, scn + self.clip_emb], 1)     # [B, 6, d]
            groups.append(ct)
            if self.head_ctx is not None:
                aux['ctx'] = (self.head_ctx(ct.mean(1)), b.yb, rn['aux_w'])
        toks = torch.cat([self.query.expand(B, -1, -1)] + groups, 1)
        valid = torch.ones(toks.shape[:2], dtype=torch.bool, device=toks.device)
        if train and self.cfg['mdrop'] and len(groups) == 2:
            u = torch.rand(B, device=toks.device)
            drop_ctx = u < rn['p_drop_ctx']
            drop_face = (u >= rn['p_drop_ctx']) & (u < rn['p_drop_ctx'] + rn['p_drop_face'])
            nf = groups[0].shape[1]
            valid[:, 1:1 + nf] &= ~drop_face.unsqueeze(1)
            valid[:, 1 + nf:] &= ~drop_ctx.unsqueeze(1)
        out = self.enc(toks, src_key_padding_mask=~valid)
        return self.head(out[:, 0]), aux


class PartiSet(PartiSetCore):
    """Token layout: [queries] + 9 face tokens (slot r, turn k at 3r+k) + 3 speech tokens (turn k) + 3 scene tokens (turn k).
    `cfg` (see config.BASE): readout 'query' | 'mean' | 'L3', nq, q_readonly, slot_emb, clip_emb, absent 'token' | 'mask',
    drop (tuple of 'L', 'speech', 'speechIII', 'scene'), speech_parts, clips."""

    def __init__(self, cfg, fdim, nvoice, d=RN['D'], rn=RN, full=None):
        from .config import FULL
        super().__init__(FULL if full is None else full, fdim, nvoice, d, rn)
        self.t = cfg
        self.nq = cfg['nq'] if cfg['readout'] == 'query' else 0
        if self.nq > 1:
            self.queries = nn.Parameter(torch.randn(1, self.nq, d) * 0.02)
        keep_clip = torch.zeros(3, dtype=torch.bool)
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
        self.register_buffer('keep_face', kf, persistent=False)
        self.register_buffer('keep_ctx', torch.cat([ks, kc]), persistent=False)
        if not bool(self.keep_ctx.any()):
            self.head_ctx = None
        if not bool(kf[2]):                                                  # A's clip-III token gone
            self.head_A = None

    def forward(self, b, train=False):
        B, t, rn = b.yb.shape[0], self.t, self.rn
        dev = b.yb.device
        slot_e = self.slot_emb_p if t['slot_emb'] else torch.zeros_like(self.slot_emb_p)
        clip_e = self.clip_emb if t['clip_emb'] else torch.zeros_like(self.clip_emb)
        aux = {}
        h, present = self.pool(b.face, b.fmask)                               # [B, 3, 3, d], [B, 3, 3]
        if t['absent'] == 'token':
            h = torch.where(present.unsqueeze(-1), h, self.absent.unsqueeze(0).expand(B, -1, -1, -1))
            vf = self.keep_face.unsqueeze(0).expand(B, -1)
        else:
            vf = self.keep_face.unsqueeze(0) & present.reshape(B, 9)
        h = h + slot_e[None, :, None] + clip_e[None, None]
        ft = h.reshape(B, 9, -1)
        parts = {'text': self.text(b.txt), 'audio': self.audio(b.aud) * b.afd.unsqueeze(-1), 'voice': self.voice(b.voi)}
        spk_ = sum(parts[p] for p in t['speech_parts']) + self.stream_emb[0] + clip_e
        scn = self.scene(b.scn) + self.stream_emb[1] + clip_e
        ct = torch.cat([spk_, scn], 1)                                        # [B, 6, d]
        vc = self.keep_ctx.unsqueeze(0).expand(B, -1)
        if self.head_face is not None:
            w = vf.float().unsqueeze(-1)
            aux['face'] = (self.head_face((ft * w).sum(1) / w.sum(1).clamp(min=1)), b.yb, rn['aux_w'])
        if self.head_A is not None:
            tA = torch.where(present[:, 0, 2], b.ya, torch.full_like(b.ya, -100))
            aux['A'] = (self.head_A(h[:, 0, 2]), tA, rn['a_w'])
        if self.head_ctx is not None:
            w = vc.float().unsqueeze(-1)
            aux['ctx'] = (self.head_ctx((ct * w).sum(1) / w.sum(1).clamp(min=1)), b.yb, rn['aux_w'])
        vf, vc = vf.clone(), vc.clone()
        if train and self.cfg['mdrop']:
            u = torch.rand(B, device=dev)
            drop_ctx = u < rn['p_drop_ctx']
            drop_face = (u >= rn['p_drop_ctx']) & (u < rn['p_drop_ctx'] + rn['p_drop_face'])
            vf &= ~drop_face.unsqueeze(1)
            vc &= ~drop_ctx.unsqueeze(1)
        nq = self.nq
        q = [] if nq == 0 else [(self.query if nq == 1 else self.queries).expand(B, -1, -1)]
        toks = torch.cat(q + [ft, ct], 1)
        valid = torch.cat([torch.ones(B, nq, dtype=torch.bool, device=dev), vf, vc], 1)
        # never leave a sample without a valid data token (an all-masked row would produce NaN through attention)
        valid[:, nq] |= ~valid[:, nq:].any(1)
        mask = None
        if nq and t['q_readonly']:
            T_ = toks.shape[1]
            mask = torch.zeros(T_, T_, dtype=torch.bool, device=dev)
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
