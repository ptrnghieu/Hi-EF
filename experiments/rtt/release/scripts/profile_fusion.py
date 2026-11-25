"""Cost of the fusion stage of PartiSet on synthetic inputs of the real shapes (no data needed): parameters, forward and
forward+backward time per sample and peak memory. Pre-processing (detection, embeddings, grouping) is not part of this script;
time it separately on the same hardware and report it next to these numbers."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from partiset import BASE, FULL, Batch, PartiSet, PartiSetCore

ap = argparse.ArgumentParser()
ap.add_argument('--batch', type=int, default=64); ap.add_argument('--fdim', type=int, default=146)
ap.add_argument('--maxf', type=int, default=24); ap.add_argument('--repeats', type=int, default=30)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, F = a.batch, a.fdim
b = Batch(torch.randn(B, 3, 3, a.maxf, F, device=dev), torch.randn(B, 1, 3, 32, F, device=dev),
          torch.rand(B, 3, 3, a.maxf, device=dev) > 0.5, torch.rand(B, 1, 3, 32, device=dev) > 0.5,
          torch.randn(B, 3, 512, device=dev), torch.nn.functional.normalize(torch.randn(B, 3, 527, device=dev), dim=-1),
          torch.ones(B, 3, device=dev), torch.randn(B, 3, 1024, device=dev), torch.randn(B, 3, 9, device=dev),
          torch.randint(0, 7, (B,), device=dev), torch.randint(0, 7, (B,), device=dev))
model = PartiSet(BASE, F, 9).to(dev)


def timed(fn):
    for _ in range(3): fn()
    if dev == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = []
    for _ in range(a.repeats):
        t0 = time.perf_counter(); fn()
        if dev == 'cuda': torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) / B * 1000)
    t.sort()
    return t[len(t) // 2], t[int(0.95 * (len(t) - 1))]


model.eval()
with torch.no_grad():
    fwd = timed(lambda: model(b))
model.train()
def step():
    logits, aux = model(b, train=True)
    (torch.nn.functional.cross_entropy(logits, b.yb) + sum(w * torch.nn.functional.cross_entropy(l, t.clamp(min=0)) for l, t, w in aux.values())).backward()
    model.zero_grad()
trn = timed(step)
print(f'device {dev} | batch {B} | parameters {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
print(f'forward (eval)    median {fwd[0]:.3f} ms/sample, p95 {fwd[1]:.3f}')
print(f'forward+backward  median {trn[0]:.3f} ms/sample, p95 {trn[1]:.3f}')
if dev == 'cuda': print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
