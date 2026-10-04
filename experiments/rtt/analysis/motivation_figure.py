# Paper Figure 1 (single column): the sample01305 frames rendered by G37, plus a speaker-centric vs listener-centric
# contrast. Input: G37's <id>_paper.png (5 panels, role boxes already drawn). Usage:
#   G37_PNG=.../sample01305_paper.png OUT=paper/figs python analysis/motivation_figure.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

SRC = os.environ['G37_PNG']
OUT = os.environ.get('OUT', 'figs')
INK, INK2, LINE = '#1a1a1a', '#5b5a56', '#b9b8b3'
A, L, O, Q, BAD, GOOD = '#eb6834', '#2a78d6', '#8a8984', '#4a3aa7', '#c23b3b', '#1f8a4c'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})

# --- cut the five frame panels out of the G37 strip; drop the burnt-in subtitles (bottom 20%)
img = np.asarray(Image.open(SRC).convert('RGB'))
g = (img.mean(2) < 235)
col = g[img.shape[0] // 5: img.shape[0] * 7 // 10].mean(0) > 0.5
edges = np.flatnonzero(np.diff(col.astype(int)))
xs = list(zip(edges[0::2] + 1, edges[1::2] + 1))
row = g[:, xs[0][0]:xs[0][1]].mean(1) > 0.5
y0, y1 = np.flatnonzero(np.diff(row.astype(int)))[:2] + 1
y1 = y0 + int(0.80 * (y1 - y0))
assert len(xs) == 5, xs
panels = [img[y0:y1, a:b] for a, b in xs]

fig = plt.figure(figsize=(3.15, 3.05))
# 2 x 2 frames: clip I (context), clip III speaker, clip III listener, clip IV target
show = [(0, 'Clip I (context)', None), (2, 'Clip III: speaker A', A), (3, 'Clip III: listener L', L),
        (4, 'Clip IV: responder B (unseen)', INK)]
W, H, gap = 0.47, 0.195, 0.02
for i, (k, title, c) in enumerate(show):
    x = 0.02 + (i % 2) * (W + gap)
    y = 0.775 - (i // 2) * (H + 0.05)
    ax = fig.add_axes([x, y, W, H]); ax.imshow(panels[k]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(c or LINE); s.set_linewidth(1.4 if c else 0.6)
        if k == 4:
            s.set_linestyle((0, (3, 2)))
    ax.set_title(title, fontsize=6.6, color=c or INK, pad=2, weight='bold' if c in (A, L) else 'normal')
fig.text(0.5, 0.468, 'Who will answer is already on screen, listening.', ha='center', fontsize=6.6, style='italic',
         color=INK2)

# --- two pipelines
ax = fig.add_axes([0, 0, 1, 0.44]); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')


def box(x, y, w, h, text, ec, fc, fs=6.4, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', ec=ec, fc=fc, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
            weight='bold' if bold else 'normal', linespacing=1.15)


def arrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=7, color=INK2, lw=0.8))


# previous: speaker-centric clip summaries
ax.text(1.5, 40.5, 'Speaker-centric (previous work)', fontsize=6.8, weight='bold', color=INK)
box(2, 25, 30, 11, 'one summary\nper clip', LINE, '#f3f2ef')
arrow(33.5, 38.5, 30.5)
box(39, 25, 27, 11, 'dominated by\nspeaker A', LINE, '#f3f2ef')
arrow(67.5, 72.5, 30.5)
box(73, 25, 25, 11, 'neutral  ✗', BAD, '#fbe4e4', fs=7, bold=True, color=BAD)
# ours: participant-centric evidence set
ax.text(1.5, 17.5, 'Listener-centric (RoleNet)', fontsize=6.8, weight='bold', color=Q)
for j, (n, c, f) in enumerate([('A', A, '#fbe1d6'), ('L', L, '#d6e6fa'), ('O', O, '#e7e6e3')]):
    box(2 + j * 10.3, 2.5, 8.2, 11, n, c, f, fs=7, bold=True, color=c)
ax.text(17.5, -0.6, 'person × turn tokens', ha='center', va='top', fontsize=5.8, color=INK2)
arrow(33.5, 38.5, 8)
box(39, 2.5, 27, 11, 'set encoder\n(listener + context)', Q, '#e6e3f6')
arrow(67.5, 72.5, 8)
box(73, 2.5, 25, 11, 'happy  ✓', GOOD, '#ddf1e4', fs=7, bold=True, color=GOOD)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'motivation.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(os.path.join(OUT, 'motivation.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
