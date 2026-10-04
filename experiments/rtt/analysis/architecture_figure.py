# RoleNet architecture figure (paper Figure 3), drawn with matplotlib at the final text width (6.3 in), so font sizes
# are the printed sizes. Optional G37_PNG (G37 <id>_paper.png) adds real input thumbnails; otherwise grey placeholders.
#   G37_PNG=.../sample01305_paper.png OUT=paper/figs python analysis/architecture_figure.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get('OUT', 'figs')
SRC = os.environ.get('G37_PNG')
INK, INK2, LINE = '#1a1a1a', '#5b5a56', '#a9a8a3'
A, L, O, SP, SC, Q = '#eb6834', '#2a78d6', '#8a8984', '#1baf7a', '#d99100', '#4a3aa7'
FILL = {A: '#fbe1d6', L: '#d6e6fa', O: '#e7e6e3', SP: '#d1efe4', SC: '#fbecc8', Q: '#e3dff6'}
STAGE = ['#f6f5f2', '#f6f5f2', '#f6f5f2', '#f6f5f2']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})

fig = plt.figure(figsize=(6.3, 2.25))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 126); ax.set_ylim(0, 45); ax.axis('off')


def rbox(x, y, w, h, ec=LINE, fc='white', lw=0.9, ls='-', r=1.0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.2,rounding_size={r}', ec=ec, fc=fc, lw=lw,
                                ls=ls, zorder=z))


def text(x, y, s, fs=7, c=INK, **kw):
    ax.text(x, y, s, fontsize=fs, color=c, ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), zorder=5, **kw)


def arrow(x1, y1, x2, y2, c=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, color=c, lw=0.9, zorder=4))


def stage(x, w, label):
    rbox(x, 5.5, w, 38.5, ec='#c9c8c3', fc='#faf9f7', lw=0.8, ls=(0, (4, 2)), r=1.6, z=0)
    text(x + w / 2, 2.4, label, fs=7.0, c=INK, weight='bold')


# ---------------- stage (a): inputs
stage(1, 19, '(a) Clips I–III')
panels = None
if SRC and os.path.exists(SRC):
    from PIL import Image
    img = np.asarray(Image.open(SRC).convert('RGB'))
    g = img.mean(2) < 235
    col = g[img.shape[0] // 5: img.shape[0] * 7 // 10].mean(0) > 0.5
    e = np.flatnonzero(np.diff(col.astype(int)))
    xs = list(zip(e[0::2] + 1, e[1::2] + 1))
    row = g[:, xs[0][0]:xs[0][1]].mean(1) > 0.5
    y0, y1 = np.flatnonzero(np.diff(row.astype(int)))[:2] + 1
    y1 = y0 + int(0.80 * (y1 - y0))
    panels = [img[y0:y1, a:b] for a, b in xs[:3]]
for k, name in enumerate(['I', 'II', 'III']):
    y = 33.5 - 10.6 * k
    if panels is not None:
        ax.imshow(panels[k], extent=(3, 15, y, y + 7.5), zorder=3, aspect='auto')
        ax.add_patch(plt.Rectangle((3, y), 12, 7.5, fill=False, ec=LINE, lw=0.6, zorder=4))
    else:
        rbox(3, y, 12, 7.5, fc='#dddcd8')
    text(17.4, y + 3.75, name, fs=7.2, weight='bold')
text(10.5, 9.0, 'frames · audio · text', fs=6.2, c=INK2)

# ---------------- stage (b): role assignment
stage(23, 22, '(b) Role assignment')
rbox(25.5, 34, 17, 7, ec=LINE, fc='white')
text(34, 37.5, 'faces → ArcFace\n→ identities', fs=6.6)
arrow(34, 33.6, 34, 30.6)
for j, (n, c, rule) in enumerate([('A', A, 'most frames\nin clip III'), ('L', L, '2nd most\nin clip III'),
                                  ('O', O, 'all others')]):
    y = 24.2 - 6.6 * j
    rbox(25.5, y, 5, 5.0, ec=c, fc=FILL[c])
    text(28, y + 2.5, n, fs=7.5, c=c, weight='bold')
    text(32.3, y + 2.5, rule, fs=6.0, c=INK2, ha='left')
text(34, 8.4, 'no clip IV, no names', fs=6.2, c=INK2, style='italic')
arrow(20.6, 24, 22.6, 24)

# ---------------- stage (c): evidence set
stage(48, 33, '(c) Evidence set (15 tokens)')
rows = [('A', A), ('L', L), ('O', O), ('speech', SP), ('scene', SC)]
for r, (name, c) in enumerate(rows):
    y = 36 - 5.9 * r
    text(57.6, y + 2.0, name, fs=6.8, c=c, ha='right', weight='bold')
    for k in range(3):
        rbox(58.8 + 7.0 * k, y, 5.6, 4.0, ec=c, fc=FILL[c])
        text(61.6 + 7.0 * k, y + 2.0, ['I', 'II', 'III'][k], fs=6.4)
ax.plot([50, 80], [17.1, 17.1], color='#d6d5d0', lw=0.6, ls=(0, (2, 2)), zorder=2)
text(64.5, 9.6, 'face token = attention pooling over\nthe role’s frames + role/clip embedding', fs=5.8, c=INK2)
arrow(45.6, 24, 47.6, 24)

# ---------------- stage (d): set encoder
stage(84, 25, '(d) Set encoder')
rbox(86.5, 15, 4.6, 18, ec=Q, fc=FILL[Q])
text(88.8, 24, 'q', fs=9, c=Q, weight='bold')
text(88.8, 12.3, 'query', fs=6.0, c=INK2)
rbox(93.5, 12, 13.5, 24, ec=Q, fc='white', lw=1.0)
text(100.25, 33.2, 'SAB × 2', fs=7.2, c=Q, weight='bold')
for j, (s, fc) in enumerate([('LN → MHA', '#efedf9'), ('+', 'white'), ('LN → FFN', '#efedf9'), ('+', 'white')]):
    y = 27.5 - 4.4 * j
    if s == '+':
        ax.add_patch(plt.Circle((100.25, y + 1.6), 1.25, ec=Q, fc='white', lw=0.8, zorder=4))
        text(100.25, y + 1.65, '+', fs=7, c=Q)
    else:
        rbox(95.2, y, 10.1, 3.3, ec='#b9b2e3', fc=fc)
        text(100.25, y + 1.65, s, fs=6.2)
text(96.5, 8.6, 'no positional encoding;\nread at q', fs=6.0, c=INK2)
arrow(81.6, 24, 83.6, 24)

# ---------------- output: forecast distribution
x0 = 112
text(118.5, 41, r'$p(y_B)$, clip IV', fs=7.2, weight='bold')
emos = ['ang', 'dis', 'fea', 'hap', 'neu', 'sad', 'sur']
vals = [0.08, 0.03, 0.01, 0.59, 0.18, 0.08, 0.03]
for i, (e, v) in enumerate(zip(emos, vals)):
    y = 35 - 4.0 * i
    text(x0 + 2.6, y + 1.2, e, fs=6.0, c=INK2, ha='right')
    ax.add_patch(plt.Rectangle((x0 + 3.3, y), 9.5 * v / 0.59, 2.4, fc=Q if e == 'hap' else '#c9c4ea', ec='none',
                               zorder=3))
arrow(107.4, 24, 111.2, 24)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'architecture.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(os.path.join(OUT, 'architecture.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
