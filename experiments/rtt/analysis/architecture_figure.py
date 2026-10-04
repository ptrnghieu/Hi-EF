# RoleNet architecture figure (paper Figure 2), drawn at the final text width (6.3 in) in the shared paper style.
# Optional G37_PNG (G37 <id>_paper.png) adds real frame thumbnails; otherwise grey placeholders.
#   G37_PNG=.../sample01305_paper.png OUT=paper/figs python analysis/architecture_figure.py
import os
import numpy as np
from figstyle import (plt, Circle, Rectangle, FILL, ACC, ROLE, box, token, container, arrow, line, INK)

OUT = os.environ.get('OUT', 'figs')
SRC = os.environ.get('G37_PNG')


def panels_from(src):
    from PIL import Image
    img = np.asarray(Image.open(src).convert('RGB'))
    g = img.mean(2) < 235
    col = g[img.shape[0] // 5: img.shape[0] * 7 // 10].mean(0) > 0.5
    e = np.flatnonzero(np.diff(col.astype(int)))
    xs = list(zip(e[0::2] + 1, e[1::2] + 1))
    row = g[:, xs[0][0]:xs[0][1]].mean(1) > 0.5
    y0, y1 = np.flatnonzero(np.diff(row.astype(int)))[:2] + 1
    y1 = y0 + int(0.80 * (y1 - y0))
    return [img[y0:y1, a:b] for a, b in xs]


def enc_box(x, y, w, h, title, sub, fill):
    box(ax, x, y, w, h, '', fill=fill)
    ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=7.2, zorder=4)
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=6.0, style='italic', zorder=4)


def center_crop(p, aspect=4 / 3):
    h, w = p.shape[:2]
    cw = int(h * aspect)
    x0 = (w - cw) // 2
    return p[:, x0:x0 + cw]


fig = plt.figure(figsize=(6.3, 2.62))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 41.6); ax.set_aspect('equal'); ax.axis('off')

# ============ (1) feature extraction
container(ax, 0.4, 5.0, 26.6, 36.2, 'Feature Extraction')
rows = {'face': 34.6, 'scene': 26.6, 'text': 18.4, 'audio': 10.4}
# frame strip (clips I-III)
P = panels_from(SRC) if SRC and os.path.exists(SRC) else None
for k in range(3):
    x = 1.4 + 3.6 * k
    if P is not None:
        ax.imshow(center_crop(P[k]), extent=(x, x + 3.5, 32.6, 36.6), zorder=3, aspect='auto')
    else:
        ax.add_patch(Rectangle((x, 32.6), 3.5, 4.0, fc='#cccccc', ec='white', zorder=3))
    ax.text(x + 1.75, 37.0, ['I', 'II', 'III'][k], ha='center', va='bottom', fontsize=7)
enc_box(13.2, 32.2, 12.4, 4.8, 'Face Encoder', 'ArcFace, HSEmotion', FILL['blue'])
arrow(ax, 12.2, rows['face'], 13.2, rows['face'])
enc_box(13.2, 24.2, 12.4, 4.8, 'Scene Encoder', 'CLIP image', FILL['green'])
line(ax, [6.6, 6.6], [32.6, rows['scene']]); arrow(ax, 6.6, rows['scene'], 13.2, rows['scene'])
# transcript
box(ax, 1.4, 15.7, 10.4, 5.4, '“Well… I’ll\nhold my\ntongue.”', fill='#f2f2f2', ec='#7f7f7f',
    fs=6.4)
enc_box(13.2, 16.0, 12.4, 4.8, 'Text Encoder', 'CLIP text', FILL['yellow'])
arrow(ax, 11.8, rows['text'], 13.2, rows['text'])
# waveform
rng = np.random.default_rng(3)
xw = np.linspace(2.0, 11.0, 22)
amp = np.abs(np.sin(np.linspace(0, 6, 22))) * 1.1 + rng.uniform(0.15, 0.6, 22)
for xv, a in zip(xw, amp):
    ax.add_patch(Rectangle((xv - 0.14, rows['audio'] - a), 0.28, 2 * a, fc=INK, ec='none', zorder=3))
enc_box(13.2, 8.0, 12.4, 4.8, 'Audio Encoder', 'AudioCLIP, ECAPA', FILL['purple'])
arrow(ax, 11.8, rows['audio'], 13.2, rows['audio'])

# ============ (2) role-based evidence set
container(ax, 28.2, 5.0, 29.6, 36.2, 'Role-Based Evidence Set')
# identity clusters -> roles
box(ax, 29.5, 29.4, 4.6, 10.4, '', fill='none', ec=INK, ls=(0, (2, 1.5)), lw=0.7, r=0.8)
ycl = {'A': 37.6, 'L': 34.6, 'O': 31.6}
for r_, yc in ycl.items():
    fn, c = ROLE[r_]
    ax.add_patch(Circle((31.8, yc), 1.25, fc=FILL[fn], ec=c, lw=0.9, zorder=3))
    ax.text(31.8, yc, r_, ha='center', va='center', fontsize=7.5, color=c, weight='bold', zorder=4)
arrow(ax, 25.6, rows['face'], 29.5, rows['face'])
for r_, yc in ycl.items():
    fn, c = ROLE[r_]
    ax.plot([33.05, 34.6, 34.6, 36.0], [yc, yc, yc, yc], color=INK, lw=0.6, zorder=3)
    box(ax, 36.0, yc - 1.2, 11.2, 2.4, 'Attention Pool', fill=FILL[fn], fs=6.8)
    arrow(ax, 47.2, yc, 49.6, yc)
    token(ax, 49.6, yc - 1.2, 4.4, 2.4, rf'$x_{{{r_},k}}$', FILL[fn], fs=7.5)
# scene
box(ax, 36.0, rows['scene'] - 1.2, 11.2, 2.4, 'Scene Projector', fill=FILL['green'], fs=6.8)
arrow(ax, 25.6, rows['scene'], 36.0, rows['scene'])
arrow(ax, 47.2, rows['scene'], 49.6, rows['scene'])
token(ax, 49.6, rows['scene'] - 1.2, 4.4, 2.4, r'$g_k$', FILL['green'], fs=7.5)
# speech: text + audio (+ turn descriptor)
ysp = (rows['text'] + rows['audio']) / 2
box(ax, 36.0, ysp - 1.2, 11.2, 2.4, 'Speech Projector', fill=FILL['yellow'], fs=6.8)
line(ax, [25.6, 31.0], [rows['text'], rows['text']]); line(ax, [25.6, 31.0], [rows['audio'], rows['audio']])
line(ax, [31.0, 31.0], [rows['audio'], rows['text']]); arrow(ax, 31.0, ysp, 36.0, ysp)
arrow(ax, 47.2, ysp, 49.6, ysp)
token(ax, 49.6, ysp - 1.2, 4.4, 2.4, r'$s_k$', FILL['yellow'], fs=7.5)
ax.text(42.9, 7.4, r'for each clip $k \in \{$I, II, III$\}$: 15 tokens', ha='center', va='center', fontsize=6.8)
# token bracket
box(ax, 48.9, 12.0, 5.8, 28.2, '', fill='none', ec=INK, ls=(0, (2, 1.5)), lw=0.7, r=0.8)

# ============ (3) set encoder (transformer-style block)
container(ax, 59.4, 5.0, 26.0, 36.2, 'Set Encoder')
xs_tok = [61.2 + 4.0 * i for i in range(6)]
labs = [(r'$q$', 'lavender'), (r'$x_{A}$', 'orange'), (r'$x_{L}$', 'blue'), (r'$x_{O}$', 'grey'),
        (r'$s$', 'yellow'), (r'$g$', 'green')]
for xt, (lab, fn) in zip(xs_tok, labs):
    token(ax, xt, 7.6, 3.0, 2.4, lab, FILL[fn], fs=7.5)
ax.add_patch(plt.matplotlib.patches.FancyBboxPatch((60.6, 13.4), 23.4, 17.6, boxstyle='round,pad=0,rounding_size=2.2',
                                                    fc='#d9d9d9', ec='#bfbfbf', lw=0.6, zorder=2))
ny = [16.6, 27.6]
cx = [xt + 1.5 for xt in xs_tok]
for i in range(6):                                       # fan connections: tokens -> layer 1 -> layer 2
    for j in range(6):
        ax.plot([cx[i], cx[j]], [10.0, ny[0] - 0.9], color='#b0b0b0', lw=0.35, zorder=2.5)
        ax.plot([cx[i], cx[j]], [ny[0] + 0.9, ny[1] - 0.9], color='#b0b0b0', lw=0.35, zorder=2.5)
for yy in ny:
    for c_ in cx:
        ax.add_patch(Circle((c_, yy), 0.9, fc='#e8e8e8', ec='#a6a6a6', lw=0.5, zorder=3))
ax.text(72.3, 22.1, 'Transformer Encoder\n(2 × SAB, no positions)', ha='center', va='center', fontsize=8,
        zorder=4)
arrow(ax, cx[0], ny[1] + 0.9, cx[0], 34.4)
token(ax, cx[0] - 1.7, 34.4, 3.4, 2.4, r'$h_q$', FILL['lavender'], fs=7.5)
ax.text(cx[0] + 2.2, 32.6, 'read at the query', ha='left', va='center', fontsize=6.6, style='italic')
line(ax, [54.7, 57.0], [26.0, 26.0]); line(ax, [57.0, 57.0], [26.0, 8.8]); arrow(ax, 57.0, 8.8, 61.2, 8.8)

# ============ (4) forecast
container(ax, 86.8, 5.0, 12.8, 36.2, 'Forecast')
line(ax, [cx[0] + 1.7, 85.6], [35.6, 35.6])
box(ax, 88.0, 21.0, 10.4, 4.4, 'Linear +\nSoftmax', fill=FILL['grey'], fs=7.2)
line(ax, [85.6, 93.2], [35.6, 35.6]); arrow(ax, 93.2, 35.6, 93.2, 25.4)
box(ax, 88.0, 10.6, 10.4, 6.0, "B's emotion\nin clip IV", fill='white', ec=INK, ls='-', fs=7.4)
arrow(ax, 93.2, 21.0, 93.2, 16.6)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'architecture.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(os.path.join(OUT, 'architecture.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
