# Paper Figure 1 (single column), in the shared paper style: the sample01305 frames rendered by G37 (left) and a
# "previous methods vs. our method" contrast (right). Input: G37's <id>_paper.png (5 panels, role boxes drawn).
#   G37_PNG=.../sample01305_paper.png OUT=paper/figs python analysis/motivation_figure.py
import os
import numpy as np
from PIL import Image
from figstyle import (plt, Circle, Rectangle, FancyBboxPatch, FILL, ROLE, box, token, hollow_arrow, INK,
                      GREY_TXT)

SRC = os.environ['G37_PNG']
OUT = os.environ.get('OUT', 'figs')
FRAME_EC = '#adb9ca'                      # light blue-grey panel outline
GOOD, BAD = '#2e7d32', '#c62828'

# --- cut the five frame panels out of the G37 strip; drop the burnt-in subtitles (bottom 20%)
img = np.asarray(Image.open(SRC).convert('RGB'))
g = img.mean(2) < 235
col = g[img.shape[0] // 5: img.shape[0] * 7 // 10].mean(0) > 0.5
edges = np.flatnonzero(np.diff(col.astype(int)))
xs = list(zip(edges[0::2] + 1, edges[1::2] + 1))
row = g[:, xs[0][0]:xs[0][1]].mean(1) > 0.5
y0, y1 = np.flatnonzero(np.diff(row.astype(int)))[:2] + 1
y1 = y0 + int(0.80 * (y1 - y0))
assert len(xs) == 5, xs
panels = [img[y0:y1, a:b] for a, b in xs]

fig = plt.figure(figsize=(3.15, 2.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 63); ax.set_ylim(0, 51); ax.set_aspect('equal'); ax.axis('off')


def panel(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec=FRAME_EC, lw=1.0, zorder=1))


# ---------------- left: inputs (clips I-III)
panel(0.6, 1.2, 21.4, 46.6)
ax.text(11.3, 48.6, 'Clips I–III', ha='center', va='bottom', fontsize=7.0, color=GREY_TXT, weight='bold',
        family='DejaVu Sans')
items = [(0, 'clip I: context', None), (2, 'clip III: speaker A', 'A'), (3, 'clip III: listener L', 'L')]
for i, (k, lab, role) in enumerate(items):
    yt = 37.0 - 14.2 * i
    ax.imshow(panels[k], extent=(2.0, 20.6, yt, yt + 7.4), zorder=3, aspect='auto')
    ax.text(11.3, yt - 1.0, lab, ha='center', va='top', fontsize=6.6, zorder=3)
ax.text(11.3, 3.3, 'clip IV (not an input):\nB = L answers', ha='center', va='center', fontsize=6.0,
        style='italic', color=GREY_TXT)

# ---------------- right: previous methods / our method
emos = ['happy', 'neutral', 'sad']
for top, title, pred, ok in ((48.0, 'previous methods', 'neutral', False), (22.6, 'our method', 'happy', True)):
    ax.text(62.6, top + 0.5, 'Forecast for B', ha='right', va='bottom', fontsize=6.6, weight='bold',
            color=GREY_TXT)
    panel(28.0, top - 21.0, 34.6, 21.0)
    ax.text(29.0, top - 1.0, title, ha='left', va='top', fontsize=7.0)
    for j, e in enumerate(emos):
        yy = top - 5.5 - 5.2 * j
        is_p = e == pred
        c = (GOOD if ok else BAD) if is_p else '#9a9a9a'
        ax.text(57.4, yy, e, ha='center', va='center', fontsize=7.0 if is_p else 6.6, color=c,
                weight='bold' if is_p else 'normal')
        if is_p:                                          # check marks need a font that has the glyphs
            ax.text(61.4, yy, '✓' if ok else '✗', ha='center', va='center', fontsize=7.0, color=c,
                    family='DejaVu Sans', weight='bold')
    # block arrow from the inputs into each box, and from the representation to the forecast
    hollow_arrow(ax, 22.4, top - 10.5, 5.2, 0, width=2.2, head_width=4.0, head_length=1.8)
    hollow_arrow(ax, 44.2, top - 10.8, 6.6, 0, width=2.6, head_width=4.6, head_length=2.0)

# previous methods: one summary per clip, dominated by the speaker
ax.add_patch(plt.matplotlib.patches.Ellipse((36.2, 37.2), 13.6, 11.2, fc='none', ec='#6d4c41', lw=0.9, zorder=3))
for j, k in enumerate(['I', 'II', 'III']):
    x = 30.7 + 3.9 * j
    ax.add_patch(Rectangle((x, 35.4), 3.2, 3.6, fc=FILL['orange'], ec=INK, lw=0.6, zorder=4))
    ax.text(x + 1.6, 37.2, k, ha='center', va='center', fontsize=6.0, zorder=5)
ax.text(36.2, 29.4, 'clip summaries\n(speaker dominates)', ha='center', va='top', fontsize=5.8, style='italic')

# our method: one token per person and turn; the listener's circle overlaps the others' (joint attention)
cxy = {'A': (33.6, 13.6), 'L': (38.4, 13.6), 'O': (36.0, 9.4)}
for r_, (cx, cy) in cxy.items():
    fn, c = ROLE[r_]
    ax.add_patch(Circle((cx, cy), 3.7, fc='none', ec='#6d4c41' if r_ != 'L' else c, lw=0.9 if r_ != 'L' else 1.4,
                        zorder=3))
for r_, (dx, dy) in {'A': (-2.2, 1.2), 'L': (2.2, 1.2), 'O': (0, -2.4)}.items():
    fn, c = ROLE[r_]
    bx, by = 36.0 + dx, 12.2 + dy
    ax.add_patch(Circle((bx, by), 1.25, fc=FILL[fn], ec=c, lw=0.8, zorder=4))
    ax.text(bx, by, r_, ha='center', va='center', fontsize=5.8, color=c, weight='bold', zorder=5)
ax.text(36.0, 3.6, 'person × turn tokens', ha='center', va='top', fontsize=5.8, style='italic')
# chips on the arrow: what the encoder reads jointly
for j, (r_, fn) in enumerate([('L', 'blue'), ('A', 'orange'), ('O', 'grey')]):
    ax.add_patch(Circle((45.5 + 1.8 * j, 11.8), 0.7, fc=FILL[fn], ec=ROLE[r_][1], lw=0.6, zorder=5))

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'motivation.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(os.path.join(OUT, 'motivation.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
