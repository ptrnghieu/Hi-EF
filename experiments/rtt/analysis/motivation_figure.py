# Paper Figure 1 (full text width): frames of one multi-party sample (G37 strip) plus a contrast between clip-level
# fusion and participant-centric evidence. Print- and colour-blind-safe: black outlines on white, hatching instead of
# colour, no blue/red/green/grey; text is 9-10 pt at final size. The role boxes burnt into the G37 frames are removed
# by inpainting and redrawn in matplotlib.
#   G37_PNG=.../sample01305_paper.png OUT=paper/figs python analysis/motivation_figure.py
import os
import numpy as np
import cv2
from PIL import Image
from figstyle import plt, Circle, Rectangle, FancyBboxPatch, INK, hollow_arrow
import matplotlib.patheffects as pe

SRC = os.environ['G37_PNG']
OUT = os.environ.get('OUT', 'figs')
FS, FS_SMALL = 9.5, 9.0
YELLOW = '#F0E442'                                      # colour-blind safe, prints as a light tone
plt.rcParams['hatch.linewidth'] = 0.8

# ---- cut the five frame panels out of the G37 strip; drop the burnt-in subtitles (bottom 20%)
img = np.asarray(Image.open(SRC).convert('RGB')).copy()
g = img.mean(2) < 235
col = g[img.shape[0] // 5: img.shape[0] * 7 // 10].mean(0) > 0.5
edges = np.flatnonzero(np.diff(col.astype(int)))
xs = list(zip(edges[0::2] + 1, edges[1::2] + 1))
row = g[:, xs[0][0]:xs[0][1]].mean(1) > 0.5
y0, y1 = np.flatnonzero(np.diff(row.astype(int)))[:2] + 1
y1 = y0 + int(0.80 * (y1 - y0))
assert len(xs) == 5, xs
panels = [img[y0:y1, a:b].copy() for a, b in xs]


def remove_box(p, rgb, tol):
    """Inpaint the burnt-in box of one colour; return the cleaned frame and the box (l, t, r, b)."""
    d = np.abs(p.astype(int) - np.array(rgb)[None, None]).max(2)
    m = (d < tol).astype(np.uint8)
    rs, cs = m.sum(1), m.sum(0)
    rows, cols = np.flatnonzero(rs > 0.5 * rs.max()), np.flatnonzero(cs > 0.5 * cs.max())
    box = (cols.min(), rows.min(), cols.max(), rows.max())
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(p, m, 5, cv2.INPAINT_TELEA), box


A_img, A_box = remove_box(panels[2], (235, 104, 52), 38)
L_img, L_box = remove_box(panels[3], (42, 120, 214), 40)


def crop_square(p, box, scale=3.2):
    l, t, r, b = box
    cx, cy, side = (l + r) / 2, (t + b) / 2, int(max(r - l, b - t) * scale)
    side = min(side, p.shape[0], p.shape[1])
    x0 = int(np.clip(cx - side / 2, 0, p.shape[1] - side)); y0_ = int(np.clip(cy - side * 0.45, 0, p.shape[0] - side))
    return p[y0_:y0_ + side, x0:x0 + side], (l - x0, t - y0_, r - x0, b - y0_), side


A_crop, A_b, A_side = crop_square(A_img, A_box)
L_crop, L_b, L_side = crop_square(L_img, L_box)

W, H = 6.3, 2.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
STROKE = [pe.withStroke(linewidth=2.6, foreground='black')]


def photo(x, y, s, crop, b, side, color, tag):
    ax.imshow(crop, extent=(x, x + s, y, y + s), zorder=2, aspect='auto', interpolation='lanczos')
    ax.add_patch(Rectangle((x, y), s, s, fc='none', ec=INK, lw=0.9, zorder=4))
    k = s / side
    l, t, r, bb = b
    bx, by, bw, bh = x + l * k, y + s - bb * k, (r - l) * k, (bb - t) * k
    ax.add_patch(Rectangle((bx, by), bw, bh, fc='none', ec='black', lw=3.2, zorder=3))
    ax.add_patch(Rectangle((bx, by), bw, bh, fc='none', ec=color, lw=1.6, zorder=3.1))
    ax.text(bx - 0.01, by + bh + 0.02, tag, ha='left', va='bottom', fontsize=FS, weight='bold', color='black', zorder=5,
            bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='black', lw=0.7))


# ---- left: observed frames of turn 3
ax.text(0.05, H - 0.04, 'Observed turn', ha='left', va='top', fontsize=FS + 0.5, weight='bold')
S_ = 1.02
photo(0.06, 1.45, S_, A_crop, A_b, A_side, 'white', 'A')
photo(1.18, 1.45, S_, L_crop, L_b, L_side, YELLOW, 'L')
ax.text(0.06 + S_ / 2, 1.38, 'speaker A', ha='center', va='top', fontsize=FS)
ax.text(1.18 + S_ / 2, 1.38, 'listener L', ha='center', va='top', fontsize=FS)
ax.text(0.06, 0.92, 'The next speaker is not\nobserved. Here it is the\nlistener L, who answers.', ha='left', va='top',
        fontsize=FS_SMALL, style='italic', linespacing=1.15)

# ---- right: two ways to turn the observed turns into a forecast
X0, WB = 2.45, 3.8


def row_box(y, h, title):
    ax.add_patch(FancyBboxPatch((X0, y), WB, h, boxstyle='round,pad=0,rounding_size=0.05', fc='white', ec=INK, lw=0.9,
                                zorder=1))
    ax.text(X0 + 0.07, y + h - 0.05, title, ha='left', va='top', fontsize=FS + 0.5, weight='bold')


YT, HT = 1.45, 1.25
row_box(YT, HT, 'Clip-level fusion')
cx, cy = X0 + 0.95, YT + 0.72
ax.add_patch(plt.matplotlib.patches.Ellipse((cx, cy), 1.55, 0.62, fc='white', ec=INK, lw=1.0, zorder=2))
for j, k in enumerate(['T1', 'T2', 'T3']):
    ax.add_patch(Rectangle((cx - 0.66 + 0.45 * j, cy - 0.15), 0.4, 0.3, fc='white', ec=INK, lw=0.8, zorder=3))
    ax.text(cx - 0.46 + 0.45 * j, cy, k, ha='center', va='center', fontsize=FS_SMALL, zorder=4)
ax.text(cx, YT + 0.14, 'one summary per turn,\nspeaker dominates', ha='center', va='center', fontsize=FS_SMALL, style='italic',
        linespacing=1.1)
hollow_arrow(ax, X0 + 1.9, cy, 0.55, 0, width=0.1, head_width=0.2, head_length=0.1)
ax.text(X0 + 2.55, cy + 0.14, 'forecast:', ha='left', va='center', fontsize=FS)
ax.text(X0 + 2.55, cy - 0.12, 'neutral', ha='left', va='center', fontsize=FS + 0.5, weight='bold')
ax.text(X0 + 3.07, cy - 0.12, '✗', ha='left', va='center', fontsize=FS + 1, weight='bold', family='DejaVu Sans')

YB, HB = 0.1, 1.25
row_box(YB, HB, 'Participant-centric')
HATCH = {'A': '////', 'L': 'xxxx', 'O': '....'}
pos = {'A': (X0 + 0.55, YB + 0.86), 'L': (X0 + 1.0, YB + 0.86), 'O': (X0 + 0.78, YB + 0.64)}
for r_, (px, py) in pos.items():
    ax.add_patch(Circle((px, py), 0.2, fc='white', ec=INK, lw=0.9, hatch=HATCH[r_], zorder=2))
for r_, (px, py) in pos.items():
    ax.text(px, py + (0.05 if r_ != 'O' else -0.02), r_, ha='center', va='center', fontsize=FS, weight='bold', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec='none'))
ax.text(X0 + 0.78, YB + 0.16, 'one token per person\nand turn', ha='center', va='center', fontsize=FS_SMALL, style='italic',
        linespacing=1.1)
hollow_arrow(ax, X0 + 1.9, YB + 0.72, 0.55, 0, width=0.1, head_width=0.2, head_length=0.1)
ax.text(X0 + 2.55, YB + 0.86, 'forecast:', ha='left', va='center', fontsize=FS)
ax.text(X0 + 2.55, YB + 0.6, 'happy', ha='left', va='center', fontsize=FS + 0.5, weight='bold')
ax.text(X0 + 3.0, YB + 0.6, '✓', ha='left', va='center', fontsize=FS + 1, weight='bold', family='DejaVu Sans')

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'motivation.pdf'), bbox_inches='tight', pad_inches=0.02, dpi=400)
fig.savefig(os.path.join(OUT, 'motivation.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
