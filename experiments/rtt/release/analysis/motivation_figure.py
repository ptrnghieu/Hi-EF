# Paper Figure 1 (one column, top right of page 1): frames of one multi-party sample (G37 strip) plus a contrast between clip-level
# fusion and participant-centric evidence. Colour-blind-safe Okabe-Ito fills with black text (no red/green/blue/grey text); text is 9-10 pt at final size. The role boxes burnt into the G37 frames are removed
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
FS, FS_SMALL = 10.0, 9.0
YELLOW = '#F0E442'; ORANGE = '#E69F00'; SKY = '#56B4E9'                                      # colour-blind safe, prints as a light tone
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

W, H = 3.05, 3.85
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def photo(x, y, s, crop, b, side, color, tag):
    ax.imshow(crop, extent=(x, x + s, y, y + s), zorder=2, aspect='auto', interpolation='lanczos')
    ax.add_patch(Rectangle((x, y), s, s, fc='none', ec=INK, lw=0.9, zorder=4))
    k = s / side
    l, t, r, bb = b
    bx, by, bw, bh = x + l * k, y + s - bb * k, (r - l) * k, (bb - t) * k
    ax.add_patch(Rectangle((bx, by), bw, bh, fc='none', ec='black', lw=3.4, zorder=3))
    ax.add_patch(Rectangle((bx, by), bw, bh, fc='none', ec=color, lw=2.0, zorder=3.1))
    ax.text(bx - 0.01, by + bh + 0.02, tag, ha='left', va='bottom', fontsize=FS, weight='bold', color='black', zorder=5,
            bbox=dict(boxstyle='square,pad=0.12', fc=color, ec='black', lw=0.7))


# ---- top: last observed turn
ax.text(0.04, H - 0.02, 'Last observed turn', ha='left', va='top', fontsize=FS, weight='bold')
S_ = 1.2
PY = H - 0.25 - S_
photo(0.05, PY, S_, A_crop, A_b, A_side, ORANGE, 'A')
photo(1.6, PY, S_, L_crop, L_b, L_side, SKY, 'L')
ax.text(0.05 + S_ / 2, PY - 0.03, 'speaker A', ha='center', va='top', fontsize=FS)
ax.text(1.6 + S_ / 2, PY - 0.03, 'listener L', ha='center', va='top', fontsize=FS)

X0, WB = 0.02, 3.01
FSB = FS


def row_box(y, h, title, bg):
    ax.add_patch(FancyBboxPatch((X0, y), WB, h, boxstyle='round,pad=0,rounding_size=0.05', fc=bg, ec=INK, lw=0.9,
                                zorder=1))
    ax.text(X0 + 0.07, y + h - 0.04, title, ha='left', va='top', fontsize=FS, weight='bold')


def forecast(y, word, mark):
    ax.text(2.0, y + 0.13, 'forecast:', ha='left', va='center', fontsize=FS)
    ax.text(2.0, y - 0.12, word, ha='left', va='center', fontsize=FS, weight='bold')
    ax.text(2.0 + 0.62, y - 0.12, mark, ha='left', va='center', fontsize=FS, weight='bold', family='DejaVu Sans')


HR = 1.0
YT = PY - 0.3 - HR
row_box(YT, HR, 'Clip-level fusion', '#FBE3B8')
cx, cy = 0.9, YT + 0.5
ax.add_patch(plt.matplotlib.patches.Ellipse((cx, cy), 1.5, 0.52, fc='white', ec=INK, lw=1.0, zorder=2))
for j, k in enumerate(['T1', 'T2', 'T3']):
    ax.add_patch(Rectangle((cx - 0.64 + 0.43 * j, cy - 0.12), 0.38, 0.24, fc='white', ec=INK, lw=0.8, zorder=3))
    ax.text(cx - 0.45 + 0.43 * j, cy, k, ha='center', va='center', fontsize=FS_SMALL, zorder=4)
ax.text(cx, YT + 0.13, 'speaker dominates', ha='center', va='center', fontsize=FS_SMALL, style='italic')
hollow_arrow(ax, 1.68, cy, 0.25, 0, width=0.09, head_width=0.18, head_length=0.09)
forecast(cy, 'neutral', '\u2717')

YB = YT - 0.1 - HR
row_box(YB, HR, 'Participant-centric', '#CFEAF8')
HATCH = {'A': None, 'L': 'xxxx', 'O': None}
COL = {'A': ORANGE, 'L': SKY, 'O': YELLOW}
pos = {'A': (0.5, YB + 0.55), 'L': (0.98, YB + 0.55), 'O': (0.74, YB + 0.37)}
for r_, (px, py) in pos.items():
    ax.add_patch(Circle((px, py), 0.19, fc=COL[r_], ec=INK, lw=0.9, zorder=2))
    if HATCH[r_]:
        ax.add_patch(Circle((px, py), 0.19, fc='none', ec=(0, 0, 0, 0.3), lw=0, hatch=HATCH[r_], zorder=2.1))
for r_, (px, py) in pos.items():
    ax.text(px, py + (0.04 if r_ != 'O' else -0.01), r_, ha='center', va='center', fontsize=FS, weight='bold', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.85))
ax.text(0.78, YB + 0.1, 'one token per person', ha='center', va='center', fontsize=FS_SMALL, style='italic')
hollow_arrow(ax, 1.68, YB + 0.5, 0.25, 0, width=0.09, head_width=0.18, head_length=0.09)
forecast(YB + 0.5, 'happy', '\u2713')

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'motivation.pdf'), bbox_inches='tight', pad_inches=0.02, dpi=400)
fig.savefig(os.path.join(OUT, 'motivation.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
