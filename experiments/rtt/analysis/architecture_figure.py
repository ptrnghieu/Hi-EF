# RoleNet architecture figure (paper Figure 2), drawn with matplotlib.
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get('OUT', 'figs')
INK, INK2, LINE = '#0b0b0b', '#52514e', '#9a9994'
A, L, O, SP, SC, Q = '#eb6834', '#2a78d6', '#8a8984', '#1baf7a', '#eda100', '#4a3aa7'
FILL = {A: '#fbe1d6', L: '#d6e6fa', O: '#e7e6e3', SP: '#d1efe4', SC: '#fbecc8', Q: '#dcd8f2'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
fig, ax = plt.subplots(figsize=(7.0, 2.75))
ax.set_xlim(0, 100); ax.set_ylim(-1, 40); ax.axis('off')


def box(x, y, w, h, text, ec=LINE, fc='white', fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', ec=ec, fc=fc, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight='bold' if bold else 'normal')


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, color=INK2, lw=0.9))


# input clips
for k, name in enumerate(['Clip I', 'Clip II', 'Clip III']):
    box(1, 29 - 10.5 * k, 11, 8, f"{name}\nframes · audio\ntranscript", fs=6)
arrow(12.6, 20.5, 15.5, 20.5)
# module 1
box(15, 12, 17.5, 17, "Module 1\nidentities\n(ArcFace clustering)\n\nA = most visible\nL = 2nd most visible\nin clip III\nO = others", fs=6)
ax.text(23.75, 7.5, 'no clip IV, no names', ha='center', fontsize=5.5, color=INK2, style='italic')
arrow(33.2, 20.5, 35.6, 20.5)
# module 2 & 3: token grid
ax.text(49.8, 37.3, 'Modules 2–3: evidence set (15 typed tokens)', ha='center', fontsize=6.5, color=INK)
rows = [('A', A), ('L', L), ('O', O), ('speech', SP), ('scene', SC)]
for r, (name, c) in enumerate(rows):
    y = 30.5 - 5.6 * r
    ax.text(38.3, y + 1.8, name, ha='right', va='center', fontsize=6, color=c, weight='bold')
    for k in range(3):
        box(39.3 + 7.2 * k, y, 5.8, 3.6, ['I', 'II', 'III'][k], ec=c, fc=FILL[c], fs=6)
ax.text(49.8, 1.2, 'face tokens: attention pooling over frames\n+ role and clip embeddings; empty cell = learned token',
        ha='center', fontsize=5.5, color=INK2)
arrow(62.0, 20.5, 64.0, 20.5)
# module 4
box(64.5, 13, 4.5, 15, 'q', ec=Q, fc=FILL[Q], fs=8, bold=True)
box(70.5, 11, 14, 19, "Module 4\nset encoder\n\n2 × SAB\n(self-attention,\nno positional\nencoding)", fs=6)
arrow(85.2, 20.5, 88.0, 20.5)
box(88.5, 15.5, 10.5, 10, "p(y_B)\n7 emotions\nof responder\nin clip IV", ec=Q, fc=FILL[Q], fs=6)
ax.text(77.5, 7.0, 'read from query position (pooling seed)', ha='center', fontsize=5.5, color=INK2)
fig.savefig(os.path.join(OUT, 'architecture.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'architecture.png'), dpi=220, bbox_inches='tight')
print('saved')
