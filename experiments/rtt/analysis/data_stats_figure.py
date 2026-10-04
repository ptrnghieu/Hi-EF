# "Who is on screen?" figure from statistics already measured (no new computation).
# Sources: G6a (README, 600 train+val MCIS for faces / 592 for voice), G16 per-MCIS table and G26 stats (2,421 MCIS).
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('OUT', 'figs')
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, BLUE, LBLUE = '#0b0b0b', '#52514e', '#d9d8d4', '#2a78d6', '#86b6ef'
ROWS = [  # (label, %, group) ; group 0 = parties, 1 = responder, 2 = editing / turn-taking
    ("$\\geq$3 people in the sample", 47.2, 0),
    ("responder B seen in clips I–III", 87.9, 1),
    ("B listening on screen in clip III", 47.9, 1),
    ("B spoke in clip I or II", 50.2, 1),
    ("A and B in the same frame (clip III)", 0.7, 2),
    ("clip I/II speaker is a third person", 38.6, 2),
]
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': GRID, 'xtick.color': INK2, 'ytick.color': INK, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(3.3, 2.2))
y = list(range(len(ROWS)))[::-1]
for yi, (lab, v, g) in zip(y, ROWS):
    ax.barh(yi, v, 0.62, color=BLUE if g == 1 else LBLUE)
    ax.text(v + 1.5, yi, f"{v:.1f}%", va='center', fontsize=7, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=7)
ax.set_xlim(0, 105); ax.set_xlabel('% of samples (train+val)', color=INK2)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'data_stats.pdf')); fig.savefig(os.path.join(OUT, 'data_stats.png'), dpi=200)
print('saved', os.path.join(OUT, 'data_stats.pdf'))
