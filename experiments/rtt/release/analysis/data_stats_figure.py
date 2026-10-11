# "Who is on screen?" figure from statistics already measured (no new computation).
# Sources: G6a (README, 600 train+val MCIS for faces / 592 for voice), G16 per-MCIS table and G26 stats (2,421 MCIS).
import os
from figstyle import plt, BAR, frame_axes

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
fig, ax = plt.subplots(figsize=(3.15, 1.85))
y = [6.4, 4.8, 3.8, 2.8, 1.2, 0.2]               # gaps between the three groups
COL = {0: BAR['grey'], 1: BAR['main'], 2: BAR['light']}
for yi, (lab, v, g) in zip(y, ROWS):
    ax.barh(yi, v, 0.72, color=COL[g], edgecolor='black', linewidth=0.4)
    inside = v > 80                                  # keep the label inside the frame
    ax.text(v - 1.5 if inside else v + 1.5, yi, f"{v:.1f}", va='center', ha='right' if inside else 'left',
            fontsize=7, color='white' if inside else 'black')
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.5)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 7.2); ax.set_xlabel('% of development samples', fontsize=8)
ax.tick_params(axis='x', labelsize=7); ax.tick_params(axis='y', length=0)
frame_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'data_stats.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(os.path.join(OUT, 'data_stats.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
print('saved', os.path.join(OUT, 'data_stats.pdf'))
