# Paper figures and numbers from saved predictions (no training).
#   test (descriptive): saved G10 seed-averaged predictions of RoleNet, RoleNet-noRole and the baseline (saved as 'PaperBest')
#   development (train+val): G14 OOF (Full, noRole), A's clip-III label from G23, clip-IV certainty from annotation.csv
#   final model: if G36_DIR holds g36_test_probs.npz / g36_cv_oof.npz, the selected resampled arm replaces RoleNet
# Usage: ARTIFACTS=<dir with g10/, g14/, g23/> [G36_DIR=<G36 output>] ANNOT=<annotation.csv> SPLIT=<split csv>
#        OUT=<figs dir> python paper_figures.py
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

A, OUT = os.environ.get('ARTIFACTS', '.'), os.environ.get('OUT', 'figs')
os.makedirs(OUT, exist_ok=True)
EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', [ORANGE, '#f3c2ad', '#ececea', '#b7d3f6', BLUE])
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 9, 'axes.titlecolor': INK, 'font.family': 'DejaVu Sans'})
ann = pd.read_csv(os.environ['ANNOT'], header=None, dtype=str).set_index(0)
sp = pd.read_csv(os.environ['SPLIT'], dtype=str).set_index('sample_id')
NUM = {}


def recalls(p, y):
    return np.array([(p[y == c] == c).mean() * 100 if (y == c).any() else np.nan for c in range(7)])


def uar(p, y):
    return np.nanmean(recalls(p, y))


def confusion(p, y):
    C = np.zeros((7, 7))
    for t, q in zip(y, p):
        C[t, q] += 1
    return C / C.sum(1, keepdims=True).clip(min=1) * 100


def draw_cm(ax, C, title, n):
    im = ax.imshow(C, cmap=SEQ, vmin=0, vmax=100)
    for i in range(7):
        for j in range(7):
            v = C[i, j]
            ax.text(j, i, f"{v:.0f}", ha='center', va='center', fontsize=6.5,
                    color='#ffffff' if v > 55 else (INK if v >= 0.5 else '#b5b4af'),
                    weight='bold' if i == j and v >= 0.5 else 'normal')
    ax.set_xticks(range(7)); ax.set_xticklabels(EMO, rotation=40, ha='right', rotation_mode='anchor', fontsize=6.8)
    ax.set_yticks(range(7)); ax.set_yticklabels([f"{e} ({k})" for e, k in zip(EMO, n)], fontsize=6.8)
    ax.set_xticks(np.arange(-.5, 7), minor=True); ax.set_yticks(np.arange(-.5, 7), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='both', length=0)
    ax.set_title(title, fontsize=8, pad=4); ax.set_xlabel('predicted', fontsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# ---------------- test (saved G10 predictions; descriptive)
g = lambda f: np.load(os.path.join(A, 'g10', f + '.npy'), allow_pickle=True)
sid = g('sample_id')
yt = sp.loc[sid].clip4_emotion.map(E2I).values
nt = np.bincount(yt, minlength=7)
G36 = os.environ.get('G36_DIR')
ours_test = g('RoleNet').mean(0)
if G36 and os.path.exists(os.path.join(G36, 'g36_test_probs.npz')):
    t36 = np.load(os.path.join(G36, 'g36_test_probs.npz'), allow_pickle=True)
    pos = {s_: i for i, s_ in enumerate(t36['sample_id'])}
    sel = str(t36['selected']).replace('-', '_')
    ours_test = t36[sel][:, [pos[s_] for s_ in sid]].mean(0)
    NUM['final_model'] = str(t36['selected'])
else:
    NUM['final_model'] = 'RoleNet (G10, uniform sampling)'
PT = {'Baseline': g('PaperBest').mean(0).argmax(1), 'RoleNet': ours_test.argmax(1)}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.75))
for ax, (k, p) in zip(axs, PT.items()):
    im = draw_cm(ax, confusion(p, yt), f"{k} (UAR {uar(p, yt):.1f})", nt)
axs[0].set_ylabel('true (test count)', fontsize=7)
fig.tight_layout(w_pad=1.0)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.015); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6, length=0); cb.set_label('row %', fontsize=6.5)
fig.savefig(f"{OUT}/confusion_test.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)
NUM['test_recall'] = {k: dict(zip(EMO, np.round(recalls(p, yt), 1))) for k, p in PT.items()}
NUM['test_counts'] = dict(zip(EMO, nt.tolist()))

# difference RoleNet - baseline (row-normalised)
D = confusion(PT['RoleNet'], yt) - confusion(PT['Baseline'], yt)
fig, ax = plt.subplots(figsize=(3.6, 3.3))
ax.imshow(D, cmap=DIV, norm=TwoSlopeNorm(0, -40, 40))
for i in range(7):
    for j in range(7):
        if abs(D[i, j]) >= 5:
            ax.text(j, i, f"{D[i, j]:+.0f}", ha='center', va='center', fontsize=6.5, color=INK)
ax.set_xticks(range(7)); ax.set_xticklabels([e[:4] for e in EMO], rotation=45)
ax.set_yticks(range(7)); ax.set_yticklabels([e[:4] for e in EMO])
ax.set_title('RoleNet − baseline (test, row %)'); ax.set_xlabel('predicted'); ax.set_ylabel('true')
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_diff_test.pdf"); plt.close(fig)

# per-class recall, test
fig, ax = plt.subplots(figsize=(4.6, 2.3))
x = np.arange(7); w = 0.38
for k, (name, col) in enumerate((('Baseline', ORANGE), ('RoleNet', BLUE))):
    r = recalls(PT[name], yt)
    ax.bar(x + (k - 0.5) * w * 1.06, r, w, color=col, label=name)
ax.set_xticks(x); ax.set_xticklabels([f"{e}\n(n={c})" for e, c in zip(EMO, nt)], fontsize=6.5)
ax.set_ylabel('recall (%)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/recall_test.pdf"); plt.close(fig)

# ---------------- development CV (G14 OOF) error analysis
z = np.load(os.path.join(A, 'g14', 'g14_results', 'g14_oof_probs.npz'), allow_pickle=True)
a23 = np.load(os.path.join(A, 'g23', 'results-2', 'g23_oof_logits.npz'), allow_pickle=True)
assert (a23['sample_id'] == z['sample_id']).all()
y, src = z['y'], z['src']
yA = a23['yA']
P = z['Full'].mean(0).argmax(1); Pn = z['noRole'].mean(0).argmax(1)
Pu = P.copy()                                     # uniform-sampling RoleNet (G14), the fair partner of noRole
if G36 and os.path.exists(os.path.join(G36, 'g36_cv_oof.npz')):
    c36 = np.load(os.path.join(G36, 'g36_cv_oof.npz'), allow_pickle=True)
    pos = {s_: i for i, s_ in enumerate(c36['sample_id'])}
    key = NUM['final_model'].replace('-', '_') if NUM['final_model'] in ('Up-sqrt', 'Up-bal') else 'RoleNet'
    P = c36[key][:, [pos[s_] for s_ in z['sample_id']]].mean(0).argmax(1)
vis = z['listener_in_III'].astype(bool)
cert = sp.loc[z['sample_id']].clip4.map(lambda c: ann.at[c, 8] if c in ann.index else 'nan').astype(str).values
mir = yA == y
ok = P == y
gidx = [np.where(src == e)[0] for e in np.unique(src)]
rng = np.random.default_rng(0)
BOOT = [np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))]) for _ in range(1000)]


def acc_ci(m):
    v = [ok[b][m[b]].mean() * 100 for b in BOOT if m[b].any()]
    return ok[m].mean() * 100, *np.percentile(v, [2.5, 97.5])


groups = [('mirror', mir), ('shift', ~mir), ('listener visible', vis), ('listener not visible', ~vis),
          ('certain label', cert == '1'), ('uncertain label', cert == '3')]
rows = [(n, m.sum(), *acc_ci(m)) for n, m in groups]
NUM['cv_accuracy_by_group'] = {n: {'n': int(c), 'acc': round(a, 1), 'ci': [round(l, 1), round(h, 1)]} for n, c, a, l, h in rows}
fig, ax = plt.subplots(figsize=(3.15, 2.0))
cols = [BLUE, '#9cc3f2'] * 3
ypos = [0, 1, 2.5, 3.5, 5, 6]
for yi, (n, c, a, l, h), col in zip(ypos, rows, cols):
    ax.barh(yi, a, color=col, height=0.7)
    ax.plot([l, h], [yi, yi], color=INK, lw=0.9)
    ax.text(h + 1.5, yi, f"{a:.1f}", va='center', fontsize=6.5, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels([f"{r[0]} ({r[1]:,})" for r in rows], fontsize=6.8, color=INK)
ax.invert_yaxis()
ax.set_xlabel('accuracy (%), development CV', fontsize=7); ax.set_xlim(0, 75)
ax.tick_params(axis='x', labelsize=6.5, length=0); ax.tick_params(axis='y', length=0)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/errors_by_group_cv.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# CV confusion: RoleNet vs without roles; recall on shifts
nc = np.bincount(y, minlength=7)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
draw_cm(axs[0], confusion(Pu, y), f"RoleNet, CV (UAR {uar(Pu, y):.1f})", nc)
draw_cm(axs[1], confusion(Pn, y), f"w/o role labels, CV (UAR {uar(Pn, y):.1f})", nc)
axs[0].set_ylabel('true (count)')
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_cv_roles.pdf"); plt.close(fig)
NUM['cv_recall_shift'] = dict(zip(EMO, np.round(recalls(P[~mir], y[~mir]), 1)))
NUM['cv_recall_mirror'] = dict(zip(EMO, np.round(recalls(P[mir], y[mir]), 1)))
NUM['cv_uar'] = {'final': round(uar(P, y), 2), 'RoleNet_uniform': round(uar(Pu, y), 2), 'noRole_uniform': round(uar(Pn, y), 2)}
NUM['cv_predicts_A_on_shift'] = round(float((P[~mir] == yA[~mir]).mean() * 100), 1)
json.dump(NUM, open(f"{OUT}/figure_numbers.json", 'w'), indent=1)
print(json.dumps(NUM, indent=1))
