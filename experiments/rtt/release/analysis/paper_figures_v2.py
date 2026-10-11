# Colour-blind- and print-safe versions of the paper figures (Okabe-Ito fills, black text, no grey/blue/red text),
# plus the appendix figures (data splits, MELD windows, Qwen2.5-VL protocol).
#   S=<scratchpad with g10_test_probs.npz, split csv, meld csvs> B2=<b2_hief_probs.npz> OUT=../paper/figs python paper_figures_v2.py
import os, json, random
import numpy as np, pandas as pd
from figstyle import plt, FancyBboxPatch, Rectangle, hollow_arrow
from matplotlib.colors import LinearSegmentedColormap

OUT = os.environ.get('OUT', 'figs')
os.makedirs(OUT, exist_ok=True)
EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
OR_, YE, SKY, VER, GRN = '#E69F00', '#F0E442', '#56B4E9', '#D55E00', '#009E73'
plt.rcParams.update({'font.size': 8.5, 'hatch.linewidth': 0.6})
INK = '#000000'


def recalls(p, y):
    return np.array([(p[y == c] == c).mean() * 100 if (y == c).any() else np.nan for c in range(7)])


def uar(p, y):
    return np.nanmean(recalls(p, y))


def confusion(p, y):
    C = np.zeros((7, 7))
    for t, q in zip(y, p):
        C[t, q] += 1
    return C / C.sum(1, keepdims=True).clip(min=1) * 100


# ---------------------------------------------------------------- data
S = os.environ['S']
sp = pd.read_csv(f"{S}/g39dry/in/split/source_folder_split_seed42.csv", dtype=str)
g = np.load(f"{S}/g10_test_probs.npz", allow_pickle=True)
b = np.load(os.environ['B2'], allow_pickle=True)
assert (g['sample_id'] == b['sample_id']).all()
spi = sp.set_index('sample_id')
yt = spi.loc[g['sample_id']].clip4_emotion.map({e: i for i, e in enumerate(EMO)}).values
nt = np.bincount(yt, minlength=7)

# ---------------------------------------------------------------- confusion matrices (test)
SEQ = LinearSegmentedColormap.from_list('yb', ['#FFFFFF', '#FDE9A9', OR_, '#8C5A00'])
PT = {'Baseline': b['row11'].mean(0).argmax(1), 'PartiSet': g['RoleNet'].mean(0).argmax(1)}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.95))
for k, (ax, (name, p)) in enumerate(zip(axs, PT.items())):
    C = confusion(p, yt)
    im = ax.imshow(C, cmap=SEQ, vmin=0, vmax=100)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{C[i, j]:.0f}", ha='center', va='center', fontsize=8,
                    color='white' if C[i, j] > 62 else 'black', weight='bold' if i == j else 'normal')
    ax.set_xticks(range(7)); ax.set_xticklabels(EMO, rotation=40, ha='right', rotation_mode='anchor', fontsize=8.5)
    ax.set_yticks(range(7)); ax.set_yticklabels([f"{e} ({n})" for e, n in zip(EMO, nt)] if k == 0 else [''] * 7, fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xlabel(f"predicted\n({'ab'[k]}) {name} (UAR {uar(p, yt):.1f})", fontsize=9)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.8)
axs[0].set_ylabel('true class (test count)', fontsize=9)
fig.tight_layout(w_pad=0.6)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.015); cb.outline.set_linewidth(0.6)
cb.ax.tick_params(labelsize=8, length=2); cb.set_label('row %', fontsize=8.5)
fig.savefig(f"{OUT}/confusion_test.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- error analysis by subgroup (numbers from figure_numbers.json)
N = json.load(open(f"{OUT}/figure_numbers.json"))['cv_accuracy_by_group']
rows = [(n, v['n'], v['acc']) for n, v in N.items()]
fig, ax = plt.subplots(figsize=(3.15, 2.1))
ypos = [0, 1, 2.5, 3.5, 5, 6]
for yi, (n, c, a), col in zip(ypos, rows, [OR_, YE] * 3):
    ax.barh(yi, a, color=col, height=0.7, edgecolor='black', linewidth=0.6)
    ax.text(a + 1.5, yi, f"{a:.1f}", va='center', fontsize=8.5)
ax.set_yticks(ypos); ax.set_yticklabels([f"{r[0]} ({r[1]:,})" for r in rows], fontsize=8.5)
ax.set_ylim(6.7, -0.7); ax.set_xlim(0, 75)
ax.set_xlabel('accuracy (%), development CV', fontsize=9); ax.tick_params(axis='x', labelsize=8.5); ax.tick_params(axis='y', length=0)
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/errors_by_group_cv.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- party statistics (data_stats)
ROWS = [("$\\geq$3 people in the sample", 47.2, 0), ("responder B seen in clips I–III", 87.9, 1),
        ("B listening on screen in clip III", 47.9, 1), ("B spoke in clip I or II", 50.2, 1),
        ("A and B in the same frame (clip III)", 0.7, 2), ("clip I/II speaker is a third person", 38.6, 2)]
fig, ax = plt.subplots(figsize=(3.15, 2.0))
y = [6.4, 4.8, 3.8, 2.8, 1.2, 0.2]
COL = {0: SKY, 1: OR_, 2: YE}
for yi, (lab, v, g_) in zip(y, ROWS):
    ax.barh(yi, v, 0.72, color=COL[g_], edgecolor='black', linewidth=0.6)
    inside = v > 80
    ax.text(v - 1.5 if inside else v + 1.5, yi, f"{v:.1f}", va='center', ha='right' if inside else 'left', fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 7.2); ax.set_xlabel('% of development samples', fontsize=9)
ax.tick_params(axis='x', labelsize=8.5); ax.tick_params(axis='y', length=0)
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/data_stats.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- appendix: Hi-EF split
eps = sp.groupby('source_folder').agg(split=('split', 'first'), n=('sample_id', 'size')).reset_index()
dev_eps = sorted(eps[eps.split != 'test'].source_folder)
sel = sorted(random.Random(2026).sample(list(dev_eps), 5))
PCOL = {'train': OR_, 'val': YE, 'test': SKY}
fig = plt.figure(figsize=(6.3, 4.4))
ax = fig.add_axes([0.02, 0.62, 0.96, 0.36]); ax.axis('off'); ax.set_xlim(0, 27); ax.set_ylim(-0.9, 3.6)
ax.text(0, 3.5, '(a) The 53 episodes of Hi-EF and their partition (number = samples in the episode)', fontsize=9, va='top', weight='bold')
for i, r in eps.reset_index(drop=True).iterrows():
    cx, cy = i % 27, 1.75 - (i // 27) * 1.15
    ax.add_patch(Rectangle((cx + 0.04, cy), 0.92, 0.95, fc=PCOL[r.split], ec='black', lw=0.7))
    ax.text(cx + 0.5, cy + 0.66, r.source_folder, ha='center', va='center', fontsize=6.5)
    ax.text(cx + 0.5, cy + 0.27, str(r.n), ha='center', va='center', fontsize=6.5)
    if r.source_folder in sel:
        ax.add_patch(Rectangle((cx + 0.04, cy), 0.92, 0.95, fc='none', ec='black', lw=1.8, hatch='////'))
for k, (lab, key) in enumerate([('train: 37 episodes, 1,993 samples', 'train'), ('validation: 8 episodes, 428', 'val'),
                                 ('test: 8 episodes, 409', 'test')]):
    ax.add_patch(Rectangle((k * 8.5 + 0.1, -0.15), 0.5, 0.5, fc=PCOL[key], ec='black', lw=0.7))
    ax.text(k * 8.5 + 0.8, 0.1, lab, fontsize=8, va='center')
ax.add_patch(Rectangle((0.1, -0.85), 0.5, 0.5, fc='white', ec='black', lw=1.2, hatch='////'))
ax.text(0.8, -0.6, 'the 5 early-stopping episodes of the test run (drawn with random.Random(2026) from the 45 development episodes)', fontsize=8, va='center')
ax2 = fig.add_axes([0.075, 0.1, 0.40, 0.40])
x = np.arange(7); w = 0.27
for k, s_ in enumerate(['train', 'val', 'test']):
    c = sp[sp.split == s_].clip4_emotion.value_counts(normalize=True).reindex(EMO).fillna(0) * 100
    ax2.bar(x + (k - 1) * w, c, w, color=PCOL[s_], edgecolor='black', linewidth=0.6, label=s_)
ax2.set_xticks(x); ax2.set_xticklabels(EMO, rotation=35, ha='right', rotation_mode='anchor', fontsize=8.5)
ax2.set_ylabel('% of samples', fontsize=9); ax2.legend(frameon=False, fontsize=8.5)
ax2.set_title('(b) Class distribution per partition', fontsize=9, weight='bold', loc='left')
for s_ in ('top', 'right'):
    ax2.spines[s_].set_visible(False)
ax3 = fig.add_axes([0.57, 0.1, 0.41, 0.40]); ax3.axis('off'); ax3.set_xlim(0, 10); ax3.set_ylim(0, 6)
ax3.text(0, 5.9, '(c) Development protocol', fontsize=9, weight='bold', va='top')
ax3.add_patch(Rectangle((0, 4.1), 10, 0.9, fc='white', ec='black', lw=0.8)); ax3.text(5, 4.55, '45 development episodes (train + validation)', ha='center', va='center', fontsize=8.5)
for k in range(5):
    ax3.add_patch(Rectangle((k * 2, 2.6), 2, 0.9, fc=OR_ if k != 2 else YE, ec='black', lw=0.8))
    ax3.text(k * 2 + 1, 3.05, f'fold {k + 1}', ha='center', va='center', fontsize=8)
ax3.text(0, 2.3, 'development CV: 5 folds of whole episodes (one held out per round)', fontsize=8, va='top')
ax3.add_patch(Rectangle((0, 0.7), 8, 0.7, fc=OR_, ec='black', lw=0.8)); ax3.add_patch(Rectangle((8, 0.7), 2, 0.7, fc=YE, ec='black', lw=0.8, hatch='////'))
ax3.text(4, 1.05, '40 episodes: fit', ha='center', va='center', fontsize=8.5); ax3.text(9, 1.05, '5', ha='center', va='center', fontsize=8.5, bbox=dict(fc='white', ec='none', pad=0.5))
ax3.text(0, 0.45, 'test run: fit on 40, early stopping on 5, test on the 8 test episodes', fontsize=8, va='top')
fig.savefig(f"{OUT}/split_hief.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)
print('selection episodes', sel)

# ---------------------------------------------------------------- appendix: MELD windows
EM = {'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'joy': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprise'}
cnt, ex = {}, None
for n, key in [('train', 'train'), ('dev', 'val'), ('test', 'test')]:
    T = pd.read_csv(f"{S}/meld/{n}_sent_emo.csv").sort_values(['Dialogue_ID', 'Utterance_ID'])
    c = []
    for d, gg in T.groupby('Dialogue_ID'):
        sp_ = gg.Speaker.tolist(); em = gg.Emotion.tolist()
        for i in range(3, len(gg)):
            if sp_[i] != sp_[i - 1]:
                c.append(EM[em[i]])
        if n == 'test' and ex is None and 7 <= len(gg) <= 9 and len(set(sp_)) == 3:
            ex = sp_
    cnt[key] = pd.Series(c).value_counts().reindex(EMO).fillna(0)
fig = plt.figure(figsize=(6.3, 3.5))
ax = fig.add_axes([0.01, 0.52, 0.98, 0.46]); ax.axis('off'); ax.set_xlim(0, 20); ax.set_ylim(0, 4.6)
ax.text(0, 4.55, '(a) Windows from a dialogue (illustrative; letters = speakers)', fontsize=9, weight='bold', va='top')
sp_ = list('ABBACCA')                                   # an illustrative dialogue, not a real one
names = {s: 'ABCDEF'[i] for i, s in enumerate(dict.fromkeys(sp_))}
sp_l = [names[s] for s in sp_]
SC = {'A': OR_, 'B': SKY, 'C': YE, 'D': GRN, 'E': VER, 'F': 'white'}
for i, s in enumerate(sp_l):
    ax.add_patch(Rectangle((i * 2.1 + 0.2, 2.6), 1.7, 0.9, fc=SC[s], ec='black', lw=0.8))
    ax.text(i * 2.1 + 1.05, 3.05, f"u{i + 1}: {s}", ha='center', va='center', fontsize=8.5)
rows_ = [(i - 3, i) for i in range(3, len(sp_l))]
for r_, (a, i) in enumerate(rows_):
    ok = sp_l[i] != sp_l[i - 1]
    y0 = 1.85 - r_ * 0.62
    ax.annotate('', xy=(i * 2.1 + 1.9, y0 + 0.2), xytext=(a * 2.1 + 0.2, y0 + 0.2), arrowprops=dict(arrowstyle='-', lw=2.2, color='black'))
    ax.text(a * 2.1 - 0.1, y0 + 0.2, 'window', ha='right', va='center', fontsize=8)
    ax.text(i * 2.1 + 2.05, y0 + 0.2, ('kept: target u%d, speaker %s differs from %s' % (i + 1, sp_l[i], sp_l[i - 1])) if ok else
            ('dropped: u%d and u%d have the same speaker' % (i, i + 1)), ha='left', va='center', fontsize=8)
    ax.add_patch(Rectangle((i * 2.1 + 0.2, y0 - 0.02), 1.7, 0.45, fc='none', ec='black', lw=1.6, hatch='////' if ok else None))
ax2 = fig.add_axes([0.07, 0.1, 0.43, 0.33])
x = np.arange(7); w = 0.27
for k, s_ in enumerate(['train', 'val', 'test']):
    ax2.bar(x + (k - 1) * w, cnt[s_] / cnt[s_].sum() * 100, w, color=PCOL[s_], edgecolor='black', linewidth=0.6,
            label={'train': 'train (5,405)', 'val': 'dev (595)', 'test': 'test (1,341)'}[s_])
ax2.set_xticks(x); ax2.set_xticklabels(EMO, rotation=35, ha='right', rotation_mode='anchor', fontsize=8.5)
ax2.set_ylabel('% of windows', fontsize=9); ax2.legend(frameon=False, fontsize=8.5)
ax2.set_title('(b) Target class per partition', fontsize=9, weight='bold', loc='left')
for s_ in ('top', 'right'):
    ax2.spines[s_].set_visible(False)
ax3 = fig.add_axes([0.57, 0.1, 0.41, 0.33]); ax3.axis('off'); ax3.set_xlim(0, 10); ax3.set_ylim(0, 6)
ax3.text(0, 5.9, '(c) Official partition is kept', fontsize=9, weight='bold', va='top')
for k, (lab, key, a, bb) in enumerate([('train: 1,038 dialogues, 9,989 utterances', 'train', 0, 0), ('dev: 114 dialogues, 1,109', 'val', 0, 0), ('test: 280 dialogues, 2,610', 'test', 0, 0)]):
    ax3.add_patch(Rectangle((0, 3.9 - k * 1.3), 0.6, 0.7, fc=PCOL[key], ec='black', lw=0.7))
    ax3.text(0.9, 4.25 - k * 1.3, lab, fontsize=8.5, va='center')
ax3.text(0, 0.1, 'windows never cross a dialogue; no speaker names are used', fontsize=8, va='bottom')
fig.savefig(f"{OUT}/split_meld.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- appendix: Qwen2.5-VL protocol
fig = plt.figure(figsize=(6.3, 3.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 63); ax.set_ylim(0, 39); ax.axis('off')


def box(x, y, w, h, text, fc='white', fs=8.5, bold=False, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.5', fc=fc, ec='black', lw=0.9, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.15)


ax.text(0.3, 38.5, '(a) Zero-shot prompt', fontsize=9, weight='bold', va='top')
box(0.5, 31.5, 28, 5.2, 'Instruction: predict the emotion of the\nnext speaker (7 labels, one word)', fc='#FDE9A9')
for k, r in enumerate(['I', 'II', 'III']):
    box(0.5 + k * 9.5, 24.3, 9, 5.6, f'Clip {r}\nvideo, 8 frames\n+ subtitle', fc=SKY)
box(0.5, 18.2, 28, 4.6, 'Next speaker\'s emotion:', fc='white')
ax.text(0.5, 16.5, 'no audio; clip IV and speaker names never shown', fontsize=8, va='top', style='italic')
ax.text(33.3, 38.5, '(b) Few-shot prompt (7 examples, one per class)', fontsize=9, weight='bold', va='top')
for k in range(3):
    ax.add_patch(FancyBboxPatch((33.5 + k * 0.9, 29.8 - k * 0.9), 27, 6.6, boxstyle='round,pad=0,rounding_size=0.5', fc='white', ec='black', lw=0.9, zorder=1 + k))
ax.add_patch(FancyBboxPatch((36.2, 28), 24.3, 6.6, boxstyle='round,pad=0,rounding_size=0.5', fc='#FDE9A9', ec='black', lw=0.9, zorder=5))
ax.text(36.2 + 13.2, 31.3, 'example: 3 subtitles + 3 middle frames\n→ true emotion of the next speaker\n(from training rows; classes in a fixed random order)', ha='center', va='center', fontsize=8.5, zorder=6, linespacing=1.15)
box(33.5, 18.2, 28.5, 7.4, 'then the target sample\nas in (a): 3 clips as video,\nsubtitles, question', fc=SKY)
# scoring
ax.add_patch(FancyBboxPatch((0.5, 0.8), 62, 12.6, boxstyle='round,pad=0,rounding_size=0.5', fc='white', ec='black', lw=0.9, ls='--'))
ax.text(1.2, 13.0, '(c) Scoring without text generation', fontsize=9, weight='bold', va='top')
box(2, 3.2, 17, 5.6, 'Qwen2.5-VL-7B\n(frozen, fp16)', fc='#FDE9A9')
hollow_arrow(ax, 19.5, 6.0, 4, 0, width=1.0, head_width=2.2, head_length=1.4)
box(24.5, 3.2, 17, 5.6, 'logits of the first token\nof the 7 label words', fc='white')
hollow_arrow(ax, 42, 6.0, 4, 0, width=1.0, head_width=2.2, head_length=1.4)
box(47, 3.2, 14, 5.6, 'softmax over 7\n→ UAR / WAR', fc=SKY)
fig.savefig(f"{OUT}/llm_protocol.pdf", bbox_inches='tight', pad_inches=0.02); plt.close(fig)
print('done')
