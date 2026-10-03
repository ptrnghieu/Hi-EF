"""MELD feasibility of the authors' research questions P1, P3, P5, P6 (text only; MELD treated on its own).

Windows: utterances u1, u2, u3 (A = speaker of u3) -> emotion of u4, spoken by B != A, inside one dialogue.
Speaker identities are allowed (MELD has them for every utterance). Development = official train+dev with 5-fold
CV grouped by dialogue; the official test split is scored once at the end with models fit on train+dev.

Text features: sentence embeddings from sentence-transformers/all-mpnet-base-v2 (general-purpose, not trained on
MELD emotion labels), PCA-64 fitted on the training rows. Model family 1: multinomial logistic regression
(C chosen by inner grouped CV on NLL). Family 2 (P3 only): histogram gradient boosting.

Pre-registered rules (written before running; 95% CI = bootstrap over dialogues, 1,000 draws, paired):

P6  value of the person-observation assignment.
    A: positional content of u1..u3. B: A + who-is-who (is-B / is-A / third-party tags for u1, u2, and the content
    of B's own and of third parties' turns). W: model B evaluated with a plausible wrong assignment (B's identity
    moved to another speaker present in u1/u2, or to nobody).
    FEASIBLE if NLL(A) - NLL(B) has CI lower bound > 0 AND the share of windows whose argmax changes under W is >= 5%.

P1  B-relative relation between B's earlier content and A's current turn (situation of B).
    On windows where B spoke in u1/u2: model B vs model B + relational terms (elementwise product and cosine of
    B's latest earlier turn and u3). FEASIBLE if NLL(B) - NLL(B+rel) has CI lower bound > 0.

P5  influence of third parties (neither A nor B).
    Full model B vs the same model with third-party turns removed (content and tags), trained separately.
    D = [NLL(no-third) - NLL(full)] on windows with a third party minus the same on windows without one (generic
    retraining cost). FEASIBLE if D has CI lower bound > 0.

P3  forecastable distinctions and what u3 adds (as Hi-EF G19).
    Pair AUC of log p(a) - log p(b) for I-II (u1, u2) vs I-III (u1..u3), families LR and GBM, eligible pairs = both
    classes >= 50 windows. CONTINUE if Spearman(LR, GBM) of pair AUCs at I-III >= 0.7 AND >= 2 pairs gain with
    CI > 0 AND >= 2 pairs are above chance at I-II (CI lower > 0.5) with a gain CI that includes 0.

usage: python feasibility_text.py <dir with MELD train/dev/test _sent_emo.csv> <out dir>
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
E = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
K, N_BOOT, CS = 64, 1000, [0.003, 0.01, 0.03, 0.1, 0.3]
rng = np.random.default_rng(0)

# ---------------- utterances and embeddings ----------------
U = []
for split, f in (('dev_pool', 'train_sent_emo.csv'), ('dev_pool', 'dev_sent_emo.csv'), ('test', 'test_sent_emo.csv')):
    d = pd.read_csv(os.path.join(DATA, f))
    d['split'] = split
    d['dlg'] = f.split('_')[0] + '_' + d.Dialogue_ID.astype(str)
    U.append(d)
U = pd.concat(U, ignore_index=True).sort_values(['split', 'dlg', 'Utterance_ID']).reset_index(drop=True)
U['y'] = U.Emotion.map(E.index)
emb_path = os.path.join(OUT, 'meld_mpnet.npy')
if os.path.exists(emb_path):
    EMB = np.load(emb_path)
else:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
    EMB = st.encode(U.Utterance.astype(str).str.replace('\x92', "'").tolist(), batch_size=64, show_progress_bar=True,
                    normalize_embeddings=True)
    np.save(emb_path, EMB)
print('utterances', len(U), 'embedding', EMB.shape, flush=True)

# ---------------- windows ----------------
W = []
for dlg, g in U.groupby('dlg', sort=False):
    ix = g.index.to_numpy(); sp = g.Speaker.to_numpy(); y = g.y.to_numpy()
    for i in range(3, len(g)):
        A, B = sp[i - 1], sp[i]
        if A == B:
            continue
        W.append(dict(dlg=dlg, split=g.split.iloc[0], u1=ix[i - 3], u2=ix[i - 2], u3=ix[i - 1], y=y[i],
                      s1=sp[i - 3], s2=sp[i - 2], A=A, B=B))
W = pd.DataFrame(W)
for k in ('1', '2'):
    W['isB' + k] = (W['s' + k] == W.B).astype(float)
    W['isA' + k] = (W['s' + k] == W.A).astype(float)
    W['isT' + k] = ((W['s' + k] != W.B) & (W['s' + k] != W.A)).astype(float)
W['Bspoke'] = (W.isB1 + W.isB2) > 0
W['third'] = (W.isT1 + W.isT2) > 0
dev = W[W.split == 'dev_pool'].reset_index(drop=True)
tst = W[W.split == 'test'].reset_index(drop=True)
print(f"windows dev_pool {len(dev)} test {len(tst)} | B spoke earlier {dev.Bspoke.mean():.3f} | third party {dev.third.mean():.3f}")


def wrong_assignment(df, seed=1):
    # B's identity moved to another speaker present in u1/u2 (not A), or to nobody; tags recomputed
    r = np.random.default_rng(seed); df = df.copy()
    for n in range(len(df)):
        cands = [s for s in {df.s1[n], df.s2[n]} if s not in (df.B[n], df.A[n])] + ['<nobody>']
        fakeB = cands[r.integers(len(cands))]
        for k in ('1', '2'):
            df.loc[n, 'isB' + k] = float(df['s' + k][n] == fakeB)
            df.loc[n, 'isT' + k] = float(df['s' + k][n] not in (fakeB, df.A[n]))
    return df


def features(df, P, kind):
    # P: fitted PCA on utterance embeddings; kind selects the arm
    z = {k: P.transform(EMB[df['u' + k].to_numpy()]) for k in ('1', '2', '3')}
    tags = lambda k: df[['isB' + k, 'isA' + k, 'isT' + k]].to_numpy()
    if kind == 'I-II':
        return np.concatenate([z['1'], z['2']], 1)
    if kind == 'I-III' or kind == 'A':
        return np.concatenate([z['1'], z['2'], z['3']], 1)
    drop_third = kind == 'B-nothird'
    zz = {k: z[k] * (1 - df['isT' + k].to_numpy()[:, None]) if drop_third else z[k] for k in ('1', '2')}
    t = {k: tags(k) * (np.array([[1, 1, 0]]) if drop_third else 1) for k in ('1', '2')}
    Bc = zz['1'] * df.isB1.to_numpy()[:, None] + zz['2'] * df.isB2.to_numpy()[:, None]
    Tc = np.zeros_like(Bc) if drop_third else z['1'] * df.isT1.to_numpy()[:, None] + z['2'] * df.isT2.to_numpy()[:, None]
    X = [zz['1'], zz['2'], z['3'], t['1'], t['2'], Bc, Tc]
    if kind == 'B+rel':
        latestB = np.where(df.isB2.to_numpy()[:, None] > 0, z['2'], z['1'] * df.isB1.to_numpy()[:, None])
        X += [latestB * z['3'], (latestB * z['3']).sum(1, keepdims=True)]
    return np.concatenate(X, 1)


def fit_lr(Xtr, ytr, gtr):
    best = None
    for C in CS:
        L = np.zeros((len(ytr), 7))
        for a, b in GroupKFold(3).split(Xtr, ytr, gtr):
            L[b] = lr_logits(Xtr[a], ytr[a], Xtr[b], C)
        s = nll(L, ytr)
        if best is None or s < best[0]:
            best = (s, C)
    return best[1]


def lr_logits(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    L = np.full((len(Xte), 7), -30.0)
    L[:, m.classes_] = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-9, 1))
    return L


def nll(L, y):
    L = L - L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def gbm_logits(Xtr, ytr, Xte):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(Xtr, ytr)
    L = np.full((len(Xte), 7), -30.0)
    L[:, m.classes_] = np.log(np.clip(m.predict_proba(Xte), 1e-9, 1))
    return L


ARMS = ['A', 'B', 'B+rel', 'B-nothird', 'I-II', 'I-III']
OOF = {a: np.zeros((len(dev), 7)) for a in ARMS + ['W', 'GBM I-II', 'GBM I-III']}
TST = {a: np.zeros((len(tst), 7)) for a in OOF}
y, g = dev.y.to_numpy(), dev.dlg.to_numpy()
devW, tstW = wrong_assignment(dev), wrong_assignment(tst)


def run(tr_df, te_df, te_wrong, sink, rows):
    utr = np.unique(np.concatenate([tr_df[c].to_numpy() for c in ('u1', 'u2', 'u3')]))
    P = PCA(K, random_state=0).fit(EMB[utr])
    ytr, gtr = tr_df.y.to_numpy(), tr_df.dlg.to_numpy()
    for a in ARMS:
        Xtr, Xte = features(tr_df, P, a), features(te_df, P, a)
        C = fit_lr(Xtr, ytr, gtr)
        sink[a][rows] = lr_logits(Xtr, ytr, Xte, C)
        if a == 'B':
            sink['W'][rows] = lr_logits(Xtr, ytr, features(te_wrong, P, 'B'), C)
    for a in ('I-II', 'I-III'):
        sink['GBM ' + a][rows] = gbm_logits(features(tr_df, P, a), ytr, features(te_df, P, a))


for f, (tr, te) in enumerate(GroupKFold(5).split(dev, y, g)):
    run(dev.iloc[tr].reset_index(drop=True), dev.iloc[te].reset_index(drop=True),
        devW.iloc[te].reset_index(drop=True), OOF, te)
    print(f"fold {f} done", flush=True)
run(dev, tst, tstW, TST, np.arange(len(tst)))

# ---------------- statistics ----------------
groups = [np.where(g == d)[0] for d in np.unique(g)]
DRAWS = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
nll_rows = lambda L, idx, yy: -(L[idx] - np.log(np.exp(L[idx]).sum(1, keepdims=True)))[np.arange(len(idx)), yy[idx]]


def ci(stat):
    pt = stat(np.arange(len(dev)))
    bs = [stat(idx) for idx in DRAWS]
    return pt, *np.nanpercentile(bs, [2.5, 97.5])


def dnll(a, b, mask=None, L=OOF, yy=y):
    def s(idx):
        idx = idx if mask is None else idx[mask[idx]]
        return nll_rows(L[a], idx, yy).mean() - nll_rows(L[b], idx, yy).mean()
    return s


uar = lambda L, yy: 100 * np.mean([np.mean(L.argmax(1)[yy == c] == c) for c in np.unique(yy)])
res = {'n_dev': int(len(dev)), 'n_test': int(len(tst))}
for a in OOF:
    res[f'dev_{a}'] = {'NLL': nll(OOF[a], y), 'UAR': uar(OOF[a], y)}
    res[f'test_{a}'] = {'NLL': nll(TST[a], tst.y.to_numpy()), 'UAR': uar(TST[a], tst.y.to_numpy())}

p6 = ci(dnll('A', 'B'))
flip = float(np.mean(OOF['B'].argmax(1) != OOF['W'].argmax(1)))
res['P6'] = {'dNLL_A_minus_B': p6, 'dNLL_W_minus_B': ci(dnll('W', 'B')), 'flip_rate': flip,
             'decision': 'FEASIBLE' if p6[1] > 0 and flip >= 0.05 else 'NOT FEASIBLE'}
mB = dev.Bspoke.to_numpy()
p1 = ci(dnll('B', 'B+rel', mB))
res['P1'] = {'dNLL_B_minus_Brel_on_Bspoke': p1, 'decision': 'FEASIBLE' if p1[1] > 0 else 'NOT FEASIBLE'}
mT = dev.third.to_numpy()


def did(idx):
    a, b = idx[mT[idx]], idx[~mT[idx]]
    return ((nll_rows(OOF['B-nothird'], a, y) - nll_rows(OOF['B'], a, y)).mean()
            - (nll_rows(OOF['B-nothird'], b, y) - nll_rows(OOF['B'], b, y)).mean())


p5 = ci(did)
res['P5'] = {'DiD': p5, 'raw_on_third': ci(dnll('B-nothird', 'B', mT)),
             'decision': 'FEASIBLE' if p5[1] > 0 else 'NOT FEASIBLE'}

cnt = np.bincount(y, minlength=7)
ELIG = [(a, b) for a, b in itertools.combinations(range(7), 2) if cnt[a] >= 50 and cnt[b] >= 50]


def pauc(L, a, b, idx):
    m = idx[np.isin(y[idx], [a, b])]
    return roc_auc_score(y[m] == a, L[m, a] - L[m, b]) if len(np.unique(y[m])) == 2 else np.nan


rows = []
for a, b in ELIG:
    r = {'pair': f'{E[a]}/{E[b]}'}
    for fam, k12, k13 in (('lr', 'I-II', 'I-III'), ('gbm', 'GBM I-II', 'GBM I-III')):
        r[f'{fam}_12'] = pauc(OOF[k12], a, b, np.arange(len(dev)))
        r[f'{fam}_13'] = pauc(OOF[k13], a, b, np.arange(len(dev)))
        bs = [(pauc(OOF[k13], a, b, idx) - pauc(OOF[k12], a, b, idx), pauc(OOF[k12], a, b, idx)) for idx in DRAWS]
        r[f'{fam}_d'] = r[f'{fam}_13'] - r[f'{fam}_12']
        r[f'{fam}_d_lo'], r[f'{fam}_d_hi'] = np.nanpercentile([x[0] for x in bs], [2.5, 97.5])
        r[f'{fam}_12_lo'] = np.nanpercentile([x[1] for x in bs], 2.5)
    rows.append(r)
T3 = pd.DataFrame(rows)
T3.to_csv(os.path.join(OUT, 'p3_pair_auc.csv'), index=False)
rho = spearmanr(T3.lr_13, T3.gbm_13).correlation
up = int((T3.lr_d_lo > 0).sum())
flat = int(((T3.lr_12_lo > 0.5) & (T3.lr_d_lo <= 0) & (T3.lr_d_hi >= 0)).sum())
res['P3'] = {'eligible_pairs': len(ELIG), 'spearman_lr_gbm_I_III': float(rho), 'pairs_up': up, 'pairs_flat': flat,
             'decision': 'CONTINUE' if rho >= 0.7 and up >= 2 and flat >= 2 else 'STOP'}
print(T3.round(3).to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k.startswith('P')}, indent=1, default=float))
json.dump(res, open(os.path.join(OUT, 'feasibility_text.json'), 'w'), indent=1, default=float)
