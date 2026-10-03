# Exploratory (train+val OOF only; test untouched): RoleNet forecast quality vs the annotated uncertainty of the
# clip-IV label. usage: python target_ambiguity.py <g19 results dir> <annotation.csv> <source_folder_split_seed42.csv>
import numpy as np, pandas as pd, itertools, sys, os
from sklearn.metrics import roc_auc_score
RES, ANN, SPLIT = sys.argv[1:4]
E=['angry','disgust','fear','happy','neutral','sad','surprise']
z=np.load(os.path.join(RES,'g19_oof_probs.npz'),allow_pickle=True)
a=pd.read_csv(ANN,header=None,dtype=str).set_index(0)
sp=pd.read_csv(SPLIT,dtype=str).set_index('sample_id').loc[z['sample_id']]
y=z['y']; src=z['src']; P=z['I_III'].mean(0)
assert (sp.clip4_emotion.map(E.index).values==y).all()
U=sp.clip4.map(a[8]).astype(int).values; U3=sp.clip3.map(a[8]).astype(int).values
pol=sp.clip4.map(a[5]).values; inten=sp.clip4.map(a[6]).values
print('target uncertainty levels:',np.bincount(U)[1:], ' share uncertain(2/3): %.1f%%'%(100*(U>1).mean()))
print(pd.crosstab(pd.Series(y).map(dict(enumerate(E))),U,margins=True))
nll=-np.log(P[np.arange(len(y)),y]+1e-12)
def uar(m): p=P[m].argmax(1); yy=y[m]; return 100*np.mean([np.mean(p[yy==c]==c) for c in np.unique(yy)])
print('\nRoleNet (G19 I-III, 10 seeds) by target uncertainty:')
for u in (1,2,3):
    m=U==u; print(f'  level {u}: n={m.sum():4d} NLL {nll[m].mean():.3f} UAR {uar(m):5.2f} acc {100*np.mean(P[m].argmax(1)==y[m]):5.1f} maxprob {P[m].max(1).mean():.3f}')
# class-controlled NLL difference: within each class, NLL(uncertain)-NLL(certain)
print('\nwithin-class NLL, certain vs uncertain(2/3):')
for c in range(7):
    m1=(y==c)&(U==1); m2=(y==c)&(U>1)
    if m2.sum()>=10: print(f'  {E[c]:8s} n {m1.sum():4d}/{m2.sum():3d}  NLL {nll[m1].mean():.3f} vs {nll[m2].mean():.3f}')
# pair AUCs on certain-only vs all
groups={e:np.where(src==e)[0] for e in np.unique(src)}
rng=np.random.default_rng(0)
def pauc(a_,b_,idx):
    m=idx[np.isin(y[idx],[a_,b_])]; 
    return roc_auc_score(y[m]==a_,np.log(P[m,a_]+1e-9)-np.log(P[m,b_]+1e-9)) if len(np.unique(y[m]))==2 else np.nan
print('\npair AUC: all targets vs certain targets only (level 1)')
for a_,b_ in [(0,5),(0,1),(1,5),(3,5),(0,3),(4,5),(0,4),(3,4)]:
    allidx=np.arange(len(y)); cert=np.where(U==1)[0]
    d=[]
    for _ in range(500):
        idx=np.concatenate([groups[e] for e in rng.choice(list(groups),len(groups))])
        d.append(pauc(a_,b_,idx[U[idx]==1])-pauc(a_,b_,idx))
    lo,hi=np.nanpercentile(d,[2.5,97.5])
    print(f'  {E[a_]}/{E[b_]}: all {pauc(a_,b_,allidx):.3f} | certain {pauc(a_,b_,cert):.3f} | diff CI [{lo:+.3f},{hi:+.3f}]')
# does context predict target uncertainty? AUC of model entropy for U>1
from sklearn.metrics import roc_auc_score as auc
ent=-(P*np.log(P+1e-12)).sum(1)
print('\nAUC of forecast entropy for detecting uncertain target: %.3f'%auc(U>1,ent))
print('AUC of clip-III uncertainty>1 for target uncertainty>1: %.3f ; P(U4>1|U3>1)=%.2f vs P(U4>1|U3=1)=%.2f'%(auc(U>1,U3),(U[U3>1]>1).mean(),(U[U3==1]>1).mean()))
