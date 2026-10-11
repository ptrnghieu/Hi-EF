# Post-hoc (not pre-registered) check of G25: is SEP better than a *tuned* weighted average of the two observations?
# usage: python g25_posthoc.py <g25_listener_obs.npz> <source_folder_split_seed42.csv>
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sys
E=['angry','disgust','fear','happy','neutral','sad','surprise']
z=np.load(sys.argv[1],allow_pickle=True)
sp=pd.read_csv(sys.argv[2],dtype=str).set_index('sample_id').loc[z['sample_id']]
yB=sp.clip4_emotion.map(E.index).values; yA=sp.clip3_emotion.map(E.index).values; src=sp.source_folder.values
Sidx=z['S']; E3,E12=z['E3'][Sidx],z['E12'][Sidx]; y=yB[Sidx]; g=src[Sidx]; shift=(yB!=yA)[Sidx]
def fitpred(Xtr,ytr,Xte,C):
    sc=StandardScaler().fit(Xtr); m=LogisticRegression(C=C,max_iter=3000).fit(sc.transform(Xtr),ytr)
    P=np.full((len(Xte),7),1e-6); P[:,m.classes_]=np.clip(m.predict_proba(sc.transform(Xte)),1e-6,1); return np.log(P/P.sum(1,keepdims=True))
def cv(make, Cs=(0.003,0.01,0.03,0.1), ws=(None,)):
    out=np.zeros((len(y),7)); chosen=[]
    for tr,te in GroupKFold(5).split(y,y,g):
        best=None
        for w in ws:
            for C in Cs:
                L=np.zeros((len(tr),7))
                for a,b in GroupKFold(5).split(tr,y[tr],g[tr]):
                    L[b]=fitpred(make(w)[tr[a]],y[tr[a]],make(w)[tr[b]],C)
                s=-L[np.arange(len(tr)),y[tr]].mean()
                if best is None or s<best[0]: best=(s,w,C)
        chosen.append(best[1:]); out[te]=fitpred(make(best[1])[tr],y[tr],make(best[1])[te],best[2])
    return out, chosen
models={'NOW':cv(lambda w:E3),'AVG':cv(lambda w:(E3+E12)/2),'SEP':cv(lambda w:np.concatenate([E3,E12],1)),
        'WAVG (w tuned)':cv(lambda w:w*E3+(1-w)*E12, ws=(0.5,0.6,0.7,0.8,0.9,1.0))}
for k,(L,ch) in models.items(): print(f"{k:16s} NLL all {-L[np.arange(len(y)),y].mean():.4f} shift {-L[shift][np.arange(shift.sum()),y[shift]].mean():.4f} | chosen {ch}")
grp=[np.where(g==e)[0] for e in np.unique(g)]; rng=np.random.default_rng(0)
def d(a,b,mask):
    La,Lb=models[a][0],models[b][0]
    f=lambda idx: (-La[idx,y[idx]].mean())-(-Lb[idx,y[idx]].mean())
    base=np.where(mask)[0]; pt=f(base); bs=[]
    for _ in range(1000):
        idx=np.concatenate([grp[i] for i in rng.integers(0,len(grp),len(grp))]); idx=idx[mask[idx]]; bs.append(f(idx))
    return pt,*np.percentile(bs,[2.5,97.5])
for a,b in [('AVG','SEP'),('WAVG (w tuned)','SEP'),('NOW','SEP'),('AVG','WAVG (w tuned)')]:
    for nm,m in (('shift',shift),('all',np.ones(len(y),bool))):
        print(f"  NLL({a}) - NLL({b}) [{nm}]: %+.4f [%+.4f, %+.4f]"%d(a,b,m))
# coefficient direction in SEP fitted on all S
sc=StandardScaler().fit(np.concatenate([E3,E12],1)); m=LogisticRegression(C=0.01,max_iter=3000).fit(sc.transform(np.concatenate([E3,E12],1)),y)
W3,W12=m.coef_[:,:10],m.coef_[:,10:]
print("per-class correlation of clip-III vs clip-I/II coefficients (positive = same direction = pooling):",
      {E[c]:round(float(np.corrcoef(W3[c],W12[c])[0,1]),2) for c in range(7)})
print("norm ratio |W12|/|W3| per class:",{E[c]:round(float(np.linalg.norm(W12[c])/np.linalg.norm(W3[c])),2) for c in range(7)})
