# Label-level replication of Hi-EF findings on MELD (gold labels only; no features).
# Windows: utterances i-3, i-2, i-1 (A = speaker of i-1) -> utterance i spoken by B != A, within a dialogue.
# usage: python labels_replication.py <dir with MELD train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv>
# (CSV files: https://github.com/declare-lab/MELD/tree/master/data/MELD)
import sys, os
os.chdir(sys.argv[1])
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
E=['anger','disgust','fear','joy','neutral','sadness','surprise']
def windows(df):
    rows=[]
    for d,g in df.sort_values(['Dialogue_ID','Utterance_ID']).groupby('Dialogue_ID'):
        sp=g.Speaker.tolist(); em=[E.index(e) for e in g.Emotion]
        for i in range(3,len(g)):
            A,B=sp[i-1],sp[i]
            if A==B: continue
            prevB=[em[j] for j in (i-2,i-3) if sp[j]==B]           # B's own earlier label within the window (latest first)
            prevO=[em[j] for j in (i-2,i-3) if sp[j] not in (B,)]  # others' earlier labels
            rows.append(dict(d=d,yA=em[i-1],yB=em[i],y2=em[i-2],y1=em[i-3],s2B=sp[i-2]==B,s1B=sp[i-3]==B,
                             prevB=prevB[0] if prevB else -1))
    return pd.DataFrame(rows)
_tr=pd.read_csv('train_sent_emo.csv'); _dv=pd.read_csv('dev_sent_emo.csv')
_dv['Dialogue_ID']=_dv['Dialogue_ID']+100000          # train and dev reuse dialogue ids; keep them apart
tr=windows(pd.concat([_tr,_dv]))
te=windows(pd.read_csv('test_sent_emo.csv'))
def uar(p,y): return 100*np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])
print(f"windows (B != A): train+dev {len(tr)}, test {len(te)}")
print("class share test:", (te.yB.value_counts(normalize=True).sort_index().round(3).rename(dict(enumerate(E)))).to_dict())
print(f"mirror P(yB==yA): train {np.mean(tr.yB==tr.yA):.3f} test {np.mean(te.yB==te.yA):.3f}")
# Markov oracle with gold A
T=np.ones((7,7)); 
for a,b in zip(tr.yA,tr.yB): T[a,b]+=1
T/=T.sum(1,keepdims=True); pB=T[te.yA.values]
print(f"[oracle] gold A -> P(B|A): UAR {uar(pB.argmax(1),te.yB.values):.2f} acc {100*np.mean(pB.argmax(1)==te.yB):.1f}")
# LA-free UAR: prior-corrected
pri=np.bincount(tr.yB,minlength=7)/len(tr)
print(f"[oracle] gold A -> P(B|A)/prior (balanced): UAR {uar((np.log(pB)-np.log(pri)).argmax(1),te.yB.values):.2f}")
m=te.prevB>=0
print(f"B spoke earlier in window: {m.mean()*100:.1f}% of test windows; copy-B UAR {uar(te.prevB[m].values,te.yB[m].values):.2f} acc {100*np.mean(te.prevB[m]==te.yB[m]):.1f} | copy-A on same subset UAR {uar(te.yA[m].values,te.yB[m].values):.2f} acc {100*np.mean(te.yA[m]==te.yB[m]):.1f}")
# inertia vs contagion: within windows where B spoke at i-2 or i-3, compare P(yB==B prev) vs P(yB==other at same position)
for pos,col,sc in [(2,'y2','s2B'),(3,'y1','s1B')]:
    a=te[te[sc]]; b=te[~te[sc]]
    print(f"utterance i-{pos}: P(yB == label) when speaker is B {np.mean(a.yB==a[col]):.3f} (n {len(a)}) vs other {np.mean(b.yB==b[col]):.3f} (n {len(b)})")
# hysteresis: additive vs interaction LR on gold labels (A label x history labels)
def X(df,inter):
    oh=lambda v: np.eye(7)[v]
    A_,H2,H1=oh(df.yA.values),oh(df.y2.values),oh(df.y1.values)
    B2=df.s2B.values[:,None].astype(float); B1=df.s1B.values[:,None].astype(float)
    base=[A_,H2,H1,H2*B2,H1*B1,B2,B1]
    if inter: base+= [np.einsum('ni,nj->nij',A_,H2).reshape(len(df),-1)]
    return np.concatenate(base,1)
from sklearn.metrics import log_loss
for inter in (False,True):
    best=None
    for C in [0.01,0.03,0.1,0.3,1]:
        m_=LogisticRegression(C=C,max_iter=3000).fit(X(tr,inter),tr.yB); ll=log_loss(te.yB,m_.predict_proba(X(te,inter)),labels=range(7))
        if best is None or ll<best[0]: best=(ll,C,uar(m_.predict_proba(X(te,inter)).argmax(1),te.yB.values))
    print(f"gold-label LR {'with A x history interaction' if inter else 'additive'}: test NLL {best[0]:.4f} (C {best[1]}) UAR {best[2]:.2f}")
L=pd.crosstab(pd.Series(te.yA).map(dict(enumerate(E))),pd.Series(te.yB).map(dict(enumerate(E))),normalize='index')/te.yB.map(dict(enumerate(E))).value_counts(normalize=True)
print("lift P(B|A)/P(B) (test):"); print(L.round(2))
rng=np.random.default_rng(0); m=te[te.prevB>=0]; ds=m.d.unique(); G={d:np.where(m.d.values==d)[0] for d in ds}
diff=[]
for _ in range(2000):
    idx=np.concatenate([G[d] for d in rng.choice(ds,len(ds))]); s=m.iloc[idx]
    diff.append(np.mean(s.prevB==s.yB)-np.mean(s.yA==s.yB))
print("paired copy-B minus copy-A accuracy on windows where B spoke earlier: %+.3f [%+.3f, %+.3f] (dialogue bootstrap)"%(np.mean(m.prevB==m.yB)-np.mean(m.yA==m.yB),*np.percentile(diff,[2.5,97.5])))
# same-speaker effect controlling position: logistic on yB==label(i-2) with speaker indicator
