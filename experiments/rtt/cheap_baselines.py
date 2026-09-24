"""Count-based and TF-IDF reference baselines on the locked source-folder split (train -> val).

Usage: python experiments/rtt/cheap_baselines.py --annotation annotation.csv --split source_folder_split_seed42.csv
The test split is dropped before anything else is computed.
"""
import argparse
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
ap=argparse.ArgumentParser(); ap.add_argument('--annotation',required=True); ap.add_argument('--split',required=True)
args=ap.parse_args()
ann=pd.read_csv(args.annotation,header=None,dtype=str).set_index(0)
sp=pd.read_csv(args.split,dtype=str)
sp=sp[sp.split!='test'].reset_index(drop=True)   # test stays locked
E=['angry','disgust','fear','happy','neutral','sad','surprise']; e2i={e:i for i,e in enumerate(E)}
P=['positive','neutral','negative']
txt=lambda c: str(ann.at[c,1]) if c in ann.index and isinstance(ann.at[c,1],str) else ''
for k in [1,2,3]: sp[f't{k}']=sp[f'clip{k}'].map(txt)
sp['pA']=sp.clip3.map(lambda c: ann.at[c,5]); sp['pB']=sp.clip4.map(lambda c: ann.at[c,5])
sp['yA']=sp.clip3_emotion.map(e2i); sp['yB']=sp.clip4_emotion.map(e2i)
tr,va=sp[sp.split=='train'],sp[sp.split=='val']
def m(pred,y,k=7):
    pred,y=np.asarray(pred),np.asarray(y); war=(pred==y).mean()*100
    uar=np.mean([(pred[y==c]==c).mean()*100 for c in range(k) if (y==c).any()]); return war,uar
rng=np.random.default_rng(0)
def ci(pred,y,src,k=7,B=2000):
    pred,y,src=map(np.asarray,(pred,y,src)); S=np.unique(src); out=[]
    for _ in range(B):
        idx=np.concatenate([np.where(src==s)[0] for s in rng.choice(S,len(S))]); out.append(m(pred[idx],y[idx],k))
    o=np.array(out); return np.percentile(o,[2.5,97.5],axis=0)
def rep(name,pred,y=va.yB,k=7):
    w,u=m(pred,y,k); lo,hi=ci(pred,y,va.source_folder,k)
    print(f"{name:<42} UAR {u:5.2f} [{lo[1]:.1f},{hi[1]:.1f}]  WAR {w:5.2f} [{lo[0]:.1f},{hi[0]:.1f}]")
print(f"train {len(tr)}  val {len(va)}  (test untouched)\n== Forecast B-emotion on VAL ==")
rep('Majority (train)', np.full(len(va), tr.yB.mode()[0]))
rep('Copy-A (gold E_A)  [oracle]', va.yA)
T=np.zeros((7,7)); 
for a,b in zip(tr.yA,tr.yB): T[a,b]+=1
rep('Markov argmax P(B|E_A gold) [oracle]', T[va.yA].argmax(1))
# Markov on (E_A, P_A) gold
key=lambda d: d.yA.astype(str)+'_'+d.pA
tab=pd.crosstab(key(tr),tr.yB); fb=T.argmax(1)
rep('Markov argmax P(B|E_A,P_A gold) [oracle]', [tab.loc[k].values.argmax() if k in tab.index else fb[a] for k,a in zip(key(va),va.yA)])
# balanced variant: argmax P(B|A)/P(B) -> boosts UAR
prior=np.bincount(tr.yB,minlength=7)/len(tr)
rep('Markov prior-corrected (gold E_A) [oracle]', (T[va.yA]/T[va.yA].sum(1,keepdims=True)/prior).argmax(1))
# text-only TF-IDF LR
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(pd.concat([tr.t1,tr.t2,tr.t3]))
X=lambda d,ks=(1,2,3): hstack([vec.transform(d[f't{k}']) for k in ks]).tocsr()
for ks,name in [((3,),'TF-IDF LR text III only'),((1,2,3),'TF-IDF LR text I+II+III')]:
    clf=LogisticRegression(max_iter=3000,C=1.0,class_weight='balanced').fit(X(tr,ks),tr.yB); rep(name, clf.predict(X(va,ks)))
print("\n== Recognize A (clip III) from its own text, VAL ==")
clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(X(tr,(3,)),tr.pA)
pp=clf.predict(X(va,(3,))); p2i={p:i for i,p in enumerate(P)}
rep('Polarity_A from text III (3-class)', [p2i[x] for x in pp], va.pA.map(p2i), 3)
rep('Polarity_A majority', np.full(len(va), p2i[tr.pA.mode()[0]]), va.pA.map(p2i), 3)
clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(X(tr,(3,)),tr.yA)
rep('Emotion_A from text III', clf.predict(X(va,(3,))), va.yA)
print("\npolarity dist train A:",tr.pA.value_counts().to_dict())
print("B polarity == A polarity rate (train):", round((tr.pA==tr.pB).mean()*100,1))
