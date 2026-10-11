# Headroom of B's earlier labelled emotion (Z) beyond persistence: Z alone vs RoleNet OOF (X) vs X+Z, on the MCIS where
# B's earlier turn is labelled (B found by clip-IV voice, analysis only). Inputs: G14 OOF probs and G16 per-MCIS table.
# Usage: ARTIFACTS=<dir with g14/... and g16/...> python prevB_headroom.py
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, log_loss
EMO=['angry','disgust','fear','happy','neutral','sad','surprise']
z=np.load(os.path.join(os.environ.get('ARTIFACTS', '.'), 'g14/g14_results/g14_oof_probs.npz'),allow_pickle=True)
per=pd.read_csv(os.path.join(os.environ.get('ARTIFACTS', '.'), 'g16/results/g16_per_mcis.csv')); assert (per.sample_id.values==z['sample_id']).all()
y=z['y']; P=z['Full'].mean(0); fold=z['fold']; src=z['src']
bII=per.B_spoke_II.values&(per.y_II.values>=0); bI=per.B_spoke_I.values&(per.y_I.values>=0)&~per.B_spoke_II.values
yB=np.where(bII,per.y_II.values,np.where(bI,per.y_I.values,-1)); m=np.where(yB>=0)[0]
y_,Z,X,f_,s_=y[m],yB[m],np.log(np.clip(P[m],1e-6,1)),fold[m],src[m]
OH=np.eye(7)[Z]
U=lambda p,yy=None: 100*recall_score(y_ if yy is None else yy,p,average='macro',labels=range(7),zero_division=0)
print(f"subset {len(m)} / {len(y)} MCIS, {len(set(s_))} episodes; stay rate {np.mean(Z==y_)*100:.1f}%")
T=pd.crosstab(pd.Series(Z,name='Z_prev').map(dict(enumerate(EMO))),pd.Series(y_,name='Y_next').map(dict(enumerate(EMO))))
print("transition counts (rows Z_prev, cols Y_next):\n",T.to_string())
def cv(F,bal):
    out=np.zeros((len(y_),7))
    for f in np.unique(f_):
        tr,te=f_!=f,f_==f
        lr=LogisticRegression(C=1.0,max_iter=3000,class_weight='balanced' if bal else None).fit(F[tr],y_[tr])
        pp=np.full((te.sum(),7),1e-6); pp[:,lr.classes_]=lr.predict_proba(F[te]); out[te]=pp/pp.sum(1,keepdims=True)
    return out
arms={'X (RoleNet, refit)':X,'Z (learned P(Y|Z))':OH,'X+Z (additive)':np.hstack([X,OH])}
R={}
for k,F in arms.items():
    pb,pn=cv(F,True),cv(F,False); R[k]=(pb,pn)
    print(f"{k:<22} UAR(balanced LR) {U(pb.argmax(1)):5.2f} | NLL(plain LR) {log_loss(y_,pn,labels=range(7)):.4f}")
print(f"{'RoleNet raw':<22} UAR {U(P[m].argmax(1)):5.2f} | NLL {log_loss(y_,P[m],labels=range(7)):.4f}")
print(f"{'copy Z':<22} UAR {U(Z):5.2f}")
# episode bootstrap on differences
rng=np.random.default_rng(0); eps=np.unique(s_); idx={e:np.where(s_==e)[0] for e in eps}
def boot(fn,B=2000):
    v=[]
    for _ in range(B):
        ii=np.concatenate([idx[e] for e in rng.choice(eps,len(eps))]); v.append(fn(ii))
    return np.percentile(v,[2.5,97.5])
pXZ,pZ,pX=R['X+Z (additive)'],R['Z (learned P(Y|Z))'],R['X (RoleNet, refit)']
for name,a,b in [('X+Z - Z',pXZ,pZ),('X+Z - X',pXZ,pX),('Z - X',pZ,pX)]:
    du=lambda ii: U(a[0][ii].argmax(1),y_[ii])-U(b[0][ii].argmax(1),y_[ii])
    dn=lambda ii: log_loss(y_[ii],b[1][ii],labels=range(7))-log_loss(y_[ii],a[1][ii],labels=range(7))
    print(f"{name:<9} dUAR {du(np.arange(len(y_))):+6.2f} {np.round(boot(du),2)} | dNLL(+ = first better) {dn(np.arange(len(y_))):+.4f} {np.round(boot(dn),4)}")
rc=lambda p: recall_score(y_,p,average=None,labels=range(7),zero_division=0)*100
print(pd.DataFrame({'n':np.bincount(y_,minlength=7),'stay%':[100*np.mean(y_[Z==c]==c) if (Z==c).any() else np.nan for c in range(7)],
  'X':rc(pX[0].argmax(1)),'Z':rc(pZ[0].argmax(1)),'X+Z':rc(pXZ[0].argmax(1))},index=EMO).round(1).to_string())
