# Usage: ARTIFACTS=<dir with g14/... and g16/...> python listener_reliability_check.py
# Does trusting the listener evidence as a function of how much of it there is (L frame counts) improve on RoleNet?
# Stack G14 OOF of Full and minus-L (both 10-seed means); constant weights vs weights by L-reliability bin.
import os, numpy as np, pandas as pd, torch
from sklearn.metrics import recall_score
A=os.environ.get('ARTIFACTS','.')
z=np.load(f'{A}/g14/g14_results/g14_oof_probs.npz',allow_pickle=True); per=pd.read_csv(f'{A}/g16/results/g16_per_mcis.csv')
assert (per.sample_id.values==z['sample_id']).all()
y=z['y']; fold=z['fold']; src=z['src']; N=len(y)
F=np.log(np.clip(z['Full'].mean(0),1e-6,1)); M=np.log(np.clip(z['minus_L'].mean(0),1e-6,1))
n3=per.nL3.values; n12=per.nL1.values+per.nL2.values
bins3=np.digitize(n3,[1,2,4]); bins12=np.digitize(n12,[1,4])          # III: 0 / 1 / 2-3 / 4+ ; I+II: 0 / 1-3 / 4+
print("L frames in clip III: 0 / 1 / 2-3 / 4+ =", np.bincount(bins3)/N)
G=np.eye(4)[bins3]; G12=np.eye(3)[bins12]
def fit(kind,tr,te,l2=1e-3):
    Ft,Mt,yt=torch.tensor(F,dtype=torch.float32),torch.tensor(M,dtype=torch.float32),torch.tensor(y)
    Gt=torch.tensor(np.hstack([G,G12]),dtype=torch.float32); k=Gt.shape[1]
    wF=torch.zeros(k,requires_grad=True); wM=torch.zeros(k,requires_grad=True); b=torch.zeros(k,7,requires_grad=True)
    a0=torch.ones(1,requires_grad=True); c0=torch.zeros(1,requires_grad=True); bb=torch.zeros(7,requires_grad=True)
    def lp(i):
        if kind=='Full': s=a0*Ft[i]+bb
        elif kind=='const': s=a0*Ft[i]+c0*Mt[i]+bb
        else: s=(a0+Gt[i]@wF)[:,None]*Ft[i]+(c0+Gt[i]@wM)[:,None]*Mt[i]+bb+Gt[i]@b
        return torch.log_softmax(s,1)
    ps=[a0,c0,bb,wF,wM,b]; opt=torch.optim.LBFGS(ps,max_iter=300,line_search_fn='strong_wolfe'); tri=torch.tensor(tr)
    def cl():
        opt.zero_grad(); l=-lp(tri).gather(1,yt[tri][:,None]).mean()+l2*((wF**2).sum()+(wM**2).sum()+(b**2).sum()); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): return lp(torch.tensor(te)).exp().numpy(), (a0+torch.eye(k)@wF).detach().numpy(), (c0+torch.eye(k)@wM).detach().numpy()
K=['Full','const','reliab']; P={k:np.zeros((N,7)) for k in K}; W=[]
for f in np.unique(fold):
    tr,te=np.where(fold!=f)[0],np.where(fold==f)[0]
    for k in K:
        P[k][te],wf,wm=fit(k,tr,te)
        if k=='reliab': W.append((wf,wm))
nll=lambda p,i: -np.log(np.clip(p[i,y[i]],1e-9,1)).mean()
uar=lambda p,i: 100*recall_score(y[i],p[i].argmax(1),average='macro',labels=range(7),zero_division=0)
al=np.arange(N)
for k in K: print(f"{k:<7} NLL {nll(P[k],al):.4f} UAR {uar(P[k],al):5.2f} | NLL by L-III bin", [round(nll(P[k],al[bins3==j]),3) for j in range(4)])
rng=np.random.default_rng(0); eps=np.unique(src); idx={e:np.where(src==e)[0] for e in eps}
def boot(fn,B=1000): return np.percentile([fn(np.concatenate([idx[e] for e in rng.choice(eps,len(eps))])) for _ in range(B)],[2.5,97.5])
for a_,b_ in [('Full','const'),('const','reliab')]:
    d=lambda i: nll(P[a_],i)-nll(P[b_],i); print(f"NLL({a_}) - NLL({b_}) {d(al):+.4f} {np.round(boot(d),4)}")
wf=np.mean([w[0] for w in W],0); wm=np.mean([w[1] for w in W],0)
print("weight on Full / minus-L per L-III bin (0,1,2-3,4+) then I+II bin (0,1-3,4+):"); print(np.round(wf,2)); print(np.round(wm,2))
