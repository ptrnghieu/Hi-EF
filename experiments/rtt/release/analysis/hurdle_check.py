# Hurdle / diagonal-effect check on RoleNet OOF (G14 Full, 10-seed mean) + A's clip-III label / recognizer (G23 RA_expr OOF).
# Usage: ARTIFACTS=<dir with g14/... and g23/...> python hurdle_check.py
import os
import numpy as np, torch
from sklearn.metrics import recall_score, roc_auc_score
torch.manual_seed(0)
a=np.load(os.path.join(os.environ.get('ARTIFACTS','.'),'g23/results-2/g23_oof_logits.npz'),allow_pickle=True); z=np.load(os.path.join(os.environ.get('ARTIFACTS','.'),'g14/g14_results/g14_oof_probs.npz'),allow_pickle=True)
y=z['y']; yA=a['yA']; fold=z['fold']; src=z['src']; N=len(y)
X=np.log(np.clip(z['Full'].mean(0),1e-6,1)); RA=a['RA_expr']; PI=np.exp(RA-RA.max(1,keepdims=True)); PI/=PI.sum(1,keepdims=True)
mir=(y==yA)
print(f"mirror rate {mir.mean():.3f} | RoleNet argmax = A's class {np.mean(X.argmax(1)==yA):.3f}")
print(f"AUC(RoleNet P(y=A_gold) -> mirror) {roc_auc_score(mir, X[np.arange(N),yA]):.3f}")
print(f"AUC(sum_a pi_a P(y=a) -> mirror)   {roc_auc_score(mir, (np.exp(X)*PI).sum(1)):.3f}")
E=torch.eye(7)
def fit_predict(kind, tr, te, l2=1e-3):
    Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y); At=torch.tensor(yA); Pt=torch.tensor(PI,dtype=torch.float32)
    W=torch.zeros(7,7,requires_grad=True); b=torch.zeros(7,requires_grad=True); W.data+=torch.eye(7)
    u=torch.zeros(8,requires_grad=True); c=torch.zeros(1,requires_grad=True); V=torch.zeros(7,7,requires_grad=True)
    ps=[W,b,u,c,V]
    def logp(idx):
        base=Xt[idx]@W.T+b
        if kind=='X': return torch.log_softmax(base,1)
        if kind=='X+RA': return torch.log_softmax(base+Pt[idx]@V.T,1)
        def diag(av):                       # hurdle == softmax + input-dependent bonus on A's class
            h=torch.cat([Xt[idx], base.gather(1,av[:,None])],1)
            return torch.log_softmax(base+(h@u+c)[:,None]*E[av],1)
        if kind=='hurdle_gold': return diag(At[idx])
        if kind=='hurdle_marg':             # A unknown: sum_a pi_a p(y | X, a)
            L=torch.stack([diag(torch.full((len(idx),),k)) for k in range(7)],1)   # [n,7a,7y]
            return torch.logsumexp(torch.log(Pt[idx]+1e-9)[:,:,None]+L,1)
    opt=torch.optim.LBFGS(ps,max_iter=300,line_search_fn='strong_wolfe')
    tri=torch.tensor(tr)
    def cl():
        opt.zero_grad(); l=-logp(tri).gather(1,yt[tri][:,None]).mean()+l2*sum((p**2).sum() for p in [W-torch.eye(7),u,V]); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): return logp(torch.tensor(te)).exp().numpy()
kinds=['X','X+RA','hurdle_marg','hurdle_gold']
OUT={k:np.zeros((N,7)) for k in kinds}
for f in np.unique(fold):
    tr,te=np.where(fold!=f)[0],np.where(fold==f)[0]
    for k in kinds: OUT[k][te]=fit_predict(k,tr,te)
nll=lambda P,i: -np.log(np.clip(P[i,y[i]],1e-9,1)).mean()
uar=lambda P,i: 100*recall_score(y[i],P[i].argmax(1),average='macro',labels=range(7),zero_division=0)
al=np.arange(N)
for k in kinds:
    print(f"{k:<12} NLL {nll(OUT[k],al):.4f} | UAR {uar(OUT[k],al):5.2f} | NLL mirror {nll(OUT[k],al[mir]):.3f} shift {nll(OUT[k],al[~mir]):.3f}")
rng=np.random.default_rng(0); eps=np.unique(src); idx={e:np.where(src==e)[0] for e in eps}
def boot(fn,B=1000):
    v=[fn(np.concatenate([idx[e] for e in rng.choice(eps,len(eps))])) for _ in range(B)]; return np.percentile(v,[2.5,97.5])
for a_,b_ in [('X','hurdle_marg'),('X+RA','hurdle_marg'),('X','hurdle_gold')]:
    d=lambda i: nll(OUT[a_],i)-nll(OUT[b_],i)
    print(f"NLL({a_}) - NLL({b_}) {d(al):+.4f} {np.round(boot(d),4)}")
