import numpy as np, itertools, warnings, sys, os
# usage: python pic.py <dir with g19_oof_probs.npz and g19_lr_oof_probs.npz>  (exploratory, train+val OOF only)
RES=sys.argv[1] if len(sys.argv)>1 else 'results'
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
E=['angry','disgust','fear','happy','neutral','sad','surprise']; H=3
z=np.load(os.path.join(RES,'g19_oof_probs.npz'),allow_pickle=True); zl=np.load(os.path.join(RES,'g19_lr_oof_probs.npz'))
y=z['y']; src=z['src']; fold=z['fold']; N=len(y)
RN={'I-II':z['I_II'],'I-III':z['I_III']}; LR={'I-II':zl['I_II'],'I-III':zl['I_III']}
eps=np.unique(src); G=[np.where(src==e)[0] for e in eps]; rng=np.random.default_rng(0)
def boot_idx(): return np.concatenate([G[k] for k in rng.integers(0,len(G),len(G))])

# ---------- 0. happy pairs vs other pairs: direct test of the difference in gain ----------
ELIG=[(a,b) for a,b in itertools.combinations(range(7),2) if 2 not in (a,b)]
HP=[p for p in ELIG if H in p]; OP=[p for p in ELIG if H not in p]
def pauc(P,a,b,idx):
    m=idx[np.isin(y[idx],[a,b])]; return roc_auc_score(y[m]==a,np.log(P[m,a]+1e-9)-np.log(P[m,b]+1e-9))
def D(P12,P13,idx):
    d={p:pauc(P13,*p,idx)-pauc(P12,*p,idx) for p in ELIG}
    return np.mean([d[p] for p in HP])-np.mean([d[p] for p in OP])
print('== 0. happy-pair gain minus other-pair gain (Δ = AUC I-III − AUC I-II) ==')
for fam in ['RN','LR']:
    full=np.arange(N)
    if fam=='RN': est=D(RN['I-II'].mean(0),RN['I-III'].mean(0),full)
    else: est=D(LR['I-II'],LR['I-III'],full)
    v=[]
    for _ in range(500):
        i=boot_idx()
        if fam=='RN':
            s=rng.integers(0,10,10); v.append(D(RN['I-II'][s].mean(0),RN['I-III'][s].mean(0),i))
        else: v.append(D(LR['I-II'],LR['I-III'],i))
    print(f"  {fam}: D = {est:+.3f}  [{np.percentile(v,2.5):+.3f}, {np.percentile(v,97.5):+.3f}]  (5 happy pairs vs 10 other eligible pairs)")

# ---------- PIC machinery ----------
def h_vec(pi): e=np.zeros(7); e[H]=1; return (e-pi[H])/np.sqrt(pi[H]*(1-pi[H]))
def pic(Q,pi,remove_h=False):
    C=Q.T@Q/len(Q); Dm=np.diag(1/np.sqrt(pi)); M=Dm@C@Dm
    cons=[np.sqrt(pi)]
    if remove_h: cons.append(np.sqrt(pi)*h_vec(pi))
    A=np.linalg.qr(np.stack(cons,1))[0]
    U,_,_=np.linalg.svd(np.eye(7)-A@A.T); B=U[:,:7-len(cons)]
    lam,V=np.linalg.eigh(B.T@M@B); o=np.argsort(-lam); lam,V=lam[o],V[:,o]
    W=B@V                     # directions in sqrt(pi)-weighted space (orthonormal)
    g=W/np.sqrt(pi)[:,None]   # label functions with E_pi g=0, E_pi g^2=1
    return lam,W,g
def principal_cos(Wa,Wb,k=2):
    return np.linalg.svd(Wa[:,:k].T@Wb[:,:k],compute_uv=False)
pi=np.bincount(y,minlength=7)/N
P={('RN',k):RN[k].mean(0) for k in RN}; P.update({('LR',k):LR[k] for k in LR})

print('\n== A. spectrum (eigenvalues = rho_X(g_k), model-implied) ==')
RES={}
for key,Q in P.items():
    lam,W,g=pic(Q,pi); RES[key]=(lam,W,g)
    bl=[]
    for _ in range(300):
        i=boot_idx()
        Qb=(RN[key[1]][rng.integers(0,10,10)].mean(0) if key[0]=='RN' else Q)[i]
        bl.append(pic(Qb,np.bincount(y[i],minlength=7)/len(i))[0])
    lo,hi=np.percentile(bl,[2.5,97.5],axis=0)
    print(f"  {key[0]} {key[1]:5s} λ: "+'  '.join(f"{l:.3f}[{a:.3f},{b:.3f}]" for l,a,b in zip(lam,lo,hi)))
    g1=g[:,0]*np.sign(g[H,0]); print(f"        g1 loadings: "+' '.join(f"{E[c][:3]} {g1[c]:+.2f}" for c in range(7))+f" | cos(g1,h) = {abs(W[:,0]@(np.sqrt(pi)*h_vec(pi))):.2f}")
print('\n  top-2 subspace similarity (cosines of principal angles):')
for a,b in [(('RN','I-II'),('LR','I-II')),(('RN','I-III'),('LR','I-III')),(('RN','I-II'),('RN','I-III')),(('LR','I-II'),('LR','I-III'))]:
    print(f"   {a[0]} {a[1]} vs {b[0]} {b[1]}: top1 |cos| {abs(RES[a][1][:,0]@RES[b][1][:,0]):.2f} | top-2 subspace {np.round(principal_cos(RES[a][1],RES[b][1]),2)}")

print('\n== B. directions orthogonal to happy-vs-rest ==')
RB={}
for key,Q in P.items():
    lam,W,g=pic(Q,pi,remove_h=True); RB[key]=(lam,W,g)
    g1=g[:,0]*np.sign(g[0,0] if abs(g[0,0])>1e-6 else 1)
    print(f"  {key[0]} {key[1]:5s} residual λ: "+' '.join(f"{l:.3f}" for l in lam)+" | r1: "+' '.join(f"{E[c][:3]} {g1[c]:+.2f}" for c in range(7)))
for a,b in [(('RN','I-II'),('LR','I-II')),(('RN','I-III'),('LR','I-III')),(('RN','I-II'),('RN','I-III'))]:
    print(f"   {a[0]} {a[1]} vs {b[0]} {b[1]}: residual top1 |cos| {abs(RB[a][1][:,0]@RB[b][1][:,0]):.2f} | top-2 {np.round(principal_cos(RB[a][1],RB[b][1]),2)}")

print('\n== C. held-out-fold check with true labels (directions found on the other 4 folds, from I-III probs) ==')
def evaluate(fam_dir, fam_eval, which, kidx, remove_h, seeds=None):
    se={k:np.zeros(N) for k in ['I-II','I-III','const']}
    for f in range(5):
        dsc,ev=fold!=f,fold==f
        pif=np.bincount(y[dsc],minlength=7)/dsc.sum()
        Qd=(RN['I-III'].mean(0) if fam_dir=='RN' else LR['I-III'])[dsc]
        if which=='h': g=h_vec(pif)
        else: g=pic(Qd,pif,remove_h)[2][:,kidx]
        tgt=g[y[ev]]; se['const'][ev]=(tgt-(g[y[dsc]]).mean())**2
        for s in ['I-II','I-III']:
            Qe=(RN[s][seeds].mean(0) if fam_eval=='RN' else LR[s])[ev] if seeds is not None or fam_eval=='LR' else RN[s].mean(0)[ev]
            se[s][ev]=(Qe@g-tgt)**2
    return se
def summarize(name,fam_dir,fam_eval,which,kidx=0,remove_h=False):
    se=evaluate(fam_dir,fam_eval,which,kidx,remove_h,None if fam_eval=='LR' else np.arange(10))
    r2=lambda s,i: 1-se[s][i].sum()/se['const'][i].sum()
    dg=lambda i: se['I-II'][i].mean()-se['I-III'][i].mean()
    allr=np.arange(N); vr,vd=[],[]
    for _ in range(300):
        i=boot_idx()
        if fam_eval=='RN':
            sb=evaluate(fam_dir,'RN',which,kidx,remove_h,rng.integers(0,10,10)); 
            vr.append(1-sb['I-III'][i].sum()/sb['const'][i].sum()); vd.append(sb['I-II'][i].mean()-sb['I-III'][i].mean())
        else:
            vr.append(r2('I-III',i)); vd.append(dg(i))
    print(f"  {name:38s} R2(I-II) {r2('I-II',allr):+.3f} | R2(I-III) {r2('I-III',allr):+.3f} [{np.percentile(vr,2.5):+.3f},{np.percentile(vr,97.5):+.3f}] | Δ_g {dg(allr):+.4f} [{np.percentile(vd,2.5):+.4f},{np.percentile(vd,97.5):+.4f}]")
for fe in ['RN','LR']:
    summarize(f'h (happy vs rest), eval {fe}','RN',fe,'h')
    for fd in ['RN','LR']:
        summarize(f'top-1 full dir from {fd}, eval {fe}',fd,fe,'pic',0,False)
        summarize(f'residual dir 1 (⟂h) from {fd}, eval {fe}',fd,fe,'pic',0,True)
        summarize(f'residual dir 2 (⟂h) from {fd}, eval {fe}',fd,fe,'pic',1,True)
