# Calibrated version (temperature fit on discovery folds, applied to held-out fold). usage: python pic_cal.py <results dir>
import os; _P=os.path.join(os.path.dirname(os.path.abspath(__file__)),'pic.py')
exec(open(_P).read().split("# ---------- 0.")[0])
exec(open(_P).read().split("# ---------- PIC machinery ----------")[1].split("pi=np.bincount")[0])
from scipy.optimize import minimize_scalar
def temper(Q,T): L=np.log(Q+1e-9)/T; L-=L.max(1,keepdims=True); E_=np.exp(L); return E_/E_.sum(1,keepdims=True)
def fitT(Q,yy): return minimize_scalar(lambda T: -np.log(temper(Q,T)[np.arange(len(yy)),yy]+1e-12).mean(),bounds=(0.2,10),method='bounded').x
def calibrated(Qfull):
    out=np.zeros_like(Qfull); Ts=[]
    for f in range(5):
        d,e=fold!=f,fold==f; T=fitT(Qfull[d],y[d]); Ts.append(T); out[e]=temper(Qfull[e],T)
    return out,Ts
C={}
for fam,src_ in [('RN',{k:v.mean(0) for k,v in RN.items()}),('LR',LR)]:
    for k,Q in src_.items():
        C[(fam,k)],Ts=calibrated(Q); print(f"{fam} {k}: fold temperatures {np.round(Ts,2)}")
pi=np.bincount(y,minlength=7)/N
print('\n== A (calibrated). spectrum and top-2 subspaces ==')
R={}
for key,Q in C.items():
    lam,W,g=pic(Q,pi); R[key]=(lam,W,g); g1=g[:,0]*np.sign(g[H,0])
    print(f"  {key[0]} {key[1]:5s} λ: "+' '.join(f"{l:.3f}" for l in lam)+" | g1: "+' '.join(f"{E[c][:3]} {g1[c]:+.2f}" for c in range(7)))
for a,b in [(('RN','I-II'),('LR','I-II')),(('RN','I-III'),('LR','I-III')),(('LR','I-II'),('LR','I-III'))]:
    print(f"   {a[0]} {a[1]} vs {b[0]} {b[1]}: top-2 subspace cosines {np.round(principal_cos(R[a][1],R[b][1]),2)}")
RBc={}
for key,Q in C.items():
    lam,W,g=pic(Q,pi,True); RBc[key]=(lam,W,g); r=g[:,0]*np.sign(g[4,0])*-1
    print(f"  ⟂h {key[0]} {key[1]:5s} λ: "+' '.join(f"{l:.3f}" for l in lam)+" | r1: "+' '.join(f"{E[c][:3]} {r[c]:+.2f}" for c in range(7)))
for a,b in [(('RN','I-III'),('LR','I-III')),(('RN','I-II'),('LR','I-II'))]:
    print(f"   ⟂h {a[0]} vs {b[0]} {a[1]}: residual top1 |cos| {abs(RBc[a][1][:,0]@RBc[b][1][:,0]):.2f}")

print('\n== C (calibrated). held-out fold, true labels ==')
def ev(fd,fe,which,remove_h=False,k=0):
    se={s:np.zeros(N) for s in ['I-II','I-III','c']}
    for f in range(5):
        d,e=fold!=f,fold==f; pif=np.bincount(y[d],minlength=7)/d.sum()
        g=h_vec(pif) if which=='h' else pic(C[(fd,'I-III')][d],pif,remove_h)[2][:,k]
        t=g[y[e]]; se['c'][e]=(t-g[y[d]].mean())**2
        for s in ['I-II','I-III']: se[s][e]=(C[(fe,s)][e]@g-t)**2
    return se
for fe in ['RN','LR']:
    for nm,fd,wh,rh,k in [('h (happy vs rest)','RN','h',False,0),('residual 1 from RN','RN','pic',True,0),('residual 1 from LR','LR','pic',True,0),('residual 2 from RN','RN','pic',True,1),('residual 2 from LR','LR','pic',True,1)]:
        se=ev(fd,fe,wh,rh,k); r2=lambda s,i:1-se[s][i].sum()/se['c'][i].sum(); dg=lambda i:se['I-II'][i].mean()-se['I-III'][i].mean()
        vr,vd=[],[]
        for _ in range(1000): i=boot_idx(); vr.append(r2('I-III',i)); vd.append(dg(i))
        a=np.arange(N)
        print(f"  eval {fe} | {nm:20s} R2 I-II {r2('I-II',a):+.3f}  I-III {r2('I-III',a):+.3f} [{np.percentile(vr,2.5):+.3f},{np.percentile(vr,97.5):+.3f}] | Δ_g {dg(a):+.4f} [{np.percentile(vd,2.5):+.4f},{np.percentile(vd,97.5):+.4f}]")
