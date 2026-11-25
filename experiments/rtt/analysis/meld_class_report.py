import sys, numpy as np
z = np.load(sys.argv[1] + '/meld_test_all.npz', allow_pickle=True); y = z['y']
EMO = ['angry','disgust','fear','happy','neutral','sad','surprise']
def wu(p): return (p == y).mean()*100, np.mean([(p[y==c]==c).mean()*100 for c in range(7) if (y==c).any()])
print(f'{"model":<18} UAR   WAR    k  | recall % by class ' + ' '.join(f'{e[:4]:>5}' for e in EMO) + ' | pred neutral %')
for k in [f'row{r}' for r in range(5, 12)] + ['PartiSet', 'PaperBest_nobody', 'zero_shot']:
    p = z[k].mean(0).argmax(1); w, u = wu(p)
    rec = [(p[y==c]==c).mean()*100 for c in range(7)]
    print(f'{k:<18} {u:5.2f} {w:6.2f} {round(w*len(y)/100):4d} | ' + ' '*18 + ' '.join(f'{r:5.1f}' for r in rec) + f' | {(p==4).mean()*100:5.1f}')
print('always-neutral: WAR %.2f UAR %.2f' % wu(np.full(len(y), 4)))
