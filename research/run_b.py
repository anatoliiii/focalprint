import sys, time; sys.path.insert(0, __file__.rsplit('/', 1)[0])
from bench import *; from ridgecorr import RidgeCorr
t=time.time(); fingers=load_fingers(); print('fingers', len(fingers), 'load %.1fs'%(time.time()-t), flush=True)
keys=sorted(fingers)[::int(sys.argv[1]) if len(sys.argv)>1 else 1]
data=build({k:fingers[k] for k in keys}, n_enroll=12, n_query=3)
print('fingers used', len(data), flush=True)
for mo in [1000, 1800]:
    t=time.time(); gen,imp,res=evaluate(data, RidgeCorr(angles=range(-24,25,4), min_overlap=mo), impostor_fingers=10)
    report(f'ridgecorr min_overlap={mo}', gen, imp, res); print(' %.0fs'%(time.time()-t), flush=True)
