"""Cohort (T-)normalisation of the dense score, A/B protocol as eval2.py.

Each query is also scored against a cohort of unrelated fingers (open
datasets never used in training, emulated 64x80 windows). The decision uses
how far the claimed finger's score stands above that background:
  raw    = s
  znorm  = (s - mean(cohort)) / std(cohort)
  margin = s - max(cohort)
Thresholds are chosen on validation only."""
import os, random, re, sys, glob
import numpy as np, cv2
sys.argv = [sys.argv[0]] + sys.argv[1:]
_src = open(os.path.join(os.path.dirname(__file__), 'eval2.py')).read().split('val_gen, val_imp = scores')[0]
exec(_src)                                   # model, features(), score(), known/unseen/templates

COHORT_SETS = ('fvc2004_DB2_B', 'neuro_UareU_sample_DB')     # held out from training
N_COHORT, PER = 24, 10
root = os.environ.get('FP_DATASETS', os.path.expanduser('~/datasets/fingerprint/raw'))
rng = random.Random(11)


def windows(im, n):
    h, w = im.shape
    out = []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        cx, cy = rng.uniform(32, w - 32), rng.uniform(40, h - 40)
        R = cv2.getRotationMatrix2D((cx, cy), rng.uniform(-40, 40), 1.0)
        R[0, 2] += 32 - cx; R[1, 2] += 40 - cy
        win = cv2.warpAffine(im, R, (64, 80), borderValue=255)
        f = win.astype(np.float32)
        std = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - cv2.blur(f, (9, 9)) ** 2, 0))
        if (std > 10).mean() > 0.85:
            lo, hi = np.percentile(win, 2), np.percentile(win, 98)
            out.append(np.clip((win - lo) / (hi - lo + 1e-3) * 255, 0, 255).astype(np.uint8))
    return out


groups = {}
for ds in COHORT_SETS:
    for p in sorted(glob.glob(f'{root}/{ds}/**/*.tif', recursive=True)):
        parts = re.findall(r'\d+', os.path.basename(p))
        groups.setdefault((ds, parts[0], parts[1] if len(parts) == 3 else ''), []).append(p)
keys = rng.sample(sorted(groups), N_COHORT)
cohort = []
for k in keys:
    wins = []
    for p in groups[k]:
        im = cv2.resize(cv2.imread(p, 0), None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
        wins += windows(im, 3)
    cohort.append([features(w) for w in wins[:PER]])
print(f'cohort: {len(cohort)} unrelated fingers x {PER} windows', flush=True)


def scored(group, split):
    """Per query: claimed-finger scores (genuine + impostor claims) and its cohort scores."""
    gen, imp = [], []
    for f, touches in group.items():
        for s, q in touches:
            if s != split:
                continue
            c = np.array([score(q, t) for t in cohort])
            gen.append((score(q, templates[f]), c))
            imp += [(score(q, templates[o]), c) for o in fingers if o != f]
    return gen, imp


NORMS = {
    'raw': lambda s, c: s,
    'znorm': lambda s, c: (s - c.mean()) / (c.std() + 0.02),
    'margin': lambda s, c: s - c.max(),
}
val = scored(known, 'val')
tests = {'A known fingers, new touches': scored(known, 'test'), 'B unseen fingers': scored(unseen, 'test')}
for name, fn in NORMS.items():
    vg = np.array([fn(s, c) for s, c in val[0]]); vi = np.sort([fn(s, c) for s, c in val[1]])
    thr0, thr1 = vi[-1] + 1e-6, vi[int(np.ceil(len(vi) * 0.99)) - 1] + 1e-6
    print(f'== {name}: validation genuine median {np.median(vg):.3f}, impostor max {vi[-1]:.3f}')
    for tname, (g, i) in tests.items():
        g = np.array([fn(s, c) for s, c in g]); i = np.array([fn(s, c) for s, c in i])
        print(f'   {tname:30s} zero-val-FAR thr: FRR {np.mean(g < thr0) * 100:5.1f}% FAR {np.mean(i >= thr0) * 100:5.2f}%'
              f'   | val-FAR-1% thr: FRR {np.mean(g < thr1) * 100:5.1f}% FAR {np.mean(i >= thr1) * 100:5.2f}%')
