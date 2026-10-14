"""Small-sensor emulation bench: 500 dpi public datasets -> 64x80 windows at
our sensor's scale; template = many windows from impressions 1-4, queries =
random windows from impressions 5-8 (genuine) / other fingers (impostor).
Never writes images anywhere."""
import cv2, numpy as np, glob, os, re, sys, random
from collections import defaultdict

ROOT = os.path.expanduser('~/datasets/fingerprint/raw')
W, H = 64, 80
SCALE = 0.72          # 500 dpi -> ~360 dpi (ridge period 9 px -> 6.5 px)

def load_fingers(max_sets=None):
    fingers = defaultdict(list)
    for path in sorted(glob.glob(f'{ROOT}/*/**/*.tif', recursive=True)):
        ds = path.split('/')[len(ROOT.split('/'))]
        m = re.match(r'(\d+)_(\d+)', os.path.basename(path))
        if not m: continue
        fingers[(ds, m.group(1))].append(path)
    out = {}
    for k, ps in fingers.items():
        if len(ps) < 6: continue
        ims = []
        for p in sorted(ps)[:8]:
            im = cv2.imread(p, 0)
            im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            ims.append(im)
        out[k] = ims
    return out

def foreground(im, block=8, thr=12):
    h, w = im.shape
    m = np.zeros_like(im, bool)
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            if im[y:y+block, x:x+block].std() > thr:
                m[y:y+block, x:x+block] = True
    return m

def windows(im, n, rng, min_fg=0.85):
    fg = foreground(im)
    h, w = im.shape
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        ang = rng.uniform(-20, 20)
        cx, cy = rng.uniform(W/2, w - W/2), rng.uniform(H/2, h - H/2)
        R = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        R[0, 2] += W/2 - cx; R[1, 2] += H/2 - cy
        win = cv2.warpAffine(im, R, (W, H), borderValue=255)
        fm = cv2.warpAffine(fg.astype(np.uint8), R, (W, H)) > 0
        if fm.mean() >= min_fg:
            out.append(win)
    return out

def build(fingers, n_enroll=20, n_query=6, seed=1):
    rng = random.Random(seed)
    data = {}
    for k, ims in fingers.items():
        enroll = []
        for im in ims[:4]:
            enroll += windows(im, n_enroll // 4, rng)
        query = []
        for im in ims[4:]:
            query += windows(im, max(1, n_query // max(1, len(ims[4:]))), rng)
        if len(enroll) >= n_enroll // 2 and query:
            data[k] = (enroll, query[:n_query])
    return data

def evaluate(data, matcher, impostor_fingers=40, seed=2):
    """matcher: object with .template(list_of_windows) and .score(template, window)."""
    rng = random.Random(seed)
    keys = list(data)
    templates = {k: matcher.template(data[k][0]) for k in keys}
    gen, imp = [], []
    for k in keys:
        for q in data[k][1]:
            gen.append(matcher.score(templates[k], q))
        for o in rng.sample([x for x in keys if x != k], min(impostor_fingers, len(keys) - 1)):
            imp.append(matcher.score(templates[o], data[k][1][0]))
    gen, imp = np.array(gen), np.array(imp)
    imp_sorted = np.sort(imp)
    res = {}
    for far in [1e-2, 1e-3]:
        idx = min(len(imp_sorted) - 1, int(np.ceil(len(imp_sorted) * (1 - far))))
        thr = imp_sorted[idx]
        res[far] = (thr, float(np.mean(gen <= thr)))
    return gen, imp, res

def report(name, gen, imp, res):
    print(f'{name:34s} n_gen={len(gen)} n_imp={len(imp)} | imp max {imp.max():.3f} '
          + ' '.join(f'FRR@FAR{far:g}={frr*100:5.1f}% (thr {thr:.3f})' for far, (thr, frr) in res.items()), flush=True)
