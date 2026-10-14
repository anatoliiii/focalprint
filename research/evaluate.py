"""Evaluate an FpNet checkpoint: (1) leave-one-out on real FT9368 frames,
(2) emulated small-sensor verification on held-out open datasets."""
import sys, os, glob, re, random, argparse, itertools
from collections import defaultdict
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from fpnet import FpNet, normalise, skin_fraction, load

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--real', default='/dev/shm/fte4800-frames')
ap.add_argument('--keypoints', type=int, default=300)
ap.add_argument('--min-sim', type=float, default=0.75)
ap.add_argument('--ratio', type=float, default=0.95, help='best/second-best similarity margin')
ap.add_argument('--nms', type=int, default=1, help='half-res suppression radius')
a = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
net = load(a.model, DEV); net.eval()


@torch.no_grad()
def extract(frame):
    d, r = net(normalise(frame)[None, None].to(DEV))
    d, r = d[0].float().cpu(), r[0, 0].float().cpu()
    h, w = r.shape
    f = frame.astype(np.float32)
    std = cv2.blur(f * f, (9, 9)) - cv2.blur(f, (9, 9)) ** 2
    skin = cv2.resize((np.sqrt(np.maximum(std, 0)) > 10).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    score = r.numpy() * skin
    if a.nms:
        k = 2 * a.nms + 1
        peak = cv2.dilate(score.astype(np.float32), np.ones((k, k), np.uint8))
        score = np.where(score >= peak, score, 0)
    idx = np.argsort(score.ravel())[::-1][:a.keypoints]
    idx = idx[score.ravel()[idx] > 0.05]
    ys, xs = np.unravel_index(idx, (h, w))
    pts = np.stack([xs * 2 + 0.5, ys * 2 + 0.5], 1).astype(np.float32)
    return pts, d.flatten(1)[:, idx].t().numpy()


def inliers(A, B):
    (pa, da), (pb, db) = A, B
    if len(pa) < 3 or len(pb) < 3:
        return 0
    s = da @ db.T
    ab, ba = s.argmax(1), s.argmax(0)
    second = np.sort(s, 1)[:, -2] if s.shape[1] > 1 else np.zeros(len(s))
    keep = [(i, j) for i, j in enumerate(ab)
            if ba[j] == i and s[i, j] > a.min_sim and second[i] < a.ratio * s[i, j]]
    if len(keep) < 3:
        return 0
    qa = np.float32([pa[i] for i, _ in keep]); qb = np.float32([pb[j] for _, j in keep])
    M, inl = cv2.estimateAffinePartial2D(qa, qb, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000, confidence=0.999)
    if M is None or not 0.85 < np.hypot(M[0, 0], M[1, 0]) < 1.15:
        return 0
    return int(inl.sum())


def real_eval(enroll=20):
    """Template = first `enroll` touches of each finger (in time order);
    queries = its later touches (genuine) and every other finger's later
    touches (impostor). Without labels.csv: leave-one-out per file prefix."""
    labels = os.path.join(a.real, 'labels.csv')
    fr = defaultdict(list)
    if os.path.exists(labels):
        import csv
        for row in sorted(csv.DictReader(open(labels)), key=lambda r: int(r['unix_ms'])):
            if row['finger']:
                fr[row['finger']].append(cv2.imread(os.path.join(a.real, row['file']), 0))
    else:
        for p in sorted(glob.glob(f'{a.real}/*.pgm')):
            lab = os.path.basename(p).rsplit('-', 1)[0]
            if not lab.startswith(('_', 'empty')):
                fr[lab].append(cv2.imread(p, 0))
    feats = {k: [extract(f) for f in v] for k, v in fr.items()}
    tmpl = {k: v[:enroll] for k, v in feats.items()}
    gen, imp, per = [], [], {}
    for k, v in feats.items():
        g = [max(inliers(q, t) for t in tmpl[k]) for q in v[enroll:]]
        per[k] = g; gen += g
        for o in feats:
            if o != k:
                imp += [max(inliers(q, t) for t in tmpl[o]) for q in v[enroll:]]
    gen, imp = np.array(gen), np.array(imp)
    print(f'REAL: template = first {enroll} touches per finger; {len(gen)} genuine, {len(imp)} other-finger comparisons')
    for k, g in per.items():
        print(f'  {k:22s} genuine median {np.median(g):5.1f}  min {min(g):3d}')
    report_rates(gen, imp)


def report_rates(gen, imp):
    print(f'  other-finger inliers max {imp.max()}, p99 {np.percentile(imp, 99):.1f}, median {np.median(imp):.1f}')
    srt = np.sort(imp)
    for far in (1e-2, 1e-3):
        thr = srt[min(len(srt) - 1, int(np.ceil(len(srt) * (1 - far))))] + 1
        print(f'  FRR at FAR {far:g}: {np.mean(gen < thr) * 100:5.1f}%  (threshold {thr})')
    thr = imp.max() + 1
    print(f'  FRR at zero observed FAR: {np.mean(gen < thr) * 100:5.1f}%  (threshold {thr})')


def emulated_eval(n_enroll=12, n_query=4, impostors=15):
    root = os.environ.get('FP_DATASETS', os.path.expanduser('~/datasets/fingerprint/raw'))
    fingers = defaultdict(list)
    for ds in ('fvc2004_DB2_B', 'neuro_UareU_sample_DB'):
        for p in sorted(glob.glob(f'{root}/{ds}/**/*.tif', recursive=True)):
            parts = re.findall(r'\d+', os.path.basename(p))
            key = (ds, parts[0], parts[1] if len(parts) == 3 else '')
            fingers[key].append(p)
    rng = random.Random(7)
    data = {}
    for k, ps in fingers.items():
        ims = [cv2.resize(cv2.imread(p, 0), None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA) for p in sorted(ps)[:8]]
        if len(ims) < 6:
            continue
        enroll = [w for im in ims[:4] for w in windows(im, n_enroll // 4, rng)]
        query = [w for im in ims[4:] for w in windows(im, 1, rng)][:n_query]
        if len(enroll) >= n_enroll // 2 and query:
            data[k] = ([extract(w) for w in enroll], [extract(w) for w in query])
    keys = list(data)
    gen, imp = [], []
    for k in keys:
        for q in data[k][1]:
            gen.append(max(inliers(q, t) for t in data[k][0]))
        for o in rng.sample([x for x in keys if x != k], min(impostors, len(keys) - 1)):
            imp.append(max(inliers(data[k][1][0], t) for t in data[o][0]))
    gen, imp = np.array(gen), np.array(imp)
    print(f'EMULATED held-out: {len(keys)} fingers, {len(gen)} genuine, {len(imp)} impostor comparisons')
    report_rates(gen, imp)


def windows(im, n, rng, W=64, H=80):
    h, w = im.shape
    out = []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        ang = rng.uniform(-30, 30)
        cx, cy = rng.uniform(W / 2, w - W / 2), rng.uniform(H / 2, h - H / 2)
        R = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        R[0, 2] += W / 2 - cx; R[1, 2] += H / 2 - cy
        win = cv2.warpAffine(im, R, (W, H), borderValue=255)
        if skin_fraction(torch.tensor(win, dtype=torch.float32)) > 0.85:
            lo, hi = np.percentile(win, 2), np.percentile(win, 98)
            out.append(np.clip((win - lo) / (hi - lo + 1e-3) * 255, 0, 255).astype(np.uint8))
    return out


real_eval()
emulated_eval()
