"""Two-scenario evaluation with a threshold chosen on validation only.

A: known fingers, later touches (everyday unlock).
B: fingers never seen in training (transfer to new skin).
Template = first 40 touches of each finger. Score of a query against a
finger = best dense aligned-descriptor similarity over its template touches
(RANSAC alignment from mutual descriptor matches, then mean cosine
similarity of the warped descriptor maps over the overlapping skin).
Thresholds come from validation impostors; test sets are scored once."""
import argparse, csv, os, sys
from collections import defaultdict
import cv2, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from fpnet import load, normalise

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--root', default='/dev/shm/fte4800')
ap.add_argument('--enroll', type=int, default=40)
ap.add_argument('--keypoints', type=int, default=400)
ap.add_argument('--min-overlap', type=int, default=60)
a = ap.parse_args()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
net = load(a.model, DEV).eval()
HH, WW = 40, 32


@torch.no_grad()
def features(frame):
    d, r = net(normalise(frame)[None, None].to(DEV))
    d, r = d[0].float(), r[0, 0].float()
    f = frame.astype(np.float32)
    std = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - cv2.blur(f, (9, 9)) ** 2, 0))
    skin = cv2.resize((std > 10).astype(np.float32), (WW, HH), interpolation=cv2.INTER_AREA) > 0.5
    score = r.cpu().numpy() * skin
    idx = np.argsort(score.ravel())[::-1][:a.keypoints]
    idx = idx[score.ravel()[idx] > 0.05]
    ys, xs = np.unravel_index(idx, (HH, WW))
    pts = np.stack([xs * 2 + 0.5, ys * 2 + 0.5], 1).astype(np.float32)
    desc = d.flatten(1)[:, torch.tensor(idx, device=DEV)].t().cpu().numpy()
    return pts, desc, d, torch.tensor(skin, device=DEV)


def align(q, t):
    """RANSAC transform mapping template pixels -> query pixels."""
    pq, dq, pt, dt = q[0], q[1], t[0], t[1]
    if len(pq) < 3 or len(pt) < 3:
        return None
    s = dq @ dt.T
    qa, ta = s.argmax(1), s.argmax(0)
    keep = [(i, j) for i, j in enumerate(qa) if ta[j] == i]
    if len(keep) < 3:
        return None
    src = np.float32([pt[j] for _, j in keep]); dst = np.float32([pq[i] for i, _ in keep])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000, confidence=0.999)
    if M is None or not 0.9 < np.hypot(M[0, 0], M[1, 0]) < 1.1:
        return None
    return M


def dense(q, t, M):
    """Mean cosine similarity of template descriptors warped into the query, over shared skin."""
    dq, sq, dt, st = q[2], q[3], t[2], t[3]
    Mi = cv2.invertAffineTransform(M)
    ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float32)
    px, py = xs * 2 + 0.5, ys * 2 + 0.5
    bx = Mi[0, 0] * px + Mi[0, 1] * py + Mi[0, 2]
    by = Mi[1, 0] * px + Mi[1, 1] * py + Mi[1, 2]
    grid = torch.tensor(np.stack([(bx - 0.5) / 2 / (WW - 1) * 2 - 1, (by - 0.5) / 2 / (HH - 1) * 2 - 1], -1),
                        device=DEV, dtype=torch.float32)[None]
    wt = F.normalize(F.grid_sample(dt[None], grid, align_corners=True)[0], dim=0)
    ws = F.grid_sample(st[None, None].float(), grid, align_corners=True, mode='nearest')[0, 0] > 0.5
    ov = sq & ws & (grid[0].abs() <= 1).all(-1)
    n = int(ov.sum())
    if n < a.min_overlap:
        return 0.0
    return float((dq * wt).sum(0)[ov].mean())


def score(q, template):
    best = 0.0
    for t in template:
        M = align(q, t)
        if M is not None:
            best = max(best, dense(q, t, M))
    return best


def load_set(name):
    rows = list(csv.DictReader(open(f'{a.root}/{name}/labels.csv')))
    rows.sort(key=lambda r: int(r['unix_ms']))
    by = defaultdict(list)
    for r in rows:
        by[r['finger']].append((r['split'], features(cv2.imread(f"{a.root}/{name}/{r['file']}", 0))))
    return by


known, unseen = load_set('known'), load_set('unseen')
fingers = {**known, **unseen}
templates = {f: [x for s, x in v if s == 'enroll'][:a.enroll] for f, v in fingers.items()}


def scores(group, split):
    gen, imp = [], []
    for f, touches in group.items():
        for s, q in touches:
            if s != split:
                continue
            gen.append(score(q, templates[f]))
            imp += [score(q, templates[o]) for o in fingers if o != f]
    return np.array(gen), np.array(imp)


val_gen, val_imp = scores(known, 'val')
srt = np.sort(val_imp)
thresholds = {
    'zero val FAR': srt[-1] + 1e-6,
    'val FAR 1%': srt[int(np.ceil(len(srt) * 0.99)) - 1] + 1e-6,
}
print(f'validation (known fingers): {len(val_gen)} genuine, {len(val_imp)} impostor; '
      f'impostor max {srt[-1]:.3f}, genuine median {np.median(val_gen):.3f}')
for name, (g, i) in {'A known fingers, new touches': scores(known, 'test'),
                     'B unseen fingers': scores(unseen, 'test')}.items():
    print(f'TEST {name}: {len(g)} genuine, {len(i)} impostor; genuine median {np.median(g):.3f}, impostor max {i.max():.3f}')
    for tname, thr in thresholds.items():
        print(f'   threshold {thr:.3f} ({tname}): FRR {np.mean(g < thr) * 100:5.1f}%   FAR {np.mean(i >= thr) * 100:5.2f}%')
