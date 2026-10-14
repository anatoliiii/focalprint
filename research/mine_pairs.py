"""Stage B, step 1: find pairs of REAL touches of the same finger that the
current model aligns with high confidence. Each pair yields a similarity
transform (touch j pixels -> touch i pixels) used as ground truth for
fine-tuning. Only training touches are used (labels.csv of the train dir)."""
import argparse, csv, json, os, sys
from collections import defaultdict
import cv2, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from fpnet import load, normalise

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--train', default='/dev/shm/fte4800/train3')
ap.add_argument('--out', default='/dev/shm/fte4800/pairs.json')
ap.add_argument('--min-score', type=float, default=0.40)
ap.add_argument('--min-inliers', type=int, default=12)
ap.add_argument('--min-overlap', type=int, default=200, help='shared skin cells (of 1280)')
a = ap.parse_args()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
net = load(a.model, DEV).eval()
GH, GW = 40, 32


@torch.no_grad()
def touch(frame):
    d, r = net(normalise(frame)[None, None].to(DEV))
    d, r = d[0].float(), r[0, 0].float()
    f = frame.astype(np.float32)
    std = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - cv2.blur(f, (9, 9)) ** 2, 0))
    skin = torch.tensor(cv2.resize((std > 10).astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA) > 0.5, device=DEV)
    score = (r * skin).flatten()
    idx = torch.argsort(score, descending=True)[:400]
    idx = idx[score[idx] > 0.05]
    pts = np.stack([(idx % GW).cpu().numpy() * 2 + 0.5, (idx // GW).cpu().numpy() * 2 + 0.5], 1).astype(np.float32)
    return pts, d.flatten(1)[:, idx].t(), d, skin


def pair(ti, tj):
    """(score, overlap, inliers, M j->i) or None."""
    (pi, di, mi, si), (pj, dj, mj, sj) = ti, tj
    s = dj @ di.t()
    ji, ij = s.argmax(1), s.argmax(0)
    keep = (ij[ji] == torch.arange(len(ji), device=DEV)).nonzero()[:, 0]
    if len(keep) < 3:
        return None
    src = pj[keep.cpu().numpy()]; dst = pi[ji[keep].cpu().numpy()]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=4000, confidence=0.999)
    if M is None or not 0.92 < np.hypot(M[0, 0], M[1, 0]) < 1.08 or int(inl.sum()) < a.min_inliers:
        return None
    # dense check: touch j's map warped into touch i
    Mi = cv2.invertAffineTransform(M)                                   # i px -> j px
    ys, xs = torch.meshgrid(torch.arange(GH, device=DEV, dtype=torch.float32), torch.arange(GW, device=DEV, dtype=torch.float32), indexing='ij')
    px, py = xs * 2 + 0.5, ys * 2 + 0.5
    bx = Mi[0, 0] * px + Mi[0, 1] * py + Mi[0, 2]; by = Mi[1, 0] * px + Mi[1, 1] * py + Mi[1, 2]
    grid = torch.stack([(bx - 0.5) / 2 / (GW - 1) * 2 - 1, (by - 0.5) / 2 / (GH - 1) * 2 - 1], -1)[None].float()
    wd = F.normalize(F.grid_sample(mj[None], grid, align_corners=True)[0], dim=0)
    ws = F.grid_sample(sj[None, None].float(), grid, align_corners=True, mode='nearest')[0, 0] > 0.5
    ov = si & ws & (grid[0].abs() <= 1).all(-1)
    n = int(ov.sum())
    if n < a.min_overlap:
        return None
    return float((wd * mi).sum(0)[ov].mean()), n, int(inl.sum()), M


rows = list(csv.DictReader(open(f'{a.train}/labels.csv')))
files = sorted(r['file'] for r in rows)                  # same order as the trainer's frame stack
index = {f: i for i, f in enumerate(files)}
by = defaultdict(list)
for r in rows:
    by[r['finger']].append(r['file'])
pairs = []
for finger, fs in by.items():
    ts = {f: touch(cv2.imread(f'{a.train}/{f}', 0)) for f in fs}
    found = 0
    for x in range(len(fs)):
        for y in range(len(fs)):
            if x == y:
                continue
            res = pair(ts[fs[x]], ts[fs[y]])
            if res and res[0] >= a.min_score:
                s, n, k, M = res
                pairs.append({'i': index[fs[x]], 'j': index[fs[y]], 'M': np.asarray(M).tolist(), 'score': s, 'overlap': n})
                found += 1
    print(f'{finger:22s} {len(fs):3d} touches, {found:5d} confident ordered pairs', flush=True)
json.dump(pairs, open(a.out, 'w'))
print('total', len(pairs), '->', a.out)
