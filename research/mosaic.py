"""Finger mosaics: the touches of an enrolment are stitched into one dense
descriptor map of the finger pad, and touches are matched against the
whole map. Evaluated with the same A/B protocol as eval2.py."""
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
ap.add_argument('--place', type=float, default=0.22, help='dense score needed to stitch a touch')
ap.add_argument('--fusion', choices=['mean', 'max'], default='max', help='mean descriptors, or per-cell best touch')
ap.add_argument('--canvas', type=int, nargs=2, default=(160, 128), help='cells (h w), half resolution')
a = ap.parse_args()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
net = load(a.model, DEV).eval()
GH, GW = 40, 32                 # touch grid (half resolution)
CH, CW = a.canvas
MIN_OVERLAP = 60


class Touch:
    @torch.no_grad()
    def __init__(self, frame):
        d, r = net(normalise(frame)[None, None].to(DEV))
        self.d, self.r = d[0].float(), r[0, 0].float()
        f = frame.astype(np.float32)
        std = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - cv2.blur(f, (9, 9)) ** 2, 0))
        self.skin = torch.tensor(cv2.resize((std > 10).astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA) > 0.5, device=DEV)
        score = (self.r * self.skin).flatten()
        idx = torch.argsort(score, descending=True)[:a.keypoints]
        idx = idx[score[idx] > 0.05]
        self.pts = np.stack([(idx % GW).cpu().numpy() * 2 + 0.5, (idx // GW).cpu().numpy() * 2 + 0.5], 1).astype(np.float32)
        self.desc = self.d.flatten(1)[:, idx].t()               # K x C


def grid_pixels(h, w):
    ys, xs = torch.meshgrid(torch.arange(h, device=DEV, dtype=torch.float32), torch.arange(w, device=DEV, dtype=torch.float32), indexing='ij')
    return xs * 2 + 0.5, ys * 2 + 0.5


def warp_sample(maps, M_inv, out_h, out_w, src_h, src_w, mode='bilinear'):
    """Sample `maps` (C,src_h,src_w) at the source positions of an out_h x out_w
    cell grid, given M_inv mapping output pixels -> source pixels."""
    px, py = grid_pixels(out_h, out_w)
    sx = M_inv[0, 0] * px + M_inv[0, 1] * py + M_inv[0, 2]
    sy = M_inv[1, 0] * px + M_inv[1, 1] * py + M_inv[1, 2]
    gx = (sx - 0.5) / 2 / (src_w - 1) * 2 - 1
    gy = (sy - 0.5) / 2 / (src_h - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None].float()
    out = F.grid_sample(maps[None], grid, mode=mode, align_corners=True)[0]
    inside = (grid[0].abs() <= 1).all(-1)
    return out, inside


def ransac(src_pts, src_desc, dst_pts, dst_desc):
    """Similarity transform src pixels -> dst pixels from mutual NN pairs."""
    if len(src_pts) < 3 or len(dst_pts) < 3:
        return None
    s = src_desc @ dst_desc.t()
    sd, ds = s.argmax(1), s.argmax(0)
    i = torch.arange(len(sd), device=DEV)
    keep = (ds[sd] == i).nonzero()[:, 0]
    if len(keep) < 3:
        return None
    src = src_pts[keep.cpu().numpy()]
    dst = dst_pts[sd[keep].cpu().numpy()]
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000, confidence=0.999)
    if M is None or not 0.9 < np.hypot(M[0, 0], M[1, 0]) < 1.1:
        return None
    return M


class Mosaic:
    def __init__(self, first):
        self.sum = torch.zeros(first.d.shape[0], CH, CW, device=DEV)
        self.w = torch.zeros(CH, CW, device=DEV)
        self.placed = []                                      # (touch, touch px -> canvas px)
        M = np.float32([[1, 0, CW - GW], [0, 1, CH - GH]])      # centre of the canvas
        self.add(first, M)

    def add(self, t, M):
        """Accumulate touch `t` placed by M (touch pixels -> canvas pixels)."""
        self.placed.append((t, np.float32(M)))
        Mi = torch.tensor(cv2.invertAffineTransform(M), device=DEV)
        stack = torch.cat([t.d, (t.r * t.skin)[None]], 0)
        sampled, inside = warp_sample(stack, Mi, CH, CW, GH, GW)
        weight = sampled[-1] * inside
        self.sum += sampled[:-1] * weight
        self.w += weight
        self._keys = None

    def keys(self):
        if self._keys is None:
            valid = (self.w > 0.5).flatten()
            idx = valid.nonzero()[:, 0]
            desc = F.normalize(self.sum.flatten(1)[:, idx], dim=0).t()
            pts = np.stack([(idx % CW).cpu().numpy() * 2 + 0.5, (idx // CW).cpu().numpy() * 2 + 0.5], 1).astype(np.float32)
            self._keys = (pts, desc)
        return self._keys

    def score(self, t, M_m2t):
        if a.fusion == 'max':
            return self.score_max(t, M_m2t)
        return self.score_mean(t, M_m2t)

    def score_max(self, t, M_m2t):
        """Every placed touch warped into the query; per query cell the best
        similarity among the touches covering it; mean over covered skin."""
        best = torch.full((GH, GW), -1.0, device=DEV)
        covered = torch.zeros(GH, GW, dtype=torch.bool, device=DEV)
        A = np.vstack([M_m2t, [0, 0, 1]])
        for tt, Mt in self.placed:
            M_t2q = (A @ np.vstack([Mt, [0, 0, 1]]))[:2]          # touch -> canvas -> query
            Mi = torch.tensor(cv2.invertAffineTransform(np.float32(M_t2q)), device=DEV)
            stack = torch.cat([tt.d, tt.skin[None].float()], 0)
            sampled, inside = warp_sample(stack, Mi, GH, GW, GH, GW)
            ok = inside & (sampled[-1] > 0.5) & t.skin
            sim = (F.normalize(sampled[:-1], dim=0) * t.d).sum(0)
            best = torch.where(ok & (sim > best), sim, best)
            covered |= ok
        n = int(covered.sum())
        return float(best[covered].mean()) if n >= MIN_OVERLAP else 0.0

    def score_mean(self, t, M_m2t):
        """Mean similarity of the mosaic (placed into the touch by M_m2t) with the touch."""
        Mi = torch.tensor(cv2.invertAffineTransform(M_m2t), device=DEV)       # touch px -> mosaic px
        stack = torch.cat([self.sum, self.w[None]], 0)
        sampled, inside = warp_sample(stack, Mi, GH, GW, CH, CW)
        w = sampled[-1]
        ov = t.skin & inside & (w > 0.5)
        n = int(ov.sum())
        if n < MIN_OVERLAP:
            return 0.0
        md = F.normalize(sampled[:-1], dim=0)
        return float((md * t.d).sum(0)[ov].mean())

    def match(self, t):
        pts, desc = self.keys()
        M = ransac(pts, desc, t.pts, t.desc)                    # mosaic -> touch
        return (self.score(t, M), M) if M is not None else (0.0, None)


def build(touches):
    """Greedy multi-pass stitching; leftovers seed further mosaics."""
    pending = list(touches)
    mosaics = []
    while pending:
        m = Mosaic(pending.pop(0))
        progress = True
        while progress:
            progress, rest = False, []
            for t in pending:
                s, M = m.match(t)
                if s >= a.place:
                    m.add(t, cv2.invertAffineTransform(M))         # touch -> mosaic
                    progress = True
                else:
                    rest.append(t)
            pending = rest
        mosaics.append(m)
    return mosaics


def score(t, mosaics):
    return max((m.match(t)[0] for m in mosaics), default=0.0)


def load_set(name):
    rows = sorted(csv.DictReader(open(f'{a.root}/{name}/labels.csv')), key=lambda r: int(r['unix_ms']))
    by = defaultdict(list)
    for r in rows:
        by[r['finger']].append((r['split'], Touch(cv2.imread(f"{a.root}/{name}/{r['file']}", 0))))
    return by


known, unseen = load_set('known'), load_set('unseen')
fingers = {**known, **unseen}
templates = {}
for f, v in fingers.items():
    ms = build([x for s, x in v if s == 'enroll'][:a.enroll])
    templates[f] = ms
    sizes = sorted((int((m.w > 0.5).sum()) for m in ms), reverse=True)
    print(f'{f:22s} {len(ms):2d} mosaic(s); covered cells {sizes[:4]} (one touch ≈ {GH * GW})')


def scores(group, split):
    gen, imp = [], []
    for f, touches in group.items():
        for s, q in touches:
            if s == split:
                gen.append(score(q, templates[f]))
                imp += [score(q, templates[o]) for o in fingers if o != f]
    return np.array(gen), np.array(imp)


val_gen, val_imp = scores(known, 'val')
srt = np.sort(val_imp)
thresholds = {'zero val FAR': srt[-1] + 1e-6, 'val FAR 1%': srt[int(np.ceil(len(srt) * 0.99)) - 1] + 1e-6}
print(f'validation: genuine median {np.median(val_gen):.3f}, impostor max {srt[-1]:.3f}')
for name, (g, i) in {'A known fingers, new touches': scores(known, 'test'), 'B unseen fingers': scores(unseen, 'test')}.items():
    print(f'TEST {name}: {len(g)} genuine, {len(i)} impostor; genuine median {np.median(g):.3f}, impostor max {i.max():.3f}')
    for tname, thr in thresholds.items():
        print(f'   threshold {thr:.3f} ({tname}): FRR {np.mean(g < thr) * 100:5.1f}%   FAR {np.mean(i >= thr) * 100:5.2f}%')
