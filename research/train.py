"""Self-supervised training of FpNet on open datasets (+ real frames)."""
import sys, os, glob, time, random, argparse
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from fpnet import *

ap = argparse.ArgumentParser()
ap.add_argument('--iters', type=int, default=3000)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--real', default='/dev/shm/fte4800-frames', help='dir of real *.pgm frames')
ap.add_argument('--real-share', type=float, default=0.5)
ap.add_argument('--out', default='/dev/shm/fte4800-models/fpnet.pt')
ap.add_argument('--resume')
ap.add_argument('--width', type=float, default=1)
ap.add_argument('--dim', type=int, default=DIM)
ap.add_argument('--context', type=int, default=2, help='3x3 layers at 1/8 resolution (receptive field)')
ap.add_argument('--workers', type=int, default=3)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--gpu-aug', action='store_true', help='generate pairs on the GPU')
ap.add_argument('--amp', action='store_true', help='fp16 autocast (RDNA2 packed math)')
ap.add_argument('--pairs', help='stage B: JSON of real touch pairs from mine_pairs.py')
ap.add_argument('--pair-share', type=float, default=0.4)
a = ap.parse_args()

ROOT = os.environ.get('FP_DATASETS', os.path.expanduser('~/datasets/fingerprint/raw'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_SETS = ('fvc2004_DB2_B', 'neuro_UareU_sample_DB')     # held out for evaluation
open_imgs = []
for p in sorted(glob.glob(f'{ROOT}/*/**/*.tif', recursive=True)):
    if any(t in p for t in TEST_SETS):
        continue
    im = cv2.imread(p, 0)
    im = cv2.resize(im, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
    if min(im.shape) >= 130:
        open_imgs.append(torch.tensor(im, dtype=torch.float32))
real = [torch.tensor(cv2.imread(p, 0), dtype=torch.float32) for p in sorted(glob.glob(f'{a.real}/*.pgm'))
        if not os.path.basename(p).startswith(('_', 'empty'))] if a.real else []
print(f'open impressions {len(open_imgs)}, real frames {len(real)}', flush=True)

class Pairs(torch.utils.data.IterableDataset):
    """Endless stream of synthetic training pairs; one RNG per worker."""
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1000 + (info.id if info else 0) + int(time.time()))
        while True:
            if real and rng.random() < a.real_share:
                v1, v2, m1, m2, t = make_pair(rng.choice(real), rng, max_angle=60, max_shift=16, real=True)
            else:
                v1, v2, m1, m2, t = make_pair(rng.choice(open_imgs), rng)
            yield normalise(v1)[None], normalise(v2)[None], m1, m2, t


if a.gpu_aug:
    from gpu_aug import GpuPairs
    import json
    pairs = json.load(open(a.pairs)) if a.pairs else None
    gen = GpuPairs([i.numpy().astype('uint8') for i in open_imgs], [r.numpy().astype('uint8') for r in real], dev, a.real_share,
                   pairs=pairs, pair_share=a.pair_share)
    if pairs:
        print(f'stage B: {len(pairs)} real pairs, share {a.pair_share}', flush=True)
    loader = iter(lambda: gen.batch(a.batch), None)
else:
  loader = iter(torch.utils.data.DataLoader(
    Pairs(), batch_size=a.batch, num_workers=a.workers, persistent_workers=True, prefetch_factor=4,
    # tiny tensors: intra-op threading in every worker only causes contention
    worker_init_fn=lambda _: torch.set_num_threads(1)))
torch.set_num_threads(4)
net = load(a.resume, dev) if a.resume else FpNet(dim=a.dim, width=a.width, context=a.context).to(dev)
print(f'device {dev}, params {sum(p.numel() for p in net.parameters()):,}', flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler(enabled=a.amp)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
t0 = time.time(); ema = None
for it in range(1, a.iters + 1):
    x1, x2, m1s, m2s, ts = next(loader)
    x1, x2 = x1.to(dev, non_blocking=True), x2.to(dev, non_blocking=True)
    m1s, m2s, ts = m1s.to(dev), m2s.to(dev), ts.to(dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=a.amp):
        d1, r1 = net(x1); d2, r2 = net(x2)
    d1, d2, r2 = d1.float(), d2.float(), r2.float()
    loss, acc, idx, correct, valid = info_nce(d1, d2, ts, m1s, m2s)
    rel = F.binary_cross_entropy(r2.flatten(1).gather(1, idx).clamp(1e-6, 1 - 1e-6), correct, reduction='none')
    loss = loss + 0.5 * (rel * valid).sum() / valid.sum().clamp(min=1)
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    acc = acc.item()
    ema = acc if ema is None else 0.98 * ema + 0.02 * acc
    if it % 100 == 0 or it == 1:
        print(f'it {it:5d} loss {loss.item():.3f} match-acc {ema:.3f} {time.time() - t0:.0f}s', flush=True)
        save(net, a.out)
save(net, a.out)
print('saved', a.out)
