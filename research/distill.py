"""Distil FpNet into a student that never sees private frames.

The teacher (trained with private FT9368 frames) is only run on public and
synthetic inputs; the student learns from its outputs there, plus the usual
geometric InfoNCE on the same public pairs. No private frame, pair or label
enters this process, so the student's weights carry no direct information
about them. `--no-teacher` trains the public-only baseline with the same data
and budget, for an honest comparison."""
import sys, os, glob, time, argparse
import cv2, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from fpnet import FpNet, load, save, info_nce
from gpu_aug import GpuPairs
import synth

ap = argparse.ArgumentParser()
ap.add_argument('--teacher', help='teacher checkpoint (.pt)')
ap.add_argument('--no-teacher', action='store_true',
                help='public-only baseline (the teacher, if given, only supplies the architecture)')
ap.add_argument('--iters', type=int, default=15000)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--synth', type=int, default=4000, help='synthetic fingerprints to add')
ap.add_argument('--tau', type=float, default=0.07)
ap.add_argument('--w-desc', type=float, default=1.0)
ap.add_argument('--w-rel', type=float, default=1.0)
ap.add_argument('--w-kl', type=float, default=1.0)
ap.add_argument('--amp', action='store_true')
ap.add_argument('--out', required=True)
a = ap.parse_args()
assert a.teacher, 'need --teacher (with --no-teacher it only supplies the architecture)'

ROOT = os.environ.get('FP_DATASETS', os.path.expanduser('~/datasets/fingerprint/raw'))
TEST_SETS = ('fvc2004_DB2_B', 'neuro_UareU_sample_DB')     # held out, as in train.py
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

imgs = []
for p in sorted(glob.glob(f'{ROOT}/*/**/*.tif', recursive=True)):
    if any(t in p for t in TEST_SETS):
        continue
    im = cv2.resize(cv2.imread(p, 0), None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
    if min(im.shape) >= 130:
        imgs.append(im)
n_public = len(imgs)
t0 = time.time()
for im in synth.generate(a.synth, dev, seed=7):
    imgs.append(cv2.resize(im, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA))
print(f'public impressions {n_public}, synthetic {a.synth} ({time.time() - t0:.0f}s), private frames 0', flush=True)

gen = GpuPairs(imgs, [], dev, real_share=0.0)
teacher = load(a.teacher, dev).eval()
cfg = teacher.config
if a.no_teacher:
    teacher = None
else:
    for p in teacher.parameters():
        p.requires_grad_(False)
student = FpNet(**cfg).to(dev)
print(f'device {dev}, student {cfg}, params {sum(p.numel() for p in student.parameters()):,}, '
      f'teacher {"none" if teacher is None else a.teacher}', flush=True)

opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler(enabled=a.amp)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)


def relational_kl(qs, ks, qt, kt, idx):
    """KL between teacher and student similarity distributions of the
    sampled view2 locations over all view1 locations."""
    B, D = qs.shape[:2]
    gather = lambda d: d.flatten(2).gather(2, idx[:, None, :].expand(B, D, idx.shape[1]))
    ls = torch.bmm(gather(qs).transpose(1, 2), ks.flatten(2)) / a.tau
    lt = torch.bmm(gather(qt).transpose(1, 2), kt.flatten(2)) / a.tau
    return F.kl_div(F.log_softmax(ls, -1), F.log_softmax(lt, -1), log_target=True, reduction='batchmean') / idx.shape[1]


t0 = time.time(); ema = None
for it in range(1, a.iters + 1):
    x1, x2, m1, m2, t21 = gen.batch(a.batch)
    with torch.autocast(dev, dtype=torch.float16, enabled=a.amp):
        d1, r1 = student(x1); d2, r2 = student(x2)
        if teacher is not None:
            with torch.no_grad():
                t1, tr1 = teacher(x1); t2, tr2 = teacher(x2)
    d1, d2, r1, r2 = d1.float(), d2.float(), r1.float(), r2.float()
    loss, acc, idx, correct, valid = info_nce(d1, d2, t21, m1, m2)
    rel = F.binary_cross_entropy(r2.flatten(1).gather(1, idx).clamp(1e-6, 1 - 1e-6), correct, reduction='none')
    loss = loss + 0.5 * (rel * valid).sum() / valid.sum().clamp(min=1)
    kd = torch.zeros((), device=dev)
    if teacher is not None:
        t1, t2, tr1, tr2 = t1.float(), t2.float(), tr1.float(), tr2.float()
        kd_desc = (2 - (d1 * t1).sum(1).mean() - (d2 * t2).sum(1).mean()) / 2
        kd_rel = (F.mse_loss(r1, tr1) + F.mse_loss(r2, tr2)) / 2
        kd_kl = relational_kl(d2, d1, t2, t1, idx)
        kd = a.w_desc * kd_desc + a.w_rel * kd_rel + a.w_kl * kd_kl
        loss = loss + kd
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    ema = acc.item() if ema is None else 0.98 * ema + 0.02 * acc.item()
    if it % 100 == 0 or it == 1:
        print(f'it {it:5d} loss {loss.item():.3f} kd {kd.item():.3f} match-acc {ema:.3f} {time.time() - t0:.0f}s', flush=True)
        save(student, a.out)
save(student, a.out)
print('saved', a.out)
