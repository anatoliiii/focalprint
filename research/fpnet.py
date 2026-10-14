"""FpNet: dense descriptors + reliability for 64x80 FT9368 frames, trained
self-supervised from synthetic warps (known correspondences)."""
import math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

H, W = 80, 64          # sensor frame
DIM = 64               # descriptor size


def conv(i, o, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, padding=dil, dilation=dil, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class FpNet(nn.Module):
    """Input 1xHxW (normalised), output descriptors DIMx(H/2)x(W/2)
    (L2-normalised) and reliability 1x(H/2)x(W/2) in [0,1].

    Pyramid (U-Net-lite) without dilated convolutions (MIOpen runs those
    ~10x slower on RDNA2): 1/2 resolution for precise positions, 1/4 for
    ridge structure, 1/8 for wide context. `context` = number of 3x3
    layers at 1/8, each adding ~16 px of receptive field:
    1 ≈ 50 px, 2 ≈ 70 px, 4 ≈ 100 px, 6 ≈ 130 px."""
    def __init__(self, dim=DIM, width=1, context=2):
        super().__init__()
        # channel counts rounded to multiples of 32: wave32 on RDNA2
        c = lambda n: max(32, int(round(n * width / 32)) * 32)
        self.config = {'dim': dim, 'width': width, 'context': context}
        self.fine = nn.Sequential(conv(1, c(16)), conv(c(16), c(32), stride=2), conv(c(32), c(32)))
        self.mid = nn.Sequential(conv(c(32), c(64), stride=2), conv(c(64), c(64)), conv(c(64), c(64)))
        self.coarse = nn.Sequential(conv(c(64), c(96), stride=2), *[conv(c(96), c(96)) for _ in range(context)])
        self.up_mid = nn.Sequential(nn.Conv2d(c(64) + c(96), c(96), 1, bias=False), nn.BatchNorm2d(c(96)),
                                    nn.ReLU(inplace=True), conv(c(96), c(96)))
        self.fuse = nn.Sequential(nn.Conv2d(c(32) + c(96), c(64), 1, bias=False), nn.BatchNorm2d(c(64)),
                                  nn.ReLU(inplace=True), conv(c(64), c(64)))
        self.desc = nn.Conv2d(c(64), dim, 1)
        self.rel = nn.Conv2d(c(64), 1, 1)

    def forward(self, x):
        h = self.fine(x)                      # 1/2
        m = self.mid(h)                       # 1/4
        k = self.coarse(m)                    # 1/8
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        m = self.up_mid(torch.cat([m, up(k, m)], 1))
        f = self.fuse(torch.cat([h, up(m, h)], 1))
        return F.normalize(self.desc(f), dim=1), torch.sigmoid(self.rel(f))


def normalise(img):
    """uint8/float HxW -> zero-mean unit-variance float tensor, as the
    daemon will do before inference."""
    x = torch.as_tensor(img, dtype=torch.float32)
    return (x - x.mean()) / (x.std() + 1e-3)


# ---------------------------------------------------------------- warps

def rigid_grid(h, w, angle, tx, ty, cx, cy, scale=1.0):
    """Pixel coords (h,w,2) of a rotated+shifted (+scaled) view, as (x,y)
    in source."""
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    xs, ys = xs - w / 2, ys - h / 2
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    return torch.stack([c * xs - s * ys + cx + tx, s * xs + c * ys + cy + ty], -1)


def elastic(h, w, amp, smooth):
    """Smooth random displacement field (h,w,2) in pixels."""
    d = torch.randn(1, 2, max(2, h // smooth), max(2, w // smooth)) * amp
    return F.interpolate(d, size=(h, w), mode='bicubic', align_corners=True)[0].permute(1, 2, 0)


def sample(src, grid):
    """Bilinear sample src (Hs,Ws) at pixel grid (h,w,2); returns image and
    validity mask."""
    hs, ws = src.shape
    g = grid.clone()
    g[..., 0] = g[..., 0] / (ws - 1) * 2 - 1
    g[..., 1] = g[..., 1] / (hs - 1) * 2 - 1
    out = F.grid_sample(src[None, None], g[None], mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
    valid = (g.abs() <= 1).all(-1)
    return out, valid


def stretch(img, valid):
    """Contrast-stretch to 0..255 over valid pixels, as the FT9368 firmware
    delivers its frames."""
    v = img[valid] if valid.any() else img.flatten()
    lo, hi = torch.quantile(v, 0.02), torch.quantile(v, 0.98)
    return ((img - lo) / (hi - lo + 1e-3) * 255).clamp(0, 255)


def skin_fraction(img, block=8, thr=10.0):
    """Fraction of blocks whose intensity varies like ridges do."""
    b = img[: img.shape[0] // block * block, : img.shape[1] // block * block]
    b = b.reshape(b.shape[0] // block, block, b.shape[1] // block, block)
    return (b.std(dim=(1, 3)) > thr).float().mean().item()


def photometric(img, valid, rng):
    """Sensor-like corruptions on a float image in [0,255]."""
    x = stretch(img, valid)
    if rng.random() < 0.7:                              # pressure: ridges thicker/thinner
        k = rng.choice([3, 5])
        pool = F.max_pool2d if rng.random() < 0.5 else (lambda t, k, s, p: -F.max_pool2d(-t, k, s, p))
        x = 0.5 * x + 0.5 * pool(x[None, None], k, 1, k // 2)[0, 0]
    if rng.random() < 0.6:                              # blur
        s = rng.uniform(0.4, 1.4); r = int(3 * s) + 1
        t = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-t ** 2 / (2 * s * s)); k /= k.sum()
        x = F.conv2d(F.conv2d(x[None, None], k.view(1, 1, 1, -1), padding=(0, r)), k.view(1, 1, -1, 1), padding=(r, 0))[0, 0]
    x = (x - x.mean()) * rng.uniform(0.75, 1.25) + x.mean() + rng.uniform(-15, 15)   # contrast/brightness
    if rng.random() < 0.3:                              # row streaks like the FT9368
        rows = torch.rand(x.shape[0]) < 0.15
        x[rows] += torch.randn(int(rows.sum()), 1) * 25
    x += torch.randn_like(x) * rng.uniform(2, 12)       # noise
    x[~valid] = rng.uniform(150, 255)                   # off-finger / outside: bright background
    return x.clamp(0, 255)


def make_pair(src, rng, max_angle=60, max_shift=22, real=False):
    """Two views of `src` (float tensor) and the map view2 -> view1 pixels.
    Returns v1, v2, m1, m2 (validity), t21 (h,w,2): view2 pixel -> view1 pixel."""
    hs, ws = src.shape
    if real:      # a 64x80 sensor frame: view1 = itself, lightly jittered
        cx, cy, a1 = ws / 2, hs / 2, math.radians(rng.uniform(-5, 5))
    else:         # a big impression: view1 = a random window on the skin
        for _ in range(30):
            cx, cy = rng.uniform(W / 2 + 10, ws - W / 2 - 10), rng.uniform(H / 2 + 10, hs - H / 2 - 10)
            a1 = rng.uniform(-math.pi, math.pi)
            zoom = rng.uniform(0.85, 1.15)       # ridge frequency varies between people
            g1 = rigid_grid(H, W, a1, 0, 0, cx, cy, zoom)
            probe, inside = sample(src, g1)
            if inside.all() and skin_fraction(probe) > 0.85:
                break
    if real:
        g1 = rigid_grid(H, W, a1, 0, 0, cx, cy)
    # view2 -> view1 coordinates: rigid + elastic
    a = math.radians(rng.uniform(-max_angle, max_angle))
    t21 = rigid_grid(H, W, a, rng.uniform(-max_shift, max_shift), rng.uniform(-max_shift, max_shift), W / 2, H / 2)
    t21 = t21 + elastic(H, W, rng.uniform(0.5, 2.5), 16)
    # view2 sampled from the source through view1's grid
    g2 = compose(g1, t21)
    v1, m1 = sample(src, g1)
    v2, m2 = sample(src, g2)
    return photometric(v1, m1, rng), photometric(v2, m2, rng), m1, m2, t21


def compose(g1, t21):
    """Source coords of view2 pixels: the rigid grid g1 evaluated at t21
    (exact, also outside view1, since g1 is affine)."""
    a = g1[0, 1] - g1[0, 0]
    b = g1[1, 0] - g1[0, 0]
    return g1[0, 0] + t21[..., 0:1] * a + t21[..., 1:2] * b


# ---------------------------------------------------------------- loss

def info_nce(d1, d2, t21, m1, m2, n=256, tau=0.07, cross=2048):
    """Vectorised over the batch. d1,d2: (B,D,h,w) at half resolution;
    t21: (B,H,W,2) view2 pixel -> view1 pixel; m1,m2: (B,H,W) validity.
    For n sampled view2 locations with a valid counterpart in view1, the
    counterpart must win among all h*w view1 locations.
    Returns (loss, accuracy, reliability BCE)."""
    B, D, h, w = d1.shape
    dev = d1.device
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij')
    full = t21[:, ys * 2, xs * 2] / 2.0                                   # (B,h,w,2) view1 half-res coords
    tx = full[..., 0].round().long().clamp(0, w - 1)
    ty = full[..., 1].round().long().clamp(0, h - 1)
    ok = (m2[:, ys * 2, xs * 2]
          & (full[..., 0] >= 0) & (full[..., 0] <= w - 1) & (full[..., 1] >= 0) & (full[..., 1] <= h - 1))
    ok &= m1[torch.arange(B, device=dev)[:, None, None], ty * 2, tx * 2]
    ok = ok.flatten(1)                                                    # (B,hw)
    # n locations per pair, uniformly among valid ones (padding with any
    # location, masked out, when a pair has fewer than n)
    weights = ok.float() + 1e-6
    idx = torch.multinomial(weights, n, replacement=False)                 # (B,n)
    valid = ok.gather(1, idx)
    q = d2.flatten(2).gather(2, idx[:, None, :].expand(B, D, n))          # (B,D,n)
    logits = torch.bmm(q.transpose(1, 2), d1.flatten(2)) / tau            # (B,n,hw)
    target = (ty.flatten(1) * w + tx.flatten(1)).gather(1, idx)           # (B,n)
    if cross and B > 1:
        # negatives from OTHER frames of the batch (other fingers / people):
        # skin that merely looks alike must not match
        src_b = torch.randint(0, B, (cross,), device=dev)
        src_c = torch.randint(0, h * w, (cross,), device=dev)
        neg = d1.flatten(2)[src_b, :, src_c]                              # (cross,D)
        neg_logits = torch.einsum('bdn,kd->bnk', q, neg) / tau            # (B,n,cross)
        own = src_b[None, None, :] == torch.arange(B, device=dev)[:, None, None]
        neg_logits = neg_logits.masked_fill(own, float('-inf'))           # not the query's own pair
        logits = torch.cat([logits, neg_logits], 2)
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none').view(B, n)
    loss = (ce * valid).sum() / valid.sum().clamp(min=1)
    pred = logits[..., :h * w].argmax(2)
    err = torch.hypot((pred % w - target % w).float(), (pred // w - target // w).float())
    correct = (err <= 1.0).float()
    acc = (correct * valid).sum() / valid.sum().clamp(min=1)
    return loss, acc, idx, correct, valid


def save(net, path):
    torch.save({'config': net.config, 'state': net.state_dict()}, path)


def load(path, device='cpu'):
    blob = torch.load(path, map_location=device)
    if 'config' not in blob:                       # bare state dict (early checkpoints)
        net = FpNet(); net.load_state_dict(blob)
    else:
        net = FpNet(**blob['config']); net.load_state_dict(blob['state'])
    return net.to(device)
