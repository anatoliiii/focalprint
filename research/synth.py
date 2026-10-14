"""SFinGe-style synthetic fingerprints on the GPU: no real person behind any
of them, so they can feed distillation in unlimited numbers.

Orientation: Sherlock-Monro zero-pole model (arch / loop / whorl / twin
loop, random singular points) plus a smooth random perturbation. Ridges:
white noise refined by iterated Gabor filtering along that orientation, with
a slowly varying ridge period. Output: uint8, dark ridges on a bright
background, fingertip-shaped skin area, ~500 dpi like the open datasets."""
import math
import numpy as np
import torch, torch.nn.functional as F

BINS = 16          # orientation quantisation of the Gabor bank
SIZE = 320


def _gabor_bank(period, dev, k=17):
    r = k // 2
    y, x = torch.meshgrid(torch.arange(-r, r + 1, dtype=torch.float32, device=dev),
                          torch.arange(-r, r + 1, dtype=torch.float32, device=dev), indexing='ij')
    out = []
    for b in range(BINS):
        t = math.pi * b / BINS                        # ridge direction
        u = -x * math.sin(t) + y * math.cos(t)        # across the ridges
        v = x * math.cos(t) + y * math.sin(t)         # along the ridges
        g = torch.exp(-(u ** 2) / (2 * (period * 0.45) ** 2) - (v ** 2) / (2 * (period * 0.9) ** 2)) \
            * torch.cos(2 * math.pi * u / period)
        out.append(g - g.mean())
    bank = torch.stack(out)[:, None]
    return bank / bank.abs().sum((1, 2, 3), keepdim=True)


def _orientation(n, gen, dev):
    """(n,S,S) ridge orientation in radians, mod pi."""
    S = SIZE
    y, x = torch.meshgrid(torch.arange(S, dtype=torch.float32, device=dev),
                          torch.arange(S, dtype=torch.float32, device=dev), indexing='ij')
    z = torch.complex(x, y)[None].expand(n, S, S)
    u = lambda *s: torch.rand(*s, device=dev, generator=gen)
    theta = torch.zeros(n, S, S, device=dev)
    kind = torch.multinomial(torch.tensor([0.07, 0.6, 0.25, 0.08], device=dev), n, True, generator=gen)
    cx = S / 2 + (u(n) - 0.5) * 60
    cy = S * 0.42 + (u(n) - 0.5) * 60
    for i in range(n):
        k = int(kind[i])
        cores, deltas = [], []
        if k == 1:      # loop, leaning left or right
            side = 1 if u(1) < 0.5 else -1
            cores.append(complex(cx[i], cy[i]))
            deltas.append(complex(cx[i] + side * (60 + 40 * float(u(1))), cy[i] + 90 + 40 * float(u(1))))
        elif k == 2:    # whorl
            d = 15 + 20 * float(u(1))
            cores += [complex(cx[i], cy[i] - d / 2), complex(cx[i], cy[i] + d / 2)]
            deltas += [complex(cx[i] - 90 - 30 * float(u(1)), cy[i] + 100), complex(cx[i] + 90 + 30 * float(u(1)), cy[i] + 100)]
        elif k == 3:    # twin loop
            cores += [complex(cx[i] - 15, cy[i] - 10), complex(cx[i] + 15, cy[i] + 10)]
            deltas += [complex(cx[i] - 100, cy[i] + 90), complex(cx[i] + 100, cy[i] + 110)]
        if k == 0:      # plain arch
            a = 0.6 + 0.6 * float(u(1))
            theta[i] = torch.atan(a * torch.tanh((x - cx[i]) / 60) * torch.exp(-((y - cy[i]) / 120) ** 2))
        else:
            t = torch.zeros(S, S, device=dev)
            for c in cores:
                t = t + torch.angle(z[i] - c)
            for d in deltas:
                t = t - torch.angle(z[i] - d)
            theta[i] = 0.5 * t
    # smooth random perturbation, stronger far from the centre
    pert = F.interpolate(torch.randn(n, 1, 6, 6, device=dev, generator=gen), size=(S, S), mode='bicubic', align_corners=True)[:, 0]
    theta = theta + pert * 0.25
    return theta % math.pi


def _skin_mask(n, gen, dev):
    S = SIZE
    y, x = torch.meshgrid(torch.linspace(-1, 1, S, device=dev), torch.linspace(-1, 1, S, device=dev), indexing='ij')
    u = lambda *s: torch.rand(*s, device=dev, generator=gen)
    a = (0.62 + 0.25 * u(n))[:, None, None]; b = (0.8 + 0.18 * u(n))[:, None, None]
    yc = (0.05 * (u(n) - 0.5))[:, None, None]
    r = (x / a) ** 2 + ((y - yc) / b) ** 2
    edge = F.interpolate(torch.randn(n, 1, 8, 8, device=dev, generator=gen), size=(S, S), mode='bicubic', align_corners=True)[:, 0]
    return (r + 0.06 * edge) < 1


@torch.no_grad()
def generate(n, device='cuda', seed=0, iters=14, batch=32):
    """Returns a list of n uint8 numpy images (SIZE x SIZE)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for start in range(0, n, batch):
        m = min(batch, n - start)
        theta = _orientation(m, gen, device)
        bins = (theta / math.pi * BINS).round().long() % BINS                 # (m,S,S)
        onehot = F.one_hot(bins, BINS).permute(0, 3, 1, 2).float()           # (m,BINS,S,S)
        period = float(torch.empty(1).uniform_(8.0, 10.5, generator=torch.Generator().manual_seed(seed + start)))
        bank = _gabor_bank(period, device)
        x = torch.randn(m, 1, SIZE, SIZE, device=device, generator=gen)
        for _ in range(iters):
            resp = F.conv2d(x, bank, padding=bank.shape[-1] // 2)            # (m,BINS,S,S)
            x = (resp * onehot).sum(1, keepdim=True)
            x = torch.tanh(x / (x.std((2, 3), keepdim=True) + 1e-6) * 1.5)
        skin = _skin_mask(m, gen, device)[:, None]
        # ink: dark ridges, bright valleys, uneven pressure, bright outside
        press = F.interpolate(torch.rand(m, 1, 5, 5, device=device, generator=gen), size=(SIZE, SIZE), mode='bicubic', align_corners=True)
        img = 128 + 100 * x * (0.6 + 0.4 * press) + 20 * (press - 0.5)
        img = torch.where(skin, img, torch.full_like(img, 245))
        img = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode='replicate'), 3, 1)
        out += list(img[:, 0].clamp(0, 255).byte().cpu().numpy())
    return out


if __name__ == '__main__':
    import sys, cv2
    dest = sys.argv[1]; count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, im in enumerate(generate(count, dev, batch=8)):
        cv2.imwrite(f'{dest}/synth{i:03d}.png', im)
