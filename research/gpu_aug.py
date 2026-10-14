"""Batched training-pair generation on the GPU (RX 6600): all sources live
in VRAM, every corruption is a batched tensor op, so the 6-core host never
limits training. Same geometry as fpnet.make_pair (view2 -> view1 map)."""
import math, random
import numpy as np
import torch, torch.nn.functional as F

H, W = 80, 64


def _skin(img, thr=10.0, k=9):
    """Local standard deviation > thr: ridges present (B,1,h,w float)."""
    m = F.avg_pool2d(img, k, 1, k // 2, count_include_pad=False)
    m2 = F.avg_pool2d(img * img, k, 1, k // 2, count_include_pad=False)
    return (m2 - m * m).clamp(min=0).sqrt() > thr


def _gauss_kernels(sigmas, dev):
    r = int(math.ceil(3 * max(sigmas)))
    t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    ks = torch.stack([torch.exp(-t ** 2 / (2 * s * s)) for s in sigmas])
    return ks / ks.sum(1, keepdim=True), r


class GpuPairs:
    def __init__(self, open_imgs, real_imgs, device, real_share=0.5, canvas=400, seed=0, pairs=None, pair_share=0.0):
        self.dev, self.real_share = device, real_share
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.rng = random.Random(seed)
        # open impressions: padded with skin-free background on a fixed canvas
        imgs = [im[:canvas, :canvas] for im in open_imgs]
        self.open_size = torch.tensor([[im.shape[1], im.shape[0]] for im in imgs], device=device, dtype=torch.float32)
        stack = np.full((len(imgs), canvas, canvas), 255, np.uint8)
        for i, im in enumerate(imgs):
            stack[i, :im.shape[0], :im.shape[1]] = im
        self.open = torch.tensor(stack, device=device)
        self.canvas = canvas
        # window centres lying on skin, per impression (sampled once)
        centres = []
        for im in imgs:
            t = torch.tensor(im, dtype=torch.float32, device=device)[None, None]
            skin = _skin(t, 12)[0, 0]
            win = F.avg_pool2d(skin[None, None].float(), (H, W), 1, (H // 2, W // 2))[0, 0, :skin.shape[0], :skin.shape[1]]
            ys, xs = torch.nonzero(win > 0.85, as_tuple=True)
            ok = (xs > W) & (xs < im.shape[1] - W) & (ys > H) & (ys < im.shape[0] - H)
            ys, xs = ys[ok], xs[ok]
            if len(xs) == 0:
                ys, xs = torch.tensor([im.shape[0] // 2], device=device), torch.tensor([im.shape[1] // 2], device=device)
            pick = torch.randint(len(xs), (256,), device=xs.device)
            centres.append(torch.stack([xs[pick], ys[pick]], 1).float())
        self.centres = torch.stack(centres).to(device)                 # (N,256,2)
        self.real = torch.tensor(np.stack(real_imgs), device=device) if real_imgs else None
        # stage B: real touch pairs with a known transform j pixels -> i pixels
        self.pair_share = pair_share if pairs else 0.0
        if pairs:
            self.pair_i = torch.tensor([p['i'] for p in pairs], device=device)
            self.pair_j = torch.tensor([p['j'] for p in pairs], device=device)
            self.pair_m = torch.tensor([p['M'] for p in pairs], device=device, dtype=torch.float32)
        self.blur, self.blur_r = _gauss_kernels([0.6, 1.0, 1.4], device)
        ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        self.base = torch.stack([xs - W / 2, ys - H / 2], -1)          # (H,W,2) centred pixel grid

    def _u(self, n, lo, hi):
        return torch.rand(n, device=self.dev, generator=self.g) * (hi - lo) + lo

    def _rigid(self, angle, tx, ty, cx, cy, scale):
        """(B,H,W,2) pixel coords: rotate/scale the centred grid, then shift."""
        c, s = torch.cos(angle) * scale, torch.sin(angle) * scale
        x, y = self.base[..., 0], self.base[..., 1]
        gx = c[:, None, None] * x - s[:, None, None] * y + (cx + tx)[:, None, None]
        gy = s[:, None, None] * x + c[:, None, None] * y + (cy + ty)[:, None, None]
        return torch.stack([gx, gy], -1)

    @staticmethod
    def _sample(src, grid, size):
        """src (B,1,Hs,Ws) float, grid pixel coords (B,H,W,2), size (B,2)=(w,h) of real content."""
        hs, ws = src.shape[-2:]
        g = torch.stack([grid[..., 0] / (ws - 1) * 2 - 1, grid[..., 1] / (hs - 1) * 2 - 1], -1)
        out = F.grid_sample(src, g, mode='bilinear', padding_mode='border', align_corners=True)
        inside = ((grid[..., 0] >= 0) & (grid[..., 1] >= 0)
                  & (grid[..., 0] <= size[:, None, None, 0] - 1) & (grid[..., 1] <= size[:, None, None, 1] - 1))
        return out[:, 0], inside

    def _photometric(self, x, valid):
        B = x.shape[0]
        dev = self.dev
        # firmware-like contrast stretch over valid pixels
        flat = torch.where(valid, x, torch.nan).flatten(1)
        lo = torch.nanquantile(flat, 0.02, dim=1, keepdim=True).nan_to_num(0)[..., None]
        hi = torch.nanquantile(flat, 0.98, dim=1, keepdim=True).nan_to_num(255)[..., None]
        x = ((x - lo) / (hi - lo + 1e-3) * 255).clamp(0, 255)[:, None]
        # pressure: ridges (dark) thicker or thinner
        k = 3
        dil = F.max_pool2d(x, k, 1, k // 2); ero = -F.max_pool2d(-x, k, 1, k // 2)
        r = torch.rand(B, 1, 1, 1, device=dev, generator=self.g)
        x = torch.where(r < 0.35, 0.5 * x + 0.5 * dil, torch.where(r < 0.7, 0.5 * x + 0.5 * ero, x))
        # blur with one of three sigmas, or none
        pad = self.blur_r
        xb = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        outs = []
        for kk in self.blur:
            t = F.conv2d(xb, kk.view(1, 1, 1, -1))
            outs.append(F.conv2d(t, kk.view(1, 1, -1, 1)))
        pick = torch.randint(0, 5, (B,), device=dev, generator=self.g)
        stacked = torch.stack(outs + [x, x], 0)                            # (5,B,1,H,W)
        x = stacked[pick, torch.arange(B, device=dev)]
        # contrast / brightness
        mean = x.mean((2, 3), keepdim=True)
        x = (x - mean) * self._u(B, 0.75, 1.25)[:, None, None, None] + mean + self._u(B, -15, 15)[:, None, None, None]
        # FT9368-like row streaks
        streak = (torch.rand(B, 1, H, 1, device=dev, generator=self.g) < 0.15) & (torch.rand(B, 1, 1, 1, device=dev, generator=self.g) < 0.3)
        x = x + streak * torch.randn(B, 1, H, 1, device=dev, generator=self.g) * 25
        x = x + torch.randn(x.shape, device=dev, generator=self.g) * self._u(B, 2, 12)[:, None, None, None]
        x = torch.where(valid[:, None], x, self._u(B, 150, 255)[:, None, None, None])
        x = x.clamp(0, 255)
        return ((x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3))

    def batch(self, B):
        dev = self.dev
        n_pair = sum(self.rng.random() < self.pair_share for _ in range(B))
        n_real = sum(self.rng.random() < self.real_share for _ in range(B - n_pair)) if self.real is not None else 0
        n_open = B - n_pair - n_real
        parts = []
        if n_open:
            idx = torch.randint(len(self.open), (n_open,), device=dev, generator=self.g)
            c = self.centres[idx, torch.randint(256, (n_open,), device=dev, generator=self.g)]
            g1 = self._rigid(self._u(n_open, -math.pi, math.pi), torch.zeros(n_open, device=dev), torch.zeros(n_open, device=dev),
                             c[:, 0], c[:, 1], self._u(n_open, 0.85, 1.15))
            src = self.open[idx][:, None].float()
            parts.append((src, g1, self.open_size[idx], 22.0))
        if n_real:
            idx = torch.randint(len(self.real), (n_real,), device=dev, generator=self.g)
            g1 = self._rigid(self._u(n_real, -0.09, 0.09), torch.zeros(n_real, device=dev), torch.zeros(n_real, device=dev),
                             torch.full((n_real,), W / 2, device=dev), torch.full((n_real,), H / 2, device=dev), torch.ones(n_real, device=dev))
            size = torch.tensor([[W, H]], device=dev, dtype=torch.float32).expand(n_real, 2)
            parts.append((self.real[idx][:, None].float(), g1, size, 16.0))
        x1s, x2s, m1s, m2s, ts = [], [], [], [], []
        for src, g1, size, shift in parts:
            n = len(g1)
            # view2 -> view1: rigid + smooth elastic displacement
            t21 = self._rigid(self._u(n, -math.radians(60), math.radians(60)), self._u(n, -shift, shift), self._u(n, -shift, shift),
                              torch.full((n,), W / 2, device=dev), torch.full((n,), H / 2, device=dev), torch.ones(n, device=dev))
            el = torch.randn(n, 2, H // 16 + 1, W // 16 + 1, device=dev, generator=self.g) * self._u(n, 0.5, 2.5)[:, None, None, None]
            t21 = t21 + F.interpolate(el, size=(H, W), mode='bicubic', align_corners=True).permute(0, 2, 3, 1)
            # g1 is affine: g2 = g1 evaluated at t21 (exact)
            a = g1[:, 0, 1] - g1[:, 0, 0]; b = g1[:, 1, 0] - g1[:, 0, 0]
            g2 = g1[:, :1, :1] + t21[..., :1] * a[:, None, None] + t21[..., 1:] * b[:, None, None]
            v1, in1 = self._sample(src, g1, size)
            v2, in2 = self._sample(src, g2, size)
            m1 = in1 & _skin(v1[:, None])[:, 0]
            m2 = in2 & _skin(v2[:, None])[:, 0]
            x1s.append(self._photometric(v1, m1)); x2s.append(self._photometric(v2, m2))
            m1s.append(m1); m2s.append(m2); ts.append(t21)
        if n_pair:
            k = torch.randint(len(self.pair_i), (n_pair,), device=dev, generator=self.g)
            v1, v2 = self.real[self.pair_i[k]].float(), self.real[self.pair_j[k]].float()
            M = self.pair_m[k]                                             # view2 (j) -> view1 (i)
            x, y = self.base[..., 0] + W / 2, self.base[..., 1] + H / 2   # pixel index coords
            t21 = torch.stack([M[:, 0, 0, None, None] * x + M[:, 0, 1, None, None] * y + M[:, 0, 2, None, None],
                               M[:, 1, 0, None, None] * x + M[:, 1, 1, None, None] * y + M[:, 1, 2, None, None]], -1)
            m1, m2 = _skin(v1[:, None])[:, 0], _skin(v2[:, None])[:, 0]
            x1s.append(self._photometric(v1, m1)); x2s.append(self._photometric(v2, m2))
            m1s.append(m1); m2s.append(m2); ts.append(t21)
        cat = lambda xs: torch.cat(xs, 0)
        return cat(x1s), cat(x2s), cat(m1s), cat(m2s), cat(ts)
