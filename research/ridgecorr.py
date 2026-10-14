"""Matcher B prototype: Gabor ridge enhancement + masked normalised
cross-correlation (Padfield 2012) over a rotation grid, via FFT."""
import cv2, numpy as np

PERIOD = 6.5

def enhance(im):
    im = im.astype(np.float32)
    hp = im - cv2.GaussianBlur(im, (0, 0), 6)
    norm = hp / (np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 6)) + 1e-3)
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 4); gyy = cv2.GaussianBlur(gy * gy, (0, 0), 4); gxy = cv2.GaussianBlur(gx * gy, (0, 0), 4)
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)
    n = 12
    resp = []
    for a in np.arange(n) * np.pi / n:
        k = cv2.getGaborKernel((19, 19), sigma=0.55 * PERIOD, theta=a, lambd=PERIOD, gamma=1.0, psi=0)
        k -= k.mean()
        resp.append(cv2.filter2D(norm, -1, k))
    idx = np.round((theta % np.pi) / np.pi * n).astype(int) % n
    out = np.take_along_axis(np.stack(resp), idx[None], 0)[0]
    out /= (np.abs(out).mean() + 1e-6)
    std = cv2.GaussianBlur(np.abs(hp), (0, 0), 4)
    mask = ((std > 6) & (coh > 0.3)).astype(np.float32)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return out.astype(np.float32), mask

def _fft(a, shape):
    return np.fft.rfft2(a, shape)

def masked_ncc(f, mf, g, mg, min_overlap):
    """NCC of moving g against fixed f for every integer shift (Padfield)."""
    shape = (f.shape[0] + g.shape[0] - 1, f.shape[1] + g.shape[1] - 1)
    f = f * mf; g = g * mg
    rg = np.rot90(g, 2); rmg = np.rot90(mg, 2)
    F, F2, MF = _fft(f, shape), _fft(f * f, shape), _fft(mf, shape)
    G, G2, MG = _fft(rg, shape), _fft(rg * rg, shape), _fft(rmg, shape)
    ir = lambda X: np.fft.irfft2(X, shape)
    overlap = np.round(ir(MF * MG))
    overlap = np.maximum(overlap, 1)
    sf, sg = ir(F * MG), ir(MF * G)
    num = ir(F * G) - sf * sg / overlap
    df = np.maximum(ir(F2 * MG) - sf * sf / overlap, 0)
    dg = np.maximum(ir(MF * G2) - sg * sg / overlap, 0)
    ncc = num / (np.sqrt(df * dg) + 1e-6)
    ncc[overlap < min_overlap] = 0
    return ncc

class RidgeCorr:
    def __init__(self, angles=range(-30, 31, 3), min_overlap=1500):
        self.angles = list(angles); self.min_overlap = min_overlap
    def template(self, wins):
        return [enhance(w) for w in wins]
    def score(self, template, query):
        q, mq = enhance(query)
        h, w = q.shape
        rots = []
        for a in self.angles:
            R = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
            rots.append((cv2.warpAffine(q, R, (w, h)), cv2.warpAffine(mq, R, (w, h), flags=cv2.INTER_NEAREST)))
        best = 0.0
        for t, mt in template:
            for qr, mqr in rots:
                best = max(best, float(masked_ncc(t, mt, qr, mqr, self.min_overlap).max()))
        return best
