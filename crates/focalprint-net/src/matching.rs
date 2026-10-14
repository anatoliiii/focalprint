//! Dense matching of two FpNet maps: RANSAC on mutual nearest-neighbour
//! descriptor pairs finds where the template sits in the query, then the
//! template's whole descriptor map is warped onto the query and compared
//! cell by cell over the skin both share. Score = mean cosine similarity.

use crate::{DenseMap, GRID_H, GRID_W, HEIGHT, WIDTH};
use rayon::prelude::*;
use zeroize::Zeroize;

const KEYPOINTS: usize = 400;
const RANSAC_ITERS: usize = 2000;
const INLIER_PX: f32 = 3.0;
const MIN_OVERLAP: usize = 60;

/// What a touch contributes to matching. Derived biometric data.
pub struct Features {
    map: DenseMap,
    /// Descriptors cell-major (`cells × dim`): one cell's vector is contiguous.
    cells: Vec<f32>,
    /// Grid cells over skin.
    skin: Vec<bool>,
    /// Keypoint grid indices, most reliable first.
    keys: Vec<usize>,
    /// Their descriptors, contiguous (`keys.len() × dim`) for fast dots.
    key_desc: Vec<f32>,
}

impl Drop for Features {
    fn drop(&mut self) {
        self.keys.zeroize();
        self.key_desc.zeroize();
        self.cells.zeroize();
    }
}

/// Dot product with eight independent accumulators, so it vectorises
/// (a plain float `sum()` may not be reordered by the compiler).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ((ac, at), (bc, bt)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = at.iter().zip(bt).map(|(x, y)| x * y).sum();
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

impl std::fmt::Debug for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Features")
            .field("keys", &self.keys.len())
            .finish_non_exhaustive()
    }
}

/// Grid cell → pixel centre.
fn pixel(cell: usize) -> (f32, f32) {
    (
        (cell % GRID_W) as f32 * 2.0 + 0.5,
        (cell / GRID_W) as f32 * 2.0 + 0.5,
    )
}

impl Features {
    /// Keypoints = the most reliable grid cells lying on skin.
    #[must_use]
    pub fn new(pixels: &[u8], map: DenseMap) -> Self {
        let skin = skin_cells(pixels);
        let mut order: Vec<usize> = (0..GRID_W * GRID_H)
            .filter(|&c| skin[c] && map.reliability[c] > 0.05)
            .collect();
        order.sort_by(|&a, &b| map.reliability[b].total_cmp(&map.reliability[a]));
        order.truncate(KEYPOINTS);
        let dim = map.dim;
        let n = GRID_W * GRID_H;
        let mut cells = vec![0.0f32; n * dim];
        for k in 0..dim {
            for c in 0..n {
                cells[c * dim + k] = map.descriptors[k * n + c];
            }
        }
        let mut key_desc = Vec::with_capacity(order.len() * dim);
        for &c in &order {
            key_desc.extend_from_slice(&cells[c * dim..(c + 1) * dim]);
        }
        Self {
            map,
            cells,
            skin,
            keys: order,
            key_desc,
        }
    }

    /// Usable keypoints: few means a small, faint or smeared touch.
    #[must_use]
    pub fn keypoints(&self) -> usize {
        self.keys.len()
    }

    fn cell(&self, c: usize) -> &[f32] {
        &self.cells[c * self.map.dim..(c + 1) * self.map.dim]
    }

    fn key(&self, i: usize) -> &[f32] {
        &self.key_desc[i * self.map.dim..(i + 1) * self.map.dim]
    }
}

/// Grid cells whose 9×9 neighbourhood at full resolution varies like ridges.
fn skin_cells(pixels: &[u8]) -> Vec<bool> {
    let std_at = |cx: usize, cy: usize| {
        let (mut s, mut s2, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for y in cy.saturating_sub(4)..(cy + 5).min(HEIGHT) {
            for x in cx.saturating_sub(4)..(cx + 5).min(WIDTH) {
                let v = f32::from(pixels[y * WIDTH + x]);
                s += v;
                s2 += v * v;
                n += 1.0;
            }
        }
        (s2 / n - (s / n).powi(2)).max(0.0).sqrt()
    };
    (0..GRID_W * GRID_H)
        .map(|c| {
            let (gx, gy) = (c % GRID_W, c / GRID_W);
            // majority of the 2×2 pixels behind the cell
            let hits = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .iter()
                .filter(|(dx, dy)| std_at(gx * 2 + dx, gy * 2 + dy) > 10.0)
                .count();
            hits >= 2
        })
        .collect()
}

/// A template pixel and the query pixel it was matched to.
type Pair = ((f32, f32), (f32, f32));

/// Similarity transform `p ↦ s·R·p + t`.
#[derive(Debug, Clone, Copy)]
struct Similarity {
    a: f32, // s·cosθ
    b: f32, // s·sinθ
    tx: f32,
    ty: f32,
}

impl Similarity {
    fn apply(&self, (x, y): (f32, f32)) -> (f32, f32) {
        (
            self.a * x - self.b * y + self.tx,
            self.b * x + self.a * y + self.ty,
        )
    }

    fn scale(&self) -> f32 {
        self.a.hypot(self.b)
    }

    fn inverse(&self) -> Self {
        let d = self.a * self.a + self.b * self.b;
        let (a, b) = (self.a / d, -self.b / d);
        Self {
            a,
            b,
            tx: -(a * self.tx - b * self.ty),
            ty: -(b * self.tx + a * self.ty),
        }
    }

    /// Least-squares fit mapping `src` onto `dst`.
    fn fit(pairs: &[Pair]) -> Option<Self> {
        let n = pairs.len() as f32;
        if pairs.len() < 2 {
            return None;
        }
        let (mut sx, mut sy, mut dx, mut dy) = (0.0, 0.0, 0.0, 0.0);
        for &((x, y), (u, v)) in pairs {
            sx += x;
            sy += y;
            dx += u;
            dy += v;
        }
        let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);
        let (mut num_a, mut num_b, mut den) = (0.0, 0.0, 0.0);
        for &((x, y), (u, v)) in pairs {
            let (x, y, u, v) = (x - sx, y - sy, u - dx, v - dy);
            num_a += x * u + y * v;
            num_b += x * v - y * u;
            den += x * x + y * y;
        }
        if den < 1e-6 {
            return None;
        }
        let (a, b) = (num_a / den, num_b / den);
        Some(Self {
            a,
            b,
            tx: dx - (a * sx - b * sy),
            ty: dy - (b * sx + a * sy),
        })
    }
}

/// Small deterministic RNG for RANSAC sampling.
struct XorShift(u64);
impl XorShift {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// Where `template` lies in `query`: transform template pixels → query pixels.
fn align(query: &Features, template: &Features) -> Option<Similarity> {
    // mutual nearest neighbours on cosine similarity, one similarity matrix
    let (nq, nt) = (query.keys.len(), template.keys.len());
    if nq == 0 || nt == 0 {
        return None;
    }
    let mut row_best = vec![(f32::MIN, 0usize); nq];
    let mut col_best = vec![(f32::MIN, 0usize); nt];
    for i in 0..nq {
        let q = query.key(i);
        for j in 0..nt {
            let s = dot(q, template.key(j));
            if s > row_best[i].0 {
                row_best[i] = (s, j);
            }
            if s > col_best[j].0 {
                col_best[j] = (s, i);
            }
        }
    }
    let pairs: Vec<Pair> = (0..nq)
        .filter(|&i| col_best[row_best[i].1].1 == i)
        .map(|i| (pixel(template.keys[row_best[i].1]), pixel(query.keys[i])))
        .collect();
    if pairs.len() < 3 {
        return None;
    }
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    let inliers_of = |t: &Similarity| -> Vec<Pair> {
        pairs
            .iter()
            .copied()
            .filter(|&(s, d)| {
                let (x, y) = t.apply(s);
                (x - d.0).hypot(y - d.1) <= INLIER_PX
            })
            .collect()
    };
    let mut best: Option<(Similarity, usize)> = None;
    let mut needed = RANSAC_ITERS;
    let mut iter = 0;
    while iter < needed {
        iter += 1;
        let (i, j) = (rng.below(pairs.len()), rng.below(pairs.len()));
        if i == j {
            continue;
        }
        let Some(t) = Similarity::fit(&[pairs[i], pairs[j]]) else {
            continue;
        };
        if !(0.9..1.1).contains(&t.scale()) {
            continue;
        }
        let n = inliers_of(&t).len();
        if best.is_none_or(|(_, b)| n > b) {
            best = Some((t, n));
            // adaptive stop: enough samples to hit an all-inlier pair with 99.9%
            let w = n as f32 / pairs.len() as f32;
            let p_good = (w * w).clamp(1e-6, 1.0 - 1e-6);
            let k = ((1.0f32 - 0.999).ln() / (1.0 - p_good).ln()).ceil();
            needed = needed.min((k as usize).max(200));
        }
    }
    let (rough, n) = best?;
    if n < 3 {
        return None;
    }
    let refined = Similarity::fit(&inliers_of(&rough)).unwrap_or(rough);
    (0.9..1.1).contains(&refined.scale()).then_some(refined)
}

/// Mean cosine similarity of `template`'s descriptors, warped by `t`, with
/// `query`'s over the skin both share; `None` if they barely overlap.
fn dense(query: &Features, template: &Features, t: &Similarity) -> Option<f32> {
    let back = t.inverse(); // query pixel → template pixel
    let dim = query.map.dim;
    let mut td = vec![0.0f32; dim];
    let (mut sum, mut overlap) = (0.0f32, 0usize);
    for cell in 0..GRID_W * GRID_H {
        if !query.skin[cell] {
            continue;
        }
        let (x, y) = back.apply(pixel(cell));
        let (gx, gy) = ((x - 0.5) / 2.0, (y - 0.5) / 2.0);
        if gx < 0.0 || gy < 0.0 || gx > (GRID_W - 1) as f32 || gy > (GRID_H - 1) as f32 {
            continue;
        }
        let near = (gy.round() as usize) * GRID_W + gx.round() as usize;
        if !template.skin[near] {
            continue;
        }
        // bilinear blend of the four surrounding template cells
        let (x0, y0) = (gx.floor() as usize, gy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(GRID_W - 1), (y0 + 1).min(GRID_H - 1));
        let (fx, fy) = (gx - x0 as f32, gy - y0 as f32);
        td.fill(0.0);
        for (cx, cy, w) in [
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x1, y0, fx * (1.0 - fy)),
            (x0, y1, (1.0 - fx) * fy),
            (x1, y1, fx * fy),
        ] {
            for (d, v) in td.iter_mut().zip(template.cell(cy * GRID_W + cx)) {
                *d += w * v;
            }
        }
        let norm = dot(&td, &td).sqrt().max(1e-6);
        sum += dot(query.cell(cell), &td) / norm;
        overlap += 1;
    }
    td.zeroize();
    (overlap >= MIN_OVERLAP).then(|| sum / overlap as f32)
}

fn score_one(query: &Features, t: &Features) -> f32 {
    align(query, t)
        .and_then(|s| dense(query, t, &s))
        .unwrap_or(0.0)
}

/// Best score of `query` against any touch of a template (all of them).
#[must_use]
pub fn score(query: &Features, template: &[Features]) -> f32 {
    template
        .par_iter()
        .map(|t| score_one(query, t))
        .reduce(|| 0.0, f32::max)
}

/// Score of `query` against each touch of a template, in order.
#[must_use]
pub fn scores(query: &Features, template: &[Features]) -> Vec<f32> {
    template.par_iter().map(|t| score_one(query, t)).collect()
}

/// Whether `query` reaches `threshold` against the template; stops at the
/// first touch that does. Returns the score that decided it.
#[must_use]
pub fn verify(query: &Features, template: &[Features], threshold: f32) -> (bool, f32) {
    match template
        .par_iter()
        .map(|t| score_one(query, t))
        .find_any(|&s| s >= threshold)
    {
        Some(s) => (true, s),
        None => (false, score(query, template).min(threshold - f32::EPSILON)),
    }
}
