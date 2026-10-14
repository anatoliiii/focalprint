//! Comparing two samples: descriptor matches, then a rigid alignment that
//! must explain enough of them, spread over the overlap, with ridge
//! directions that agree.

use crate::{
    features::{Feature, axis_between},
    prep::BLOCK,
    sample::Sample,
};
use std::f32::consts::PI;

/// Descriptor distance above which two features are unrelated.
const MAX_HAMMING: u32 = 64;
/// Best match must beat the runner-up by this factor (Lowe's ratio test).
const RATIO: f32 = 0.8;
/// Pixel distance within which an aligned feature counts as an inlier.
const INLIER_PX: f32 = 3.0;
/// Feature directions must agree after alignment within this angle.
const INLIER_ANGLE: f32 = 0.35;
/// Pairs closer than this give a poor rotation estimate.
const MIN_PAIR_SPAN: f32 = 6.0;
/// Grid used to measure how spread out the inliers are.
const CELL: f32 = 16.0;
/// Blocks this coherent carry a trustworthy ridge direction.
const MIN_COHERENCE: f32 = 0.3;

/// Everything learnt from one comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Evidence {
    /// Mutual, unambiguous descriptor matches.
    pub candidates: usize,
    /// Of those, explained by the best alignment.
    pub inliers: usize,
    /// Distinct 16 × 16 px cells holding inliers.
    pub cells: usize,
    /// Foreground blocks present in both samples after alignment.
    pub overlap: usize,
    /// Mean `cos 2Δθ` of ridge directions over the overlap, in `[-1, 1]`.
    pub agreement: f32,
    /// Rotation of the alignment, radians.
    pub rotation: f32,
    /// Translation of the alignment, pixels.
    pub shift: (f32, f32),
    /// Mean descriptor distance (bits of 256) over the inliers: low for
    /// the same skin, near the matching limit for chance alignments.
    pub hamming: f32,
}

impl Evidence {
    /// A single number for ranking and for threshold studies: inliers
    /// weighted by how well the ridge flow agrees.
    #[must_use]
    pub fn score(&self) -> f32 {
        self.inliers as f32 * self.agreement.max(0.0)
    }
}

/// Rigid transform `p ↦ R·p + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rigid {
    pub(crate) rotation: f32,
    pub(crate) tx: f32,
    pub(crate) ty: f32,
}

impl Rigid {
    pub(crate) const IDENTITY: Self = Self {
        rotation: 0.0,
        tx: 0.0,
        ty: 0.0,
    };

    pub(crate) fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let (s, c) = self.rotation.sin_cos();
        (c * x - s * y + self.tx, s * x + c * y + self.ty)
    }

    fn invert(&self, x: f32, y: f32) -> (f32, f32) {
        let (s, c) = self.rotation.sin_cos();
        let (x, y) = (x - self.tx, y - self.ty);
        (c * x + s * y, -s * x + c * y)
    }

    /// `self ∘ inner`: first `inner`, then `self`.
    pub(crate) fn after(&self, inner: &Self) -> Self {
        let (tx, ty) = self.apply(inner.tx, inner.ty);
        Self {
            rotation: self.rotation + inner.rotation,
            tx,
            ty,
        }
    }

    pub(crate) fn inverse(&self) -> Self {
        let (tx, ty) = self.invert(0.0, 0.0);
        Self {
            rotation: -self.rotation,
            tx,
            ty,
        }
    }
}

/// Compares `query` against `reference`.
#[must_use]
pub fn compare(query: &Sample, reference: &Sample) -> Evidence {
    compare_rigid(query, reference).0
}

/// [`compare`], also returning the alignment `reference → query`.
pub(crate) fn compare_rigid(query: &Sample, reference: &Sample) -> (Evidence, Option<Rigid>) {
    let (candidates, aligned) = align_features(&query.features, &reference.features, 0.0);
    let Some((transform, inliers)) = aligned else {
        return (
            Evidence {
                candidates,
                ..Evidence::default()
            },
            None,
        );
    };
    let (overlap, agreement) = flow_agreement(query, &[(reference, transform)]);
    let evidence = Evidence {
        candidates,
        inliers: inliers.len(),
        cells: cells(&query.features, &inliers),
        overlap,
        agreement,
        rotation: transform.rotation,
        shift: (transform.tx, transform.ty),
        hamming: mean_hamming(&query.features, &reference.features, &inliers),
    };
    (evidence, Some(transform))
}

/// Matched feature indices `(query, reference)`.
pub(crate) type Pairs = Vec<(usize, usize)>;

/// Mutual matches and, if any, the best alignment `reference → query`
/// with its inlier pairs.
pub(crate) fn align_features(
    query: &[Feature],
    reference: &[Feature],
    same_point_px: f32,
) -> (usize, Option<(Rigid, Pairs)>) {
    let pairs = mutual_matches(query, reference, same_point_px);
    (pairs.len(), align(&pairs, query, reference))
}

/// Mean descriptor distance over `inliers`.
pub(crate) fn mean_hamming(
    query: &[Feature],
    reference: &[Feature],
    inliers: &[(usize, usize)],
) -> f32 {
    if inliers.is_empty() {
        return 0.0;
    }
    let total: u32 = inliers
        .iter()
        .map(|&(q, r)| query[q].distance(&reference[r]))
        .sum();
    total as f32 / inliers.len() as f32
}

/// Distinct 16 × 16 px cells of the query holding inliers.
pub(crate) fn cells(query: &[Feature], inliers: &[(usize, usize)]) -> usize {
    let mut cells: Vec<(i32, i32)> = inliers
        .iter()
        .map(|&(q, _)| ((query[q].x / CELL) as i32, (query[q].y / CELL) as i32))
        .collect();
    cells.sort_unstable();
    cells.dedup();
    cells.len()
}

/// Pairs `(query index, reference index)` that are each other's clear best.
///
/// "Clear" is Lowe's ratio test against the runner-up. In a mosaic the same
/// physical point can appear several times (seen by overlapping touches,
/// descriptors slightly different), so reference features within
/// `same_point_px` of the best one are not counted as its competitors.
fn mutual_matches(query: &[Feature], reference: &[Feature], same_point_px: f32) -> Pairs {
    let nearest = |from: &Feature, among: &[Feature]| -> Option<(usize, u32)> {
        among
            .iter()
            .map(|other| from.distance(other))
            .enumerate()
            .min_by_key(|&(_, d)| d)
    };
    query
        .iter()
        .enumerate()
        .filter_map(|(qi, q)| {
            let (ri, d) = nearest(q, reference)?;
            let best = &reference[ri];
            let runner_up = reference
                .iter()
                .enumerate()
                .filter(|&(i, r)| i != ri && (r.x - best.x).hypot(r.y - best.y) > same_point_px)
                .map(|(_, r)| q.distance(r))
                .min()
                .unwrap_or(u32::MAX);
            let clear = d <= MAX_HAMMING && (d as f32) < RATIO * runner_up as f32;
            let (back, _) = nearest(best, query)?;
            (clear && back == qi).then_some((qi, ri))
        })
        .collect()
}

/// Best rigid alignment over all two-point hypotheses, refined by least
/// squares. Returns it with its inlier pairs.
fn align(
    pairs: &[(usize, usize)],
    query: &[Feature],
    reference: &[Feature],
) -> Option<(Rigid, Vec<(usize, usize)>)> {
    let mut best: Option<(Rigid, usize)> = None;
    for (i, &(qa, ra)) in pairs.iter().enumerate() {
        for &(qb, rb) in &pairs[i + 1..] {
            let (q0, q1, r0, r1) = (&query[qa], &query[qb], &reference[ra], &reference[rb]);
            let (qdx, qdy, rdx, rdy) = (q1.x - q0.x, q1.y - q0.y, r1.x - r0.x, r1.y - r0.y);
            let (qlen, rlen) = (qdx.hypot(qdy), rdx.hypot(rdy));
            if qlen < MIN_PAIR_SPAN || rlen < MIN_PAIR_SPAN || (qlen / rlen - 1.0).abs() > 0.1 {
                continue;
            }
            let rotation = qdy.atan2(qdx) - rdy.atan2(rdx);
            let (s, c) = rotation.sin_cos();
            let transform = Rigid {
                rotation,
                tx: q0.x - (c * r0.x - s * r0.y),
                ty: q0.y - (s * r0.x + c * r0.y),
            };
            let count = inliers(pairs, query, reference, &transform).len();
            if best.is_none_or(|(_, n)| count > n) {
                best = Some((transform, count));
            }
        }
    }
    let (rough, count) = best?;
    if count < 3 {
        return None;
    }
    let refined =
        least_squares(&inliers(pairs, query, reference, &rough), query, reference).unwrap_or(rough);
    let kept = inliers(pairs, query, reference, &refined);
    Some((refined, kept))
}

fn inliers(
    pairs: &[(usize, usize)],
    query: &[Feature],
    reference: &[Feature],
    t: &Rigid,
) -> Vec<(usize, usize)> {
    pairs
        .iter()
        .copied()
        .filter(|&(q, r)| {
            let (q, r) = (&query[q], &reference[r]);
            let (x, y) = t.apply(r.x, r.y);
            (x - q.x).hypot(y - q.y) <= INLIER_PX
                && axis_between(q.angle, r.angle + t.rotation) <= INLIER_ANGLE
        })
        .collect()
}

/// 2-D Procrustes fit (rotation + translation) of reference onto query.
fn least_squares(
    pairs: &[(usize, usize)],
    query: &[Feature],
    reference: &[Feature],
) -> Option<Rigid> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f32;
    let centroid = |points: &mut dyn Iterator<Item = (f32, f32)>| {
        let (sx, sy) = points.fold((0.0, 0.0), |(ax, ay), (x, y)| (ax + x, ay + y));
        (sx / n, sy / n)
    };
    let (qcx, qcy) = centroid(&mut pairs.iter().map(|&(q, _)| (query[q].x, query[q].y)));
    let (rcx, rcy) = centroid(&mut pairs.iter().map(|&(_, r)| (reference[r].x, reference[r].y)));
    let (mut sin, mut cos) = (0.0f32, 0.0f32);
    for &(q, r) in pairs {
        let (qx, qy) = (query[q].x - qcx, query[q].y - qcy);
        let (rx, ry) = (reference[r].x - rcx, reference[r].y - rcy);
        cos += rx * qx + ry * qy;
        sin += rx * qy - ry * qx;
    }
    let rotation = sin.atan2(cos);
    let (s, c) = rotation.sin_cos();
    Some(Rigid {
        rotation,
        tx: qcx - (c * rcx - s * rcy),
        ty: qcy - (s * rcx + c * rcy),
    })
}

/// Ridge-flow check of an alignment: over the query's coherent blocks that
/// some reference covers (first match wins, so overlapping references are
/// not counted twice), the number of such blocks and the mean `cos 2Δθ`.
/// Each reference comes with its transform `reference → query`.
pub(crate) fn flow_agreement(query: &Sample, references: &[(&Sample, Rigid)]) -> (usize, f32) {
    let (mut overlap, mut sum) = (0usize, 0.0f32);
    let centre = |b: usize| (b * BLOCK) as f32 + BLOCK as f32 / 2.0;
    for by in 0..query.blocks_y() {
        for bx in 0..query.blocks_x() {
            let Some((q_angle, q_coherence)) = query.block(bx, by) else {
                continue;
            };
            if q_coherence < MIN_COHERENCE {
                continue;
            }
            let covering = references.iter().find_map(|(reference, t)| {
                let (rx, ry) = t.invert(centre(bx), centre(by));
                if rx < 0.0 || ry < 0.0 {
                    return None;
                }
                let (r_angle, r_coherence) =
                    reference.block(rx as usize / BLOCK, ry as usize / BLOCK)?;
                (r_coherence >= MIN_COHERENCE).then_some(r_angle + t.rotation)
            });
            if let Some(r_angle) = covering {
                overlap += 1;
                sum += (2.0 * (q_angle - r_angle)).rem_euclid(2.0 * PI).cos();
            }
        }
    }
    (
        overlap,
        if overlap == 0 {
            0.0
        } else {
            sum / overlap as f32
        },
    )
}

#[cfg(test)]
mod tests {
    use super::Rigid;

    #[test]
    fn rigid_algebra() {
        let a = Rigid {
            rotation: 0.4,
            tx: 3.0,
            ty: -2.0,
        };
        let b = Rigid {
            rotation: -1.1,
            tx: 7.0,
            ty: 5.0,
        };
        let (x, y) = (11.0, -4.0);
        let (ax, ay) = b.apply(x, y);
        let (cx, cy) = a.after(&b).apply(x, y);
        let (ex, ey) = a.apply(ax, ay);
        assert!((cx - ex).abs() < 1e-4 && (cy - ey).abs() < 1e-4);
        let (ix, iy) = a.inverse().apply(a.apply(x, y).0, a.apply(x, y).1);
        assert!((ix - x).abs() < 1e-4 && (iy - y).abs() < 1e-4);
    }
}
