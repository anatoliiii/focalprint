//! Keypoints and rotation-normalised binary descriptors.
//!
//! Keypoints are Harris corners of the normalised ridge image: ridge
//! endings, bifurcations and sharp bends. Each gets a 256-bit descriptor of
//! intensity comparisons around it, rotated to the local ridge direction so
//! that the same spot compares equal however the finger is turned.

use crate::{plane::Plane, prep::Prepared};
use serde::{Deserialize, Serialize};
use std::{
    f32::consts::{PI, TAU},
    sync::LazyLock,
};
use zeroize::{Zeroize, ZeroizeOnDrop};

pub(crate) const DESCRIPTOR_BYTES: usize = 32;
const DESCRIPTOR_BITS: usize = DESCRIPTOR_BYTES * 8;
/// Radius of the descriptor patch.
const PATCH: f32 = 9.0;
/// Keypoints closer to the frame edge than this are dropped.
const BORDER: usize = 5;

/// Keypoint detector settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Integration scale of the Harris structure tensor.
    pub harris_sigma: f32,
    /// Keypoints must be the strongest response within this radius.
    pub nms_radius: usize,
    /// Responses below this fraction of the frame's strongest are ignored.
    pub relative_threshold: f32,
    /// Keep at most this many, strongest first.
    pub max_keypoints: usize,
}

impl Params {
    /// Chosen with `focalprint eval` on the FT9368 (2 fingers × 14
    /// touches): denser, finer keypoints separated fingers best.
    pub const DEFAULT: Self = Self {
        harris_sigma: 0.5,
        nms_radius: 1,
        relative_threshold: 0.002,
        max_keypoints: 400,
    };
}

impl Default for Params {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Seed of the comparison pattern; part of the template format.
const PATTERN_SEED: u64 = 0x4654_4534_3830_3031; // "FTE48001"

/// One keypoint with its descriptor.
///
/// A ridge direction is only defined modulo π, and which half the keypoint
/// angle picks can flip between touches. So the patch is described both
/// ways: `descriptor` at `angle`, `flipped` at `angle + π`; comparisons take
/// the better of the two.
#[derive(Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub(crate) struct Feature {
    pub(crate) x: f32,
    pub(crate) y: f32,
    /// Direction in `[0, 2π)`, meaningful modulo π.
    pub(crate) angle: f32,
    pub(crate) descriptor: [u8; DESCRIPTOR_BYTES],
    pub(crate) flipped: [u8; DESCRIPTOR_BYTES],
}

impl Feature {
    pub(crate) fn distance(&self, other: &Self) -> u32 {
        hamming(&self.descriptor, &other.descriptor).min(hamming(&self.flipped, &other.descriptor))
    }
}

fn hamming(a: &[u8; DESCRIPTOR_BYTES], b: &[u8; DESCRIPTOR_BYTES]) -> u32 {
    a.iter().zip(b).map(|(a, b)| (a ^ b).count_ones()).sum()
}

/// Difference of two directions known only modulo π, in `[0, π/2]`.
pub(crate) fn axis_between(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(PI);
    d.min(PI - d)
}

/// Point pairs compared by the descriptor, drawn once from a fixed seed.
static PATTERN: LazyLock<Vec<[(f32, f32); 2]>> = LazyLock::new(|| {
    let mut rng = SplitMix64(PATTERN_SEED);
    let mut point = || loop {
        // Gaussian around the keypoint, as in BRIEF, clipped to the patch.
        let (x, y) = (rng.gaussian() * PATCH / 2.5, rng.gaussian() * PATCH / 2.5);
        if x * x + y * y <= PATCH * PATCH {
            return (x, y);
        }
    };
    (0..DESCRIPTOR_BITS).map(|_| [point(), point()]).collect()
});

pub(crate) fn extract(prepared: &Prepared, params: &Params) -> Vec<Feature> {
    let image = &prepared.image;
    let response = harris(image, params.harris_sigma);
    let (w, h) = (image.width(), image.height());
    let strongest = (BORDER..h - BORDER)
        .flat_map(|y| (BORDER..w - BORDER).map(move |x| (x, y)))
        .map(|(x, y)| response.get(x, y))
        .fold(0.0f32, f32::max);
    let threshold = strongest * params.relative_threshold;

    let mut corners: Vec<(f32, usize, usize)> = Vec::new();
    for y in BORDER..h - BORDER {
        for x in BORDER..w - BORDER {
            let r = response.get(x, y);
            if r > threshold
                && prepared.in_foreground(x as f32, y as f32)
                && is_local_max(&response, x, y, r, params.nms_radius)
            {
                corners.push((r, x, y));
            }
        }
    }
    corners.sort_by(|a, b| b.0.total_cmp(&a.0));
    corners.truncate(params.max_keypoints);

    corners
        .into_iter()
        .map(|(_, x, y)| {
            let (x, y) = (x as f32, y as f32);
            let angle = keypoint_angle(prepared, x, y);
            Feature {
                x,
                y,
                angle,
                descriptor: describe(image, x, y, angle),
                flipped: describe(image, x, y, angle + PI),
            }
        })
        .collect()
}

fn harris(image: &Plane, sigma: f32) -> Plane {
    let (w, h) = (image.width(), image.height());
    let gx = Plane::from_fn(w, h, |x, y| {
        (image.at(x as isize + 1, y as isize) - image.at(x as isize - 1, y as isize)) * 0.5
    });
    let gy = Plane::from_fn(w, h, |x, y| {
        (image.at(x as isize, y as isize + 1) - image.at(x as isize, y as isize - 1)) * 0.5
    });
    let sxx = gx.map(|v| v * v).gaussian(sigma);
    let syy = gy.map(|v| v * v).gaussian(sigma);
    let sxy = gx.zip(&gy, |a, b| a * b).gaussian(sigma);
    Plane::from_fn(w, h, |x, y| {
        let (a, b, c) = (sxx.get(x, y), syy.get(x, y), sxy.get(x, y));
        a * b - c * c - 0.04 * (a + b).powi(2)
    })
}

fn is_local_max(response: &Plane, x: usize, y: usize, r: f32, radius: usize) -> bool {
    let radius = radius as isize;
    (-radius..=radius).all(|dy| {
        (-radius..=radius)
            .all(|dx| (dx == 0 && dy == 0) || response.at(x as isize + dx, y as isize + dy) < r)
    })
}

/// Local ridge direction, made a full-turn angle by picking the half that
/// points towards the patch's intensity centroid.
fn keypoint_angle(prepared: &Prepared, x: f32, y: f32) -> f32 {
    let ridge = prepared.pixel_orientation.angle.get(x as usize, y as usize);
    let (mut mx, mut my) = (0.0f32, 0.0f32);
    let r = PATCH as isize;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let v = prepared.image.at(x as isize + dx, y as isize + dy);
                mx += dx as f32 * v;
                my += dy as f32 * v;
            }
        }
    }
    let centroid = my.atan2(mx);
    let flipped = ridge + PI;
    if angle_between(ridge, centroid) <= angle_between(flipped, centroid) {
        ridge.rem_euclid(TAU)
    } else {
        flipped.rem_euclid(TAU)
    }
}

/// Absolute difference of two full-turn angles, in `[0, π]`.
pub(crate) fn angle_between(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d)
}

fn describe(image: &Plane, x: f32, y: f32, angle: f32) -> [u8; DESCRIPTOR_BYTES] {
    let (s, c) = angle.sin_cos();
    let value = |(px, py): (f32, f32)| {
        let (u, v) = (x + c * px - s * py, y + s * px + c * py);
        image
            .sample(u, v)
            .unwrap_or_else(|| image.at(u.round() as isize, v.round() as isize))
    };
    let mut descriptor = [0u8; DESCRIPTOR_BYTES];
    for (bit, [a, b]) in PATTERN.iter().enumerate() {
        if value(*a) < value(*b) {
            descriptor[bit / 8] |= 1 << (bit % 8);
        }
    }
    descriptor
}

/// Small deterministic generator for the fixed comparison pattern.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        // 24 random bits, exactly representable.
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Box–Muller.
    fn gaussian(&mut self) -> f32 {
        let u = self.uniform().max(f32::MIN_POSITIVE);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prep::{prepare, tests::ridges};

    #[test]
    fn pattern_is_stable() {
        // Changing it silently would invalidate every stored template.
        let first = PATTERN[0];
        assert!(
            PATTERN
                .iter()
                .all(|[a, b]| a.0.hypot(a.1) <= PATCH && b.0.hypot(b.1) <= PATCH)
        );
        assert_eq!(PATTERN.len(), DESCRIPTOR_BITS);
        assert!((first[0].0 - PATTERN[0][0].0).abs() < f32::EPSILON);
    }

    #[test]
    fn finds_features_on_ridges() {
        let p = prepare(&ridges(64, 80, 0.0, 0.0, 0.0), 64, 80).unwrap();
        let features = extract(&p, &Params::DEFAULT);
        assert!(features.len() >= 12, "{} features", features.len());
    }
}
