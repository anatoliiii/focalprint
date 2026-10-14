//! Frame quality, normalisation and the ridge orientation field.

use crate::plane::Plane;
use std::f32::consts::FRAC_PI_2;

/// Side of the square blocks used for the foreground mask and the stored
/// orientation field.
pub(crate) const BLOCK: usize = 8;

/// Why a frame cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum Rejected {
    /// Too little of the frame shows ridges (partial or very light touch).
    #[error("finger covers only {coverage:.0}% of the sensor")]
    Coverage { coverage: f32 },
    /// Ridges are too faint to tell apart (wet, dry or no finger).
    #[error("ridge contrast too low ({contrast:.1})")]
    Contrast { contrast: f32 },
    /// Not enough distinctive points for a reliable comparison.
    #[error("only {found} usable features")]
    Features { found: usize },
}

/// Tunables for [`prepare`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct QualityPolicy {
    /// Raw-intensity standard deviation that makes a block foreground.
    pub(crate) block_std: f32,
    /// Minimum foreground fraction of the frame.
    pub(crate) coverage: f32,
    /// Minimum mean foreground standard deviation.
    pub(crate) contrast: f32,
}

pub(crate) const QUALITY: QualityPolicy = QualityPolicy {
    block_std: 10.0,
    coverage: 0.45,
    contrast: 18.0,
};

/// Per-block ridge orientation (radians, `[0, π)`) and its coherence
/// (`0` = no dominant direction, `1` = perfectly parallel ridges).
#[derive(Clone)]
pub(crate) struct Orientation {
    pub(crate) angle: Plane,
    pub(crate) coherence: Plane,
}

pub(crate) struct Prepared {
    /// Zero-mean, unit-variance image, lightly smoothed.
    pub(crate) image: Plane,
    /// Pixel-level orientation, for rotating descriptors.
    pub(crate) pixel_orientation: Orientation,
    /// Block-level orientation, for the template.
    pub(crate) block_orientation: Orientation,
    /// Foreground blocks.
    pub(crate) mask: Vec<bool>,
    pub(crate) blocks_x: usize,
    pub(crate) blocks_y: usize,
}

impl Prepared {
    pub(crate) fn in_foreground(&self, x: f32, y: f32) -> bool {
        if x < 0.0 || y < 0.0 {
            return false;
        }
        let (bx, by) = (x as usize / BLOCK, y as usize / BLOCK);
        bx < self.blocks_x && by < self.blocks_y && self.mask[by * self.blocks_x + bx]
    }
}

pub(crate) fn prepare(pixels: &[u8], width: usize, height: usize) -> Result<Prepared, Rejected> {
    assert_eq!(
        pixels.len(),
        width * height,
        "pixel buffer does not match dimensions"
    );
    let raw = Plane::from_fn(width, height, |x, y| f32::from(pixels[y * width + x]));
    let (blocks_x, blocks_y) = (width / BLOCK, height / BLOCK);

    // Foreground: blocks where the intensity actually varies (ridges/valleys).
    let mut mask = vec![false; blocks_x * blocks_y];
    let mut contrast_sum = 0.0;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let std = block_std(&raw, bx, by);
            if std >= QUALITY.block_std {
                mask[by * blocks_x + bx] = true;
                contrast_sum += std;
            }
        }
    }
    let foreground = mask.iter().filter(|&&m| m).count();
    let coverage = foreground as f32 / mask.len() as f32;
    if coverage < QUALITY.coverage {
        return Err(Rejected::Coverage {
            coverage: coverage * 100.0,
        });
    }
    let contrast = contrast_sum / foreground as f32;
    if contrast < QUALITY.contrast {
        return Err(Rejected::Contrast { contrast });
    }

    // Local normalisation removes pressure and dryness gradients.
    let mean = raw.box_mean(6);
    let centred = raw.zip(&mean, |v, m| v - m);
    let variance = centred.map(|v| v * v).box_mean(6);
    let image = centred
        .zip(&variance, |v, var| v / (var.sqrt() + 4.0))
        .gaussian(0.8);

    let pixel_orientation = orientation(&image, 3.0);
    let block_orientation = block_orientation(&pixel_orientation, blocks_x, blocks_y);
    Ok(Prepared {
        image,
        pixel_orientation,
        block_orientation,
        mask,
        blocks_x,
        blocks_y,
    })
}

fn block_std(raw: &Plane, bx: usize, by: usize) -> f32 {
    let values =
        (0..BLOCK * BLOCK).map(|i| raw.get(bx * BLOCK + i % BLOCK, by * BLOCK + i / BLOCK));
    let n = (BLOCK * BLOCK) as f32;
    let mean = values.clone().sum::<f32>() / n;
    (values.map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt()
}

/// Ridge orientation from the smoothed gradient structure tensor.
fn orientation(image: &Plane, sigma: f32) -> Orientation {
    let (w, h) = (image.width(), image.height());
    let gx = Plane::from_fn(w, h, |x, y| {
        let (x, y) = (x as isize, y as isize);
        (image.at(x + 1, y - 1) + 2.0 * image.at(x + 1, y) + image.at(x + 1, y + 1))
            - (image.at(x - 1, y - 1) + 2.0 * image.at(x - 1, y) + image.at(x - 1, y + 1))
    });
    let gy = Plane::from_fn(w, h, |x, y| {
        let (x, y) = (x as isize, y as isize);
        (image.at(x - 1, y + 1) + 2.0 * image.at(x, y + 1) + image.at(x + 1, y + 1))
            - (image.at(x - 1, y - 1) + 2.0 * image.at(x, y - 1) + image.at(x + 1, y - 1))
    });
    let gxx = gx.map(|v| v * v).gaussian(sigma);
    let gyy = gy.map(|v| v * v).gaussian(sigma);
    let gxy = gx.zip(&gy, |a, b| a * b).gaussian(sigma);
    tensor_orientation(&gxx, &gyy, &gxy)
}

fn tensor_orientation(gxx: &Plane, gyy: &Plane, gxy: &Plane) -> Orientation {
    let (w, h) = (gxx.width(), gxx.height());
    let angle = Plane::from_fn(w, h, |x, y| {
        let (a, b, c) = (gxx.get(x, y), gyy.get(x, y), gxy.get(x, y));
        // Gradient direction plus 90°: ridges run across the gradient.
        normalise_half_turn(0.5 * (2.0 * c).atan2(a - b) + FRAC_PI_2)
    });
    let coherence = Plane::from_fn(w, h, |x, y| {
        let (a, b, c) = (gxx.get(x, y), gyy.get(x, y), gxy.get(x, y));
        let trace = a + b;
        if trace <= f32::EPSILON {
            0.0
        } else {
            ((a - b).powi(2) + 4.0 * c * c).sqrt() / trace
        }
    });
    Orientation { angle, coherence }
}

/// Averages the doubled-angle vectors of each block.
fn block_orientation(pixel: &Orientation, blocks_x: usize, blocks_y: usize) -> Orientation {
    let mut angle = Plane::new(blocks_x, blocks_y);
    let mut coherence = Plane::new(blocks_x, blocks_y);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let (mut sx, mut sy, mut sc) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..BLOCK * BLOCK {
                let (x, y) = (bx * BLOCK + i % BLOCK, by * BLOCK + i / BLOCK);
                let (theta, c) = (pixel.angle.get(x, y), pixel.coherence.get(x, y));
                sx += c * (2.0 * theta).cos();
                sy += c * (2.0 * theta).sin();
                sc += c;
            }
            angle.set(bx, by, normalise_half_turn(0.5 * sy.atan2(sx)));
            coherence.set(
                bx,
                by,
                if sc > 0.0 {
                    (sx * sx + sy * sy).sqrt() / sc
                } else {
                    0.0
                },
            );
        }
    }
    Orientation { angle, coherence }
}

/// Maps an orientation (defined modulo π) into `[0, π)`.
pub(crate) fn normalise_half_turn(theta: f32) -> f32 {
    theta.rem_euclid(std::f32::consts::PI)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Synthetic fingerprint: curved ridges (period ~9 px) with phase
    /// singularities that create bifurcations and ridge endings, seen through
    /// a window shifted by (`dx`, `dy`) and turned by `rot`.
    pub(crate) fn ridges(width: usize, height: usize, dx: f32, dy: f32, rot: f32) -> Vec<u8> {
        const MINUTIAE: [(f32, f32, f32); 14] = [
            (-18.0, -25.0, 1.0),
            (12.0, -30.0, -1.0),
            (-6.0, -8.0, 1.0),
            (20.0, 4.0, -1.0),
            (-22.0, 18.0, -1.0),
            (6.0, 22.0, 1.0),
            (-2.0, 38.0, 1.0),
            (-26.0, -6.0, -1.0),
            (24.0, -14.0, 1.0),
            (-12.0, 30.0, 1.0),
            (16.0, 34.0, -1.0),
            (2.0, -36.0, 1.0),
            (-14.0, 8.0, -1.0),
            (10.0, 12.0, 1.0),
        ];
        let (s, c) = rot.sin_cos();
        (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32 - 32.0, (i / width) as f32 - 40.0);
                let (u, v) = (c * x - s * y + dx, s * x + c * y + dy);
                let r = ((u + 40.0).powi(2) * 0.6 + (v - 25.0).powi(2)).sqrt();
                let wobble = (u * 0.21).sin() * 2.0 + (v * 0.17).cos() * 1.5;
                let twist: f32 = MINUTIAE
                    .iter()
                    .map(|&(mx, my, k)| k * (v - my).atan2(u - mx))
                    .sum();
                let phase = (r + wobble) * std::f32::consts::TAU / 9.0 + twist;
                (128.0 + 90.0 * phase.sin()) as u8
            })
            .collect()
    }

    #[test]
    fn rejects_blank_frame() {
        let blank = vec![200u8; 64 * 80];
        assert!(matches!(
            prepare(&blank, 64, 80),
            Err(Rejected::Coverage { .. })
        ));
    }

    #[test]
    fn accepts_ridges_and_finds_their_direction() {
        // Horizontal ridges: intensity varies with y only.
        let horizontal: Vec<u8> = (0..64 * 80)
            .map(|i| (128.0 + 90.0 * ((i / 64) as f32 * std::f32::consts::TAU / 9.0).sin()) as u8)
            .collect();
        let p = prepare(&horizontal, 64, 80).unwrap();
        let theta = p.block_orientation.angle.get(4, 5);
        let off = theta.min(std::f32::consts::PI - theta);
        assert!(off < 0.1, "ridge angle {theta}");
        assert!(p.block_orientation.coherence.get(4, 5) > 0.9);
    }
}
