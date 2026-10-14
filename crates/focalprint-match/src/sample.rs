//! What is kept of one touch: features and a coarse ridge-flow map. No
//! pixels survive extraction.

use crate::{
    features::{self, Feature, Params},
    prep::{self, Rejected},
};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Fewer usable features than this make a comparison meaningless.
const MIN_FEATURES: usize = 12;

/// Features extracted from one frame.
#[derive(Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct Sample {
    pub(crate) features: Vec<Feature>,
    blocks_x: u8,
    blocks_y: u8,
    /// Per block: ridge angle quantised to `0..=254` over `[0, π)`,
    /// or [`BACKGROUND`].
    orientation: Vec<u8>,
    /// Per block: coherence quantised to `0..=255`.
    coherence: Vec<u8>,
}

const BACKGROUND: u8 = 255;

impl Sample {
    /// Extracts a sample from an 8-bit grayscale frame.
    ///
    /// # Errors
    ///
    /// [`Rejected`] if the frame is unusable; the caller should ask for
    /// another touch.
    ///
    /// # Panics
    ///
    /// If `pixels.len() != width * height` or the frame is larger than
    /// 2040 × 2040 pixels.
    pub fn extract(pixels: &[u8], width: usize, height: usize) -> Result<Self, Rejected> {
        Self::extract_with(pixels, width, height, &Params::DEFAULT)
    }

    /// [`Self::extract`] with non-default detector settings.
    ///
    /// # Errors
    ///
    /// As [`Self::extract`].
    ///
    /// # Panics
    ///
    /// As [`Self::extract`].
    pub fn extract_with(
        pixels: &[u8],
        width: usize,
        height: usize,
        params: &Params,
    ) -> Result<Self, Rejected> {
        let prepared = prep::prepare(pixels, width, height)?;
        let features = features::extract(&prepared, params);
        if features.len() < MIN_FEATURES {
            return Err(Rejected::Features {
                found: features.len(),
            });
        }
        let blocks = prepared.blocks_x * prepared.blocks_y;
        let mut orientation = Vec::with_capacity(blocks);
        let mut coherence = Vec::with_capacity(blocks);
        for by in 0..prepared.blocks_y {
            for bx in 0..prepared.blocks_x {
                let fg = prepared.mask[by * prepared.blocks_x + bx];
                let angle = prepared.block_orientation.angle.get(bx, by);
                let c = prepared.block_orientation.coherence.get(bx, by);
                orientation.push(if fg {
                    quantise(angle / PI, 254.0)
                } else {
                    BACKGROUND
                });
                coherence.push(if fg { quantise(c, 255.0) } else { 0 });
            }
        }
        Ok(Self {
            features,
            blocks_x: u8::try_from(prepared.blocks_x).expect("frame too large"),
            blocks_y: u8::try_from(prepared.blocks_y).expect("frame too large"),
            orientation,
            coherence,
        })
    }

    /// Number of features, for diagnostics.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    pub(crate) fn blocks_x(&self) -> usize {
        self.blocks_x.into()
    }

    pub(crate) fn blocks_y(&self) -> usize {
        self.blocks_y.into()
    }

    /// Ridge angle and coherence of a foreground block.
    pub(crate) fn block(&self, bx: usize, by: usize) -> Option<(f32, f32)> {
        if bx >= self.blocks_x() || by >= self.blocks_y() {
            return None;
        }
        let i = by * self.blocks_x() + bx;
        let angle = *self.orientation.get(i)?;
        (angle != BACKGROUND).then(|| {
            (
                f32::from(angle) / 254.0 * PI,
                f32::from(self.coherence[i]) / 255.0,
            )
        })
    }

    /// Structural sanity after deserialisation.
    pub(crate) fn is_consistent(&self) -> bool {
        let blocks = self.blocks_x() * self.blocks_y();
        self.orientation.len() == blocks
            && self.coherence.len() == blocks
            && self.features.len() >= MIN_FEATURES
            && self
                .features
                .iter()
                .all(|f| f.x.is_finite() && f.y.is_finite() && f.angle.is_finite())
    }
}

fn quantise(unit: f32, scale: f32) -> u8 {
    (unit.clamp(0.0, 1.0) * scale).round() as u8
}

/// Deliberately says nothing about the features.
impl std::fmt::Debug for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sample")
            .field("features", &self.features.len())
            .finish_non_exhaustive()
    }
}
