//! FpNet inference with `tract` (pure Rust, CPU): a 64×80 frame becomes a
//! 40×32 grid of L2-normalised skin descriptors plus per-cell reliability.

// Grid/pixel arithmetic on a 64×80 frame: index/float casts cannot overflow,
// and x/y/a/b/s are the customary names in geometry code.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::missing_asserts_for_indexing
)]

pub mod matching;

use std::path::Path;
use tract_onnx::prelude::*;
use zeroize::Zeroize;

/// Frame size the network was exported for.
pub const WIDTH: usize = 64;
pub const HEIGHT: usize = 80;
/// Output grid (half resolution).
pub const GRID_W: usize = WIDTH / 2;
pub const GRID_H: usize = HEIGHT / 2;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model: {0}")]
    Model(String),
}

impl From<TractError> for Error {
    fn from(e: TractError) -> Self {
        Self::Model(e.to_string())
    }
}

/// Dense descriptors of one frame. Derived biometric data: wiped on drop.
pub struct DenseMap {
    /// Descriptor dimension.
    pub dim: usize,
    /// `dim × GRID_H × GRID_W`, channel-major, each cell L2-normalised.
    pub descriptors: Vec<f32>,
    /// `GRID_H × GRID_W` in `[0, 1]`.
    pub reliability: Vec<f32>,
}

impl Drop for DenseMap {
    fn drop(&mut self) {
        self.descriptors.zeroize();
        self.reliability.zeroize();
    }
}

impl std::fmt::Debug for DenseMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseMap")
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

type Plan = std::sync::Arc<TypedRunnableModel>;

/// A loaded, optimised network.
pub struct FpNet {
    plan: Plan,
}

impl std::fmt::Debug for FpNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FpNet")
    }
}

impl FpNet {
    /// Loads an ONNX export with input `frame: f32[1,1,80,64]`.
    ///
    /// # Errors
    ///
    /// Unreadable or incompatible model.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, 1, HEIGHT, WIDTH]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan })
    }

    /// Runs the network on an 8-bit frame (normalised like in training:
    /// zero mean, unit variance over the frame).
    ///
    /// # Errors
    ///
    /// Inference failures (never expected for a valid frame).
    ///
    /// # Panics
    ///
    /// If `pixels.len() != WIDTH * HEIGHT`.
    pub fn run(&self, pixels: &[u8]) -> Result<DenseMap, Error> {
        assert_eq!(pixels.len(), WIDTH * HEIGHT, "wrong frame size");
        let n = pixels.len() as f32;
        let mean = pixels.iter().map(|&p| f32::from(p)).sum::<f32>() / n;
        let var = pixels
            .iter()
            .map(|&p| (f32::from(p) - mean).powi(2))
            .sum::<f32>()
            / (n - 1.0);
        let std = var.sqrt() + 1e-3;
        let input: Tensor =
            tract_ndarray::Array4::from_shape_fn((1, 1, HEIGHT, WIDTH), |(_, _, y, x)| {
                (f32::from(pixels[y * WIDTH + x]) - mean) / std
            })
            .into();
        let out = self.plan.run(tvec!(input.into()))?;
        let desc = out[0].to_plain_array_view::<f32>()?;
        let rel = out[1].to_plain_array_view::<f32>()?;
        let dim = desc.shape()[1];
        Ok(DenseMap {
            dim,
            descriptors: desc.iter().copied().collect(),
            reliability: rel.iter().copied().collect(),
        })
    }
}
