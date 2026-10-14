//! A captured fingerprint frame. Biometric data: never logged, wiped on drop.

use zeroize::{Zeroize, ZeroizeOnDrop};

/// One 8-bit grayscale frame from the sensor.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Image {
    pixels: Box<[u8; Self::WIDTH * Self::HEIGHT]>,
}

impl Image {
    /// Columns of the FT9368 matrix.
    pub const WIDTH: usize = 64;
    /// Rows of the FT9368 matrix.
    pub const HEIGHT: usize = 80;

    pub(crate) fn blank() -> Self {
        Self {
            pixels: Box::new([0; Self::WIDTH * Self::HEIGHT]),
        }
    }

    pub(crate) fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels[..]
    }

    /// Row-major pixels, [`Self::WIDTH`] per row.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels[..]
    }

    /// Pixel at (`x`, `y`).
    ///
    /// # Panics
    ///
    /// If the coordinates are outside the frame.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        assert!(
            x < Self::WIDTH && y < Self::HEIGHT,
            "pixel ({x}, {y}) out of frame"
        );
        self.pixels[y * Self::WIDTH + x]
    }
}

/// Deliberately says nothing about the pixels.
impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &Self::WIDTH)
            .field("height", &Self::HEIGHT)
            .finish_non_exhaustive()
    }
}
