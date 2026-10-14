//! A dense 2-D grid of samples, wiped on drop (it is derived biometric data).

use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub(crate) struct Plane {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Plane {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub(crate) fn from_fn(
        width: usize,
        height: usize,
        mut f: impl FnMut(usize, usize) -> f32,
    ) -> Self {
        let mut plane = Self::new(width, height);
        for y in 0..height {
            for x in 0..width {
                plane.data[y * width + x] = f(x, y);
            }
        }
        plane
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    pub(crate) fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }

    /// Clamped-to-edge lookup with signed coordinates.
    pub(crate) fn at(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.get(x, y)
    }

    /// Bilinear sample; `None` outside the plane.
    pub(crate) fn sample(&self, x: f32, y: f32) -> Option<f32> {
        let max_x = (self.width - 1) as f32;
        let max_y = (self.height - 1) as f32;
        if !(0.0..=max_x).contains(&x) || !(0.0..=max_y).contains(&y) {
            return None;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let top = self.get(x0, y0) * (1.0 - fx) + self.get(x1, y0) * fx;
        let bottom = self.get(x0, y1) * (1.0 - fx) + self.get(x1, y1) * fx;
        Some(top * (1.0 - fy) + bottom * fy)
    }

    /// Mean over a `(2r+1)²` window, via an integral image.
    pub(crate) fn box_mean(&self, r: usize) -> Self {
        let (w, h) = (self.width, self.height);
        let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0.0f64;
            for x in 0..w {
                row += f64::from(self.get(x, y));
                integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row;
            }
        }
        let out = Self::from_fn(w, h, |x, y| {
            let (x0, y0) = (x.saturating_sub(r), y.saturating_sub(r));
            let (x1, y1) = ((x + r + 1).min(w), (y + r + 1).min(h));
            let sum = integral[y1 * (w + 1) + x1]
                - integral[y0 * (w + 1) + x1]
                - integral[y1 * (w + 1) + x0]
                + integral[y0 * (w + 1) + x0];
            (sum / ((x1 - x0) * (y1 - y0)) as f64) as f32
        });
        integral.zeroize();
        out
    }

    /// Separable Gaussian blur.
    pub(crate) fn gaussian(&self, sigma: f32) -> Self {
        let r = (sigma * 3.0).ceil() as isize;
        let kernel: Vec<f32> = (-r..=r)
            .map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp())
            .collect();
        let norm: f32 = kernel.iter().sum();
        let pass = |src: &Self, dx: isize, dy: isize| {
            Self::from_fn(src.width, src.height, |x, y| {
                kernel
                    .iter()
                    .zip(-r..=r)
                    .map(|(k, i)| k * src.at(x as isize + i * dx, y as isize + i * dy))
                    .sum::<f32>()
                    / norm
            })
        };
        pass(&pass(self, 1, 0), 0, 1)
    }

    pub(crate) fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self::from_fn(self.width, self.height, |x, y| f(self.get(x, y)))
    }

    pub(crate) fn zip(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Self {
        Self::from_fn(self.width, self.height, |x, y| {
            f(self.get(x, y), other.get(x, y))
        })
    }
}
