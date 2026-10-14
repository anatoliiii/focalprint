//! What the daemon needs from any supported sensor, and picking one.

use crate::{Device, Ft9368, Image, Result, Wait};
use std::{fmt, path::Path, str::FromStr};

/// A touch sensor that delivers 64×80 8-bit frames.
pub trait Sensor: Send + fmt::Debug {
    /// Waits for a finger and returns its frame.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Timeout`], [`crate::Error::Cancelled`], or device/protocol failures.
    fn capture(&mut self, wait: Wait<'_>) -> Result<Image>;

    /// Waits until the finger has been lifted.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Timeout`], [`crate::Error::Cancelled`], or device/protocol failures.
    fn wait_release(&mut self, wait: Wait<'_>) -> Result<()>;

    /// Reads a frame right now, finger or not (sensor background).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Protocol`] where the sensor can't do that.
    fn read_now(&mut self) -> Result<Image>;

    /// Model and firmware, for logs.
    fn describe(&self) -> &str;
}

/// Which sensor family to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The SPI sensor if its device node exists, otherwise USB.
    #[default]
    Auto,
    /// FTE4800 / FT9368 through the `focaltech_spi_fp` kernel module.
    Spi,
    /// FT9201 over USB.
    Usb,
}

/// ACPI node of the SPI sensor; exists whether or not a driver is bound.
pub const SPI_ACPI_NODE: &str = "/sys/bus/acpi/devices/FTE4800:00";

impl Backend {
    /// Which supported sensor this machine has, without opening it: the
    /// SPI sensor by its ACPI node, the USB one by its vendor/product id.
    #[must_use]
    pub fn detect() -> Option<Self> {
        if Path::new(SPI_ACPI_NODE).exists() {
            return Some(Self::Spi);
        }
        #[cfg(feature = "usb")]
        if crate::usb::present() {
            return Some(Self::Usb);
        }
        None
    }

    /// Settles [`Backend::Auto`] by looking for the SPI device node.
    #[must_use]
    pub fn resolve(self, spi_path: &Path) -> Self {
        match self {
            Self::Auto if spi_path.exists() => Self::Spi,
            Self::Auto => Self::Usb,
            other => other,
        }
    }
}

impl FromStr for Backend {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "spi" => Ok(Self::Spi),
            "usb" => Ok(Self::Usb),
            _ => Err("expected auto, spi or usb"),
        }
    }
}

#[derive(Debug)]
struct Labeled<S> {
    sensor: S,
    label: String,
}

impl Sensor for Labeled<Ft9368<Device>> {
    fn capture(&mut self, wait: Wait<'_>) -> Result<Image> {
        self.sensor.capture(wait)
    }
    fn wait_release(&mut self, wait: Wait<'_>) -> Result<()> {
        self.sensor.wait_release(wait)
    }
    fn read_now(&mut self) -> Result<Image> {
        self.sensor.read_image()
    }
    fn describe(&self) -> &str {
        &self.label
    }
}

#[cfg(feature = "usb")]
impl Sensor for Labeled<crate::Ft9201<crate::UsbLink>> {
    fn capture(&mut self, wait: Wait<'_>) -> Result<Image> {
        self.sensor.capture(wait)
    }
    fn wait_release(&mut self, wait: Wait<'_>) -> Result<()> {
        self.sensor.wait_release(wait)
    }
    fn read_now(&mut self) -> Result<Image> {
        Err(crate::Error::Protocol(
            "FT9201 only delivers frames for a finger",
        ))
    }
    fn describe(&self) -> &str {
        &self.label
    }
}

fn open_spi(path: &Path) -> Result<Box<dyn Sensor>> {
    let mut sensor = Ft9368::new(Device::open(path)?);
    let identity = sensor.identify()?;
    Ok(Box::new(Labeled {
        sensor,
        label: format!("FT9368 (SPI), firmware {:#04x}", identity.firmware),
    }))
}

/// FT9201 pixel decoding, pending hardware confirmation: environment
/// variable [`FT9201_PIXELS`], comma-separated `le`/`be` (16-bit byte order,
/// default `le`) and `invert` (ridges come out bright).
#[cfg(feature = "usb")]
pub const FT9201_PIXELS: &str = "FOCALPRINT_FT9201_PIXELS";

#[cfg(feature = "usb")]
fn ft9201_pixels() -> Result<(crate::ft9201::PixelOrder, bool)> {
    use crate::ft9201::PixelOrder;
    let mut setting = (PixelOrder::Little, false);
    for word in std::env::var(FT9201_PIXELS).unwrap_or_default().split(',') {
        match word.trim() {
            "" => {}
            "le" => setting.0 = PixelOrder::Little,
            "be" => setting.0 = PixelOrder::Big,
            "invert" => setting.1 = true,
            _ => return Err(crate::Error::Protocol("bad FOCALPRINT_FT9201_PIXELS")),
        }
    }
    Ok(setting)
}

#[cfg(feature = "usb")]
fn open_usb() -> Result<Box<dyn Sensor>> {
    let (order, invert) = ft9201_pixels()?;
    let mut sensor = crate::Ft9201::new(crate::UsbLink::open()?)
        .with_pixel_order(order)
        .with_inverted(invert);
    let identity = sensor.identify()?;
    Ok(Box::new(Labeled {
        sensor,
        label: format!("FT9201 (USB), id {:#06x}", identity.id),
    }))
}

#[cfg(not(feature = "usb"))]
fn open_usb() -> Result<Box<dyn Sensor>> {
    Err(crate::Error::NoSensor)
}

/// Opens and initialises a sensor. `spi_path` is the kernel module's device
/// node.
///
/// # Errors
///
/// [`crate::Error::NoSensor`] if nothing supported is present, or the backend's
/// own open/identify errors.
pub fn open(backend: Backend, spi_path: &Path) -> Result<Box<dyn Sensor>> {
    match backend.resolve(spi_path) {
        Backend::Spi => open_spi(spi_path),
        Backend::Usb | Backend::Auto => open_usb(),
    }
}
