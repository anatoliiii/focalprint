//! FocalTech fingerprint sensors: the FTE4800 (FT9368, SPI) of the TECNO
//! MEGABOOK T1 and the FT9201 (USB) of the MegaBook K16S.
//!
//! [`Device`] talks to the `focaltech_spi_fp` kernel module; [`Ft9368`] implements the
//! SPI protocol on top of any [`Transport`]. [`Ft9201`] implements the USB
//! protocol on top of any [`ft9201::Link`], [`UsbLink`] being the real one.
//! Both can be tested without hardware. [`open`] picks whichever is present.

mod capture;
mod device;
pub mod ft9201;
pub mod ft9368;
mod image;
mod sensor;
mod uapi;
#[cfg(feature = "usb")]
pub mod usb;

pub use capture::Wait;
pub use device::{DEFAULT_PATH, Device, DeviceInfo};
pub use ft9201::Ft9201;
pub use ft9368::{Ft9368, Identity, Status};
pub use image::Image;
#[cfg(feature = "usb")]
pub use sensor::FT9201_PIXELS;
pub use sensor::{Backend, SPI_ACPI_NODE, Sensor, open};
#[cfg(feature = "usb")]
pub use usb::UsbLink;

use std::time::Duration;

/// Errors from the device or the protocol.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A system call failed.
    #[error(transparent)]
    Io(#[from] rustix::io::Errno),
    /// Another process (usually `focalprintd`) has the sensor open.
    #[error("the sensor is in use by another process")]
    Busy,
    /// The kernel module and this library disagree on the ABI.
    #[error("kernel module speaks ABI {kernel}, expected {expected}")]
    Abi { kernel: u32, expected: u32 },
    /// No supported sensor is present.
    #[error("no supported fingerprint sensor found")]
    NoSensor,
    /// USB access failed.
    #[error("USB: {0}")]
    Usb(#[source] std::io::Error),
    /// No FT9368 answered.
    #[error("no FT9368 found (chip id {chip_id:#06x})")]
    NotFound { chip_id: u16 },
    /// The sensor said something we don't understand.
    #[error("protocol error: {0}")]
    Protocol(&'static str),
    /// Nothing happened within the allowed time.
    #[error("timed out")]
    Timeout,
    /// The caller gave up.
    #[error("cancelled")]
    Cancelled,
}

/// Result alias for this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// How the protocol reaches the sensor.
pub trait Transport {
    /// One full-duplex SPI frame, replaced in place by what came back.
    ///
    /// # Errors
    ///
    /// Transport-specific.
    fn transfer(&mut self, frame: &mut [u8]) -> Result<()>;

    /// Pulses the hardware reset line.
    ///
    /// # Errors
    ///
    /// Transport-specific.
    fn reset(&mut self) -> Result<()>;

    /// Waits; overridable so tests don't actually sleep.
    fn sleep(&mut self, duration: Duration);
}

/// The sensor's touch interrupt.
pub trait Interrupt {
    /// Current interrupt counter, without waiting.
    ///
    /// # Errors
    ///
    /// Transport-specific.
    fn irq_count(&mut self) -> Result<u64>;

    /// Waits until the counter moves past `seen`; `None` after `timeout`.
    ///
    /// # Errors
    ///
    /// Transport-specific.
    fn wait_irq(&mut self, seen: u64, timeout: Duration) -> Result<Option<u64>>;
}
