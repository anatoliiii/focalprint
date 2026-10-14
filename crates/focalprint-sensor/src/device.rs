//! `/dev/focaltech-fp`: raw SPI transfers, reset and interrupt waits.

use crate::{Error, Interrupt, Result, Transport, uapi};
use rustix::{
    fs::{Mode, OFlags},
    io::Errno,
    ioctl::{Getter, NoArg, Setter, Updater, ioctl},
};
use std::{os::fd::OwnedFd, path::Path, time::Duration};

/// Default device node created by the kernel module.
pub const DEFAULT_PATH: &str = "/dev/focaltech-fp";

/// Static facts about the bound sensor, as reported by the kernel module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Largest transfer the kernel accepts, in bytes.
    pub max_transfer: usize,
    /// SPI clock.
    pub speed_hz: u32,
    /// SPI mode bits (`SPI_CS_HIGH` etc.).
    pub spi_mode: u32,
}

/// An open sensor. Only one process can hold it at a time.
#[derive(Debug)]
pub struct Device {
    fd: OwnedFd,
    info: DeviceInfo,
}

impl Device {
    /// Opens the sensor and checks that the kernel speaks our ABI.
    ///
    /// # Errors
    ///
    /// [`Error::Busy`] if another process has it open, [`Error::Abi`] on a
    /// kernel/userspace version mismatch, [`Error::Io`] otherwise.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let fd = rustix::fs::open(path.as_ref(), OFlags::RDWR | OFlags::CLOEXEC, Mode::empty())
            .map_err(|errno| match errno {
                Errno::BUSY => Error::Busy,
                errno => errno.into(),
            })?;
        // SAFETY: `IOC_INFO` is a read ioctl producing a `uapi::Info`.
        let raw = unsafe { ioctl(&fd, Getter::<{ uapi::IOC_INFO }, uapi::Info>::new()) }?;
        if raw.abi != uapi::ABI_VERSION {
            return Err(Error::Abi {
                kernel: raw.abi,
                expected: uapi::ABI_VERSION,
            });
        }
        let info = DeviceInfo {
            max_transfer: raw.max_transfer as usize,
            speed_hz: raw.speed_hz,
            spi_mode: raw.mode,
        };
        Ok(Self { fd, info })
    }

    /// What the kernel reported at open time.
    #[must_use]
    pub fn info(&self) -> DeviceInfo {
        self.info
    }

    fn wait_irq_raw(&self, seen: u64, timeout_ms: u32) -> Result<u64> {
        let mut wait = uapi::WaitIrq {
            seen,
            timeout_ms,
            reserved: 0,
        };
        // SAFETY: `IOC_WAIT_IRQ` reads and writes back a `uapi::WaitIrq`.
        unsafe {
            ioctl(
                &self.fd,
                Updater::<{ uapi::IOC_WAIT_IRQ }, _>::new(&mut wait),
            )
        }?;
        Ok(wait.seen)
    }
}

impl Transport for Device {
    fn transfer(&mut self, frame: &mut [u8]) -> Result<()> {
        if frame.is_empty() || frame.len() > self.info.max_transfer {
            return Err(Error::Io(Errno::INVAL));
        }
        let ptr = frame.as_mut_ptr() as u64;
        let request = uapi::Transfer {
            tx: ptr,
            rx: ptr,
            len: u32::try_from(frame.len()).map_err(|_| Errno::INVAL)?,
            reserved: 0,
        };
        // SAFETY: `IOC_TRANSFER` takes a `uapi::Transfer`; the kernel reads
        // `len` bytes from `tx` before writing `len` bytes to `rx`, both of
        // which point into `frame`, exclusively borrowed for this call.
        unsafe { ioctl(&self.fd, Setter::<{ uapi::IOC_TRANSFER }, _>::new(request)) }?;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // SAFETY: `IOC_RESET` takes no argument.
        unsafe { ioctl(&self.fd, NoArg::<{ uapi::IOC_RESET }>::new()) }?;
        Ok(())
    }

    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

impl Interrupt for Device {
    fn irq_count(&mut self) -> Result<u64> {
        self.wait_irq_raw(0, 0)
    }

    fn wait_irq(&mut self, seen: u64, timeout: Duration) -> Result<Option<u64>> {
        let timeout_ms = u32::try_from(timeout.as_millis())
            .unwrap_or(u32::MAX)
            .max(1);
        match self.wait_irq_raw(seen, timeout_ms) {
            Err(Error::Io(Errno::TIMEDOUT)) => Ok(None),
            other => other.map(Some),
        }
    }
}
