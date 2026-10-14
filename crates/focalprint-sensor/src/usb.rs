//! FT9201 over usbfs, through `nusb` (no libusb, no kernel module).

use crate::{Error, Result, ft9201::Link};
use nusb::{
    Endpoint, Interface, MaybeFuture as _,
    transfer::{Buffer, Bulk, In, Out},
};
use std::time::Duration;

/// FocalTech FT9201 (enumerates as HOLTEK).
pub const VENDOR_ID: u16 = 0x2808;
/// FocalTech FT9201 (enumerates as HOLTEK).
pub const PRODUCT_ID: u16 = 0xc652;

const INTERFACE: u8 = 0;
const EP_OUT: u8 = 0x03;
const EP_IN: u8 = 0x81;
const TIMEOUT: Duration = Duration::from_secs(2);

fn usb(err: impl Into<std::io::Error>) -> Error {
    Error::Usb(err.into())
}

/// Whether an FT9201 is plugged in (does not open it).
#[must_use]
pub fn present() -> bool {
    nusb::list_devices().wait().is_ok_and(|mut devices| {
        devices.any(|d| d.vendor_id() == VENDOR_ID && d.product_id() == PRODUCT_ID)
    })
}

/// The claimed sensor interface. Only one process can hold it at a time.
#[derive(Debug)]
pub struct UsbLink {
    out: Endpoint<Bulk, Out>,
    inp: Endpoint<Bulk, In>,
    _interface: Interface,
}

impl UsbLink {
    /// Finds the first FT9201 and claims its interface.
    ///
    /// # Errors
    ///
    /// [`Error::NoSensor`] if none is plugged in, [`Error::Busy`] if another
    /// process (fprintd with the vendor driver) holds it.
    pub fn open() -> Result<Self> {
        let info = nusb::list_devices()
            .wait()
            .map_err(usb)?
            .find(|d| d.vendor_id() == VENDOR_ID && d.product_id() == PRODUCT_ID)
            .ok_or(Error::NoSensor)?;
        let device = info.open().wait().map_err(usb)?;
        let interface = device
            .detach_and_claim_interface(INTERFACE)
            .wait()
            .map_err(|e| match e.kind() {
                nusb::ErrorKind::Busy => Error::Busy,
                _ => usb(e),
            })?;
        Ok(Self {
            out: interface.endpoint(EP_OUT).map_err(usb)?,
            inp: interface.endpoint(EP_IN).map_err(usb)?,
            _interface: interface,
        })
    }
}

impl Link for UsbLink {
    fn exchange(&mut self, request: &[u8], reply: &mut [u8]) -> Result<usize> {
        self.out
            .transfer_blocking(Buffer::from(request), TIMEOUT)
            .status
            .map_err(usb)?;
        // IN requests must be whole packets; the reply ends with a short one.
        let packet = self.inp.max_packet_size();
        let mut done = self
            .inp
            .transfer_blocking(Buffer::new(reply.len().div_ceil(packet) * packet), TIMEOUT);
        let got = done.buffer.len();
        let fits = got <= reply.len();
        if fits {
            reply[..got].copy_from_slice(&done.buffer);
        }
        // May hold image data.
        done.buffer.fill(0);
        done.status.map_err(usb)?;
        if fits {
            Ok(got)
        } else {
            Err(Error::Protocol("reply longer than expected"))
        }
    }

    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}
