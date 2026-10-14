// SPDX-License-Identifier: GPL-2.0-only
//! Minimal safe surface over the C SPI core for one bound `spi_device`.

use crate::sys;
use core::ptr::{addr_of_mut, NonNull};
use kernel::{error::to_result, prelude::*};

/// `SPI_CS_HIGH` from `include/uapi/linux/spi/spi.h`.
pub(crate) const SPI_CS_HIGH: u32 = 1 << 2;

/// A `spi_device` this driver is bound to.
///
/// # Invariants
///
/// Points to a live `spi_device` for as long as the driver stays bound to it;
/// the owner ([`crate::sensor::Sensor`]) drops it in `remove()` at the latest.
pub(crate) struct SpiDevice(NonNull<sys::spi_device>);

// SAFETY: the SPI core serialises messages internally; `spi_device` may be
// used from any thread.
unsafe impl Send for SpiDevice {}

impl SpiDevice {
    /// # Safety
    ///
    /// `spi` must stay valid until the returned value is dropped.
    pub(crate) unsafe fn new(spi: NonNull<sys::spi_device>) -> Self {
        Self(spi)
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::spi_device {
        self.0.as_ptr()
    }

    pub(crate) fn device(&self) -> *mut sys::device {
        // SAFETY: valid by the type invariant.
        unsafe { addr_of_mut!((*self.as_ptr()).dev) }
    }

    pub(crate) fn speed_hz(&self) -> u32 {
        // SAFETY: valid by the type invariant; plain read of a config field.
        unsafe { (*self.as_ptr()).max_speed_hz }
    }

    pub(crate) fn mode(&self) -> u32 {
        // SAFETY: as above.
        unsafe { (*self.as_ptr()).mode }
    }

    pub(crate) fn irq(&self) -> Result<u32> {
        // SAFETY: as above.
        let irq = unsafe { (*self.as_ptr()).irq };
        u32::try_from(irq).ok().filter(|&irq| irq != 0).ok_or(ENXIO)
    }

    /// Adds `mode` bits on top of what firmware described and applies them.
    pub(crate) fn setup(&self, extra_mode: u32) -> Result {
        // SAFETY: valid by the type invariant; probe context owns the device
        // configuration, nobody else issues transfers yet.
        unsafe {
            (*self.as_ptr()).mode |= extra_mode;
            (*self.as_ptr()).bits_per_word = 8;
            to_result(sys::spi_setup(self.as_ptr()))
        }
    }

    /// One full-duplex transfer with chip select held for its whole length.
    ///
    /// Both buffers must be DMA-safe (`kmalloc`ed), which `KVec` guarantees.
    pub(crate) fn transfer(&self, tx: &KVec<u8>, rx: &mut KVec<u8>) -> Result {
        let len = u32::try_from(tx.len()).map_err(|_| EINVAL)?;
        if rx.len() != tx.len() {
            return Err(EINVAL);
        }
        let mut xfer = sys::spi_transfer {
            tx_buf: tx.as_ptr().cast(),
            rx_buf: rx.as_mut_ptr().cast(),
            len,
            ..Default::default()
        };
        let mut msg = sys::spi_message::default();
        // SAFETY: `msg` and `xfer` live on this stack frame and are neither
        // moved nor touched again until `spi_sync` has completed the message;
        // this is `spi_message_init()` + `spi_message_add_tail()`.
        unsafe {
            list_init(&raw mut msg.transfers);
            list_init(&raw mut msg.resources);
            list_add_tail(&raw mut xfer.transfer_list, &raw mut msg.transfers);
            to_result(sys::spi_sync(self.as_ptr(), &raw mut msg))
        }
    }
}

/// `INIT_LIST_HEAD()`.
///
/// # Safety
///
/// `head` must be valid for writes.
unsafe fn list_init(head: *mut sys::list_head) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        (*head).next = head;
        (*head).prev = head;
    }
}

/// `list_add_tail()`.
///
/// # Safety
///
/// `head` must be an initialised list and `new` valid for writes.
unsafe fn list_add_tail(new: *mut sys::list_head, head: *mut sys::list_head) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let prev = (*head).prev;
        (*new).next = head;
        (*new).prev = prev;
        (*prev).next = new;
        (*head).prev = new;
    }
}
