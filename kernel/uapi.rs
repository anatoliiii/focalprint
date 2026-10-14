// SPDX-License-Identifier: GPL-2.0-only
//! Userspace ABI of `/dev/focaltech-fp`. Mirrored by `crates/focalprint-sensor/src/uapi.rs`;
//! keep both in sync and bump [`ABI_VERSION`] on any change.

use kernel::{
    ioctl::{_IO, _IOR, _IOW, _IOWR},
    transmute::{AsBytes, FromBytes},
};

pub(crate) const ABI_VERSION: u32 = 1;

/// Largest single full-duplex transfer: a 5120-byte image frame plus header.
pub(crate) const MAX_TRANSFER: u32 = 8192;

const MAGIC: u32 = 0xF4;

pub(crate) const IOC_INFO: u32 = _IOR::<Info>(MAGIC, 0);
pub(crate) const IOC_RESET: u32 = _IO(MAGIC, 1);
pub(crate) const IOC_TRANSFER: u32 = _IOW::<Transfer>(MAGIC, 2);
pub(crate) const IOC_WAIT_IRQ: u32 = _IOWR::<WaitIrq>(MAGIC, 3);

/// `IOC_INFO` result.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Info {
    pub(crate) abi: u32,
    pub(crate) max_transfer: u32,
    pub(crate) speed_hz: u32,
    pub(crate) mode: u32,
}

/// `IOC_TRANSFER` argument: full duplex, `len` bytes out of `tx`, into `rx`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Transfer {
    pub(crate) tx: u64,
    pub(crate) rx: u64,
    pub(crate) len: u32,
    pub(crate) reserved: u32,
}

/// `IOC_WAIT_IRQ` argument. Blocks until the interrupt counter differs from
/// `seen` (or `timeout_ms` elapses; 0 = don't wait) and returns it in `seen`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct WaitIrq {
    pub(crate) seen: u64,
    pub(crate) timeout_ms: u32,
    pub(crate) reserved: u32,
}

// SAFETY: plain `repr(C)` integers without padding; every bit pattern is valid.
unsafe impl FromBytes for Transfer {}
// SAFETY: as above.
unsafe impl FromBytes for WaitIrq {}
// SAFETY: as above, and no padding bytes can leak kernel memory.
unsafe impl AsBytes for Info {}
// SAFETY: as above.
unsafe impl AsBytes for WaitIrq {}
