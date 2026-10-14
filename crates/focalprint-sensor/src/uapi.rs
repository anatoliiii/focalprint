//! ABI of `/dev/focaltech-fp`, mirrored from `kernel/uapi.rs`.

use rustix::ioctl::{Opcode, opcode};

pub(crate) const ABI_VERSION: u32 = 1;

const MAGIC: u8 = 0xF4;

pub(crate) const IOC_INFO: Opcode = opcode::read::<Info>(MAGIC, 0);
pub(crate) const IOC_RESET: Opcode = opcode::none(MAGIC, 1);
pub(crate) const IOC_TRANSFER: Opcode = opcode::write::<Transfer>(MAGIC, 2);
pub(crate) const IOC_WAIT_IRQ: Opcode = opcode::read_write::<WaitIrq>(MAGIC, 3);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Info {
    pub(crate) abi: u32,
    pub(crate) max_transfer: u32,
    pub(crate) speed_hz: u32,
    pub(crate) mode: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Transfer {
    pub(crate) tx: u64,
    pub(crate) rx: u64,
    pub(crate) len: u32,
    pub(crate) reserved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WaitIrq {
    pub(crate) seen: u64,
    pub(crate) timeout_ms: u32,
    pub(crate) reserved: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same numbers the kernel computes with `_IOR`/`_IO`/`_IOW`/`_IOWR`.
    #[test]
    fn opcodes_match_kernel() {
        assert_eq!(IOC_INFO, 0x8010_f400);
        assert_eq!(IOC_RESET, 0x0000_f401);
        assert_eq!(IOC_TRANSFER, 0x4018_f402);
        assert_eq!(IOC_WAIT_IRQ, 0xc010_f403);
    }
}
