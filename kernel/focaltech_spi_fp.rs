// SPDX-License-Identifier: GPL-2.0-only
//! FocalTech FTE4800 (FT9368) fingerprint sensor, as found in the TECNO
//! MEGABOOK T1: SPI transport, reset and interrupt for `focalprintd`.

use core::ptr::NonNull;
use kernel::{error::to_result, prelude::*, ThisModule};

mod reset;
mod sensor;
mod spi;
mod sys;
mod uapi;

module! {
    type: FocaltechSpiFp,
    name: "focaltech_spi_fp",
    authors: ["Mozgolika"],
    description: "FocalTech FTE4800 (FT9368) fingerprint sensor",
    license: "GPL",
    alias: ["acpi*:FTE4800:*"],
}

static ACPI_IDS: [sys::acpi_device_id; 2] = [acpi_id(b"FTE4800"), acpi_id(b"")];

const fn acpi_id(hid: &[u8]) -> sys::acpi_device_id {
    let mut id = [0u8; 16];
    let mut i = 0;
    while i < hid.len() {
        id[i] = hid[i];
        i += 1;
    }
    sys::acpi_device_id {
        id,
        driver_data: 0,
        cls: 0,
        cls_msk: 0,
    }
}

unsafe extern "C" fn probe(spi: *mut sys::spi_device) -> i32 {
    let Some(spi) = NonNull::new(spi) else {
        return EINVAL.to_errno();
    };
    // SAFETY: the SPI core keeps `spi` alive while we are bound, and the
    // `Sensor` drops its handle in `remove()` at the latest.
    let device = unsafe { spi::SpiDevice::new(spi) };
    match sensor::Sensor::probe(device, &THIS_MODULE) {
        Ok(sensor) => {
            sensor::store(spi, sensor);
            pr_info!("FT9368 ready on /dev/focaltech-fp\n");
            0
        }
        Err(err) => err.to_errno(),
    }
}

unsafe extern "C" fn remove(spi: *mut sys::spi_device) {
    if let Some(sensor) = NonNull::new(spi).and_then(sensor::take) {
        sensor.unbind();
    }
}

/// A registered `spi_driver`, pinned on the heap.
struct Registration(Pin<KBox<sys::spi_driver>>);

// SAFETY: after registration the driver core owns all mutation of the
// struct; we only hand its address back to `driver_unregister()`.
unsafe impl Send for Registration {}
// SAFETY: as above, there is no shared access from Rust.
unsafe impl Sync for Registration {}

struct FocaltechSpiFp {
    driver: Registration,
}

impl kernel::Module for FocaltechSpiFp {
    fn init(module: &'static ThisModule) -> Result<Self> {
        let driver = sys::spi_driver {
            probe: Some(probe),
            remove: Some(remove),
            driver: sys::device_driver {
                name: c"focaltech_spi_fp".as_ptr().cast(),
                acpi_match_table: ACPI_IDS.as_ptr(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut driver = KBox::pin(driver, GFP_KERNEL)?;
        // SAFETY: the driver struct is pinned on the heap and unregistered in
        // `drop()` before it is freed.
        to_result(unsafe {
            sys::__spi_register_driver(module.as_ptr().cast(), driver.as_mut().get_unchecked_mut())
        })?;
        Ok(Self {
            driver: Registration(driver),
        })
    }
}

impl Drop for FocaltechSpiFp {
    fn drop(&mut self) {
        // SAFETY: registered in `init()`; `spi_unregister_driver()`.
        unsafe {
            sys::driver_unregister(&raw mut self.driver.0.as_mut().get_unchecked_mut().driver);
        }
    }
}
