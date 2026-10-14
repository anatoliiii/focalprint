// SPDX-License-Identifier: GPL-2.0-only
//! The sensor's reset line, located through the ACPI `_CRS` of its device.
//!
//! The firmware lists one `GpioIo` (reset) and one `GpioInt` (interrupt, which
//! the SPI core already turned into `spi->irq`) but no `_DSD` names, so the
//! reset line is found by walking `_CRS` instead of hard-coding pin numbers.

use crate::{spi::SpiDevice, sys};
use core::{ffi::c_void, ptr::NonNull};
use kernel::{
    error::{from_err_ptr, to_result},
    prelude::*,
    time::{delay::fsleep, Delta},
};

const AE_OK: sys::acpi_status = 0;

/// Owned reset GPIO. Logical 1 = reset asserted (the line is active-low).
///
/// # Invariants
///
/// `desc` was obtained with `gpiod_get()` through `mapping`, which is
/// registered on `adev` until drop.
pub(crate) struct ResetLine {
    desc: NonNull<sys::gpio_desc>,
    adev: NonNull<sys::acpi_device>,
    // Referenced by `adev->driver_gpios` until drop, hence boxed.
    _mapping: KBox<[sys::acpi_gpio_mapping; 2]>,
    _params: KBox<sys::acpi_gpio_params>,
}

// SAFETY: gpiod and ACPI device handles may be used from any thread; all
// accesses are serialised by the owning `Sensor` mutex.
unsafe impl Send for ResetLine {}

impl ResetLine {
    /// Claims the reset line and leaves the sensor running (deasserted).
    pub(crate) fn claim(spi: &SpiDevice) -> Result<Self> {
        let adev = acpi_companion(spi).ok_or(ENODEV)?;
        let index = reset_crs_index(adev)?;

        let params = KBox::new(
            sys::acpi_gpio_params {
                crs_entry_index: index,
                line_index: 0,
                active_low: true,
            },
            GFP_KERNEL,
        )?;
        let mapping = KBox::new(
            [
                sys::acpi_gpio_mapping {
                    name: c"reset-gpios".as_ptr().cast(),
                    data: &raw const *params,
                    size: 1,
                    quirks: 0,
                },
                sys::acpi_gpio_mapping::default(),
            ],
            GFP_KERNEL,
        )?;
        // SAFETY: `adev` is the live companion of the bound device; `mapping`
        // is NULL-terminated and outlives the registration (removed in drop).
        to_result(unsafe { sys::acpi_dev_add_driver_gpios(adev.as_ptr(), mapping.as_ptr()) })?;

        // SAFETY: the device is bound; the mapping above names "reset".
        let desc = from_err_ptr(unsafe {
            sys::gpiod_get(
                spi.device(),
                c"reset".as_ptr().cast(),
                sys::gpiod_flags_GPIOD_OUT_LOW,
            )
        });
        let desc = match desc.map(NonNull::new) {
            Ok(Some(desc)) => desc,
            failed => {
                // SAFETY: registered above, nothing obtained through it.
                unsafe { sys::acpi_dev_remove_driver_gpios(adev.as_ptr()) };
                return Err(failed.err().unwrap_or(ENOENT));
            }
        };
        Ok(Self {
            desc,
            adev,
            _mapping: mapping,
            _params: params,
        })
    }

    fn set(&self, asserted: bool) -> Result {
        // SAFETY: `desc` is a valid, owned descriptor (type invariant).
        to_result(unsafe { sys::gpiod_set_value_cansleep(self.desc.as_ptr(), asserted.into()) })
    }

    /// Hardware reset: 10 ms released, 20 ms held, then released again. The
    /// sensor needs ~200 ms afterwards before it answers; that wait is left
    /// to userspace so the caller doesn't hold the device lock meanwhile.
    pub(crate) fn pulse(&self) -> Result {
        self.set(false)?;
        fsleep(Delta::from_millis(10));
        self.set(true)?;
        fsleep(Delta::from_millis(20));
        self.set(false)
    }
}

impl Drop for ResetLine {
    fn drop(&mut self) {
        // SAFETY: owned descriptor and registered mapping (type invariant);
        // released in reverse order of acquisition.
        unsafe {
            sys::gpiod_put(self.desc.as_ptr());
            sys::acpi_dev_remove_driver_gpios(self.adev.as_ptr());
        }
    }
}

/// `ACPI_COMPANION(&spi->dev)`.
fn acpi_companion(spi: &SpiDevice) -> Option<NonNull<sys::acpi_device>> {
    // SAFETY: the device is bound, so its fwnode pointer is stable; an ACPI
    // fwnode is always embedded in a `struct acpi_device`.
    unsafe {
        let fwnode = (*spi.device()).fwnode;
        if fwnode.is_null() || !core::ptr::eq((*fwnode).ops, &raw const sys::acpi_device_fwnode_ops)
        {
            return None;
        }
        NonNull::new(kernel::container_of!(fwnode, sys::acpi_device, fwnode))
    }
}

#[derive(Default)]
struct CrsScan {
    gpio_index: u32,
    reset: Option<u32>,
    resets: u32,
}

/// Index of the single one-pin `GpioIo` among the device's GPIO resources.
fn reset_crs_index(adev: NonNull<sys::acpi_device>) -> Result<u32> {
    unsafe extern "C" fn visit(res: *mut sys::acpi_resource, ctx: *mut c_void) -> sys::acpi_status {
        // SAFETY: ACPICA passes a valid resource and our `CrsScan` context.
        let (res, scan) = unsafe { (&*res, &mut *ctx.cast::<CrsScan>()) };
        if res.type_ != sys::ACPI_RESOURCE_TYPE_GPIO {
            return AE_OK;
        }
        // SAFETY: `type_` says the union holds a GPIO descriptor.
        let gpio = unsafe { res.data.gpio.as_ref() };
        if u32::from(gpio.connection_type) == sys::ACPI_RESOURCE_GPIO_TYPE_IO
            && gpio.pin_table_length == 1
        {
            scan.reset = Some(scan.gpio_index);
            scan.resets += 1;
        }
        scan.gpio_index += 1;
        AE_OK
    }

    let mut scan = CrsScan::default();
    // SAFETY: valid handle of a live ACPI device; `scan` outlives the walk.
    let status = unsafe {
        sys::acpi_walk_resources(
            (*adev.as_ptr()).handle,
            c"_CRS".as_ptr().cast_mut().cast(),
            Some(visit),
            (&raw mut scan).cast(),
        )
    };
    match (status, scan.resets, scan.reset) {
        (AE_OK, 1, Some(index)) => Ok(index),
        (AE_OK, ..) => Err(ENODEV),
        _ => Err(EIO),
    }
}
