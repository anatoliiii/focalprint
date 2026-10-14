// SPDX-License-Identifier: GPL-2.0-only
//! Per-device state and the `/dev/focaltech-fp` character device.
//!
//! The kernel side is deliberately thin: SPI transfers, the reset line and the
//! interrupt. The FT9368 protocol lives in userspace (`focalprintd`), where it
//! can be written in safe Rust and tested without hardware.

use crate::{
    reset::ResetLine,
    spi::{SpiDevice, SPI_CS_HIGH},
    sys, uapi,
};
use core::{ffi::c_void, ptr::NonNull};
use kernel::{
    error::to_result,
    fs::File,
    miscdevice::{MiscDevice, MiscDeviceOptions, MiscDeviceRegistration},
    new_condvar, new_mutex,
    prelude::*,
    sync::{Arc, ArcBorrow, CondVar, CondVarTimeoutResult, Mutex},
    time::msecs_to_jiffies,
    uaccess::{UserPtr, UserSlice},
    ThisModule,
};

/// Everything that only exists while the driver is bound to the hardware.
struct Hardware {
    spi: SpiDevice,
    reset: ResetLine,
    irq: u32,
}

struct State {
    /// `None` once the device has been unbound; open files then get `ENODEV`.
    hw: Option<Hardware>,
    /// At most one opener (the daemon) at a time.
    open: bool,
    /// IRQ line currently requested for the opener, if any.
    irq_requested: Option<u32>,
    /// Interrupts seen since probe; userspace waits for it to change.
    irq_count: u64,
}

#[pin_data]
pub(crate) struct Sensor {
    #[pin]
    state: Mutex<State>,
    #[pin]
    irq_event: CondVar,
    module: &'static ThisModule,
    #[pin]
    misc: MiscDeviceRegistration<OpenSensor>,
}

impl Sensor {
    /// Takes over the bound `spi` device and publishes `/dev/focaltech-fp`.
    pub(crate) fn probe(spi: SpiDevice, module: &'static ThisModule) -> Result<Arc<Self>> {
        // FTE4800 firmware declares an active-low chip select, but the
        // FT9368 only answers with CS active-high.
        spi.setup(SPI_CS_HIGH)?;
        let irq = spi.irq()?;
        let reset = ResetLine::claim(&spi)?;
        let hw = Hardware { spi, reset, irq };

        Arc::pin_init(
            try_pin_init!(Self {
                state <- new_mutex!(State {
                    hw: Some(hw),
                    open: false,
                    irq_requested: None,
                    irq_count: 0,
                }),
                irq_event <- new_condvar!(),
                module,
                misc <- MiscDeviceRegistration::register(MiscDeviceOptions { name: c"focaltech-fp" }),
            }),
            GFP_KERNEL,
        )
    }

    /// Driver unbind: release the hardware now, even if a file is still open.
    pub(crate) fn unbind(&self) {
        let (hw, irq) = {
            let mut state = self.state.lock();
            (state.hw.take(), state.irq_requested.take())
        };
        if let Some(irq) = irq {
            self.free_irq(irq);
        }
        drop(hw);
        self.irq_event.notify_all();
    }

    fn open(&self) -> Result {
        let mut state = self.state.lock();
        let irq = state.hw.as_ref().ok_or(ENODEV)?.irq;
        if state.open {
            return Err(EBUSY);
        }
        // misc device fops carry no owner; pin the module while open instead.
        // SAFETY: `self.module` is this module.
        if !unsafe { sys::try_module_get(self.module.as_ptr().cast()) } {
            return Err(ENODEV);
        }
        // Threaded handler only: it takes `state`, which may sleep.
        // SAFETY: `self` is pinned in an `Arc` that outlives the request: it
        // is freed in `close()`/`unbind()`, both before the last reference
        // can go away.
        let ret = unsafe {
            sys::request_threaded_irq(
                irq,
                None,
                Some(Self::irq_thread),
                sys::IRQF_ONESHOT as kernel::ffi::c_ulong,
                c"focaltech-fp".as_ptr().cast(),
                core::ptr::from_ref(self).cast_mut().cast(),
            )
        };
        if let Err(err) = to_result(ret) {
            // SAFETY: taken just above.
            unsafe { sys::module_put(self.module.as_ptr().cast()) };
            return Err(err);
        }
        state.open = true;
        state.irq_requested = Some(irq);
        Ok(())
    }

    fn close(&self) {
        let irq = {
            let mut state = self.state.lock();
            state.open = false;
            state.irq_requested.take()
        };
        if let Some(irq) = irq {
            self.free_irq(irq);
        }
        // SAFETY: paired with `try_module_get()` in `open()`.
        unsafe { sys::module_put(self.module.as_ptr().cast()) };
    }

    /// Must be called without `state` held: `free_irq` waits for the handler.
    fn free_irq(&self, irq: u32) {
        // SAFETY: requested in `open()` with this very `dev_id`.
        unsafe { sys::free_irq(irq, core::ptr::from_ref(self).cast_mut().cast()) };
    }

    unsafe extern "C" fn irq_thread(_irq: i32, data: *mut c_void) -> sys::irqreturn_t {
        // SAFETY: `data` is the `Sensor` registered in `open()`, alive until
        // `free_irq()` has returned.
        let sensor = unsafe { &*data.cast::<Self>() };
        sensor.state.lock().irq_count += 1;
        sensor.irq_event.notify_all();
        sys::irqreturn_IRQ_HANDLED
    }

    fn info(&self) -> Result<uapi::Info> {
        let state = self.state.lock();
        let hw = state.hw.as_ref().ok_or(ENODEV)?;
        Ok(uapi::Info {
            abi: uapi::ABI_VERSION,
            max_transfer: uapi::MAX_TRANSFER,
            speed_hz: hw.spi.speed_hz(),
            mode: hw.spi.mode(),
        })
    }

    fn reset(&self) -> Result {
        self.state.lock().hw.as_ref().ok_or(ENODEV)?.reset.pulse()
    }

    fn transfer(&self, request: uapi::Transfer) -> Result {
        let len = request.len as usize;
        if request.len == 0 || request.len > uapi::MAX_TRANSFER || request.reserved != 0 {
            return Err(EINVAL);
        }
        let mut tx = Wiped(KVec::from_elem(0u8, len, GFP_KERNEL)?);
        let mut rx = Wiped(KVec::from_elem(0u8, len, GFP_KERNEL)?);
        user_slice(request.tx, len)?
            .reader()
            .read_slice(&mut tx.0)?;
        self.state
            .lock()
            .hw
            .as_ref()
            .ok_or(ENODEV)?
            .spi
            .transfer(&tx.0, &mut rx.0)?;
        user_slice(request.rx, len)?.writer().write_slice(&rx.0)
        // `tx`/`rx` may hold fingerprint image data: wiped on drop.
    }

    fn wait_irq(&self, mut request: uapi::WaitIrq) -> Result<uapi::WaitIrq> {
        if request.reserved != 0 {
            return Err(EINVAL);
        }
        let mut remaining = msecs_to_jiffies(request.timeout_ms);
        let mut state = self.state.lock();
        while request.timeout_ms != 0 && state.irq_count == request.seen {
            if state.hw.is_none() {
                return Err(ENODEV);
            }
            remaining = match self
                .irq_event
                .wait_interruptible_timeout(&mut state, remaining)
            {
                CondVarTimeoutResult::Woken { jiffies } => jiffies,
                CondVarTimeoutResult::Timeout => return Err(ETIMEDOUT),
                CondVarTimeoutResult::Signal { .. } => return Err(ERESTARTSYS),
            };
        }
        request.seen = state.irq_count;
        Ok(request)
    }
}

fn user_slice(addr: u64, len: usize) -> Result<UserSlice> {
    let addr = usize::try_from(addr).map_err(|_| EFAULT)?;
    Ok(UserSlice::new(UserPtr::from_addr(addr), len))
}

/// A kernel buffer that is zeroed before being freed.
struct Wiped(KVec<u8>);

impl Drop for Wiped {
    fn drop(&mut self) {
        for byte in self.0.iter_mut() {
            // SAFETY: `byte` is a valid, exclusive reference. Volatile so the
            // stores can't be elided as dead before the free.
            unsafe { core::ptr::write_volatile(byte, 0) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// An open `/dev/focaltech-fp`.
pub(crate) struct OpenSensor {
    sensor: Arc<Sensor>,
}

#[vtable]
impl MiscDevice for OpenSensor {
    type Ptr = KBox<Self>;

    fn open(_file: &File, misc: &MiscDeviceRegistration<Self>) -> Result<KBox<Self>> {
        // SAFETY: every registration of `OpenSensor` is the `misc` field of a
        // `Sensor` that lives in an `Arc`, and `misc_open()` keeps it
        // registered (hence alive) for the duration of this call.
        let sensor: Arc<Sensor> = unsafe {
            ArcBorrow::from_raw(kernel::container_of!(
                core::ptr::from_ref(misc).cast_mut(),
                Sensor,
                misc
            ))
        }
        .into();
        let this = KBox::new(Self { sensor }, GFP_KERNEL)?;
        this.sensor.open()?;
        Ok(this)
    }

    fn release(this: KBox<Self>, _file: &File) {
        this.sensor.close();
    }

    fn ioctl(this: &Self, _file: &File, cmd: u32, arg: usize) -> Result<isize> {
        let arg = UserSlice::new(UserPtr::from_addr(arg), kernel::ioctl::_IOC_SIZE(cmd));
        match cmd {
            uapi::IOC_INFO => arg.writer().write(&this.sensor.info()?)?,
            uapi::IOC_RESET => this.sensor.reset()?,
            uapi::IOC_TRANSFER => this.sensor.transfer(arg.reader().read()?)?,
            uapi::IOC_WAIT_IRQ => {
                let (mut reader, mut writer) = arg.reader_writer();
                writer.write(&this.sensor.wait_irq(reader.read()?)?)?;
            }
            _ => return Err(ENOTTY),
        }
        Ok(0)
    }
}

/// Lets `remove()` find its `Sensor` again.
pub(crate) fn store(spi: NonNull<sys::spi_device>, sensor: Arc<Sensor>) {
    // SAFETY: bound device; `driver_data` is ours until `take()` in remove.
    unsafe { (*spi.as_ptr()).dev.driver_data = Arc::into_raw(sensor).cast_mut().cast() };
}

pub(crate) fn take(spi: NonNull<sys::spi_device>) -> Option<Arc<Sensor>> {
    // SAFETY: as in `store()`; the pointer came from `Arc::into_raw`.
    unsafe {
        let data = core::mem::replace(&mut (*spi.as_ptr()).dev.driver_data, core::ptr::null_mut());
        (!data.is_null()).then(|| Arc::from_raw(data.cast::<Sensor>()))
    }
}
