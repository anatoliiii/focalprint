//! Touch-driven capture: wait for a finger, read its frame, wait for release.

use crate::{Error, Ft9368, Image, Interrupt, Result, Transport};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// How often a blocking wait looks at its cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(100);
/// While a finger rests, the contact flag pulses a few times per second
/// instead of staying set: a release is this long without contact.
const RELEASE_QUIET: Duration = Duration::from_millis(700);
const RELEASE_POLL: Duration = Duration::from_millis(20);

/// Limits for a blocking capture step.
#[derive(Debug, Clone, Copy)]
pub struct Wait<'a> {
    /// Give up with [`Error::Timeout`] after this long.
    pub timeout: Duration,
    /// Give up with [`Error::Cancelled`] once this becomes `true`.
    pub cancel: &'a AtomicBool,
}

impl Wait<'_> {
    pub(crate) fn check(&self, deadline: Instant) -> Result<Duration> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or(Error::Timeout)
    }
}

impl<T: Transport + Interrupt> Ft9368<T> {
    /// Waits for a finger and returns its frame.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`], [`Error::Cancelled`], or device/protocol failures.
    pub fn capture(&mut self, wait: Wait<'_>) -> Result<Image> {
        let deadline = Instant::now() + wait.timeout;
        // Interrupts that happened before arming must not count as a touch.
        let mut seen = self.io_mut().irq_count()?;
        self.wake()?;
        self.clear_notification()?;
        loop {
            let slice = wait.check(deadline)?.min(CANCEL_POLL);
            let Some(count) = self.io_mut().wait_irq(seen, slice)? else {
                continue;
            };
            seen = count;
            // The interrupt is only a hint; the status block is the authority.
            if self.contact()? {
                return self.read_image();
            }
        }
    }

    /// Waits until the finger has been lifted.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`], [`Error::Cancelled`], or device/protocol failures.
    pub fn wait_release(&mut self, wait: Wait<'_>) -> Result<()> {
        let deadline = Instant::now() + wait.timeout;
        let mut quiet = Duration::ZERO;
        while quiet < RELEASE_QUIET {
            wait.check(deadline)?;
            if self.contact()? {
                // Re-arm, so a finger still resting raises it again instead
                // of the notification lingering past the release.
                self.clear_notification()?;
                quiet = Duration::ZERO;
            } else {
                quiet += RELEASE_POLL;
            }
            self.io_mut().sleep(RELEASE_POLL);
        }
        Ok(())
    }
}
