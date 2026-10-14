//! FocalTech FT9368 command protocol.
//!
//! Every exchange is a single full-duplex SPI frame:
//!
//! ```text
//! tx: [cmd_hi, cmd_lo, len_hi, len_lo, 0, 0, 0, 0 × len]
//! rx: [ ...      header (4, +3 turnaround if len > 0) ..., payload × len]
//! ```

use crate::{Error, Result, Transport, image::Image};
use std::time::Duration;
use zeroize::Zeroizing;

const IMAGE_BYTES: usize = Image::WIDTH * Image::HEIGHT;

/// The value at bytes 19..21 of the status block.
pub const CHIP_ID: u16 = 0x9368;

const STATUS_BYTES: usize = 32;
const MAX_PAYLOAD: usize = IMAGE_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
enum Command {
    /// Wake the sensor from doze. No payload.
    Wake = 0xff00,
    /// Read the status block.
    Status = 0x9180,
    /// Read the latched frame; a short read just clears the notification.
    Frame = 0x9080,
}

/// Status byte 2 while the sensor is busy processing a touch.
const BUSY: u8 = 0x22;
/// Status bytes 1 and 2 while a finger is on the sensor.
const CONTACT: u8 = 0x11;

const fn header_len(payload: usize) -> usize {
    if payload == 0 { 4 } else { 7 }
}

/// The sensor's 32-byte status block.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Status([u8; STATUS_BYTES]);

impl Status {
    /// Chip identifier, [`CHIP_ID`] for a running FT9368 application.
    #[must_use]
    pub fn chip_id(&self) -> u16 {
        u16::from_be_bytes([self.0[19], self.0[20]])
    }

    /// Application firmware version.
    #[must_use]
    pub fn firmware(&self) -> u8 {
        self.0[21]
    }

    /// The raw block, for diagnostics.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; STATUS_BYTES] {
        &self.0
    }

    /// Whether this block can be trusted to say anything about contact:
    /// right chip, not busy, not the all-ones pattern of a missing sensor.
    fn is_healthy(&self) -> bool {
        self.chip_id() == CHIP_ID && self.0[2] != BUSY && self.0[..2] != [0xff, 0xff]
    }

    /// A finger is on the sensor (only meaningful when healthy).
    fn has_contact(&self) -> bool {
        self.0[1] == CONTACT && self.0[2] == CONTACT
    }

    /// A dozing sensor clocks out one repeated filler byte instead of data.
    fn is_filler(&self) -> bool {
        let first = self.0[0];
        first != 0 && self.0[..4].iter().all(|&b| b == first)
    }
}

impl std::fmt::Debug for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Status")
            .field("chip_id", &format_args!("{:#06x}", self.chip_id()))
            .field("firmware", &format_args!("{:#04x}", self.firmware()))
            .finish_non_exhaustive()
    }
}

/// Who answered, as established by [`Ft9368::identify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// Application firmware version (0x11, 0x13, 0x14 are known).
    pub firmware: u8,
}

/// An FT9368 behind some [`Transport`].
#[derive(Debug)]
pub struct Ft9368<T> {
    io: T,
}

impl<T: Transport> Ft9368<T> {
    /// Wraps a transport; does not talk to the sensor yet.
    pub fn new(io: T) -> Self {
        Self { io }
    }

    /// The underlying transport.
    pub fn io_mut(&mut self) -> &mut T {
        &mut self.io
    }

    /// Gives the transport back.
    pub fn into_inner(self) -> T {
        self.io
    }

    /// Sends `command` and reads `payload.len()` bytes of answer into it.
    /// The frame buffer is wiped afterwards, as it may carry image data.
    fn command(&mut self, command: Command, payload: &mut [u8]) -> Result<()> {
        if payload.len() > MAX_PAYLOAD {
            return Err(Error::Protocol("payload too large"));
        }
        let header = header_len(payload.len());
        let mut frame = Zeroizing::new(vec![0u8; header + payload.len()]);
        frame[..2].copy_from_slice(&(command as u16).to_be_bytes());
        // Bounded by MAX_PAYLOAD above.
        #[allow(clippy::cast_possible_truncation)]
        frame[2..4].copy_from_slice(&(payload.len() as u16).to_be_bytes());
        self.io.transfer(&mut frame)?;
        payload.copy_from_slice(&frame[header..]);
        Ok(())
    }

    /// Reads the status block once, as is.
    ///
    /// # Errors
    ///
    /// Transport failures.
    pub fn status(&mut self) -> Result<Status> {
        let mut block = [0u8; STATUS_BYTES];
        self.command(Command::Status, &mut block)?;
        Ok(Status(block))
    }

    /// Brings the sensor out of doze. It falls back asleep ~100 ms after the
    /// last command, so call this before every interaction.
    ///
    /// # Errors
    ///
    /// Transport failures.
    pub fn wake(&mut self) -> Result<()> {
        for _ in 0..3 {
            self.command(Command::Wake, &mut [])?;
            self.io.sleep(Duration::from_millis(10));
            let mut probe = [0u8; 4];
            self.command(Command::Status, &mut probe)?;
            let mut status = [0u8; STATUS_BYTES];
            status[..4].copy_from_slice(&probe);
            if !Status(status).is_filler() {
                break;
            }
        }
        Ok(())
    }

    /// Whether a finger is on the sensor right now.
    ///
    /// The sensor answers busy or filler for a moment while it processes a
    /// touch; those are retried, only a persistently odd answer is an error.
    ///
    /// # Errors
    ///
    /// [`Error::Protocol`] if no healthy status arrives.
    pub fn contact(&mut self) -> Result<bool> {
        for _ in 0..5 {
            self.wake()?;
            let status = self.status()?;
            if status.is_healthy() {
                return Ok(status.has_contact());
            }
            self.io.sleep(Duration::from_millis(20));
        }
        Err(Error::Protocol("no healthy status while checking contact"))
    }

    /// Clears a latched touch notification. Until cleared, it holds the
    /// edge-triggered interrupt line high, so a new touch raises no edge.
    ///
    /// # Errors
    ///
    /// Transport failures.
    pub fn clear_notification(&mut self) -> Result<()> {
        self.command(Command::Frame, &mut [0u8; 6])
    }

    /// Reads the frame captured for the current touch.
    ///
    /// # Errors
    ///
    /// Transport failures.
    pub fn read_image(&mut self) -> Result<Image> {
        let mut image = Image::blank();
        self.wake()?;
        self.command(Command::Frame, image.pixels_mut())?;
        Ok(image)
    }

    /// Makes sure an FT9368 application is running, resetting the sensor if
    /// it doesn't answer, and reports its firmware.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] if no FT9368 answers even after a hardware reset.
    pub fn identify(&mut self) -> Result<Identity> {
        self.wake()?;
        let status = self.status()?;
        if status.chip_id() == CHIP_ID {
            return Ok(Identity {
                firmware: status.firmware(),
            });
        }

        self.io.reset()?;
        let mut last = status;
        for _ in 0..5 {
            self.io.sleep(Duration::from_millis(200));
            self.wake()?;
            last = self.status()?;
            if last.chip_id() == CHIP_ID {
                return Ok(Identity {
                    firmware: last.firmware(),
                });
            }
        }
        Err(Error::NotFound {
            chip_id: last.chip_id(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Scripted sensor: answers each frame with the next queued reply.
    #[derive(Default)]
    struct Script {
        sent: Vec<Vec<u8>>,
        replies: VecDeque<Vec<u8>>,
        resets: usize,
    }

    impl Transport for Script {
        fn transfer(&mut self, frame: &mut [u8]) -> Result<()> {
            self.sent.push(frame.to_vec());
            let reply = self.replies.pop_front().unwrap_or_default();
            frame.fill(0);
            let n = reply.len().min(frame.len());
            frame[..n].copy_from_slice(&reply[..n]);
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            self.resets += 1;
            Ok(())
        }
        fn sleep(&mut self, _: Duration) {}
    }

    fn contact_reply(b1: u8, b2: u8) -> Vec<u8> {
        let mut reply = status_reply(CHIP_ID, 0x14);
        reply[7 + 1] = b1;
        reply[7 + 2] = b2;
        reply
    }

    #[test]
    fn contact_skips_busy_answers() {
        let mut script = Script::default();
        let wake = [vec![], vec![0; 11]];
        script.replies.extend(wake.clone());
        script.replies.push_back(contact_reply(0x11, BUSY));
        script.replies.extend(wake);
        script.replies.push_back(contact_reply(0x11, 0x11));
        assert!(Ft9368::new(script).contact().unwrap());
    }

    #[test]
    fn image_is_read_with_frame_command() {
        let mut script = Script::default();
        script.replies.extend([vec![], vec![0; 11]]);
        let mut frame = vec![0u8; 7 + IMAGE_BYTES];
        frame[7] = 0x42;
        frame[7 + IMAGE_BYTES - 1] = 0x24;
        script.replies.push_back(frame);
        let mut sensor = Ft9368::new(script);
        let image = sensor.read_image().unwrap();
        assert_eq!(image.pixels()[0], 0x42);
        assert_eq!(image.pixels()[IMAGE_BYTES - 1], 0x24);
        assert_eq!(sensor.into_inner().sent[2][..4], [0x90, 0x80, 0x14, 0x00]);
    }

    fn status_reply(chip: u16, firmware: u8) -> Vec<u8> {
        let mut reply = vec![0u8; 7 + STATUS_BYTES];
        reply[7 + 19..7 + 21].copy_from_slice(&chip.to_be_bytes());
        reply[7 + 21] = firmware;
        reply
    }

    #[test]
    fn frames_carry_command_and_length() {
        let mut sensor = Ft9368::new(Script::default());
        sensor.status().unwrap();
        sensor.command(Command::Wake, &mut []).unwrap();
        let sent = sensor.into_inner().sent;
        assert_eq!(sent[0].len(), 7 + 32);
        assert_eq!(sent[0][..4], [0x91, 0x80, 0x00, 0x20]);
        assert_eq!(sent[1], [0xff, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn identifies_running_sensor() {
        let mut script = Script::default();
        script
            .replies
            .extend([vec![], vec![0; 11], status_reply(CHIP_ID, 0x14)]);
        let mut sensor = Ft9368::new(script);
        assert_eq!(sensor.identify().unwrap(), Identity { firmware: 0x14 });
        assert_eq!(sensor.into_inner().resets, 0);
    }

    #[test]
    fn wake_retries_while_dozing() {
        let mut script = Script::default();
        let filler = vec![0, 0, 0, 0, 0, 0, 0, 0xa5, 0xa5, 0xa5, 0xa5];
        script.replies.extend([vec![], filler, vec![], vec![0; 11]]);
        let mut sensor = Ft9368::new(script);
        sensor.wake().unwrap();
        assert_eq!(sensor.into_inner().sent.len(), 4);
    }

    #[test]
    fn resets_silent_sensor_then_gives_up() {
        let mut sensor = Ft9368::new(Script::default());
        assert!(matches!(
            sensor.identify(),
            Err(Error::NotFound { chip_id: 0 })
        ));
        assert_eq!(sensor.into_inner().resets, 1);
    }
}
