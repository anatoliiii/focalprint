//! FocalTech FT9201 USB protocol (vendor-class interface, `2808:c652`).
//!
//! Every exchange is one bulk OUT request and one bulk IN reply, both framed
//! the same way:
//!
//! ```text
//! 02 | len (u16 BE) | payload × len | xor(len_hi, len_lo, payload…)
//! ```
//!
//! A reply payload starts with a status byte, `04` for success. The sensor
//! only captures; matching happens on the host, like with the FT9368.

use crate::{Error, Image, Result, Wait};
use std::time::{Duration, Instant};
use zeroize::Zeroizing;

const START: u8 = 0x02;
const OK: u8 = 0x04;
/// Request/reply framing around the payload: start, length, checksum.
const OVERHEAD: usize = 4;

const PIXELS: usize = Image::WIDTH * Image::HEIGHT;
/// Image payload: status byte, then 16 bits per pixel.
const IMAGE_REPLY: usize = OVERHEAD + 1 + 2 * PIXELS;
const SHORT_REPLY: usize = 64;

const CMD_ID: &[u8] = &[0x80, 0x05];
const CMD_WIDTH: &[u8] = &[0x80, 0x03];
const CMD_HEIGHT: &[u8] = &[0x80, 0x04];
/// Sent by the vendor library right after the geometry queries; its meaning
/// is unknown, the sensor just acknowledges it.
const CMD_SETUP: &[u8] = &[0x82, 0x78];
/// `01` once a frame has been captured for a finger, `00` otherwise.
const CMD_FRAME_READY: &[u8] = &[0x80, 0x02];
const CMD_IMAGE: &[u8] = &[0x81];

/// The vendor library polls at this rate.
const POLL: Duration = Duration::from_millis(20);
/// While a finger rests, a new frame becomes ready every 0.1–0.4 s; a release
/// is this long without one.
const RELEASE_QUIET: Duration = Duration::from_millis(700);

/// How one exchange reaches the sensor.
pub trait Link {
    /// Sends `request` and reads one reply frame into `reply`; returns its
    /// length.
    ///
    /// # Errors
    ///
    /// Transport-specific.
    fn exchange(&mut self, request: &[u8], reply: &mut [u8]) -> Result<usize>;

    /// Waits; overridable so tests don't actually sleep.
    fn sleep(&mut self, duration: Duration);
}

/// Byte order of the 16-bit pixels in an image reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelOrder {
    /// Least significant byte first.
    #[default]
    Little,
    /// Most significant byte first.
    Big,
}

/// Who answered, as established by [`Ft9201::identify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// Answer to the id query (`0x6293` on the MegaBook K16S).
    pub id: u16,
}

/// An FT9201 behind some [`Link`].
#[derive(Debug)]
pub struct Ft9201<L> {
    link: L,
    order: PixelOrder,
    invert: bool,
}

fn checksum(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0, |acc, b| acc ^ b)
}

impl<L: Link> Ft9201<L> {
    /// Wraps a link; does not talk to the sensor yet.
    pub fn new(link: L) -> Self {
        Self {
            link,
            order: PixelOrder::default(),
            invert: false,
        }
    }

    /// Overrides how image pixels are decoded.
    #[must_use]
    pub fn with_pixel_order(mut self, order: PixelOrder) -> Self {
        self.order = order;
        self
    }

    /// Inverts the stretched frame, for a sensor that reports ridges bright:
    /// FpNet expects dark ridges and a light background, like the FT9368.
    #[must_use]
    pub fn with_inverted(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Gives the link back.
    pub fn into_inner(self) -> L {
        self.link
    }

    /// Sends `payload`, checks the reply's framing and status, and returns
    /// the reply payload after the status byte. The receive buffer is wiped
    /// on drop, as it may carry image data.
    fn command(&mut self, payload: &[u8], reply_len: usize) -> Result<Zeroizing<Vec<u8>>> {
        let len = u16::try_from(payload.len())
            .map_err(|_| Error::Protocol("request too long"))?
            .to_be_bytes();
        let mut request = Vec::with_capacity(payload.len() + OVERHEAD);
        request.push(START);
        request.extend_from_slice(&len);
        request.extend_from_slice(payload);
        request.push(checksum(&request[1..]));

        let mut reply = Zeroizing::new(vec![0u8; reply_len]);
        let got = self.link.exchange(&request, &mut reply)?;
        let frame = reply.get(..got).ok_or(Error::Protocol("reply overflow"))?;
        let [START, hi, lo, ref rest @ ..] = *frame else {
            return Err(Error::Protocol("malformed reply frame"));
        };
        let n = usize::from(u16::from_be_bytes([hi, lo]));
        if rest.len() != n + 1 {
            return Err(Error::Protocol("reply length mismatch"));
        }
        if checksum(&frame[1..got - 1]) != frame[got - 1] {
            return Err(Error::Protocol("reply checksum mismatch"));
        }
        match rest.first() {
            Some(&OK) => Ok(Zeroizing::new(rest[1..n].to_vec())),
            Some(_) => Err(Error::Protocol("sensor rejected the command")),
            None => Err(Error::Protocol("empty reply")),
        }
    }

    fn byte(&mut self, payload: &[u8]) -> Result<u8> {
        self.command(payload, SHORT_REPLY)?
            .first()
            .copied()
            .ok_or(Error::Protocol("short reply"))
    }

    /// Runs the vendor start-up sequence and checks the matrix geometry.
    ///
    /// # Errors
    ///
    /// [`Error::Protocol`] if the sensor is not a 64×80 FT9201.
    pub fn identify(&mut self) -> Result<Identity> {
        let id = self.command(CMD_ID, SHORT_REPLY)?;
        let &[hi, lo] = id.as_slice() else {
            return Err(Error::Protocol("unexpected id reply"));
        };
        let (width, height) = (self.byte(CMD_WIDTH)?, self.byte(CMD_HEIGHT)?);
        if (usize::from(width), usize::from(height)) != (Image::WIDTH, Image::HEIGHT) {
            return Err(Error::Protocol("unexpected sensor geometry"));
        }
        self.command(CMD_SETUP, SHORT_REPLY)?;
        Ok(Identity {
            id: u16::from_be_bytes([hi, lo]),
        })
    }

    /// Whether a frame is waiting to be read.
    ///
    /// # Errors
    ///
    /// Transport or protocol failures.
    pub fn frame_ready(&mut self) -> Result<bool> {
        Ok(self.byte(CMD_FRAME_READY)? == 0x01)
    }

    /// Reads the captured frame, stretched to 8 bits.
    ///
    /// # Errors
    ///
    /// Transport or protocol failures.
    pub fn read_image(&mut self) -> Result<Image> {
        let raw = self.command(CMD_IMAGE, IMAGE_REPLY)?;
        if raw.len() != 2 * PIXELS {
            return Err(Error::Protocol("unexpected image size"));
        }
        let order = self.order;
        let samples: Zeroizing<Vec<u16>> = Zeroizing::new(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|&p| match order {
                    PixelOrder::Little => u16::from_le_bytes(p),
                    PixelOrder::Big => u16::from_be_bytes(p),
                })
                .collect(),
        );
        let mut image = stretch(&samples);
        if self.invert {
            for p in image.pixels_mut() {
                *p = 255 - *p;
            }
        }
        Ok(image)
    }

    /// Waits for a finger and returns its frame.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`], [`Error::Cancelled`], or device/protocol failures.
    pub fn capture(&mut self, wait: Wait<'_>) -> Result<Image> {
        let deadline = Instant::now() + wait.timeout;
        loop {
            wait.check(deadline)?;
            if self.frame_ready()? {
                return self.read_image();
            }
            self.link.sleep(POLL);
        }
    }

    /// Waits until the finger has been lifted. Frames captured meanwhile
    /// are read (so the sensor keeps reporting new ones) and wiped.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`], [`Error::Cancelled`], or device/protocol failures.
    pub fn wait_release(&mut self, wait: Wait<'_>) -> Result<()> {
        let deadline = Instant::now() + wait.timeout;
        let mut quiet = Duration::ZERO;
        while quiet < RELEASE_QUIET {
            wait.check(deadline)?;
            if self.frame_ready()? {
                drop(self.read_image()?);
                quiet = Duration::ZERO;
            } else {
                quiet += POLL;
            }
            self.link.sleep(POLL);
        }
        Ok(())
    }
}

/// Maps the 2nd..98th percentile of the raw samples linearly onto 0..=255,
/// so frames from different sensors and pressures land in the same range.
fn stretch(samples: &[u16]) -> Image {
    let mut sorted = Zeroizing::new(samples.to_vec());
    let lo_at = sorted.len() / 50;
    let hi_at = sorted.len() - 1 - lo_at;
    let lo = f32::from(*sorted.select_nth_unstable(lo_at).1);
    let hi = f32::from(*sorted.select_nth_unstable(hi_at).1);
    let scale = 255.0 / (hi - lo).max(1.0);
    let mut image = Image::blank();
    for (dst, &src) in image.pixels_mut().iter_mut().zip(samples) {
        // Clamped to 0..=255 first.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v = ((f32::from(src) - lo) * scale).clamp(0.0, 255.0).round() as u8;
        *dst = v;
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::VecDeque, sync::atomic::AtomicBool};

    /// Scripted sensor: answers each request with the next queued payload.
    #[derive(Default)]
    struct Script {
        sent: Vec<Vec<u8>>,
        replies: VecDeque<Vec<u8>>,
    }

    fn frame(payload: &[u8]) -> Vec<u8> {
        let mut f = vec![START];
        f.extend_from_slice(&u16::try_from(payload.len()).unwrap().to_be_bytes());
        f.extend_from_slice(payload);
        f.push(checksum(&f[1..]));
        f
    }

    impl Script {
        fn answer(&mut self, payload: &[u8]) {
            self.replies.push_back(frame(payload));
        }
    }

    impl Link for Script {
        fn exchange(&mut self, request: &[u8], reply: &mut [u8]) -> Result<usize> {
            self.sent.push(request.to_vec());
            let r = self.replies.pop_front().ok_or(Error::Timeout)?;
            reply[..r.len()].copy_from_slice(&r);
            Ok(r.len())
        }
        fn sleep(&mut self, _: Duration) {}
    }

    /// Byte-exact exchanges recorded from the vendor library.
    #[test]
    fn matches_vendor_capture() {
        let mut s = Script::default();
        for r in [
            "020003046293f6",
            "020002044046",
            "020002045056",
            "0200010405",
        ] {
            s.replies.push_back(hex(r));
        }
        let mut sensor = Ft9201::new(s);
        assert_eq!(sensor.identify().unwrap(), Identity { id: 0x6293 });
        let sent = sensor.into_inner().sent;
        let want = [
            "020002800587",
            "020002800381",
            "020002800486",
            "0200028278f8",
        ];
        assert_eq!(sent, want.map(hex));
        assert_eq!(frame(CMD_IMAGE), hex("0200018180"));
        assert_eq!(frame(CMD_FRAME_READY), hex("020002800280"));
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut s = Script::default();
        s.replies.push_back(hex("020002040007"));
        let err = Ft9201::new(s).frame_ready().unwrap_err();
        assert!(matches!(err, Error::Protocol("reply checksum mismatch")));
    }

    #[test]
    fn capture_polls_then_reads_stretched_frame() {
        let mut s = Script::default();
        s.answer(&[OK, 0]);
        s.answer(&[OK, 1]);
        let mut image = vec![OK];
        for i in 0..PIXELS {
            // A ramp from 1000 to ~6000, big-endian.
            let v = 1000 + u16::try_from(i).unwrap();
            image.extend_from_slice(&v.to_be_bytes());
        }
        s.answer(&image);
        let cancel = AtomicBool::new(false);
        let mut sensor = Ft9201::new(s).with_pixel_order(PixelOrder::Big);
        let img = sensor
            .capture(Wait {
                timeout: Duration::from_secs(1),
                cancel: &cancel,
            })
            .unwrap();
        assert_eq!(img.pixels()[0], 0);
        assert_eq!(img.pixels()[PIXELS - 1], 255);
        assert!(img.pixels().windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(sensor.into_inner().sent[2], hex("0200018180"));
    }

    #[test]
    fn inverted_frame_flips_polarity() {
        let mut s = Script::default();
        let mut image = vec![OK];
        for i in 0..PIXELS {
            let v = 1000 + u16::try_from(i).unwrap();
            image.extend_from_slice(&v.to_le_bytes());
        }
        s.answer(&image);
        let img = Ft9201::new(s).with_inverted(true).read_image().unwrap();
        assert_eq!(img.pixels()[0], 255);
        assert_eq!(img.pixels()[PIXELS - 1], 0);
    }

    #[test]
    fn release_needs_a_quiet_spell() {
        let mut s = Script::default();
        s.answer(&[OK, 1]);
        s.answer(&[OK; 1 + 2 * PIXELS]);
        let quiet_polls = RELEASE_QUIET.as_millis() / POLL.as_millis();
        for _ in 0..quiet_polls {
            s.answer(&[OK, 0]);
        }
        let cancel = AtomicBool::new(false);
        let mut sensor = Ft9201::new(s);
        sensor
            .wait_release(Wait {
                timeout: Duration::from_secs(5),
                cancel: &cancel,
            })
            .unwrap();
        assert!(sensor.into_inner().replies.is_empty());
    }
}
