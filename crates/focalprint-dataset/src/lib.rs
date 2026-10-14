//! Opt-in, encrypted collection of fingerprint frames for algorithm research.
//!
//! Each touch becomes one file `MAGIC ‖ nonce ‖ XChaCha20-Poly1305(record)`
//! under the collection directory. The 256-bit key is never stored in the
//! clear: it is sealed to this machine's TPM with `systemd-creds` and handed
//! to the daemon by systemd (`LoadCredentialEncrypted=`).

use chacha20poly1305::{
    KeyInit, XChaCha20Poly1305, XNonce,
    aead::{Aead, Payload},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write as _},
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

const MAGIC: &[u8; 8] = b"FTE4DS01";
const NONCE: usize = 24;
pub const KEY_BYTES: usize = 32;
/// Name of the systemd credential carrying the key.
pub const CREDENTIAL: &str = "focalprint-key";

/// Errors reading or writing the collection.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("dataset key must be {KEY_BYTES} bytes")]
    Key,
    #[error("not a dataset record, wrong key, or damaged")]
    Record,
}

/// What the touch was used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Zeroize)]
pub enum Kind {
    Verify,
    Enroll,
    /// Deliberate collection session (`focalprint dataset capture/import`).
    Capture,
}

/// One touch with its context. Labels come from what the daemon knew:
/// the finger being enrolled, or the finger a verification matched.
#[derive(Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct Record {
    pub unix_ms: u64,
    pub kind: Kind,
    pub user: String,
    /// Finger name when known (enrolment, or a successful match).
    pub finger: Option<String>,
    /// fprintd-style result, e.g. `verify-match`, `enroll-stage-passed`.
    pub outcome: String,
    /// Matcher evidence for verifications (0 otherwise).
    pub inliers: u16,
    pub width: u16,
    pub height: u16,
    pub pixels: Vec<u8>,
}

impl std::fmt::Debug for Record {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Record")
            .field("unix_ms", &self.unix_ms)
            .field("kind", &self.kind)
            .field("finger", &self.finger)
            .field("outcome", &self.outcome)
            .finish_non_exhaustive()
    }
}

/// The collection key, wiped on drop.
pub struct Key(Zeroizing<[u8; KEY_BYTES]>);

impl Key {
    /// Reads a raw key (e.g. `$CREDENTIALS_DIRECTORY/focalprint-key`).
    ///
    /// # Errors
    ///
    /// I/O errors, or [`Error::Key`] if the file is not exactly 32 bytes.
    pub fn load(path: &Path) -> Result<Self, Error> {
        Self::from_bytes(&Zeroizing::new(fs::read(path)?))
    }

    /// # Errors
    ///
    /// [`Error::Key`] unless `bytes` is exactly 32 bytes long.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let key: [u8; KEY_BYTES] = bytes.try_into().map_err(|_| Error::Key)?;
        Ok(Self(Zeroizing::new(key)))
    }

    /// A fresh random key.
    ///
    /// # Errors
    ///
    /// If the system RNG fails.
    pub fn generate() -> Result<Self, Error> {
        let mut key = Zeroizing::new([0u8; KEY_BYTES]);
        getrandom::fill(&mut *key).map_err(|e| io::Error::other(e.to_string()))?;
        Ok(Self(key))
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8; KEY_BYTES] {
        &self.0
    }

    fn cipher(&self) -> XChaCha20Poly1305 {
        XChaCha20Poly1305::new(&(*self.0).into())
    }

    /// A second handle to the same key (e.g. for templates and records).
    #[must_use]
    pub fn duplicate(&self) -> Self {
        Self(Zeroizing::new(*self.0))
    }

    /// `nonce ‖ XChaCha20-Poly1305(plain)`, bound to `context` as AAD.
    ///
    /// # Errors
    ///
    /// RNG failure.
    pub fn seal(&self, context: &[u8], plain: &[u8]) -> Result<Vec<u8>, Error> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).map_err(|e| io::Error::other(e.to_string()))?;
        let sealed = self
            .cipher()
            .encrypt(
                &XNonce::from(nonce),
                Payload {
                    msg: plain,
                    aad: context,
                },
            )
            .map_err(|_| Error::Record)?;
        let mut out = nonce.to_vec();
        out.extend_from_slice(&sealed);
        Ok(out)
    }

    /// Inverse of [`Self::seal`].
    ///
    /// # Errors
    ///
    /// [`Error::Record`] for damaged data, the wrong key or the wrong context.
    pub fn open(&self, context: &[u8], sealed: &[u8]) -> Result<Zeroizing<Vec<u8>>, Error> {
        if sealed.len() < NONCE {
            return Err(Error::Record);
        }
        let (nonce, body) = sealed.split_at(NONCE);
        let nonce: [u8; NONCE] = nonce.try_into().map_err(|_| Error::Record)?;
        self.cipher()
            .decrypt(
                &XNonce::from(nonce),
                Payload {
                    msg: body,
                    aad: context,
                },
            )
            .map(Zeroizing::new)
            .map_err(|_| Error::Record)
    }
}

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Key(…)")
    }
}

/// Appends encrypted records to a directory (0700, files 0600).
#[derive(Debug)]
pub struct Collection {
    dir: PathBuf,
    key: Key,
}

impl Collection {
    /// # Errors
    ///
    /// If the directory cannot be created.
    pub fn open(dir: impl Into<PathBuf>, key: Key) -> Result<Self, Error> {
        let dir = dir.into();
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)?;
        Ok(Self { dir, key })
    }

    /// Encrypts and stores one record; returns its path.
    ///
    /// # Errors
    ///
    /// I/O or RNG failures.
    pub fn add(&self, record: &Record) -> Result<PathBuf, Error> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).map_err(|e| io::Error::other(e.to_string()))?;
        let plain = Zeroizing::new(postcard::to_stdvec(record).map_err(|_| Error::Record)?);
        let sealed = self
            .key
            .cipher()
            .encrypt(
                &XNonce::from(nonce),
                Payload {
                    msg: &plain,
                    aad: MAGIC,
                },
            )
            .map_err(|_| Error::Record)?;
        let suffix = u32::from_be_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]);
        let path = self
            .dir
            .join(format!("{}-{suffix:08x}.rec", record.unix_ms));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        file.write_all(MAGIC)?;
        file.write_all(&nonce)?;
        file.write_all(&sealed)?;
        file.sync_all()?;
        Ok(path)
    }

    /// Decrypts one stored record.
    ///
    /// # Errors
    ///
    /// [`Error::Record`] for foreign, damaged or wrongly keyed files.
    pub fn read(&self, path: &Path) -> Result<Record, Error> {
        let bytes = fs::read(path)?;
        if bytes.len() < MAGIC.len() + NONCE || !bytes.starts_with(MAGIC) {
            return Err(Error::Record);
        }
        let (nonce, sealed) = bytes[MAGIC.len()..].split_at(NONCE);
        let nonce: [u8; NONCE] = nonce.try_into().map_err(|_| Error::Record)?;
        let plain = Zeroizing::new(
            self.key
                .cipher()
                .decrypt(
                    &XNonce::from(nonce),
                    Payload {
                        msg: sealed,
                        aad: MAGIC,
                    },
                )
                .map_err(|_| Error::Record)?,
        );
        postcard::from_bytes(&plain).map_err(|_| Error::Record)
    }

    /// Stored record files, oldest first.
    ///
    /// # Errors
    ///
    /// If the directory cannot be read.
    pub fn list(&self) -> Result<Vec<PathBuf>, Error> {
        let mut files: Vec<PathBuf> = fs::read_dir(&self.dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "rec"))
            .collect();
        files.sort();
        Ok(files)
    }
}

/// Milliseconds since the Unix epoch.
#[must_use]
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_wrong_key() {
        let dir = std::env::temp_dir().join(format!("focalprint-ds-test-{}", std::process::id()));
        let key = Key::generate().unwrap();
        let other = Key::from_bytes(&[7; KEY_BYTES]).unwrap();
        let collection = Collection::open(&dir, Key::from_bytes(key.as_bytes()).unwrap()).unwrap();
        let record = Record {
            unix_ms: 1,
            kind: Kind::Verify,
            user: "u".into(),
            finger: Some("right-index-finger".into()),
            outcome: "verify-match".into(),
            inliers: 12,
            width: 2,
            height: 2,
            pixels: vec![1, 2, 3, 4],
        };
        let path = collection.add(&record).unwrap();
        assert!(
            !fs::read(&path)
                .unwrap()
                .windows(4)
                .any(|w| w == [1, 2, 3, 4])
        );
        assert_eq!(collection.read(&path).unwrap().pixels, record.pixels);
        let wrong = Collection::open(&dir, other).unwrap();
        assert!(matches!(wrong.read(&path), Err(Error::Record)));
        fs::remove_dir_all(dir).unwrap();
    }
}
