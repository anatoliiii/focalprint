//! Enrolled fingers on disk: `<state>/<user>/<finger>`, 0600, written
//! atomically. A template is the touch frames themselves, encrypted with
//! the TPM-sealed key and bound (AAD) to user and finger, so the neural
//! features can be recomputed whenever the model changes.
//!
//! Format `FTE4NT02`: sealed(`enrolled` as u16 LE ‖ frames). The first
//! `enrolled` frames come from enrolment and are never replaced; the rest
//! were added by verification (adaptive template). `FTE4NT01` (frames only,
//! all from enrolment) is still read.

use focalprint_dataset::Key;
use std::{
    fs,
    io::{self, Write as _},
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _},
    path::{Path, PathBuf},
};

/// Finger names as used by the fprintd API.
pub(crate) const FINGERS: [&str; 10] = [
    "left-thumb",
    "left-index-finger",
    "left-middle-finger",
    "left-ring-finger",
    "left-little-finger",
    "right-thumb",
    "right-index-finger",
    "right-middle-finger",
    "right-ring-finger",
    "right-little-finger",
];

pub(crate) fn is_finger(name: &str) -> bool {
    FINGERS.contains(&name)
}

/// Raw touch frames of one finger, wiped on drop.
pub(crate) type Frames = Vec<zeroize::Zeroizing<Vec<u8>>>;

/// One enrolled finger: enrolment touches first, then adaptive ones.
#[derive(Clone)]
pub(crate) struct Template {
    pub(crate) frames: Frames,
    /// How many leading frames come from enrolment.
    pub(crate) enrolled: usize,
}

impl std::fmt::Debug for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Template")
            .field("touches", &self.frames.len())
            .field("enrolled", &self.enrolled)
            .finish_non_exhaustive()
    }
}

impl Template {
    pub(crate) fn enrolled(frames: Frames) -> Self {
        let enrolled = frames.len();
        Self { frames, enrolled }
    }
}

const MAGIC_V1: &[u8; 8] = b"FTE4NT01";
const MAGIC: &[u8; 8] = b"FTE4NT02";
const FRAME_BYTES: usize = 64 * 80;

#[derive(Debug)]
pub(crate) struct Store {
    root: PathBuf,
    key: Key,
}

impl Store {
    pub(crate) fn new(root: impl Into<PathBuf>, key: Key) -> io::Result<Self> {
        let root = root.into();
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&root)?;
        Ok(Self { root, key })
    }

    fn context(user: &str, finger: &str) -> Vec<u8> {
        format!("fte4800-template:{user}:{finger}").into_bytes()
    }

    /// Users that have a template directory.
    pub(crate) fn users(&self) -> Vec<String> {
        fs::read_dir(&self.root)
            .map(|d| {
                d.flatten()
                    .filter(|e| e.path().is_dir())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The per-user directory. User names come from the system account
    /// database, but are still checked so they can never escape `root`.
    fn user_dir(&self, user: &str) -> io::Result<PathBuf> {
        let safe = !user.is_empty()
            && user != "."
            && user != ".."
            && user
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-@$".contains(&b));
        if safe {
            Ok(self.root.join(user))
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsafe user name",
            ))
        }
    }

    pub(crate) fn list(&self, user: &str) -> io::Result<Vec<&'static str>> {
        let dir = self.user_dir(user)?;
        Ok(FINGERS
            .iter()
            .copied()
            .filter(|f| dir.join(f).is_file())
            .collect())
    }

    pub(crate) fn load(&self, user: &str, finger: &str) -> io::Result<Template> {
        let bytes = fs::read(self.user_dir(user)?.join(finger))?;
        let invalid = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let (v1, body) = match (bytes.strip_prefix(MAGIC), bytes.strip_prefix(MAGIC_V1)) {
            (Some(body), _) => (false, body),
            (None, Some(body)) => (true, body),
            (None, None) => return Err(invalid("not a neural template")),
        };
        let plain = self
            .key
            .open(&Self::context(user, finger), body)
            .map_err(|e| invalid(&e.to_string()))?;
        let (enrolled, data) = if v1 {
            (None, &plain[..])
        } else {
            let (head, data) = plain
                .split_first_chunk::<2>()
                .ok_or_else(|| invalid("bad header"))?;
            (Some(usize::from(u16::from_le_bytes(*head))), data)
        };
        if data.is_empty() || data.len() % FRAME_BYTES != 0 {
            return Err(invalid("bad frame data"));
        }
        let frames: Frames = data
            .chunks(FRAME_BYTES)
            .map(|c| zeroize::Zeroizing::new(c.to_vec()))
            .collect();
        let enrolled = enrolled.unwrap_or(frames.len());
        if enrolled == 0 || enrolled > frames.len() {
            return Err(invalid("bad enrolment count"));
        }
        Ok(Template { frames, enrolled })
    }

    /// Every readable template of `user`, with its finger name. Damaged or
    /// outdated files are skipped (and reported) rather than failing.
    pub(crate) fn load_all(&self, user: &str) -> io::Result<Vec<(&'static str, Template)>> {
        let mut out = Vec::new();
        for finger in self.list(user)? {
            match self.load(user, finger) {
                Ok(t) => out.push((finger, t)),
                Err(err) => tracing::warn!(user, finger, %err, "skipping unreadable template"),
            }
        }
        Ok(out)
    }

    pub(crate) fn save(&self, user: &str, finger: &str, template: &Template) -> io::Result<()> {
        let dir = self.user_dir(user)?;
        let enrolled = u16::try_from(template.enrolled)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many touches"))?;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)?;
        let plain = zeroize::Zeroizing::new(
            enrolled
                .to_le_bytes()
                .into_iter()
                .chain(template.frames.iter().flat_map(|f| f.iter().copied()))
                .collect::<Vec<u8>>(),
        );
        let sealed = self
            .key
            .seal(&Self::context(user, finger), &plain)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let target = dir.join(finger);
        let tmp = dir.join(format!(".{finger}.tmp"));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(MAGIC)?;
        file.write_all(&sealed)?;
        file.sync_all()?;
        fs::rename(&tmp, &target)?;
        fs::File::open(&dir)?.sync_all()
    }

    pub(crate) fn delete(&self, user: &str, finger: &str) -> io::Result<()> {
        remove_if_present(&self.user_dir(user)?.join(finger))
    }

    pub(crate) fn delete_all(&self, user: &str) -> io::Result<()> {
        let dir = self.user_dir(user)?;
        for finger in FINGERS {
            remove_if_present(&dir.join(finger))?;
        }
        Ok(())
    }
}

fn remove_if_present(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_tricks() {
        let store = Store {
            root: PathBuf::from("/nonexistent"),
            key: Key::from_bytes(&[0; 32]).unwrap(),
        };
        for bad in ["", ".", "..", "a/b", "../root", "x\0y"] {
            assert!(store.user_dir(bad).is_err(), "{bad:?}");
        }
        assert!(store.user_dir("alice").is_ok());
    }

    fn temp_store() -> (Store, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "focalprintd-store-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        let store = Store::new(&root, Key::from_bytes(&[7; 32]).unwrap()).unwrap();
        (store, root)
    }

    fn frame(v: u8) -> zeroize::Zeroizing<Vec<u8>> {
        zeroize::Zeroizing::new(vec![v; FRAME_BYTES])
    }

    #[test]
    fn round_trip_keeps_enrolment_count() {
        let (store, root) = temp_store();
        let t = Template {
            frames: vec![frame(1), frame(2), frame(3)],
            enrolled: 2,
        };
        store.save("alice", "right-thumb", &t).unwrap();
        let back = store.load("alice", "right-thumb").unwrap();
        assert_eq!(back.enrolled, 2);
        assert_eq!(back.frames.len(), 3);
        assert_eq!(back.frames[2][0], 3);
        // bound to user and finger
        fs::rename(
            root.join("alice/right-thumb"),
            root.join("alice/left-thumb"),
        )
        .unwrap();
        assert!(store.load("alice", "left-thumb").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reads_v1_templates() {
        let (store, root) = temp_store();
        let plain: Vec<u8> = [frame(4), frame(5)]
            .iter()
            .flat_map(|f| f.to_vec())
            .collect();
        let sealed = store
            .key
            .seal(&Store::context("bob", "right-thumb"), &plain)
            .unwrap();
        fs::create_dir_all(root.join("bob")).unwrap();
        fs::write(
            root.join("bob/right-thumb"),
            [&MAGIC_V1[..], &sealed].concat(),
        )
        .unwrap();
        let t = store.load("bob", "right-thumb").unwrap();
        assert_eq!((t.frames.len(), t.enrolled), (2, 2));
        fs::remove_dir_all(root).unwrap();
    }
}
