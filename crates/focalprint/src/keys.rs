//! The machine key: encrypts templates and research records, sealed to the
//! TPM with `systemd-creds` so it is useless off this machine. focalprintd
//! receives it through `LoadCredentialEncrypted=`.

use focalprint_dataset::{CREDENTIAL, KEY_BYTES, Key};
use std::{
    fs,
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use zeroize::Zeroizing;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Where the sealed key lives inside the state directory.
pub(crate) fn path(state_dir: &Path) -> PathBuf {
    state_dir.join("key.cred")
}

pub(crate) fn seal(key: &Key, cred: &Path) -> Result {
    let mut child = Command::new("systemd-creds")
        .args([
            "encrypt",
            "--with-key=tpm2",
            &format!("--name={CREDENTIAL}"),
            "-",
        ])
        .arg(cred)
        .stdin(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(key.as_bytes())?;
    if !child.wait()?.success() {
        return Err("systemd-creds encrypt failed".into());
    }
    fs::set_permissions(cred, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    Ok(())
}

pub(crate) fn unseal(cred: &Path) -> Result<Key> {
    let mut child = Command::new("systemd-creds")
        .args(["decrypt", &format!("--name={CREDENTIAL}")])
        .arg(cred)
        .arg("-")
        .stdout(Stdio::piped())
        .spawn()?;
    let mut bytes = Zeroizing::new(Vec::with_capacity(KEY_BYTES));
    child
        .stdout
        .take()
        .ok_or("no stdout")?
        .read_to_end(&mut bytes)?;
    if !child.wait()?.success() {
        return Err("systemd-creds decrypt failed".into());
    }
    Ok(Key::from_bytes(&bytes)?)
}
