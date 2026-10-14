//! `focalprint setup`: one-time machine setup, as root. Creates the state
//! directory and the TPM-sealed machine key, installs a matcher model and
//! reports which sensor was found. Safe to run again.

use crate::keys;
use focalprint_dataset::Key;
use focalprint_sensor::{Backend, DEFAULT_PATH};
use std::{
    fs,
    io::Write as _,
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _, PermissionsExt as _},
    path::Path,
};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Model location inside the state directory, as focalprintd expects it.
const MODEL: &str = "model/fpnet.onnx";

pub(crate) fn run(state_dir: &Path, model: Option<&Path>) -> Result {
    private_dir(state_dir)?;

    let cred = keys::path(state_dir);
    if cred.exists() {
        keys::unseal(&cred)?;
        println!("machine key:  {} (sealed to this TPM, OK)", cred.display());
    } else {
        keys::seal(&Key::generate()?, &cred)?;
        println!("machine key:  created {}", cred.display());
    }

    let installed = state_dir.join(MODEL);
    if let Some(model) = model {
        // Refuse anything the daemon could not load.
        focalprint_net::FpNet::load(model)?;
        private_dir(installed.parent().ok_or("bad model path")?)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&installed)?;
        file.write_all(&fs::read(model)?)?;
        println!("model:        installed {}", installed.display());
    } else if installed.exists() {
        println!("model:        {}", installed.display());
    } else {
        println!("model:        MISSING — focalprintd cannot start without one:");
        println!("              sudo focalprint setup --model /path/to/fpnet.onnx");
    }

    match Backend::detect() {
        Some(Backend::Spi) if Path::new(DEFAULT_PATH).exists() => {
            println!("sensor:       FTE4800 / FT9368 (SPI), kernel module loaded");
        }
        Some(Backend::Spi) => {
            println!("sensor:       FTE4800 / FT9368 (SPI), but the kernel module is NOT loaded;");
            println!("              see docs/megabook-t1.md (needs focalprint-dkms)");
        }
        Some(_) => println!("sensor:       FT9201 (USB)"),
        None => println!("sensor:       none of the supported sensors found"),
    }

    println!("\nnext: sudo systemctl restart fprintd   # runs focalprintd");
    println!("      fprintd-enroll                     # or your desktop's settings");
    Ok(())
}

fn private_dir(dir: &Path) -> Result {
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    Ok(())
}
