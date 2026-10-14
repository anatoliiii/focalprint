//! `focalprint dump`: DEVELOPMENT ONLY. Writes raw frames as PGM into a
//! RAM-backed directory (`/dev/shm/...`) for offline algorithm work, owned
//! by the invoking user, mode 0700. Refuses anything not on tmpfs.

use focalprint_sensor::{Image, Sensor, Wait};
use std::{
    fs,
    io::Write as _,
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _},
    path::Path,
};

pub(crate) fn run(
    sensor: &mut dyn Sensor,
    wait: Wait<'_>,
    out: &Path,
    label: &str,
    count: u32,
    empty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !out.starts_with("/dev/shm") {
        return Err("refusing to write frames outside /dev/shm".into());
    }
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(out)?;
    give_to_caller(out)?;
    for n in 1..=count {
        let image = if empty {
            sensor.read_now()?
        } else {
            println!("[{n}/{count}] {label}: touch…");
            sensor.capture(wait)?
        };
        let path = out.join(format!("{label}-{n:02}.pgm"));
        save(&path, &image)?;
        give_to_caller(&path)?;
        if empty {
            println!("[{n}/{count}] {} (no finger)", path.display());
            std::thread::sleep(std::time::Duration::from_millis(300));
        } else {
            println!("        {} — lift", path.display());
            sensor.wait_release(wait)?;
        }
    }
    Ok(())
}

fn save(path: &Path, image: &Image) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    write!(file, "P5\n{} {}\n255\n", Image::WIDTH, Image::HEIGHT)?;
    file.write_all(image.pixels())
}

/// Hands the file to the user who ran `sudo`, so analysis needs no root.
fn give_to_caller(path: &Path) -> std::io::Result<()> {
    let id = |var| std::env::var(var).ok().and_then(|v| v.parse::<u32>().ok());
    if let (Some(uid), Some(gid)) = (id("SUDO_UID"), id("SUDO_GID")) {
        std::os::unix::fs::chown(path, Some(uid), Some(gid))?;
    }
    Ok(())
}
