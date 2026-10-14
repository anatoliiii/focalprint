//! `focalprint dataset`: manage the opt-in research collection of
//! focalprintd. Records are encrypted with a key sealed to this machine's TPM
//! (`systemd-creds --with-key=tpm2`); `export` decrypts only into /dev/shm.

use crate::keys;
use clap::Subcommand;
use focalprint_dataset::Collection;
use std::{
    fmt::Write as _,
    fs,
    io::Write as _,
    os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _},
    path::{Path, PathBuf},
};
use zeroize::Zeroizing;

#[derive(Debug, Subcommand)]
pub(crate) enum Action {
    /// Start collecting every touch. Takes effect when focalprintd starts
    /// next.
    Enable,
    /// Stop collecting; stored records stay.
    Disable,
    /// Show whether collection is on and how much is stored.
    Stats,
    /// Decrypt everything into a RAM directory for research.
    Export {
        #[arg(long, default_value = "/dev/shm/focalprint-dataset")]
        out: PathBuf,
    },
    /// Delete all records. Irreversible. The machine key stays: it also
    /// protects the enrolled fingers.
    Wipe,
    /// Collect labelled touches straight into the encrypted set (needs the
    /// sensor: stop focalprintd first, e.g. `dev-switch rust`).
    Capture {
        /// fprintd finger name, e.g. right-index-finger.
        #[arg(long)]
        finger: String,
        #[arg(short = 'n', long, default_value_t = 50)]
        count: u32,
        /// Account the finger belongs to (default: the sudo caller).
        #[arg(long)]
        user: Option<String>,
    },
    /// Encrypt PGM frames from a `dump` directory into the set; files named
    /// `<label>-NN.pgm`, labels mapped with --map label=finger.
    Import {
        #[arg(long, default_value = "/dev/shm/focalprint-frames")]
        dir: PathBuf,
        /// e.g. index=right-index-finger (repeatable); unmapped labels skipped.
        #[arg(long = "map")]
        maps: Vec<String>,
        #[arg(long)]
        user: Option<String>,
    },
}

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn run(state_dir: &Path, action: &Action) -> Result {
    let dir = state_dir.join("dataset");
    let cred = keys::path(state_dir);
    match action {
        Action::Enable => {
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&dir)?;
            if !cred.exists() {
                return Err("no machine key yet: run `focalprint setup` first".into());
            }
            fs::write(dir.join("enabled"), b"")?;
            println!("collection enabled; restart focalprintd to apply");
        }
        Action::Disable => {
            remove(&dir.join("enabled"))?;
            println!("collection disabled; restart focalprintd to apply");
        }
        Action::Stats => {
            let on = dir.join("enabled").exists();
            let records = fs::read_dir(dir.join("records")).map_or(0, |d| d.flatten().count());
            println!("collection: {}", if on { "ON" } else { "off" });
            println!(
                "key:        {}",
                if cred.exists() {
                    "sealed to TPM"
                } else {
                    "none"
                }
            );
            println!("records:    {records}");
        }
        Action::Export { out } => export(&dir, &cred, out)?,
        Action::Capture {
            finger,
            count,
            user,
        } => capture(&dir, &cred, finger, *count, &owner(user.as_deref())?)?,
        Action::Import {
            dir: from,
            maps,
            user,
        } => {
            import(&dir, &cred, from, maps, &owner(user.as_deref())?)?;
        }
        Action::Wipe => {
            if dir.exists() {
                fs::remove_dir_all(&dir)?;
            }
            println!("dataset deleted");
        }
    }
    Ok(())
}

fn export(dir: &Path, cred: &Path, out: &Path) -> Result {
    if !out.starts_with("/dev/shm") {
        return Err("refusing to export outside /dev/shm".into());
    }
    let collection = Collection::open(dir.join("records"), keys::unseal(cred)?)?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(out)?;
    let mut labels = String::from("file,unix_ms,kind,user,finger,outcome,inliers\n");
    let mut count = 0usize;
    for path in collection.list()? {
        let record = match collection.read(&path) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("skipping {}: {err}", path.display());
                continue;
            }
        };
        let name = format!("{count:05}.pgm");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(out.join(&name))?;
        write!(file, "P5\n{} {}\n255\n", record.width, record.height)?;
        file.write_all(&record.pixels)?;
        let _ = writeln!(
            labels,
            "{name},{},{:?},{},{},{},{}",
            record.unix_ms,
            record.kind,
            record.user,
            record.finger.as_deref().unwrap_or(""),
            record.outcome,
            record.inliers
        );
        count += 1;
    }
    fs::write(out.join("labels.csv"), labels)?;
    give_to_caller(out)?;
    println!(
        "exported {count} records to {} (RAM only; delete when done)",
        out.display()
    );
    Ok(())
}

fn remove(path: &Path) -> std::io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

fn give_to_caller(dir: &Path) -> std::io::Result<()> {
    let id = |var| std::env::var(var).ok().and_then(|v| v.parse::<u32>().ok());
    let (Some(uid), Some(gid)) = (id("SUDO_UID"), id("SUDO_GID")) else {
        return Ok(());
    };
    std::os::unix::fs::chown(dir, Some(uid), Some(gid))?;
    for entry in fs::read_dir(dir)?.flatten() {
        std::os::unix::fs::chown(entry.path(), Some(uid), Some(gid))?;
    }
    Ok(())
}

fn owner(user: Option<&str>) -> Result<String> {
    user.map(String::from)
        .or_else(|| std::env::var("SUDO_USER").ok())
        .ok_or_else(|| "pass --user".into())
}

fn record(
    user: &str,
    finger: &str,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
) -> focalprint_dataset::Record {
    focalprint_dataset::Record {
        unix_ms: focalprint_dataset::now_ms(),
        kind: focalprint_dataset::Kind::Capture,
        user: user.into(),
        finger: Some(finger.into()),
        outcome: "capture".into(),
        inliers: 0,
        width: u16::try_from(width).unwrap_or(0),
        height: u16::try_from(height).unwrap_or(0),
        pixels,
    }
}

fn capture(dir: &Path, cred: &Path, finger: &str, count: u32, user: &str) -> Result {
    use focalprint_sensor::{Backend, DEFAULT_PATH, Image, Wait};
    let collection = Collection::open(dir.join("records"), keys::unseal(cred)?)?;
    let mut sensor = focalprint_sensor::open(Backend::Auto, DEFAULT_PATH.as_ref())?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let wait = Wait {
        timeout: std::time::Duration::from_secs(120),
        cancel: &cancel,
    };
    println!("{finger}: {count} touches. Vary position and angle: centre, edges, tip, tilted.");
    for n in 1..=count {
        print!("  [{n}/{count}] touch… ");
        std::io::stdout().flush()?;
        let image = sensor.capture(wait)?;
        collection.add(&record(
            user,
            finger,
            image.pixels().to_vec(),
            Image::WIDTH,
            Image::HEIGHT,
        ))?;
        print!("stored, lift… ");
        std::io::stdout().flush()?;
        sensor.wait_release(wait)?;
        println!("ok");
    }
    Ok(())
}

fn import(dir: &Path, cred: &Path, from: &Path, maps: &[String], user: &str) -> Result {
    let collection = Collection::open(dir.join("records"), keys::unseal(cred)?)?;
    let maps: Vec<(&str, &str)> = maps.iter().filter_map(|m| m.split_once('=')).collect();
    let mut entries: Vec<PathBuf> = fs::read_dir(from)?.flatten().map(|e| e.path()).collect();
    entries.sort();
    let mut count = 0;
    for path in entries {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((label, _)) = name.strip_suffix(".pgm").and_then(|n| n.rsplit_once('-')) else {
            continue;
        };
        let Some((_, finger)) = maps.iter().find(|(l, _)| *l == label) else {
            continue;
        };
        let bytes = Zeroizing::new(fs::read(&path)?);
        let Some((width, height, pixels)) = parse_pgm(&bytes) else {
            eprintln!("skipping {}: not a binary PGM", path.display());
            continue;
        };
        collection.add(&record(user, finger, pixels.to_vec(), width, height))?;
        count += 1;
    }
    println!(
        "imported {count} frames; you can now delete {}",
        from.display()
    );
    Ok(())
}

/// Minimal `P5` reader for files written by `dump` (no comments).
fn parse_pgm(bytes: &[u8]) -> Option<(usize, usize, &[u8])> {
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes.get(at)?.is_ascii_whitespace() {
            at += 1;
        }
        let start = at;
        while !bytes.get(at)?.is_ascii_whitespace() {
            at += 1;
        }
        fields.push(std::str::from_utf8(&bytes[start..at]).ok()?);
    }
    let (w, h): (usize, usize) = (fields[1].parse().ok()?, fields[2].parse().ok()?);
    let data = bytes.get(at + 1..at + 1 + w * h)?;
    (fields[0] == "P5" && fields[3] == "255").then_some((w, h, data))
}
