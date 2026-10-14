//! `focalprintd`: fingerprint daemon for FocalTech FTE4800 (FT9368, SPI) and
//! FT9201 (USB) sensors.
//!
//! Speaks fprintd's `net.reactivated.Fprint` D-Bus API. Frames exist only in
//! locked, never-dumped memory of this process and are wiped after feature
//! extraction; only templates (features) reach the disk.

mod dbus;
mod sensor;
mod store;

use clap::Parser;
use std::{path::PathBuf, process::ExitCode, time::Duration};
use tokio::signal::unix::{SignalKind, signal};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Sensor family: `spi` (FTE4800 via the kernel module), `usb` (FT9201),
    /// or `auto` (SPI if its device node exists, else USB).
    #[arg(long, default_value = "auto")]
    backend: focalprint_sensor::Backend,
    /// Device node of the kernel module (SPI backend).
    #[arg(long, default_value = focalprint_sensor::DEFAULT_PATH)]
    device: PathBuf,
    /// Template storage; systemd's `StateDirectory=` wins when set.
    #[arg(long, env = "STATE_DIRECTORY", default_value = "/var/lib/focalprint")]
    state_dir: PathBuf,
    /// Exit after this many idle seconds (0 = never). With the neural
    /// matcher, staying up keeps template features warm.
    #[arg(long, default_value_t = 0)]
    idle_timeout: u64,
    /// FpNet ONNX model.
    #[arg(long, default_value = "/var/lib/focalprint/model/fpnet.onnx")]
    model: PathBuf,
    /// Decision threshold on the dense similarity score.
    #[arg(long, default_value_t = 0.30)]
    threshold: f32,
    /// Adaptive template: matches scoring at least this much join the
    /// finger's template (never below `--threshold`; above 1 disables).
    #[arg(long, default_value_t = 0.50)]
    adapt_score: f32,
}

fn harden() {
    use rustix::{
        fs::Mode,
        mm::{MlockAllFlags, mlockall},
        process::{DumpableBehavior, set_dumpable_behavior, umask},
    };
    // No core dumps or ptrace by other users: memory may hold frames.
    if let Err(err) = set_dumpable_behavior(DumpableBehavior::NotDumpable) {
        tracing::warn!(%err, "cannot disable core dumps");
    }
    // Keep everything out of swap.
    if let Err(err) = mlockall(MlockAllFlags::CURRENT | MlockAllFlags::FUTURE) {
        tracing::warn!(%err, "cannot lock memory; frames could reach swap");
    }
    umask(Mode::from_raw_mode(0o077));
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .without_time()
        .init();
    let args = Args::parse();
    harden();
    match run(&args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!("{err}");
            ExitCode::FAILURE
        }
    }
}

/// The TPM-sealed key systemd hands us (`LoadCredentialEncrypted=`):
/// encrypts templates and research records.
fn credential_key() -> Result<focalprint_dataset::Key, Box<dyn std::error::Error>> {
    let creds = std::env::var_os("CREDENTIALS_DIRECTORY")
        .ok_or("no credentials directory: the TPM-sealed key is required")?;
    Ok(focalprint_dataset::Key::load(
        &std::path::Path::new(&creds).join(focalprint_dataset::CREDENTIAL),
    )?)
}

/// The opt-in research collection, enabled by `focalprint dataset enable`.
fn research_collection(
    state_dir: &std::path::Path,
    key: &focalprint_dataset::Key,
) -> Option<focalprint_dataset::Collection> {
    let dir = state_dir.join("dataset");
    if !dir.join("enabled").exists() {
        return None;
    }
    match focalprint_dataset::Collection::open(dir.join("records"), key.duplicate()) {
        Ok(collection) => {
            tracing::info!("research dataset collection is ON (encrypted)");
            Some(collection)
        }
        Err(err) => {
            tracing::warn!(%err, "cannot open dataset collection; not collecting");
            None
        }
    }
}

async fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let key = credential_key()?;
    let store = store::Store::new(args.state_dir.join("templates"), key.duplicate())?;
    let net = focalprint_net::FpNet::load(&args.model)?;
    tracing::info!(
        model = %args.model.display(),
        threshold = args.threshold,
        adapt_score = args.adapt_score,
        "neural matcher loaded"
    );
    let matcher = sensor::Matcher {
        net,
        threshold: args.threshold,
        adapt_score: args.adapt_score.max(args.threshold),
    };
    let sensor = sensor::Sensor::start(
        args.backend,
        &args.device,
        matcher,
        research_collection(&args.state_dir, &key),
    )?;
    // Precompute every enrolled template so the first unlock is fast.
    let templates: Vec<_> = store
        .users()
        .iter()
        .flat_map(|u| store.load_all(u).unwrap_or_default())
        .map(|(_, t)| std::sync::Arc::new(t))
        .collect();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    sensor.submit(sensor::Job::Warm { templates }, tx);
    let daemon = dbus::Daemon::new(store, sensor);

    let conn = zbus::connection::Builder::system()?
        .serve_at(dbus::MANAGER_PATH, dbus::Manager)?
        .serve_at(
            dbus::DEVICE_PATH,
            dbus::Device {
                daemon: daemon.clone(),
            },
        )?
        .name(dbus::BUS_NAME)?
        .build()
        .await?;
    tracing::info!("serving {} on the system bus", dbus::BUS_NAME);
    tokio::spawn({
        let (conn, daemon) = (conn.clone(), daemon.clone());
        async move {
            if let Err(err) = dbus::watch_clients(conn, daemon).await {
                tracing::error!(%err, "client watch failed");
            }
        }
    });

    let mut term = signal(SignalKind::terminate())?;
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    let idle = Duration::from_secs(args.idle_timeout);
    loop {
        tokio::select! {
            _ = term.recv() => break,
            _ = tokio::signal::ctrl_c() => break,
            _ = tick.tick() => {
                if args.idle_timeout != 0 && daemon.idle_for().await.is_some_and(|t| t >= idle) {
                    tracing::info!("idle, exiting");
                    break;
                }
            }
        }
    }
    Ok(())
}
