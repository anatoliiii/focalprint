//! Diagnostics for FocalTech FTE4800 (SPI) and FT9201 (USB) fingerprint sensors.
//!
//! Frames are only ever shown on the terminal, never written anywhere.

mod dataset;
mod dump;
mod eval;
mod identify;
mod keys;
mod neural;
mod setup;

use clap::{Parser, Subcommand};
use focalprint_sensor::{Backend, DEFAULT_PATH, Device, Ft9368, Image, Interrupt, Transport, Wait};
use std::{
    fmt::Write as _,
    path::PathBuf,
    process::ExitCode,
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Sensor family: `spi`, `usb`, or `auto` (SPI if its node exists).
    #[arg(long, default_value = "auto")]
    backend: Backend,
    /// Device node of the kernel module (SPI backend).
    #[arg(long, default_value = DEFAULT_PATH)]
    device: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// One-time machine setup (root): state directory, TPM-sealed key,
    /// matcher model, sensor check. Safe to repeat.
    Setup {
        /// FpNet ONNX model to install.
        #[arg(long)]
        model: Option<PathBuf>,
        /// State directory of focalprintd.
        #[arg(long, default_value = "/var/lib/focalprint")]
        state_dir: PathBuf,
    },
    /// Opt-in research collection of focalprintd (encrypted, TPM-sealed key).
    Dataset {
        /// Template storage of focalprintd.
        #[arg(long, default_value = "/var/lib/focalprint")]
        state_dir: PathBuf,
        #[command(subcommand)]
        action: dataset::Action,
    },
    /// Identify the sensor; for SPI also what the kernel module reports.
    Info,
    /// SPI only: pulse the hardware reset line and identify the sensor again.
    Reset,
    /// Wait for touches and preview each frame in the terminal.
    Capture {
        /// Number of touches to capture.
        #[arg(short = 'n', long, default_value_t = 1)]
        count: u32,
        /// Seconds to wait for each touch.
        #[arg(short, long, default_value_t = 30)]
        timeout: u64,
    },
    /// Tell which enrolled finger touches, at what angle and offset.
    /// Needs the daemon stopped (`dev-switch rust`).
    Identify {
        /// Number of touches.
        #[arg(short = 'n', long, default_value_t = 10)]
        count: u32,
        /// Template storage of focalprintd.
        #[arg(long, default_value = "/var/lib/focalprint")]
        state_dir: PathBuf,
    },
    /// DEVELOPMENT ONLY: save frames as PGM under /dev/shm (RAM) for
    /// offline algorithm work. Delete the directory when done.
    Dump {
        /// Target directory; must be under /dev/shm.
        #[arg(long, default_value = "/dev/shm/focalprint-frames")]
        out: PathBuf,
        /// File name prefix, e.g. the finger.
        #[arg(long)]
        label: String,
        /// Number of frames.
        #[arg(short = 'n', long, default_value_t = 12)]
        count: u32,
        /// Read frames without waiting for a touch (sensor background).
        #[arg(long)]
        empty: bool,
    },
    /// Live test of the neural matcher: enrol fingers in memory, then score
    /// touches. Nothing is stored.
    Neural {
        /// ONNX export of FpNet.
        #[arg(long, default_value = "/dev/shm/focalprint-models/fpnet.onnx")]
        model: PathBuf,
        #[arg(long, default_value_t = 1)]
        fingers: usize,
        #[arg(long, default_value_t = 20)]
        enroll: usize,
        #[arg(short = 'n', long, default_value_t = 20)]
        tries: usize,
        #[arg(long, default_value_t = 0.28)]
        threshold: f32,
    },
    /// Capture several fingers and measure how well they are told apart.
    Eval {
        /// Number of different fingers.
        #[arg(long, default_value_t = 2)]
        fingers: usize,
        /// Touches per finger.
        #[arg(short = 'n', long, default_value_t = 14)]
        touches: usize,
        /// Of those, how many form the template; the rest are verified.
        #[arg(long, default_value_t = 8)]
        enroll: usize,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("focalprint: {err}");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)] // one arm per subcommand
fn run(cli: &Cli) -> focalprint_sensor::Result<()> {
    if let Command::Setup { model, state_dir } = &cli.command {
        if let Err(err) = setup::run(state_dir, model.as_deref()) {
            eprintln!("focalprint: {err}");
        }
        return Ok(());
    }
    if let Command::Dataset { state_dir, action } = &cli.command {
        if let Err(err) = dataset::run(state_dir, action) {
            eprintln!("focalprint: {err}");
        }
        return Ok(());
    }
    let spi = cli.backend.resolve(&cli.device) == Backend::Spi;
    if spi && matches!(cli.command, Command::Info | Command::Reset) {
        return spi_info(cli);
    }
    let mut sensor = focalprint_sensor::open(cli.backend, &cli.device)?;
    println!("sensor:    {}", sensor.describe());
    let sensor = sensor.as_mut();

    match cli.command {
        // Handled above, before the device is opened.
        Command::Setup { .. } | Command::Dataset { .. } | Command::Info => {}
        Command::Reset => return Err(focalprint_sensor::Error::Protocol("reset is SPI only")),
        Command::Capture { count, timeout } => {
            let cancel = AtomicBool::new(false);
            let wait = Wait {
                timeout: Duration::from_secs(timeout),
                cancel: &cancel,
            };
            for n in 1..=count {
                println!("\n[{n}/{count}] touch the sensor…");
                let started = Instant::now();
                let image = sensor.capture(wait)?;
                let touched = started.elapsed();
                print!("{}", preview(&image));
                println!("{}", Stats::of(&image));
                println!(
                    "touch after {:.1} s; lift your finger…",
                    touched.as_secs_f32()
                );
                sensor.wait_release(wait)?;
            }
        }
        Command::Dump {
            ref out,
            ref label,
            count,
            empty,
        } => {
            let cancel = AtomicBool::new(false);
            let wait = Wait {
                timeout: Duration::from_secs(60),
                cancel: &cancel,
            };
            if let Err(err) = dump::run(sensor, wait, out, label, count, empty) {
                eprintln!("focalprint: {err}");
            }
        }
        Command::Neural {
            ref model,
            fingers,
            enroll,
            tries,
            threshold,
        } => {
            let cancel = AtomicBool::new(false);
            let wait = Wait {
                timeout: Duration::from_secs(120),
                cancel: &cancel,
            };
            if let Err(err) = neural::run(sensor, wait, model, fingers, enroll, tries, threshold) {
                eprintln!("focalprint: {err}");
            }
        }
        Command::Identify {
            count,
            ref state_dir,
        } => {
            let cancel = AtomicBool::new(false);
            let wait = Wait {
                timeout: Duration::from_secs(60),
                cancel: &cancel,
            };
            identify::run(sensor, wait, state_dir, count)?;
        }
        Command::Eval {
            fingers,
            touches,
            enroll,
        } => {
            let cancel = AtomicBool::new(false);
            let wait = Wait {
                timeout: Duration::from_secs(60),
                cancel: &cancel,
            };
            eval::run(sensor, wait, fingers, touches, enroll)?;
        }
    }
    Ok(())
}

/// SPI diagnostics: what the kernel module reports, then the status block.
fn spi_info(cli: &Cli) -> focalprint_sensor::Result<()> {
    let mut device = Device::open(&cli.device)?;
    let info = device.info();
    println!("device:    {}", cli.device.display());
    println!(
        "spi:       {} kHz, mode {:#x}",
        info.speed_hz / 1000,
        info.spi_mode
    );
    println!("irq count: {}", device.irq_count()?);

    let mut sensor = Ft9368::new(device);
    if matches!(cli.command, Command::Reset) {
        sensor.io_mut().reset()?;
        println!("reset:     done");
    }
    let identity = sensor.identify()?;
    println!("firmware:  {:#04x}", identity.firmware);

    let status = sensor.status()?;
    println!("status:    {}", hex(status.as_bytes()));
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Two pixel rows per text row: upper half block in the foreground colour,
/// lower one in the background colour, contrast-stretched to the frame.
fn preview(image: &Image) -> String {
    let stats = Stats::of(image);
    let span = u32::from(stats.max.saturating_sub(stats.min)).max(1);
    let level = |v: u8| u32::from(v.saturating_sub(stats.min)) * 255 / span;
    let mut out = String::new();
    for y in (0..Image::HEIGHT).step_by(2) {
        for x in 0..Image::WIDTH {
            let (top, bottom) = (level(image.at(x, y)), level(image.at(x, y + 1)));
            let _ = write!(
                out,
                "\x1b[38;2;{top};{top};{top}m\x1b[48;2;{bottom};{bottom};{bottom}m▀"
            );
        }
        out.push_str("\x1b[0m\n");
    }
    out
}

struct Stats {
    min: u8,
    max: u8,
    mean: f64,
    stddev: f64,
}

impl Stats {
    fn of(image: &Image) -> Self {
        let pixels = image.pixels();
        let n = f64::from(u32::try_from(pixels.len()).unwrap_or(u32::MAX));
        let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        let var = pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n;
        Self {
            min: pixels.iter().copied().min().unwrap_or(0),
            max: pixels.iter().copied().max().unwrap_or(0),
            mean,
            stddev: var.sqrt(),
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "range {}..{}, mean {:.1}, contrast (σ) {:.1}",
            self.min, self.max, self.mean, self.stddev
        )
    }
}
