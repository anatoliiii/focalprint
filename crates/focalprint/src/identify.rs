//! `focalprint identify`: on every touch, show which enrolled finger it is,
//! at what angle and offset relative to the template, and how close each
//! finger came. Read-only: templates are never changed.

use focalprint_match::{Policy, Sample, Template};
use focalprint_sensor::{Image, Sensor, Wait};
use std::{fs, path::Path};

pub(crate) fn run(
    sensor: &mut dyn Sensor,
    wait: Wait<'_>,
    state_dir: &Path,
    count: u32,
) -> focalprint_sensor::Result<()> {
    let templates = load(state_dir);
    if templates.is_empty() {
        println!("no templates in {}", state_dir.display());
        return Ok(());
    }
    for (name, t) in &templates {
        println!(
            "enrolled: {name}: {} touches ({} learnt) in {} mosaic(s)",
            t.len(),
            t.learned(),
            t.mosaics()
        );
    }
    let policy = Policy::DEFAULT;
    for n in 1..=count {
        println!("\n[{n}/{count}] touch the sensor…");
        let image = sensor.capture(wait)?;
        let sample = Sample::extract(image.pixels(), Image::WIDTH, Image::HEIGHT);
        drop(image);
        match sample {
            Err(why) => println!("  frame rejected: {why}"),
            Ok(sample) => {
                let mut rows: Vec<_> = templates
                    .iter()
                    .map(|(name, t)| (name, t.verify(&sample, &policy)))
                    .collect();
                rows.sort_by(|a, b| b.1.best.score().total_cmp(&a.1.best.score()));
                for (name, v) in &rows {
                    let e = &v.best;
                    let verdict = if v.matched { "MATCH" } else { "  -  " };
                    let placed = if e.inliers >= 3 {
                        format!(
                            "turned {:+5.0}°, shifted {:+3.0},{:+3.0} px",
                            e.rotation.to_degrees(),
                            e.shift.0,
                            e.shift.1
                        )
                    } else {
                        "no common area with the template".into()
                    };
                    println!(
                        "  {verdict} {name:<28} {:>3} points (need {}) of {:>3} candidates, {:>2} cells, flow {:.2}, hamming {:>4.1}  {placed}",
                        e.inliers,
                        policy.min_inliers,
                        e.candidates,
                        e.cells,
                        e.agreement,
                        e.hamming
                    );
                }
            }
        }
        sensor.wait_release(wait)?;
    }
    Ok(())
}

/// `user/finger` → template, for every readable template under `dir`.
fn load(dir: &Path) -> Vec<(String, Template)> {
    let mut out = Vec::new();
    let Ok(users) = fs::read_dir(dir) else {
        return out;
    };
    for user in users.flatten() {
        let Ok(fingers) = fs::read_dir(user.path()) else {
            continue;
        };
        for finger in fingers.flatten() {
            let name = finger.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let bytes = zeroize::Zeroizing::new(fs::read(finger.path()).unwrap_or_default());
            match Template::from_bytes(&bytes) {
                Ok(t) => out.push((format!("{}/{name}", user.file_name().to_string_lossy()), t)),
                Err(err) => eprintln!("  skipping {}: {err}", finger.path().display()),
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
