//! `focalprint neural`: live test of the FpNet matcher on the sensor.
//! Enrols fingers into memory-only templates, then scores every touch
//! against each of them and reports timings. Nothing is stored.

use focalprint_net::{FpNet, matching};
use focalprint_sensor::{Image, Sensor, Wait};
use std::{path::Path, time::Instant};

pub(crate) fn run(
    sensor: &mut dyn Sensor,
    wait: Wait<'_>,
    model: &Path,
    fingers: usize,
    enroll: usize,
    tries: usize,
    threshold: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let t = Instant::now();
    let net = FpNet::load(model)?;
    println!(
        "model loaded in {:.0} ms",
        t.elapsed().as_secs_f32() * 1000.0
    );

    let touch = |sensor: &mut dyn Sensor| -> Result<(matching::Features, f32, f32), Box<dyn std::error::Error>> {
        let t0 = Instant::now();
        let image = sensor.capture(wait)?;
        let captured = t0.elapsed().as_secs_f32() * 1000.0;
        let t1 = Instant::now();
        let map = net.run(image.pixels())?;
        let features = matching::Features::new(image.pixels(), map);
        let inferred = t1.elapsed().as_secs_f32() * 1000.0;
        sensor.wait_release(wait)?;
        Ok((features, captured, inferred))
    };

    let mut templates: Vec<(String, Vec<matching::Features>)> = Vec::new();
    for f in 0..fingers {
        let name = char::from(b'A' + u8::try_from(f).unwrap_or(0)).to_string();
        println!("\nfinger {name}: {enroll} touches, cover the pad (centre, edges, tip, turned)");
        let mut touches = Vec::with_capacity(enroll);
        for n in 1..=enroll {
            print!("  [{n}/{enroll}] ");
            flush();
            let (features, _, inferred) = touch(sensor)?;
            println!("ok ({inferred:.0} ms network)");
            touches.push(features);
        }
        templates.push((name, touches));
    }

    println!("\nverification: touch any enrolled finger (or another one); threshold {threshold}");
    for n in 1..=tries {
        print!("[{n}/{tries}] touch… ");
        flush();
        let (query, captured, inferred) = touch(sensor)?;
        let t = Instant::now();
        let mut scores: Vec<(&str, f32)> = templates
            .iter()
            .map(|(name, t)| (name.as_str(), matching::score(&query, t)))
            .collect();
        let matched = t.elapsed().as_secs_f32() * 1000.0;
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let verdict = match scores.first() {
            Some((name, s)) if *s >= threshold => format!("MATCH {name}"),
            _ => "no match".into(),
        };
        let all: Vec<String> = scores
            .iter()
            .map(|(name, s)| format!("{name}={s:.3}"))
            .collect();
        println!(
            "{verdict:<9} [{}]  capture {captured:.0} ms, network {inferred:.0} ms, matching {matched:.0} ms",
            all.join(" ")
        );
    }
    Ok(())
}

fn flush() {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
}

#[allow(dead_code)]
fn _image_size_check() {
    const { assert!(Image::WIDTH == focalprint_net::WIDTH && Image::HEIGHT == focalprint_net::HEIGHT) };
}
