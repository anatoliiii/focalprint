//! Matching speed on synthetic ridge-like frames: `cargo run --release -p focalprint-net --example bench -- model.onnx`
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
use focalprint_net::{FpNet, HEIGHT, WIDTH, matching};
use std::time::Instant;

fn frame(seed: f32) -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let r =
                ((x - 32.0 + seed * 7.0).powi(2) * 0.7 + (y - 50.0 + seed * 3.0).powi(2)).sqrt();
            (128.0 + 100.0 * (r * 0.95 + (x * 0.13 + seed).sin() * 1.5).sin()) as u8
        })
        .collect()
}

fn main() {
    let model = std::env::args().nth(1).expect("model path");
    let net = FpNet::load(model.as_ref()).unwrap();
    let feats = |s: f32| {
        let f = frame(s);
        matching::Features::new(&f, net.run(&f).unwrap())
    };
    let template: Vec<_> = (0..40).map(|i| feats(i as f32 * 0.37)).collect();
    let query = feats(100.0);
    for _ in 0..2 {
        let t = Instant::now();
        let s = matching::score(&query, &template);
        println!(
            "score {s:.3} in {:.0} ms (full scan)",
            t.elapsed().as_secs_f32() * 1000.0
        );
        let t = Instant::now();
        let v = matching::verify(&query, &template, 0.28);
        println!(
            "verify {v:?} in {:.0} ms (early exit)",
            t.elapsed().as_secs_f32() * 1000.0
        );
    }
}
