//! `focalprint eval`: measure how well the matcher separates fingers on this
//! very sensor. Frames stay in memory and are wiped when the command ends.

use focalprint_match::{Enrollment, Evidence, Params, Policy, Progress, Sample, compare};
use focalprint_sensor::{Image, Sensor, Wait};

/// Detector settings tried against the same captured frames.
const GRID: &[Params] = &[
    Params {
        harris_sigma: 0.7,
        nms_radius: 2,
        relative_threshold: 0.002,
        max_keypoints: 200,
    },
    Params {
        harris_sigma: 0.7,
        nms_radius: 2,
        relative_threshold: 0.002,
        max_keypoints: 300,
    },
    Params {
        harris_sigma: 0.7,
        nms_radius: 1,
        relative_threshold: 0.002,
        max_keypoints: 300,
    },
    Params {
        harris_sigma: 0.5,
        nms_radius: 2,
        relative_threshold: 0.002,
        max_keypoints: 300,
    },
    Params {
        harris_sigma: 0.5,
        nms_radius: 1,
        relative_threshold: 0.002,
        max_keypoints: 400,
    },
];

/// `min_inliers` values reported side by side; the rest of the policy is
/// [`Policy::DEFAULT`].
const INLIER_THRESHOLDS: [usize; 6] = [4, 5, 6, 7, 8, 10];

pub(crate) fn run(
    sensor: &mut dyn Sensor,
    wait: Wait<'_>,
    fingers: usize,
    touches: usize,
    enroll: usize,
) -> focalprint_sensor::Result<()> {
    let mut frames: Vec<Vec<Image>> = Vec::with_capacity(fingers);
    for finger in 0..fingers {
        let name = finger_name(finger);
        println!(
            "\nfinger {name}: touch {touches} times, lifting in between; vary the spot a little."
        );
        let mut set = Vec::with_capacity(touches);
        for n in 1..=touches {
            print!("  [{n}/{touches}] touch… ");
            flush();
            set.push(sensor.capture(wait)?);
            print!("ok, lift… ");
            flush();
            sensor.wait_release(wait)?;
            println!("done");
        }
        frames.push(set);
    }

    println!("\npolicy: {:?}", Policy::DEFAULT);
    println!(
        "columns: comparisons │ accepted with min_inliers = {}\n",
        INLIER_THRESHOLDS.map(|t| t.to_string()).join(" ")
    );
    for (i, params) in GRID.iter().enumerate() {
        println!("── set {i}: {params:?}");
        evaluate(&frames, params, enroll);
    }
    Ok(())
}

fn evaluate(frames: &[Vec<Image>], params: &Params, enroll: usize) {
    let policy = Policy::DEFAULT;
    let mut rejected = 0usize;
    let samples: Vec<Vec<Sample>> = frames
        .iter()
        .map(|set| {
            set.iter()
                .filter_map(|image| {
                    match Sample::extract_with(image.pixels(), Image::WIDTH, Image::HEIGHT, params)
                    {
                        Ok(sample) => Some(sample),
                        Err(why) => {
                            rejected += 1;
                            println!("   rejected frame: {why}");
                            None
                        }
                    }
                })
                .collect()
        })
        .collect();
    let features: Vec<usize> = samples
        .iter()
        .flatten()
        .map(Sample::feature_count)
        .collect();
    println!(
        "   features per frame: {}..{} (mean {:.0}), {rejected} frames rejected",
        features.iter().min().unwrap_or(&0),
        features.iter().max().unwrap_or(&0),
        mean(&features)
    );

    // Single-sample comparisons.
    let (mut genuine, mut impostor) = (Vec::new(), Vec::new());
    for (fa, a) in samples.iter().enumerate() {
        for (fb, b) in samples.iter().enumerate().skip(fa) {
            for (i, sa) in a.iter().enumerate() {
                for (j, sb) in b.iter().enumerate() {
                    if fa == fb && j <= i {
                        continue;
                    }
                    let e = compare(sa, sb);
                    if fa == fb {
                        genuine.push(e);
                    } else {
                        impostor.push(e);
                    }
                }
            }
        }
    }
    report("pairs  genuine ", &genuine);
    report("pairs  impostor", &impostor);

    // Template (first `enroll` usable touches) against the remaining ones.
    let (mut genuine, mut impostor) = (Vec::new(), Vec::new());
    for (fa, set) in samples.iter().enumerate() {
        let mut enrollment = Enrollment::new(enroll, policy);
        let mut used = 0;
        for sample in set {
            if enrollment.is_complete() {
                break;
            }
            used += 1;
            // Duplicates and unrelated touches are simply not kept.
            let _: Progress = enrollment.add(sample.clone());
        }
        let Some(template) = enrollment.finish() else {
            println!("   finger {}: enrollment incomplete", finger_name(fa));
            continue;
        };
        genuine.extend(set[used..].iter().map(|s| template.verify(s, &policy).best));
        for (fb, other) in samples.iter().enumerate() {
            if fb != fa {
                impostor.extend(other.iter().map(|s| template.verify(s, &policy).best));
            }
        }
    }
    report("templ  genuine ", &genuine);
    report("templ  impostor", &impostor);
}

fn report(label: &str, results: &[Evidence]) {
    if results.is_empty() {
        println!("   {label}: none");
        return;
    }
    let accepted: Vec<String> = INLIER_THRESHOLDS
        .iter()
        .map(|&min_inliers| {
            let policy = Policy {
                min_inliers,
                ..Policy::DEFAULT
            };
            format!(
                "{:>3}",
                results.iter().filter(|e| policy.accepts(e)).count()
            )
        })
        .collect();
    let mut inliers: Vec<usize> = results.iter().map(|e| e.inliers).collect();
    inliers.sort_unstable();
    println!(
        "   {label} {:>3} │ {} │ inliers median {:>2} max {:>2}",
        results.len(),
        accepted.join(" "),
        inliers[inliers.len() / 2],
        inliers[inliers.len() - 1],
    );
}

fn finger_name(i: usize) -> char {
    char::from(b'A' + u8::try_from(i % 26).unwrap_or(0))
}

fn flush() {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
}

#[allow(clippy::cast_precision_loss)] // a handful of small counts
fn mean(values: &[usize]) -> f32 {
    values.iter().sum::<usize>() as f32 / values.len().max(1) as f32
}
