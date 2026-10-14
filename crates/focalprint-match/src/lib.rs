//! Fingerprint feature extraction and matching for small-area sensors
//! (64 × 80 px on the FT9368).
//!
//! A frame becomes a [`Sample`]: keypoints with rotation-normalised binary
//! descriptors plus a coarse ridge-flow map. Pixels are not kept. Several
//! samples of one finger form a [`Template`]; a touch matches when a rigid
//! alignment explains enough well-spread descriptor matches *and* the ridge
//! flow agrees over the overlap.

// Image arithmetic on frames of at most 2040 × 2040 px: index/float casts
// cannot overflow or lose meaningful precision, and x/y/r/s/c are the
// customary names in geometry code.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]

mod features;
mod matcher;
mod plane;
mod prep;
mod sample;
mod template;

pub use features::Params;
pub use matcher::{Evidence, compare};
pub use prep::Rejected;
pub use sample::Sample;
pub use template::{Enrollment, MAX_LEARNED, Policy, Progress, Template, TemplateError, Verdict};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prep::tests::ridges;

    fn sample(dx: f32, dy: f32, rot: f32) -> Sample {
        Sample::extract(&ridges(64, 80, dx, dy, rot), 64, 80).unwrap()
    }

    #[test]
    fn same_pattern_shifted_and_turned_matches() {
        let reference = sample(0.0, 0.0, 0.0);
        let query = sample(4.0, -3.0, 0.15);
        let e = compare(&query, &reference);
        assert!(Policy::DEFAULT.accepts(&e), "{e:?}");
    }

    #[test]
    fn different_pattern_does_not_match() {
        let reference = sample(0.0, 0.0, 0.0);
        let other: Vec<u8> = (0..64 * 80)
            .map(|i| {
                let (x, y) = ((i % 64) as f32, (i / 64) as f32);
                (128.0 + 90.0 * ((x * 0.7 + y * 0.35 + (y * 0.2).sin() * 4.0) * 0.55).sin()) as u8
            })
            .collect();
        if let Ok(query) = Sample::extract(&other, 64, 80) {
            let e = compare(&query, &reference);
            assert!(!Policy::DEFAULT.accepts(&e), "{e:?}");
        }
    }

    #[test]
    fn guided_enrollment_grows_one_mosaic() {
        let mut enrollment = Enrollment::new(3, Policy::DEFAULT);
        assert!(matches!(
            enrollment.add(sample(0.0, 0.0, 0.0)),
            Progress::Accepted { .. }
        ));
        // Same spot again adds nothing.
        assert_eq!(enrollment.add(sample(1.0, 1.0, 0.0)), Progress::Duplicate);
        // Far away, no overlap: starts a new piece of the map.
        assert!(matches!(
            enrollment.add(sample(-70.0, 0.0, 0.0)),
            Progress::Accepted { .. }
        ));
        // Overlapping and new.
        assert!(matches!(
            enrollment.add(sample(18.0, 10.0, 0.0)),
            Progress::Accepted { .. }
        ));
        assert!(matches!(
            enrollment.add(sample(-16.0, 14.0, 0.05)),
            Progress::Accepted { .. }
        ));
        let template = enrollment.finish().unwrap();
        assert_eq!(template.mosaics(), 2, "{template:?}");
        // A touch between the enrolled ones matches the stitched mosaic.
        let verdict = template.verify(&sample(4.0, 12.0, 0.1), &Policy::DEFAULT);
        assert!(verdict.matched, "{verdict:?}");
    }

    #[test]
    fn learning_is_bounded_and_needs_a_match() {
        let mut enrollment = Enrollment::new(1, Policy::DEFAULT);
        enrollment.add(sample(0.0, 0.0, 0.0));
        let mut template = enrollment.finish().unwrap();

        let unrelated = Verdict {
            matched: false,
            best: Evidence::default(),
            placement: None,
        };
        assert!(!template.learn(sample(2.0, 1.0, 0.1), &unrelated));

        let mut learnt = 0;
        for i in 0..MAX_LEARNED + 3 {
            let touch = sample(4.0 + i as f32 * 0.5, -3.0, 0.12);
            let mut verdict = template.verify(&touch, &Policy::DEFAULT);
            assert!(verdict.matched, "{verdict:?}");
            // Pretend every touch shows something new.
            verdict.best.inliers = 12;
            learnt += usize::from(template.learn(touch, &verdict));
        }
        assert_eq!(learnt, MAX_LEARNED + 3);
        assert_eq!(template.learned(), MAX_LEARNED);
        assert_eq!(template.mosaics(), 1);
        let round = Template::from_bytes(&template.to_bytes()).unwrap();
        assert_eq!(round.learned(), MAX_LEARNED);
    }

    #[test]
    fn template_round_trips_and_detects_tampering() {
        let mut enrollment = Enrollment::new(1, Policy::DEFAULT);
        enrollment.add(sample(0.0, 0.0, 0.0));
        let template = enrollment.finish().unwrap();
        let bytes = template.to_bytes();
        assert_eq!(Template::from_bytes(&bytes).unwrap().len(), 1);
        let mut damaged = bytes.to_vec();
        damaged[20] ^= 1;
        assert!(matches!(
            Template::from_bytes(&damaged),
            Err(TemplateError::Integrity)
        ));
    }
}
