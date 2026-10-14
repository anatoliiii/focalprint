//! The sensor thread. It owns the sensor and the neural matcher and runs
//! one job at a time; frames never leave this thread except as encrypted
//! templates (enrolment) or encrypted research records.

use crate::store::{Frames, Template};
use focalprint_dataset::{Collection, Kind, Record};
use focalprint_net::{
    FpNet,
    matching::{self, Features},
};
use focalprint_sensor::{Backend, Error as DeviceError, Image, Wait};
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash as _, Hasher as _},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc::UnboundedSender;

/// How long a single blocking wait may last before re-checking for
/// cancellation of the whole job.
const SLICE: Duration = Duration::from_secs(2);
/// Touches with fewer usable keypoints are too small/faint to judge.
const MIN_KEYPOINTS: usize = 80;
/// Adaptive template: at most this many verification touches per finger,
/// on top of the enrolment ones.
const MAX_ADAPTIVE: usize = 20;
/// A touch this similar to one already in the template adds no new skin.
const DUPLICATE_SCORE: f32 = 0.8;

pub(crate) enum Job {
    /// Wait for a finger and compare it with the user's templates.
    Verify {
        user: String,
        candidates: Vec<(&'static str, Arc<Template>)>,
    },
    /// Collect `stages` touches of `finger`.
    Enroll {
        user: String,
        finger: &'static str,
        stages: usize,
    },
    /// Precompute features of templates (daemon start, after enrolment).
    Warm { templates: Vec<Arc<Template>> },
}

/// Progress reported back to the D-Bus side.
#[derive(Debug)]
pub(crate) enum Event {
    FingerNeeded(bool),
    FingerPresent(bool),
    /// fprintd result string and whether the action is over.
    Verify(&'static str, bool),
    Enroll(&'static str, bool),
    /// Enrolment finished; the D-Bus side stores the frames (encrypted),
    /// then reports `enroll-completed`.
    Enrolled(Frames),
    /// A verified touch joined `finger`'s template; the D-Bus side stores
    /// it if the template on disk is still the one with id `base`.
    Adapted {
        finger: &'static str,
        base: u64,
        template: Template,
    },
}

struct Request {
    job: Job,
    cancel: Arc<AtomicBool>,
    events: UnboundedSender<Event>,
}

/// Handle to the sensor thread.
#[derive(Debug, Clone)]
pub(crate) struct Sensor {
    requests: mpsc::Sender<Request>,
}

/// Matcher settings.
pub(crate) struct Matcher {
    pub(crate) net: FpNet,
    pub(crate) threshold: f32,
    /// Matches scoring at least this much join the template.
    pub(crate) adapt_score: f32,
}

impl Sensor {
    /// Opens the device and starts the thread. With a `collection`, every
    /// touch is also stored there, encrypted, for algorithm research.
    pub(crate) fn start(
        backend: Backend,
        path: &Path,
        matcher: Matcher,
        collection: Option<Collection>,
    ) -> Result<Self, DeviceError> {
        let mut sensor = focalprint_sensor::open(backend, path)?;
        tracing::info!(sensor = sensor.describe(), "sensor ready");
        let (requests, queue) = mpsc::channel::<Request>();
        thread::Builder::new()
            .name("sensor".into())
            .spawn(move || {
                let mut cache: HashMap<u64, Vec<Features>> = HashMap::new();
                for request in queue {
                    let mut worker = Worker {
                        sensor: sensor.as_mut(),
                        cancel: &request.cancel,
                        events: &request.events,
                        collection: collection.as_ref(),
                        matcher: &matcher,
                        cache: &mut cache,
                    };
                    worker.run(request.job);
                }
            })
            .map_err(|_| DeviceError::Protocol("cannot start sensor thread"))?;
        Ok(Self { requests })
    }

    /// Queues a job; returns its cancel flag.
    pub(crate) fn submit(&self, job: Job, events: UnboundedSender<Event>) -> Arc<AtomicBool> {
        let cancel = Arc::new(AtomicBool::new(false));
        let request = Request {
            job,
            cancel: cancel.clone(),
            events,
        };
        if self.requests.send(request).is_err() {
            tracing::error!("sensor thread is gone");
        }
        cancel
    }
}

pub(crate) fn frames_id(frames: &Frames) -> u64 {
    let mut h = DefaultHasher::new();
    for f in frames {
        f.as_slice().hash(&mut h);
    }
    h.finish()
}

/// Ends a job early: cancelled by the client or the device failed.
enum Stop {
    Cancelled,
    Failed(String),
}

impl From<DeviceError> for Stop {
    fn from(e: DeviceError) -> Self {
        match e {
            DeviceError::Cancelled => Self::Cancelled,
            e => Self::Failed(e.to_string()),
        }
    }
}

struct Worker<'a> {
    sensor: &'a mut dyn focalprint_sensor::Sensor,
    cancel: &'a AtomicBool,
    events: &'a UnboundedSender<Event>,
    collection: Option<&'a Collection>,
    matcher: &'a Matcher,
    cache: &'a mut HashMap<u64, Vec<Features>>,
}

impl Worker<'_> {
    fn send(&self, event: Event) {
        let _ = self.events.send(event);
    }

    fn run(&mut self, job: Job) {
        let (outcome, failure) = match job {
            Job::Verify { user, candidates } => (
                self.verify(&user, &candidates),
                Some(Event::Verify("verify-unknown-error", true)),
            ),
            Job::Enroll {
                user,
                finger,
                stages,
            } => (
                self.enroll(&user, finger, stages),
                Some(Event::Enroll("enroll-unknown-error", true)),
            ),
            Job::Warm { templates } => {
                for t in &templates {
                    self.features_of(t);
                }
                (Ok(()), None)
            }
        };
        if let (Err(Stop::Failed(err)), Some(failure)) = (outcome, failure) {
            tracing::error!(%err, "sensor job failed");
            self.send(failure);
        }
        self.send(Event::FingerNeeded(false));
    }

    fn features(&self, pixels: &[u8]) -> Result<Features, Stop> {
        let map = self
            .matcher
            .net
            .run(pixels)
            .map_err(|e| Stop::Failed(e.to_string()))?;
        Ok(Features::new(pixels, map))
    }

    /// Template features, computed once per distinct template.
    fn features_of(&mut self, template: &Template) -> u64 {
        let id = frames_id(&template.frames);
        if !self.cache.contains_key(&id) {
            let t = Instant::now();
            let feats: Vec<Features> = template
                .frames
                .iter()
                .filter_map(|f| self.features(f).ok())
                .collect();
            tracing::info!(
                touches = feats.len(),
                ms = t.elapsed().as_millis(),
                "template features ready"
            );
            self.cache.insert(id, feats);
        }
        id
    }

    /// Stores the touch in the research collection, if enabled.
    fn collect(
        &self,
        image: &Image,
        kind: Kind,
        user: &str,
        finger: Option<&str>,
        outcome: &str,
        score: f32,
    ) {
        let Some(collection) = self.collection else {
            return;
        };
        let record = Record {
            unix_ms: focalprint_dataset::now_ms(),
            kind,
            user: user.into(),
            finger: finger.map(Into::into),
            outcome: outcome.into(),
            // score × 1000 (the field predates neural scores); 0..=1000 fits
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            inliers: (score.clamp(0.0, 1.0) * 1000.0).round() as u16,
            width: u16::try_from(Image::WIDTH).unwrap_or(0),
            height: u16::try_from(Image::HEIGHT).unwrap_or(0),
            pixels: image.pixels().to_vec(),
        };
        if let Err(err) = collection.add(&record) {
            tracing::warn!(%err, "cannot store dataset record");
        }
    }

    /// Blocks until a finger is down and returns its frame. A finger left on
    /// the sensor from before has to be lifted first.
    fn touch(&mut self) -> Result<Image, Stop> {
        self.lift()?;
        self.send(Event::FingerNeeded(true));
        let image = loop {
            match self.sensor.capture(Wait {
                timeout: SLICE,
                cancel: self.cancel,
            }) {
                Err(DeviceError::Timeout) if !self.cancel.load(Ordering::Relaxed) => {}
                other => break other?,
            }
        };
        self.send(Event::FingerNeeded(false));
        self.send(Event::FingerPresent(true));
        Ok(image)
    }

    fn lift(&mut self) -> Result<(), Stop> {
        loop {
            match self.sensor.wait_release(Wait {
                timeout: SLICE,
                cancel: self.cancel,
            }) {
                Err(DeviceError::Timeout) if !self.cancel.load(Ordering::Relaxed) => {}
                other => break other?,
            }
        }
        self.send(Event::FingerPresent(false));
        Ok(())
    }

    fn verify(
        &mut self,
        user: &str,
        candidates: &[(&'static str, Arc<Template>)],
    ) -> Result<(), Stop> {
        let ids: Vec<(&'static str, u64, &Template)> = candidates
            .iter()
            .map(|(f, t)| (*f, self.features_of(t), &**t))
            .collect();
        loop {
            let image = self.touch()?;
            let started = Instant::now();
            let query = self.features(image.pixels())?;
            if query.keypoints() < MIN_KEYPOINTS {
                self.collect(&image, Kind::Verify, user, None, "verify-retry-scan", 0.0);
                self.send(Event::Verify("verify-retry-scan", false));
                continue;
            }
            let mut best: (f32, Option<(&'static str, u64, &Template)>) = (0.0, None);
            for &(finger, id, template) in &ids {
                let (ok, score) =
                    matching::verify(&query, &self.cache[&id], self.matcher.threshold);
                if ok {
                    best = (score, Some((finger, id, template)));
                    break;
                }
                best.0 = best.0.max(score);
            }
            let ms = started.elapsed().as_millis();
            match best {
                (score, Some((finger, id, template))) => {
                    tracing::info!(
                        finger,
                        score = format_args!("{score:.3}"),
                        ms,
                        "verify: match"
                    );
                    self.collect(
                        &image,
                        Kind::Verify,
                        user,
                        Some(finger),
                        "verify-match",
                        score,
                    );
                    self.send(Event::Verify("verify-match", true));
                    // after the reply: unlocking does not wait for this
                    self.adapt(finger, template, id, query, image.pixels());
                }
                (score, None) => {
                    tracing::info!(score = format_args!("{score:.3}"), ms, "verify: no match");
                    self.collect(&image, Kind::Verify, user, None, "verify-no-match", score);
                    self.send(Event::Verify("verify-no-match", true));
                }
            }
            return self.lift();
        }
    }

    /// Adaptive template: a confidently matched touch joins the finger's
    /// template, so it grows to cover the whole pad instead of only the
    /// skin seen at enrolment. Enrolment touches are never replaced, and
    /// only confident matches are added, so a lucky impostor touch cannot
    /// slowly take a template over. When full, the new touch replaces the
    /// adaptive touch most similar to it.
    fn adapt(
        &mut self,
        finger: &'static str,
        template: &Template,
        id: u64,
        query: Features,
        pixels: &[u8],
    ) {
        let Some(feats) = self.cache.get(&id) else {
            return;
        };
        // a touch whose features failed would shift the indices
        if feats.len() != template.frames.len() {
            return;
        }
        let scores = matching::scores(&query, feats);
        let best = scores.iter().copied().fold(0.0, f32::max);
        if best < self.matcher.adapt_score || best >= DUPLICATE_SCORE {
            return;
        }
        let mut next = template.clone();
        let frame = zeroize::Zeroizing::new(pixels.to_vec());
        let slot = if next.frames.len() - next.enrolled < MAX_ADAPTIVE {
            None
        } else {
            (next.enrolled..next.frames.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        };
        let Some(mut feats) = self.cache.remove(&id) else {
            return;
        };
        if let Some(i) = slot {
            next.frames[i] = frame;
            feats[i] = query;
        } else {
            next.frames.push(frame);
            feats.push(query);
        }
        self.cache.insert(frames_id(&next.frames), feats);
        tracing::info!(
            finger,
            score = format_args!("{best:.3}"),
            touches = next.frames.len(),
            replaced = slot.is_some(),
            "template adapted"
        );
        self.send(Event::Adapted {
            finger,
            base: id,
            template: next,
        });
    }

    fn enroll(&mut self, user: &str, finger: &'static str, stages: usize) -> Result<(), Stop> {
        let mut frames: Frames = Vec::with_capacity(stages);
        while frames.len() < stages {
            let image = self.touch()?;
            let ok = self.features(image.pixels())?.keypoints() >= MIN_KEYPOINTS;
            let outcome = if ok {
                "enroll-stage-passed"
            } else {
                "enroll-retry-scan"
            };
            self.collect(&image, Kind::Enroll, user, Some(finger), outcome, 0.0);
            if ok {
                frames.push(zeroize::Zeroizing::new(image.pixels().to_vec()));
            }
            self.send(Event::Enroll(outcome, false));
        }
        self.send(Event::Enrolled(frames));
        self.lift()
    }
}
