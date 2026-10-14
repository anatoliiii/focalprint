//! Enrolled fingers: several samples covering different parts of the finger,
//! the decision policy, and the on-disk format.

use crate::{
    features::{Feature, axis_between},
    matcher::{
        Evidence, Rigid, align_features, cells, compare_rigid, flow_agreement, mean_hamming,
    },
    prep::BLOCK,
    sample::Sample,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::f32::consts::TAU;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

/// When two samples count as the same finger.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    pub min_inliers: usize,
    pub min_cells: usize,
    pub min_overlap: usize,
    pub min_agreement: f32,
    /// Upper bound on [`Evidence::hamming`].
    pub max_hamming: f32,
}

impl Policy {
    /// From `focalprint eval` on the FT9368 (3 fingers, 672 impostor
    /// comparisons incl. mosaics): impostors never exceeded 4 inliers, so 8
    /// keeps a 2× margin while recovering partial-overlap genuine touches.
    /// `max_hamming` is not used yet (equal to the matching limit).
    pub const DEFAULT: Self = Self {
        min_inliers: 8,
        min_cells: 3,
        min_overlap: 10,
        min_agreement: 0.7,
        max_hamming: 64.0,
    };

    #[must_use]
    pub fn accepts(&self, e: &Evidence) -> bool {
        e.inliers >= self.min_inliers
            && e.cells >= self.min_cells
            && e.overlap >= self.min_overlap
            && e.agreement >= self.min_agreement
            && e.hamming <= self.max_hamming
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Result of checking a touch against a template.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    pub matched: bool,
    /// Strongest comparison over the template's mosaics.
    pub best: Evidence,
    /// Which mosaic gave `best`, and its alignment `mosaic → query`.
    pub(crate) placement: Option<(u16, Rigid)>,
}

/// Where a sample sits: transform `sample → mosaic` of its component.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Zeroize)]
struct Pose {
    rotation: f32,
    tx: f32,
    ty: f32,
    component: u16,
}

impl Pose {
    fn new(rigid: Rigid, component: u16) -> Self {
        Self {
            rotation: rigid.rotation,
            tx: rigid.tx,
            ty: rigid.ty,
            component,
        }
    }

    fn rigid(&self) -> Rigid {
        Rigid {
            rotation: self.rotation,
            tx: self.tx,
            ty: self.ty,
        }
    }
}

/// An enrolled finger.
///
/// The sensor sees only a 3 × 4 mm patch per touch, so the touches are
/// stitched into mosaics: every sample has a pose in the coordinate frame of
/// its connected component (samples that overlap, directly or through
/// others). A touch is matched against whole mosaics, collecting evidence
/// from every sample it covers.
#[derive(Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct Template {
    samples: Vec<Sample>,
    poses: Vec<Pose>,
    /// `samples[..enrolled]` came from enrollment and are never replaced;
    /// the rest were learnt from successful verifications.
    enrolled: u16,
}

/// Learnt samples kept on top of the enrolled ones.
pub const MAX_LEARNED: usize = 12;
/// A verified touch sharing at least this many inliers with the template
/// adds nothing new and is not learnt.
const NOVELTY_INLIERS: usize = 30;
/// Mosaic features this close (px, rad, bits) to one already there are the
/// same physical point seen in another touch.
const MERGE_PX: f32 = 2.5;
const MERGE_ANGLE: f32 = 0.35;
const MERGE_HAMMING: u32 = 48;
/// Mosaic features closer than this to a match are treated as the same
/// point by the ratio test.
const SAME_POINT_PX: f32 = 4.0;

/// Why stored bytes are not a template.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("not a focalprint template (or an unsupported version)")]
    Format,
    #[error("template is corrupted")]
    Integrity,
}

/// Version 4: features carry both half-turn descriptors. Earlier versions
/// are not readable; those fingers have to be enrolled again.
const MAGIC: &[u8; 8] = b"FTE4TPL\x04";
const DIGEST: usize = 32;

impl Template {
    fn stitch(samples: Vec<Sample>, enrolled: u16) -> Self {
        let poses = assemble(&samples, &Policy::DEFAULT);
        Self {
            samples,
            poses,
            enrolled,
        }
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Never true for a finished enrollment.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Samples learnt after enrollment.
    #[must_use]
    pub fn learned(&self) -> usize {
        self.samples.len() - usize::from(self.enrolled)
    }

    /// Number of separate mosaics (1 when every touch overlapped another).
    #[must_use]
    pub fn mosaics(&self) -> usize {
        self.components().len()
    }

    fn components(&self) -> Vec<u16> {
        let mut ids: Vec<u16> = self.poses.iter().map(|p| p.component).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Adaptive update after a successful verification: keeps `query` when
    /// it shows the finger from a side the template knows little about,
    /// placing it into the mosaic it matched and evicting the oldest learnt
    /// sample when full. Only ever fed touches that already matched, so an
    /// impostor cannot teach it anything.
    ///
    /// Returns whether the template changed (and should be saved).
    pub fn learn(&mut self, query: Sample, verdict: &Verdict) -> bool {
        let Some((component, mosaic_to_query)) = verdict.placement else {
            return false;
        };
        if !verdict.matched || verdict.best.inliers >= NOVELTY_INLIERS {
            return false;
        }
        if self.learned() >= MAX_LEARNED {
            let oldest = usize::from(self.enrolled);
            self.samples.remove(oldest);
            self.poses.remove(oldest);
        }
        self.samples.push(query);
        self.poses
            .push(Pose::new(mosaic_to_query.inverse(), component));
        true
    }

    /// Features of one mosaic in its own frame, merged where touches
    /// overlap, plus each member sample with its pose.
    fn mosaic(&self, component: u16) -> (Vec<Feature>, Vec<(&Sample, Rigid)>) {
        let members: Vec<(&Sample, Rigid)> = self
            .samples
            .iter()
            .zip(&self.poses)
            .filter(|(_, p)| p.component == component)
            .map(|(s, p)| (s, p.rigid()))
            .collect();
        let mut features: Vec<Feature> = Vec::new();
        for (sample, pose) in &members {
            for f in &sample.features {
                let (x, y) = pose.apply(f.x, f.y);
                let placed = Feature {
                    x,
                    y,
                    angle: (f.angle + pose.rotation).rem_euclid(TAU),
                    descriptor: f.descriptor,
                    flipped: f.flipped,
                };
                let seen = features.iter().any(|g| {
                    (g.x - x).hypot(g.y - y) <= MERGE_PX
                        && axis_between(g.angle, placed.angle) <= MERGE_ANGLE
                        && g.distance(&placed) <= MERGE_HAMMING
                });
                if !seen {
                    features.push(placed);
                }
            }
        }
        (features, members)
    }

    /// Compares `query` with every mosaic of this finger.
    #[must_use]
    pub fn verify(&self, query: &Sample, policy: &Policy) -> Verdict {
        let mut best: Option<(Evidence, u16, Rigid)> = None;
        for component in self.components() {
            let (features, members) = self.mosaic(component);
            let (candidates, aligned) = align_features(&query.features, &features, SAME_POINT_PX);
            let Some((to_query, inliers)) = aligned else {
                continue;
            };
            let placed: Vec<(&Sample, Rigid)> = members
                .iter()
                .map(|(s, pose)| (*s, to_query.after(pose)))
                .collect();
            let (overlap, agreement) = flow_agreement(query, &placed);
            let evidence = Evidence {
                candidates,
                inliers: inliers.len(),
                cells: cells(&query.features, &inliers),
                overlap,
                agreement,
                rotation: to_query.rotation,
                shift: (to_query.tx, to_query.ty),
                hamming: mean_hamming(&query.features, &features, &inliers),
            };
            let better = best.as_ref().is_none_or(|(b, _, _)| {
                (policy.accepts(&evidence), evidence.score()) > (policy.accepts(b), b.score())
            });
            if better {
                best = Some((evidence, component, to_query));
            }
        }
        match best {
            Some((best, component, to_query)) => Verdict {
                matched: policy.accepts(&best),
                best,
                placement: Some((component, to_query)),
            },
            None => Verdict {
                matched: false,
                best: Evidence::default(),
                placement: None,
            },
        }
    }

    /// Serialises as `MAGIC ‖ postcard(template) ‖ SHA-256(MAGIC ‖ body)`.
    ///
    /// # Panics
    ///
    /// Never in practice: serialising plain data into a `Vec` cannot fail.
    #[must_use]
    pub fn to_bytes(&self) -> Zeroizing<Vec<u8>> {
        let mut out = Zeroizing::new(MAGIC.to_vec());
        let body = Zeroizing::new(postcard::to_stdvec(self).expect("serialising to memory"));
        out.extend_from_slice(&body);
        let digest = Sha256::digest(&*out);
        out.extend_from_slice(&digest);
        out
    }

    /// Parses and checks bytes produced by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// [`TemplateError`] for foreign, outdated, damaged or implausible data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TemplateError> {
        if bytes.len() < MAGIC.len() + DIGEST || !bytes.starts_with(MAGIC) {
            return Err(TemplateError::Format);
        }
        let (signed, digest) = bytes.split_at(bytes.len() - DIGEST);
        if Sha256::digest(signed).as_slice() != digest {
            return Err(TemplateError::Integrity);
        }
        let template: Self =
            postcard::from_bytes(&signed[MAGIC.len()..]).map_err(|_| TemplateError::Format)?;
        let plausible = !template.samples.is_empty()
            && template.poses.len() == template.samples.len()
            && usize::from(template.enrolled) <= template.samples.len()
            && template.samples.iter().all(Sample::is_consistent)
            && template
                .poses
                .iter()
                .all(|p| p.rotation.is_finite() && p.tx.is_finite() && p.ty.is_finite());
        if plausible {
            Ok(template)
        } else {
            Err(TemplateError::Format)
        }
    }
}

/// Poses for `samples`: overlapping samples (per `policy`) are chained into
/// connected components, each in the frame of its first sample.
fn assemble(samples: &[Sample], policy: &Policy) -> Vec<Pose> {
    let n = samples.len();
    // edges[a] = (b, transform b → a)
    let mut edges: Vec<Vec<(usize, Rigid)>> = vec![Vec::new(); n];
    for a in 0..n {
        for b in a + 1..n {
            let (evidence, transform) = compare_rigid(&samples[a], &samples[b]);
            if let (true, Some(b_to_a)) = (policy.accepts(&evidence), transform) {
                edges[a].push((b, b_to_a));
                edges[b].push((a, b_to_a.inverse()));
            }
        }
    }
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    let mut component = 0u16;
    for root in 0..n {
        if poses[root].is_some() {
            continue;
        }
        poses[root] = Some(Pose::new(Rigid::IDENTITY, component));
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(a) = queue.pop_front() {
            let a_to_mosaic = poses[a].map_or(Rigid::IDENTITY, |p| p.rigid());
            for &(b, b_to_a) in &edges[a] {
                if poses[b].is_none() {
                    poses[b] = Some(Pose::new(a_to_mosaic.after(&b_to_a), component));
                    queue.push_back(b);
                }
            }
        }
        component += 1;
    }
    poses
        .into_iter()
        .map(|p| p.unwrap_or(Pose::new(Rigid::IDENTITY, u16::MAX)))
        .collect()
}

impl std::fmt::Debug for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Template")
            .field("samples", &self.samples.len())
            .field("enrolled", &self.enrolled)
            .field("mosaics", &self.mosaics())
            .finish_non_exhaustive()
    }
}

/// Outcome of offering one touch to an [`Enrollment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// Kept; `done` of `target` samples collected.
    Accepted { done: usize, target: usize },
    /// Covers (almost) nothing new: move the finger a little further.
    Duplicate,
    /// Does not overlap what is enrolled so far: move the finger back
    /// towards the last spot, so the map grows in one piece.
    Unrelated,
}

/// Side of the cells used to measure coverage, in px.
const COVERAGE_CELL: f32 = 8.0;
/// A new touch must cover at least this fraction of cells not covered yet.
const MIN_NOVELTY: f32 = 0.2;
/// After this many unrelated touches in a row, start a new piece instead of
/// insisting. 1 = never insist: with today's matcher, overlapping touches
/// are often not recognised as such, and insisting made enrollment a chore.
const MAX_UNRELATED_STREAK: usize = 1;

/// Guided enrollment of one finger: touches are stitched into a mosaic as
/// they come, each one has to overlap the map so far (so it grows in one
/// piece, without holes) and add new area to it.
#[derive(Debug)]
pub struct Enrollment {
    template: Template,
    target: usize,
    policy: Policy,
    unrelated_streak: usize,
}

impl Enrollment {
    /// Starts an enrollment that needs `target` accepted touches.
    #[must_use]
    pub fn new(target: usize, policy: Policy) -> Self {
        Self {
            template: Template {
                samples: Vec::new(),
                poses: Vec::new(),
                enrolled: 0,
            },
            target: target.max(1),
            policy,
            unrelated_streak: 0,
        }
    }

    /// Offers a touch.
    pub fn add(&mut self, sample: Sample) -> Progress {
        let pose = if self.template.is_empty() {
            Pose::new(Rigid::IDENTITY, 0)
        } else {
            let verdict = self.template.verify(&sample, &self.policy);
            match verdict.placement {
                Some((component, mosaic_to_query)) if verdict.matched => {
                    self.unrelated_streak = 0;
                    let pose = Pose::new(mosaic_to_query.inverse(), component);
                    if self.novelty(&sample, &pose) < MIN_NOVELTY {
                        return Progress::Duplicate;
                    }
                    pose
                }
                _ => {
                    self.unrelated_streak += 1;
                    if self.unrelated_streak < MAX_UNRELATED_STREAK {
                        return Progress::Unrelated;
                    }
                    self.unrelated_streak = 0;
                    let next = self
                        .template
                        .poses
                        .iter()
                        .map(|p| p.component + 1)
                        .max()
                        .unwrap_or(0);
                    Pose::new(Rigid::IDENTITY, next)
                }
            }
        };
        self.template.samples.push(sample);
        self.template.poses.push(pose);
        Progress::Accepted {
            done: self.template.len(),
            target: self.target,
        }
    }

    /// Fraction of `sample`'s foreground, placed at `pose`, that no
    /// enrolled touch covers yet.
    fn novelty(&self, sample: &Sample, pose: &Pose) -> f32 {
        let covered: std::collections::HashSet<(u16, i32, i32)> = self
            .template
            .samples
            .iter()
            .zip(&self.template.poses)
            .flat_map(|(s, p)| footprint(s, p))
            .collect();
        let mine = footprint(sample, pose);
        if mine.is_empty() {
            return 0.0;
        }
        mine.iter().filter(|cell| !covered.contains(cell)).count() as f32 / mine.len() as f32
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.template.len() >= self.target
    }

    /// The finished template, or `None` if more touches are needed.
    #[must_use]
    pub fn finish(self) -> Option<Template> {
        let enrolled = u16::try_from(self.template.len()).ok()?;
        // Re-stitch from all pairwise overlaps: more robust than the
        // incremental placement, and joins pieces that later touches bridged.
        self.is_complete()
            .then(|| Template::stitch(self.template.samples.clone(), enrolled))
    }
}

/// Coverage cells (in mosaic coordinates) of a sample's foreground blocks.
fn footprint(sample: &Sample, pose: &Pose) -> Vec<(u16, i32, i32)> {
    let to_mosaic = pose.rigid();
    let mut cells = Vec::new();
    for by in 0..sample.blocks_y() {
        for bx in 0..sample.blocks_x() {
            if sample.block(bx, by).is_none() {
                continue;
            }
            // Four cells per 8 px block, so a quarter-block shift counts.
            for (ox, oy) in [(2.0, 2.0), (6.0, 2.0), (2.0, 6.0), (6.0, 6.0)] {
                let (x, y) = to_mosaic.apply((bx * BLOCK) as f32 + ox, (by * BLOCK) as f32 + oy);
                let cell = (
                    (x / COVERAGE_CELL * 2.0).floor() as i32,
                    (y / COVERAGE_CELL * 2.0).floor() as i32,
                );
                cells.push((pose.component, cell.0, cell.1));
            }
        }
    }
    cells.sort_unstable();
    cells.dedup();
    cells
}
