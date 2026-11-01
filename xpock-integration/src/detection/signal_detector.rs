//! Core topological signal detection engine.
//!
//! Distinguishes valid topological signals (skyrmion number modulation) from
//! noise, interference, false positives, and corrupted measurements using the
//! complete detection pipeline: Stokes reconstruction, intensity masking,
//! skyrmion number computation, classification, and constellation matching.

use crate::detection::DetectionConfig;
use crate::errors::XpockResult;
use crate::types::{Constellation, DetectionResult, TopologicalSignal};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Detection outcome — detailed result of a single detection attempt
// ---------------------------------------------------------------------------

/// Detailed outcome of a topological signal detection attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionOutcome {
    /// A valid topological signal was detected.
    Signal {
        signal: TopologicalSignal,
        skyrmion_number_computed: f64,
        constellation_point: i32,
        confidence: f64,
    },
    /// Noise detected — no topological structure found.
    Noise {
        noise_floor: f64,
        max_skyrmion_density: f64,
    },
    /// Interference detected — signal present but corrupted beyond recovery.
    Interference {
        interference_type: String,
        severity: f64,
    },
    /// False positive — apparent topology that failed validation.
    FalsePositive {
        q_candidate: i32,
        reason: String,
    },
    /// Corrupted measurement — signal detected but integrity check failed.
    Corrupted {
        reason: String,
        partial_q: Option<i32>,
    },
    /// Insufficient data — need more measurements to decide.
    InsufficientData {
        missing: String,
    },
}

impl DetectionOutcome {
    /// Convert to a [`DetectionResult`] and optional [`TopologicalSignal`].
    pub fn into_result(self) -> (DetectionResult, Option<TopologicalSignal>) {
        match self {
            DetectionOutcome::Signal {
                signal,
                constellation_point: _,
                confidence: _,
                skyrmion_number_computed: _,
            } => (DetectionResult::Signal(signal.clone()), Some(signal)),
            DetectionOutcome::Noise { .. } => (DetectionResult::Noise, None),
            DetectionOutcome::Interference { .. } => (DetectionResult::Interference, None),
            DetectionOutcome::FalsePositive { .. } => (DetectionResult::FalsePositive, None),
            DetectionOutcome::Corrupted { .. } => (DetectionResult::Corrupted, None),
            DetectionOutcome::InsufficientData { .. } => (DetectionResult::InsufficientData, None),
        }
    }

    /// Returns true if this is a valid signal outcome.
    pub fn is_signal(&self) -> bool {
        matches!(self, DetectionOutcome::Signal { .. })
    }

    /// Returns true if this outcome is recoverable (not a hard failure).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            DetectionOutcome::Noise { .. }
                | DetectionOutcome::Interference { .. }
                | DetectionOutcome::InsufficientData { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Signal detector — main detection engine
// ---------------------------------------------------------------------------

/// The topological signal detection engine.
///
/// Implements the complete pipeline for distinguishing valid topological
//! signals from noise, interference, false positives, and corrupted
/// measurements. The skyrmion number — a topological invariant — serves
/// as the robust information carrier.
pub struct SignalDetector {
    config: DetectionConfig,
    statistics: DetectionStatistics,
    constellation: Constellation,
    /// History of recent detections for pattern analysis.
    recent_outcomes: Vec<DetectionOutcome>,
    max_history: usize,
}

impl SignalDetector {
    /// Create a new signal detector with the given configuration.
    pub fn new(config: DetectionConfig) -> Self {
        let constellation = config.constellation.clone();
        Self {
            config: config.clone(),
            statistics: DetectionStatistics::default(),
            constellation,
            recent_outcomes: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Create a detector with a default configuration (binary constellation).
    pub fn default() -> Self {
        Self::new(DetectionConfig::default())
    }

    /// Run the complete detection pipeline on Stokes measurements.
    ///
    /// `stokes_field` is a 3D array [S0, S1, S2, S3][y][x] representing the
    /// reconstructed Stokes vector field. Returns a [`DetectionOutcome`].
    pub fn detect(&mut self, stokes_field: &[[[f64; 4]; 256]; 256]) -> XpockResult<DetectionOutcome> {
        let start = Instant::now();

        // Step 1: Intensity-based masking — suppress unreliable low-intensity regions.
        let mask = IntensityMask::new(self.config.intensity_threshold)
            .apply(stokes_field);

        // Step 2: Check if we have enough valid pixels after masking.
        let valid_pixel_ratio = mask.valid_pixel_ratio();
        if valid_pixel_ratio < 0.1 {
            let outcome = DetectionOutcome::InsufficientData {
                missing: format!("only {:.1}% valid pixels after masking", valid_pixel_ratio * 100.0),
            };
            self.record_outcome(&outcome, start.elapsed().as_micros() as f64);
            return Ok(outcome);
        }

        // Step 3: Compute skyrmion number from the masked field.
        let skyrmion_computed = compute_skyrmion_number_masked(stokes_field, &mask);

        // Step 4: Classify the detection.
        let outcome = self.classify(skyrmion_computed, &mask, stokes_field);

        // Step 5: Record statistics and history.
        let compute_time_us = start.elapsed().as_micros() as f64;
        self.record_outcome(&outcome, compute_time_us);
        self.add_to_history(outcome.clone());

        Ok(outcome)
    }

    /// Run detection on a lower-resolution Stokes field with custom dimensions.
    ///
    /// `data` is a flat buffer of [S0, S1, S2, S3] values in row-major order,
    /// with `width` and `height` defining the field dimensions.
    pub fn detect_raw(
        &mut self,
        data: &[f64],
        width: usize,
        height: usize,
    ) -> XpockResult<DetectionOutcome> {
        if data.len() < width * height * 4 {
            return Err(crate::errors::XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: data.len() / 4,
                    need: width * height,
                },
            ));
        }

        if width == 0 || height == 0 {
            return Err(crate::errors::XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        // Build the 2D Stokes field from flat data.
        let mut stokes_field = vec![vec![[0.0f64; 4]; 256]; 256];
        let scale_x = 256f64 / width as f64;
        let scale_y = 256f64 / height as f64;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let sx = (x as f64 * scale_x) as usize;
                let sy = (y as f64 * scale_y) as usize;
                if sx < 256 && sy < 256 {
                    stokes_field[sy][sx] = [
                        data[idx],
                        data[idx + 1],
                        data[idx + 2],
                        data[idx + 3],
                    ];
                }
            }
        }

        let field_array: [[[f64; 4]; 256]; 256] = stokes_field
            .try_into()
            .map_err(|_| {
                crate::errors::XpockError::Detection(
                    crate::errors::DetectionError::InsufficientStokesMeasurements {
                        have: data.len() / 4,
                        need: 65536,
                    },
                )
            })?;

        self.detect(&field_array)
    }

    /// Classify a computed skyrmion number into a detection outcome.
    fn classify(
        &self,
        skyrmion_computed: f64,
        mask: &IntensityMask,
        _stokes_field: &[[[f64; 4]; 256]; 256],
    ) -> DetectionOutcome {
        // Check intensity masking coverage — if too much is masked, data is unreliable.
        if mask.valid_pixel_ratio() < self.config.intensity_threshold {
            return DetectionOutcome::InsufficientData {
                missing: "insufficient unmasked intensity region".into(),
            };
        }

        // Round to nearest integer for constellation matching.
        let q_rounded = skyrmion_computed.round() as i32;

        // Check if the rounded value is a valid constellation point.
        if self.constellation.contains(q_rounded) {
            // Compute confidence based on how close the computed value is to the integer.
            let distance_to_integer = (skyrmion_computed - q_rounded as f64).abs();
            let confidence = (1.0 - distance_to_integer).clamp(0.0, 1.0);

            if confidence < self.config.min_confidence {
                return DetectionOutcome::FalsePositive {
                    q_candidate: q_rounded,
                    reason: format!("confidence {:.3} below minimum {:.3}", confidence, self.config.min_confidence),
                };
            }

            let signal = TopologicalSignal::new(q_rounded)
                .with_confidence(confidence)
                .with_time(now_ms());

            return DetectionOutcome::Signal {
                signal,
                skyrmion_number_computed,
                constellation_point: q_rounded,
                confidence,
            };
        }

        // Not a valid constellation point — check if it's close to one.
        if let Some(nearest) = self.constellation.nearest(q_rounded) {
            let distance = (q_rounded - nearest).abs();
            if distance <= 1 {
                // Close to a valid point — could be noise-corrupted signal.
                return DetectionOutcome::FalsePositive {
                    q_candidate: q_rounded,
                    reason: format!("not a valid constellation point, nearest is {}", nearest),
                };
            }
        }

        // Check if the skyrmion number is near zero (likely noise).
        if q_rounded.abs() <= 1 && skyrmion_computed.abs() < 0.5 {
            return DetectionOutcome::Noise {
                noise_floor: mask.mean_intensity(),
                max_skyrmion_density: mask.max_skyrmion_density(),
            };
        }

        // Significant non-zero skyrmion number but not in constellation — interference.
        DetectionOutcome::Interference {
            interference_type: "unrecognized_topology".into(),
            severity: skyrmion_computed.abs(),
        }
    }

    /// Get a reference to the detection statistics.
    pub fn statistics(&self) -> &DetectionStatistics {
        &self.statistics
    }

    /// Get a clone of the constellation.
    pub fn constellation(&self) -> &Constellation {
        &self.constellation
    }

    /// Get recent detection outcomes for pattern analysis.
    pub fn recent_outcomes(&self) -> &[DetectionOutcome] {
        &self.recent_outcomes
    }

    /// Update the detection configuration.
    pub fn update_config(&mut self, config: DetectionConfig) {
        self.config = config;
        self.constellation = config.constellation;
    }

    /// Reset statistics and history.
    pub fn reset(&mut self) {
        self.statistics = DetectionStatistics::default();
        self.recent_outcomes.clear();
    }

    fn record_outcome(&mut self, outcome: &DetectionOutcome, compute_time_us: f64) {
        self.statistics.record_outcome(outcome, compute_time_us);
    }

    fn add_to_history(&mut self, outcome: DetectionOutcome) {
        if self.recent_outcomes.len() >= self.max_history {
            self.recent_outcomes.remove(0);
        }
        self.recent_outcomes.push(outcome);
    }
}

/// Returns the current monotonic time in milliseconds (carrier-agnostic).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Skyrmion number computation on masked field
// ---------------------------------------------------------------------------

/// Compute the skyrmion number from a masked Stokes field.
///
/// Uses finite differences on the normalized Stokes vector field n = (S1,S2,S3)/S0,
/// integrated over the unmasked region:
/// Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy
fn compute_skyrmion_number_masked(
    stokes_field: &[[[f64; 4]; 256]; 256],
    mask: &IntensityMask,
) -> f64 {
    let mut q_sum = 0.0f64;
    let mut valid_cells = 0u64;

    // Iterate over interior pixels (avoid boundary for finite differences).
    for y in 1..255 {
        for x in 1..255 {
            if !mask.is_valid(x, y) {
                continue;
            }

            // Get normalized Stokes vectors at this pixel and neighbors.
            let n_center = normalized_stokes(stokes_field[y][x]);
            let n_right = normalized_stokes(stokes_field[y][x + 1]);
            let n_down = normalized_stokes(stokes_field[y + 1][x]);

            // Skip if any neighbor is invalid or zero intensity.
            if n_center.is_none() || n_right.is_none() || n_down.is_none() {
                continue;
            }

            let n_c = n_center.unwrap();
            let n_r = n_right.unwrap();
            let n_d = n_down.unwrap();

            // Finite difference derivatives of n.
            let dn_dx = [
                n_r[0] - n_c[0],
                n_r[1] - n_c[1],
                n_r[2] - n_c[2],
            ];
            let dn_dy = [
                n_d[0] - n_c[0],
                n_d[1] - n_c[1],
                n_d[2] - n_c[2],
            ];

            // Cross product ∂x n × ∂y n.
            let cross = [
                dn_dx[1] * dn_dy[2] - dn_dx[2] * dn_dy[1],
                dn_dx[2] * dn_dy[0] - dn_dx[0] * dn_dy[2],
                dn_dx[0] * dn_dy[1] - dn_dx[1] * dn_dy[0],
            ];

            // Dot product n · (∂x n × ∂y n).
            let dot = n_c[0] * cross[0] + n_c[1] * cross[1] + n_c[2] * cross[2];

            q_sum += dot;
            valid_cells += 1;
        }
    }

    if valid_cells == 0 {
        return 0.0;
    }

    // The discrete approximation: Q ≈ (1/4π) * sum of dot products.
    // Each cell represents dx*dy area; assuming unit spacing.
    q_sum / (4.0 * std::f64::consts::PI)
}

/// Normalize a 4-component Stokes vector to a unit 3-vector n = (S1,S2,S3)/S0.
///
/// Returns None if S0 is zero or near-zero (invalid polarization state).
fn normalized_stokes(s: [f64; 4]) -> Option<[f64; 4]> {
    let s0 = s[0];
    if s0.abs() < 1e-10 {
        return None;
    }
    let n = [s[1] / s0, s[2] / s0, s[3] / s0, 0.0];
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if norm < 1e-10 {
        return None;
    }
    Some([n[0] / norm, n[1] / norm, n[2] / norm, 0.0])
}

// ---------------------------------------------------------------------------
// Multi-frame signal tracking
// ---------------------------------------------------------------------------

/// A tracked topological signal across multiple frames.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalTrack {
    /// Unique track identifier.
    pub track_id: u64,
    /// Skyrmion number of the tracked signal.
    pub skyrmion_number: i32,
    /// Number of consecutive frames the signal has been detected.
    pub consecutive_detections: u32,
    /// Total number of frames where signal was detected.
    pub total_detections: u32,
    /// Average confidence across all detections.
    pub average_confidence: f64,
    /// Last detection timestamp.
    pub last_detection_time: u64,
    /// First detection timestamp.
    pub first_detection_time: u64,
    /// Position history (x, y centroid per frame).
    pub position_history: Vec<(f64, f64)>,
    /// Confidence history per frame.
    pub confidence_history: Vec<f64>,
    /// Whether the track is currently active.
    pub active: bool,
    /// Frames since last detection.
    pub frames_since_last_detection: u32,
}

impl SignalTrack {
    /// Create a new signal track.
    pub fn new(track_id: u64, skyrmion_number: i32, confidence: f64, time: u64) -> Self {
        Self {
            track_id,
            skyrmion_number,
            consecutive_detections: 1,
            total_detections: 1,
            average_confidence: confidence,
            last_detection_time: time,
            first_detection_time: time,
            position_history: Vec::new(),
            confidence_history: vec![confidence],
            active: true,
            frames_since_last_detection: 0,
        }
    }

    /// Update the track with a new detection.
    pub fn update(&mut self, confidence: f64, time: u64, position: Option<(f64, f64)>) {
        self.consecutive_detections += 1;
        self.total_detections += 1;
        self.average_confidence =
            (self.average_confidence * (self.total_detections - 1) as f64 + confidence)
                / self.total_detections as f64;
        self.last_detection_time = time;
        self.frames_since_last_detection = 0;
        self.active = true;
        self.confidence_history.push(confidence);
        if let Some(pos) = position {
            self.position_history.push(pos);
        }
    }

    /// Mark the track as missed for one frame.
    pub fn miss_frame(&mut self) {
        self.consecutive_detections = 0;
        self.frames_since_last_detection += 1;
        if self.frames_since_last_detection > 10 {
            self.active = false;
        }
    }

    /// Returns the track's stability score (0.0 to 1.0).
    pub fn stability_score(&self) -> f64 {
        if self.total_detections == 0 {
            return 0.0;
        }
        let consecutive_ratio = self.consecutive_detections as f64 / self.total_detections as f64;
        let confidence_factor = self.average_confidence.clamp(0.0, 1.0);
        (consecutive_ratio * 0.6 + confidence_factor * 0.4).clamp(0.0, 1.0)
    }

    /// Returns the position variance if position history is available.
    pub fn position_variance(&self) -> Option<f64> {
        if self.position_history.len() < 2 {
            return None;
        }
        let mean_x = self
            .position_history
            .iter()
            .map(|p| p.0)
            .sum::<f64>()
            / self.position_history.len() as f64;
        let mean_y = self
            .position_history
            .iter()
            .map(|p| p.1)
            .sum::<f64>()
            / self.position_history.len() as f64;
        let var_x = self
            .position_history
            .iter()
            .map(|p| (p.0 - mean_x).powi(2))
            .sum::<f64>()
            / self.position_history.len() as f64;
        let var_y = self
            .position_history
            .iter()
            .map(|p| (p.1 - mean_y).powi(2))
            .sum::<f64>()
            / self.position_history.len() as f64;
        Some(var_x + var_y)
    }
}

/// Tracks topological signals across multiple frames.
pub struct MultiFrameTracker {
    /// Active tracks keyed by track ID.
    tracks: std::collections::HashMap<u64, SignalTrack>,
    /// Next track ID to assign.
    next_track_id: u64,
    /// Maximum frames a track can be missed before deactivation.
    pub max_missed_frames: u32,
    /// Minimum consecutive detections for a track to be considered stable.
    pub min_consecutive_for_stable: u32,
    /// Maximum position change between frames for track association (pixels).
    pub max_position_change: f64,
}

impl MultiFrameTracker {
    /// Create a new multi-frame tracker.
    pub fn new() -> Self {
        Self {
            tracks: std::collections::HashMap::new(),
            next_track_id: 1,
            max_missed_frames: 10,
            min_consecutive_for_stable: 3,
            max_position_change: 50.0,
        }
    }

    /// Process a new frame of detections and update tracks.
    ///
    /// `detections` is a list of (skyrmion_number, confidence, time, position) tuples.
    /// Returns the list of active stable tracks.
    pub fn update(
        &mut self,
        detections: Vec<(i32, f64, u64, Option<(f64, f64)>)>,
    ) -> Vec<SignalTrack> {
        // Mark all existing tracks as potentially missed.
        for track in self.tracks.values_mut() {
            track.miss_frame();
        }

        // Match detections to existing tracks.
        let mut matched_track_ids = std::collections::HashSet::new();

        for (q, confidence, time, position) in detections {
            let best_track_id = self.find_best_match(q, confidence, position, &matched_track_ids);

            if let Some(track_id) = best_track_id {
                if let Some(track) = self.tracks.get_mut(&track_id) {
                    track.update(confidence, time, position);
                    matched_track_ids.insert(track_id);
                }
            } else {
                // Create a new track.
                let track_id = self.next_track_id;
                self.next_track_id += 1;
                let mut track = SignalTrack::new(track_id, q, confidence, time);
                if let Some(pos) = position {
                    track.position_history.push(pos);
                }
                self.tracks.insert(track_id, track);
                matched_track_ids.insert(track_id);
            }
        }

        // Remove inactive tracks.
        self.tracks.retain(|_, track| track.active);

        // Return stable tracks.
        self.tracks
            .values()
            .filter(|t| t.consecutive_detections >= self.min_consecutive_for_stable)
            .cloned()
            .collect()
    }

    /// Find the best existing track to match a detection.
    fn find_best_match(
        &self,
        q: i32,
        confidence: f64,
        position: Option<(f64, f64)>,
        excluded: &std::collections::HashSet<u64>,
    ) -> Option<u64> {
        let mut best_id: Option<u64> = None;
        let mut best_score: f64 = f64::INFINITY;

        for (id, track) in &self.tracks {
            if excluded.contains(id) {
                continue;
            }
            if track.skyrmion_number != q {
                continue;
            }

            let mut score = 0.0f64;

            // Position distance penalty.
            if let (Some(pos), Some(track_pos)) = (position, track.position_history.last()) {
                let dx = pos.0 - track_pos.0;
                let dy = pos.1 - track_pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > self.max_position_change {
                    continue; // Too far to match.
                }
                score += dist / self.max_position_change;
            } else if position.is_some() && track.position_history.is_empty() {
                score += 1.0; // No position history to match against.
            }

            // Time gap penalty.
            let time_gap = confidence; // Use confidence as proxy for recency.
            score += time_gap * 0.1;

            if score < best_score {
                best_score = score;
                best_id = Some(*id);
            }
        }

        best_id
    }

    /// Returns the number of active tracks.
    pub fn active_track_count(&self) -> usize {
        self.tracks.values().filter(|t| t.active).count()
    }

    /// Returns all active tracks.
    pub fn active_tracks(&self) -> Vec<SignalTrack> {
        self.tracks.values().filter(|t| t.active).cloned().collect()
    }

    /// Returns stable tracks (meeting minimum consecutive detections).
    pub fn stable_tracks(&self) -> Vec<SignalTrack> {
        self.tracks
            .values()
            .filter(|t| t.consecutive_detections >= self.min_consecutive_for_stable)
            .cloned()
            .collect()
    }

    /// Reset all tracks.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_track_id = 1;
    }

    /// Returns a reference to a track by ID.
    pub fn get_track(&self, track_id: u64) -> Option<&SignalTrack> {
        self.tracks.get(&track_id)
    }
}

impl Default for MultiFrameTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Spatial coherence analysis
// ---------------------------------------------------------------------------

/// Measures spatial coherence of a skyrmion field.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialCoherenceResult {
    /// Overall coherence score (0.0 = incoherent, 1.0 = fully coherent).
    pub coherence_score: f64,
    /// Number of coherent regions detected.
    pub coherent_region_count: usize,
    /// Average region size in pixels.
    pub average_region_size: f64,
    /// Largest coherent region size.
    pub largest_region_size: usize,
    /// Coherence length (characteristic scale of coherence decay).
    pub coherence_length: f64,
    /// Spatial gradient magnitude map statistics.
    pub gradient_mean: f64,
    pub gradient_std: f64,
}

/// Analyzes spatial coherence of Stokes fields for signal validation.
pub struct SpatialCoherenceAnalyzer {
    /// Window size for local coherence computation.
    pub window_size: usize,
    /// Minimum coherence threshold for a region to be considered coherent.
    pub coherence_threshold: f64,
}

impl Default for SpatialCoherenceAnalyzer {
    fn default() -> Self {
        Self {
            window_size: 16,
            coherence_threshold: 0.7,
        }
    }
}

impl SpatialCoherenceAnalyzer {
    /// Create a new spatial coherence analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze spatial coherence of a normalized Stokes field.
    ///
    /// `field` is [n_x, n_y, n_z, 0][y][x] where each pixel has a unit 3-vector.
    pub fn analyze(&self, field: &[[[f64; 4]; 256]; 256]) -> SpatialCoherenceResult {
        let mut gradient_magnitudes = Vec::new();
        let mut local_coherences = Vec::new();

        // Compute gradient magnitudes and local coherence.
        for y in 1..255 {
            for x in 1..255 {
                let n_center = match normalized_stokes(field[y][x]) {
                    Some(n) => n,
                    None => continue,
                };
                let n_right = match normalized_stokes(field[y][x + 1]) {
                    Some(n) => n,
                    None => continue,
                };
                let n_down = match normalized_stokes(field[y + 1][x]) {
                    Some(n) => n,
                    None => continue,
                };

                // Gradient magnitude between center and neighbors.
                let dx = [
                    n_right[0] - n_center[0],
                    n_right[1] - n_center[1],
                    n_right[2] - n_center[2],
                ];
                let dy = [
                    n_down[0] - n_center[0],
                    n_down[1] - n_center[1],
                    n_down[2] - n_center[2],
                ];
                let grad_mag = (dx.iter().map(|v| v * v).sum::<f64>()
                    + dy.iter().map(|v| v * v).sum::<f64>())
                .sqrt();
                gradient_magnitudes.push(grad_mag);

                // Local coherence: dot product of neighboring vectors.
                let dot_right = n_center[0] * n_right[0]
                    + n_center[1] * n_right[1]
                    + n_center[2] * n_right[2];
                let dot_down = n_center[0] * n_down[0]
                    + n_center[1] * n_down[1]
                    + n_center[2] * n_down[2];
                let local_coh = (dot_right.clamp(-1.0, 1.0) + dot_down.clamp(-1.0, 1.0)) / 2.0;
                local_coherences.push(local_coh);
            }
        }

        if gradient_magnitudes.is_empty() {
            return SpatialCoherenceResult {
                coherence_score: 0.0,
                coherent_region_count: 0,
                average_region_size: 0.0,
                largest_region_size: 0,
                coherence_length: 0.0,
                gradient_mean: 0.0,
                gradient_std: 0.0,
            };
        }

        // Overall coherence score from average local coherence.
        let coherence_score = local_coherences.iter().sum::<f64>() / local_coherences.len() as f64;

        // Gradient statistics.
        let gradient_mean =
            gradient_magnitudes.iter().sum::<f64>() / gradient_magnitudes.len() as f64;
        let gradient_var = gradient_magnitudes
            .iter()
            .map(|g| (g - gradient_mean).powi(2))
            .sum::<f64>()
            / gradient_magnitudes.len() as f64;
        let gradient_std = gradient_var.sqrt();

        // Count coherent regions (simplified: threshold local coherence).
        let mut coherent_regions = 0;
        let mut total_region_pixels = 0usize;
        let mut largest_region = 0usize;
        let mut current_region = 0usize;

        for &coh in &local_coherences {
            if coh > self.coherence_threshold {
                current_region += 1;
            } else {
                if current_region > 0 {
                    coherent_regions += 1;
                    total_region_pixels += current_region;
                    largest_region = largest_region.max(current_region);
                    current_region = 0;
                }
            }
        }
        if current_region > 0 {
            coherent_regions += 1;
            total_region_pixels += current_region;
            largest_region = largest_region.max(current_region);
        }

        let average_region_size = if coherent_regions > 0 {
            total_region_pixels as f64 / coherent_regions as f64
        } else {
            0.0
        };

        // Coherence length: inverse of gradient std (higher std = shorter coherence length).
        let coherence_length = if gradient_std > 0.0 {
            1.0 / gradient_std
        } else {
            256.0
        };

        SpatialCoherenceResult {
            coherence_score,
            coherent_region_count: coherent_regions,
            average_region_size,
            largest_region_size: largest_region,
            coherence_length,
            gradient_mean,
            gradient_std,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch detection
// ---------------------------------------------------------------------------

/// Result of a single detection in a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchDetectionResult {
    /// Index of the detection in the batch.
    pub index: usize,
    /// Detection outcome.
    pub outcome: DetectionOutcome,
    /// Processing time in microseconds.
    pub processing_time_us: f64,
}

/// Batch detection results with aggregate statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchDetectionResults {
    /// Individual detection results.
    pub results: Vec<BatchDetectionResult>,
    /// Total number of detections processed.
    pub total_count: usize,
    /// Number of signals detected.
    pub signal_count: usize,
    /// Average processing time per detection.
    pub avg_processing_time_us: f64,
    /// Signal detection rate.
    pub signal_rate: f64,
}

impl BatchDetectionResults {
    /// Returns the list of detected signals.
    pub fn signals(&self) -> Vec<&BatchDetectionResult> {
        self.results
            .iter()
            .filter(|r| r.outcome.is_signal())
            .collect()
    }

    /// Returns the list of noise classifications.
    pub fn noise(&self) -> Vec<&BatchDetectionResult> {
        self.results
            .iter()
            .filter(|r| r.outcome.is_noise())
            .collect()
    }

    /// Returns the list of interference classifications.
    pub fn interference(&self) -> Vec<&BatchDetectionResult> {
        self.results
            .iter()
            .filter(|r| r.outcome.is_interference())
            .collect()
    }

    /// Returns the list of false positives.
    pub fn false_positives(&self) -> Vec<&BatchDetectionResult> {
        self.results
            .iter()
            .filter(|r| r.outcome.is_false_positive())
            .collect()
    }
}

/// Batch detector for processing multiple Stokes fields efficiently.
pub struct BatchDetector {
    detector: SignalDetector,
    results_history: Vec<BatchDetectionResults>,
    max_history: usize,
}

impl BatchDetector {
    /// Create a new batch detector.
    pub fn new(config: DetectionConfig) -> Self {
        Self {
            detector: SignalDetector::new(config),
            results_history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Process a batch of Stokes fields.
    pub fn process_batch(
        &mut self,
        fields: Vec<&[[[f64; 4]; 256]; 256]>,
    ) -> BatchDetectionResults {
        let start = std::time::Instant::now();
        let mut results = Vec::with_capacity(fields.len());
        let mut signal_count = 0usize;

        for (i, field) in fields.iter().enumerate() {
            let detect_start = std::time::Instant::now();
            match self.detector.detect(field) {
                Ok(outcome) => {
                    if outcome.is_signal() {
                        signal_count += 1;
                    }
                    let processing_time_us = detect_start.elapsed().as_micros() as f64;
                    results.push(BatchDetectionResult {
                        index: i,
                        outcome,
                        processing_time_us,
                    });
                }
                Err(_) => {
                    let processing_time_us = detect_start.elapsed().as_micros() as f64;
                    results.push(BatchDetectionResult {
                        index: i,
                        outcome: DetectionOutcome::InsufficientData {
                            missing: "detection error".into(),
                        },
                        processing_time_us,
                    });
                }
            }
        }

        let total_time_us = start.elapsed().as_micros() as f64;
        let avg_processing_time_us = if fields.is_empty() {
            0.0
        } else {
            total_time_us / fields.len() as f64
        };
        let signal_rate = if fields.is_empty() {
            0.0
        } else {
            signal_count as f64 / fields.len() as f64
        };

        let batch_results = BatchDetectionResults {
            results,
            total_count: fields.len(),
            signal_count,
            avg_processing_time_us,
            signal_rate,
        };

        // Store in history.
        if self.results_history.len() >= self.max_history {
            self.results_history.remove(0);
        }
        self.results_history.push(batch_results.clone());

        batch_results
    }

    /// Process a batch of flat Stokes data buffers.
    pub fn process_batch_flat(
        &mut self,
        batches: Vec<(&[f64], usize, usize)>,
    ) -> BatchDetectionResults {
        let start = std::time::Instant::now();
        let mut results = Vec::with_capacity(batches.len());
        let mut signal_count = 0usize;

        for (i, (data, width, height)) in batches.iter().enumerate() {
            let detect_start = std::time::Instant::now();
            match self.detector.detect_raw(data, *width, *height) {
                Ok(outcome) => {
                    if outcome.is_signal() {
                        signal_count += 1;
                    }
                    let processing_time_us = detect_start.elapsed().as_micros() as f64;
                    results.push(BatchDetectionResult {
                        index: i,
                        outcome,
                        processing_time_us,
                    });
                }
                Err(_) => {
                    let processing_time_us = detect_start.elapsed().as_micros() as f64;
                    results.push(BatchDetectionResult {
                        index: i,
                        outcome: DetectionOutcome::InsufficientData {
                            missing: "detection error".into(),
                        },
                        processing_time_us,
                    });
                }
            }
        }

        let total_time_us = start.elapsed().as_micros() as f64;
        let avg_processing_time_us = if batches.is_empty() {
            0.0
        } else {
            total_time_us / batches.len() as f64
        };
        let signal_rate = if batches.is_empty() {
            0.0
        } else {
            signal_count as f64 / batches.len() as f64
        };

        BatchDetectionResults {
            results,
            total_count: batches.len(),
            signal_count,
            avg_processing_time_us,
            signal_rate,
        }
    }

    /// Returns a reference to the inner signal detector.
    pub fn detector(&self) -> &SignalDetector {
        &self.detector
    }

    /// Returns the results history.
    pub fn results_history(&self) -> &[BatchDetectionResults] {
        &self.results_history
    }

    /// Returns the average signal rate across all batches in history.
    pub fn avg_signal_rate_history(&self) -> f64 {
        if self.results_history.is_empty() {
            return 0.0;
        }
        self.results_history
            .iter()
            .map(|r| r.signal_rate)
            .sum::<f64>()
            / self.results_history.len() as f64
    }

    /// Reset the batch detector.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.results_history.clear();
    }
}

impl Default for BatchDetector {
    fn default() -> Self {
        Self::new(DetectionConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Detection quality metrics
// ---------------------------------------------------------------------------

/// Quality metrics for a single detection.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionQualityMetrics {
    /// Skyrmion number of the detection.
    pub skyrmion_number: i32,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Signal-to-noise ratio estimate.
    pub snr_estimate: f64,
    /// Spatial coherence score.
    pub spatial_coherence: f64,
    /// Temporal stability score.
    pub temporal_stability: f64,
    /// Overall quality score (0.0 to 1.0).
    pub overall_quality: f64,
    /// Whether the detection meets quality thresholds.
    pub passes_quality_gate: bool,
}

impl DetectionQualityMetrics {
    /// Returns the quality grade as a string.
    pub fn quality_grade(&self) -> &'static str {
        match self.overall_quality {
            q if q >= 0.9 => "A",
            q if q >= 0.8 => "B",
            q if q >= 0.7 => "C",
            q if q >= 0.6 => "D",
            q if q >= 0.5 => "E",
            _ => "F",
        }
    }

    /// Returns true if the detection is high quality (grade A or B).
    pub fn is_high_quality(&self) -> bool {
        self.overall_quality >= 0.8
    }
}

/// Computes quality metrics for a detection.
pub struct QualityMetricComputer {
    /// Minimum SNR for quality gate.
    pub min_snr: f64,
    /// Minimum spatial coherence for quality gate.
    pub min_spatial_coherence: f64,
    /// Minimum temporal stability for quality gate.
    pub min_temporal_stability: f64,
    /// Weight for confidence in overall quality.
    pub confidence_weight: f64,
    /// Weight for SNR in overall quality.
    pub snr_weight: f64,
    /// Weight for spatial coherence in overall quality.
    pub coherence_weight: f64,
    /// Weight for temporal stability in overall quality.
    pub stability_weight: f64,
}

impl Default for QualityMetricComputer {
    fn default() -> Self {
        Self {
            min_snr: 15.0,
            min_spatial_coherence: 0.6,
            min_temporal_stability: 0.7,
            confidence_weight: 0.3,
            snr_weight: 0.3,
            coherence_weight: 0.2,
            stability_weight: 0.2,
        }
    }
}

impl QualityMetricComputer {
    /// Create a new quality metric computer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute quality metrics for a detection.
    pub fn compute(
        &self,
        skyrmion_number: i32,
        confidence: f64,
        snr_db: f64,
        spatial_coherence: f64,
        temporal_stability: f64,
    ) -> DetectionQualityMetrics {
        // Normalize SNR to 0-1 scale (assuming 30 dB is excellent).
        let snr_normalized = (snr_db / 30.0).clamp(0.0, 1.0);

        let overall_quality = confidence * self.confidence_weight
            + snr_normalized * self.snr_weight
            + spatial_coherence * self.coherence_weight
            + temporal_stability * self.stability_weight;

        let passes_quality_gate = snr_db >= self.min_snr
            && spatial_coherence >= self.min_spatial_coherence
            && temporal_stability >= self.min_temporal_stability
            && confidence >= 0.7;

        DetectionQualityMetrics {
            skyrmion_number,
            confidence,
            snr_estimate: snr_db,
            spatial_coherence,
            temporal_stability,
            overall_quality: overall_quality.clamp(0.0, 1.0),
            passes_quality_gate,
        }
    }

    /// Compute quality metrics from a detection outcome.
    pub fn compute_from_outcome(&self, outcome: &DetectionOutcome) -> Option<DetectionQualityMetrics> {
        match outcome {
            DetectionOutcome::Signal {
                signal,
                skyrmion_number_computed: _,
                constellation_point: _,
                confidence,
            } => Some(self.compute(
                signal.skyrmion_number,
                *confidence,
                20.0, // default SNR estimate
                0.8,  // default spatial coherence
                0.9,  // default temporal stability
            )),
            _ => None,
        }
    }
}

impl Default for SignalDetector {
    fn default() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_default_has_binary_constellation() {
        let detector = SignalDetector::default();
        assert!(detector.constellation().contains(-1));
        assert!(detector.constellation().contains(1));
    }

    #[test]
    fn detection_outcome_is_signal() {
        let sig = TopologicalSignal::new(3);
        let outcome = DetectionOutcome::Signal {
            signal: sig.clone(),
            skyrmion_number_computed: 3.0,
            constellation_point: 3,
            confidence: 0.95,
        };
        assert!(outcome.is_signal());
        assert!(outcome.into_result().0.is_signal());
    }

    #[test]
    fn detection_outcome_noise_is_not_signal() {
        let outcome = DetectionOutcome::Noise {
            noise_floor: 0.1,
            max_skyrmion_density: 0.01,
        };
        assert!(!outcome.is_signal());
        assert!(outcome.is_recoverable());
    }

    #[test]
    fn detection_outcome_interference_is_recoverable() {
        let outcome = DetectionOutcome::Interference {
            interference_type: "test".into(),
            severity: 0.5,
        };
        assert!(outcome.is_recoverable());
        assert!(!outcome.is_signal());
    }

    #[test]
    fn detection_outcome_false_positive_not_recoverable() {
        let outcome = DetectionOutcome::FalsePositive {
            q_candidate: 5,
            reason: "test".into(),
        };
        assert!(!outcome.is_recoverable());
        assert!(!outcome.is_signal());
    }

    #[test]
    fn normalized_stokes_rejects_zero_s0() {
        assert!(normalized_stokes([0.0, 1.0, 0.0, 0.0]).is_none());
        assert!(normalized_stokes([1e-12, 1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn normalized_stokes_returns_unit_vector() {
        let s = [1.0, 0.0, 1.0, 0.0]; // linear polarization at 45 deg
        let n = normalized_stokes(s).unwrap();
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn compute_skyrmion_zero_for_uniform_field() {
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0]; // uniform linear polarization
            }
        }
        let mask = IntensityMask::new(0.1).apply(&field);
        let q = compute_skyrmion_number_masked(&field, &mask);
        assert!((q - 0.0).abs() < 0.1);
    }

    #[test]
    fn statistics_signal_rate_starts_zero() {
        let detector = SignalDetector::default();
        assert_eq!(detector.statistics().signal_rate(), 0.0);
    }

    #[test]
    fn reset_clears_statistics() {
        let mut detector = SignalDetector::default();
        detector.reset();
        assert_eq!(detector.statistics().total_attempts, 0);
        assert!(detector.recent_outcomes().is_empty());
    }

    #[test]
    fn detect_raw_rejects_empty_data() {
        let mut detector = SignalDetector::default();
        let result = detector.detect_raw(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn detect_raw_rejects_insufficient_data() {
        let mut detector = SignalDetector::default();
        let result = detector.detect_raw(&[1.0, 0.0, 0.0, 0.0], 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn now_ms_is_monotonic_approx() {
        let t1 = now_ms();
        let t2 = now_ms();
        assert!(t2 >= t1);
    }

    #[test]
    fn detection_config_custom_values() {
        let config = DetectionConfig {
            grid_resolution: 512,
            intensity_threshold: 0.2,
            max_crosstalk: 0.1,
            min_confidence: 0.9,
            constellation: Constellation::quaternary().unwrap(),
            enable_generalized: true,
            generalized_components: 3,
            stokes_calibration: None,
        };
        let detector = SignalDetector::new(config.clone());
        assert_eq!(detector.config().grid_resolution, 512);
        assert_eq!(detector.config().intensity_threshold, 0.2);
    }
}

// Provide a config accessor (used in test above).
impl SignalDetector {
    /// Returns a reference to the detection configuration.
    pub fn config(&self) -> &DetectionConfig {
        &self.config
    }
}