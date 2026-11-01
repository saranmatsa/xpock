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

// ---------------------------------------------------------------------------
// Detection pipeline orchestration
// ---------------------------------------------------------------------------

/// Stages of the topological signal detection pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionStage {
    /// Preprocessing: noise reduction and field normalization.
    Preprocessing,
    /// Intensity masking: suppress low-reliability regions.
    IntensityMasking,
    /// Skyrmion number computation.
    SkyrmionComputation,
    /// Constellation matching.
    ConstellationMatching,
    /// Noise classification.
    NoiseClassification,
    /// Post-processing: track integration and quality assessment.
    PostProcessing,
}

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageConfig {
    /// Whether this stage is enabled.
    pub enabled: bool,
    /// Timeout in milliseconds for this stage.
    pub timeout_ms: u64,
    /// Retry count on failure.
    pub retry_count: u32,
    /// Minimum quality threshold to proceed to next stage.
    pub quality_threshold: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: 1000,
            retry_count: 3,
            quality_threshold: 0.5,
        }
    }
}

/// Result from a single pipeline stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageResult {
    /// Stage that produced this result.
    pub stage: DetectionStage,
    /// Whether the stage succeeded.
    pub success: bool,
    /// Processing time in microseconds.
    pub processing_time_us: f64,
    /// Quality score (0.0 to 1.0).
    pub quality_score: f64,
    /// Output data as opaque bytes (stage-specific).
    pub output: Vec<u8>,
    /// Error message if stage failed.
    pub error: Option<String>,
}

impl StageResult {
    /// Create a successful stage result.
    pub fn success(stage: DetectionStage, processing_time_us: f64, quality_score: f64) -> Self {
        Self {
            stage,
            success: true,
            processing_time_us,
            quality_score,
            output: Vec::new(),
            error: None,
        }
    }

    /// Create a failed stage result.
    pub fn failure(stage: DetectionStage, processing_time_us: f64, error: String) -> Self {
        Self {
            stage,
            success: false,
            processing_time_us,
            quality_score: 0.0,
            output: Vec::new(),
            error: Some(error),
        }
    }
}

/// Complete detection pipeline orchestrating all stages.
pub struct DetectionPipeline {
    /// Configuration for each stage.
    stage_configs: std::collections::HashMap<DetectionStage, StageConfig>,
    /// Whether to continue on stage failure.
    pub continue_on_failure: bool,
    /// Pipeline statistics.
    pub stats: PipelineStats,
}

/// Statistics accumulated by the detection pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineStats {
    /// Total number of detections processed.
    pub total_detections: u64,
    /// Number of successful detections.
    pub successful_detections: u64,
    /// Number of failed detections.
    pub failed_detections: u64,
    /// Average processing time per detection.
    pub avg_processing_time_us: f64,
    /// Stage failure counts.
    pub stage_failures: std::collections::HashMap<DetectionStage, u64>,
    /// Stage average quality scores.
    pub stage_quality: std::collections::HashMap<DetectionStage, f64>,
}

impl DetectionPipeline {
    /// Create a new detection pipeline with default stage configuration.
    pub fn new() -> Self {
        let mut stage_configs = std::collections::HashMap::new();
        for stage in [
            DetectionStage::Preprocessing,
            DetectionStage::IntensityMasking,
            DetectionStage::SkyrmionComputation,
            DetectionStage::ConstellationMatching,
            DetectionStage::NoiseClassification,
            DetectionStage::PostProcessing,
        ] {
            stage_configs.insert(stage, StageConfig::default());
        }
        Self {
            stage_configs,
            continue_on_failure: true,
            stats: PipelineStats::default(),
        }
    }

    /// Create a pipeline with custom stage configuration.
    pub fn with_config(stage_configs: std::collections::HashMap<DetectionStage, StageConfig>) -> Self {
        Self {
            stage_configs,
            continue_on_failure: true,
            stats: PipelineStats::default(),
        }
    }

    /// Run the complete detection pipeline on a Stokes field.
    pub fn run(
        &mut self,
        detector: &mut SignalDetector,
        field: &[[[f64; 4]; 256]; 256],
    ) -> PipelineResult {
        let start = crate::detection::signal_detector::lib_ms();
        let mut stage_results = Vec::with_capacity(6);
        let mut overall_success = true;

        // Stage 1: Preprocessing
        if self.stage_configs[&DetectionStage::Preprocessing].enabled {
            let stage_start = lib_ms();
            let result = self.run_preprocessing(field);
            stage_results.push(result.clone());
            if !result.success {
                overall_success = false;
                *self.stats.stage_failures.entry(DetectionStage::Preprocessing).or_insert(0) += 1;
                if !self.continue_on_failure {
                    return self.finalize_pipeline(stage_results, start, overall_success);
                }
            }
        }

        // Stage 2: Intensity masking
        if self.stage_configs[&DetectionStage::IntensityMasking].enabled {
            let stage_start = lib_ms();
            let result = self.run_intensity_masking(detector, field);
            stage_results.push(result.clone());
            if !result.success {
                overall_success = false;
                *self.stats.stage_failures.entry(DetectionStage::IntensityMasking).or_insert(0) += 1;
                if !self.continue_on_failure {
                    return self.finalize_pipeline(stage_results, start, overall_success);
                }
            }
        }

        // Stage 3: Skyrmion computation
        if self.stage_configs[&DetectionStage::SkyrmionComputation].enabled {
            let stage_start = lib_ms();
            let result = self.run_skyrmion_computation(detector, field);
            stage_results.push(result.clone());
            if !result.success {
                overall_success = false;
                *self.stats.stage_failures.entry(DetectionStage::SkyrmionComputation).or_insert(0) += 1;
                if !self.continue_on_failure {
                    return self.finalize_pipeline(stage_results, start, overall_success);
                }
            }
        }

        // Stage 4: Constellation matching
        if self.stage_configs[&DetectionStage::ConstellationMatching].enabled {
            let stage_start = lib_ms();
            let result = self.run_constellation_matching(detector);
            stage_results.push(result.clone());
            if !result.success {
                overall_success = false;
                *self.stats.stage_failures.entry(DetectionStage::ConstellationMatching).or_insert(0) += 1;
                if !self.continue_on_failure {
                    return self.finalize_pipeline(stage_results, start, overall_success);
                }
            }
        }

        // Stage 5: Noise classification
        if self.stage_configs[&DetectionStage::NoiseClassification].enabled {
            let stage_start = lib_ms();
            let result = self.run_noise_classification(detector);
            stage_results.push(result.clone());
            if !result.success {
                overall_success = false;
                *self.stats.stage_failures.entry(DetectionStage::NoiseClassification).or_insert(0) += 1;
                if !self.continue_on_failure {
                    return self.finalize_pipeline(stage_results, start, overall_success);
                }
            }
        }

        // Stage 6: Post-processing
        if self.stage_configs[&DetectionStage::PostProcessing].enabled {
            let stage_start = lib_ms();
            let result = self.run_post_processing(detector);
            stage_results.push(result.clone());
            if !result.success {
                overall_success = false;
                *self.stats.stage_failures.entry(DetectionStage::PostProcessing).or_insert(0) += 1;
            }
        }

        self.finalize_pipeline(stage_results, start, overall_success)
    }

    fn run_preprocessing(&self, field: &[[[f64; 4]; 256]; 256]) -> StageResult {
        let start = lib_ms();
        // Validate field structure.
        let mut valid_pixels = 0u64;
        let mut total_intensity = 0.0f64;
        for y in 0..256 {
            for x in 0..256 {
                let s0 = field[y][x][0];
                if s0.abs() > 1e-10 {
                    valid_pixels += 1;
                    total_intensity += s0.abs();
                }
            }
        }
        let quality = valid_pixels as f64 / 65536.0;
        StageResult::success(
            DetectionStage::Preprocessing,
            (lib_ms() - start) as f64,
            quality,
        )
    }

    fn run_intensity_masking(&self, _detector: &SignalDetector, _field: &[[[f64; 4]; 256]; 256]) -> StageResult {
        let start = lib_ms();
        StageResult::success(
            DetectionStage::IntensityMasking,
            (lib_ms() - start) as f64,
            0.85,
        )
    }

    fn run_skyrmion_computation(&self, detector: &SignalDetector, field: &[[[f64; 4]; 256]; 256]) -> StageResult {
        let start = lib_ms();
        // Use the detector's internal skyrmion computation.
        let q = compute_skyrmion_number_masked(field, &IntensityMask::new(0.1).apply(field));
        let quality = if q.abs() < 0.1 { 0.3 } else { 0.9 };
        StageResult::success(
            DetectionStage::SkyrmionComputation,
            (lib_ms() - start) as f64,
            quality,
        )
    }

    fn run_constellation_matching(&self, detector: &SignalDetector) -> StageResult {
        let start = lib_ms();
        let constellation = &detector.constellation;
        let quality = if constellation.len() >= 2 { 0.9 } else { 0.4 };
        StageResult::success(
            DetectionStage::ConstellationMatching,
            (lib_ms() - start) as f64,
            quality,
        )
    }

    fn run_noise_classification(&self, detector: &SignalDetector) -> StageResult {
        let start = lib_ms();
        let stats = detector.statistics();
        let quality = if stats.total_attempts > 0 {
            stats.signal_rate()
        } else {
            0.5
        };
        StageResult::success(
            DetectionStage::NoiseClassification,
            (lib_ms() - start) as f64,
            quality,
        )
    }

    fn run_post_processing(&self, detector: &SignalDetector) -> StageResult {
        let start = lib_ms();
        let stats = detector.statistics();
        let quality = stats.false_positive_rate().clamp(0.0, 1.0);
        StageResult::success(
            DetectionStage::PostProcessing,
            (lib_ms() - start) as f64,
            1.0 - quality,
        )
    }

    fn finalize_pipeline(
        &mut self,
        stage_results: Vec<StageResult>,
        start_ms: u64,
        overall_success: bool,
    ) -> PipelineResult {
        let total_time = (lib_ms() - start_ms) as f64;
        self.stats.total_detections += 1;
        if overall_success {
            self.stats.successful_detections += 1;
        } else {
            self.stats.failed_detections += 1;
        }
        self.stats.avg_processing_time_us =
            (self.stats.avg_processing_time_us * (self.stats.total_detections - 1) as f64 + total_time)
                / self.stats.total_detections as f64;

        for result in &stage_results {
            let quality_entry = self
                .stats
                .stage_quality
                .entry(result.stage)
                .or_insert(0.0);
            let count = self.stats.total_detections as f64;
            *quality_entry = (*quality_entry * (count - 1.0) + result.quality_score) / count;
        }

        PipelineResult {
            success: overall_success,
            stage_results,
            total_processing_time_us: total_time,
            timestamp: lib_ms(),
        }
    }

    /// Returns pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Returns the stage configuration for a given stage.
    pub fn stage_config(&self, stage: DetectionStage) -> Option<&StageConfig> {
        self.stage_configs.get(&stage)
    }

    /// Update the configuration for a pipeline stage.
    pub fn update_stage_config(&mut self, stage: DetectionStage, config: StageConfig) {
        self.stage_configs.insert(stage, config);
    }

    /// Enable or disable a pipeline stage.
    pub fn set_stage_enabled(&mut self, stage: DetectionStage, enabled: bool) {
        if let Some(cfg) = self.stage_configs.get_mut(&stage) {
            cfg.enabled = enabled;
        }
    }

    /// Reset pipeline statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStats::default();
    }
}

impl Default for DetectionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete pipeline execution result.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineResult {
    /// Whether the pipeline completed successfully.
    pub success: bool,
    /// Per-stage results.
    pub stage_results: Vec<StageResult>,
    /// Total processing time in microseconds.
    pub total_processing_time_us: f64,
    /// Execution timestamp.
    pub timestamp: u64,
}

impl PipelineResult {
    /// Returns the average quality across all stages.
    pub fn avg_quality(&self) -> f64 {
        if self.stage_results.is_empty() {
            return 0.0;
        }
        self.stage_results
            .iter()
            .map(|r| r.quality_score)
            .sum::<f64>()
            / self.stage_results.len() as f64
    }

    /// Returns true if all stages succeeded.
    pub fn all_stages_succeeded(&self) -> bool {
        self.stage_results.iter().all(|r| r.success)
    }

    /// Returns the first stage error, if any.
    pub fn first_error(&self) -> Option<&str> {
        self.stage_results
            .iter()
            .find(|r| !r.success)
            .and_then(|r| r.error.as_deref())
    }
}

// ---------------------------------------------------------------------------
// Signal quality assessment
// ---------------------------------------------------------------------------

/// Comprehensive quality assessment for a detected topological signal.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalQualityAssessment {
    /// Overall quality score (0.0 to 1.0).
    pub overall_score: f64,
    /// Component quality scores.
    pub components: QualityComponents,
    /// Quality grade letter.
    pub grade: QualityGrade,
    /// Whether the signal passes quality gates.
    pub passes_gate: bool,
    /// Recommended action based on quality.
    pub recommendation: QualityRecommendation,
}

/// Component quality scores.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityComponents {
    /// Topological stability (how close Q is to integer).
    pub topological_stability: f64,
    /// Signal-to-noise ratio quality.
    pub snr_quality: f64,
    /// Spatial coherence quality.
    pub spatial_coherence: f64,
    /// Temporal stability quality.
    pub temporal_stability: f64,
    /// Intensity reliability quality.
    pub intensity_reliability: f64,
    /// Classification confidence quality.
    pub classification_confidence: f64,
}

/// Quality grade letter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    A, // Excellent (0.9-1.0)
    B, // Good (0.8-0.9)
    C, // Fair (0.7-0.8)
    D, // Poor (0.6-0.7)
    E, // Very poor (0.5-0.6)
    F, // Unacceptable (<0.5)
}

/// Recommended action based on signal quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityRecommendation {
    /// Signal is reliable, proceed with normal processing.
    Accept,
    /// Signal is acceptable but monitor closely.
    AcceptWithCaution,
    /// Signal needs re-acquisition or averaging.
    Reacquire,
    /// Signal should be rejected.
    Reject,
    /// Signal requires manual review.
    ManualReview,
}

impl SignalQualityAssessment {
    /// Create a new quality assessment.
    pub fn new(
        topological_stability: f64,
        snr_quality: f64,
        spatial_coherence: f64,
        temporal_stability: f64,
        intensity_reliability: f64,
        classification_confidence: f64,
    ) -> Self {
        let components = QualityComponents {
            topological_stability,
            snr_quality,
            spatial_coherence,
            temporal_stability,
            intensity_reliability,
            classification_confidence,
        };

        // Weighted overall score.
        let overall_score = topological_stability * 0.25
            + snr_quality * 0.20
            + spatial_coherence * 0.20
            + temporal_stability * 0.15
            + intensity_reliability * 0.10
            + classification_confidence * 0.10;

        let grade = if overall_score >= 0.9 {
            QualityGrade::A
        } else if overall_score >= 0.8 {
            QualityGrade::B
        } else if overall_score >= 0.7 {
            QualityGrade::C
        } else if overall_score >= 0.6 {
            QualityGrade::D
        } else if overall_score >= 0.5 {
            QualityGrade::E
        } else {
            QualityGrade::F
        };

        let passes_gate = overall_score >= 0.6;

        let recommendation = match grade {
            QualityGrade::A => QualityRecommendation::Accept,
            QualityGrade::B => QualityRecommendation::AcceptWithCaution,
            QualityGrade::C => QualityRecommendation::AcceptWithCaution,
            QualityGrade::D => QualityRecommendation::Reacquire,
            QualityGrade::E => QualityRecommendation::Reject,
            QualityGrade::F => QualityRecommendation::Reject,
        };

        Self {
            overall_score: overall_score.clamp(0.0, 1.0),
            components,
            grade,
            passes_gate,
            recommendation,
        }
    }

    /// Returns true if the signal is high quality (grade A or B).
    pub fn is_high_quality(&self) -> bool {
        matches!(self.grade, QualityGrade::A | QualityGrade::B)
    }

    /// Returns the lowest component score (bottleneck).
    pub fn bottleneck(&self) -> f64 {
        self.components
            .topological_stability
            .min(self.components.snr_quality)
            .min(self.components.spatial_coherence)
            .min(self.components.temporal_stability)
            .min(self.components.intensity_reliability)
            .min(self.components.classification_confidence)
    }
}

/// Assesses signal quality from detection results.
pub struct QualityAssessor {
    /// Minimum overall score for acceptance.
    pub min_overall_score: f64,
    /// Minimum topological stability.
    pub min_topological_stability: f64,
    /// Minimum SNR quality.
    pub min_snr_quality: f64,
    /// Whether to use temporal stability in assessment.
    pub use_temporal_stability: bool,
}

impl QualityAssessor {
    /// Create a new quality assessor.
    pub fn new() -> Self {
        Self {
            min_overall_score: 0.6,
            min_topological_stability: 0.7,
            min_snr_quality: 0.5,
            use_temporal_stability: true,
        }
    }

    /// Assess the quality of a detection outcome.
    pub fn assess(&self, outcome: &DetectionOutcome) -> Option<SignalQualityAssessment> {
        match outcome {
            DetectionOutcome::Signal {
                signal,
                skyrmion_number_computed: _,
                constellation_point: _,
                confidence,
            } => {
                let topological_stability = (1.0 - (skyrmion_number_computed.fract().abs())).clamp(0.0, 1.0);
                let snr_quality = (*confidence).clamp(0.0, 1.0);
                let spatial_coherence = (*confidence * 0.9).clamp(0.0, 1.0);
                let temporal_stability = if self.use_temporal_stability {
                    (*confidence * 0.85).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let intensity_reliability = (*confidence * 0.95).clamp(0.0, 1.0);
                let classification_confidence = *confidence;

                Some(SignalQualityAssessment::new(
                    topological_stability,
                    snr_quality,
                    spatial_coherence,
                    temporal_stability,
                    intensity_reliability,
                    classification_confidence,
                ))
            }
            DetectionOutcome::Noise { .. } => Some(SignalQualityAssessment::new(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )),
            DetectionOutcome::Interference { severity, .. } => {
                Some(SignalQualityAssessment::new(
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    (1.0 - severity).clamp(0.0, 1.0),
                ))
            }
            DetectionOutcome::FalsePositive { confidence, .. } => {
                Some(SignalQualityAssessment::new(
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    (1.0 - confidence).clamp(0.0, 1.0),
                ))
            }
            DetectionOutcome::Corrupted { .. } => Some(SignalQualityAssessment::new(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )),
            DetectionOutcome::InsufficientData { .. } => Some(SignalQualityAssessment::new(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )),
        }
    }

    /// Returns true if the assessment passes the quality gate.
    pub fn passes_gate(&self, assessment: &SignalQualityAssessment) -> bool {
        assessment.overall_score >= self.min_overall_score
            && assessment.components.topological_stability >= self.min_topological_stability
            && assessment.components.snr_quality >= self.min_snr_quality
    }

    /// Update quality thresholds.
    pub fn update_thresholds(
        &mut self,
        min_overall: f64,
        min_topo: f64,
        min_snr: f64,
    ) {
        self.min_overall_score = min_overall.clamp(0.0, 1.0);
        self.min_topological_stability = min_topo.clamp(0.0, 1.0);
        self.min_snr_quality = min_snr.clamp(0.0, 1.0);
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Constellation analysis
// ---------------------------------------------------------------------------

/// Analysis of constellation point distribution and separation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationAnalysis {
    /// Number of constellation points analyzed.
    pub point_count: usize,
    /// Minimum separation between adjacent points.
    pub min_separation: f64,
    /// Average separation between adjacent points.
    pub avg_separation: f64,
    /// Separation quality score (0.0 to 1.0).
    pub separation_quality: f64,
    /// Whether constellation points are well-separated.
    pub well_separated: bool,
    /// Recommended modulation order based on separation.
    pub recommended_order: usize,
}

/// Analyzes topological signal constellation quality.
pub struct ConstellationAnalyzer {
    /// Minimum separation for reliable discrimination.
    pub min_reliable_separation: f64,
    /// Maximum recommended modulation order.
    pub max_order: usize,
}

impl ConstellationAnalyzer {
    /// Create a new constellation analyzer.
    pub fn new() -> Self {
        Self {
            min_reliable_separation: 2.0,
            max_order: 8,
        }
    }

    /// Analyze a constellation.
    pub fn analyze(&self, constellation: &crate::types::Constellation) -> ConstellationAnalysis {
        let points: Vec<i32> = constellation.iter().copied().collect();
        let point_count = points.len();

        if point_count < 2 {
            return ConstellationAnalysis {
                point_count,
                min_separation: 0.0,
                avg_separation: 0.0,
                separation_quality: 0.0,
                well_separated: false,
                recommended_order: 1,
            };
        }

        // Compute separations between adjacent points.
        let mut separations: Vec<f64> = Vec::new();
        for window in points.windows(2) {
            separations.push((window[1] - window[0]).abs() as f64);
        }

        let min_sep = separations.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg_sep = separations.iter().sum::<f64>() / separations.len() as f64;

        // Separation quality: how well separated relative to minimum reliable.
        let separation_quality = (min_sep / self.min_reliable_separation).clamp(0.0, 1.0);
        let well_separated = min_sep >= self.min_reliable_separation;

        // Recommended order based on separation quality.
        let recommended_order = if separation_quality >= 0.9 {
            self.max_order.min(8)
        } else if separation_quality >= 0.7 {
            4
        } else if separation_quality >= 0.5 {
            2
        } else {
            1
        };

        ConstellationAnalysis {
            point_count,
            min_separation: min_sep,
            avg_separation: avg_sep,
            separation_quality,
            well_separated,
            recommended_order,
        }
    }

    /// Returns true if the constellation supports reliable k-bit modulation.
    pub fn supports_order(&self, analysis: &ConstellationAnalysis, order: usize) -> bool {
        let required_points = 2usize.pow(order as u32);
        analysis.point_count >= required_points && analysis.well_separated
    }

    /// Update separation threshold.
    pub fn update_threshold(&mut self, min_separation: f64) {
        self.min_reliable_separation = min_separation.max(0.0);
    }
}

impl Default for ConstellationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Frequency-domain analysis
// ---------------------------------------------------------------------------

/// Frequency-domain characteristics of a topological signal.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyAnalysis {
    /// Dominant frequency component.
    pub dominant_frequency: f64,
    /// Spectral bandwidth.
    pub bandwidth: f64,
    /// Spectral centroid.
    pub spectral_centroid: f64,
    /// Frequency-domain SNR.
    pub spectral_snr: f64,
    /// Whether the signal has a clean spectral signature.
    pub clean_spectrum: bool,
}

/// Performs frequency-domain analysis of Stokes fields.
pub struct FrequencyDomainAnalyzer {
    /// FFT size for analysis.
    pub fft_size: usize,
    /// Frequency resolution.
    pub frequency_resolution: f64,
    /// Noise floor estimate in frequency domain.
    pub noise_floor_db: f64,
}

impl FrequencyDomainAnalyzer {
    /// Create a new frequency domain analyzer.
    pub fn new() -> Self {
        Self {
            fft_size: 256,
            frequency_resolution: 1.0,
            noise_floor_db: -60.0,
        }
    }

    /// Analyze frequency content of a Stokes field.
    pub fn analyze(&self, field: &[[[f64; 4]; 256]; 256]) -> FrequencyAnalysis {
        // Compute spatial frequency content via gradient magnitude spectrum.
        let mut freq_magnitudes = Vec::new();

        for y in 1..255 {
            for x in 1..255 {
                let gx = field[y][x + 1][0] - field[y][x - 1][0];
                let gy = field[y + 1][x][0] - field[y - 1][x][0];
                let mag = (gx * gx + gy * gy).sqrt();
                freq_magnitudes.push(mag);
            }
        }

        if freq_magnitudes.is_empty() {
            return FrequencyAnalysis {
                dominant_frequency: 0.0,
                bandwidth: 0.0,
                spectral_centroid: 0.0,
                spectral_snr: 0.0,
                clean_spectrum: false,
            };
        }

        // Find dominant frequency (peak magnitude location).
        let max_idx = freq_magnitudes
            .iter()
            .enumerate()
            .max_by(|_, b| b.1.partial_cmp(a.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let dominant_frequency = max_idx as f64 * self.frequency_resolution;

        // Compute bandwidth (spread of significant frequencies).
        let max_mag = freq_magnitudes.iter().cloned().fold(0.0f64, f64::max);
        let threshold = max_mag * 0.1;
        let significant_count = freq_magnitudes.iter().filter(|&&m| m > threshold).count();
        let bandwidth = significant_count as f64 * self.frequency_resolution;

        // Spectral centroid.
        let total_mag: f64 = freq_magnitudes.iter().sum();
        let spectral_centroid = if total_mag > 0.0 {
            freq_magnitudes
                .iter()
                .enumerate()
                .map(|(i, &m)| i as f64 * m * self.frequency_resolution)
                .sum::<f64>()
                / total_mag
        } else {
            0.0
        };

        // Spectral SNR.
        let signal_power: f64 = freq_magnitudes.iter().map(|m| m * m).sum();
        let noise_power = self.noise_floor_db.exp() * freq_magnitudes.len() as f64;
        let spectral_snr = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        let clean_spectrum = spectral_snr > 20.0 && bandwidth < 50.0;

        FrequencyAnalysis {
            dominant_frequency,
            bandwidth,
            spectral_centroid,
            spectral_snr,
            clean_spectrum,
        }
    }

    /// Returns true if the frequency signature is suitable for detection.
    pub fn is_suitable(&self, analysis: &FrequencyAnalysis) -> bool {
        analysis.clean_spectrum && analysis.spectral_snr > 15.0
    }

    /// Update FFT size.
    pub fn update_fft_size(&mut self, size: usize) {
        self.fft_size = size.max(64);
    }
}

impl Default for FrequencyDomainAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-resolution analysis
// ---------------------------------------------------------------------------

/// Analysis result at a single resolution level.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionLevelResult {
    /// Resolution level (0 = finest).
    pub level: usize,
    /// Skyrmion number at this level.
    pub skyrmion_number: i32,
    /// Confidence at this level.
    pub confidence: f64,
    /// Processing time in microseconds.
    pub processing_time_us: f64,
}

/// Multi-resolution analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiResolutionAnalysis {
    /// Results per resolution level.
    pub level_results: Vec<ResolutionLevelResult>,
    /// Consensus skyrmion number across levels.
    pub consensus_number: i32,
    /// Consensus confidence.
    pub consensus_confidence: f64,
    /// Whether all levels agree.
    pub level_agreement: bool,
}

/// Performs multi-resolution skyrmion analysis.
pub struct MultiResolutionAnalyzer {
    /// Number of resolution levels.
    pub num_levels: usize,
    /// Downsampling factor between levels.
    pub downsample_factor: usize,
    /// Minimum confidence for level agreement.
    pub min_agreement_confidence: f64,
}

impl MultiResolutionAnalyzer {
    /// Create a new multi-resolution analyzer.
    pub fn new() -> Self {
        Self {
            num_levels: 3,
            downsample_factor: 2,
            min_agreement_confidence: 0.7,
        }
    }

    /// Analyze a field at multiple resolutions.
    pub fn analyze(
        &self,
        field: &[[[f64; 4]; 256]; 256],
    ) -> MultiResolutionAnalysis {
        let mut level_results = Vec::with_capacity(self.num_levels);
        let mut computer = SkyrmionComputer::new();

        for level in 0..self.num_levels {
            let start = lib_ms();
            let scale = self.downsample_factor.pow(level as u32) as f64;

            // Downsample field.
            let downsampled = self.downsample_field(field, scale);

            // Compute skyrmion number.
            let result = computer.compute(&downsampled).unwrap_or_else(|_| {
                SkyrmionComputationResult {
                    skyrmion_number: SkyrmionNumber::default(),
                    field: SkyrmionField::default(),
                    compute_time_us: 0.0,
                    iterations: 0,
                    converged: false,
                }
            });

            level_results.push(ResolutionLevelResult {
                level,
                skyrmion_number: result.skyrmion_number.value,
                confidence: if result.skyrmion_number.precision < 0.1 {
                    0.9
                } else {
                    0.5
                },
                processing_time_us: (lib_ms() - start) as f64,
            });
        }

        // Compute consensus.
        let consensus_number = self.compute_consensus(&level_results);
        let consensus_confidence = level_results
            .iter()
            .map(|r| r.confidence)
            .sum::<f64>()
            / level_results.len() as f64;

        // Check agreement.
        let level_agreement = level_results
            .iter()
            .all(|r| r.skyrmion_number == consensus_number);

        MultiResolutionAnalysis {
            level_results,
            consensus_number,
            consensus_confidence,
            level_agreement,
        }
    }

    fn downsample_field(
        &self,
        field: &[[[f64; 4]; 256]; 256],
        scale: f64,
    ) -> [[[f64; 4]; 256]; 256] {
        let mut result = [[[0.0f64; 4]; 256]; 256];
        let inv_scale = 1.0 / scale;

        for y in 0..256 {
            for x in 0..256 {
                let src_y = (y as f64 * inv_scale) as usize;
                let src_x = (x as f64 * inv_scale) as usize;
                if src_y < 256 && src_x < 256 {
                    result[y][x] = field[src_y][src_x];
                }
            }
        }
        result
    }

    fn compute_consensus(&self, results: &[ResolutionLevelResult]) -> i32 {
        if results.is_empty() {
            return 0;
        }

        // Vote for most common skyrmion number.
        let mut counts: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
        for r in results {
            *counts.entry(r.skyrmion_number).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|_, &v| v)
            .map(|(k, _)| k)
            .unwrap_or(0)
    }

    /// Returns true if multi-resolution analysis is reliable.
    pub fn is_reliable(&self, analysis: &MultiResolutionAnalysis) -> bool {
        analysis.level_agreement && analysis.consensus_confidence >= self.min_agreement_confidence
    }

    /// Update number of levels.
    pub fn update_levels(&mut self, num_levels: usize) {
        self.num_levels = num_levels.max(1).min(8);
    }
}

impl Default for MultiResolutionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Temporal filtering
// ---------------------------------------------------------------------------

/// A filtered detection over time.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredDetection {
    /// Filtered skyrmion number.
    pub skyrmion_number: i32,
    /// Filtered confidence.
    pub confidence: f64,
    /// Number of frames contributing to filter.
    pub frame_count: u32,
    /// Filter stability (0.0 to 1.0).
    pub stability: f64,
}

/// Temporal filter for smoothing detection results over time.
pub struct TemporalFilter {
    /// Filter window size.
    pub window_size: usize,
    /// Smoothing factor (0.0 = no smoothing, 1.0 = heavy smoothing).
    pub smoothing_factor: f64,
    /// History of detections.
    pub history: Vec<DetectionOutcome>,
    /// Maximum history length.
    pub max_history: usize,
}

impl TemporalFilter {
    /// Create a new temporal filter.
    pub fn new() -> Self {
        Self {
            window_size: 5,
            smoothing_factor: 0.3,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Filter a new detection outcome.
    pub fn filter(&mut self, outcome: DetectionOutcome) -> FilteredDetection {
        if let DetectionOutcome::Signal {
            signal,
            skyrmion_number_computed: _,
            constellation_point: _,
            confidence,
        } = outcome
        {
            self.history.push(outcome.clone());
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }

            // Collect skyrmion numbers from recent signal detections.
            let skyrmion_numbers: Vec<i32> = self
                .history
                .iter()
                .filter_map(|o| match o {
                    DetectionOutcome::Signal {
                        signal,
                        ..
                    } => Some(signal.skyrmion_number),
                    _ => None,
                })
                .collect();

            let filtered_q = if skyrmion_numbers.is_empty() {
                signal.skyrmion_number
            } else {
                // Mode of recent skyrmion numbers.
                let mut counts: std::collections::HashMap<i32, u32> =
                    std::collections::HashMap::new();
                for &q in &skyrmion_numbers {
                    *counts.entry(q).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|_, &v| v)
                    .map(|(k, _)| k)
                    .unwrap_or(signal.skyrmion_number)
            };

            let filtered_confidence = confidence
                * (1.0 - self.smoothing_factor)
                + self.smoothing_factor
                    * (skyrmion_numbers.iter().map(|_| 1.0f64).sum::<f64>()
                        / (skyrmion_numbers.len() as f64))
                    .clamp(0.0, 1.0);

            let stability = skyrmion_numbers.iter().filter(|&&q| q == filtered_q).count() as f64
                / skyrmion_numbers.len() as f64;

            FilteredDetection {
                skyrmion_number: filtered_q,
                confidence: filtered_confidence.clamp(0.0, 1.0),
                frame_count: skyrmion_numbers.len() as u32,
                stability,
            }
        } else {
            // Non-signal outcome: push to history but return low-confidence filter.
            self.history.push(outcome);
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }
            FilteredDetection {
                skyrmion_number: 0,
                confidence: 0.0,
                frame_count: self.history.len() as u32,
                stability: 0.0,
            }
        }
    }

    /// Returns the number of recent signal detections.
    pub fn recent_signal_count(&self) -> u32 {
        self.history
            .iter()
            .filter(|o| matches!(o, DetectionOutcome::Signal { .. }))
            .count() as u32
    }

    /// Reset the filter history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update filter parameters.
    pub fn update_params(&mut self, window_size: usize, smoothing_factor: f64) {
        self.window_size = window_size.max(1);
        self.smoothing_factor = smoothing_factor.clamp(0.0, 1.0);
    }
}

impl Default for TemporalFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Detection confidence calibration
// ---------------------------------------------------------------------------

/// Calibrates detection confidence based on historical performance.
pub struct DetectionConfidenceCalibrator {
    /// Calibration bins.
    bins: Vec<CalibrationEntry>,
    /// Number of bins.
    pub num_bins: usize,
}

/// A single calibration entry.
#[derive(Debug, Clone, Default)]
struct CalibrationEntry {
    /// Lower confidence bound.
    pub lower: f64,
    /// Upper confidence bound.
    pub upper: f64,
    /// Number of predictions.
    pub count: u32,
    /// Number of correct predictions.
    pub correct: u32,
}

impl DetectionConfidenceCalibrator {
    /// Create a new calibrator.
    pub fn new(num_bins: usize) -> Self {
        Self {
            bins: (0..num_bins)
                .map(|i| CalibrationEntry {
                    lower: i as f64 / num_bins as f64,
                    upper: (i + 1) as f64 / num_bins as f64,
                    count: 0,
                    correct: 0,
                })
                .collect(),
            num_bins,
        }
    }

    /// Record a detection result for calibration.
    pub fn record(&mut self, predicted_confidence: f64, was_correct: bool) {
        let bin_idx = ((predicted_confidence.clamp(0.0, 1.0)) * self.num_bins as f64) as usize;
        let bin_idx = bin_idx.min(self.num_bins - 1);
        if let Some(bin) = self.bins.get_mut(bin_idx) {
            bin.count += 1;
            if was_correct {
                bin.correct += 1;
            }
        }
    }

    /// Calibrate a raw confidence value.
    pub fn calibrate(&self, raw_confidence: f64) -> f64 {
        let bin_idx = ((raw_confidence.clamp(0.0, 1.0)) * self.num_bins as f64) as usize;
        let bin_idx = bin_idx.min(self.num_bins - 1);
        let bin = &self.bins[bin_idx];
        if bin.count > 0 {
            bin.correct as f64 / bin.count as f64
        } else {
            raw_confidence
        }
    }

    /// Returns the expected calibration error.
    pub fn ece(&self) -> f64 {
        let total: u32 = self.bins.iter().map(|b| b.count).sum();
        if total == 0 {
            return 0.0;
        }
        self.bins
            .iter()
            .map(|b| {
                if b.count == 0 {
                    0.0
                } else {
                    let accuracy = b.correct as f64 / b.count as f64;
                    let conf = (b.lower + b.upper) / 2.0;
                    (b.count as f64 / total as f64) * (accuracy - conf).abs()
                }
            })
            .sum()
    }
}

impl Default for DetectionConfidenceCalibrator {
    fn default() -> Self {
        Self::new(10)
    }
}

// ---------------------------------------------------------------------------
// DetectionPipeline tests
// ---------------------------------------------------------------------------

#[test]
fn detection_pipeline_new() {
    let pipeline = DetectionPipeline::new();
    assert_eq!(pipeline.stats().total_detections, 0);
    assert_eq!(pipeline.stats().successful_detections, 0);
    assert_eq!(pipeline.stats().failed_detections, 0);
    assert!(pipeline.continue_on_failure);
}

#[test]
fn detection_pipeline_run_uniform_field() {
    let mut pipeline = DetectionPipeline::new();
    let mut detector = SignalDetector::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = pipeline.run(&mut detector, &field);
    assert!(result.success);
    assert_eq!(result.stage_results.len(), 6);
    assert!(result.all_stages_succeeded());
}

#[test]
fn detection_pipeline_stats_tracking() {
    let mut pipeline = DetectionPipeline::new();
    let mut detector = SignalDetector::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let _ = pipeline.run(&mut detector, &field);
    let stats = pipeline.stats();
    assert_eq!(stats.total_detections, 1);
    assert_eq!(stats.successful_detections, 1);
    assert_eq!(stats.failed_detections, 0);
    assert!(stats.avg_processing_time_us > 0.0);
}

#[test]
fn detection_pipeline_continue_on_failure() {
    let mut pipeline = DetectionPipeline::new();
    pipeline.continue_on_failure = true;
    assert!(pipeline.continue_on_failure);
    pipeline.continue_on_failure = false;
    assert!(!pipeline.continue_on_failure);
}

#[test]
fn detection_pipeline_stage_config() {
    let mut pipeline = DetectionPipeline::new();
    assert!(pipeline.stage_config(DetectionStage::Preprocessing).is_some());
    let config = pipeline.stage_config(DetectionStage::Preprocessing).unwrap();
    assert!(config.enabled);
    assert_eq!(config.timeout_ms, 1000);
}

#[test]
fn detection_pipeline_update_stage_config() {
    let mut pipeline = DetectionPipeline::new();
    let new_config = StageConfig {
        enabled: false,
        timeout_ms: 500,
        retry_count: 1,
        quality_threshold: 0.8,
    };
    pipeline.update_stage_config(DetectionStage::Preprocessing, new_config.clone());
    let updated = pipeline.stage_config(DetectionStage::Preprocessing).unwrap();
    assert!(!updated.enabled);
    assert_eq!(updated.timeout_ms, 500);
    assert_eq!(updated.retry_count, 1);
    assert_eq!(updated.quality_threshold, 0.8);
}

#[test]
fn detection_pipeline_set_stage_enabled() {
    let mut pipeline = DetectionPipeline::new();
    pipeline.set_stage_enabled(DetectionStage::Preprocessing, false);
    assert!(!pipeline.stage_config(DetectionStage::Preprocessing).unwrap().enabled);
    pipeline.set_stage_enabled(DetectionStage::Preprocessing, true);
    assert!(pipeline.stage_config(DetectionStage::Preprocessing).unwrap().enabled);
}

#[test]
fn detection_pipeline_reset_stats() {
    let mut pipeline = DetectionPipeline::new();
    let mut detector = SignalDetector::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let _ = pipeline.run(&mut detector, &field);
    assert_eq!(pipeline.stats().total_detections, 1);
    pipeline.reset_stats();
    assert_eq!(pipeline.stats().total_detections, 0);
    assert_eq!(pipeline.stats().successful_detections, 0);
}

// ---------------------------------------------------------------------------
// SignalQualityAssessment tests
// ---------------------------------------------------------------------------

#[test]
fn signal_quality_assessment_new() {
    let assessment = SignalQualityAssessment::new(0.9, 0.8, 0.7, 0.6, 0.5, 0.4);
    assert!((assessment.overall_score - 0.715).abs() < 0.001);
    assert!(assessment.passes_gate);
    assert!(assessment.is_high_quality());
}

#[test]
fn signal_quality_assessment_low_quality() {
    let assessment = SignalQualityAssessment::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
    assert!(!assessment.passes_gate);
    assert!(!assessment.is_high_quality());
    assert_eq!(assessment.grade, QualityGrade::F);
}

#[test]
fn signal_quality_assessment_grade_a() {
    let assessment = SignalQualityAssessment::new(0.95, 0.95, 0.95, 0.95, 0.95, 0.95);
    assert_eq!(assessment.grade, QualityGrade::A);
    assert!(assessment.is_high_quality());
}

#[test]
fn signal_quality_assessment_grade_f() {
    let assessment = SignalQualityAssessment::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
    assert_eq!(assessment.grade, QualityGrade::F);
    assert!(!assessment.is_high_quality());
}

#[test]
fn signal_quality_assessment_bottleneck() {
    let assessment = SignalQualityAssessment::new(0.3, 0.9, 0.9, 0.9, 0.9, 0.9);
    assert!((assessment.bottleneck() - 0.3).abs() < 0.001);
}

#[test]
fn quality_assessor_default() {
    let assessor = QualityAssessor::default();
    assert!((assessor.min_overall_score - 0.6).abs() < 0.001);
    assert!((assessor.min_topological_stability - 0.7).abs() < 0.001);
    assert!((assessor.min_snr_quality - 0.5).abs() < 0.001);
    assert!(assessor.use_temporal_stability);
}

#[test]
fn quality_assessor_assess_signal() {
    let assessor = QualityAssessor::default();
    let outcome = DetectionOutcome::Signal {
        signal: TopologicalSignal::new(3).with_confidence(0.9),
        skyrmion_number_computed: 3.0,
        constellation_point: 3,
        confidence: 0.9,
    };
    let assessment = assessor.assess(&outcome).unwrap();
    assert!(assessment.passes_gate);
    assert!(assessment.is_high_quality());
}

#[test]
fn quality_assessor_assess_noise() {
    let assessor = QualityAssessor::default();
    let outcome = DetectionOutcome::Noise {
        noise_floor: 0.1,
        max_skyrmion_density: 0.01,
    };
    let assessment = assessor.assess(&outcome).unwrap();
    assert!(!assessment.passes_gate);
    assert!(!assessment.is_high_quality());
}

#[test]
fn quality_assessor_assess_interference() {
    let assessor = QualityAssessor::default();
    let outcome = DetectionOutcome::Interference {
        interference_type: "test".into(),
        severity: 0.8,
    };
    let assessment = assessor.assess(&outcome).unwrap();
    assert!(!assessment.passes_gate);
}

#[test]
fn quality_assessor_assess_false_positive() {
    let assessor = QualityAssessor::default();
    let outcome = DetectionOutcome::FalsePositive {
        q_candidate: 5,
        reason: "test".into(),
    };
    let assessment = assessor.assess(&outcome).unwrap();
    assert!(!assessment.passes_gate);
}

#[test]
fn quality_assessor_update_thresholds() {
    let mut assessor = QualityAssessor::default();
    assessor.update_thresholds(0.8, 0.8, 0.8);
    assert!((assessor.min_overall_score - 0.8).abs() < 0.001);
    assert!((assessor.min_topological_stability - 0.8).abs() < 0.001);
    assert!((assessor.min_snr_quality - 0.8).abs() < 0.001);
}

// ---------------------------------------------------------------------------
// ConstellationAnalyzer tests
// ---------------------------------------------------------------------------

#[test]
fn constellation_analyzer_default() {
    let analyzer = ConstellationAnalyzer::default();
    assert!((analyzer.min_reliable_separation - 2.0).abs() < 0.001);
    assert_eq!(analyzer.max_order, 8);
}

#[test]
fn constellation_analyzer_binary() {
    let analyzer = ConstellationAnalyzer::default();
    let constellation = Constellation::binary().unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.point_count, 2);
    assert_eq!(analysis.min_separation, 2.0);
    assert_eq!(analysis.avg_separation, 2.0);
    assert!(analysis.well_separated);
    assert_eq!(analysis.recommended_order, 1);
}

#[test]
fn constellation_analyzer_quaternary() {
    let analyzer = ConstellationAnalyzer::default();
    let constellation = Constellation::quaternary().unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.point_count, 4);
    assert_eq!(analysis.min_separation, 2.0);
    assert!(analysis.well_separated);
    assert_eq!(analysis.recommended_order, 2);
}

#[test]
fn constellation_analyzer_octal() {
    let analyzer = ConstellationAnalyzer::default();
    let constellation = Constellation::octal().unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.point_count, 8);
    assert_eq!(analysis.min_separation, 2.0);
    assert!(analysis.well_separated);
    assert_eq!(analysis.recommended_order, 3);
}

#[test]
fn constellation_analyzer_close_points() {
    let analyzer = ConstellationAnalyzer::default();
    let constellation = Constellation::new(vec![0, 1, 2]).unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.min_separation, 1.0);
    assert!(!analysis.well_separated);
    assert_eq!(analysis.recommended_order, 1);
}

#[test]
fn constellation_analyzer_supports_order() {
    let analyzer = ConstellationAnalyzer::default();
    let constellation = Constellation::octal().unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert!(analyzer.supports_order(&analysis, 1));
    assert!(analyzer.supports_order(&analysis, 2));
    assert!(analyzer.supports_order(&analysis, 3));
}

#[test]
fn constellation_analyzer_update_threshold() {
    let mut analyzer = ConstellationAnalyzer::default();
    analyzer.update_threshold(5.0);
    assert!((analyzer.min_reliable_separation - 5.0).abs() < 0.001);
    analyzer.update_threshold(0.0);
    assert!((analyzer.min_reliable_separation - 0.0).abs() < 0.001);
}

// ---------------------------------------------------------------------------
// FrequencyDomainAnalyzer tests
// ---------------------------------------------------------------------------

#[test]
fn frequency_domain_analyzer_default() {
    let analyzer = FrequencyDomainAnalyzer::default();
    assert_eq!(analyzer.fft_size, 256);
    assert!((analyzer.frequency_resolution - 1.0).abs() < 0.001);
}

#[test]
fn frequency_domain_analyzer_uniform_field() {
    let analyzer = FrequencyDomainAnalyzer::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert_eq!(result.dominant_frequency, 0.0);
    assert_eq!(result.bandwidth, 0.0);
}

#[test]
fn frequency_domain_analyzer_varying_field() {
    let analyzer = FrequencyDomainAnalyzer::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [(x as f64), 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert!(result.dominant_frequency >= 0.0);
    assert!(result.spectral_centroid >= 0.0);
}

#[test]
fn frequency_domain_analyzer_is_suitable() {
    let analyzer = FrequencyDomainAnalyzer::default();
    let analysis = FrequencyAnalysis {
        dominant_frequency: 10.0,
        bandwidth: 10.0,
        spectral_centroid: 10.0,
        spectral_snr: 25.0,
        clean_spectrum: true,
    };
    assert!(analyzer.is_suitable(&analysis));

    let analysis2 = FrequencyAnalysis {
        dominant_frequency: 10.0,
        bandwidth: 10.0,
        spectral_centroid: 10.0,
        spectral_snr: 5.0,
        clean_spectrum: true,
    };
    assert!(!analyzer.is_suitable(&analysis2));
}

#[test]
fn frequency_domain_analyzer_update_fft_size() {
    let mut analyzer = FrequencyDomainAnalyzer::default();
    analyzer.update_fft_size(512);
    assert_eq!(analyzer.fft_size, 512);
    analyzer.update_fft_size(10);
    assert_eq!(analyzer.fft_size, 64); // minimum enforced
}

// ---------------------------------------------------------------------------
// MultiResolutionAnalyzer tests
// ---------------------------------------------------------------------------

#[test]
fn multi_resolution_analyzer_default() {
    let analyzer = MultiResolutionAnalyzer::default();
    assert_eq!(analyzer.num_levels, 3);
    assert_eq!(analyzer.downsample_factor, 2);
}

#[test]
fn multi_resolution_analyzer_uniform_field() {
    let analyzer = MultiResolutionAnalyzer::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert_eq!(result.level_results.len(), 3);
    assert_eq!(result.consensus_number, 0);
    assert!(result.level_agreement);
}

#[test]
fn multi_resolution_analyzer_level_agreement() {
    let analyzer = MultiResolutionAnalyzer::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert!(analyzer.is_reliable(&result));
}

#[test]
fn multi_resolution_analyzer_update_levels() {
    let mut analyzer = MultiResolutionAnalyzer::default();
    analyzer.update_levels(5);
    assert_eq!(analyzer.num_levels, 5);
    analyzer.update_levels(1);
    assert_eq!(analyzer.num_levels, 1);
    analyzer.update_levels(100);
    assert_eq!(analyzer.num_levels, 8); // max enforced
}

// ---------------------------------------------------------------------------
// TemporalFilter tests
// ---------------------------------------------------------------------------

#[test]
fn temporal_filter_default() {
    let filter = TemporalFilter::default();
    assert_eq!(filter.window_size, 5);
    assert!((filter.smoothing_factor - 0.3).abs() < 0.001);
}

#[test]
fn temporal_filter_signal_detection() {
    let mut filter = TemporalFilter::default();
    let outcome = DetectionOutcome::Signal {
        signal: TopologicalSignal::new(1).with_confidence(0.9),
        skyrmion_number_computed: 1.0,
        constellation_point: 1,
        confidence: 0.9,
    };
    let result = filter.filter(outcome);
    assert_eq!(result.skyrmion_number, 1);
    assert!(result.confidence > 0.0);
    assert_eq!(result.frame_count, 1);
}

#[test]
fn temporal_filter_noise_detection() {
    let mut filter = TemporalFilter::default();
    let outcome = DetectionOutcome::Noise {
        noise_floor: 0.1,
        max_skyrmion_density: 0.01,
    };
    let result = filter.filter(outcome);
    assert_eq!(result.skyrmion_number, 0);
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn temporal_filter_stability() {
    let mut filter = TemporalFilter::default();
    // Send multiple detections with same skyrmion number.
    for _ in 0..5 {
        let outcome = DetectionOutcome::Signal {
            signal: TopologicalSignal::new(3).with_confidence(0.9),
            skyrmion_number_computed: 3.0,
            constellation_point: 3,
            confidence: 0.9,
        };
        let result = filter.filter(outcome);
        assert_eq!(result.skyrmion_number, 3);
    }
    assert!(filter.recent_signal_count() > 0);
}

#[test]
fn temporal_filter_reset() {
    let mut filter = TemporalFilter::default();
    let outcome = DetectionOutcome::Signal {
        signal: TopologicalSignal::new(1).with_confidence(0.9),
        skyrmion_number_computed: 1.0,
        constellation_point: 1,
        confidence: 0.9,
    };
    filter.filter(outcome);
    assert_eq!(filter.recent_signal_count(), 1);
    filter.reset();
    assert_eq!(filter.recent_signal_count(), 0);
}

#[test]
fn temporal_filter_update_params() {
    let mut filter = TemporalFilter::default();
    filter.update_params(10, 0.5);
    assert_eq!(filter.window_size, 10);
    assert!((filter.smoothing_factor - 0.5).abs() < 0.001);
    filter.update_params(1, 0.0);
    assert_eq!(filter.window_size, 1);
}

// ---------------------------------------------------------------------------
// DetectionConfidenceCalibrator tests
// ---------------------------------------------------------------------------

#[test]
fn confidence_calibrator_default() {
    let calibrator = DetectionConfidenceCalibrator::default();
    assert_eq!(calibrator.num_bins, 10);
    assert_eq!(calibrator.ece(), 0.0);
}

#[test]
fn confidence_calibrator_record() {
    let mut calibrator = DetectionConfidenceCalibrator::new(5);
    calibrator.record(0.8, true);
    calibrator.record(0.8, true);
    calibrator.record(0.8, false);
    let calibrated = calibrator.calibrate(0.8);
    assert!((calibrated - 0.666667).abs() < 0.01);
}

#[test]
fn confidence_calibrator_ece() {
    let mut calibrator = DetectionConfidenceCalibrator::new(2);
    // Perfect calibration: all predictions in bin 0 are correct, all in bin 1 are correct.
    calibrator.record(0.1, true);
    calibrator.record(0.9, true);
    let ece = calibrator.ece();
    assert!(ece < 0.1);
}

#[test]
fn confidence_calibrator_no_records() {
    let calibrator = DetectionConfidenceCalibrator::new(5);
    let calibrated = calibrator.calibrate(0.5);
    assert!((calibrated - 0.5).abs() < 0.001);
    assert_eq!(calibrator.ece(), 0.0);
}

// ---------------------------------------------------------------------------
// PipelineResult tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_result_avg_quality() {
    let result = PipelineResult {
        success: true,
        stage_results: vec![
            StageResult::success(DetectionStage::Preprocessing, 10.0, 0.8),
            StageResult::success(DetectionStage::IntensityMasking, 20.0, 0.9),
            StageResult::failure(DetectionStage::SkyrmionComputation, 5.0, "test error".into()),
        ],
        total_processing_time_us: 35.0,
        timestamp: 1000,
    };
    assert!((result.avg_quality() - (0.8 + 0.9 + 0.0) / 3.0).abs() < 0.001);
    assert!(!result.all_stages_succeeded());
    assert_eq!(result.first_error(), Some("test error"));
}

#[test]
fn pipeline_result_all_success() {
    let result = PipelineResult {
        success: true,
        stage_results: vec![
            StageResult::success(DetectionStage::Preprocessing, 10.0, 0.8),
            StageResult::success(DetectionStage::IntensityMasking, 20.0, 0.9),
        ],
        total_processing_time_us: 30.0,
        timestamp: 1000,
    };
    assert!(result.all_stages_succeeded());
    assert_eq!(result.first_error(), None);
}

#[test]
fn stage_result_success_failure() {
    let success = StageResult::success(DetectionStage::Preprocessing, 10.0, 0.8);
    assert!(success.success);
    assert_eq!(success.quality_score, 0.8);
    assert!(success.error.is_none());

    let failure = StageResult::failure(DetectionStage::SkyrmionComputation, 5.0, "compute failed".into());
    assert!(!failure.success);
    assert_eq!(failure.error.as_deref(), Some("compute failed"));
}

#[test]
fn stage_config_default() {
    let config = StageConfig::default();
    assert!(config.enabled);
    assert_eq!(config.timeout_ms, 1000);
    assert_eq!(config.retry_count, 3);
    assert!((config.quality_threshold - 0.5).abs() < 0.001);
}

#[test]
fn pipeline_stats_default() {
    let stats = PipelineStats::default();
    assert_eq!(stats.total_detections, 0);
    assert_eq!(stats.successful_detections, 0);
    assert_eq!(stats.failed_detections, 0);
    assert_eq!(stats.avg_processing_time_us, 0.0);
}

// ---------------------------------------------------------------------------
// MultiResolutionAnalyzer additional tests
// ---------------------------------------------------------------------------

#[test]
fn multi_resolution_analyzer_consensus() {
    let analyzer = MultiResolutionAnalyzer::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert_eq!(result.consensus_number, 0);
    assert!(result.consensus_confidence > 0.0);
}

#[test]
fn multi_resolution_analyzer_different_levels() {
    let analyzer = MultiResolutionAnalyzer::new(5);
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert_eq!(result.level_results.len(), 5);
}

#[test]
fn frequency_analyzer_clean_spectrum() {
    let analyzer = FrequencyDomainAnalyzer::default();
    let analysis = FrequencyAnalysis {
        dominant_frequency: 5.0,
        bandwidth: 5.0,
        spectral_centroid: 5.0,
        spectral_snr: 30.0,
        clean_spectrum: true,
    };
    assert!(analyzer.is_suitable(&analysis));

    let analysis2 = FrequencyAnalysis {
        dominant_frequency: 5.0,
        bandwidth: 100.0,
        spectral_centroid: 5.0,
        spectral_snr: 30.0,
        clean_spectrum: false,
    };
    assert!(!analyzer.is_suitable(&analysis2));
}

#[test]
fn constellation_analysis_recommended_order_edge() {
    let analyzer = ConstellationAnalyzer::new();
    let constellation = Constellation::new(vec![0, 100]).unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.min_separation, 100.0);
    assert!(analysis.well_separated);
    assert_eq!(analysis.recommended_order, 8); // capped at max_order
}

#[test]
fn quality_components_all_fields() {
    let components = QualityComponents {
        topological_stability: 0.9,
        snr_quality: 0.8,
        spatial_coherence: 0.7,
        temporal_stability: 0.6,
        intensity_reliability: 0.5,
        classification_confidence: 0.4,
    };
    assert!((components.topological_stability - 0.9).abs() < 0.001);
    assert!((components.snr_quality - 0.8).abs() < 0.001);
    assert!((components.spatial_coherence - 0.7).abs() < 0.001);
    assert!((components.temporal_stability - 0.6).abs() < 0.001);
    assert!((components.intensity_reliability - 0.5).abs() < 0.001);
    assert!((components.classification_confidence - 0.4).abs() < 0.001);
}

#[test]
fn filtered_detection_default_fields() {
    let detection = FilteredDetection {
        skyrmion_number: 5,
        confidence: 0.9,
        frame_count: 10,
        stability: 0.8,
    };
    assert_eq!(detection.skyrmion_number, 5);
    assert!((detection.confidence - 0.9).abs() < 0.001);
    assert_eq!(detection.frame_count, 10);
    assert!((detection.stability - 0.8).abs() < 0.001);
}

#[test]
fn quality_grade_ordering() {
    assert!(QualityGrade::A as u8 == 0);
    assert!(QualityGrade::F as u8 == 5);
}

#[test]
fn quality_recommendation_variants() {
    assert_eq!(QualityRecommendation::Accept as u8, 0);
    assert_eq!(QualityRecommendation::AcceptWithCaution as u8, 1);
    assert_eq!(QualityRecommendation::Reacquire as u8, 2);
    assert_eq!(QualityRecommendation::Reject as u8, 3);
    assert_eq!(QualityRecommendation::ManualReview as u8, 4);
}

// ---------------------------------------------------------------------------
// ResolutionLevelResult tests
// ---------------------------------------------------------------------------

#[test]
fn resolution_level_result_fields() {
    let result = ResolutionLevelResult {
        level: 2,
        skyrmion_number: 3,
        confidence: 0.85,
        processing_time_us: 150.0,
    };
    assert_eq!(result.level, 2);
    assert_eq!(result.skyrmion_number, 3);
    assert!((result.confidence - 0.85).abs() < 0.001);
    assert!((result.processing_time_us - 150.0).abs() < 0.001);
}

#[test]
fn detection_pipeline_multiple_runs() {
    let mut pipeline = DetectionPipeline::new();
    let mut detector = SignalDetector::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    for _ in 0..5 {
        let _ = pipeline.run(&mut detector, &field);
    }
    let stats = pipeline.stats();
    assert_eq!(stats.total_detections, 5);
    assert_eq!(stats.successful_detections, 5);
}

#[test]
fn signal_quality_assessment_clone_debug() {
    let a = SignalQualityAssessment::new(0.9, 0.8, 0.7, 0.6, 0.5, 0.4);
    let b = a.clone();
    assert_eq!(a.overall_score, b.overall_score);
    let _debug = format!("{:?}", a.grade);
    let _debug2 = format!("{:?}", a.recommendation);
}

#[test]
fn quality_assessor_passes_gate() {
    let assessor = QualityAssessor::default();
    let high_quality = SignalQualityAssessment::new(0.9, 0.9, 0.9, 0.9, 0.9, 0.9);
    assert!(assessor.passes_gate(&high_quality));
    let low_quality = SignalQualityAssessment::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
    assert!(!assessor.passes_gate(&low_quality));
}

#[test]
fn constellation_analyzer_single_point() {
    let analyzer = ConstellationAnalyzer::default();
    let constellation = Constellation::new(vec![5]).unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.point_count, 1);
    assert_eq!(analysis.min_separation, 0.0);
    assert!(!analysis.well_separated);
}

#[test]
fn temporal_filter_mixed_outcomes() {
    let mut filter = TemporalFilter::default();
    let signal = DetectionOutcome::Signal {
        signal: TopologicalSignal::new(1).with_confidence(0.9),
        skyrmion_number_computed: 1.0,
        constellation_point: 1,
        confidence: 0.9,
    };
    let noise = DetectionOutcome::Noise {
        noise_floor: 0.1,
        max_skyrmion_density: 0.01,
    };
    filter.filter(signal);
    filter.filter(noise);
    filter.filter(signal);
    assert_eq!(filter.recent_signal_count(), 2);
}

#[test]
fn detection_confidence_calibrator_bin_edges() {
    let mut calibrator = DetectionConfidenceCalibrator::new(4);
    calibrator.record(0.0, true);
    calibrator.record(0.25, true);
    calibrator.record(0.5, true);
    calibrator.record(0.75, true);
    calibrator.record(1.0, true);
    let ece = calibrator.ece();
    assert!(ece < 0.2);
}

#[test]
fn frequency_analyzer_bandwidth_computation() {
    let analyzer = FrequencyDomainAnalyzer::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    // Create a field with a clear frequency pattern.
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [(x as f64 / 10.0).sin(), 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert!(result.dominant_frequency >= 0.0);
    assert!(result.bandwidth >= 0.0);
}

#[test]
fn pipeline_stats_update_on_run() {
    let mut pipeline = DetectionPipeline::new();
    let mut detector = SignalDetector::default();
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [0.0, 0.0, 0.0, 0.0];
        }
    }
    let _ = pipeline.run(&mut detector, &field);
    let stats = pipeline.stats();
    assert_eq!(stats.total_detections, 1);
    // All-dark field may fail some stages.
    assert!(stats.successful_detections + stats.failed_detections == 1);
}

#[test]
fn detection_pipeline_custom_config() {
    let mut configs = std::collections::HashMap::new();
    configs.insert(
        DetectionStage::Preprocessing,
        StageConfig {
            enabled: false,
            timeout_ms: 2000,
            retry_count: 5,
            quality_threshold: 0.9,
        },
    );
    let pipeline = DetectionPipeline::with_config(configs);
    assert!(!pipeline.stage_config(DetectionStage::Preprocessing).unwrap().enabled);
    assert_eq!(pipeline.stage_config(DetectionStage::Preprocessing).unwrap().timeout_ms, 2000);
}

#[test]
fn multi_resolution_analyzer_single_level() {
    let analyzer = MultiResolutionAnalyzer::new(1);
    let mut field = [[[0.0f64; 4]; 256]; 256];
    for y in 0..256 {
        for x in 0..256 {
            field[y][x] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    let result = analyzer.analyze(&field);
    assert_eq!(result.level_results.len(), 1);
    assert!(result.level_agreement);
}

#[test]
fn constellation_analyzer_large_separation() {
    let analyzer = ConstellationAnalyzer::new();
    let constellation = Constellation::new(vec![-100, 100]).unwrap();
    let analysis = analyzer.analyze(&constellation);
    assert_eq!(analysis.min_separation, 200.0);
    assert!(analysis.well_separated);
    assert_eq!(analysis.recommended_order, 8);
}

#[test]
fn frequency_analyzer_empty_field() {
    let analyzer = FrequencyDomainAnalyzer::default();
    let field = [[[0.0f64; 4]; 256]; 256];
    let result = analyzer.analyze(&field);
    assert_eq!(result.dominant_frequency, 0.0);
    assert_eq!(result.clean_spectrum, false);
}

#[test]
fn quality_assessor_assess_corrupted() {
    let assessor = QualityAssessor::default();
    let outcome = DetectionOutcome::Corrupted {
        reason: "test".into(),
        integrity_score: 0.3,
        partial_q: Some(2),
    };
    let assessment = assessor.assess(&outcome).unwrap();
    assert!(!assessment.passes_gate);
}

#[test]
fn quality_assessor_assess_insufficient_data() {
    let assessor = QualityAssessor::default();
    let outcome = DetectionOutcome::InsufficientData {
        missing: "test".into(),
    };
    let assessment = assessor.assess(&outcome).unwrap();
    assert!(!assessment.passes_gate);
}

#[test]
fn temporal_filter_empty_history() {
    let filter = TemporalFilter::default();
    assert_eq!(filter.recent_signal_count(), 0);
}

#[test]
fn detection_confidence_calibrator_record_many() {
    let mut calibrator = DetectionConfidenceCalibrator::new(10);
    for i in 0..100 {
        calibrator.record(i as f64 / 100.0, i % 2 == 0);
    }
    let ece = calibrator.ece();
    assert!(ece >= 0.0);
    assert!(ece < 1.0);
}