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
// Default implementation for SignalDetector
// ---------------------------------------------------------------------------

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