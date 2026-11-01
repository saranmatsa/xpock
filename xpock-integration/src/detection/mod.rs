//! Topological signal detection module.
//!
//! Implements the core detection pipeline for distinguishing valid topological
//! signals (skyrmion number modulation) from noise, interference, false positives,
//! and corrupted measurements.
//!
//! Pipeline stages:
//! 1. Stokes polarimetry acquisition (via [`crate::receiver`])
//! 2. Stokes vector field reconstruction via demodulation matrix
//! 3. Intensity-based masking (suppresses unreliable low-intensity regions)
//! 4. Skyrmion number computation Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy
//! 5. Classification: Signal vs Noise vs Interference vs FalsePositive vs Corrupted
//! 6. Constellation matching against valid skyrmion number points
//!
//! All types are designed for HX-241 carrier-agnostic integration — no assumptions
//! about the physical medium.

pub mod signal_detector;
pub mod skyrmion_number;
pub mod noise_classifier;
pub mod threshold_engine;
pub mod intensity_masking;
pub mod generalized_skyrmion;
pub mod crosstalk_matrix;
pub mod constellation_integration;
pub mod demodulator;
pub mod recovery;
pub mod validators;

// Re-export key types for convenient access.
pub use signal_detector::{SignalDetector, DetectionConfig, DetectionStatistics};
pub use skyrmion_number::{SkyrmionNumber, SkyrmionDensity, SkyrmionField};
pub use noise_classifier::{NoiseClassifier, NoiseClassification, InterferenceType};
pub use threshold_engine::{ThresholdEngine, AdaptiveThreshold, ThresholdPolicy};
pub use intensity_masking::{IntensityMask, MaskingConfig, MaskedRegion};
pub use generalized_skyrmion::{GeneralizedSkyrmionNumber, SkyrmionTuple};
pub use crosstalk_matrix::{CrosstalkMatrix, CrosstalkAnalysis};
pub use constellation_integration::{ConstellationMatcher, MatchedPoint};
pub use demodulator::{SkyrmionDemodulator, DemodulationResult, DemodulationConfig};
pub use recovery::{DetectionRecovery, RecoveryStrategy, RecoveryOutcome};

// Re-export validator types.
pub use validators::{
    TopologicalValidator, BoundaryValidator, NumericalValidator, ValidationResult,
};

// ---------------------------------------------------------------------------
// Detection pipeline configuration
// ---------------------------------------------------------------------------

/// Configuration for the topological signal detection pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionConfig {
    /// Skyrmion number computation grid resolution (pixels per dimension).
    pub grid_resolution: usize,
    /// Intensity masking threshold (fraction of max intensity).
    pub intensity_threshold: f64,
    /// Maximum allowed crosstalk between constellation points (0.0 to 1.0).
    pub max_crosstalk: f64,
    /// Minimum confidence for accepting a detection (0.0 to 1.0).
    pub min_confidence: f64,
    /// Constellation to match against.
    pub constellation: crate::types::Constellation,
    /// Enable generalized skyrmion number (Z^k tuple) detection.
    pub enable_generalized: bool,
    /// Number of generalized components when enabled.
    pub generalized_components: usize,
    /// Stokes demodulation matrix calibration data (6 measurements).
    pub stokes_calibration: Option<[[f64; 4]; 6]>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 256,
            intensity_threshold: 0.1,
            max_crosstalk: 0.15,
            min_confidence: 0.7,
            constellation: crate::types::Constellation::binary().expect("binary constellation"),
            enable_generalized: false,
            generalized_components: 1,
            stokes_calibration: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Detection statistics for observability
// ---------------------------------------------------------------------------

/// Statistics collected during detection for monitoring and debugging.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectionStatistics {
    /// Total number of detection attempts.
    pub total_attempts: u64,
    /// Number of valid signals detected.
    pub signals_detected: u64,
    /// Number of noise classifications.
    pub noise_detected: u64,
    /// Number of interference classifications.
    pub interference_detected: u64,
    /// Number of false positives rejected.
    pub false_positives_rejected: u64,
    /// Number of corrupted measurements.
    pub corrupted_measurements: u64,
    /// Average skyrmion number confidence.
    pub average_confidence: f64,
    /// Average computation time in microseconds.
    pub avg_compute_time_us: f64,
    /// Detection errors by type.
    pub error_counts: std::collections::HashMap<String, u64>,
}

impl DetectionStatistics {
    /// Returns the signal detection rate (signals / total attempts).
    pub fn signal_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            return 0.0;
        }
        self.signals_detected as f64 / self.total_attempts as f64
    }

    /// Returns the false positive rejection rate.
    pub fn false_positive_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            return 0.0;
        }
        self.false_positives_rejected as f64 / self.total_attempts as f64
    }

    /// Records a detection outcome.
    pub fn record_outcome(&mut self, outcome: &crate::detection::signal_detector::DetectionOutcome, compute_time_us: f64) {
        self.total_attempts += 1;
        match outcome {
            crate::detection::signal_detector::DetectionOutcome::Signal { .. } => {
                self.signals_detected += 1;
            }
            crate::detection::signal_detector::DetectionOutcome::Noise => {
                self.noise_detected += 1;
            }
            crate::detection::signal_detector::DetectionOutcome::Interference { .. } => {
                self.interference_detected += 1;
            }
            crate::detection::signal_detector::DetectionOutcome::FalsePositive { .. } => {
                self.false_positives_rejected += 1;
            }
            crate::detection::signal_detector::DetectionOutcome::Corrupted { .. } => {
                self.corrupted_measurements += 1;
            }
        }
        // Running average of compute time.
        self.avg_compute_time_us =
            (self.avg_compute_time_us * (self.total_attempts - 1) as f64 + compute_time_us)
                / self.total_attempts as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_config_default_is_valid() {
        let config = DetectionConfig::default();
        assert_eq!(config.grid_resolution, 256);
        assert_eq!(config.intensity_threshold, 0.1);
        assert_eq!(config.max_crosstalk, 0.15);
        assert_eq!(config.min_confidence, 0.7);
        assert!(!config.enable_generalized);
        assert_eq!(config.generalized_components, 1);
    }

    #[test]
    fn detection_statistics_signal_rate() {
        let mut stats = DetectionStatistics::default();
        assert_eq!(stats.signal_rate(), 0.0);
        assert_eq!(stats.false_positive_rate(), 0.0);
    }

    #[test]
    fn constellation_default_is_binary() {
        let config = DetectionConfig::default();
        assert!(config.constellation.contains(-1));
        assert!(config.constellation.contains(1));
        assert!(!config.constellation.contains(0));
    }
}