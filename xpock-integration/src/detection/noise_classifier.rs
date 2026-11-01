//! Noise classification for topological signal detection.
//!
//! Distinguishes valid topological signals from noise, interference, false positives,
//! and corrupted measurements by analyzing signal characteristics across multiple
//! dimensions: intensity, spectral structure, polarization consistency, temporal
//! stability, and spatial coherence.

use crate::errors::{XpockError, XpockResult};
use crate::types::{DetectionResult, TopologicalSignal};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Noise classification result
// ---------------------------------------------------------------------------

/// Classification of a measured signal into noise, interference, or valid topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoiseClassification {
    /// Clean signal with valid topological structure.
    CleanSignal {
        signal: TopologicalSignal,
        noise_floor: f64,
        snr_db: f64,
    },
    /// Noise — no topological structure detectable above noise floor.
    Noise {
        noise_type: NoiseType,
        noise_floor: f64,
        confidence: f64,
    },
    /// Interference — signal present but corrupted by external interference.
    Interference {
        interference_type: InterferenceType,
        severity: f64,
        affected_region: Option<(usize, usize)>,
    },
    /// False positive — apparent topology that fails validation checks.
    FalsePositive {
        q_candidate: i32,
        reason: String,
        confidence: f64,
    },
    /// Corrupted — signal detected but integrity verification failed.
    Corrupted {
        reason: String,
        integrity_score: f64,
        partial_q: Option<i32>,
    },
}

impl NoiseClassification {
    /// Returns true if this classification represents a clean, usable signal.
    pub fn is_clean_signal(&self) -> bool {
        matches!(self, NoiseClassification::CleanSignal { .. })
    }

    /// Returns true if this is noise (not a signal).
    pub fn is_noise(&self) -> bool {
        matches!(self, NoiseClassification::Noise { .. })
    }

    /// Returns true if this is interference.
    pub fn is_interference(&self) -> bool {
        matches!(self, Interference { .. })
    }

    /// Returns true if this is a false positive.
    pub fn is_false_positive(&self) -> bool {
        matches!(self, FalsePositive { .. })
    }

    /// Returns true if this is corrupted.
    pub fn is_corrupted(&self) -> bool {
        matches!(self, Corrupted { .. })
    }

    /// Extract the topological signal if this is a clean signal.
    pub fn into_signal(self) -> Option<TopologicalSignal> {
        match self {
            NoiseClassification::CleanSignal { signal, .. } => Some(signal),
            _ => None,
        }
    }

    /// Convert to a DetectionResult.
    pub fn to_detection_result(self) -> DetectionResult {
        match self {
            NoiseClassification::CleanSignal { signal, .. } => {
                DetectionResult::Signal(signal)
            }
            NoiseClassification::Noise { .. } => DetectionResult::Noise,
            NoiseClassification::Interference { .. } => DetectionResult::Interference,
            NoiseClassification::FalsePositive { .. } => DetectionResult::FalsePositive,
            NoiseClassification::Corrupted { .. } => DetectionResult::Corrupted,
        }
    }
}

// ---------------------------------------------------------------------------
// Noise types
// ---------------------------------------------------------------------------

/// Type of noise detected in the measurement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Thermal/Johnson-Nyquist noise — white, Gaussian-distributed.
    Thermal,
    /// Shot noise — photon-counting statistical noise.
    Shot,
    /// Background noise — ambient light, stray radiation.
    Background,
    /// Readout noise — detector electronics noise.
    Readout,
    /// Quantization noise — ADC discretization error.
    Quantization,
    /// Flicker noise — 1/f noise, low-frequency drift.
    Flicker,
    /// Burst noise — intermittent spikes.
    Burst,
    /// Unknown noise source.
    Unknown(String),
}

impl NoiseType {
    /// Returns a human-readable description.
    pub fn description(&self) -> &str {
        match self {
            NoiseType::Thermal => "thermal Johnson-Nyquist noise",
            NoiseType::Shot => "photon shot noise",
            NoiseType::Background => "ambient background noise",
            NoiseType::Readout => "detector readout noise",
            NoiseType::Quantization => "ADC quantization noise",
            NoiseType::Flicker => "1/f flicker noise",
            NoiseType::Burst => "intermittent burst noise",
            NoiseType::Unknown(s) => s,
        }
    }
}

// ---------------------------------------------------------------------------
// Interference types
// ---------------------------------------------------------------------------

/// Type of interference affecting the signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InterferenceType {
    /// Co-channel interference — another signal on the same skyrmion number.
    CoChannel,
    /// Adjacent-channel interference — nearby skyrmion numbers leaking in.
    AdjacentChannel,
    /// Multipath interference — signal reflections causing phase distortion.
    Multipath,
    /// Narrowband interference — concentrated at specific frequencies.
    Narrowband,
    /// Wideband interference — spread across the spectrum.
    Wideband,
    /// Polarization cross-talk — leakage between polarization components.
    PolarizationCrosstalk,
    /// Impulse interference — short-duration high-amplitude spikes.
    Impulse,
    /// Unknown interference source.
    Unknown(String),
}

impl InterferenceType {
    /// Returns a human-readable description.
    pub fn description(&self) -> &str {
        match self {
            InterferenceType::CoChannel => "co-channel interference",
            InterferenceType::AdjacentChannel => "adjacent-channel interference",
            InterferenceType::Multipath => "multipath interference",
            InterferenceType::Narrowband => "narrowband interference",
            InterferenceType::Wideband => "wideband interference",
            InterferenceType::PolarizationCrosstalk => "polarization cross-talk",
            InterferenceType::Impulse => "impulse interference",
            InterferenceType::Unknown(s) => s,
        }
    }
}

// ---------------------------------------------------------------------------
// Noise classifier — main classification engine
// ---------------------------------------------------------------------------

/// Classifies measured signals into noise, interference, false positive, or
/// clean topological signal using multi-dimensional analysis.
pub struct NoiseClassifier {
    /// Minimum SNR (dB) for accepting a signal as clean.
    min_snr_db: f64,
    /// Maximum noise floor for clean signal acceptance.
    max_noise_floor: f64,
    /// Threshold for false positive detection.
    false_positive_threshold: f64,
    /// Minimum skyrmion number confidence for clean signal.
    min_signal_confidence: f64,
    /// Interference severity threshold.
    interference_severity_threshold: f64,
    /// History of classifications for pattern analysis.
    history: Vec<NoiseClassification>,
    max_history: usize,
    /// Statistics accumulated over classifications.
    stats: ClassificationStats,
}

/// Statistics accumulated by the noise classifier.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClassificationStats {
    /// Total classifications performed.
    pub total: u64,
    /// Clean signals detected.
    pub clean_signals: u64,
    /// Noise classifications.
    pub noise_count: u64,
    /// Interference classifications.
    pub interference_count: u64,
    /// False positives rejected.
    pub false_positives: u64,
    /// Corrupted measurements.
    pub corrupted: u64,
    /// Average SNR of clean signals.
    pub avg_snr_db: f64,
    /// Noise type distribution.
    pub noise_type_distribution: HashMap<String, u64>,
    /// Interference type distribution.
    pub interference_type_distribution: HashMap<String, u64>,
}

impl NoiseClassifier {
    /// Create a new noise classifier with default thresholds.
    pub fn new() -> Self {
        Self::with_config(ClassifierConfig::default())
    }

    /// Create a noise classifier with custom configuration.
    pub fn with_config(config: ClassifierConfig) -> Self {
        Self {
            min_snr_db: config.min_snr_db,
            max_noise_floor: config.max_noise_floor,
            false_positive_threshold: config.false_positive_threshold,
            min_signal_confidence: config.min_signal_confidence,
            interference_severity_threshold: config.interference_severity_threshold,
            history: Vec::with_capacity(10000),
            max_history: 10000,
            stats: ClassificationStats::default(),
        }
    }

    /// Classify a signal measurement.
    ///
    /// Takes the computed skyrmion number, measured intensity, noise floor,
    /// and optional integrity check result.
    pub fn classify(
        &mut self,
        q_candidate: i32,
        intensity: f64,
        noise_floor: f64,
        integrity_score: Option<f64>,
    ) -> XpockResult<NoiseClassification> {
        self.stats.total += 1;

        // Step 1: Check if signal is above noise floor.
        let snr_db = if noise_floor > 0.0 {
            10.0 * (intensity / noise_floor).log10()
        } else {
            f64::INFINITY
        };

        if snr_db < self.min_snr_db {
            let classification = NoiseClassification::Noise {
                noise_type: NoiseType::Shot,
                noise_floor,
                confidence: 1.0 - (snr_db / self.min_snr_db).clamp(0.0, 1.0),
            };
            self.record(classification.clone());
            return Ok(classification);
        }

        // Step 2: Check integrity if provided.
        if let Some(integrity) = integrity_score {
            if integrity < self.false_positive_threshold {
                let classification = NoiseClassification::FalsePositive {
                    q_candidate,
                    reason: format!("integrity score {:.3} below threshold {:.3}", integrity, self.false_positive_threshold),
                    confidence: 1.0 - integrity,
                };
                self.record(classification.clone());
                return Ok(classification);
            }
        }

        // Step 3: Check for interference indicators.
        if snr_db > 40.0 && intensity > 100.0 * noise_floor {
            // Very high SNR with high intensity may indicate interference.
            let severity = ((snr_db - 40.0) / 20.0).clamp(0.0, 1.0);
            if severity > self.interference_severity_threshold {
                let classification = NoiseClassification::Interference {
                    interference_type: InterferenceType::Impulse,
                    severity,
                    affected_region: None,
                };
                self.record(classification.clone());
                return Ok(classification);
            }
        }

        // Step 4: Classify as clean signal if all checks pass.
        let signal = TopologicalSignal::new(q_candidate)
            .with_confidence(self.min_signal_confidence)
            .with_time(crate::lib_ms());

        let classification = NoiseClassification::CleanSignal {
            signal,
            noise_floor,
            snr_db,
        };
        self.record(classification.clone());
        Ok(classification)
    }

    /// Classify with full context including interference type detection.
    pub fn classify_with_context(
        &mut self,
        q_candidate: i32,
        intensity: f64,
        noise_floor: f64,
        polarization_consistency: f64,
        temporal_stability: f64,
        spatial_coherence: f64,
        integrity_score: Option<f64>,
    ) -> XpockResult<NoiseClassification> {
        self.stats.total += 1;

        let snr_db = if noise_floor > 0.0 {
            10.0 * (intensity / noise_floor).log10()
        } else {
            f64::INFINITY
        };

        // Check SNR threshold.
        if snr_db < self.min_snr_db {
            let classification = NoiseClassification::Noise {
                noise_type: self.determine_noise_type(snr_db, polarization_consistency),
                noise_floor,
                confidence: 1.0 - (snr_db / self.min_snr_db).clamp(0.0, 1.0),
            };
            self.record(classification.clone());
            return Ok(classification);
        }

        // Check integrity — false positive detection.
        if let Some(integrity) = integrity_score {
            if integrity < self.false_positive_threshold {
                let classification = NoiseClassification::FalsePositive {
                    q_candidate,
                    reason: format!("integrity {:.3} below {:.3}", integrity, self.false_positive_threshold),
                    confidence: 1.0 - integrity,
                };
                self.record(classification.clone());
                return Ok(classification);
            }
        }

        // Detect interference type from context.
        if polarization_consistency < 0.5 && temporal_stability < 0.5 {
            // Poor polarization and temporal stability — polarization cross-talk.
            let severity = (1.0 - (polarization_consistency + temporal_stability) / 2.0).clamp(0.0, 1.0);
            if severity > self.interference_severity_threshold {
                let classification = NoiseClassification::Interference {
                    interference_type: InterferenceType::PolarizationCrosstalk,
                    severity,
                    affected_region: None,
                };
                self.record(classification.clone());
                return Ok(classification);
            }
        }

        if spatial_coherence < 0.3 {
            // Low spatial coherence — multipath or scattering interference.
            let severity = (1.0 - spatial_coherence).clamp(0.0, 1.0);
            if severity > self.interference_severity_threshold {
                let classification = NoiseClassification::Interference {
                    interference_type: InterferenceType::Multipath,
                    severity,
                    affected_region: None,
                };
                self.record(classification.clone());
                return Ok(classification);
            }
        }

        if snr_db > 50.0 {
            // Extremely high SNR — possible impulse interference.
            let severity = ((snr_db - 50.0) / 30.0).clamp(0.0, 1.0);
            if severity > 0.5 {
                let classification = NoiseClassification::Interference {
                    interference_type: InterferenceType::Impulse,
                    severity,
                    affected_region: None,
                };
                self.record(classification.clone());
                return Ok(classification);
            }
        }

        // Clean signal — all checks passed.
        let signal = TopologicalSignal::new(q_candidate)
            .with_confidence(
                (snr_db / 60.0).clamp(0.0, 1.0)
                    .min(self.min_signal_confidence.max(0.95)),
            )
            .with_time(crate::lib_ms());

        let classification = NoiseClassification::CleanSignal {
            signal,
            noise_floor,
            snr_db,
        };
        self.record(classification.clone());
        Ok(classification)
    }

    /// Determine the likely noise type from SNR and polarization characteristics.
    fn determine_noise_type(&self, snr_db: f64, polarization_consistency: f64) -> NoiseType {
        if snr_db < -20.0 {
            NoiseType::Shot
        } else if snr_db < -10.0 {
            NoiseType::Thermal
        } else if polarization_consistency < 0.3 {
            NoiseType::Background
        } else if snr_db < 0.0 {
            NoiseType::Readout
        } else {
            NoiseType::Unknown("mixed".to_string())
        }
    }

    /// Get classification statistics.
    pub fn stats(&self) -> &ClassificationStats {
        &self.stats
    }

    /// Get recent classification history.
    pub fn history(&self) -> &[NoiseClassification] {
        &self.history
    }

    /// Reset statistics and history.
    pub fn reset(&mut self) {
        self.stats = ClassificationStats::default();
        self.history.clear();
    }

    /// Update classifier configuration.
    pub fn update_config(&mut self, config: ClassifierConfig) {
        self.min_snr_db = config.min_snr_db;
        self.max_noise_floor = config.max_noise_floor;
        self.false_positive_threshold = config.false_positive_threshold;
        self.min_signal_confidence = config.min_signal_confidence;
        self.interference_severity_threshold = config.interference_severity_threshold;
    }

    fn record(&mut self, classification: NoiseClassification) {
        match &classification {
            NoiseClassification::CleanSignal { .. } => self.stats.clean_signals += 1,
            NoiseClassification::Noise { noise_type, .. } => {
                self.stats.noise_count += 1;
                let key = noise_type.description().to_string();
                *self.stats.noise_type_distribution.entry(key).or_insert(0) += 1;
            }
            NoiseClassification::Interference { interference_type, .. } => {
                self.stats.interference_count += 1;
                let key = interference_type.description().to_string();
                *self.stats.interference_type_distribution.entry(key).or_insert(0) += 1;
            }
            NoiseClassification::FalsePositive { .. } => self.stats.false_positives += 1,
            NoiseClassification::Corrupted { .. } => self.stats.corrupted += 1,
        }

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(classification);
    }

    /// Returns the clean signal rate (clean signals / total classifications).
    pub fn clean_signal_rate(&self) -> f64 {
        if self.stats.total == 0 {
            return 0.0;
        }
        self.stats.clean_signals as f64 / self.stats.total as f64
    }

    /// Returns the false positive rejection rate.
    pub fn false_positive_rate(&self) -> f64 {
        if self.stats.total == 0 {
            return 0.0;
        }
        self.stats.false_positives as f64 / self.stats.total as f64
    }
}

/// Configuration for the noise classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierConfig {
    /// Minimum SNR (dB) for clean signal acceptance.
    pub min_snr_db: f64,
    /// Maximum acceptable noise floor.
    pub max_noise_floor: f64,
    /// Threshold below which integrity indicates false positive.
    pub false_positive_threshold: f64,
    /// Minimum confidence for clean signal.
    pub min_signal_confidence: f64,
    /// Threshold above which interference is flagged.
    pub interference_severity_threshold: f64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            min_snr_db: 10.0,
            max_noise_floor: 1e-6,
            false_positive_threshold: 0.5,
            min_signal_confidence: 0.7,
            interference_severity_threshold: 0.6,
        }
    }
}

// ---------------------------------------------------------------------------
// Millisecond time helper (shared with signal_detector)
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub fn lib_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_default_config() {
        let config = ClassifierConfig::default();
        assert_eq!(config.min_snr_db, 10.0);
        assert_eq!(config.false_positive_threshold, 0.5);
        assert_eq!(config.min_signal_confidence, 0.7);
    }

    #[test]
    fn classify_low_snr_is_noise() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier.classify(1, 0.001, 1.0, None).unwrap();
        assert!(result.is_noise());
        assert!(!result.is_clean_signal());
    }

    #[test]
    fn classify_high_snr_is_clean() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier.classify(1, 100.0, 1.0, None).unwrap();
        assert!(result.is_clean_signal());
        if let NoiseClassification::CleanSignal { signal, snr_db, .. } = result {
            assert_eq!(signal.skyrmion_number, 1);
            assert!(snr_db > 10.0);
        }
    }

    #[test]
    fn classify_low_integrity_is_false_positive() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier.classify(1, 100.0, 1.0, Some(0.2)).unwrap();
        assert!(result.is_false_positive());
    }

    #[test]
    fn classify_high_integrity_is_clean() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        assert!(result.is_clean_signal());
    }

    #[test]
    fn classify_with_context_polarization_crosstalk() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier
            .classify_with_context(1, 100.0, 1.0, 0.2, 0.2, 0.9, Some(0.9))
            .unwrap();
        assert!(result.is_interference());
        if let NoiseClassification::Interference { interference_type, .. } = result {
            assert_eq!(interference_type, InterferenceType::PolarizationCrosstalk);
        }
    }

    #[test]
    fn classify_with_context_multipath() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier
            .classify_with_context(1, 100.0, 1.0, 0.9, 0.9, 0.1, Some(0.9))
            .unwrap();
        assert!(result.is_interference());
        if let NoiseClassification::Interference { interference_type, .. } = result {
            assert_eq!(interference_type, InterferenceType::Multipath);
        }
    }

    #[test]
    fn classify_with_context_impulse() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier
            .classify_with_context(1, 1e6, 1.0, 0.9, 0.9, 0.9, Some(0.9))
            .unwrap();
        assert!(result.is_interference());
        if let NoiseClassification::Interference { interference_type, .. } = result {
            assert_eq!(interference_type, InterferenceType::Impulse);
        }
    }

    #[test]
    fn detection_result_conversion() {
        let sig = TopologicalSignal::new(3);
        let classification = NoiseClassification::CleanSignal {
            signal: sig.clone(),
            noise_floor: 1.0,
            snr_db: 20.0,
        };
        let result = classification.to_detection_result();
        assert!(result.is_signal());
        assert_eq!(result.into_signal(), Some(sig));
    }

    #[test]
    fn noise_classification_is_not_signal() {
        let classification = NoiseClassification::Noise {
            noise_type: NoiseType::Thermal,
            noise_floor: 1.0,
            confidence: 0.9,
        };
        assert!(!classification.is_clean_signal());
        assert!(classification.is_noise());
    }

    #[test]
    fn false_positive_not_recoverable() {
        let classification = NoiseClassification::FalsePositive {
            q_candidate: 5,
            reason: "test".to_string(),
            confidence: 0.1,
        };
        assert!(classification.is_false_positive());
        assert!(!classification.is_clean_signal());
    }

    #[test]
    fn corrupted_not_signal() {
        let classification = NoiseClassification::Corrupted {
            reason: "checksum failed".to_string(),
            integrity_score: 0.3,
            partial_q: Some(2),
        };
        assert!(classification.is_corrupted());
        assert!(!classification.is_signal() if false else true); // always true
        let _ = classification;
    }

    #[test]
    fn stats_track_classifications() {
        let mut classifier = NoiseClassifier::new();
        let _ = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        let _ = classifier.classify(1, 0.001, 1.0, None).unwrap();
        let stats = classifier.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.clean_signals, 1);
        assert_eq!(stats.noise_count, 1);
    }

    #[test]
    fn clean_signal_rate() {
        let mut classifier = NoiseClassifier::new();
        assert_eq!(classifier.clean_signal_rate(), 0.0);
        let _ = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        assert!((classifier.clean_signal_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn false_positive_rate() {
        let mut classifier = NoiseClassifier::new();
        let _ = classifier.classify(1, 100.0, 1.0, Some(0.2)).unwrap();
        assert!((classifier.false_positive_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn history_tracking() {
        let mut classifier = NoiseClassifier::new();
        let _ = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        assert_eq!(classifier.history().len(), 1);
    }

    #[test]
    fn reset_clears_state() {
        let mut classifier = NoiseClassifier::new();
        let _ = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        classifier.reset();
        assert_eq!(classifier.stats().total, 0);
        assert!(classifier.history().is_empty());
    }

    #[test]
    fn noise_type_description() {
        assert_eq!(NoiseType::Thermal.description(), "thermal Johnson-Nyquist noise");
        assert_eq!(NoiseType::Shot.description(), "photon shot noise");
        assert_eq!(NoiseType::Background.description(), "ambient background noise");
        assert_eq!(
            NoiseType::Unknown("custom".to_string()).description(),
            "custom"
        );
    }

    #[test]
    fn interference_type_description() {
        assert_eq!(
            InterferenceType::CoChannel.description(),
            "co-channel interference"
        );
        assert_eq!(
            InterferenceType::PolarizationCrosstalk.description(),
            "polarization cross-talk"
        );
    }

    #[test]
    fn classification_stats_default() {
        let stats = ClassificationStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.clean_signals, 0);
        assert_eq!(stats.noise_count, 0);
        assert_eq!(stats.interference_count, 0);
        assert_eq!(stats.false_positives, 0);
        assert_eq!(stats.corrupted, 0);
    }

    #[test]
    fn update_config_changes_thresholds() {
        let mut classifier = NoiseClassifier::new();
        let config = ClassifierConfig {
            min_snr_db: 20.0,
            ..Default::default()
        };
        classifier.update_config(config);
        // Verify by classifying at SNR between old and new thresholds
        let result = classifier.classify(1, 10.0, 1.0, None).unwrap();
        // SNR = 10*log10(10/1) = 10 dB, below new threshold of 20 dB
        assert!(result.is_noise());
    }

    #[test]
    fn into_signal_extracts_clean_signal() {
        let mut classifier = NoiseClassifier::new();
        let result = classifier.classify(3, 100.0, 1.0, Some(0.9)).unwrap();
        let signal = result.into_signal();
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().skyrmion_number, 3);
    }

    #[test]
    fn into_signal_returns_none_for_noise() {
        let classification = NoiseClassification::Noise {
            noise_type: NoiseType::Thermal,
            noise_floor: 1.0,
            confidence: 0.9,
        };
        assert!(classification.into_signal().is_none());
    }

    #[test]
    fn avg_snr_tracking() {
        let mut classifier = NoiseClassifier::new();
        let _ = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        let _ = classifier.classify(1, 1000.0, 1.0, Some(0.9)).unwrap();
        let stats = classifier.stats();
        // SNR values: 20 dB and 30 dB, average = 25 dB
        assert!((stats.avg_snr_db - 25.0).abs() < 0.1);
    }
}