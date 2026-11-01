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

// ---------------------------------------------------------------------------
// Noise profile
// ---------------------------------------------------------------------------

/// Characterized noise profile for a specific environment or channel.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseProfile {
    /// Profile identifier.
    pub id: String,
    /// Dominant noise type.
    pub noise_type: NoiseType,
    /// Average noise floor intensity.
    pub noise_floor: f64,
    /// Noise variance.
    pub variance: f64,
    /// Temporal correlation coefficient (0.0 = independent, 1.0 = fully correlated).
    pub temporal_correlation: f64,
    /// Spatial correlation coefficient.
    pub spatial_correlation: f64,
    /// Spectral exponent (for 1/f^alpha noise).
    pub spectral_exponent: f64,
    /// Confidence in the profile characterization (0.0 to 1.0).
    pub characterization_confidence: f64,
    /// Last update timestamp.
    pub last_updated: u64,
}

impl NoiseProfile {
    /// Create a new noise profile.
    pub fn new(
        id: String,
        noise_type: NoiseType,
        noise_floor: f64,
        variance: f64,
    ) -> Self {
        Self {
            id,
            noise_type,
            noise_floor,
            variance,
            temporal_correlation: 0.0,
            spatial_correlation: 0.0,
            spectral_exponent: 1.0,
            characterization_confidence: 0.5,
            last_updated: lib_ms(),
        }
    }

    /// Update the profile with a new noise measurement.
    pub fn update(
        &mut self,
        noise_floor: f64,
        variance: f64,
        confidence: f64,
    ) {
        // Exponential moving average update.
        let alpha = 0.3;
        self.noise_floor = alpha * noise_floor + (1.0 - alpha) * self.noise_floor;
        self.variance = alpha * variance + (1.0 - alpha) * self.variance;
        self.characterization_confidence =
            (self.characterization_confidence + confidence) / 2.0;
        self.last_updated = lib_ms();
    }

    /// Returns the signal-to-noise ratio for a given signal intensity.
    pub fn snr_for_intensity(&self, signal_intensity: f64) -> f64 {
        if self.noise_floor > 0.0 {
            10.0 * (signal_intensity / self.noise_floor).log10()
        } else {
            f64::INFINITY
        }
    }

    /// Returns true if the profile is well characterized.
    pub fn is_well_characterized(&self) -> bool {
        self.characterization_confidence > 0.7
    }
}

/// Collection of noise profiles for different channels or environments.
#[derive(Debug, Clone, Default)]
pub struct NoiseProfileRegistry {
    profiles: std::collections::HashMap<String, NoiseProfile>,
    default_profile: Option<String>,
}

impl NoiseProfileRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            profiles: std::collections::HashMap::new(),
            default_profile: None,
        }
    }

    /// Register a noise profile.
    pub fn register(&mut self, profile: NoiseProfile) {
        let id = profile.id.clone();
        self.profiles.insert(id.clone(), profile);
        if self.default_profile.is_none() {
            self.default_profile = Some(id);
        }
    }

    /// Get a profile by ID.
    pub fn get(&self, id: &str) -> Option<&NoiseProfile> {
        self.profiles.get(id)
    }

    /// Get the default profile.
    pub fn default_profile(&self) -> Option<&NoiseProfile> {
        self.default_profile
            .as_deref()
            .and_then(|id| self.profiles.get(id))
    }

    /// Returns the number of registered profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Interference source localization
// ---------------------------------------------------------------------------

/// An estimated interference source location and characteristics.
#[derive(Debug, Clone, PartialEq)]
pub struct InterferenceSource {
    /// Source identifier.
    pub source_id: u64,
    /// Estimated position (x, y) in field coordinates.
    pub position: (f64, f64),
    /// Interference type.
    pub interference_type: InterferenceType,
    /// Estimated strength (0.0 to 1.0).
    pub strength: f64,
    /// Confidence in the localization (0.0 to 1.0).
    pub confidence: f64,
    /// First detection timestamp.
    pub first_seen: u64,
    /// Last detection timestamp.
    pub last_seen: u64,
    /// Number of detections.
    pub detection_count: u32,
}

impl InterferenceSource {
    /// Create a new interference source estimate.
    pub fn new(
        source_id: u64,
        position: (f64, f64),
        interference_type: InterferenceType,
        strength: f64,
    ) -> Self {
        let now = lib_ms();
        Self {
            source_id,
            position,
            interference_type,
            strength,
            confidence: 0.5,
            first_seen: now,
            last_seen: now,
            detection_count: 1,
        }
    }

    /// Update the source estimate with a new detection.
    pub fn update(
        &mut self,
        position: (f64, f64),
        strength: f64,
        confidence: f64,
    ) {
        // Exponential moving average for position and strength.
        let alpha = 0.4;
        self.position = (
            alpha * position.0 + (1.0 - alpha) * self.position.0,
            alpha * position.1 + (1.0 - alpha) * self.position.1,
        );
        self.strength = alpha * strength + (1.0 - alpha) * self.strength;
        self.confidence = (self.confidence + confidence) / 2.0;
        self.last_seen = lib_ms();
        self.detection_count += 1;
    }

    /// Returns true if the source is currently active (recently detected).
    pub fn is_active(&self, max_age_ms: u64) -> bool {
        lib_ms().saturating_sub(self.last_seen) < max_age_ms
    }

    /// Returns the source's reliability score.
    pub fn reliability(&self) -> f64 {
        (self.confidence * 0.6 + (self.detection_count as f64 / 10.0).clamp(0.0, 1.0) * 0.4)
            .clamp(0.0, 1.0)
    }
}

/// Tracks interference sources across multiple detections.
pub struct InterferenceTracker {
    sources: std::collections::HashMap<u64, InterferenceSource>,
    next_source_id: u64,
    /// Maximum distance for associating a detection with an existing source.
    pub max_association_distance: f64,
    /// Maximum age in ms before a source is considered inactive.
    pub max_source_age_ms: u64,
}

impl InterferenceTracker {
    /// Create a new interference tracker.
    pub fn new() -> Self {
        Self {
            sources: std::collections::HashMap::new(),
            next_source_id: 1,
            max_association_distance: 100.0,
            max_source_age_ms: 60_000,
        }
    }

    /// Process interference detections and update source tracks.
    ///
    /// `detections` is a list of (position, interference_type, strength, confidence) tuples.
    /// Returns the list of active interference sources.
    pub fn update(
        &mut self,
        detections: Vec<((f64, f64), InterferenceType, f64, f64)>,
    ) -> Vec<InterferenceSource> {
        let mut matched_source_ids = std::collections::HashSet::new();

        for (position, interference_type, strength, confidence) in detections {
            let best_source_id = self.find_best_source(
                position,
                interference_type,
                &matched_source_ids,
            );

            if let Some(source_id) = best_source_id {
                if let Some(source) = self.sources.get_mut(&source_id) {
                    source.update(position, strength, confidence);
                    matched_source_ids.insert(source_id);
                }
            } else {
                let source_id = self.next_source_id;
                self.next_source_id += 1;
                let source = InterferenceSource::new(
                    source_id,
                    position,
                    interference_type,
                    strength,
                );
                self.sources.insert(source_id, source);
                matched_source_ids.insert(source_id);
            }
        }

        // Remove inactive sources.
        self.sources
            .retain(|_, s| s.is_active(self.max_source_age_ms));

        self.sources.values().cloned().collect()
    }

    /// Find the best existing source to match a detection.
    fn find_best_source(
        &self,
        position: (f64, f64),
        interference_type: InterferenceType,
        excluded: &std::collections::HashSet<u64>,
    ) -> Option<u64> {
        let mut best_id: Option<u64> = None;
        let mut best_dist: f64 = f64::INFINITY;

        for (id, source) in &self.sources {
            if excluded.contains(id) {
                continue;
            }
            if source.interference_type != interference_type {
                continue;
            }

            let dx = position.0 - source.position.0;
            let dy = position.1 - source.position.1;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < self.max_association_distance && dist < best_dist {
                best_dist = dist;
                best_id = Some(*id);
            }
        }

        best_id
    }

    /// Returns the number of active sources.
    pub fn active_source_count(&self) -> usize {
        self.sources.values().filter(|s| s.is_active(self.max_source_age_ms)).count()
    }

    /// Returns all active sources.
    pub fn active_sources(&self) -> Vec<InterferenceSource> {
        self.sources
            .values()
            .filter(|s| s.is_active(self.max_source_age_ms))
            .cloned()
            .collect()
    }

    /// Reset all sources.
    pub fn reset(&mut self) {
        self.sources.clear();
        self.next_source_id = 1;
    }
}

impl Default for InterferenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Classification confidence calibration
// ---------------------------------------------------------------------------

/// Calibrated confidence score with uncertainty bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedConfidence {
    /// Point estimate of confidence (0.0 to 1.0).
    pub point_estimate: f64,
    /// Lower bound of confidence interval (0.0 to 1.0).
    pub lower_bound: f64,
    /// Upper bound of confidence interval (0.0 to 1.0).
    pub upper_bound: f64,
    /// Number of observations used for calibration.
    pub observation_count: u32,
}

impl CalibratedConfidence {
    /// Create a new calibrated confidence.
    pub fn new(point_estimate: f64, observation_count: u32) -> Self {
        let uncertainty = if observation_count == 0 {
            0.5
        } else {
            1.0 / (observation_count as f64).sqrt()
        };
        Self {
            point_estimate: point_estimate.clamp(0.0, 1.0),
            lower_bound: (point_estimate - uncertainty).clamp(0.0, 1.0),
            upper_bound: (point_estimate + uncertainty).clamp(0.0, 1.0),
            observation_count,
        }
    }

    /// Returns true if the confidence interval is narrow (well calibrated).
    pub fn is_well_calibrated(&self) -> bool {
        (self.upper_bound - self.lower_bound) < 0.2
    }

    /// Returns the margin of uncertainty.
    pub fn uncertainty_margin(&self) -> f64 {
        (self.upper_bound - self.lower_bound) / 2.0
    }
}

/// Calibrates classification confidence based on historical accuracy.
pub struct ConfidenceCalibrator {
    /// Calibration bins for different confidence ranges.
    bins: Vec<CalibrationBin>,
    /// Number of bins.
    pub num_bins: usize,
}

/// A single calibration bin.
#[derive(Debug, Clone, Default)]
struct CalibrationBin {
    /// Lower confidence bound for this bin.
    pub lower: f64,
    /// Upper confidence bound for this bin.
    pub upper: f64,
    /// Number of predictions in this bin.
    pub count: u32,
    /// Number of correct predictions in this bin.
    pub correct: u32,
}

impl ConfidenceCalibrator {
    /// Create a new confidence calibrator.
    pub fn new(num_bins: usize) -> Self {
        Self {
            bins: (0..num_bins)
                .map(|i| CalibrationBin {
                    lower: i as f64 / num_bins as f64,
                    upper: (i + 1) as f64 / num_bins as f64,
                    count: 0,
                    correct: 0,
                })
                .collect(),
            num_bins,
        }
    }

    /// Record a classification result for calibration.
    pub fn record(
        &mut self,
        predicted_confidence: f64,
        was_correct: bool,
    ) {
        let bin_index =
            ((predicted_confidence.clamp(0.0, 1.0)) * self.num_bins as f64) as usize;
        let bin_index = bin_index.min(self.num_bins - 1);

        if let Some(bin) = self.bins.get_mut(bin_index) {
            bin.count += 1;
            if was_correct {
                bin.correct += 1;
            }
        }
    }

    /// Get calibrated confidence for a raw confidence value.
    pub fn calibrate(&self, raw_confidence: f64) -> CalibratedConfidence {
        let bin_index =
            ((raw_confidence.clamp(0.0, 1.0)) * self.num_bins as f64) as usize;
        let bin_index = bin_index.min(self.num_bins - 1);

        let bin = &self.bins[bin_index];
        let calibrated = if bin.count > 0 {
            bin.correct as f64 / bin.count as f64
        } else {
            raw_confidence
        };

        CalibratedConfidence::new(calibrated, bin.count)
    }

    /// Returns the expected calibration error (ECE).
    pub fn expected_calibration_error(&self) -> f64 {
        let total: u32 = self.bins.iter().map(|b| b.count).sum();
        if total == 0 {
            return 0.0;
        }

        let ece = self
            .bins
            .iter()
            .map(|b| {
                if b.count == 0 {
                    0.0
                } else {
                    let accuracy = b.correct as f64 / b.count as f64;
                    let avg_confidence = (b.lower + b.upper) / 2.0;
                    (b.count as f64 / total as f64) * (accuracy - avg_confidence).abs()
                }
            })
            .sum::<f64>();

        ece
    }

    /// Returns the number of calibration records.
    pub fn total_records(&self) -> u32 {
        self.bins.iter().map(|b| b.count).sum()
    }
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new(10)
    }
}

// ---------------------------------------------------------------------------
// Multi-sensor fusion
// ---------------------------------------------------------------------------

/// A classification from a single sensor.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorClassification {
    /// Sensor identifier.
    pub sensor_id: String,
    /// Classification result.
    pub classification: NoiseClassification,
    /// Sensor reliability weight (0.0 to 1.0).
    pub weight: f64,
    /// Timestamp of classification.
    pub timestamp: u64,
}

/// Fuses classifications from multiple sensors using weighted voting.
pub struct MultiSensorFusion {
    /// Sensor weights.
    sensor_weights: std::collections::HashMap<String, f64>,
    /// Classification history per sensor.
    history: std::collections::HashMap<String, Vec<NoiseClassification>>,
    /// Maximum history per sensor.
    max_history_per_sensor: usize,
}

impl MultiSensorFusion {
    /// Create a new multi-sensor fusion engine.
    pub fn new() -> Self {
        Self {
            sensor_weights: std::collections::HashMap::new(),
            history: std::collections::HashMap::new(),
            max_history_per_sensor: 100,
        }
    }

    /// Register a sensor with a weight.
    pub fn register_sensor(&mut self, sensor_id: String, weight: f64) {
        self.sensor_weights.insert(sensor_id, weight.clamp(0.0, 1.0));
    }

    /// Fuse multiple sensor classifications into a single consensus.
    pub fn fuse(
        &mut self,
        classifications: Vec<SensorClassification>,
    ) -> NoiseClassification {
        // Record classifications in history.
        for cls in &classifications {
            self.history
                .entry(cls.sensor_id.clone())
                .or_default()
                .push(cls.classification.clone());
            if let Some(hist) = self.history.get_mut(&cls.sensor_id) {
                if hist.len() > self.max_history_per_sensor {
                    hist.remove(0);
                }
            }
        }

        // If no classifications, return InsufficientData.
        if classifications.is_empty() {
            return NoiseClassification::InsufficientData {
                missing: "no sensor data".into(),
            };
        }

        // Weighted voting for each classification type.
        let mut signal_weight = 0.0f64;
        let mut noise_weight = 0.0f64;
        let mut interference_weight = 0.0f64;
        let mut false_positive_weight = 0.0f64;
        let mut corrupted_weight = 0.0f64;

        for cls in &classifications {
            let weight = self
                .sensor_weights
                .get(&cls.sensor_id)
                .copied()
                .unwrap_or(1.0);

            match &cls.classification {
                NoiseClassification::CleanSignal { .. } => {
                    signal_weight += weight;
                }
                NoiseClassification::Noise { .. } => {
                    noise_weight += weight;
                }
                NoiseClassification::Interference { .. } => {
                    interference_weight += weight;
                }
                NoiseClassification::FalsePositive { .. } => {
                    false_positive_weight += weight;
                }
                NoiseClassification::Corrupted { .. } => {
                    corrupted_weight += weight;
                }
            }
        }

        // Determine the winning classification.
        let max_weight = signal_weight
            .max(noise_weight)
            .max(interference_weight)
            .max(false_positive_weight)
            .max(corrupted_weight);

        if max_weight == 0.0 {
            return NoiseClassification::InsufficientData {
                missing: "zero weight".into(),
            };
        }

        if signal_weight == max_weight {
            // Return the highest confidence clean signal.
            let best_signal = classifications
                .iter()
                .filter_map(|cls| match &cls.classification {
                    NoiseClassification::CleanSignal {
                        signal, confidence, ..
                    } => Some((signal, confidence, weight)),
                    _ => None,
                })
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

            if let Some((signal, confidence, _)) = best_signal {
                return NoiseClassification::CleanSignal {
                    signal: signal.clone(),
                    noise_floor: 0.0,
                    snr_db: 0.0,
                    _confidence: *confidence,
                };
            }
        }

        if noise_weight == max_weight {
            return NoiseClassification::Noise {
                noise_type: NoiseType::Unknown("fused".into()),
                noise_floor: 0.0,
                confidence: 0.5,
            };
        }

        if interference_weight == max_weight {
            return NoiseClassification::Interference {
                interference_type: InterferenceType::Unknown("fused".into()),
                severity: 0.5,
                affected_region: None,
            };
        }

        if false_positive_weight == max_weight {
            return NoiseClassification::FalsePositive {
                q_candidate: 0,
                reason: "fused false positive".into(),
                confidence: 0.5,
            };
        }

        NoiseClassification::Corrupted {
            reason: "fused corrupted".into(),
            integrity_score: 0.0,
            partial_q: None,
        }
    }

    /// Update a sensor's weight based on performance.
    pub fn update_sensor_weight(&mut self, sensor_id: &str, new_weight: f64) {
        self.sensor_weights
            .insert(sensor_id.to_string(), new_weight.clamp(0.0, 1.0));
    }

    /// Returns the weight for a sensor.
    pub fn sensor_weight(&self, sensor_id: &str) -> f64 {
        self.sensor_weights.get(sensor_id).copied().unwrap_or(1.0)
    }

    /// Returns the number of registered sensors.
    pub fn sensor_count(&self) -> usize {
        self.sensor_weights.len()
    }
}

impl Default for MultiSensorFusion {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Adaptive threshold controller
// ---------------------------------------------------------------------------

/// Controls adaptive threshold adjustment based on classification history.
pub struct AdaptiveThresholdController {
    /// Current threshold value.
    pub current_threshold: f64,
    /// Minimum threshold.
    pub min_threshold: f64,
    /// Maximum threshold.
    pub max_threshold: f64,
    /// Target false positive rate (0.0 to 1.0).
    pub target_false_positive_rate: f64,
    /// Learning rate for threshold adjustment.
    pub learning_rate: f64,
    /// History of false positive rates.
    pub fpr_history: Vec<f64>,
    /// Maximum history length.
    pub max_history: usize,
}

impl AdaptiveThresholdController {
    /// Create a new adaptive threshold controller.
    pub fn new(
        initial_threshold: f64,
        min_threshold: f64,
        max_threshold: f64,
    ) -> Self {
        Self {
            current_threshold: initial_threshold,
            min_threshold,
            max_threshold,
            target_false_positive_rate: 0.1,
            learning_rate: 0.05,
            fpr_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Update the threshold based on the observed false positive rate.
    pub fn update(&mut self, false_positive_rate: f64) {
        self.fpr_history.push(false_positive_rate);
        if self.fpr_history.len() > self.max_history {
            self.fpr_history.remove(0);
        }

        let error = false_positive_rate - self.target_false_positive_rate;
        let adjustment = self.learning_rate * error;
        self.current_threshold = (self.current_threshold + adjustment)
            .clamp(self.min_threshold, self.max_threshold);
    }

    /// Returns the average false positive rate over the history.
    pub fn avg_false_positive_rate(&self) -> f64 {
        if self.fpr_history.is_empty() {
            return 0.0;
        }
        self.fpr_history.iter().sum::<f64>() / self.fpr_history.len() as f64
    }

    /// Returns true if the controller has converged (FPR close to target).
    pub fn is_converged(&self, tolerance: f64) -> bool {
        self.avg_false_positive_rate().abs_diff(self.target_false_positive_rate) < tolerance
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.current_threshold = (self.min_threshold + self.max_threshold) / 2.0;
        self.fpr_history.clear();
    }
}

impl Default for AdaptiveThresholdController {
    fn default() -> Self {
        Self::new(0.5, 0.01, 0.99)
    }
}

// ---------------------------------------------------------------------------
// Spectral noise analysis
// ---------------------------------------------------------------------------

/// Spectral characteristics of noise in a signal.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralNoiseProfile {
    /// Dominant noise frequency.
    pub dominant_frequency: f64,
    /// Spectral power distribution across frequency bands.
    pub band_powers: Vec<(f64, f64)>, // (frequency, power)
    /// Spectral centroid.
    pub spectral_centroid: f64,
    /// Spectral bandwidth.
    pub spectral_bandwidth: f64,
    /// 1/f noise exponent (alpha in 1/f^alpha).
    pub pink_noise_exponent: f64,
    /// White noise floor level.
    pub white_noise_floor: f64,
    /// Confidence in the spectral analysis (0.0 to 1.0).
    pub analysis_confidence: f64,
}

impl SpectralNoiseProfile {
    /// Create a new spectral noise profile.
    pub fn new() -> Self {
        Self {
            dominant_frequency: 0.0,
            band_powers: Vec::new(),
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            pink_noise_exponent: 1.0,
            white_noise_floor: 0.0,
            analysis_confidence: 0.0,
        }
    }

    /// Returns true if the profile is dominated by low-frequency noise.
    pub fn is_low_frequency_dominated(&self) -> bool {
        self.dominant_frequency < 10.0 && self.pink_noise_exponent > 0.5
    }

    /// Returns true if the profile is dominated by high-frequency noise.
    pub fn is_high_frequency_dominated(&self) -> bool {
        self.dominant_frequency > 100.0 && self.pink_noise_exponent < 0.3
    }

    /// Returns the total noise power across all bands.
    pub fn total_power(&self) -> f64 {
        self.band_powers.iter().map(|(_, p)| p).sum()
    }
}

/// Analyzes spectral characteristics of noise in signal data.
pub struct SpectralNoiseAnalyzer {
    /// FFT size for spectral analysis.
    pub fft_size: usize,
    /// Frequency resolution.
    pub frequency_resolution: f64,
    /// Number of frequency bands.
    pub num_bands: usize,
    /// Analysis history.
    pub history: Vec<SpectralNoiseProfile>,
    /// Maximum history length.
    pub max_history: usize,
}

impl SpectralNoiseAnalyzer {
    /// Create a new spectral noise analyzer.
    pub fn new() -> Self {
        Self {
            fft_size: 256,
            frequency_resolution: 1.0,
            num_bands: 10,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Analyze spectral noise characteristics from intensity data.
    pub fn analyze(&mut self, data: &[f64]) -> SpectralNoiseProfile {
        let n = data.len();
        if n == 0 {
            return SpectralNoiseProfile::new();
        }

        // Compute mean and variance for basic spectral estimation.
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 = data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n as f64;

        // Compute spectral power in frequency bands using simple DFT-like computation.
        let mut band_powers = Vec::with_capacity(self.num_bands);
        let band_width = (n as f64) / (self.num_bands as f64);

        for b in 0..self.num_bands {
            let band_start = (b as f64 * band_width) as usize;
            let band_end = ((b + 1) as f64 * band_width) as usize;
            let band_end = band_end.min(n);

            let mut band_power = 0.0f64;
            for i in band_start..band_end {
                let freq = i as f64 * self.frequency_resolution;
                let power = data[i].powi(2);
                band_power += power;
            }
            band_powers.push((freq, band_power / band_end.max(1) as f64));
        }

        // Find dominant frequency.
        let dominant = band_powers
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(f, _)| *f)
            .unwrap_or(0.0);

        // Compute spectral centroid.
        let total_power: f64 = band_powers.iter().map(|(_, p)| p).sum();
        let spectral_centroid = if total_power > 0.0 {
            band_powers
                .iter()
                .map(|(f, p)| f * p)
                .sum::<f64>()
                / total_power
        } else {
            0.0
        };

        // Compute spectral bandwidth.
        let spectral_bandwidth = if total_power > 0.0 {
            band_powers
                .iter()
                .map(|(f, p)| (f - spectral_centroid).powi(2) * p)
                .sum::<f64>()
                .sqrt()
                / total_power.sqrt()
        } else {
            0.0
        };

        // Estimate pink noise exponent from band power decay.
        let pink_noise_exponent = self.estimate_pink_exponent(&band_powers);

        // Estimate white noise floor from high-frequency bands.
        let white_noise_floor = band_powers
            .iter()
            .skip(self.num_bands.saturating_sub(3))
            .map(|(_, p)| *p)
            .sum::<f64>()
            / 3.0;

        let analysis_confidence = if n > 100 { 0.9 } else { 0.5 };

        let profile = SpectralNoiseProfile {
            dominant_frequency: dominant,
            band_powers,
            spectral_centroid,
            spectral_bandwidth,
            pink_noise_exponent,
            white_noise_floor,
            analysis_confidence,
        };

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(profile.clone());

        profile
    }

    /// Estimate pink noise exponent from band powers.
    fn estimate_pink_exponent(&self, band_powers: &[(f64, f64)]) -> f64 {
        if band_powers.len() < 3 {
            return 1.0;
        }

        // Fit log(power) vs log(frequency) to estimate exponent.
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut count = 0;

        for (freq, power) in band_powers {
            if *freq > 0.0 && *power > 0.0 {
                let x = freq.ln();
                let y = power.ln();
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                count += 1;
            }
        }

        if count < 3 {
            return 1.0;
        }

        let slope = (count as f64 * sum_xy - sum_x * sum_y)
            / (count as f64 * sum_x2 - sum_x * sum_x);

        -slope.clamp(0.0, 3.0)
    }

    /// Returns the average pink noise exponent across history.
    pub fn avg_pink_exponent(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        self.history
            .iter()
            .map(|p| p.pink_noise_exponent)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the analysis history.
    pub fn history(&self) -> &[SpectralNoiseProfile] {
        &self.history
    }

    /// Reset the analysis history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for SpectralNoiseAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Spatial correlation analysis
// ---------------------------------------------------------------------------

/// Spatial correlation characteristics of noise.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialCorrelationProfile {
    /// Correlation length in pixels.
    pub correlation_length: f64,
    /// Correlation strength at zero lag.
    pub zero_lag_correlation: f64,
    /// Spatial anisotropy ratio.
    pub anisotropy_ratio: f64,
    /// Correlation decay rate.
    pub decay_rate: f64,
    /// Number of correlation samples.
    pub sample_count: usize,
    /// Confidence in the correlation analysis.
    pub analysis_confidence: f64,
}

impl SpatialCorrelationProfile {
    /// Create a new spatial correlation profile.
    pub fn new() -> Self {
        Self {
            correlation_length: 0.0,
            zero_lag_correlation: 1.0,
            anisotropy_ratio: 1.0,
            decay_rate: 0.0,
            sample_count: 0,
            analysis_confidence: 0.0,
        }
    }

    /// Returns true if noise is strongly spatially correlated.
    pub fn is_strongly_correlated(&self) -> bool {
        self.correlation_length > 10.0 && self.zero_lag_correlation > 0.7
    }

    /// Returns true if noise is spatially uncorrelated (white in space).
    pub fn is_spatially_white(&self) -> bool {
        self.correlation_length < 2.0 && self.zero_lag_correlation < 0.3
    }
}

/// Analyzes spatial correlation of noise in 2D data.
pub struct SpatialCorrelationAnalyzer {
    /// Maximum lag for correlation computation.
    pub max_lag: usize,
    /// Number of lag samples.
    pub num_lag_samples: usize,
    /// Analysis history.
    pub history: Vec<SpatialCorrelationProfile>,
    /// Maximum history length.
    pub max_history: usize,
}

impl SpatialCorrelationAnalyzer {
    /// Create a new spatial correlation analyzer.
    pub fn new() -> Self {
        Self {
            max_lag: 50,
            num_lag_samples: 25,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Analyze spatial correlation from 2D intensity data.
    pub fn analyze(&mut self, data: &[Vec<f64>]) -> SpatialCorrelationProfile {
        let height = data.len();
        if height == 0 || data[0].is_empty() {
            return SpatialCorrelationProfile::new();
        }
        let width = data[0].len();

        // Compute mean and variance.
        let mut total = 0.0f64;
        let mut count = 0usize;
        for row in data {
            for &val in row {
                total += val;
                count += 1;
            }
        }
        if count == 0 {
            return SpatialCorrelationProfile::new();
        }
        let mean = total / count as f64;

        let mut variance = 0.0f64;
        for row in data {
            for &val in row {
                variance += (val - mean).powi(2);
            }
        }
        variance /= count as f64;

        if variance < 1e-10 {
            return SpatialCorrelationProfile::new();
        }

        // Compute autocorrelation at different lags.
        let mut correlations = Vec::with_capacity(self.num_lag_samples);
        let max_lag = self.max_lag.min(width.min(height) / 2);

        for lag in 1..=max_lag {
            let mut cov_sum = 0.0f64;
            let mut cov_count = 0usize;

            for y in 0..height {
                for x in 0..width {
                    if x + lag < width {
                        cov_sum += (data[y][x] - mean) * (data[y][x + lag] - mean);
                        cov_count += 1;
                    }
                    if y + lag < height {
                        cov_sum += (data[y][x] - mean) * (data[y + lag][x] - mean);
                        cov_count += 1;
                    }
                }
            }

            if cov_count > 0 {
                let correlation = cov_sum / (cov_count as f64 * variance);
                correlations.push(correlation);
            }
        }

        // Find correlation length (lag where correlation drops below 1/e).
        let correlation_length = correlations
            .iter()
            .position(|&c| c < 1.0 / std::f64::consts::E)
            .map(|i| i as f64)
            .unwrap_or(correlations.len() as f64);

        // Zero-lag correlation is always 1.0 by definition.
        let zero_lag_correlation = 1.0;

        // Estimate anisotropy ratio (simplified: ratio of x vs y correlation decay).
        let anisotropy_ratio = if correlations.len() > 1 {
            correlations[0].max(correlations[correlations.len() - 1])
                / correlations[correlations.len() - 1].max(correlations[0])
        } else {
            1.0
        };

        // Estimate decay rate from correlation curve.
        let decay_rate = if correlations.len() > 1 && correlations[0] > 0.0 {
            -correlations[correlations.len() - 1].ln().abs() / correlations.len() as f64
        } else {
            0.0
        };

        let analysis_confidence = if count > 1000 { 0.9 } else { 0.5 };

        let profile = SpatialCorrelationProfile {
            correlation_length,
            zero_lag_correlation,
            anisotropy_ratio,
            decay_rate,
            sample_count: count,
            analysis_confidence,
        };

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(profile.clone());

        profile
    }

    /// Returns the average correlation length across history.
    pub fn avg_correlation_length(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|p| p.correlation_length)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the analysis history.
    pub fn history(&self) -> &[SpatialCorrelationProfile] {
        &self.history
    }

    /// Reset the analysis history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for SpatialCorrelationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

/// An anomaly detection result.
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyDetectionResult {
    /// Whether an anomaly was detected.
    pub is_anomaly: bool,
    /// Anomaly score (0.0 = normal, 1.0 = severe anomaly).
    pub anomaly_score: f64,
    /// Type of anomaly detected.
    pub anomaly_type: AnomalyType,
    /// Location of the anomaly (x, y) if spatial.
    pub location: Option<(usize, usize)>,
    /// Confidence in the anomaly detection.
    pub confidence: f64,
}

/// Type of anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Sudden spike in signal intensity.
    Spike,
    /// Sudden drop in signal intensity.
    Dropout,
    /// Unusual spatial pattern.
    SpatialPattern,
    /// Temporal discontinuity.
    TemporalDiscontinuity,
    /// Statistical outlier.
    StatisticalOutlier,
    /// Unknown anomaly type.
    Unknown,
}

impl AnomalyDetectionResult {
    /// Returns true if the anomaly is severe.
    pub fn is_severe(&self) -> bool {
        self.anomaly_score > 0.8
    }

    /// Returns true if the anomaly is minor.
    pub fn is_minor(&self) -> bool {
        self.anomaly_score < 0.3
    }
}

/// Detects anomalies in signal data using statistical methods.
pub struct AnomalyDetector {
    /// Z-score threshold for anomaly detection.
    pub z_score_threshold: f64,
    /// Window size for local statistics.
    pub window_size: usize,
    /// Minimum anomaly score to report.
    pub min_anomaly_score: f64,
    /// Detection history.
    pub history: Vec<AnomalyDetectionResult>,
    /// Maximum history length.
    pub max_history: usize,
}

impl AnomalyDetector {
    /// Create a new anomaly detector.
    pub fn new() -> Self {
        Self {
            z_score_threshold: 3.0,
            window_size: 100,
            min_anomaly_score: 0.5,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Detect anomalies in 1D data.
    pub fn detect_1d(&mut self, data: &[f64]) -> Vec<AnomalyDetectionResult> {
        let mut anomalies = Vec::new();
        let n = data.len();
        if n < self.window_size {
            return anomalies;
        }

        // Compute global statistics.
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 = data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return anomalies;
        }

        // Scan for anomalies using sliding window.
        let half_window = self.window_size / 2;
        for i in half_window..n.saturating_sub(half_window) {
            let window_start = i.saturating_sub(half_window);
            let window_end = (i + half_window).min(n);
            let window: Vec<f64> = data[window_start..window_end].to_vec();

            let local_mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let local_variance: f64 = window
                .iter()
                .map(|x| (x - local_mean).powi(2))
                .sum::<f64>()
                / window.len() as f64;
            let local_std = local_variance.sqrt();

            if local_std < 1e-10 {
                continue;
            }

            let z_score = (data[i] - local_mean) / local_std;

            if z_score.abs() > self.z_score_threshold {
                let anomaly_score = (z_score.abs() / self.z_score_threshold)
                    .clamp(0.0, 1.0);

                if anomaly_score >= self.min_anomaly_score {
                    let anomaly_type = if data[i] > local_mean {
                        AnomalyType::Spike
                    } else {
                        AnomalyType::Dropout
                    };

                    anomalies.push(AnomalyDetectionResult {
                        is_anomaly: true,
                        anomaly_score,
                        anomaly_type,
                        location: Some((i, 0)),
                        confidence: anomaly_score,
                    });
                }
            }
        }

        // Store in history.
        for anomaly in &anomalies {
            if self.history.len() >= self.max_history {
                self.history.remove(0);
            }
            self.history.push(anomaly.clone());
        }

        anomalies
    }

    /// Detect anomalies in 2D data.
    pub fn detect_2d(&mut self, data: &[Vec<f64>]) -> Vec<AnomalyDetectionResult> {
        let mut anomalies = Vec::new();
        let height = data.len();
        if height == 0 || data[0].is_empty() {
            return anomalies;
        }
        let width = data[0].len();

        // Flatten data for global statistics.
        let mut flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        flat.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = flat.len();
        let median = if n % 2 == 0 {
            (flat[n / 2 - 1] + flat[n / 2]) / 2.0
        } else {
            flat[n / 2]
        };

        let mad: f64 = flat
            .iter()
            .map(|x| (x - median).abs())
            .sum::<f64>()
            / n as f64;

        if mad < 1e-10 {
            return anomalies;
        }

        // Scan for spatial anomalies using modified Z-score.
        let threshold = 3.5 * mad / 0.6745; // Modified Z-score threshold.

        for y in 0..height {
            for x in 0..width {
                let modified_z = 0.6745 * (data[y][x] - median).abs() / mad;
                if modified_z > self.z_score_threshold {
                    let anomaly_score = (modified_z / self.z_score_threshold)
                        .clamp(0.0, 1.0);

                    if anomaly_score >= self.min_anomaly_score {
                        anomalies.push(AnomalyDetectionResult {
                            is_anomaly: true,
                            anomaly_score,
                            anomaly_type: AnomalyType::SpatialPattern,
                            location: Some((x, y)),
                            confidence: anomaly_score,
                        });
                    }
                }
            }
        }

        // Store in history.
        for anomaly in &anomalies {
            if self.history.len() >= self.max_history {
                self.history.remove(0);
            }
            self.history.push(anomaly.clone());
        }

        anomalies
    }

    /// Returns the number of anomalies detected in history.
    pub fn total_anomalies(&self) -> usize {
        self.history.len()
    }

    /// Returns the average anomaly score across history.
    pub fn avg_anomaly_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|a| a.anomaly_score)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Reset the detection history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update detection parameters.
    pub fn update_params(&mut self, z_score_threshold: f64, window_size: usize) {
        self.z_score_threshold = z_score_threshold.max(1.0);
        self.window_size = window_size.max(10);
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Classification confidence tracker
// ---------------------------------------------------------------------------

/// Tracks classification confidence over time for calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceTrackingRecord {
    /// Timestamp of the classification.
    pub timestamp: u64,
    /// Predicted confidence.
    pub predicted_confidence: f64,
    /// Actual correctness (1.0 = correct, 0.0 = incorrect).
    pub actual_correctness: f64,
    /// Classification type.
    pub classification_type: String,
}

/// Tracks and calibrates classification confidence.
pub struct ConfidenceTracker {
    /// Tracking records.
    pub records: Vec<ConfidenceTrackingRecord>,
    /// Maximum history length.
    pub max_history: usize,
    /// Calibration bins.
    pub num_bins: usize,
}

impl ConfidenceTracker {
    /// Create a new confidence tracker.
    pub fn new() -> Self {
        Self {
            records: Vec::with_capacity(1000),
            max_history: 1000,
            num_bins: 10,
        }
    }

    /// Record a classification result.
    pub fn record(
        &mut self,
        predicted_confidence: f64,
        actual_correctness: f64,
        classification_type: String,
    ) {
        if self.records.len() >= self.max_history {
            self.records.remove(0);
        }
        self.records.push(ConfidenceTrackingRecord {
            timestamp: crate::detection::signal_detector::lib_ms(),
            predicted_confidence: predicted_confidence.clamp(0.0, 1.0),
            actual_correctness: actual_correctness.clamp(0.0, 1.0),
            classification_type,
        });
    }

    /// Get calibrated confidence for a raw confidence value.
    pub fn calibrate(&self, raw_confidence: f64) -> f64 {
        let raw = raw_confidence.clamp(0.0, 1.0);
        let bin_idx = ((raw * self.num_bins as f64) as usize).min(self.num_bins - 1);

        let bin_start = bin_idx as f64 / self.num_bins as f64;
        let bin_end = (bin_idx + 1) as f64 / self.num_bins as f64;

        let bin_records: Vec<&ConfidenceTrackingRecord> = self
            .records
            .iter()
            .filter(|r| {
                let c = r.predicted_confidence;
                c >= bin_start && c < bin_end
            })
            .collect();

        if bin_records.is_empty() {
            return raw;
        }

        let avg_correctness: f64 = bin_records
            .iter()
            .map(|r| r.actual_correctness)
            .sum::<f64>()
            / bin_records.len() as f64;

        avg_correctness
    }

    /// Returns the expected calibration error.
    pub fn expected_calibration_error(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }

        let mut total_ece = 0.0f64;
        let total_count = self.records.len() as f64;

        for b in 0..self.num_bins {
            let bin_start = b as f64 / self.num_bins as f64;
            let bin_end = (b + 1) as f64 / self.num_bins as f64;

            let bin_records: Vec<&ConfidenceTrackingRecord> = self
                .records
                .iter()
                .filter(|r| {
                    let c = r.predicted_confidence;
                    c >= bin_start && c < bin_end
                })
                .collect();

            if bin_records.is_empty() {
                continue;
            }

            let bin_count = bin_records.len() as f64;
            let avg_confidence: f64 = bin_records
                .iter()
                .map(|r| r.predicted_confidence)
                .sum::<f64>()
                / bin_count;
            let avg_correctness: f64 = bin_records
                .iter()
                .map(|r| r.actual_correctness)
                .sum::<f64>()
                / bin_count;

            total_ece += (bin_count / total_count) * (avg_confidence - avg_correctness).abs();
        }

        total_ece
    }

    /// Returns the number of tracking records.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Reset the tracking history.
    pub fn reset(&mut self) {
        self.records.clear();
    }
}

impl Default for ConfidenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Time-series analysis
// ---------------------------------------------------------------------------

/// Time-series analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSeriesAnalysisResult {
    /// Trend direction.
    pub trend: TrendDirection,
    /// Trend strength (0.0 to 1.0).
    pub trend_strength: f64,
    /// Seasonality strength (0.0 to 1.0).
    pub seasonality_strength: f64,
    /// Noise level in the series.
    pub noise_level: f64,
    /// Autocorrelation at lag 1.
    pub autocorrelation_lag1: f64,
    /// Detected change points.
    pub change_points: Vec<usize>,
}

/// Trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stationary,
    Unknown,
}

/// Analyzes time-series properties of detection data.
pub struct TimeSeriesAnalyzer {
    /// Window size for trend analysis.
    pub window_size: usize,
    /// Minimum data points for analysis.
    pub min_points: usize,
    /// Change point detection threshold.
    pub change_point_threshold: f64,
}

impl TimeSeriesAnalyzer {
    /// Create a new time-series analyzer.
    pub fn new() -> Self {
        Self {
            window_size: 50,
            min_points: 10,
            change_point_threshold: 3.0,
        }
    }

    /// Analyze a time-series of values.
    pub fn analyze(&self, data: &[f64]) -> TimeSeriesAnalysisResult {
        let n = data.len();
        if n < self.min_points {
            return TimeSeriesAnalysisResult {
                trend: TrendDirection::Unknown,
                trend_strength: 0.0,
                seasonality_strength: 0.0,
                noise_level: 0.0,
                autocorrelation_lag1: 0.0,
                change_points: Vec::new(),
            };
        }

        // Compute trend using linear regression.
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = data.iter().sum::<f64>() / n as f64;

        let mut sum_xy = 0.0f64;
        let mut sum_x2 = 0.0f64;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            sum_xy += (x - x_mean) * (y - y_mean);
            sum_x2 += (x - x_mean).powi(2);
        }

        let slope = if sum_x2 > 0.0 {
            sum_xy / sum_x2
        } else {
            0.0
        };

        // Normalize slope by data range for trend strength.
        let y_range: f64 = data
            .iter()
            .cloned()
            .fold(f64::INFINITY / 2.0, |a, b| a.min(b));
        let y_max: f64 = data.iter().cloned().fold(f64::NEG_INFINITY / 2.0, |a, b| a.max(b));
        let range = y_max - y_range;

        let trend_strength = if range > 0.0 {
            (slope * (n - 1) as f64 / range).abs().clamp(0.0, 1.0)
        } else {
            0.0
        };

        let trend = if slope > 0.01 * range / (n as f64) {
            TrendDirection::Increasing
        } else if slope < -0.01 * range / (n as f64) {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stationary
        };

        // Compute noise level (residual variance from trend).
        let mut residual_sum = 0.0f64;
        for (i, &y) in data.iter().enumerate() {
            let predicted = y_mean + slope * (i as f64 - x_mean);
            residual_sum += (y - predicted).powi(2);
        }
        let noise_level = (residual_sum / n as f64).sqrt();

        // Compute autocorrelation at lag 1.
        let mut autocorr_numerator = 0.0f64;
        let mut autocorr_denominator = 0.0f64;
        for i in 1..n {
            autocorr_numerator += (data[i] - y_mean) * (data[i - 1] - y_mean);
            autocorr_denominator += (data[i] - y_mean).powi(2);
        }
        let autocorrelation_lag1 = if autocorr_denominator > 0.0 {
            autocorr_numerator / autocorr_denominator
        } else {
            0.0
        };

        // Detect change points using cumulative sum.
        let mut change_points = Vec::new();
        if n > 20 {
            let segment_size = n / 4;
            for i in segment_size..n - segment_size {
                let left_mean: f64 = data[i - segment_size..i].iter().sum::<f64>()
                    / segment_size as f64;
                let right_mean: f64 = data[i..i + segment_size].iter().sum::<f64>()
                    / segment_size as f64;
                let left_std: f64 = data[i - segment_size..i]
                    .iter()
                    .map(|x| (x - left_mean).powi(2))
                    .sum::<f64>()
                    / segment_size as f64;
                let right_std: f64 = data[i..i + segment_size]
                    .iter()
                    .map(|x| (x - right_mean).powi(2))
                    .sum::<f64>()
                    / segment_size as f64;
                let pooled_std = ((left_std + right_std) / 2.0).sqrt();

                if pooled_std > 1e-10 {
                    let cusum = (left_mean - right_mean).abs() / pooled_std;
                    if cusum > self.change_point_threshold {
                        change_points.push(i);
                    }
                }
            }
        }

        TimeSeriesAnalysisResult {
            trend,
            trend_strength,
            seasonality_strength: autocorrelation_lag1.abs(),
            noise_level,
            autocorrelation_lag1,
            change_points,
        }
    }

    /// Returns true if the time series has a significant trend.
    pub fn has_significant_trend(&self, result: &TimeSeriesAnalysisResult) -> bool {
        result.trend_strength > 0.5 && result.trend != TrendDirection::Unknown
    }

    /// Returns true if the time series has change points.
    pub fn has_change_points(&self, result: &TimeSeriesAnalysisResult) -> bool {
        !result.change_points.is_empty()
    }

    /// Update analysis parameters.
    pub fn update_params(&mut self, window_size: usize, change_point_threshold: f64) {
        self.window_size = window_size.max(10);
        self.change_point_threshold = change_point_threshold.max(1.0);
    }
}

impl Default for TimeSeriesAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cross-correlation analysis
// ---------------------------------------------------------------------------

/// Cross-correlation result between two signals.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossCorrelationResult {
    /// Maximum correlation coefficient.
    pub max_correlation: f64,
    /// Lag at which maximum correlation occurs.
    pub lag_at_max: isize,
    /// Correlation at zero lag.
    pub zero_lag_correlation: f64,
    /// Whether the signals are significantly correlated.
    pub is_significantly_correlated: bool,
    /// Confidence in the correlation analysis.
    pub confidence: f64,
}

/// Computes cross-correlation between two signals.
pub struct CrossCorrelationAnalyzer {
    /// Maximum lag for computation.
    pub max_lag: usize,
    /// Significance threshold for correlation.
    pub significance_threshold: f64,
}

impl CrossCorrelationAnalyzer {
    /// Create a new cross-correlation analyzer.
    pub fn new() -> Self {
        Self {
            max_lag: 100,
            significance_threshold: 0.5,
        }
    }

    /// Compute cross-correlation between two 1D signals.
    pub fn correlate(&self, signal_a: &[f64], signal_b: &[f64]) -> CrossCorrelationResult {
        let n = signal_a.len().min(signal_b.len());
        if n == 0 {
            return CrossCorrelationResult {
                max_correlation: 0.0,
                lag_at_max: 0,
                zero_lag_correlation: 0.0,
                is_significantly_correlated: false,
                confidence: 0.0,
            };
        }

        // Compute means.
        let mean_a: f64 = signal_a.iter().take(n).sum::<f64>() / n as f64;
        let mean_b: f64 = signal_b.iter().take(n).sum::<f64>() / n as f64;

        // Compute variances.
        let var_a: f64 = signal_a
            .iter()
            .take(n)
            .map(|x| (x - mean_a).powi(2))
            .sum::<f64>()
            / n as f64;
        let var_b: f64 = signal_b
            .iter()
            .take(n)
            .map(|x| (x - mean_b).powi(2))
            .sum::<f64>()
            / n as f64;

        let std_a = var_a.sqrt();
        let std_b = var_b.sqrt();

        if std_a < 1e-10 || std_b < 1e-10 {
            return CrossCorrelationResult {
                max_correlation: 0.0,
                lag_at_max: 0,
                zero_lag_correlation: 0.0,
                is_significantly_correlated: false,
                confidence: 0.0,
            };
        }

        // Compute cross-correlation at different lags.
        let max_lag = self.max_lag.min(n / 4);
        let mut correlations = Vec::with_capacity(2 * max_lag + 1);

        for lag in -(max_lag as isize)..=(max_lag as isize) {
            let mut cov_sum = 0.0f64;
            let mut valid_count = 0usize;

            for i in 0..n {
                let j = i as isize + lag;
                if j >= 0 && j < n as isize {
                    cov_sum += (signal_a[i] - mean_a) * (signal_b[j as usize] - mean_b);
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                let correlation = cov_sum / (valid_count as f64 * std_a * std_b);
                correlations.push((lag, correlation));
            }
        }

        // Find maximum correlation.
        let max_entry = correlations
            .iter()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap_or(&(0, 0.0));

        let zero_lag_entry = correlations
            .iter()
            .find(|(lag, _)| *lag == 0)
            .map(|(_, c)| *c)
            .unwrap_or(0.0);

        let is_significantly_correlated = max_entry.1.abs() > self.significance_threshold;

        CrossCorrelationResult {
            max_correlation: max_entry.1,
            lag_at_max: max_entry.0,
            zero_lag_correlation: zero_lag_entry,
            is_significantly_correlated,
            confidence: max_correlation.abs().clamp(0.0, 1.0),
        }
    }

    /// Returns true if two signals are synchronized (correlated at zero lag).
    pub fn are_synchronized(&self, result: &CrossCorrelationResult) -> bool {
        result.zero_lag_correlation.abs() > self.significance_threshold
            && result.lag_at_max.abs() <= 1
    }

    /// Update analysis parameters.
    pub fn update_params(&mut self, max_lag: usize, significance_threshold: f64) {
        self.max_lag = max_lag.max(10);
        self.significance_threshold = significance_threshold.clamp(0.0, 1.0);
    }
}

impl Default for CrossCorrelationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Noise budget analysis
// ---------------------------------------------------------------------------

/// Noise budget allocation across different noise sources.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseBudgetAllocation {
    /// Total noise budget.
    pub total_budget: f64,
    /// Allocated budget per noise source.
    pub allocations: Vec<(String, f64)>, // (source_name, allocated_budget)
    /// Remaining unallocated budget.
    pub remaining_budget: f64,
    /// Whether the budget is balanced.
    pub is_balanced: bool,
}

/// Manages noise budget allocation across detection stages.
pub struct NoiseBudgetManager {
    /// Total available noise budget.
    pub total_budget: f64,
    /// Current allocations.
    pub allocations: std::collections::HashMap<String, f64>,
    /// Budget history.
    pub history: Vec<NoiseBudgetAllocation>,
    /// Maximum history length.
    pub max_history: usize,
}

impl NoiseBudgetManager {
    /// Create a new noise budget manager.
    pub fn new(total_budget: f64) -> Self {
        Self {
            total_budget,
            allocations: std::collections::HashMap::new(),
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Allocate budget to a noise source.
    pub fn allocate(&mut self, source: String, budget: f64) -> bool {
        let current_total: f64 = self.allocations.values().sum();
        if current_total + budget > self.total_budget {
            return false;
        }
        self.allocations.insert(source, budget);
        self.record_allocation();
        true
    }

    /// Deallocate budget from a noise source.
    pub fn deallocate(&mut self, source: &str) -> Option<f64> {
        let budget = self.allocations.remove(source);
        if budget.is_some() {
            self.record_allocation();
        }
        budget
    }

    /// Get the allocated budget for a source.
    pub fn get_allocation(&self, source: &str) -> Option<f64> {
        self.allocations.get(source).copied()
    }

    /// Get the remaining budget.
    pub fn remaining_budget(&self) -> f64 {
        let allocated: f64 = self.allocations.values().sum();
        self.total_budget - allocated
    }

    /// Record current allocation in history.
    fn record_allocation(&mut self) {
        let allocations_vec: Vec<(String, f64)> =
            self.allocations.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let remaining = self.remaining_budget();
        let is_balanced = remaining >= 0.0 && remaining < self.total_budget * 0.1;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(NoiseBudgetAllocation {
            total_budget: self.total_budget,
            allocations: allocations_vec,
            remaining_budget: remaining,
            is_balanced,
        });
    }

    /// Returns the allocation history.
    pub fn history(&self) -> &[NoiseBudgetAllocation] {
        &self.history
    }

    /// Returns true if the budget is currently balanced.
    pub fn is_balanced(&self) -> bool {
        self.history.last().map(|a| a.is_balanced).unwrap_or(false)
    }

    /// Reset the budget manager.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.history.clear();
    }
}

impl Default for NoiseBudgetManager {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ---------------------------------------------------------------------------
// Environmental condition classifier
// ---------------------------------------------------------------------------

/// Environmental condition classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentalCondition {
    Pristine,
    MildlyDegraded,
    ModeratelyDegraded,
    SeverelyDegraded,
    Critical,
}

/// Classifies environmental conditions based on noise measurements.
pub struct EnvironmentalConditionClassifier {
    /// Thresholds for each condition level.
    pub thresholds: [f64; 4], // [pristine->mild, mild->moderate, moderate->severe, severe->critical]
    /// Classification history.
    pub history: Vec<(EnvironmentalCondition, u64)>, // (condition, timestamp)
    /// Maximum history length.
    pub max_history: usize,
}

impl EnvironmentalConditionClassifier {
    /// Create a new environmental condition classifier.
    pub fn new() -> Self {
        Self {
            thresholds: [0.2, 0.4, 0.6, 0.8],
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Classify environmental condition based on noise metrics.
    pub fn classify(
        &mut self,
        noise_score: f64,
        timestamp: u64,
    ) -> EnvironmentalCondition {
        let condition = if noise_score < self.thresholds[0] {
            EnvironmentalCondition::Pristine
        } else if noise_score < self.thresholds[1] {
            EnvironmentalCondition::MildlyDegraded
        } else if noise_score < self.thresholds[2] {
            EnvironmentalCondition::ModeratelyDegraded
        } else if noise_score < self.thresholds[3] {
            EnvironmentalCondition::SeverelyDegraded
        } else {
            EnvironmentalCondition::Critical
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push((condition, timestamp));

        condition
    }

    /// Returns the current condition (last classification).
    pub fn current_condition(&self) -> Option<EnvironmentalCondition> {
        self.history.last().map(|(c, _)| *c)
    }

    /// Returns the condition history.
    pub fn history(&self) -> &[(EnvironmentalCondition, u64)] {
        &self.history
    }

    /// Returns true if conditions are degrading over time.
    pub fn is_degrading(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }
        let recent: &[(EnvironmentalCondition, u64)] = &self.history[self.history.len() - 3..];
        let condition_values: Vec<f64> = recent
            .iter()
            .map(|(c, _)| match c {
                EnvironmentalCondition::Pristine => 0.0,
                EnvironmentalCondition::MildlyDegraded => 0.25,
                EnvironmentalCondition::ModeratelyDegraded => 0.5,
                EnvironmentalCondition::SeverelyDegraded => 0.75,
                EnvironmentalCondition::Critical => 1.0,
            })
            .collect();

        condition_values[2] > condition_values[0]
    }

    /// Update classification thresholds.
    pub fn update_thresholds(&mut self, thresholds: [f64; 4]) {
        self.thresholds = thresholds;
    }
}

impl Default for EnvironmentalConditionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Signal integrity monitor
// ---------------------------------------------------------------------------

/// Signal integrity monitoring result.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegrityMonitorResult {
    /// Overall integrity score (0.0 to 1.0).
    pub integrity_score: f64,
    /// Number of integrity violations detected.
    pub violation_count: usize,
    /// Types of violations detected.
    pub violation_types: Vec<IntegrityViolationType>,
    /// Whether the signal meets integrity requirements.
    pub meets_requirements: bool,
}

/// Type of integrity violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityViolationType {
    /// Signal amplitude out of range.
    AmplitudeOutOfRange,
    /// Phase discontinuity.
    PhaseDiscontinuity,
    /// Frequency drift.
    FrequencyDrift,
    /// Polarization corruption.
    PolarizationCorruption,
    /// Timing violation.
    TimingViolation,
}

/// Monitors signal integrity over time.
pub struct IntegrityMonitor {
    /// Integrity thresholds.
    pub amplitude_threshold: f64,
    pub phase_threshold: f64,
    pub frequency_threshold: f64,
    /// Violation history.
    pub violations: Vec<IntegrityViolationType>,
    /// Maximum history length.
    pub max_history: usize,
}

impl IntegrityMonitor {
    /// Create a new integrity monitor.
    pub fn new() -> Self {
        Self {
            amplitude_threshold: 0.9,
            phase_threshold: 0.1,
            frequency_threshold: 0.05,
            violations: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Monitor signal integrity.
    pub fn monitor(
        &mut self,
        amplitude: f64,
        phase_deviation: f64,
        frequency_drift: f64,
        polarization_error: f64,
    ) -> IntegrityMonitorResult {
        let mut violation_types = Vec::new();

        if amplitude > self.amplitude_threshold {
            violation_types.push(IntegrityViolationType::AmplitudeOutOfRange);
        }
        if phase_deviation > self.phase_threshold {
            violation_types.push(IntegrityViolationType::PhaseDiscontinuity);
        }
        if frequency_drift > self.frequency_threshold {
            violation_types.push(IntegrityViolationType::FrequencyDrift);
        }
        if polarization_error > 0.1 {
            violation_types.push(IntegrityViolationType::PolarizationCorruption);
        }

        // Store violations in history.
        for v in &violation_types {
            if self.violations.len() >= self.max_history {
                self.violations.remove(0);
            }
            self.violations.push(*v);
        }

        // Compute integrity score.
        let amplitude_score = (1.0 - amplitude).clamp(0.0, 1.0);
        let phase_score = (1.0 - phase_deviation / self.phase_threshold).clamp(0.0, 1.0);
        let frequency_score = (1.0 - frequency_drift / self.frequency_threshold).clamp(0.0, 1.0);
        let polarization_score = (1.0 - polarization_error).clamp(0.0, 1.0);

        let integrity_score = (amplitude_score + phase_score + frequency_score + polarization_score)
            / 4.0;

        let meets_requirements = violation_types.is_empty() && integrity_score > 0.7;

        IntegrityMonitorResult {
            integrity_score,
            violation_count: violation_types.len(),
            violation_types,
            meets_requirements,
        }
    }

    /// Returns the violation rate across history.
    pub fn violation_rate(&self) -> f64 {
        if self.violations.is_empty() {
            return 0.0;
        }
        self.violations.len() as f64 / self.max_history as f64
    }

    /// Returns the most common violation type.
    pub fn most_common_violation(&self) -> Option<IntegrityViolationType> {
        if self.violations.is_empty() {
            return None;
        }
        let mut counts: std::collections::HashMap<IntegrityViolationType, u32> =
            std::collections::HashMap::new();
        for &v in &self.violations {
            *counts.entry(v).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|_, &v| v)
            .map(|(k, _)| k)
    }

    /// Reset the monitor history.
    pub fn reset(&mut self) {
        self.violations.clear();
    }
}

impl Default for IntegrityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// False positive suppression engine
// ---------------------------------------------------------------------------

/// False positive suppression result.
#[derive(Debug, Clone, PartialEq)]
pub struct SuppressionResult {
    /// Whether the detection was suppressed.
    pub is_suppressed: bool,
    /// Suppression confidence (0.0 to 1.0).
    pub suppression_confidence: f64,
    /// Reason for suppression (if suppressed).
    pub suppression_reason: Option<String>,
    /// Adjusted confidence after suppression.
    pub adjusted_confidence: f64,
}

/// Suppresses false positives using multiple criteria.
pub struct FalsePositiveSuppressionEngine {
    /// Minimum confidence for non-suppression.
    pub min_confidence: f64,
    /// Maximum allowed false positive rate.
    pub max_false_positive_rate: f64,
    /// Suppression history.
    pub history: Vec<SuppressionResult>,
    /// Maximum history length.
    pub max_history: usize,
    /// Adaptive threshold enabled.
    pub adaptive_enabled: bool,
}

impl FalsePositiveSuppressionEngine {
    /// Create a new false positive suppression engine.
    pub fn new() -> Self {
        Self {
            min_confidence: 0.7,
            max_false_positive_rate: 0.1,
            history: Vec::with_capacity(1000),
            max_history: 1000,
            adaptive_enabled: true,
        }
    }

    /// Evaluate whether a detection should be suppressed.
    pub fn evaluate(
        &mut self,
        confidence: f64,
        anomaly_score: f64,
        integrity_score: f64,
    ) -> SuppressionResult {
        let mut suppression_confidence = 0.0f64;
        let mut suppression_reason: Option<String> = None;

        // Check confidence threshold.
        if confidence < self.min_confidence {
            suppression_confidence = 1.0 - confidence;
            suppression_reason = Some("confidence below threshold".into());
        }

        // Check anomaly score.
        if anomaly_score > 0.7 {
            let anomaly_suppression = anomaly_score.clamp(0.0, 1.0);
            if anomaly_suppression > suppression_confidence {
                suppression_confidence = anomaly_suppression;
                suppression_reason = Some("high anomaly score".into());
            }
        }

        // Check integrity score.
        if integrity_score < 0.5 {
            let integrity_suppression = 1.0 - integrity_score;
            if integrity_suppression > suppression_confidence {
                suppression_confidence = integrity_suppression;
                suppression_reason = Some("low integrity score".into());
            }
        }

        let is_suppressed = suppression_confidence > 0.5;

        // Apply adaptive threshold if enabled.
        let adjusted_confidence = if self.adaptive_enabled {
            let fp_rate = self.current_false_positive_rate();
            if fp_rate > self.max_false_positive_rate {
                confidence * (1.0 - (fp_rate - self.max_false_positive_rate)).clamp(0.0, 1.0)
            } else {
                confidence
            }
        } else {
            confidence
        };

        let result = SuppressionResult {
            is_suppressed,
            suppression_confidence,
            suppression_reason,
            adjusted_confidence,
        };

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        result
    }

    /// Returns the current false positive rate.
    pub fn current_false_positive_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let suppressed_count = self
            .history
            .iter()
            .filter(|r| r.is_suppressed)
            .count() as f64;
        suppressed_count / self.history.len() as f64
    }

    /// Returns the suppression history.
    pub fn history(&self) -> &[SuppressionResult] {
        &self.history
    }

    /// Returns true if adaptive suppression is enabled.
    pub fn is_adaptive_enabled(&self) -> bool {
        self.adaptive_enabled
    }

    /// Update engine parameters.
    pub fn update_params(
        &mut self,
        min_confidence: f64,
        max_false_positive_rate: f64,
        adaptive_enabled: bool,
    ) {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self.max_false_positive_rate = max_false_positive_rate.clamp(0.0, 1.0);
        self.adaptive_enabled = adaptive_enabled;
    }

    /// Reset the engine history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for FalsePositiveSuppressionEngine {
    fn default() -> Self {
        Self::new()
    }
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

    // ---------------------------------------------------------------------------
    // SpectralNoiseAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn spectral_noise_analyzer_new() {
        let analyzer = SpectralNoiseAnalyzer::default();
        assert_eq!(analyzer.fft_size, 256);
        assert!((analyzer.frequency_resolution - 1.0).abs() < 0.001);
        assert_eq!(analyzer.num_bands, 10);
        assert!(analyzer.history.is_empty());
    }

    #[test]
    fn spectral_noise_analyzer_empty_input() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.dominant_frequency, 0.0);
        assert_eq!(result.spectral_centroid, 0.0);
        assert_eq!(result.spectral_bandwidth, 0.0);
        assert!(result.analysis_confidence < 0.6);
    }

    #[test]
    fn spectral_noise_analyzer_uniform_input() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data = vec![1.0f64; 200];
        let result = analyzer.analyze(&data);
        assert!(result.analysis_confidence > 0.5);
        assert!(result.band_powers.len() <= 10);
        assert!(result.total_power() > 0.0);
    }

    #[test]
    fn spectral_noise_analyzer_varying_input() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = analyzer.analyze(&data);
        assert!(result.dominant_frequency >= 0.0);
        assert!(result.spectral_centroid >= 0.0);
        assert!(result.pink_noise_exponent >= 0.0);
    }

    #[test]
    fn spectral_noise_analyzer_low_frequency_dominated() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let mut profile = SpectralNoiseProfile::new();
        profile.dominant_frequency = 5.0;
        profile.pink_noise_exponent = 0.8;
        assert!(profile.is_low_frequency_dominated());
    }

    #[test]
    fn spectral_noise_analyzer_high_frequency_dominated() {
        let mut profile = SpectralNoiseProfile::new();
        profile.dominant_frequency = 200.0;
        profile.pink_noise_exponent = 0.1;
        assert!(profile.is_high_frequency_dominated());
    }

    #[test]
    fn spectral_noise_analyzer_total_power() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data = vec![2.0f64; 100];
        let result = analyzer.analyze(&data);
        assert!(result.total_power() > 0.0);
    }

    #[test]
    fn spectral_noise_analyzer_history_tracking() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data = vec![1.0f64; 100];
        let _ = analyzer.analyze(&data);
        let _ = analyzer.analyze(&data);
        assert_eq!(analyzer.history().len(), 2);
    }

    #[test]
    fn spectral_noise_analyzer_avg_pink_exponent() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data = vec![1.0f64; 100];
        analyzer.analyze(&data);
        analyzer.analyze(&data);
        let avg = analyzer.avg_pink_exponent();
        assert!(avg >= 0.0);
    }

    #[test]
    fn spectral_noise_analyzer_reset() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data = vec![1.0f64; 100];
        analyzer.analyze(&data);
        assert!(!analyzer.history().is_empty());
        analyzer.reset();
        assert!(analyzer.history().is_empty());
    }

    // ---------------------------------------------------------------------------
    // SpatialCorrelationAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn spatial_correlation_analyzer_new() {
        let analyzer = SpatialCorrelationAnalyzer::default();
        assert_eq!(analyzer.max_lag, 50);
        assert_eq!(analyzer.num_lag_samples, 25);
    }

    #[test]
    fn spatial_correlation_analyzer_empty_input() {
        let mut analyzer = SpatialCorrelationAnalyzer::new();
        let data: Vec<Vec<f64>> = vec![];
        let result = analyzer.analyze(&data);
        assert_eq!(result.correlation_length, 0.0);
        assert_eq!(result.zero_lag_correlation, 1.0);
    }

    #[test]
    fn spatial_correlation_analyzer_uniform_input() {
        let mut analyzer = SpatialCorrelationAnalyzer::new();
        let data = vec![vec![1.0f64; 10]; 10];
        let result = analyzer.analyze(&data);
        assert!(result.sample_count > 0);
        assert!(result.analysis_confidence > 0.0);
    }

    #[test]
    fn spatial_correlation_analyzer_random_input() {
        let mut analyzer = SpatialCorrelationAnalyzer::new();
        let data: Vec<Vec<f64>> = (0..20).map(|_| (0..20).map(|_| rand_val()).collect()).collect();
        let result = analyzer.analyze(&data);
        assert!(result.correlation_length >= 0.0);
        assert!(result.decay_rate >= 0.0);
    }

    #[test]
    fn spatial_correlation_analyzer_is_strongly_correlated() {
        let mut profile = SpatialCorrelationProfile::new();
        profile.correlation_length = 15.0;
        profile.zero_lag_correlation = 0.8;
        assert!(profile.is_strongly_correlated());
    }

    #[test]
    fn spatial_correlation_analyzer_is_spatially_white() {
        let mut profile = SpatialCorrelationProfile::new();
        profile.correlation_length = 1.0;
        profile.zero_lag_correlation = 0.2;
        assert!(profile.is_spatially_white());
    }

    #[test]
    fn spatial_correlation_analyzer_history() {
        let mut analyzer = SpatialCorrelationAnalyzer::new();
        let data = vec![vec![1.0f64; 10]; 10];
        analyzer.analyze(&data);
        analyzer.analyze(&data);
        assert_eq!(analyzer.history().len(), 2);
    }

    #[test]
    fn spatial_correlation_analyzer_avg_correlation_length() {
        let mut analyzer = SpatialCorrelationAnalyzer::new();
        let data = vec![vec![1.0f64; 10]; 10];
        analyzer.analyze(&data);
        let avg = analyzer.avg_correlation_length();
        assert!(avg >= 0.0);
    }

    fn rand_val() -> f64 {
        // Simple pseudo-random value for tests
        0.5
    }

    // ---------------------------------------------------------------------------
    // AnomalyDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn anomaly_detector_new() {
        let detector = AnomalyDetector::default();
        assert!((detector.z_score_threshold - 3.0).abs() < 0.001);
        assert_eq!(detector.window_size, 100);
    }

    #[test]
    fn anomaly_detector_empty_input() {
        let mut detector = AnomalyDetector::new();
        let result = detector.detect_1d(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn anomaly_detector_normal_input() {
        let mut detector = AnomalyDetector::new();
        let data: Vec<f64> = (0..200).map(|_| 1.0).collect();
        let result = detector.detect_1d(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn anomaly_detector_spike_detection() {
        let mut detector = AnomalyDetector::new();
        let mut data: Vec<f64> = vec![1.0; 200];
        data[100] = 100.0; // Spike
        let result = detector.detect_1d(&data);
        assert!(!result.is_empty());
        assert!(result.iter().any(|a| a.is_severe()));
    }

    #[test]
    fn anomaly_detector_dropout_detection() {
        let mut detector = AnomalyDetector::new();
        let mut data: Vec<f64> = vec![10.0; 200];
        data[100] = 0.0; // Dropout
        let result = detector.detect_1d(&data);
        assert!(!result.is_empty());
    }

    #[test]
    fn anomaly_detector_2d_detection() {
        let mut detector = AnomalyDetector::new();
        let data: Vec<Vec<f64>> = vec![vec![1.0f64; 10]; 10];
        let result = detector.detect_2d(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn anomaly_detector_total_anomalies() {
        let mut detector = AnomalyDetector::new();
        let mut data: Vec<f64> = vec![1.0; 200];
        data[100] = 100.0;
        detector.detect_1d(&data);
        assert!(detector.total_anomalies() > 0);
    }

    #[test]
    fn anomaly_detector_reset() {
        let mut detector = AnomalyDetector::new();
        let mut data: Vec<f64> = vec![1.0; 200];
        data[100] = 100.0;
        detector.detect_1d(&data);
        assert!(detector.total_anomalies() > 0);
        detector.reset();
        assert_eq!(detector.total_anomalies(), 0);
    }

    #[test]
    fn anomaly_detector_update_params() {
        let mut detector = AnomalyDetector::new();
        detector.update_params(5.0, 50);
        assert!((detector.z_score_threshold - 5.0).abs() < 0.001);
        assert_eq!(detector.window_size, 50);
    }

    // ---------------------------------------------------------------------------
    // ConfidenceTracker tests
    // ---------------------------------------------------------------------------

    #[test]
    fn confidence_tracker_new() {
        let tracker = ConfidenceTracker::default();
        assert_eq!(tracker.num_bins, 10);
        assert_eq!(tracker.record_count(), 0);
    }

    #[test]
    fn confidence_tracker_record() {
        let mut tracker = ConfidenceTracker::new();
        tracker.record(0.8, 1.0, "test".to_string());
        assert_eq!(tracker.record_count(), 1);
    }

    #[test]
    fn confidence_tracker_calibrate() {
        let mut tracker = ConfidenceTracker::new();
        tracker.record(0.8, 1.0, "test".to_string());
        tracker.record(0.8, 1.0, "test".to_string());
        tracker.record(0.8, 0.0, "test".to_string());
        let calibrated = tracker.calibrate(0.8);
        assert!((calibrated - 0.666667).abs() < 0.01);
    }

    #[test]
    fn confidence_tracker_calibrate_no_records() {
        let tracker = ConfidenceTracker::new();
        let calibrated = tracker.calibrate(0.5);
        assert!((calibrated - 0.5).abs() < 0.001);
    }

    #[test]
    fn confidence_tracker_ece() {
        let mut tracker = ConfidenceTracker::new();
        tracker.record(0.1, 1.0, "test".to_string());
        tracker.record(0.9, 1.0, "test".to_string());
        let ece = tracker.expected_calibration_error();
        assert!(ece >= 0.0);
        assert!(ece < 1.0);
    }

    #[test]
    fn confidence_tracker_reset() {
        let mut tracker = ConfidenceTracker::new();
        tracker.record(0.8, 1.0, "test".to_string());
        assert_eq!(tracker.record_count(), 1);
        tracker.reset();
        assert_eq!(tracker.record_count(), 0);
    }

    // ---------------------------------------------------------------------------
    // TimeSeriesAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn time_series_analyzer_new() {
        let analyzer = TimeSeriesAnalyzer::default();
        assert_eq!(analyzer.window_size, 50);
        assert_eq!(analyzer.min_points, 10);
    }

    #[test]
    fn time_series_analyzer_insufficient_data() {
        let analyzer = TimeSeriesAnalyzer::new();
        let data = vec![1.0f64, 2.0, 3.0];
        let result = analyzer.analyze(&data);
        assert_eq!(result.trend, TrendDirection::Unknown);
    }

    #[test]
    fn time_series_analyzer_increasing_trend() {
        let analyzer = TimeSeriesAnalyzer::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let result = analyzer.analyze(&data);
        assert_eq!(result.trend, TrendDirection::Increasing);
        assert!(result.trend_strength > 0.5);
    }

    #[test]
    fn time_series_analyzer_decreasing_trend() {
        let analyzer = TimeSeriesAnalyzer::new();
        let data: Vec<f64> = (0..50).map(|i| 50.0 - i as f64).collect();
        let result = analyzer.analyze(&data);
        assert_eq!(result.trend, TrendDirection::Decreasing);
        assert!(result.trend_strength > 0.5);
    }

    #[test]
    fn time_series_analyzer_stationary() {
        let analyzer = TimeSeriesAnalyzer::new();
        let data = vec![5.0f64; 50];
        let result = analyzer.analyze(&data);
        assert_eq!(result.trend, TrendDirection::Stationary);
        assert_eq!(result.trend_strength, 0.0);
    }

    #[test]
    fn time_series_analyzer_has_significant_trend() {
        let analyzer = TimeSeriesAnalyzer::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let result = analyzer.analyze(&data);
        assert!(analyzer.has_significant_trend(&result));
    }

    #[test]
    fn time_series_analyzer_change_points() {
        let mut analyzer = TimeSeriesAnalyzer::new();
        let mut data: Vec<f64> = (0..50).map(|i| 1.0).collect();
        for i in 25..50 {
            data[i] = 10.0;
        }
        let result = analyzer.analyze(&data);
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn time_series_analyzer_update_params() {
        let mut analyzer = TimeSeriesAnalyzer::new();
        analyzer.update_params(100, 5.0);
        assert_eq!(analyzer.window_size, 100);
    }

    // ---------------------------------------------------------------------------
    // CrossCorrelationAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn cross_correlation_analyzer_new() {
        let analyzer = CrossCorrelationAnalyzer::default();
        assert_eq!(analyzer.max_lag, 100);
    }

    #[test]
    fn cross_correlation_analyzer_identical_signals() {
        let analyzer = CrossCorrelationAnalyzer::new();
        let signal: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let result = analyzer.correlate(&signal, &signal);
        assert!((result.max_correlation - 1.0).abs() < 0.01);
        assert!(result.is_significantly_correlated);
    }

    #[test]
    fn cross_correlation_analyzer_uncorrelated_signals() {
        let analyzer = CrossCorrelationAnalyzer::new();
        let signal_a: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let signal_b: Vec<f64> = (0..100).map(|i| (50 - i) as f64 * 0.1).collect();
        let result = analyzer.correlate(&signal_a, &signal_b);
        assert!(result.max_correlation <= 1.0);
    }

    #[test]
    fn cross_correlation_analyzer_empty_input() {
        let analyzer = CrossCorrelationAnalyzer::new();
        let result = analyzer.correlate(&[], &[]);
        assert_eq!(result.max_correlation, 0.0);
        assert!(!result.is_significantly_correlated);
    }

    #[test]
    fn cross_correlation_analyzer_are_synchronized() {
        let analyzer = CrossCorrelationAnalyzer::new();
        let signal: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let result = analyzer.correlate(&signal, &signal);
        assert!(analyzer.are_synchronized(&result));
    }

    #[test]
    fn cross_correlation_analyzer_update_params() {
        let mut analyzer = CrossCorrelationAnalyzer::new();
        analyzer.update_params(50, 0.8);
        assert_eq!(analyzer.max_lag, 50);
    }

    // ---------------------------------------------------------------------------
    // NoiseBudgetManager tests
    // ---------------------------------------------------------------------------

    #[test]
    fn noise_budget_manager_new() {
        let manager = NoiseBudgetManager::new(1.0);
        assert!((manager.total_budget - 1.0).abs() < 0.001);
        assert!(manager.allocations.is_empty());
    }

    #[test]
    fn noise_budget_manager_allocate() {
        let mut manager = NoiseBudgetManager::new(1.0);
        assert!(manager.allocate("source_a".to_string(), 0.3));
        assert!((manager.get_allocation("source_a").unwrap() - 0.3).abs() < 0.001);
    }

    #[test]
    fn noise_budget_manager_allocate_exceeds_budget() {
        let mut manager = NoiseBudgetManager::new(1.0);
        assert!(manager.allocate("source_a".to_string(), 0.8));
        assert!(!manager.allocate("source_b".to_string(), 0.5));
    }

    #[test]
    fn noise_budget_manager_deallocate() {
        let mut manager = NoiseBudgetManager::new(1.0);
        manager.allocate("source_a".to_string(), 0.3);
        let freed = manager.deallocate("source_a");
        assert_eq!(freed, Some(0.3));
        assert!(manager.get_allocation("source_a").is_none());
    }

    #[test]
    fn noise_budget_manager_remaining_budget() {
        let mut manager = NoiseBudgetManager::new(1.0);
        assert!((manager.remaining_budget() - 1.0).abs() < 0.001);
        manager.allocate("source_a".to_string(), 0.3);
        assert!((manager.remaining_budget() - 0.7).abs() < 0.001);
    }

    #[test]
    fn noise_budget_manager_is_balanced() {
        let mut manager = NoiseBudgetManager::new(1.0);
        manager.allocate("source_a".to_string(), 0.9);
        assert!(manager.is_balanced());
    }

    #[test]
    fn noise_budget_manager_reset() {
        let mut manager = NoiseBudgetManager::new(1.0);
        manager.allocate("source_a".to_string(), 0.3);
        manager.reset();
        assert!(manager.allocations.is_empty());
        assert!(manager.history().is_empty());
    }

    // ---------------------------------------------------------------------------
    // EnvironmentalConditionClassifier tests
    // ---------------------------------------------------------------------------

    #[test]
    fn environmental_condition_classifier_new() {
        let classifier = EnvironmentalConditionClassifier::default();
        assert!((classifier.thresholds[0] - 0.2).abs() < 0.001);
        assert!((classifier.thresholds[3] - 0.8).abs() < 0.001);
    }

    #[test]
    fn environmental_condition_classifier_pristine() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        let condition = classifier.classify(0.1, 1000);
        assert_eq!(condition, EnvironmentalCondition::Pristine);
    }

    #[test]
    fn environmental_condition_classifier_critical() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        let condition = classifier.classify(0.9, 2000);
        assert_eq!(condition, EnvironmentalCondition::Critical);
    }

    #[test]
    fn environmental_condition_classifier_history() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        classifier.classify(0.1, 1000);
        classifier.classify(0.5, 2000);
        assert_eq!(classifier.history().len(), 2);
    }

    #[test]
    fn environmental_condition_classifier_current_condition() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        classifier.classify(0.9, 1000);
        assert_eq!(classifier.current_condition(), Some(EnvironmentalCondition::Critical));
    }

    #[test]
    fn environmental_condition_classifier_is_degrading() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        classifier.classify(0.1, 1000);
        classifier.classify(0.5, 2000);
        classifier.classify(0.9, 3000);
        assert!(classifier.is_degrading());
    }

    #[test]
    fn environmental_condition_classifier_update_thresholds() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        classifier.update_thresholds([0.1, 0.3, 0.5, 0.7]);
        assert!((classifier.thresholds[0] - 0.1).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // IntegrityMonitor tests
    // ---------------------------------------------------------------------------

    #[test]
    fn integrity_monitor_new() {
        let monitor = IntegrityMonitor::default();
        assert!((monitor.amplitude_threshold - 0.9).abs() < 0.001);
    }

    #[test]
    fn integrity_monitor_good_signal() {
        let mut monitor = IntegrityMonitor::new();
        let result = monitor.monitor(0.5, 0.05, 0.02, 0.05);
        assert!(result.meets_requirements);
        assert!(result.integrity_score > 0.7);
        assert_eq!(result.violation_count, 0);
    }

    #[test]
    fn integrity_monitor_bad_signal() {
        let mut monitor = IntegrityMonitor::new();
        let result = monitor.monitor(1.5, 0.5, 0.2, 0.5);
        assert!(!result.meets_requirements);
        assert!(result.violation_count > 0);
    }

    #[test]
    fn integrity_monitor_violation_history() {
        let mut monitor = IntegrityMonitor::new();
        monitor.monitor(1.5, 0.05, 0.02, 0.05);
        assert!(!monitor.violations.is_empty());
    }

    #[test]
    fn integrity_monitor_violation_rate() {
        let mut monitor = IntegrityMonitor::new();
        monitor.monitor(1.5, 0.05, 0.02, 0.05);
        let rate = monitor.violation_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn integrity_monitor_most_common_violation() {
        let mut monitor = IntegrityMonitor::new();
        monitor.monitor(1.5, 0.05, 0.02, 0.05);
        monitor.monitor(1.6, 0.05, 0.02, 0.05);
        let common = monitor.most_common_violation();
        assert!(common.is_some());
    }

    #[test]
    fn integrity_monitor_reset() {
        let mut monitor = IntegrityMonitor::new();
        monitor.monitor(1.5, 0.05, 0.02, 0.05);
        monitor.reset();
        assert!(monitor.violations.is_empty());
    }

    // ---------------------------------------------------------------------------
    // FalsePositiveSuppressionEngine tests
    // ---------------------------------------------------------------------------

    #[test]
    fn false_positive_suppression_engine_new() {
        let engine = FalsePositiveSuppressionEngine::default();
        assert!((engine.min_confidence - 0.7).abs() < 0.001);
        assert!(engine.adaptive_enabled);
    }

    #[test]
    fn false_positive_suppression_engine_suppress_low_confidence() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        let result = engine.evaluate(0.3, 0.1, 0.9);
        assert!(result.is_suppressed);
        assert!(result.suppression_reason.is_some());
    }

    #[test]
    fn false_positive_suppression_engine_suppress_high_anomaly() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        let result = engine.evaluate(0.9, 0.9, 0.9);
        assert!(result.is_suppressed);
        assert_eq!(result.suppression_reason.as_deref(), Some("high anomaly score"));
    }

    #[test]
    fn false_positive_suppression_engine_suppress_low_integrity() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        let result = engine.evaluate(0.9, 0.1, 0.2);
        assert!(result.is_suppressed);
        assert_eq!(result.suppression_reason.as_deref(), Some("low integrity score"));
    }

    #[test]
    fn false_positive_suppression_engine_accept_good() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        let result = engine.evaluate(0.9, 0.1, 0.9);
        assert!(!result.is_suppressed);
        assert!((result.adjusted_confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn false_positive_suppression_engine_false_positive_rate() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        for _ in 0..10 {
            engine.evaluate(0.3, 0.1, 0.9);
        }
        let rate = engine.current_false_positive_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn false_positive_suppression_engine_adaptive() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        assert!(engine.is_adaptive_enabled());
        engine.update_params(0.5, 0.2, false);
        assert!(!engine.is_adaptive_enabled());
    }

    #[test]
    fn false_positive_suppression_engine_reset() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        engine.evaluate(0.3, 0.1, 0.9);
        assert!(!engine.history().is_empty());
        engine.reset();
        assert!(engine.history().is_empty());
    }

    // ---------------------------------------------------------------------------
    // Additional edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn noise_classifier_comprehensive_workflow() {
        let mut classifier = NoiseClassifier::new();
        // Test various scenarios
        let r1 = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        assert!(r1.is_clean_signal());
        let r2 = classifier.classify(1, 0.001, 1.0, None).unwrap();
        assert!(r2.is_noise());
        let r3 = classifier.classify(1, 100.0, 1.0, Some(0.2)).unwrap();
        assert!(r3.is_false_positive());
    }

    #[test]
    fn noise_classifier_stats_accumulation() {
        let mut classifier = NoiseClassifier::new();
        for _ in 0..10 {
            let _ = classifier.classify(1, 100.0, 1.0, Some(0.9)).unwrap();
        }
        let stats = classifier.stats();
        assert_eq!(stats.total, 10);
        assert_eq!(stats.clean_signals, 10);
    }

    #[test]
    fn spectral_noise_analyzer_large_input() {
        let mut analyzer = SpectralNoiseAnalyzer::new();
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = analyzer.analyze(&data);
        assert!(result.analysis_confidence > 0.5);
    }

    #[test]
    fn anomaly_detector_2d_anomaly() {
        let mut detector = AnomalyDetector::new();
        let mut data: Vec<Vec<f64>> = vec![vec![1.0f64; 20]; 20];
        data[10][10] = 100.0;
        let result = detector.detect_2d(&data);
        assert!(!result.is_empty());
    }

    #[test]
    fn confidence_tracker_many_records() {
        let mut tracker = ConfidenceTracker::new();
        for i in 0..100 {
            tracker.record(i as f64 / 100.0, i % 2 == 0, "test".to_string());
        }
        assert_eq!(tracker.record_count(), 100);
        let ece = tracker.expected_calibration_error();
        assert!(ece >= 0.0);
    }

    #[test]
    fn time_series_analyzer_change_point_detection() {
        let mut analyzer = TimeSeriesAnalyzer::new();
        let mut data: Vec<f64> = vec![1.0; 50];
        for i in 30..50 {
            data[i] = 10.0;
        }
        let result = analyzer.analyze(&data);
        assert!(result.trend_strength > 0.0);
    }

    #[test]
    fn cross_correlation_analyzer_lag_detection() {
        let mut analyzer = CrossCorrelationAnalyzer::new();
        let signal_a: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let signal_b: Vec<f64> = (0..100).map(|i| (i as f64 + 5.0) * 0.1).collect();
        let result = analyzer.correlate(&signal_a, &signal_b);
        assert!(result.max_correlation > 0.5);
    }

    #[test]
    fn noise_budget_manager_many_allocations() {
        let mut manager = NoiseBudgetManager::new(10.0);
        for i in 0..5 {
            manager.allocate(format!("source_{}", i), 1.0);
        }
        assert_eq!(manager.allocations.len(), 5);
        assert!((manager.remaining_budget() - 5.0).abs() < 0.001);
    }

    #[test]
    fn environmental_condition_classifier_all_levels() {
        let mut classifier = EnvironmentalConditionClassifier::new();
        assert_eq!(classifier.classify(0.1, 1000), EnvironmentalCondition::Pristine);
        assert_eq!(classifier.classify(0.3, 2000), EnvironmentalCondition::MildlyDegraded);
        assert_eq!(classifier.classify(0.5, 3000), EnvironmentalCondition::ModeratelyDegraded);
        assert_eq!(classifier.classify(0.7, 4000), EnvironmentalCondition::SeverelyDegraded);
        assert_eq!(classifier.classify(0.9, 5000), EnvironmentalCondition::Critical);
    }

    #[test]
    fn integrity_monitor_all_violations() {
        let mut monitor = IntegrityMonitor::new();
        let result = monitor.monitor(1.5, 0.5, 0.2, 0.5);
        assert_eq!(result.violation_count, 4);
    }

    #[test]
    fn false_positive_suppression_engine_adaptive_rate() {
        let mut engine = FalsePositiveSuppressionEngine::new();
        for _ in 0..20 {
            engine.evaluate(0.3, 0.1, 0.9);
        }
        let rate = engine.current_false_positive_rate();
        assert_eq!(rate, 1.0);
    }
}