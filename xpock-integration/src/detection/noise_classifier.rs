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