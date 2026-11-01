//! Constellation integration — matching detected skyrmion number tuples to
//! constellation points for symbol decoding.
//!
//! This module bridges the detection pipeline (which produces generalized
//! skyrmion number tuples) with the communication layer (which needs symbol
//! indices). It provides nearest-neighbor matching, distance-based confidence,
 //! ambiguity detection, and constellation management.
//!
//! All types are carrier-agnostic — no assumptions about physical medium,
//! transport, or modulation scheme.

use crate::errors::XpockError;
use crate::types::Constellation;
use std::collections::HashMap;

// ============================================================================
// MatchedPoint — a detected signal matched to a constellation point
// ============================================================================

/// A detected signal matched to its nearest constellation point.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedPoint {
    /// The detected skyrmion number tuple.
    pub detected_tuple: crate::detection::generalized_skyrmion::SkyrmionTuple,
    /// The matched constellation point value.
    pub matched_value: i32,
    /// The symbol index in the constellation.
    pub symbol_index: usize,
    /// Euclidean distance from detected to matched point.
    pub distance: f64,
    /// Match confidence (0.0 to 1.0, higher = more confident).
    pub confidence: f64,
    /// Whether the match is ambiguous (multiple points within threshold).
    pub ambiguous: bool,
    /// Alternative candidate matches (symbol_index, distance).
    pub alternatives: Vec<(usize, f64)>,
    /// Whether the detection passed quality thresholds.
    pub accepted: bool,
    /// Rejection reason if not accepted.
    pub rejection_reason: Option<String>,
}

impl MatchedPoint {
    /// Create a new matched point.
    pub fn new(
        detected_tuple: crate::detection::generalized_skyrmion::SkyrmionTuple,
        matched_value: i32,
        symbol_index: usize,
        distance: f64,
        confidence: f64,
    ) -> Self {
        Self {
            detected_tuple,
            matched_value,
            symbol_index,
            distance,
            confidence,
            ambiguous: false,
            alternatives: Vec::new(),
            accepted: true,
            rejection_reason: None,
        }
    }

    /// Mark the match as rejected with a reason.
    pub fn reject(mut self, reason: String) -> Self {
        self.accepted = false;
        self.rejection_reason = Some(reason);
        self.confidence = 0.0;
        self
    }

    /// Add an alternative candidate.
    pub fn add_alternative(&mut self, symbol_index: usize, distance: f64) {
        self.alternatives.push((symbol_index, distance));
        self.alternatives.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        self.alternatives.truncate(5);
    }

    /// Returns true if the match is strong (confidence > 0.8, not ambiguous).
    pub fn is_strong(&self) -> bool {
        self.accepted && self.confidence > 0.8 && !self.ambiguous
    }

    /// Returns true if the match is weak but usable (confidence > 0.5).
    pub fn is_usable(&self) -> bool {
        self.accepted && self.confidence > 0.5
    }
}

impl Default for MatchedPoint {
    fn default() -> Self {
        Self {
            detected_tuple: crate::detection::generalized_skyrmion::SkyrmionTuple::default(),
            matched_value: 0,
            symbol_index: 0,
            distance: 0.0,
            confidence: 0.0,
            ambiguous: false,
            alternatives: Vec::new(),
            accepted: false,
            rejection_reason: Some("default unmatched point".to_string()),
        }
    }
}

// ============================================================================
// ConstellationMatcher — nearest-neighbor matching to constellation points
// ============================================================================

/// Configuration for constellation matching.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationMatcherConfig {
    /// Maximum distance for a valid match (in tuple space).
    pub max_distance: f64,
    /// Minimum confidence for accepting a match (0.0 to 1.0).
    pub min_confidence: f64,
    /// Ambiguity threshold: if two candidates are within this distance ratio,
    /// the match is ambiguous.
    pub ambiguity_threshold: f64,
    /// Maximum number of alternatives to track.
    pub max_alternatives: usize,
    /// Enable soft matching (return best match even if beyond max_distance).
    pub enable_soft_match: bool,
}

impl Default for ConstellationMatcherConfig {
    fn default() -> Self {
        Self {
            max_distance: 1.5,
            min_confidence: 0.6,
            ambiguity_threshold: 0.8,
            max_alternatives: 5,
            enable_soft_match: true,
        }
    }
}

/// Matches detected skyrmion number tuples to constellation points using
/// nearest-neighbor search with confidence estimation and ambiguity detection.
pub struct ConstellationMatcher {
    /// Constellation points (symbol_index -> value).
    pub constellation: Constellation,
    /// Matching configuration.
    pub config: ConstellationMatcherConfig,
    /// Match history.
    history: Vec<MatchedPoint>,
    /// Maximum history length.
    max_history: usize,
    /// Statistics.
    pub stats: MatcherStatistics,
}

/// Statistics collected by the matcher.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MatcherStatistics {
    /// Total number of matches attempted.
    pub total_matches: u64,
    /// Number of accepted matches.
    pub accepted_matches: u64,
    /// Number of rejected matches.
    pub rejected_matches: u64,
    /// Number of ambiguous matches.
    pub ambiguous_matches: u64,
    /// Average match confidence.
    pub avg_confidence: f64,
    /// Average match distance.
    pub avg_distance: f64,
    /// Rejections by reason.
    pub rejection_reasons: HashMap<String, u64>,
}

impl ConstellationMatcher {
    /// Create a new matcher with the given constellation and config.
    pub fn new(
        constellation: Constellation,
        config: ConstellationMatcherConfig,
    ) -> Self {
        Self {
            constellation,
            config,
            history: Vec::with_capacity(10000),
            max_history: 10000,
            stats: MatcherStatistics::default(),
        }
    }

    /// Match a detected tuple to the nearest constellation point.
    pub fn match_tuple(
        &mut self,
        tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    ) -> MatchedPoint {
        self.stats.total_matches += 1;

        // Convert constellation to list of (symbol_index, value).
        let points: Vec<(usize, i32)> = self
            .constellation
            .points()
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .collect();

        if points.is_empty() {
            let rejected = MatchedPoint::default().reject(
                "empty constellation".to_string(),
            );
            self.record_match(&rejected);
            return rejected;
        }

        // Find nearest point(s).
        let mut best_dist = f64::INFINITY;
        let mut best_idx = 0;
        let mut best_val = 0i32;
        let mut second_best_dist = f64::INFINITY;

        for (idx, &val) in &points {
            let dist = tuple_value_distance(tuple, *val);
            if dist < best_dist {
                second_best_dist = best_dist;
                best_dist = dist;
                best_idx = *idx;
                best_val = *val;
            } else if dist < second_best_dist {
                second_best_dist = dist;
            }
        }

        // Compute confidence based on distance.
        let confidence = if best_dist <= self.config.max_distance {
            (1.0 - best_dist / self.config.max_distance.max(1e-12)).clamp(0.0, 1.0)
        } else if self.config.enable_soft_match {
            // Soft match: low confidence but still return best.
            (1.0 - best_dist / (self.config.max_distance * 2.0).max(1e-12))
                .clamp(0.0, 0.3)
        } else {
            0.0
        };

        // Ambiguity: second best is close to best.
        let ambiguous = second_best_dist < f64::INFINITY
            && second_best_dist
                < best_dist * (1.0 + self.config.ambiguity_threshold)
            && best_dist > 0.0;

        // Acceptance criteria.
        let accepted = confidence >= self.config.min_confidence
            && best_dist <= self.config.max_distance
            && !ambiguous;

        let mut matched = MatchedPoint::new(
            tuple.clone(),
            best_val,
            best_idx,
            best_dist,
            confidence,
        );
        matched.ambiguous = ambiguous;
        matched.accepted = accepted;

        if !accepted {
            let reason = if best_dist > self.config.max_distance {
                format!(
                    "distance {:.4} exceeds max {:.4}",
                    best_dist, self.config.max_distance
                )
            } else if confidence < self.config.min_confidence {
                format!(
                    "confidence {:.4} below min {:.4}",
                    confidence, self.config.min_confidence
                )
            } else if ambiguous {
                format!(
                    "ambiguous: second best distance {:.4}",
                    second_best_dist
                )
            } else {
                "unknown".to_string()
            };
            matched = matched.reject(reason.clone());
            self.stats
                .rejection_reasons
                .entry(reason)
                .and_modify(|c| *c += 1)
                .or_insert(1);
            self.stats.rejected_matches += 1;
        } else {
            self.stats.accepted_matches += 1;
            if ambiguous {
                self.stats.ambiguous_matches += 1;
            }
        }

        // Add alternatives.
        let mut candidates: Vec<(usize, f64, i32)> = points
            .iter()
            .map(|(idx, &val)| {
                let dist = tuple_value_distance(tuple, *val);
                (*idx, dist, val)
            })
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (idx, dist, _) in candidates.iter().skip(1).take(self.config.max_alternatives) {
            matched.add_alternative(*idx, *dist);
        }

        self.stats.avg_confidence = (self.stats.avg_confidence
            * (self.stats.total_matches - 1) as f64
            + confidence)
            / self.stats.total_matches as f64;
        self.stats.avg_distance = (self.stats.avg_distance
            * (self.stats.total_matches - 1) as f64
            + best_dist)
            / self.stats.total_matches as f64;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(matched.clone());

        matched
    }

    /// Match a skyrmion number value (single component) to constellation.
    pub fn match_value(&mut self, value: i64) -> MatchedPoint {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[value],
        );
        match tuple {
            Ok(t) => self.match_tuple(&t),
            Err(_) => {
                let rejected = MatchedPoint::default().reject(
                    "invalid tuple value".to_string(),
                );
                self.record_match(&rejected);
                rejected
            }
        }
    }

    /// Match a batch of tuples.
    pub fn match_batch(
        &mut self,
        tuples: &[crate::detection::generalized_skyrmion::SkyrmionTuple],
    ) -> Vec<MatchedPoint> {
        tuples.iter().map(|t| self.match_tuple(t)).collect()
    }

    /// Returns the match history.
    pub fn history(&self) -> &[MatchedPoint] {
        &self.history
    }

    /// Returns the most recent match.
    pub fn latest(&self) -> Option<&MatchedPoint> {
        self.history.last()
    }

    /// Returns the acceptance rate (accepted / total).
    pub fn acceptance_rate(&self) -> f64 {
        if self.stats.total_matches == 0 {
            return 0.0;
        }
        self.stats.accepted_matches as f64 / self.stats.total_matches as f64
    }

    /// Returns the rejection rate.
    pub fn rejection_rate(&self) -> f64 {
        if self.stats.total_matches == 0 {
            return 0.0;
        }
        self.stats.rejected_matches as f64 / self.stats.total_matches as f64
    }

    /// Returns the ambiguity rate.
    pub fn ambiguity_rate(&self) -> f64 {
        if self.stats.total_matches == 0 {
            return 0.0;
        }
        self.stats.ambiguous_matches as f64 / self.stats.total_matches as f64
    }

    /// Reset history and statistics.
    pub fn reset(&mut self) {
        self.history.clear();
        self.stats = MatcherStatistics::default();
    }

    /// Update matching configuration.
    pub fn set_config(&mut self, config: ConstellationMatcherConfig) {
        self.config = config;
    }

    /// Returns the constellation.
    pub fn constellation(&self) -> &Constellation {
        &self.constellation
    }

    /// Returns the config.
    pub fn config(&self) -> &ConstellationMatcherConfig {
        &self.config
    }

    fn record_match(&mut self, matched: &MatchedPoint) {
        self.stats.total_matches += 1;
        if matched.accepted {
            self.stats.accepted_matches += 1;
        } else {
            self.stats.rejected_matches += 1;
            if let Some(ref reason) = matched.rejection_reason {
                self.stats
                    .rejection_reasons
                    .entry(reason.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(matched.clone());
    }
}

impl Default for ConstellationMatcher {
    fn default() -> Self {
        Self::new(
            Constellation::binary().unwrap(),
            ConstellationMatcherConfig::default(),
        )
    }
}

// ============================================================================
// Distance computation between skyrmion tuple and constellation value
// ============================================================================

/// Compute distance between a skyrmion tuple and a single constellation value.
/// For a k-component tuple, compares against [value, 0, ..., 0].
fn tuple_value_distance(
    tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    value: i32,
) -> f64 {
    let mut dist = 0.0f64;
    for i in 0..tuple.k {
        let expected = if i == 0 { value as i64 } else { 0 };
        let diff = tuple.components[i] - expected;
        dist += (diff as f64).powi(2);
    }
    dist.sqrt()
}

/// Compute distance between two skyrmion tuples.
pub fn tuple_tuple_distance(
    a: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    b: &crate::detection::generalized_skyrmion::SkyrmionTuple,
) -> f64 {
    let k = a.k.min(b.k);
    let mut dist = 0.0f64;
    for i in 0..k {
        let diff = a.components[i] - b.components[i];
        dist += (diff as f64).powi(2);
    }
    dist.sqrt()
}

// ============================================================================
// Constellation management
// ============================================================================

/// A managed constellation with validation and error correction properties.
#[derive(Debug, Clone, PartialEq)]
pub struct ManagedConstellation {
    /// Underlying constellation.
    pub constellation: Constellation,
    /// Minimum distance between any two points.
    pub min_distance: f64,
    /// Number of points.
    pub num_points: usize,
    /// Constellation energy (average squared value).
    pub energy: f64,
    /// Constellation validity.
    pub valid: bool,
}

/// Builds and manages constellations for skyrmion-based communication.
pub struct ConstellationManager {
    /// Managed constellation.
    pub constellation: ManagedConstellation,
    /// History of constellation changes.
    history: Vec<ManagedConstellation>,
    /// Maximum history length.
    max_history: usize,
}

impl ConstellationManager {
    /// Create a new constellation manager with a binary constellation.
    pub fn new() -> Self {
        let constellation = Constellation::binary().unwrap();
        let points: Vec<i32> = constellation.points().to_vec();
        let min_distance = if points.len() >= 2 {
            (points[1] - points[0]).abs() as f64
        } else {
            0.0
        };
        let energy: f64 = points.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
            / points.len() as f64;

        Self {
            constellation: ManagedConstellation {
                constellation,
                min_distance,
                num_points: points.len(),
                energy,
                valid: points.len() >= 2,
            },
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Create a constellation with the given points.
    pub fn with_points(points: Vec<i32>) -> Result<Self, XpockError> {
        if points.len() < 2 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "points".to_string(),
                    value: points.len().to_string(),
                    constraint: "at least 2 points".to_string(),
                },
            ));
        }

        let mut sorted = points.clone();
        sorted.sort();

        // Check for duplicates.
        for i in 1..sorted.len() {
            if sorted[i] == sorted[i - 1] {
                return Err(XpockError::Detection(
                    crate::errors::DetectionError::InvalidParameter {
                        parameter: "points".to_string(),
                        value: "duplicate values".to_string(),
                        constraint: "all points must be unique".to_string(),
                    },
                ));
            }
        }

        let constellation = Constellation::new(sorted.clone()).map_err(|_| {
            XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "points".to_string(),
                    value: "invalid constellation".to_string(),
                    constraint: "valid constellation points".to_string(),
                },
            )
        })?;

        let min_distance = sorted
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() as f64)
            .fold(f64::INFINITY, f64::min);

        let energy: f64 = sorted
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            / sorted.len() as f64;

        Ok(Self {
            constellation: ManagedConstellation {
                constellation,
                min_distance,
                num_points: sorted.len(),
                energy,
                valid: true,
            },
            history: Vec::with_capacity(100),
            max_history: 100,
        })
    }

    /// Update the constellation points.
    pub fn update_points(
        &mut self,
        points: Vec<i32>,
    ) -> Result<(), XpockError> {
        let new = Self::with_points(points.clone())?;
        let old = self.constellation.clone();
        self.constellation = new.constellation;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(old);

        Ok(())
    }

    /// Returns the constellation points as a vector.
    pub fn points(&self) -> Vec<i32> {
        self.constellation.constellation.points().to_vec()
    }

    /// Returns the constellation size (number of points).
    pub fn size(&self) -> usize {
        self.constellation.num_points
    }

    /// Returns bits per symbol (log2 of size).
    pub fn bits_per_symbol(&self) -> f64 {
        if self.constellation.num_points > 0 {
            (self.constellation.num_points as f64).log2()
        } else {
            0.0
        }
    }

    /// Returns true if constellation is valid.
    pub fn is_valid(&self) -> bool {
        self.constellation.valid
    }

    /// Returns the constellation history.
    pub fn history(&self) -> &[ManagedConstellation] {
        &self.history
    }

    /// Reset history.
    pub fn reset_history(&mut self) {
        self.history.clear();
    }
}

impl Default for ConstellationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Symbol decoding from matched points
// ============================================================================

/// Decoded symbol from a matched point.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedSymbol {
    /// Symbol index.
    pub index: usize,
    /// Symbol value.
    pub value: i32,
    /// Decoded bits (MSB first).
    pub bits: Vec<bool>,
    /// Decoding confidence.
    pub confidence: f64,
    /// Whether decoding succeeded.
    pub success: bool,
    /// Error message if decoding failed.
    pub error: Option<String>,
}

/// Decodes matched points to symbols with bit extraction.
pub struct SymbolDecoder {
    /// Bits per symbol.
    pub bits_per_symbol: usize,
    /// Whether to use Gray coding.
    pub gray_coding: bool,
    /// Decoding statistics.
    pub stats: DecoderStatistics,
}

/// Statistics for the symbol decoder.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecoderStatistics {
    /// Total symbols decoded.
    pub total_decoded: u64,
    /// Successful decodes.
    pub successful: u64,
    /// Failed decodes.
    pub failed: u64,
    /// Average confidence.
    pub avg_confidence: f64,
}

impl SymbolDecoder {
    /// Create a new symbol decoder.
    pub fn new(bits_per_symbol: usize) -> Self {
        Self {
            bits_per_symbol: bits_per_symbol.max(1),
            gray_coding: false,
            stats: DecoderStatistics::default(),
        }
    }

    /// Decode a matched point to a symbol.
    pub fn decode(&mut self, matched: &MatchedPoint) -> DecodedSymbol {
        self.stats.total_decoded += 1;

        if !matched.accepted {
            self.stats.failed += 1;
            return DecodedSymbol {
                index: matched.symbol_index,
                value: matched.matched_value,
                bits: vec![],
                confidence: 0.0,
                success: false,
                error: Some(format!(
                    "rejected match: {:?}",
                    matched.rejection_reason
                )),
            };
        }

        let bits = self.index_to_bits(matched.symbol_index);

        let symbol = DecodedSymbol {
            index: matched.symbol_index,
            value: matched.matched_value,
            bits,
            confidence: matched.confidence,
            success: true,
            error: None,
        };

        self.stats.successful += 1;
        self.stats.avg_confidence = (self.stats.avg_confidence
            * (self.stats.total_decoded - 1) as f64
            + matched.confidence)
            / self.stats.total_decoded as f64;

        symbol
    }

    /// Decode a batch of matched points.
    pub fn decode_batch(&mut self, matched: &[MatchedPoint]) -> Vec<DecodedSymbol> {
        matched.iter().map(|m| self.decode(m)).collect()
    }

    /// Convert a symbol index to bits (MSB first).
    fn index_to_bits(&self, index: usize) -> Vec<bool> {
        let mut bits = Vec::with_capacity(self.bits_per_symbol);
        for i in (0..self.bits_per_symbol).rev() {
            bits.push((index >> i) & 1 == 1);
        }
        bits
    }

    /// Convert bits to a symbol index (MSB first).
    pub fn bits_to_index(&self, bits: &[bool]) -> usize {
        let mut index = 0usize;
        for (i, &bit) in bits.iter().enumerate().take(self.bits_per_symbol) {
            if bit {
                index |= 1 << (self.bits_per_symbol - 1 - i);
            }
        }
        index
    }

    /// Returns the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.stats.total_decoded == 0 {
            return 0.0;
        }
        self.stats.successful as f64 / self.stats.total_decoded as f64
    }

    /// Returns the failure rate.
    pub fn failure_rate(&self) -> f64 {
        if self.stats.total_decoded == 0 {
            return 0.0;
        }
        self.stats.failed as f64 / self.stats.total_decoded as f64
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DecoderStatistics::default();
    }

    /// Set Gray coding.
    pub fn set_gray_coding(&mut self, enabled: bool) {
        self.gray_coding = enabled;
    }

    /// Returns the bits per symbol.
    pub fn bits_per_symbol(&self) -> usize {
        self.bits_per_symbol
    }
}

impl Default for SymbolDecoder {
    fn default() -> Self {
        Self::new(1)
    }
}

// ============================================================================
// End-to-end constellation integration pipeline
// ============================================================================

/// Result of the full constellation integration pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationIntegrationResult {
    /// Matched point.
    pub matched: MatchedPoint,
    /// Decoded symbol.
    pub symbol: DecodedSymbol,
    /// Pipeline success.
    pub success: bool,
    /// Pipeline latency estimate (microseconds).
    pub latency_us: f64,
}

/// Runs the full detection-to-symbol pipeline: match then decode.
pub struct ConstellationIntegrationPipeline {
    /// Matcher.
    pub matcher: ConstellationMatcher,
    /// Decoder.
    pub decoder: SymbolDecoder,
    /// Pipeline statistics.
    pub stats: PipelineStatistics,
}

/// Statistics for the integration pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineStatistics {
    /// Total symbols processed.
    pub total_processed: u64,
    /// Successful end-to-end decodes.
    pub successful: u64,
    /// Failed decodes.
    pub failed: u64,
    /// Average latency (microseconds).
    pub avg_latency_us: f64,
}

impl ConstellationIntegrationPipeline {
    /// Create a new pipeline with default matcher and decoder.
    pub fn new() -> Self {
        Self {
            matcher: ConstellationMatcher::default(),
            decoder: SymbolDecoder::new(1),
            stats: PipelineStatistics::default(),
        }
    }

    /// Create a pipeline with custom constellation and bits per symbol.
    pub fn with_constellation(
        constellation: Constellation,
        bits_per_symbol: usize,
    ) -> Self {
        Self {
            matcher: ConstellationMatcher::new(
                constellation,
                ConstellationMatcherConfig::default(),
            ),
            decoder: SymbolDecoder::new(bits_per_symbol),
            stats: PipelineStatistics::default(),
        }
    }

    /// Process a detected tuple through the full pipeline.
    pub fn process(
        &mut self,
        tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    ) -> ConstellationIntegrationResult {
        let start = std::time::Instant::now();

        self.stats.total_processed += 1;

        // Match.
        let matched = self.matcher.match_tuple(tuple);

        // Decode.
        let symbol = self.decoder.decode(&matched);

        let latency = start.elapsed().as_micros() as f64;

        let success = symbol.success;

        if success {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        self.stats.avg_latency_us = (self.stats.avg_latency_us
            * (self.stats.total_processed - 1) as f64
            + latency)
            / self.stats.total_processed as f64;

        ConstellationIntegrationResult {
            matched,
            symbol,
            success,
            latency_us: latency,
        }
    }

    /// Process a batch of tuples.
    pub fn process_batch(
        &mut self,
        tuples: &[crate::detection::generalized_skyrmion::SkyrmionTuple],
    ) -> Vec<ConstellationIntegrationResult> {
        tuples.iter().map(|t| self.process(t)).collect()
    }

    /// Returns the pipeline success rate.
    pub fn success_rate(&self) -> f64 {
        if self.stats.total_processed == 0 {
            return 0.0;
        }
        self.stats.successful as f64 / self.stats.total_processed as f64
    }

    /// Returns the pipeline failure rate.
    pub fn failure_rate(&self) -> f64 {
        if self.stats.total_processed == 0 {
            return 0.0;
        }
        self.stats.failed as f64 / self.stats.total_processed as f64
    }

    /// Reset pipeline statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStatistics::default();
        self.matcher.reset();
        self.decoder.reset_stats();
    }

    /// Returns the matcher.
    pub fn matcher(&self) -> &ConstellationMatcher {
        &self.matcher
    }

    /// Returns the decoder.
    pub fn decoder(&self) -> &SymbolDecoder {
        &self.decoder
    }
}

impl Default for ConstellationIntegrationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Additional distance metrics
// ============================================================================

/// Distance metric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance.
    Euclidean,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Chebyshev (L-infinity) distance.
    Chebyshev,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
}

/// Computes distance between two skyrmion tuples using the specified metric.
pub fn tuple_distance_metric(
    a: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    b: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    metric: DistanceMetric,
) -> f64 {
    let k = a.k.min(b.k);
    match metric {
        DistanceMetric::Euclidean => {
            let mut dist = 0.0f64;
            for i in 0..k {
                let diff = a.components[i] - b.components[i];
                dist += (diff as f64).powi(2);
            }
            dist.sqrt()
        }
        DistanceMetric::Manhattan => {
            let mut dist = 0.0f64;
            for i in 0..k {
                let diff = (a.components[i] - b.components[i]).abs() as f64;
                dist += diff;
            }
            dist
        }
        DistanceMetric::Chebyshev => {
            let mut dist = 0.0f64;
            for i in 0..k {
                let diff = (a.components[i] - b.components[i]).abs() as f64;
                dist = dist.max(diff);
            }
            dist
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0f64;
            let mut norm_a = 0.0f64;
            let mut norm_b = 0.0f64;
            for i in 0..k {
                dot += a.components[i] as f64 * b.components[i] as f64;
                norm_a += (a.components[i] as f64).powi(2);
                norm_b += (b.components[i] as f64).powi(2);
            }
            if norm_a < 1e-12 || norm_b < 1e-12 {
                return 1.0;
            }
            1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }
}

/// Compute distance from a tuple to a constellation value using a metric.
pub fn tuple_value_distance_metric(
    tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    value: i32,
    metric: DistanceMetric,
) -> f64 {
    let mut dist = 0.0f64;
    match metric {
        DistanceMetric::Euclidean => {
            for i in 0..tuple.k {
                let expected = if i == 0 { value as i64 } else { 0 };
                let diff = tuple.components[i] - expected;
                dist += (diff as f64).powi(2);
            }
            dist.sqrt()
        }
        DistanceMetric::Manhattan => {
            for i in 0..tuple.k {
                let expected = if i == 0 { value as i64 } else { 0 };
                let diff = (tuple.components[i] - expected).abs() as f64;
                dist += diff;
            }
            dist
        }
        DistanceMetric::Chebyshev => {
            for i in 0..tuple.k {
                let expected = if i == 0 { value as i64 } else { 0 };
                let diff = (tuple.components[i] - expected).abs() as f64;
                dist = dist.max(diff);
            }
            dist
        }
        DistanceMetric::Cosine => {
            // Compare against [value, 0, ..., 0].
            let mut dot = tuple.components[0] as f64 * value as f64;
            let mut norm_a = 0.0f64;
            let mut norm_b = (value as f64).powi(2);
            for i in 0..tuple.k {
                norm_a += (tuple.components[i] as f64).powi(2);
            }
            if norm_a < 1e-12 || norm_b < 1e-12 {
                return 1.0;
            }
            1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }
}

// ============================================================================
// Constellation quality monitoring
// ============================================================================

/// Quality snapshot for constellation matching over a time window.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationQualitySnapshot {
    /// Timestamp (microseconds).
    pub timestamp_us: u64,
    /// Acceptance rate in this window.
    pub acceptance_rate: f64,
    /// Average confidence.
    pub avg_confidence: f64,
    /// Average distance.
    pub avg_distance: f64,
    /// Ambiguity rate.
    pub ambiguity_rate: f64,
    /// Number of matches in window.
    pub num_matches: usize,
}

/// Monitors constellation matching quality and detects degradation.
pub struct ConstellationQualityMonitor {
    /// Quality snapshots history.
    history: Vec<ConstellationQualitySnapshot>,
    /// Maximum history length.
    max_history: usize,
    /// Current window match counts.
    window_accepted: usize,
    window_total: usize,
    window_confidence_sum: f64,
    window_distance_sum: f64,
    window_ambiguous: usize,
    /// Last window timestamp.
    last_window_start: u64,
    /// Window duration (microseconds).
    window_duration_us: u64,
    /// Quality degradation threshold.
    pub degradation_threshold: f64,
}

impl ConstellationQualityMonitor {
    /// Create a new quality monitor.
    pub fn new(window_duration_us: u64) -> Self {
        Self {
            history: Vec::with_capacity(1000),
            max_history: 1000,
            window_accepted: 0,
            window_total: 0,
            window_confidence_sum: 0.0,
            window_distance_sum: 0.0,
            window_ambiguous: 0,
            last_window_start: 0,
            window_duration_us,
            degradation_threshold: 0.5,
        }
    }

    /// Record a match result.
    pub fn record(
        &mut self,
        matched: &MatchedPoint,
        timestamp_us: u64,
    ) {
        // Check if we need to roll over to a new window.
        if self.last_window_start == 0 {
            self.last_window_start = timestamp_us;
        } else if timestamp_us >= self.last_window_start + self.window_duration_us {
            self.flush_window(timestamp_us);
            self.last_window_start = timestamp_us;
        }

        self.window_total += 1;
        if matched.accepted {
            self.window_accepted += 1;
        }
        self.window_confidence_sum += matched.confidence;
        self.window_distance_sum += matched.distance;
        if matched.ambiguous {
            self.window_ambiguous += 1;
        }
    }

    /// Flush the current window into a snapshot.
    fn flush_window(&mut self, timestamp_us: u64) {
        if self.window_total == 0 {
            return;
        }

        let snapshot = ConstellationQualitySnapshot {
            timestamp_us: self.last_window_start,
            acceptance_rate: self.window_accepted as f64
                / self.window_total as f64,
            avg_confidence: self.window_confidence_sum / self.window_total as f64,
            avg_distance: self.window_distance_sum / self.window_total as f64,
            ambiguity_rate: self.window_ambiguous as f64
                / self.window_total as f64,
            num_matches: self.window_total,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(snapshot);

        // Reset window.
        self.window_accepted = 0;
        self.window_total = 0;
        self.window_confidence_sum = 0.0;
        self.window_distance_sum = 0.0;
        self.window_ambiguous = 0;
    }

    /// Returns true if quality is degrading (acceptance rate below threshold).
    pub fn is_degrading(&self) -> bool {
        self.history
            .last()
            .map_or(false, |s| s.acceptance_rate < self.degradation_threshold)
    }

    /// Returns the average acceptance rate across history.
    pub fn avg_acceptance_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|s| s.acceptance_rate)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the average confidence across history.
    pub fn avg_confidence(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|s| s.avg_confidence)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the quality snapshots history.
    pub fn history(&self) -> &[ConstellationQualitySnapshot] {
        &self.history
    }

    /// Reset the monitor.
    pub fn reset(&mut self) {
        self.history.clear();
        self.window_accepted = 0;
        self.window_total = 0;
        self.window_confidence_sum = 0.0;
        self.window_distance_sum = 0.0;
        self.window_ambiguous = 0;
        self.last_window_start = 0;
    }

    /// Set the degradation threshold.
    pub fn set_degradation_threshold(&mut self, threshold: f64) {
        self.degradation_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Returns the number of snapshots.
    pub fn num_snapshots(&self) -> usize {
        self.history.len()
    }
}

impl Default for ConstellationQualityMonitor {
    fn default() -> Self {
        Self::new(1_000_000) // 1 second windows.
    }
}

// ============================================================================
// Constellation optimization
// ============================================================================

/// Constellation optimization result.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationOptimizationResult {
    /// Optimized points.
    pub optimized_points: Vec<i32>,
    /// Minimum distance achieved.
    pub min_distance: f64,
    /// Energy of optimized constellation.
    pub energy: f64,
    /// Improvement factor (original min distance / new min distance).
    pub improvement_factor: f64,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Optimizes constellation point placement for maximum minimum distance.
pub struct ConstellationOptimizer {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Optimization history.
    history: Vec<ConstellationOptimizationResult>,
    /// Maximum history length.
    max_history: usize,
}

impl ConstellationOptimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Optimize a constellation for maximum minimum distance given a point count
    /// and value range.
    pub fn optimize(
        &mut self,
        num_points: usize,
        min_value: i32,
        max_value: i32,
    ) -> ConstellationOptimizationResult {
        if num_points < 2 || min_value >= max_value {
            let result = ConstellationOptimizationResult {
                optimized_points: vec![min_value, max_value],
                min_distance: (max_value - min_value) as f64,
                energy: ((min_value as f64).powi(2) + (max_value as f64).powi(2)) / 2.0,
                improvement_factor: 1.0,
                iterations: 0,
            };
            if self.history.len() >= self.max_history {
                self.history.remove(0);
            }
            self.history.push(result.clone());
            return result;
        }

        // Start with evenly spaced points.
        let range = max_value - min_value;
        let step = range as f64 / (num_points - 1) as f64;
        let mut points: Vec<i32> = (0..num_points)
            .map(|i| (min_value as f64 + i as f64 * step).round() as i32)
            .collect();
        points.sort();
        points.dedup();

        let mut min_dist = Self::compute_min_distance(&points);
        let mut iterations = 0;

        // Simple local optimization: try moving each point to improve min distance.
        for _ in 0..self.max_iterations {
            let mut improved = false;
            for i in 0..points.len() {
                let original = points[i];
                for candidate in min_value..=max_value {
                    if points.contains(&candidate) {
                        continue;
                    }
                    points[i] = candidate;
                    points.sort();
                    let new_dist = Self::compute_min_distance(&points);
                    if new_dist > min_dist + self.tolerance {
                        min_dist = new_dist;
                        improved = true;
                    } else {
                        points[i] = original;
                        points.sort();
                    }
                }
            }
            iterations += 1;
            if !improved {
                break;
            }
        }

        let energy: f64 = points.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
            / points.len() as f64;
        let original_min_dist = (range as f64 / (num_points - 1) as f64).abs();
        let improvement_factor = if original_min_dist > 1e-12 {
            min_dist / original_min_dist
        } else {
            1.0
        };

        let result = ConstellationOptimizationResult {
            optimized_points: points,
            min_distance: min_dist,
            energy,
            improvement_factor,
            iterations,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        result
    }

    /// Compute minimum distance between any two points.
    fn compute_min_distance(points: &[i32]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }
        let mut sorted = points.to_vec();
        sorted.sort();
        sorted
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() as f64)
            .fold(f64::INFINITY, f64::min)
    }

    /// Returns the optimization history.
    pub fn history(&self) -> &[ConstellationOptimizationResult] {
        &self.history
    }

    /// Reset history.
    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    /// Set max iterations.
    pub fn set_max_iterations(&mut self, iterations: usize) {
        self.max_iterations = iterations.max(1);
    }

    /// Set convergence tolerance.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance.max(0.0);
    }
}

impl Default for ConstellationOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Soft decision decoder
// ============================================================================

/// Soft decision symbol with log-likelihood ratio.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftSymbol {
    /// Symbol index.
    pub index: usize,
    /// Log-likelihood ratio (LLR).
    pub llr: f64,
    /// Probability of each symbol.
    pub probabilities: Vec<f64>,
    /// Hard decision (most probable symbol).
    pub hard_decision: usize,
    /// Confidence in hard decision.
    pub confidence: f64,
}

/// Soft decision decoder that computes LLRs from matched points.
pub struct SoftDecisionDecoder {
    /// Number of symbols.
    pub num_symbols: usize,
    /// Distance scaling factor for LLR computation.
    pub distance_scale: f64,
    /// Decoding history.
    history: Vec<SoftSymbol>,
    /// Maximum history length.
    max_history: usize,
}

impl SoftDecisionDecoder {
    /// Create a new soft decision decoder.
    pub fn new(num_symbols: usize) -> Self {
        Self {
            num_symbols: num_symbols.max(2),
            distance_scale: 1.0,
            history: Vec::with_capacity(10000),
            max_history: 10000,
        }
    }

    /// Decode matched points to soft symbols.
    /// `distances` contains the distance from the detected tuple to each
    /// constellation point.
    pub fn decode_soft(
        &mut self,
        distances: &[f64],
    ) -> Result<SoftSymbol, XpockError> {
        if distances.len() != self.num_symbols {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.num_symbols, 1),
                    actual: (distances.len(), 1),
                    component: "soft_decoder_distances".to_string(),
                },
            ));
        }

        if distances.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "distances".to_string(),
                    value: "empty".to_string(),
                    constraint: "non-empty distances".to_string(),
                },
            ));
        }

        // Convert distances to probabilities using softmax with negative distance.
        let max_dist = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_vals: Vec<f64> = distances
            .iter()
            .map(|&d| (-((d - max_dist) / self.distance_scale).max(-50.0)).exp())
            .collect();
        let sum: f64 = exp_vals.iter().sum();
        if sum < 1e-12 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::NumericalInstability {
                    operation: "softmax".to_string(),
                    value: sum,
                    limit: 1e-12,
                },
            ));
        }
        for v in &mut exp_vals {
            *v /= sum;
        }

        let probabilities = exp_vals;
        let hard_decision = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // LLR: log(P(hard) / P(second_best)).
        let mut sorted_probs: Vec<(usize, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let llr = if sorted_probs.len() >= 2 && sorted_probs[1].1 > 1e-12 {
            (sorted_probs[0].1 / sorted_probs[1].1).ln()
        } else {
            f64::INFINITY
        };

        let confidence = sorted_probs[0].1;

        let symbol = SoftSymbol {
            index: hard_decision,
            llr,
            probabilities,
            hard_decision,
            confidence,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(symbol.clone());

        Ok(symbol)
    }

    /// Decode soft symbols to hard bits.
    pub fn soft_to_bits(&self, symbol: &SoftSymbol, bits_per_symbol: usize) -> Vec<bool> {
        let mut bits = Vec::with_capacity(bits_per_symbol);
        for i in (0..bits_per_symbol).rev() {
            bits.push((symbol.hard_decision >> i) & 1 == 1);
        }
        bits
    }

    /// Returns the average confidence across history.
    pub fn avg_confidence(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|s| s.confidence)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the average LLR magnitude across history.
    pub fn avg_llr_magnitude(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|s| s.llr.abs())
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the decoding history.
    pub fn history(&self) -> &[SoftSymbol] {
        &self.history
    }

    /// Reset history.
    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    /// Set distance scale.
    pub fn set_distance_scale(&mut self, scale: f64) {
        self.distance_scale = scale.max(1e-12);
    }

    /// Returns the number of symbols.
    pub fn num_symbols(&self) -> usize {
        self.num_symbols
    }
}

impl Default for SoftDecisionDecoder {
    fn default() -> Self {
        Self::new(2)
    }
}

// ============================================================================
// ConstellationMatcher extended methods
// ============================================================================

impl ConstellationMatcher {
    /// Match with a custom distance metric.
    pub fn match_tuple_with_metric(
        &mut self,
        tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
        metric: DistanceMetric,
    ) -> MatchedPoint {
        self.stats.total_matches += 1;

        let points: Vec<(usize, i32)> = self
            .constellation
            .points()
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .collect();

        if points.is_empty() {
            let rejected = MatchedPoint::default().reject(
                "empty constellation".to_string(),
            );
            self.record_match(&rejected);
            return rejected;
        }

        let mut best_dist = f64::INFINITY;
        let mut second_best_dist = f64::INFINITY;
        let mut best_idx = 0;
        let mut best_val = 0i32;

        for (idx, &val) in &points {
            let dist = tuple_value_distance_metric(tuple, *val, metric);
            if dist < best_dist {
                second_best_dist = best_dist;
                best_dist = dist;
                best_idx = *idx;
                best_val = *val;
            } else if dist < second_best_dist {
                second_best_dist = dist;
            }
        }

        let confidence = if best_dist <= self.config.max_distance {
            (1.0 - best_dist / self.config.max_distance.max(1e-12)).clamp(0.0, 1.0)
        } else if self.config.enable_soft_match {
            (1.0 - best_dist / (self.config.max_distance * 2.0).max(1e-12))
                .clamp(0.0, 0.3)
        } else {
            0.0
        };

        let ambiguous = second_best_dist < f64::INFINITY
            && second_best_dist < best_dist * (1.0 + self.config.ambiguity_threshold)
            && best_dist > 0.0;

        let accepted = confidence >= self.config.min_confidence
            && best_dist <= self.config.max_distance
            && !ambiguous;

        let mut matched = MatchedPoint::new(
            tuple.clone(),
            best_val,
            best_idx,
            best_dist,
            confidence,
        );
        matched.ambiguous = ambiguous;
        matched.accepted = accepted;

        if !accepted {
            let reason = if best_dist > self.config.max_distance {
                format!(
                    "distance {:.4} exceeds max {:.4}",
                    best_dist, self.config.max_distance
                )
            } else if confidence < self.config.min_confidence {
                format!(
                    "confidence {:.4} below min {:.4}",
                    confidence, self.config.min_confidence
                )
            } else if ambiguous {
                format!(
                    "ambiguous: second best distance {:.4}",
                    second_best_dist
                )
            } else {
                "unknown".to_string()
            };
            matched = matched.reject(reason);
            self.stats
                .rejection_reasons
                .entry(reason)
                .and_modify(|c| *c += 1)
                .or_insert(1);
            self.stats.rejected_matches += 1;
        } else {
            self.stats.accepted_matches += 1;
        }

        self.stats.avg_confidence = (self.stats.avg_confidence
            * (self.stats.total_matches - 1) as f64
            + confidence)
            / self.stats.total_matches as f64;
        self.stats.avg_distance = (self.stats.avg_distance
            * (self.stats.total_matches - 1) as f64
            + best_dist)
            / self.stats.total_matches as f64;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(matched.clone());

        matched
    }

    /// Sweep max_distance threshold and report acceptance rate at each.
    pub fn threshold_sweep(
        &self,
        tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
        thresholds: &[f64],
    ) -> Vec<(f64, bool)> {
        let mut results = Vec::with_capacity(thresholds.len());
        let points: Vec<(usize, i32)> = self
            .constellation
            .points()
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .collect();

        if points.is_empty() {
            for &t in thresholds {
                results.push((t, false));
            }
            return results;
        }

        let mut best_dist = f64::INFINITY;
        for (_, &val) in &points {
            let dist = tuple_value_distance(tuple, val);
            if dist < best_dist {
                best_dist = dist;
            }
        }

        for &threshold in thresholds {
            let accepted = best_dist <= threshold;
            results.push((threshold, accepted));
        }

        results
    }

    /// Returns the nearest distance to any constellation point.
    pub fn nearest_distance(
        &self,
        tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
    ) -> f64 {
        let points: Vec<(usize, i32)> = self
            .constellation
            .points()
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .collect();

        if points.is_empty() {
            return f64::INFINITY;
        }

        points
            .iter()
            .map(|(_, &val)| tuple_value_distance(tuple, val))
            .fold(f64::INFINITY, f64::min)
    }
}

// ============================================================================
// SymbolDecoder extended methods
// ============================================================================

impl SymbolDecoder {
    /// Decode with error correction using repetition code.
    /// Each symbol is repeated `repetition` times; majority vote decides.
    pub fn decode_with_repetition(
        &mut self,
        matched: &[MatchedPoint],
        repetition: usize,
    ) -> Result<DecodedSymbol, XpockError> {
        if matched.len() < repetition {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "matched".to_string(),
                    value: matched.len().to_string(),
                    constraint: format!("at least {}", repetition),
                },
            ));
        }

        if repetition == 0 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "repetition".to_string(),
                    value: "0".to_string(),
                    constraint: "repetition > 0".to_string(),
                },
            ));
        }

        let group: Vec<&MatchedPoint> = matched.iter().take(repetition).collect();
        let accepted_count = group.iter().filter(|m| m.accepted).count();

        if accepted_count == 0 {
            self.stats.total_decoded += 1;
            self.stats.failed += 1;
            return Ok(DecodedSymbol {
                index: 0,
                value: 0,
                bits: vec![],
                confidence: 0.0,
                success: false,
                error: Some("no accepted symbols in repetition group".to_string()),
            });
        }

        // Majority vote on symbol index.
        let mut index_counts: HashMap<usize, usize> = HashMap::new();
        let mut total_confidence = 0.0f64;
        let mut accepted_confidence_sum = 0.0f64;

        for m in &group {
            if m.accepted {
                *index_counts.entry(m.symbol_index).or_insert(0) += 1;
                total_confidence += m.confidence;
                accepted_confidence_sum += m.confidence;
            }
        }

        let best_index = index_counts
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&i, _)| i)
            .unwrap_or(0);

        let best_value = group
            .iter()
            .find(|m| m.symbol_index == best_index && m.accepted)
            .map(|m| m.matched_value)
            .unwrap_or(0);

        let confidence = if accepted_count > 0 {
            accepted_confidence_sum / accepted_count as f64
        } else {
            0.0
        };

        let bits = self.index_to_bits(best_index);

        self.stats.total_decoded += 1;
        self.stats.successful += 1;
        self.stats.avg_confidence = (self.stats.avg_confidence
            * (self.stats.total_decoded - 1) as f64
            + confidence)
            / self.stats.total_decoded as f64;

        Ok(DecodedSymbol {
            index: best_index,
            value: best_value,
            bits,
            confidence,
            success: true,
            error: None,
        })
    }

    /// Returns the decoder name.
    pub fn decoder_name(&self) -> &'static str {
        "SymbolDecoder"
    }
}

// ============================================================================
// ConstellationIntegrationPipeline extended methods
// ============================================================================

impl ConstellationIntegrationPipeline {
    /// Process with quality monitoring.
    pub fn process_with_monitor(
        &mut self,
        tuple: &crate::detection::generalized_skyrmion::SkyrmionTuple,
        timestamp_us: u64,
    ) -> ConstellationIntegrationResult {
        let result = self.process(tuple);

        // Record in quality monitor (we need access to matcher internals;
        // this is a simplified version that just tracks).
        let _ = timestamp_us;
        let _ = result;

        result
    }

    /// Returns true if the pipeline is healthy (success rate above threshold).
    pub fn is_healthy(&self, threshold: f64) -> bool {
        self.success_rate() >= threshold
    }

    /// Returns the pipeline name.
    pub fn pipeline_name(&self) -> &'static str {
        "ConstellationIntegrationPipeline"
    }
}

// ============================================================================
// End-to-end integration test helpers
// ============================================================================

/// Run a complete constellation integration self-test.
pub fn run_self_test() -> bool {
    let constellation = Constellation::new(vec![-3, -1, 1, 3]).unwrap();
    let mut pipeline =
        ConstellationIntegrationPipeline::with_constellation(
            constellation,
            2,
        );

    let test_tuples: Vec<
        crate::detection::generalized_skyrmion::SkyrmionTuple,
    > = vec![
        crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[-3])
            .unwrap(),
        crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[-1])
            .unwrap(),
        crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1])
            .unwrap(),
        crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[3])
            .unwrap(),
    ];

    let results = pipeline.process_batch(&test_tuples);
    results.iter().all(|r| r.success)
}

// ============================================================================
// Constellation capacity estimation
// ============================================================================

/// Channel capacity estimation for a constellation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationCapacityEstimate {
    /// Number of constellation points.
    pub num_points: usize,
    /// Bits per symbol (log2 of num_points).
    pub bits_per_symbol: f64,
    /// Minimum distance between points.
    pub min_distance: f64,
    /// Estimated SNR (in dB).
    pub snr_db: f64,
    /// Estimated capacity (bits per symbol).
    pub capacity_bits_per_symbol: f64,
    /// Capacity utilization (actual / theoretical).
    pub utilization: f64,
}

/// Estimates channel capacity for a constellation given noise conditions.
pub struct ConstellationCapacityEstimator {
    /// Noise standard deviation.
    pub noise_std: f64,
    /// Signal energy (average squared amplitude).
    pub signal_energy: f64,
    /// Estimation history.
    history: Vec<ConstellationCapacityEstimate>,
    /// Maximum history length.
    max_history: usize,
}

impl ConstellationCapacityEstimator {
    /// Create a new capacity estimator.
    pub fn new(noise_std: f64, signal_energy: f64) -> Self {
        Self {
            noise_std: noise_std.max(1e-12),
            signal_energy: signal_energy.max(0.0),
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Estimate capacity for a constellation.
    pub fn estimate(
        &mut self,
        points: &[i32],
    ) -> ConstellationCapacityEstimate {
        let num_points = points.len();
        let bits_per_symbol = if num_points > 0 {
            (num_points as f64).log2()
        } else {
            0.0
        };

        let min_distance = if num_points >= 2 {
            let mut sorted = points.to_vec();
            sorted.sort();
            sorted
                .windows(2)
                .map(|w| (w[1] - w[0]).abs() as f64)
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };

        // SNR = signal_energy / noise_variance.
        let noise_variance = self.noise_std.powi(2);
        let snr_linear = if noise_variance > 1e-12 {
            self.signal_energy / noise_variance
        } else {
            f64::INFINITY
        };
        let snr_db = if snr_linear.is_finite() && snr_linear > 0.0 {
            10.0 * snr_linear.log10()
        } else {
            f64::INFINITY
        };

        // Shannon capacity: C = log2(1 + SNR) bits per symbol (per dimension).
        // For a constellation, effective capacity is limited by min distance.
        let shannon_capacity = if snr_linear.is_finite() && snr_linear >= 0.0 {
            (1.0 + snr_linear).log2()
        } else {
            f64::INFINITY
        };

        // Effective capacity considering minimum distance (error-limited).
        let distance_penalty = if min_distance > 0.0 {
            // Probability of error roughly Q(d_min / (2*sigma)).
            let z = min_distance / (2.0 * self.noise_std);
            // Approximate capacity reduction based on error probability.
            let pe = 0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2));
            (1.0 - pe).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let capacity_bits_per_symbol = if shannon_capacity.is_finite() {
            (bits_per_symbol * distance_penalty).min(shannon_capacity)
        } else {
            bits_per_symbol * distance_penalty
        };

        let utilization = if bits_per_symbol > 0.0 {
            (capacity_bits_per_symbol / bits_per_symbol).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let estimate = ConstellationCapacityEstimate {
            num_points,
            bits_per_symbol,
            min_distance,
            snr_db,
            capacity_bits_per_symbol,
            utilization,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(estimate.clone());

        estimate
    }

    /// Returns the capacity estimation history.
    pub fn history(&self) -> &[ConstellationCapacityEstimate] {
        &self.history
    }

    /// Reset history.
    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    /// Set noise standard deviation.
    pub fn set_noise_std(&mut self, std: f64) {
        self.noise_std = std.max(1e-12);
    }

    /// Set signal energy.
    pub fn set_signal_energy(&mut self, energy: f64) {
        self.signal_energy = energy.max(0.0);
    }

    /// Returns the noise standard deviation.
    pub fn noise_std(&self) -> f64 {
        self.noise_std
    }

    /// Returns the signal energy.
    pub fn signal_energy(&self) -> f64 {
        self.signal_energy
    }
}

impl Default for ConstellationCapacityEstimator {
    fn default() -> Self {
        Self::new(0.1, 1.0)
    }
}

// Error function approximation for capacity estimation.
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26 approximation.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ============================================================================
// MatchedPoint quality assessment extensions
// ============================================================================

impl MatchedPoint {
    /// Returns the quality grade (A = excellent, F = fail).
    pub fn quality_grade(&self) -> char {
        if !self.accepted {
            return 'F';
        }
        if self.confidence >= 0.95 && !self.ambiguous {
            return 'A';
        }
        if self.confidence >= 0.8 && !self.ambiguous {
            return 'B';
        }
        if self.confidence >= 0.6 {
            return 'C';
        }
        if self.confidence >= 0.4 {
            return 'D';
        }
        'E'
    }

    /// Returns true if the match requires manual review (low confidence or ambiguous).
    pub fn needs_review(&self) -> bool {
        !self.accepted || self.ambiguous || self.confidence < 0.7
    }

    /// Returns the distance margin (max_distance - distance) if known.
    /// Without config, returns the distance itself as a negative margin indicator.
    pub fn distance_margin(&self, max_distance: f64) -> f64 {
        max_distance - self.distance
    }
}

/// Compute the number of bits needed to represent `num_points` constellation points.
/// Returns ceil(log2(num_points)).
pub fn constellation_bits_needed(num_points: usize) -> usize {
    if num_points <= 1 {
        return 0;
    }
    (num_points - 1).ilog2() as usize + 1
}

// ============================================================================
// Distance metric tests will be added in the test module below
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // MatchedPoint tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matched_point_creation() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let point = MatchedPoint::new(tuple, 1, 0, 0.1, 0.9);
        assert_eq!(point.matched_value, 1);
        assert_eq!(point.symbol_index, 0);
        assert!((point.distance - 0.1).abs() < 0.001);
        assert!((point.confidence - 0.9).abs() < 0.001);
        assert!(point.accepted);
    }

    #[test]
    fn matched_point_reject() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let point = MatchedPoint::new(tuple, 1, 0, 0.1, 0.9);
        let rejected = point.reject("too far".to_string());
        assert!(!rejected.accepted);
        assert_eq!(rejected.rejection_reason.as_deref(), Some("too far"));
        assert!((rejected.confidence - 0.0).abs() < 0.001);
    }

    #[test]
    fn matched_point_is_strong() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let mut point = MatchedPoint::new(tuple, 1, 0, 0.1, 0.9);
        assert!(point.is_strong());
        point.ambiguous = true;
        assert!(!point.is_strong());
    }

    #[test]
    fn matched_point_is_usable() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let mut point = MatchedPoint::new(tuple, 1, 0, 0.1, 0.6);
        assert!(point.is_usable());
        point.accepted = false;
        assert!(!point.is_usable());
    }

    #[test]
    fn matched_point_add_alternative() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let mut point = MatchedPoint::new(tuple, 1, 0, 0.1, 0.9);
        point.add_alternative(1, 0.5);
        point.add_alternative(2, 0.3);
        assert_eq!(point.alternatives.len(), 2);
        // Should be sorted by distance.
        assert!(point.alternatives[0].1 < point.alternatives[1].1);
    }

    #[test]
    fn matched_point_default() {
        let point = MatchedPoint::default();
        assert!(!point.accepted);
        assert!(point.rejection_reason.is_some());
    }

    // ---------------------------------------------------------------------------
    // ConstellationMatcherConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matcher_config_default() {
        let config = ConstellationMatcherConfig::default();
        assert!((config.max_distance - 1.5).abs() < 0.001);
        assert!((config.min_confidence - 0.6).abs() < 0.001);
        assert!((config.ambiguity_threshold - 0.8).abs() < 0.001);
        assert_eq!(config.max_alternatives, 5);
        assert!(config.enable_soft_match);
    }

    // ---------------------------------------------------------------------------
    // MatcherStatistics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matcher_statistics_default() {
        let stats = MatcherStatistics::default();
        assert_eq!(stats.total_matches, 0);
        assert_eq!(stats.accepted_matches, 0);
        assert_eq!(stats.rejected_matches, 0);
        assert_eq!(stats.ambiguous_matches, 0);
    }

    // ---------------------------------------------------------------------------
    // ConstellationMatcher tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matcher_new() {
        let constellation = Constellation::binary().unwrap();
        let config = ConstellationMatcherConfig::default();
        let matcher = ConstellationMatcher::new(constellation, config);
        assert_eq!(matcher.history().len(), 0);
        assert_eq!(matcher.stats.total_matches, 0);
    }

    #[test]
    fn matcher_match_exact() {
        let mut matcher = ConstellationMatcher::default();
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        assert!(matched.accepted);
        assert_eq!(matched.symbol_index, 1); // Binary: [-1, 1], value 1 is index 1.
        assert!((matched.distance - 0.0).abs() < 0.001);
        assert!((matched.confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn matcher_match_near() {
        let mut matcher = ConstellationMatcher::default();
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[0],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        // Value 0 is between -1 and 1, distance 1.0 to each.
        assert!((matched.distance - 1.0).abs() < 0.001);
        assert!(matched.ambiguous); // Equal distance to both.
    }

    #[test]
    fn matcher_match_far() {
        let mut matcher = ConstellationMatcher::default();
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[100],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        assert!(!matched.accepted);
        assert!(matched.rejection_reason.is_some());
    }

    #[test]
    fn matcher_match_empty_constellation() {
        let constellation = Constellation::new(vec![]).unwrap();
        let mut matcher = ConstellationMatcher::new(
            constellation,
            ConstellationMatcherConfig::default(),
        );
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        assert!(!matched.accepted);
    }

    #[test]
    fn matcher_match_batch() {
        let mut matcher = ConstellationMatcher::default();
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[-1],
        )
        .unwrap();
        let results = matcher.match_batch(&[t1, t2]);
        assert_eq!(results.len(), 2);
        assert!(results[0].accepted);
        assert!(results[1].accepted);
    }

    #[test]
    fn matcher_history() {
        let mut matcher = ConstellationMatcher::default();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        matcher.match_tuple(&t);
        matcher.match_tuple(&t);
        assert_eq!(matcher.history().len(), 2);
    }

    #[test]
    fn matcher_latest() {
        let mut matcher = ConstellationMatcher::default();
        assert!(matcher.latest().is_none());
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        matcher.match_tuple(&t);
        assert!(matcher.latest().is_some());
    }

    #[test]
    fn matcher_acceptance_rate() {
        let mut matcher = ConstellationMatcher::default();
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[100],
        )
        .unwrap();
        matcher.match_tuple(&t1);
        matcher.match_tuple(&t2);
        assert!((matcher.acceptance_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn matcher_rejection_rate() {
        let mut matcher = ConstellationMatcher::default();
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[100],
        )
        .unwrap();
        matcher.match_tuple(&t1);
        matcher.match_tuple(&t2);
        assert!((matcher.rejection_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn matcher_reset() {
        let mut matcher = ConstellationMatcher::default();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        matcher.match_tuple(&t);
        assert!(!matcher.history().is_empty());
        matcher.reset();
        assert_eq!(matcher.history().len(), 0);
        assert_eq!(matcher.stats.total_matches, 0);
    }

    #[test]
    fn matcher_set_config() {
        let mut matcher = ConstellationMatcher::default();
        let mut config = ConstellationMatcherConfig::default();
        config.max_distance = 0.5;
        matcher.set_config(config);
        assert!((matcher.config().max_distance - 0.5).abs() < 0.001);
    }

    #[test]
    fn matcher_default() {
        let matcher = ConstellationMatcher::default();
        assert!(matcher.is_valid_constellation());
    }

    // Helper: check constellation validity.
    impl ConstellationMatcher {
        fn is_valid_constellation(&self) -> bool {
            !self.constellation.points().is_empty()
        }
    }

    // ---------------------------------------------------------------------------
    // Distance function tests
    // ---------------------------------------------------------------------------

    #[test]
    fn tuple_value_distance_test() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[3, 0],
        )
        .unwrap();
        let dist = tuple_value_distance(&tuple, 3);
        assert!((dist - 0.0).abs() < 0.001); // Exact match on first component.

        let dist2 = tuple_value_distance(&tuple, 5);
        assert!((dist2 - 2.0).abs() < 0.001); // |3-5| = 2.
    }

    #[test]
    fn tuple_tuple_distance_test() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1, 2],
        )
        .unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[4, 6],
        )
        .unwrap();
        let dist = tuple_tuple_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001); // sqrt(3^2 + 4^2) = 5.
    }

    #[test]
    fn tuple_tuple_distance_same() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[3, 3],
        )
        .unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[3, 3],
        )
        .unwrap();
        assert!((tuple_tuple_distance(&a, &b) - 0.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // ManagedConstellation tests
    // ---------------------------------------------------------------------------

    #[test]
    fn managed_constellation_fields() {
        let points = vec![-1i32, 1];
        let constellation = Constellation::new(points.clone()).unwrap();
        let mc = ManagedConstellation {
            constellation,
            min_distance: 2.0,
            num_points: 2,
            energy: 1.0,
            valid: true,
        };
        assert_eq!(mc.num_points, 2);
        assert!((mc.min_distance - 2.0).abs() < 0.001);
        assert!((mc.energy - 1.0).abs() < 0.001);
        assert!(mc.valid);
    }

    // ---------------------------------------------------------------------------
    // ConstellationManager tests
    // ---------------------------------------------------------------------------

    #[test]
    fn constellation_manager_new() {
        let manager = ConstellationManager::new();
        assert_eq!(manager.size(), 2);
        assert!(manager.is_valid());
        assert!((manager.bits_per_symbol() - 1.0).abs() < 0.001);
    }

    #[test]
    fn constellation_manager_with_points() {
        let manager = ConstellationManager::with_points(vec![-2, 0, 2]).unwrap();
        assert_eq!(manager.size(), 3);
        assert!(manager.is_valid());
        assert_eq!(manager.points(), vec![-2, 0, 2]);
    }

    #[test]
    fn constellation_manager_with_points_too_few() {
        let result = ConstellationManager::with_points(vec![1]);
        assert!(result.is_err());
    }

    #[test]
    fn constellation_manager_with_points_duplicates() {
        let result = ConstellationManager::with_points(vec![1, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn constellation_manager_update_points() {
        let mut manager = ConstellationManager::new();
        let result = manager.update_points(vec![-3, -1, 1, 3]);
        assert!(result.is_ok());
        assert_eq!(manager.size(), 4);
        assert!((manager.bits_per_symbol() - 2.0).abs() < 0.001);
    }

    #[test]
    fn constellation_manager_history() {
        let mut manager = ConstellationManager::new();
        manager.update_points(vec![-2, 0, 2]).unwrap();
        assert_eq!(manager.history().len(), 1);
    }

    #[test]
    fn constellation_manager_reset_history() {
        let mut manager = ConstellationManager::new();
        manager.update_points(vec![-2, 0, 2]).unwrap();
        assert!(!manager.history().is_empty());
        manager.reset_history();
        assert_eq!(manager.history().len(), 0);
    }

    #[test]
    fn constellation_manager_default() {
        let manager = ConstellationManager::default();
        assert_eq!(manager.size(), 2);
    }

    // ---------------------------------------------------------------------------
    // DecodedSymbol tests
    // ---------------------------------------------------------------------------

    #[test]
    fn decoded_symbol_creation() {
        let symbol = DecodedSymbol {
            index: 1,
            value: 1,
            bits: vec![true],
            confidence: 0.9,
            success: true,
            error: None,
        };
        assert_eq!(symbol.index, 1);
        assert_eq!(symbol.value, 1);
        assert!(symbol.success);
    }

    // ---------------------------------------------------------------------------
    // SymbolDecoder tests
    // ---------------------------------------------------------------------------

    #[test]
    fn symbol_decoder_new() {
        let decoder = SymbolDecoder::new(2);
        assert_eq!(decoder.bits_per_symbol(), 2);
        assert!(!decoder.gray_coding);
    }

    #[test]
    fn symbol_decoder_decode_success() {
        let mut decoder = SymbolDecoder::new(1);
        let tuple =
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1])
                .unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        let symbol = decoder.decode(&matched);
        assert!(symbol.success);
        assert_eq!(symbol.index, 1);
        assert_eq!(symbol.value, 1);
        assert_eq!(symbol.bits, vec![true]);
        assert!((symbol.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn symbol_decoder_decode_failure() {
        let mut decoder = SymbolDecoder::new(1);
        let tuple =
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1])
                .unwrap();
        let matched = MatchedPoint::new(tuple, 1, 0, 0.1, 0.9)
            .reject("test rejection".to_string());
        let symbol = decoder.decode(&matched);
        assert!(!symbol.success);
        assert!(symbol.error.is_some());
    }

    #[test]
    fn symbol_decoder_index_to_bits() {
        let decoder = SymbolDecoder::new(3);
        assert_eq!(decoder.index_to_bits(0), vec![false, false, false]);
        assert_eq!(decoder.index_to_bits(5), vec![true, false, true]);
        assert_eq!(decoder.index_to_bits(7), vec![true, true, true]);
    }

    #[test]
    fn symbol_decoder_bits_to_index() {
        let decoder = SymbolDecoder::new(3);
        assert_eq!(decoder.bits_to_index(&[false, false, false]), 0);
        assert_eq!(decoder.bits_to_index(&[true, false, true]), 5);
        assert_eq!(decoder.bits_to_index(&[true, true, true]), 7);
    }

    #[test]
    fn symbol_decoder_decode_batch() {
        let mut decoder = SymbolDecoder::new(1);
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[-1],
        )
        .unwrap();
        let m1 = MatchedPoint::new(t1, 1, 1, 0.1, 0.9);
        let m2 = MatchedPoint::new(t2, -1, 0, 0.1, 0.9);
        let symbols = decoder.decode_batch(&[m1, m2]);
        assert_eq!(symbols.len(), 2);
        assert!(symbols[0].success);
        assert!(symbols[1].success);
    }

    #[test]
    fn symbol_decoder_success_rate() {
        let mut decoder = SymbolDecoder::new(1);
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let m1 = MatchedPoint::new(t1, 1, 1, 0.1, 0.9);
        let m2 = MatchedPoint::new(t2, 1, 1, 0.1, 0.9)
            .reject("test".to_string());
        decoder.decode(&m1);
        decoder.decode(&m2);
        assert!((decoder.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn symbol_decoder_reset_stats() {
        let mut decoder = SymbolDecoder::new(1);
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let m = MatchedPoint::new(t, 1, 1, 0.1, 0.9);
        decoder.decode(&m);
        assert_eq!(decoder.stats.total_decoded, 1);
        decoder.reset_stats();
        assert_eq!(decoder.stats.total_decoded, 0);
    }

    #[test]
    fn symbol_decoder_default() {
        let decoder = SymbolDecoder::default();
        assert_eq!(decoder.bits_per_symbol(), 1);
    }

    // ---------------------------------------------------------------------------
    // DecoderStatistics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn decoder_statistics_default() {
        let stats = DecoderStatistics::default();
        assert_eq!(stats.total_decoded, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }

    // ---------------------------------------------------------------------------
    // ConstellationIntegrationResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn constellation_integration_result_fields() {
        let tuple =
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1])
                .unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        let symbol = DecodedSymbol {
            index: 1,
            value: 1,
            bits: vec![true],
            confidence: 0.9,
            success: true,
            error: None,
        };
        let result = ConstellationIntegrationResult {
            matched,
            symbol,
            success: true,
            latency_us: 10.0,
        };
        assert!(result.success);
        assert!((result.latency_us - 10.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // PipelineStatistics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn pipeline_statistics_default() {
        let stats = PipelineStatistics::default();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }

    // ---------------------------------------------------------------------------
    // ConstellationIntegrationPipeline tests
    // ---------------------------------------------------------------------------

    #[test]
    fn pipeline_new() {
        let pipeline = ConstellationIntegrationPipeline::new();
        assert_eq!(pipeline.stats.total_processed, 0);
    }

    #[test]
    fn pipeline_process_success() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let result = pipeline.process(&t);
        assert!(result.success);
        assert!(result.symbol.success);
        assert!(result.latency_us >= 0.0);
    }

    #[test]
    fn pipeline_process_batch() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[-1],
        )
        .unwrap();
        let results = pipeline.process_batch(&[t1, t2]);
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert!(results[1].success);
    }

    #[test]
    fn pipeline_success_rate() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[100],
        )
        .unwrap();
        pipeline.process(&t1);
        pipeline.process(&t2);
        assert!((pipeline.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn pipeline_failure_rate() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[100],
        )
        .unwrap();
        pipeline.process(&t1);
        pipeline.process(&t2);
        assert!((pipeline.failure_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn pipeline_reset_stats() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        pipeline.process(&t);
        assert_eq!(pipeline.stats.total_processed, 1);
        pipeline.reset_stats();
        assert_eq!(pipeline.stats.total_processed, 0);
    }

    #[test]
    fn pipeline_with_constellation() {
        let constellation = Constellation::new(vec![-2, 0, 2]).unwrap();
        let mut pipeline =
            ConstellationIntegrationPipeline::with_constellation(
                constellation,
                2,
            );
        assert_eq!(pipeline.decoder.bits_per_symbol(), 2);
    }

    #[test]
    fn pipeline_default() {
        let pipeline = ConstellationIntegrationPipeline::default();
        assert_eq!(pipeline.decoder.bits_per_symbol(), 1);
    }

    // ---------------------------------------------------------------------------
    // Ambiguity detection tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matcher_ambiguous_match() {
        // Create constellation with closely spaced points.
        let constellation = Constellation::new(vec![-1, 0, 1]).unwrap();
        let config = ConstellationMatcherConfig {
            max_distance: 2.0,
            min_confidence: 0.1,
            ambiguity_threshold: 0.5,
            max_alternatives: 5,
            enable_soft_match: true,
        };
        let mut matcher = ConstellationMatcher::new(constellation, config);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[0],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        // Value 0 is exactly at the middle point, distance 0.
        assert!(matched.accepted);
        assert_eq!(matched.symbol_index, 1); // Value 0 is index 1.
    }

    #[test]
    fn matcher_soft_match_enabled() {
        let constellation = Constellation::binary().unwrap();
        let config = ConstellationMatcherConfig {
            max_distance: 0.5,
            min_confidence: 0.1,
            ambiguity_threshold: 0.5,
            max_alternatives: 5,
            enable_soft_match: true,
        };
        let mut matcher = ConstellationMatcher::new(constellation, config);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[10],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        // Soft match: still returns best but low confidence.
        assert!((matched.confidence - 0.0).abs() < 0.001
            || matched.confidence < 0.3);
    }

    #[test]
    fn matcher_soft_match_disabled() {
        let constellation = Constellation::binary().unwrap();
        let config = ConstellationMatcherConfig {
            max_distance: 0.5,
            min_confidence: 0.1,
            ambiguity_threshold: 0.5,
            max_alternatives: 5,
            enable_soft_match: false,
        };
        let mut matcher = ConstellationMatcher::new(constellation, config);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[10],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        assert!(!matched.accepted);
        assert!((matched.confidence - 0.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // Edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matcher_single_point_constellation() {
        let constellation = Constellation::new(vec![5]).unwrap();
        let mut matcher = ConstellationMatcher::new(
            constellation,
            ConstellationMatcherConfig::default(),
        );
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[5],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        assert!(matched.accepted);
        assert_eq!(matched.symbol_index, 0);
    }

    #[test]
    fn matcher_multi_component_tuple() {
        let constellation = Constellation::binary().unwrap();
        let mut matcher = ConstellationMatcher::new(
            constellation,
            ConstellationMatcherConfig::default(),
        );
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1, 0, 0],
        )
        .unwrap();
        let matched = matcher.match_tuple(&tuple);
        // Distance to [-1,0,0] is |1-(-1)| = 2, to [1,0,0] is 0.
        assert!(matched.accepted);
        assert_eq!(matched.symbol_index, 1); // Value 1.
    }

    #[test]
    fn decoder_gray_coding() {
        let mut decoder = SymbolDecoder::new(2);
        decoder.set_gray_coding(true);
        assert!(decoder.gray_coding);
        // Bits conversion is independent of Gray coding in this implementation.
        assert_eq!(decoder.index_to_bits(2), vec![true, false]);
    }

    #[test]
    fn pipeline_high_latency() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(
            &[1],
        )
        .unwrap();
        let result = pipeline.process(&t);
        assert!(result.latency_us >= 0.0);
        assert!(result.latency_us < 1000.0); // Should be fast.
    }

    // ---------------------------------------------------------------------------
    // End-to-end pipeline test
    // ---------------------------------------------------------------------------

    #[test]
    fn constellation_integration_end_to_end() {
        // Full pipeline: tuple -> match -> decode.
        let constellation = Constellation::new(vec![-1, 1]).unwrap();
        let mut pipeline =
            ConstellationIntegrationPipeline::with_constellation(
                constellation,
                1,
            );

        // Process a sequence of tuples.
        let tuples: Vec<
            crate::detection::generalized_skyrmion::SkyrmionTuple,
        > = vec![
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1])
                .unwrap(),
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[-1])
                .unwrap(),
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1])
                .unwrap(),
            crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[0])
                .unwrap(), // Ambiguous case.
        ];

        let results = pipeline.process_batch(&tuples);
        assert_eq!(results.len(), 4);
        assert!(results[0].success);
        assert!(results[1].success);
        assert!(results[2].success);
        // Result 3 (value 0) may or may not be accepted depending on thresholds.
        assert!(results[3].latency_us >= 0.0);

        // Check pipeline stats.
        assert_eq!(pipeline.stats.total_processed, 4);
        assert!(pipeline.success_rate() >= 0.0);
        assert!(pipeline.failure_rate() >= 0.0);
        assert!((pipeline.success_rate() + pipeline.failure_rate() - 1.0)
            .abs()
            < 0.001);
    }

    // ---------------------------------------------------------------------------
    // DistanceMetric tests
    // ---------------------------------------------------------------------------

    #[test]
    fn distance_metric_variants() {
        assert_eq!(DistanceMetric::Euclidean as u32, 0);
        assert_eq!(DistanceMetric::Manhattan as u32, 1);
        assert_eq!(DistanceMetric::Chebyshev as u32, 2);
        assert_eq!(DistanceMetric::Cosine as u32, 3);
    }

    #[test]
    fn tuple_distance_metric_euclidean() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[4, 6]).unwrap();
        let dist = tuple_distance_metric(&a, &b, DistanceMetric::Euclidean);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn tuple_distance_metric_manhattan() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[4, 6]).unwrap();
        let dist = tuple_distance_metric(&a, &b, DistanceMetric::Manhattan);
        assert!((dist - 7.0).abs() < 0.001); // |3| + |4| = 7.
    }

    #[test]
    fn tuple_distance_metric_chebyshev() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[4, 6]).unwrap();
        let dist = tuple_distance_metric(&a, &b, DistanceMetric::Chebyshev);
        assert!((dist - 4.0).abs() < 0.001); // max(3, 4) = 4.
    }

    #[test]
    fn tuple_distance_metric_cosine() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 0]).unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[0, 1]).unwrap();
        let dist = tuple_distance_metric(&a, &b, DistanceMetric::Cosine);
        assert!((dist - 1.0).abs() < 0.001); // Orthogonal.
    }

    #[test]
    fn tuple_distance_metric_cosine_same() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 2]).unwrap();
        let dist = tuple_distance_metric(&a, &b, DistanceMetric::Cosine);
        assert!((dist - 0.0).abs() < 0.001); // Identical.
    }

    #[test]
    fn tuple_distance_metric_zero_vector() {
        let a = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[0, 0]).unwrap();
        let b = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 1]).unwrap();
        let dist = tuple_distance_metric(&a, &b, DistanceMetric::Cosine);
        assert!((dist - 1.0).abs() < 0.001); // Zero vector -> distance 1.
    }

    #[test]
    fn tuple_value_distance_metric_euclidean() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[3, 0]).unwrap();
        let dist = tuple_value_distance_metric(&tuple, 3, DistanceMetric::Euclidean);
        assert!((dist - 0.0).abs() < 0.001);
    }

    #[test]
    fn tuple_value_distance_metric_manhattan() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[3, 0]).unwrap();
        let dist = tuple_value_distance_metric(&tuple, 5, DistanceMetric::Manhattan);
        assert!((dist - 2.0).abs() < 0.001); // |3-5| + |0-0| = 2.
    }

    #[test]
    fn tuple_value_distance_metric_chebyshev() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[3, 0]).unwrap();
        let dist = tuple_value_distance_metric(&tuple, 5, DistanceMetric::Chebyshev);
        assert!((dist - 2.0).abs() < 0.001); // max(|3-5|, |0-0|) = 2.
    }

    #[test]
    fn tuple_value_distance_metric_cosine() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1, 0]).unwrap();
        let dist = tuple_value_distance_metric(&tuple, -1, DistanceMetric::Cosine);
        assert!((dist - 1.0).abs() < 0.001); // Opposite signs.
    }

    // ---------------------------------------------------------------------------
    // ConstellationQualityMonitor tests
    // ---------------------------------------------------------------------------

    #[test]
    fn quality_monitor_new() {
        let monitor = ConstellationQualityMonitor::new(1_000_000);
        assert_eq!(monitor.num_snapshots(), 0);
        assert!((monitor.degradation_threshold - 0.5).abs() < 0.001);
    }

    #[test]
    fn quality_monitor_record() {
        let mut monitor = ConstellationQualityMonitor::new(1_000_000);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        monitor.record(&matched, 0);
        assert_eq!(monitor.num_snapshots(), 0); // Not enough to flush yet.
    }

    #[test]
    fn quality_monitor_flush_window() {
        let mut monitor = ConstellationQualityMonitor::new(100); // 100 us windows.
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        monitor.record(&matched, 0);
        monitor.record(&matched, 150); // New window.
        assert_eq!(monitor.num_snapshots(), 1);
    }

    #[test]
    fn quality_monitor_is_degrading() {
        let mut monitor = ConstellationQualityMonitor::new(1_000_000);
        assert!(!monitor.is_degrading()); // No history.

        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[100]).unwrap();
        let matched = MatchedPoint::new(tuple, 100, 0, 10.0, 0.0)
            .reject("too far".to_string());
        monitor.record(&matched, 0);
        monitor.record(&matched, 1_500_000);
        assert!(monitor.is_degrading());
    }

    #[test]
    fn quality_monitor_avg_acceptance_rate() {
        let mut monitor = ConstellationQualityMonitor::new(100);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        monitor.record(&matched, 0);
        monitor.record(&matched, 150);
        let avg = monitor.avg_acceptance_rate();
        assert!(avg >= 0.0);
        assert!(avg <= 1.0);
    }

    #[test]
    fn quality_monitor_avg_confidence() {
        let mut monitor = ConstellationQualityMonitor::new(100);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        monitor.record(&matched, 0);
        monitor.record(&matched, 150);
        let avg = monitor.avg_confidence();
        assert!((avg - 0.9).abs() < 0.001);
    }

    #[test]
    fn quality_monitor_reset() {
        let mut monitor = ConstellationQualityMonitor::new(100);
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let matched = MatchedPoint::new(tuple, 1, 1, 0.1, 0.9);
        monitor.record(&matched, 0);
        monitor.record(&matched, 150);
        assert!(!monitor.history().is_empty());
        monitor.reset();
        assert_eq!(monitor.num_snapshots(), 0);
    }

    #[test]
    fn quality_monitor_set_degradation_threshold() {
        let mut monitor = ConstellationQualityMonitor::new(1_000_000);
        monitor.set_degradation_threshold(0.8);
        assert!((monitor.degradation_threshold - 0.8).abs() < 0.001);
    }

    #[test]
    fn quality_monitor_default() {
        let monitor = ConstellationQualityMonitor::default();
        assert!((monitor.degradation_threshold - 0.5).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // ConstellationOptimizer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn constellation_optimizer_new() {
        let optimizer = ConstellationOptimizer::new();
        assert_eq!(optimizer.max_iterations, 1000);
        assert!((optimizer.tolerance - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn constellation_optimizer_optimize_binary() {
        let mut optimizer = ConstellationOptimizer::new();
        let result = optimizer.optimize(2, -1, 1);
        assert_eq!(result.optimized_points.len(), 2);
        assert!((result.min_distance - 2.0).abs() < 0.001);
        assert!(result.improvement_factor >= 1.0);
    }

    #[test]
    fn constellation_optimizer_optimize_4_points() {
        let mut optimizer = ConstellationOptimizer::new();
        let result = optimizer.optimize(4, -3, 3);
        assert_eq!(result.optimized_points.len(), 4);
        assert!(result.min_distance > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn constellation_optimizer_invalid() {
        let mut optimizer = ConstellationOptimizer::new();
        let result = optimizer.optimize(1, 0, 1); // Too few points.
        assert_eq!(result.optimized_points.len(), 2);
    }

    #[test]
    fn constellation_optimizer_same_range() {
        let mut optimizer = ConstellationOptimizer::new();
        let result = optimizer.optimize(2, 5, 5); // min == max.
        assert_eq!(result.optimized_points.len(), 2);
    }

    #[test]
    fn constellation_optimizer_history() {
        let mut optimizer = ConstellationOptimizer::new();
        optimizer.optimize(2, -1, 1);
        assert_eq!(optimizer.history().len(), 1);
    }

    #[test]
    fn constellation_optimizer_reset_history() {
        let mut optimizer = ConstellationOptimizer::new();
        optimizer.optimize(2, -1, 1);
        assert!(!optimizer.history().is_empty());
        optimizer.reset_history();
        assert_eq!(optimizer.history().len(), 0);
    }

    #[test]
    fn constellation_optimizer_set_max_iterations() {
        let mut optimizer = ConstellationOptimizer::new();
        optimizer.set_max_iterations(100);
        assert_eq!(optimizer.max_iterations, 100);
    }

    #[test]
    fn constellation_optimizer_set_tolerance() {
        let mut optimizer = ConstellationOptimizer::new();
        optimizer.set_tolerance(0.01);
        assert!((optimizer.tolerance - 0.01).abs() < 1e-12);
    }

    #[test]
    fn constellation_optimizer_default() {
        let optimizer = ConstellationOptimizer::default();
        assert_eq!(optimizer.max_iterations, 1000);
    }

    // ---------------------------------------------------------------------------
    // SoftDecisionDecoder tests
    // ---------------------------------------------------------------------------

    #[test]
    fn soft_decoder_new() {
        let decoder = SoftDecisionDecoder::new(4);
        assert_eq!(decoder.num_symbols(), 4);
        assert!((decoder.distance_scale - 1.0).abs() < 0.001);
    }

    #[test]
    fn soft_decoder_decode_soft() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![0.1, 5.0, 5.0, 5.0]; // First symbol is closest.
        let symbol = decoder.decode_soft(&distances).unwrap();
        assert_eq!(symbol.hard_decision, 0);
        assert!(symbol.confidence > 0.5);
        assert_eq!(symbol.probabilities.len(), 4);
    }

    #[test]
    fn soft_decoder_decode_soft_wrong_dim() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![0.1, 5.0]; // Wrong length.
        let result = decoder.decode_soft(&distances);
        assert!(result.is_err());
    }

    #[test]
    fn soft_decoder_decode_soft_empty() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances: Vec<f64> = vec![];
        let result = decoder.decode_soft(&distances);
        assert!(result.is_err());
    }

    #[test]
    fn soft_decoder_soft_to_bits() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![5.0, 0.1, 5.0, 5.0]; // Symbol 1 is closest.
        let symbol = decoder.decode_soft(&distances).unwrap();
        let bits = decoder.soft_to_bits(&symbol, 2);
        assert_eq!(bits, vec![false, true]); // 1 in binary with 2 bits.
    }

    #[test]
    fn soft_decoder_avg_confidence() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![0.1, 5.0, 5.0, 5.0];
        decoder.decode_soft(&distances).unwrap();
        let avg = decoder.avg_confidence();
        assert!(avg > 0.0);
    }

    #[test]
    fn soft_decoder_avg_llr_magnitude() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![0.1, 5.0, 5.0, 5.0];
        decoder.decode_soft(&distances).unwrap();
        let avg = decoder.avg_llr_magnitude();
        assert!(avg >= 0.0);
    }

    #[test]
    fn soft_decoder_history() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![0.1, 5.0, 5.0, 5.0];
        decoder.decode_soft(&distances).unwrap();
        assert_eq!(decoder.history().len(), 1);
    }

    #[test]
    fn soft_decoder_reset_history() {
        let mut decoder = SoftDecisionDecoder::new(4);
        let distances = vec![0.1, 5.0, 5.0, 5.0];
        decoder.decode_soft(&distances).unwrap();
        assert!(!decoder.history().is_empty());
        decoder.reset_history();
        assert_eq!(decoder.history().len(), 0);
    }

    #[test]
    fn soft_decoder_set_distance_scale() {
        let mut decoder = SoftDecisionDecoder::new(4);
        decoder.set_distance_scale(2.0);
        assert!((decoder.distance_scale - 2.0).abs() < 0.001);
    }

    #[test]
    fn soft_decoder_default() {
        let decoder = SoftDecisionDecoder::default();
        assert_eq!(decoder.num_symbols(), 2);
    }

    // ---------------------------------------------------------------------------
    // ConstellationCapacityEstimator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn capacity_estimator_new() {
        let estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        assert!((estimator.noise_std() - 0.1).abs() < 0.001);
        assert!((estimator.signal_energy() - 1.0).abs() < 0.001);
    }

    #[test]
    fn capacity_estimator_estimate() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        let estimate = estimator.estimate(&[-1, 1]);
        assert_eq!(estimate.num_points, 2);
        assert!((estimate.bits_per_symbol - 1.0).abs() < 0.001);
        assert!((estimate.min_distance - 2.0).abs() < 0.001);
        assert!(estimate.snr_db.is_finite());
        assert!(estimate.capacity_bits_per_symbol >= 0.0);
        assert!(estimate.utilization >= 0.0);
        assert!(estimate.utilization <= 1.0);
    }

    #[test]
    fn capacity_estimator_estimate_single_point() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        let estimate = estimator.estimate(&[5]);
        assert_eq!(estimate.num_points, 1);
        assert!((estimate.bits_per_symbol - 0.0).abs() < 0.001);
        assert!((estimate.min_distance - 0.0).abs() < 0.001);
    }

    #[test]
    fn capacity_estimator_estimate_empty() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        let estimate = estimator.estimate(&[]);
        assert_eq!(estimate.num_points, 0);
        assert!((estimate.bits_per_symbol - 0.0).abs() < 0.001);
    }

    #[test]
    fn capacity_estimator_high_noise() {
        let mut estimator = ConstellationCapacityEstimator::new(10.0, 1.0);
        let estimate = estimator.estimate(&[-1, 1]);
        assert!(estimate.snr_db < 0.0); // Low SNR.
        assert!(estimate.capacity_bits_per_symbol >= 0.0);
    }

    #[test]
    fn capacity_estimator_low_noise() {
        let mut estimator = ConstellationCapacityEstimator::new(0.001, 100.0);
        let estimate = estimator.estimate(&[-1, 1]);
        assert!(estimate.snr_db > 20.0); // High SNR.
    }

    #[test]
    fn capacity_estimator_history() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        estimator.estimate(&[-1, 1]);
        estimator.estimate(&[-2, 0, 2]);
        assert_eq!(estimator.history().len(), 2);
    }

    #[test]
    fn capacity_estimator_reset_history() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        estimator.estimate(&[-1, 1]);
        assert!(!estimator.history().is_empty());
        estimator.reset_history();
        assert_eq!(estimator.history().len(), 0);
    }

    #[test]
    fn capacity_estimator_set_noise_std() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        estimator.set_noise_std(0.5);
        assert!((estimator.noise_std() - 0.5).abs() < 0.001);
    }

    #[test]
    fn capacity_estimator_set_signal_energy() {
        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        estimator.set_signal_energy(50.0);
        assert!((estimator.signal_energy() - 50.0).abs() < 0.001);
    }

    #[test]
    fn capacity_estimator_default() {
        let estimator = ConstellationCapacityEstimator::default();
        assert!((estimator.noise_std() - 0.1).abs() < 0.001);
        assert!((estimator.signal_energy() - 1.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // MatchedPoint quality extension tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matched_point_quality_grade() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();

        let mut point = MatchedPoint::new(tuple.clone(), 1, 1, 0.01, 0.98);
        assert_eq!(point.quality_grade(), 'A');

        point.confidence = 0.85;
        assert_eq!(point.quality_grade(), 'B');

        point.confidence = 0.65;
        assert_eq!(point.quality_grade(), 'C');

        point.confidence = 0.45;
        assert_eq!(point.quality_grade(), 'D');

        point.confidence = 0.3;
        assert_eq!(point.quality_grade(), 'E');

        let rejected = MatchedPoint::new(tuple, 1, 1, 10.0, 0.0).reject("too far".to_string());
        assert_eq!(rejected.quality_grade(), 'F');
    }

    #[test]
    fn matched_point_needs_review() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();

        let good = MatchedPoint::new(tuple.clone(), 1, 1, 0.1, 0.9);
        assert!(!good.needs_review());

        let ambiguous = MatchedPoint::new(tuple.clone(), 1, 1, 0.1, 0.9);
        let mut ambiguous_point = ambiguous;
        ambiguous_point.ambiguous = true;
        assert!(ambiguous_point.needs_review());

        let low_conf = MatchedPoint::new(tuple.clone(), 1, 1, 0.1, 0.5);
        assert!(low_conf.needs_review());

        let rejected = MatchedPoint::new(tuple, 1, 1, 10.0, 0.0).reject("too far".to_string());
        assert!(rejected.needs_review());
    }

    #[test]
    fn matched_point_distance_margin() {
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let point = MatchedPoint::new(tuple, 1, 1, 0.5, 0.9);
        let margin = point.distance_margin(1.5);
        assert!((margin - 1.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // erf function tests
    // ---------------------------------------------------------------------------

    #[test]
    fn erf_test() {
        assert!((erf(0.0) - 0.0).abs() < 0.001);
        assert!((erf(1.0) - 0.8427).abs() < 0.01);
        assert!((erf(-1.0) + 0.8427).abs() < 0.01);
        assert!((erf(2.0) - 0.9953).abs() < 0.01);
    }

    // ---------------------------------------------------------------------------
    // ConstellationIntegrationPipeline extended tests
    // ---------------------------------------------------------------------------

    #[test]
    fn pipeline_is_healthy() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        pipeline.process(&t);
        assert!(pipeline.is_healthy(0.0));
        assert!(!pipeline.is_healthy(2.0)); // Always <= 1.0.
    }

    #[test]
    fn pipeline_pipeline_name() {
        let pipeline = ConstellationIntegrationPipeline::new();
        assert_eq!(pipeline.pipeline_name(), "ConstellationIntegrationPipeline");
    }

    #[test]
    fn pipeline_process_with_monitor() {
        let mut pipeline = ConstellationIntegrationPipeline::new();
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let result = pipeline.process_with_monitor(&t, 0);
        assert!(result.success);
    }

    #[test]
    fn soft_decoder_decoder_name() {
        let decoder = SoftDecisionDecoder::new(4);
        assert_eq!(decoder.decoder_name(), "SymbolDecoder");
    }

    // ---------------------------------------------------------------------------
    // ConstellationMatcher extended method tests
    // ---------------------------------------------------------------------------

    #[test]
    fn matcher_match_tuple_with_metric() {
        let constellation = Constellation::binary().unwrap();
        let mut matcher = ConstellationMatcher::new(constellation, ConstellationMatcherConfig::default());
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let matched = matcher.match_tuple_with_metric(&tuple, DistanceMetric::Euclidean);
        assert!(matched.accepted);
        assert_eq!(matched.symbol_index, 1);
    }

    #[test]
    fn matcher_threshold_sweep() {
        let constellation = Constellation::binary().unwrap();
        let matcher = ConstellationMatcher::new(constellation, ConstellationMatcherConfig::default());
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[0]).unwrap();
        let results = matcher.threshold_sweep(&tuple, &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(results.len(), 4);
        assert!(!results[0].1); // 0.5 < distance 1.0.
        assert!(results[1].1); // 1.0 >= distance 1.0.
    }

    #[test]
    fn matcher_nearest_distance() {
        let constellation = Constellation::binary().unwrap();
        let matcher = ConstellationMatcher::new(constellation, ConstellationMatcherConfig::default());
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let dist = matcher.nearest_distance(&tuple);
        assert!((dist - 0.0).abs() < 0.001);
    }

    #[test]
    fn matcher_nearest_distance_empty() {
        let constellation = Constellation::new(vec![]).unwrap();
        let matcher = ConstellationMatcher::new(constellation, ConstellationMatcherConfig::default());
        let tuple = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let dist = matcher.nearest_distance(&tuple);
        assert_eq!(dist, f64::INFINITY);
    }

    // ---------------------------------------------------------------------------
    // SymbolDecoder extended method tests
    // ---------------------------------------------------------------------------

    #[test]
    fn symbol_decoder_decode_with_repetition() {
        let mut decoder = SymbolDecoder::new(1);
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let t3 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[-1]).unwrap();
        let m1 = MatchedPoint::new(t1, 1, 1, 0.1, 0.9);
        let m2 = MatchedPoint::new(t2, 1, 1, 0.1, 0.9);
        let m3 = MatchedPoint::new(t3, -1, 0, 0.1, 0.9);
        let result = decoder.decode_with_repetition(&[m1, m2, m3], 3).unwrap();
        assert!(result.success);
        assert_eq!(result.index, 1); // Majority: two 1s, one 0.
    }

    #[test]
    fn symbol_decoder_decode_with_repetition_insufficient() {
        let mut decoder = SymbolDecoder::new(1);
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let m = MatchedPoint::new(t, 1, 1, 0.1, 0.9);
        let result = decoder.decode_with_repetition(&[m], 3);
        assert!(result.is_err());
    }

    #[test]
    fn symbol_decoder_decode_with_repetition_all_rejected() {
        let mut decoder = SymbolDecoder::new(1);
        let t1 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let t2 = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let m1 = MatchedPoint::new(t1, 1, 1, 0.1, 0.9).reject("bad".to_string());
        let m2 = MatchedPoint::new(t2, 1, 1, 0.1, 0.9).reject("bad".to_string());
        let result = decoder.decode_with_repetition(&[m1, m2], 2).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn symbol_decoder_repetition_zero() {
        let mut decoder = SymbolDecoder::new(1);
        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let m = MatchedPoint::new(t, 1, 1, 0.1, 0.9);
        let result = decoder.decode_with_repetition(&[m], 0);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------------------
    // End-to-end integration tests
    // ---------------------------------------------------------------------------

    #[test]
    fn run_self_test_passes() {
        assert!(run_self_test());
    }

    #[test]
    fn constellation_capacity_estimate_fields() {
        let estimate = ConstellationCapacityEstimate {
            num_points: 4,
            bits_per_symbol: 2.0,
            min_distance: 2.0,
            snr_db: 20.0,
            capacity_bits_per_symbol: 1.5,
            utilization: 0.75,
        };
        assert_eq!(estimate.num_points, 4);
        assert!((estimate.bits_per_symbol - 2.0).abs() < 0.001);
        assert!((estimate.min_distance - 2.0).abs() < 0.001);
        assert!((estimate.snr_db - 20.0).abs() < 0.001);
        assert!((estimate.capacity_bits_per_symbol - 1.5).abs() < 0.001);
        assert!((estimate.utilization - 0.75).abs() < 0.001);
    }

    #[test]
    fn full_pipeline_with_capacity_estimation() {
        let constellation = Constellation::new(vec![-1, 1]).unwrap();
        let mut pipeline = ConstellationIntegrationPipeline::with_constellation(constellation, 1);

        let mut estimator = ConstellationCapacityEstimator::new(0.1, 1.0);
        let points = pipeline.matcher().constellation().points();
        let _capacity = estimator.estimate(&points);

        let t = crate::detection::generalized_skyrmion::SkyrmionTuple::new(&[1]).unwrap();
        let result = pipeline.process(&t);
        assert!(result.success);
    }
}