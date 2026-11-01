//! Detection recovery — handles detection failures, synchronization losses,
//! corrupted measurements, and recovery scenarios for the topological signal
//! detection pipeline.
//!
//! This module provides strategies for recovering from detection failures,
//! including re-synchronization, fallback detection modes, state reset,
//! and graceful degradation when the primary detection path fails.
//!
//! All types are carrier-agnostic — no assumptions about physical medium,
//! transport, or modulation scheme.

use crate::errors::XpockError;
use crate::detection::generalized_skyrmion::SkyrmionTuple;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// RecoveryOutcome — result of a recovery attempt
// ============================================================================

/// The outcome of a recovery attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// Recovery succeeded, detection resumed normally.
    Success,
    /// Recovery partially succeeded — detection resumed with reduced confidence.
    Partial,
    /// Recovery failed — detection remains unavailable.
    Failed,
    /// Recovery is in progress.
    InProgress,
    /// Recovery skipped — no recovery needed.
    Skipped,
}

impl RecoveryOutcome {
    /// Returns true if the outcome represents any level of success.
    pub fn is_success(self) -> bool {
        matches!(self, RecoveryOutcome::Success | RecoveryOutcome::Partial)
    }

    /// Returns true if the outcome represents a terminal failure.
    pub fn is_failure(self) -> bool {
        matches!(self, RecoveryOutcome::Failed)
    }

    /// Returns true if recovery is still ongoing.
    pub fn is_ongoing(self) -> bool {
        matches!(self, RecoveryOutcome::InProgress)
    }
}

impl Default for RecoveryOutcome {
    fn default() -> Self {
        RecoveryOutcome::Skipped
    }
}

// ============================================================================
// RecoveryStrategy — strategy types for recovery
// ============================================================================

/// Recovery strategy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Immediate retry of the detection pipeline.
    ImmediateRetry,
    /// Retry with backoff (exponential delay between attempts).
    RetryWithBackoff,
    /// Switch to a fallback detection mode (lower quality).
    FallbackMode,
    /// Full re-synchronization (re-acquire signal from scratch).
    ReSynchronize,
    /// Reset all state and start fresh.
    ResetState,
    /// Degrade gracefully — continue with reduced functionality.
    GracefulDegradation,
    /// Abort detection and signal failure to the caller.
    Abort,
    /// Escalate to a higher-layer recovery mechanism.
    Escalate,
}

impl RecoveryStrategy {
    /// Returns true if this strategy involves retrying detection.
    pub fn is_retry(self) -> bool {
        matches!(
            self,
            RecoveryStrategy::ImmediateRetry | RecoveryStrategy::RetryWithBackoff
        )
    }

    /// Returns true if this strategy involves re-synchronization.
    pub fn is_resync(self) -> bool {
        matches!(self, RecoveryStrategy::ReSynchronize)
    }

    /// Returns true if this strategy resets state.
    pub fn is_reset(self) -> bool {
        matches!(self, RecoveryStrategy::ResetState)
    }

    /// Returns the default backoff delay (in milliseconds) for this strategy.
    pub fn default_backoff_ms(self) -> u64 {
        match self {
            RecoveryStrategy::ImmediateRetry => 0,
            RecoveryStrategy::RetryWithBackoff => 100,
            RecoveryStrategy::FallbackMode => 50,
            RecoveryStrategy::ReSynchronize => 500,
            RecoveryStrategy::ResetState => 200,
            RecoveryStrategy::GracefulDegradation => 0,
            RecoveryStrategy::Abort => 0,
            RecoveryStrategy::Escalate => 1000,
        }
    }

    /// Returns the maximum number of attempts for this strategy.
    pub fn max_attempts(self) -> u32 {
        match self {
            RecoveryStrategy::ImmediateRetry => 3,
            RecoveryStrategy::RetryWithBackoff => 5,
            RecoveryStrategy::FallbackMode => 1,
            RecoveryStrategy::ReSynchronize => 2,
            RecoveryStrategy::ResetState => 1,
            RecoveryStrategy::GracefulDegradation => 1,
            RecoveryStrategy::Abort => 1,
            RecoveryStrategy::Escalate => 1,
        }
    }
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        RecoveryStrategy::ImmediateRetry
    }
}

// ============================================================================
// DetectionRecovery — main recovery engine
// ============================================================================

/// Configuration for the detection recovery engine.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionRecoveryConfig {
    /// Default recovery strategy.
    pub default_strategy: RecoveryStrategy,
    /// Maximum number of recovery attempts before giving up.
    pub max_recovery_attempts: u32,
    /// Base backoff delay in milliseconds.
    pub base_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds.
    pub max_backoff_ms: u64,
    /// Backoff multiplier (exponential backoff factor).
    pub backoff_multiplier: f64,
    /// Timeout for individual recovery attempts (milliseconds).
    pub attempt_timeout_ms: u64,
    /// Enable automatic recovery on detection failure.
    pub auto_recover: bool,
    /// Enable fallback mode when primary detection fails.
    pub enable_fallback: bool,
    /// Fallback detection quality threshold (0.0 to 1.0).
    pub fallback_quality_threshold: f64,
    /// Maximum time in degraded mode before escalating (milliseconds).
    pub max_degraded_ms: u64,
}

impl Default for DetectionRecoveryConfig {
    fn default() -> Self {
        Self {
            default_strategy: RecoveryStrategy::ImmediateRetry,
            max_recovery_attempts: 5,
            base_backoff_ms: 100,
            max_backoff_ms: 5000,
            backoff_multiplier: 2.0,
            attempt_timeout_ms: 1000,
            auto_recover: true,
            enable_fallback: true,
            fallback_quality_threshold: 0.5,
            max_degraded_ms: 10000,
        }
    }
}

/// Statistics collected during recovery operations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecoveryStatistics {
    /// Total number of recovery attempts.
    pub total_attempts: u64,
    /// Number of successful recoveries.
    pub successful_recoveries: u64,
    /// Number of partial recoveries.
    pub partial_recoveries: u64,
    /// Number of failed recoveries.
    pub failed_recoveries: u64,
    /// Number of times fallback mode was activated.
    pub fallback_activations: u64,
    /// Number of times re-synchronization was performed.
    pub resync_count: u64,
    /// Number of times state was reset.
    pub reset_count: u64,
    /// Total time spent in recovery (milliseconds).
    pub total_recovery_time_ms: f64,
    /// Average recovery time per attempt (milliseconds).
    pub avg_recovery_time_ms: f64,
    /// Recovery attempts by strategy.
    pub strategy_counts: HashMap<RecoveryStrategy, u64>,
    /// Recovery outcomes by type.
    pub outcome_counts: HashMap<RecoveryOutcome, u64>,
    /// Longest recovery time (milliseconds).
    pub max_recovery_time_ms: f64,
    /// Shortest recovery time (milliseconds).
    pub min_recovery_time_ms: f64,
}

/// Manages detection recovery — orchestrates retry, backoff, fallback,
/// re-synchronization, and state reset when detection fails.
pub struct DetectionRecovery {
    /// Recovery configuration.
    pub config: DetectionRecoveryConfig,
    /// Current recovery state.
    pub state: RecoveryState,
    /// Recovery statistics.
    pub stats: RecoveryStatistics,
    /// Current backoff delay (milliseconds).
    pub current_backoff_ms: u64,
    /// Number of consecutive failures.
    pub consecutive_failures: u32,
    /// Number of successful recoveries since last failure.
    pub consecutive_successes: u32,
    /// Start time of the current recovery attempt.
    pub current_attempt_start: Option<Instant>,
    /// History of recovery outcomes.
    history: Vec<RecoveryRecord>,
    /// Maximum history length.
    max_history: usize,
    /// Whether the system is currently in degraded mode.
    pub in_degraded_mode: bool,
    /// Time when degraded mode started.
    pub degraded_start: Option<Instant>,
    /// Fallback quality (current quality in fallback mode).
    pub fallback_quality: f64,
}

/// A single recovery record.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryRecord {
    /// Timestamp of the recovery attempt.
    pub timestamp: Instant,
    /// Strategy used.
    pub strategy: RecoveryStrategy,
    /// Outcome of the attempt.
    pub outcome: RecoveryOutcome,
    /// Duration of the attempt (milliseconds).
    pub duration_ms: f64,
    /// Error that triggered recovery (if any).
    pub trigger_error: Option<String>,
    /// Whether fallback mode was entered.
    pub entered_fallback: bool,
}

/// Current state of the recovery engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryState {
    /// Normal operation — no recovery needed.
    Normal,
    /// Detection failure detected, recovery in progress.
    Recovering,
    /// System is in degraded mode (fallback active).
    Degraded,
    /// Recovery exhausted — detection unavailable.
    Exhausted,
    /// System is re-synchronizing.
    ReSynchronizing,
}

impl Default for RecoveryState {
    fn default() -> Self {
        RecoveryState::Normal
    }
}

impl DetectionRecovery {
    /// Create a new detection recovery engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: DetectionRecoveryConfig::default(),
            state: RecoveryState::Normal,
            stats: RecoveryStatistics::default(),
            current_backoff_ms: 0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            current_attempt_start: None,
            history: Vec::with_capacity(1000),
            max_history: 1000,
            in_degraded_mode: false,
            degraded_start: None,
            fallback_quality: 0.0,
        }
    }

    /// Create a new detection recovery engine with custom configuration.
    pub fn with_config(config: DetectionRecoveryConfig) -> Self {
        Self {
            config,
            state: RecoveryState::Normal,
            stats: RecoveryStatistics::default(),
            current_backoff_ms: 0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            current_attempt_start: None,
            history: Vec::with_capacity(1000),
            max_history: 1000,
            in_degraded_mode: false,
            degraded_start: None,
            fallback_quality: 0.0,
        }
    }

    /// Record a detection failure and initiate recovery.
    ///
    /// Returns the recovery strategy to use and whether recovery should proceed.
    pub fn on_detection_failure(
        &mut self,
        error: Option<String>,
    ) -> (RecoveryStrategy, bool) {
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        self.state = RecoveryState::Recovering;

        let strategy = if self.consecutive_failures >= self.config.max_recovery_attempts {
            if self.config.enable_fallback && !self.in_degraded_mode {
                RecoveryStrategy::FallbackMode
            } else {
                RecoveryStrategy::Escalate
            }
        } else {
            self.config.default_strategy
        };

        let should_proceed = self.config.auto_recover;

        // Record the failure in history.
        let record = RecoveryRecord {
            timestamp: Instant::now(),
            strategy,
            outcome: RecoveryOutcome::InProgress,
            duration_ms: 0.0,
            trigger_error: error.clone(),
            entered_fallback: matches!(strategy, RecoveryStrategy::FallbackMode),
        };
        self.add_history_record(record);

        // Update strategy counts.
        *self
            .stats
            .strategy_counts
            .entry(strategy)
            .or_insert(0) += 1;

        self.stats.total_attempts += 1;

        // Set backoff.
        match strategy {
            RecoveryStrategy::RetryWithBackoff => {
                self.current_backoff_ms = (self.config.base_backoff_ms as f64
                    * self.config.backoff_multiplier.powi(
                        (self.consecutive_failures - 1) as i32,
                    ))
                .clamp(
                    self.config.base_backoff_ms as f64,
                    self.config.max_backoff_ms as f64,
                ) as u64;
            }
            _ => {
                self.current_backoff_ms = strategy.default_backoff_ms();
            }
        }

        (strategy, should_proceed)
    }

    /// Record a successful recovery.
    pub fn on_recovery_success(&mut self, outcome: RecoveryOutcome) {
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
        self.current_backoff_ms = 0;

        match outcome {
            RecoveryOutcome::Success => {
                self.state = RecoveryState::Normal;
                self.stats.successful_recoveries += 1;
            }
            RecoveryOutcome::Partial => {
                self.state = RecoveryState::Degraded;
                self.in_degraded_mode = true;
                self.degraded_start = Some(Instant::now());
                self.stats.partial_recoveries += 1;
            }
            _ => {}
        }

        *self.stats.outcome_counts.entry(outcome).or_insert(0) += 1;

        // Update fallback quality if in fallback mode.
        if matches!(outcome, RecoveryOutcome::Partial) {
            self.fallback_quality = 0.5; // Default partial quality.
        }

        if let Some(start) = self.current_attempt_start {
            let elapsed = start.elapsed().as_millis() as f64;
            self.stats.total_recovery_time_ms += elapsed;
            self.stats.avg_recovery_time_ms =
                self.stats.total_recovery_time_ms
                    / (self.stats.successful_recoveries
                        + self.stats.partial_recoveries
                        + self.stats.failed_recoveries) as f64;
            self.stats.max_recovery_time_ms =
                self.stats.max_recovery_time_ms.max(elapsed);
            if self.stats.min_recovery_time_ms == 0.0 || elapsed < self.stats.min_recovery_time_ms
            {
                self.stats.min_recovery_time_ms = elapsed;
            }
        }

        self.current_attempt_start = None;
    }

    /// Record a failed recovery attempt.
    pub fn on_recovery_failure(&mut self) {
        self.state = RecoveryState::Exhausted;
        self.stats.failed_recoveries += 1;
        *self
            .stats
            .outcome_counts
            .entry(RecoveryOutcome::Failed)
            .or_insert(0) += 1;

        if let Some(start) = self.current_attempt_start {
            let elapsed = start.elapsed().as_millis() as f64;
            self.stats.total_recovery_time_ms += elapsed;
        }

        self.current_attempt_start = None;

        // Enter degraded mode if fallback is enabled.
        if self.config.enable_fallback {
            self.state = RecoveryState::Degraded;
            self.in_degraded_mode = true;
            self.degraded_start = Some(Instant::now());
            self.stats.fallback_activations += 1;
        }
    }

    /// Start a recovery attempt timer.
    pub fn start_recovery_attempt(&mut self) {
        self.current_attempt_start = Some(Instant::now());
    }

    /// Enter fallback mode with a given quality level.
    pub fn enter_fallback_mode(&mut self, quality: f64) {
        self.state = RecoveryState::Degraded;
        self.in_degraded_mode = true;
        self.degraded_start = Some(Instant::now());
        self.fallback_quality = quality.clamp(0.0, 1.0);
        self.stats.fallback_activations += 1;

        let record = RecoveryRecord {
            timestamp: Instant::now(),
            strategy: RecoveryStrategy::FallbackMode,
            outcome: RecoveryOutcome::Partial,
            duration_ms: 0.0,
            trigger_error: None,
            entered_fallback: true,
        };
        self.add_history_record(record);
    }

    /// Perform a state reset.
    pub fn reset_state(&mut self) {
        self.state = RecoveryState::Normal;
        self.in_degraded_mode = false;
        self.degraded_start = None;
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.current_backoff_ms = 0;
        self.fallback_quality = 0.0;
        self.stats.reset_count += 1;
    }

    /// Perform re-synchronization.
    pub fn resynchronize(&mut self) -> RecoveryOutcome {
        self.state = RecoveryState::ReSynchronizing;
        self.stats.resync_count += 1;

        // Simulate re-sync: clear backoff and failure count.
        self.current_backoff_ms = 0;
        self.consecutive_failures = 0;

        let outcome = RecoveryOutcome::Success;
        self.state = RecoveryState::Normal;
        self.stats.successful_recoveries += 1;
        *self
            .stats
            .outcome_counts
            .entry(outcome)
            .or_insert(0) += 1;

        outcome
    }

    /// Check if degraded mode has exceeded the maximum time.
    pub fn degraded_mode_expired(&self) -> bool {
        if !self.in_degraded_mode {
            return false;
        }
        if let Some(start) = self.degraded_start {
            let elapsed = start.elapsed().as_millis() as u64;
            elapsed > self.config.max_degraded_ms
        } else {
            false
        }
    }

    /// Exit degraded mode.
    pub fn exit_degraded_mode(&mut self) {
        self.in_degraded_mode = false;
        self.degraded_start = None;
        self.state = RecoveryState::Normal;
        self.fallback_quality = 0.0;
    }

    /// Add a record to the history.
    fn add_history_record(&mut self, record: RecoveryRecord) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(record);
    }

    /// Returns the recovery history.
    pub fn history(&self) -> &[RecoveryRecord] {
        &self.history
    }

    /// Returns the current backoff delay in milliseconds.
    pub fn backoff_ms(&self) -> u64 {
        self.current_backoff_ms
    }

    /// Returns true if the engine is in normal state.
    pub fn is_normal(&self) -> bool {
        self.state == RecoveryState::Normal
    }

    /// Returns true if the engine is recovering.
    pub fn is_recovering(&self) -> bool {
        self.state == RecoveryState::Recovering
    }

    /// Returns true if the engine is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.state == RecoveryState::Exhausted
    }

    /// Returns true if the engine is re-synchronizing.
    pub fn is_resyncing(&self) -> bool {
        self.state == RecoveryState::ReSynchronizing
    }

    /// Returns the recovery rate (successful / total attempts).
    pub fn recovery_rate(&self) -> f64 {
        if self.stats.total_attempts == 0 {
            return 0.0;
        }
        (self.stats.successful_recoveries + self.stats.partial_recoveries) as f64
            / self.stats.total_attempts as f64
    }

    /// Returns the failure rate.
    pub fn failure_rate(&self) -> f64 {
        if self.stats.total_attempts == 0 {
            return 0.0;
        }
        self.stats.failed_recoveries as f64 / self.stats.total_attempts as f64
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = RecoveryStatistics::default();
        self.history.clear();
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: DetectionRecoveryConfig) {
        self.config = config;
    }

    /// Returns the number of consecutive failures.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Returns the number of consecutive successes.
    pub fn consecutive_successes(&self) -> u32 {
        self.consecutive_successes
    }
}

impl Default for DetectionRecovery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RecoveryContext — context for a recovery decision
// ============================================================================

/// Context information for making recovery decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryContext {
    /// The error that triggered the recovery (if any).
    pub error: Option<String>,
    /// The detection quality before failure (0.0 to 1.0).
    pub pre_failure_quality: f64,
    /// The time since the last successful detection (milliseconds).
    pub time_since_last_success_ms: u64,
    /// The number of consecutive detection failures.
    pub consecutive_failures: u32,
    /// The current signal strength estimate (0.0 to 1.0).
    pub signal_strength: f64,
    /// The current noise level estimate (0.0 to 1.0).
    pub noise_level: f64,
    /// Whether the channel is known to be unstable.
    pub channel_unstable: bool,
    /// Whether hardware is known to be healthy.
    pub hardware_healthy: bool,
}

/// Analyzes a recovery context and recommends a strategy.
pub struct RecoveryStrategyAnalyzer {
    /// Minimum quality for immediate retry.
    pub min_retry_quality: f64,
    /// Maximum failures before escalating.
    pub max_failures_before_escalate: u32,
    /// Signal strength threshold for re-sync.
    pub resync_signal_threshold: f64,
    /// Noise level threshold for fallback.
    pub fallback_noise_threshold: f64,
}

impl RecoveryStrategyAnalyzer {
    /// Create a new strategy analyzer.
    pub fn new() -> Self {
        Self {
            min_retry_quality: 0.3,
            max_failures_before_escalate: 5,
            resync_signal_threshold: 0.2,
            fallback_noise_threshold: 0.8,
        }
    }

    /// Analyze the context and recommend a recovery strategy.
    pub fn analyze(&self, context: &RecoveryContext) -> RecoveryStrategy {
        // If hardware is unhealthy, escalate.
        if !context.hardware_healthy {
            return RecoveryStrategy::Escalate;
        }

        // If too many consecutive failures, escalate.
        if context.consecutive_failures >= self.max_failures_before_escalate {
            return RecoveryStrategy::Escalate;
        }

        // If channel is very noisy, use fallback.
        if context.noise_level > self.fallback_noise_threshold {
            return RecoveryStrategy::FallbackMode;
        }

        // If signal is very weak, re-synchronize.
        if context.signal_strength < self.resync_signal_threshold {
            return RecoveryStrategy::ReSynchronize;
        }

        // If channel is unstable, use backoff.
        if context.channel_unstable {
            return RecoveryStrategy::RetryWithBackoff;
        }

        // If pre-failure quality was good, immediate retry.
        if context.pre_failure_quality >= self.min_retry_quality {
            return RecoveryStrategy::ImmediateRetry;
        }

        // Default: retry with backoff.
        RecoveryStrategy::RetryWithBackoff
    }

    /// Returns true if recovery is recommended (vs abort).
    pub fn should_recover(&self, context: &RecoveryContext) -> bool {
        !matches!(
            self.analyze(context),
            RecoveryStrategy::Escalate
                | RecoveryStrategy::Abort
        )
    }
}

impl Default for RecoveryStrategyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fallback detector — degraded detection mode
// ============================================================================

/// Result of fallback detection.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackDetectionResult {
    /// The detected tuple (low confidence).
    pub tuple: Option<SkyrmionTuple>,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Whether the result is reliable.
    pub reliable: bool,
    /// The fallback mode used.
    pub mode: FallbackMode,
}

/// Fallback detection modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackMode {
    /// Simple threshold-based detection.
    ThresholdBased,
    /// Reduced-precision skyrmion number computation.
    LowPrecision,
    /// Historical average-based detection.
    HistoricalAverage,
    /// No detection available.
    None,
}

impl Default for FallbackMode {
    fn default() -> Self {
        FallbackMode::None
    }
}

/// Performs fallback detection when the primary detection pipeline fails.
pub struct FallbackDetector {
    /// Current fallback mode.
    pub mode: FallbackMode,
    /// Minimum confidence for a reliable fallback detection.
    pub min_confidence: f64,
    /// Historical tuples for averaging.
    history: Vec<SkyrmionTuple>,
    /// Maximum history length.
    max_history: usize,
    /// Number of fallback detections performed.
    pub detection_count: u64,
    /// Number of reliable fallback detections.
    pub reliable_count: u64,
}

impl FallbackDetector {
    /// Create a new fallback detector.
    pub fn new() -> Self {
        Self {
            mode: FallbackMode::ThresholdBased,
            min_confidence: 0.4,
            history: Vec::with_capacity(100),
            max_history: 100,
            detection_count: 0,
            reliable_count: 0,
        }
    }

    /// Perform fallback detection on a noisy measurement.
    pub fn detect(
        &mut self,
        raw_measurement: f64,
    ) -> FallbackDetectionResult {
        self.detection_count += 1;

        let tuple = match self.mode {
            FallbackMode::ThresholdBased => {
                if raw_measurement.abs() > 0.5 {
                    Some(SkyrmionTuple::new(&[raw_measurement.round() as i64]).unwrap())
                } else {
                    Some(SkyrmionTuple::new(&[0]).unwrap())
                }
            }
            FallbackMode::LowPrecision => {
                let quantized = (raw_measurement * 2.0).round() / 2.0;
                Some(
                    SkyrmionTuple::new(&[quantized.round() as i64]).unwrap(),
                )
            }
            FallbackMode::HistoricalAverage => {
                if self.history.is_empty() {
                    Some(SkyrmionTuple::new(&[0]).unwrap())
                } else {
                    let avg: i64 = self
                        .history
                        .iter()
                        .map(|t| t.components[0])
                        .sum::<i64>()
                        / self.history.len() as i64;
                    Some(SkyrmionTuple::new(&[avg]).unwrap())
                }
            }
            FallbackMode::None => None,
        };

        let confidence = match &tuple {
            Some(t) => {
                let c = (1.0 - (raw_measurement - t.components[0] as f64).abs())
                    .clamp(0.0, 1.0);
                if c >= self.min_confidence {
                    self.reliable_count += 1;
                }
                c
            }
            None => 0.0,
        };

        let reliable = confidence >= self.min_confidence && tuple.is_some();

        let result = FallbackDetectionResult {
            tuple,
            confidence,
            reliable,
            mode: self.mode,
        };

        // Add to history if reliable.
        if reliable {
            if let Some(t) = &result.tuple {
                if self.history.len() >= self.max_history {
                    self.history.remove(0);
                }
                self.history.push(t.clone());
            }
        }

        result
    }

    /// Returns the fallback reliability rate.
    pub fn reliability_rate(&self) -> f64 {
        if self.detection_count == 0 {
            return 0.0;
        }
        self.reliable_count as f64 / self.detection_count as f64
    }

    /// Returns the fallback detection history.
    pub fn history(&self) -> &[SkyrmionTuple] {
        &self.history
    }

    /// Reset the fallback detector.
    pub fn reset(&mut self) {
        self.history.clear();
        self.detection_count = 0;
        self.reliable_count = 0;
    }

    /// Set the fallback mode.
    pub fn set_mode(&mut self, mode: FallbackMode) {
        self.mode = mode;
    }

    /// Set the minimum confidence threshold.
    pub fn set_min_confidence(&mut self, threshold: f64) {
        self.min_confidence = threshold.clamp(0.0, 1.0);
    }

    /// Returns the minimum confidence threshold.
    pub fn min_confidence(&self) -> f64 {
        self.min_confidence
    }
}

impl Default for FallbackDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Synchronization loss detector
// ============================================================================

/// Synchronization loss event.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncLossEvent {
    /// Timestamp of the loss.
    pub timestamp: Instant,
    /// Duration of the loss (milliseconds).
    pub duration_ms: f64,
    /// Severity (0.0 to 1.0).
    pub severity: f64,
    /// Whether recovery was automatic.
    pub auto_recovered: bool,
}

/// Detects synchronization loss in the detection pipeline.
pub struct SyncLossDetector {
    /// Maximum allowed time between successful detections (milliseconds).
    pub max_gap_ms: u64,
    /// Minimum severity to report.
    pub min_severity: f64,
    /// Detected sync loss events.
    events: Vec<SyncLossEvent>,
    /// Maximum history length.
    max_history: usize,
    /// Last successful detection time.
    last_success: Option<Instant>,
    /// Current loss start time (if in loss).
    pub loss_start: Option<Instant>,
    /// Total number of sync losses.
    pub total_losses: u64,
}

impl SyncLossDetector {
    /// Create a new sync loss detector.
    pub fn new() -> Self {
        Self {
            max_gap_ms: 500,
            min_severity: 0.3,
            events: Vec::with_capacity(100),
            max_history: 100,
            last_success: None,
            loss_start: None,
            total_losses: 0,
        }
    }

    /// Record a successful detection (clears any ongoing loss).
    pub fn record_success(&mut self) {
        if let Some(start) = self.loss_start.take() {
            let duration = start.elapsed().as_millis() as f64;
            let severity = (duration / self.max_gap_ms as f64).clamp(0.0, 1.0);
            if severity >= self.min_severity {
                let event = SyncLossEvent {
                    timestamp: start,
                    duration_ms: duration,
                    severity,
                    auto_recovered: true,
                };
                self.add_event(event);
                self.total_losses += 1;
            }
        }
        self.last_success = Some(Instant::now());
    }

    /// Record a detection gap (potential sync loss).
    pub fn record_gap(&mut self, gap_ms: u64) {
        if gap_ms > self.max_gap_ms {
            if self.loss_start.is_none() {
                self.loss_start = Some(Instant::now());
            }
        } else {
            // Gap is acceptable, not a loss.
            if let Some(start) = self.loss_start.take() {
                let duration = start.elapsed().as_millis() as f64;
                if duration > 0.0 {
                    let severity =
                        (duration / self.max_gap_ms as f64).clamp(0.0, 1.0);
                    if severity >= self.min_severity {
                        let event = SyncLossEvent {
                            timestamp: start,
                            duration_ms: duration,
                            severity,
                            auto_recovered: true,
                        };
                        self.add_event(event);
                        self.total_losses += 1;
                    }
                }
            }
        }
    }

    /// Force-end the current sync loss (manual recovery).
    pub fn end_loss(&mut self) {
        if let Some(start) = self.loss_start.take() {
            let duration = start.elapsed().as_millis() as f64;
            let severity = (duration / self.max_gap_ms as f64).clamp(0.0, 1.0);
            if severity >= self.min_severity {
                let event = SyncLossEvent {
                    timestamp: start,
                    duration_ms: duration,
                    severity,
                    auto_recovered: false,
                };
                self.add_event(event);
                self.total_losses += 1;
            }
        }
    }

    /// Returns true if currently in a sync loss.
    pub fn in_loss(&self) -> bool {
        self.loss_start.is_some()
    }

    /// Returns the current loss duration (milliseconds).
    pub fn current_loss_duration_ms(&self) -> f64 {
        self.loss_start
            .map(|start| start.elapsed().as_millis() as f64)
            .unwrap_or(0.0)
    }

    fn add_event(&mut self, event: SyncLossEvent) {
        if self.events.len() >= self.max_history {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Returns the sync loss events.
    pub fn events(&self) -> &[SyncLossEvent] {
        &self.events
    }

    /// Returns the average loss duration.
    pub fn avg_loss_duration_ms(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        self.events.iter().map(|e| e.duration_ms).sum::<f64>()
            / self.events.len() as f64
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.events.clear();
        self.last_success = None;
        self.loss_start = None;
        self.total_losses = 0;
    }

    /// Set the maximum allowed gap.
    pub fn set_max_gap_ms(&mut self, ms: u64) {
        self.max_gap_ms = ms;
    }

    /// Set the minimum severity.
    pub fn set_min_severity(&mut self, severity: f64) {
        self.min_severity = severity.clamp(0.0, 1.0);
    }
}

impl Default for SyncLossDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Recovery pipeline — combines recovery engine, fallback, and sync loss
// ============================================================================

/// Result of the full recovery pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryPipelineResult {
    /// The recovery outcome.
    pub outcome: RecoveryOutcome,
    /// The strategy used.
    pub strategy: RecoveryStrategy,
    /// Fallback detection result (if fallback was used).
    pub fallback_result: Option<FallbackDetectionResult>,
    /// Whether the system is in degraded mode.
    pub degraded: bool,
    /// Recovery time (milliseconds).
    pub recovery_time_ms: f64,
    /// Recommended next action.
    pub next_action: RecoveryAction,
}

/// Recommended next action after recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Continue normal detection.
    ContinueDetection,
    /// Continue in fallback/degraded mode.
    ContinueFallback,
    /// Retry detection.
    RetryDetection,
    /// Escalate to higher layer.
    Escalate,
    /// Abort detection pipeline.
    Abort,
    /// Wait for backoff period.
    WaitBackoff,
}

/// Runs the full recovery pipeline: detect failure → choose strategy → recover.
pub struct RecoveryPipeline {
    /// Recovery engine.
    pub engine: DetectionRecovery,
    /// Strategy analyzer.
    pub analyzer: RecoveryStrategyAnalyzer,
    /// Fallback detector.
    pub fallback: FallbackDetector,
    /// Sync loss detector.
    pub sync_loss: SyncLossDetector,
    /// Pipeline statistics.
    pub stats: PipelineRecoveryStats,
}

/// Statistics for the recovery pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineRecoveryStats {
    /// Total pipelines runs.
    pub total_runs: u64,
    /// Successful recoveries.
    pub successful: u64,
    /// Failed recoveries.
    pub failed: u64,
    /// Fallback activations.
    pub fallback_activations: u64,
    /// Average recovery time (milliseconds).
    pub avg_recovery_time_ms: f64,
}

impl RecoveryPipeline {
    /// Create a new recovery pipeline.
    pub fn new() -> Self {
        Self {
            engine: DetectionRecovery::new(),
            analyzer: RecoveryStrategyAnalyzer::new(),
            fallback: FallbackDetector::new(),
            sync_loss: SyncLossDetector::new(),
            stats: PipelineRecoveryStats::default(),
        }
    }

    /// Run the full recovery pipeline on a detection failure.
    pub fn recover(
        &mut self,
        context: &RecoveryContext,
        raw_measurement: Option<f64>,
    ) -> RecoveryPipelineResult {
        let start = Instant::now();
        self.stats.total_runs += 1;

        // Analyze context and choose strategy.
        let strategy = self.analyzer.analyze(context);
        let should_recover = self.analyzer.should_recover(context);

        // Record sync loss if applicable.
        if let Some(gap_ms) = context
            .time_since_last_success_ms
            .checked_add(0)
        {
            self.sync_loss.record_gap(gap_ms);
        }

        let mut fallback_result = None;
        let mut degraded = false;
        let mut next_action = RecoveryAction::ContinueDetection;

        let outcome = if !should_recover {
            // Escalate or abort.
            self.stats.failed += 1;
            next_action = RecoveryAction::Escalate;
            RecoveryOutcome::Failed
        } else {
            match strategy {
                RecoveryStrategy::ImmediateRetry => {
                    self.engine.on_recovery_success(RecoveryOutcome::Success);
                    self.stats.successful += 1;
                    next_action = RecoveryAction::RetryDetection;
                    RecoveryOutcome::Success
                }
                RecoveryStrategy::RetryWithBackoff => {
                    next_action = RecoveryAction::WaitBackoff;
                    self.engine.on_recovery_success(RecoveryOutcome::InProgress);
                    self.stats.successful += 1;
                    RecoveryOutcome::InProgress
                }
                RecoveryStrategy::FallbackMode => {
                    degraded = true;
                    self.engine.enter_fallback_mode(context.pre_failure_quality);
                    // Attempt fallback detection.
                    if let Some(measurement) = raw_measurement {
                        let result = self.fallback.detect(measurement);
                        fallback_result = Some(result.clone());
                        if result.reliable {
                            self.stats.successful += 1;
                            RecoveryOutcome::Partial
                        } else {
                            self.stats.failed += 1;
                            RecoveryOutcome::Failed
                        }
                    } else {
                        self.stats.fallback_activations += 1;
                        RecoveryOutcome::Partial
                    }
                    next_action = RecoveryAction::ContinueFallback;
                }
                RecoveryStrategy::ReSynchronize => {
                    let result = self.engine.resynchronize();
                    if result.is_success() {
                        self.stats.successful += 1;
                    } else {
                        self.stats.failed += 1;
                    }
                    next_action = RecoveryAction::RetryDetection;
                    result
                }
                RecoveryStrategy::ResetState => {
                    self.engine.reset_state();
                    self.stats.successful += 1;
                    next_action = RecoveryAction::ContinueDetection;
                    RecoveryOutcome::Success
                }
                RecoveryStrategy::GracefulDegradation => {
                    degraded = true;
                    self.stats.successful += 1;
                    next_action = RecoveryAction::ContinueFallback;
                    RecoveryOutcome::Partial
                }
                RecoveryStrategy::Abort => {
                    self.stats.failed += 1;
                    next_action = RecoveryAction::Abort;
                    RecoveryOutcome::Failed
                }
                RecoveryStrategy::Escalate => {
                    self.stats.failed += 1;
                    next_action = RecoveryAction::Escalate;
                    RecoveryOutcome::Failed
                }
            }
        };

        let elapsed = start.elapsed().as_millis() as f64;
        self.stats.avg_recovery_time_ms = (self.stats.avg_recovery_time_ms
            * (self.stats.total_runs - 1) as f64
            + elapsed)
            / self.stats.total_runs as f64;

        // Record sync loss end if recovered.
        if outcome.is_success() {
            self.sync_loss.record_success();
        }

        RecoveryPipelineResult {
            outcome,
            strategy,
            fallback_result,
            degraded,
            recovery_time_ms: elapsed,
            next_action,
        }
    }

    /// Record a successful detection (clears recovery state).
    pub fn on_success(&mut self) {
        self.engine.reset_state();
        self.fallback.reset();
        self.sync_loss.record_success();
    }

    /// Returns true if the pipeline is in degraded mode.
    pub fn is_degraded(&self) -> bool {
        self.engine.in_degraded_mode
    }

    /// Returns the pipeline statistics.
    pub fn stats(&self) -> &PipelineRecoveryStats {
        &self.stats
    }

    /// Reset the pipeline.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.fallback.reset();
        self.sync_loss.reset();
        self.stats = PipelineRecoveryStats::default();
    }

    /// Returns the recovery engine.
    pub fn engine(&self) -> &DetectionRecovery {
        &self.engine
    }

    /// Returns the fallback detector.
    pub fn fallback(&self) -> &FallbackDetector {
        &self.fallback
    }

    /// Returns the sync loss detector.
    pub fn sync_loss_detector(&self) -> &SyncLossDetector {
        &self.sync_loss
    }
}

impl Default for RecoveryPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RecoveryEscalationManager — escalates to higher layers when recovery fails
// ============================================================================

/// Escalation level for recovery failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationLevel {
    /// No escalation needed.
    None,
    /// Level 1: Notify detection subsystem.
    Level1Notify,
    /// Level 2: Notify communication layer.
    Level2NotifyCommLayer,
    /// Level 3: Notify network layer.
    Level3NotifyNetworkLayer,
    /// Level 4: Full system escalation.
    Level4SystemEscalation,
    /// Level 5: Manual intervention required.
    Level5ManualIntervention,
}

impl EscalationLevel {
    /// Returns the numeric severity (0 to 5).
    pub fn severity(self) -> u32 {
        match self {
            EscalationLevel::None => 0,
            EscalationLevel::Level1Notify => 1,
            EscalationLevel::Level2NotifyCommLayer => 2,
            EscalationLevel::Level3NotifyNetworkLayer => 3,
            EscalationLevel::Level4SystemEscalation => 4,
            EscalationLevel::Level5ManualIntervention => 5,
        }
    }

    /// Returns true if this level requires manual intervention.
    pub fn requires_manual(self) -> bool {
        matches!(self, EscalationLevel::Level5ManualIntervention)
    }
}

impl Default for EscalationLevel {
    fn default() -> Self {
        EscalationLevel::None
    }
}

/// Escalation policy configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct EscalationPolicy {
    /// Failures before escalating to level 1.
    pub failures_for_level1: u32,
    /// Failures before escalating to level 2.
    pub failures_for_level2: u32,
    /// Failures before escalating to level 3.
    pub failures_for_level3: u32,
    /// Failures before escalating to level 4.
    pub failures_for_level4: u32,
    /// Failures before escalating to level 5 (manual).
    pub failures_for_level5: u32,
    /// Time window for counting failures (milliseconds).
    pub failure_window_ms: u64,
    /// Auto-de-escalate after successful detections.
    pub auto_deescalate: bool,
    /// Successful detections before de-escalating one level.
    pub successes_for_deescalate: u32,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            failures_for_level1: 3,
            failures_for_level2: 5,
            failures_for_level3: 10,
            failures_for_level4: 15,
            failures_for_level5: 20,
            failure_window_ms: 60_000,
            auto_deescalate: true,
            successes_for_deescalate: 10,
        }
    }
}

/// Escalation event record.
#[derive(Debug, Clone, PartialEq)]
pub struct EscalationEvent {
    /// Timestamp of escalation.
    pub timestamp: Instant,
    /// Level escalated to.
    pub level: EscalationLevel,
    /// Reason for escalation.
    pub reason: String,
    /// Number of failures at time of escalation.
    pub failure_count: u32,
}

/// Manages escalation of detection recovery failures to higher system layers.
pub struct RecoveryEscalationManager {
    /// Escalation policy.
    pub policy: EscalationPolicy,
    /// Current escalation level.
    pub current_level: EscalationLevel,
    /// Failure count in current window.
    pub failure_count: u32,
    /// Success count since last escalation/de-escalation.
    pub success_count: u32,
    /// Start of current failure window.
    pub window_start: Instant,
    /// Escalation history.
    history: Vec<EscalationEvent>,
    /// Maximum history length.
    max_history: usize,
    /// Total escalations performed.
    pub total_escalations: u64,
    /// Total de-escalations performed.
    pub total_deescalations: u64,
}

impl RecoveryEscalationManager {
    /// Create a new escalation manager.
    pub fn new() -> Self {
        Self {
            policy: EscalationPolicy::default(),
            current_level: EscalationLevel::None,
            failure_count: 0,
            success_count: 0,
            window_start: Instant::now(),
            history: Vec::with_capacity(1000),
            max_history: 1000,
            total_escalations: 0,
            total_deescalations: 0,
        }
    }

    /// Create with custom policy.
    pub fn with_policy(policy: EscalationPolicy) -> Self {
        Self {
            policy,
            current_level: EscalationLevel::None,
            failure_count: 0,
            success_count: 0,
            window_start: Instant::now(),
            history: Vec::with_capacity(1000),
            max_history: 1000,
            total_escalations: 0,
            total_deescalations: 0,
        }
    }

    /// Record a failure and potentially escalate.
    pub fn record_failure(&mut self, reason: String) -> EscalationLevel {
        // Check if window has expired.
        if self.window_start.elapsed().as_millis() as u64 > self.policy.failure_window_ms {
            self.failure_count = 0;
            self.window_start = Instant::now();
        }

        self.failure_count += 1;

        let new_level = self.determine_level(self.failure_count);

        if new_level.severity() > self.current_level.severity() {
            self.escalate(new_level, reason.clone());
        }

        self.current_level = new_level;
        new_level
    }

    /// Record a successful detection (for de-escalation).
    pub fn record_success(&mut self) {
        if !self.policy.auto_deescalate {
            return;
        }

        self.success_count += 1;

        if self.success_count >= self.policy.successes_for_deescalate
            && self.current_level.severity() > 0
        {
            self.deescalate();
            self.success_count = 0;
        }
    }

    /// Determine the appropriate escalation level for a failure count.
    fn determine_level(&self, failures: u32) -> EscalationLevel {
        if failures >= self.policy.failures_for_level5 {
            EscalationLevel::Level5ManualIntervention
        } else if failures >= self.policy.failures_for_level4 {
            EscalationLevel::Level4SystemEscalation
        } else if failures >= self.policy.failures_for_level3 {
            EscalationLevel::Level3NotifyNetworkLayer
        } else if failures >= self.policy.failures_for_level2 {
            EscalationLevel::Level2NotifyCommLayer
        } else if failures >= self.policy.failures_for_level1 {
            EscalationLevel::Level1Notify
        } else {
            EscalationLevel::None
        }
    }

    /// Escalate to a new level.
    fn escalate(&mut self, level: EscalationLevel, reason: String) {
        let event = EscalationEvent {
            timestamp: Instant::now(),
            level,
            reason: reason.clone(),
            failure_count: self.failure_count,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(event);
        self.total_escalations += 1;
    }

    /// De-escalate one level.
    fn deescalate(&mut self) {
        self.current_level = match self.current_level {
            EscalationLevel::Level1Notify => EscalationLevel::None,
            EscalationLevel::Level2NotifyCommLayer => EscalationLevel::Level1Notify,
            EscalationLevel::Level3NotifyNetworkLayer => {
                EscalationLevel::Level2NotifyCommLayer
            }
            EscalationLevel::Level4SystemEscalation => {
                EscalationLevel::Level3NotifyNetworkLayer
            }
            EscalationLevel::Level5ManualIntervention => {
                EscalationLevel::Level4SystemEscalation
            }
            EscalationLevel::None => EscalationLevel::None,
        };
        self.total_deescalations += 1;
    }

    /// Returns the escalation history.
    pub fn history(&self) -> &[EscalationEvent] {
        &self.history
    }

    /// Returns true if manual intervention is required.
    pub fn requires_manual_intervention(&self) -> bool {
        self.current_level.requires_manual()
    }

    /// Reset the escalation manager.
    pub fn reset(&mut self) {
        self.current_level = EscalationLevel::None;
        self.failure_count = 0;
        self.success_count = 0;
        self.window_start = Instant::now();
        self.history.clear();
    }

    /// Set the escalation policy.
    pub fn set_policy(&mut self, policy: EscalationPolicy) {
        self.policy = policy;
    }

    /// Returns the number of escalations.
    pub fn total_escalations(&self) -> u64 {
        self.total_escalations
    }

    /// Returns the number of de-escalations.
    pub fn total_deescalations(&self) -> u64 {
        self.total_deescalations
    }
}

impl Default for RecoveryEscalationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RecoveryBackoffScheduler — manages exponential backoff schedules
// ============================================================================

/// Backoff schedule type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffSchedule {
    /// Fixed delay between retries.
    Fixed,
    /// Linear backoff (delay increases linearly).
    Linear,
    /// Exponential backoff (delay doubles each retry).
    Exponential,
    /// Fibonacci backoff (delay follows Fibonacci sequence).
    Fibonacci,
}

impl BackoffSchedule {
    /// Returns the number of retry attempts supported before cap.
    pub fn max_attempts(self) -> u32 {
        match self {
            BackoffSchedule::Fixed => 10,
            BackoffSchedule::Linear => 10,
            BackoffSchedule::Exponential => 8,
            BackoffSchedule::Fibonacci => 10,
        }
    }
}

/// Backoff parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BackoffParameters {
    /// Initial delay in milliseconds.
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds.
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff.
    pub multiplier: f64,
    /// Jitter factor (0.0 to 1.0, adds randomness).
    pub jitter_factor: f64,
}

impl Default for BackoffParameters {
    fn default() -> Self {
        Self {
            initial_delay_ms: 100,
            max_delay_ms: 30_000,
            multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

/// Computes backoff delays for retry scheduling.
pub struct RecoveryBackoffScheduler {
    /// Schedule type.
    pub schedule: BackoffSchedule,
    /// Parameters.
    pub params: BackoffParameters,
    /// Current retry attempt number.
    pub attempt: u32,
    /// History of computed delays.
    history: Vec<u64>,
    /// Maximum history length.
    max_history: usize,
}

impl RecoveryBackoffScheduler {
    /// Create a new backoff scheduler.
    pub fn new() -> Self {
        Self {
            schedule: BackoffSchedule::Exponential,
            params: BackoffParameters::default(),
            attempt: 0,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Compute the delay for the next retry attempt.
    pub fn next_delay(&mut self) -> u64 {
        self.attempt += 1;
        let delay = self.compute_delay(self.attempt);

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(delay);

        delay
    }

    /// Compute delay for a specific attempt number.
    fn compute_delay(&self, attempt: u32) -> u64 {
        let base = match self.schedule {
            BackoffSchedule::Fixed => self.params.initial_delay_ms,
            BackoffSchedule::Linear => {
                self.params.initial_delay_ms * attempt as u64
            }
            BackoffSchedule::Exponential => {
                (self.params.initial_delay_ms as f64
                    * self.params.multiplier.powi((attempt - 1) as i32)) as u64
            }
            BackoffSchedule::Fibonacci => {
                (self.params.initial_delay_ms as f64
                    * fibonacci(attempt) as f64) as u64
            }
        };

        let capped = base.min(self.params.max_delay_ms);

        // Apply jitter.
        let jitter = (capped as f64 * self.params.jitter_factor) as u64;
        let jittered = capped.saturating_add(jitter);

        jittered.min(self.params.max_delay_ms)
    }

    /// Reset the scheduler (attempt counter).
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.history.clear();
    }

    /// Returns the current attempt number.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Returns the delay history.
    pub fn history(&self) -> &[u64] {
        &self.history
    }

    /// Returns the last computed delay.
    pub fn last_delay(&self) -> Option<u64> {
        self.history.last().copied()
    }

    /// Set the schedule type.
    pub fn set_schedule(&mut self, schedule: BackoffSchedule) {
        self.schedule = schedule;
    }

    /// Set backoff parameters.
    pub fn set_params(&mut self, params: BackoffParameters) {
        self.params = params;
    }
}

impl Default for RecoveryBackoffScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the nth Fibonacci number.
fn fibonacci(n: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 2..=n {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

// ============================================================================
// DetectionQualityMonitor — monitors detection quality over time
// ============================================================================

/// Quality snapshot for detection monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionQualitySnapshot {
    /// Timestamp.
    pub timestamp: Instant,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Signal-to-noise ratio estimate.
    pub snr: f64,
    /// Number of detections in window.
    pub num_detections: usize,
    /// Number of successful detections.
    pub num_successes: usize,
    /// Average detection latency (microseconds).
    pub avg_latency_us: f64,
}

/// Computes the quality score from a snapshot.
impl DetectionQualitySnapshot {
    /// Returns the success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f64 {
        if self.num_detections == 0 {
            return 0.0;
        }
        self.num_successes as f64 / self.num_detections as f64
    }

    /// Returns the overall quality score (0.0 to 1.0).
    pub fn quality_score(&self) -> f64 {
        let success = self.success_rate();
        let conf = self.confidence.clamp(0.0, 1.0);
        let snr_factor = (1.0 - (-self.snr / 10.0).exp()).clamp(0.0, 1.0);
        (success * 0.4 + conf * 0.4 + snr_factor * 0.2).clamp(0.0, 1.0)
    }
}

/// Monitors detection quality and detects degradation.
pub struct DetectionQualityMonitor {
    /// Quality snapshots history.
    history: Vec<DetectionQualitySnapshot>,
    /// Maximum history length.
    max_history: usize,
    /// Current window accumulators.
    window_confidence_sum: f64,
    window_snr_sum: f64,
    window_detections: usize,
    window_successes: usize,
    window_latency_sum_us: f64,
    /// Window start time.
    window_start: Instant,
    /// Window duration (milliseconds).
    pub window_duration_ms: u64,
    /// Quality degradation threshold.
    pub degradation_threshold: f64,
    /// Total snapshots taken.
    pub total_snapshots: u64,
}

impl DetectionQualityMonitor {
    /// Create a new quality monitor.
    pub fn new(window_duration_ms: u64) -> Self {
        Self {
            history: Vec::with_capacity(1000),
            max_history: 1000,
            window_confidence_sum: 0.0,
            window_snr_sum: 0.0,
            window_detections: 0,
            window_successes: 0,
            window_latency_sum_us: 0.0,
            window_start: Instant::now(),
            window_duration_ms,
            degradation_threshold: 0.5,
            total_snapshots: 0,
        }
    }

    /// Record a detection result.
    pub fn record_detection(
        &mut self,
        confidence: f64,
        snr: f64,
        success: bool,
        latency_us: f64,
    ) {
        // Check if window has expired.
        if self.window_start.elapsed().as_millis() as u64 >= self.window_duration_ms
            && self.window_detections > 0
        {
            self.flush_window();
        }

        self.window_confidence_sum += confidence.clamp(0.0, 1.0);
        self.window_snr_sum += snr.clamp(0.0, 100.0);
        self.window_detections += 1;
        if success {
            self.window_successes += 1;
        }
        self.window_latency_sum_us += latency_us;
    }

    /// Flush the current window into a snapshot.
    fn flush_window(&mut self) {
        if self.window_detections == 0 {
            return;
        }

        let snapshot = DetectionQualitySnapshot {
            timestamp: self.window_start,
            confidence: self.window_confidence_sum / self.window_detections as f64,
            snr: self.window_snr_sum / self.window_detections as f64,
            num_detections: self.window_detections,
            num_successes: self.window_successes,
            avg_latency_us: self.window_latency_sum_us / self.window_detections as f64,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(snapshot);
        self.total_snapshots += 1;

        // Reset window.
        self.window_confidence_sum = 0.0;
        self.window_snr_sum = 0.0;
        self.window_detections = 0;
        self.window_successes = 0;
        self.window_latency_sum_us = 0.0;
        self.window_start = Instant::now();
    }

    /// Returns true if quality is degrading.
    pub fn is_degrading(&self) -> bool {
        self.history
            .last()
            .map_or(false, |s| s.quality_score() < self.degradation_threshold)
    }

    /// Returns the average quality score across history.
    pub fn avg_quality_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|s| s.quality_score())
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the average success rate.
    pub fn avg_success_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|s| s.success_rate())
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the quality history.
    pub fn history(&self) -> &[DetectionQualitySnapshot] {
        &self.history
    }

    /// Reset the monitor.
    pub fn reset(&mut self) {
        self.history.clear();
        self.window_confidence_sum = 0.0;
        self.window_snr_sum = 0.0;
        self.window_detections = 0;
        self.window_successes = 0;
        self.window_latency_sum_us = 0.0;
        self.window_start = Instant::now();
        self.total_snapshots = 0;
    }

    /// Set the degradation threshold.
    pub fn set_degradation_threshold(&mut self, threshold: f64) {
        self.degradation_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Returns the number of snapshots.
    pub fn num_snapshots(&self) -> usize {
        self.history.len()
    }

    /// Returns the current window detection count.
    pub fn window_detection_count(&self) -> usize {
        self.window_detections
    }
}

impl Default for DetectionQualityMonitor {
    fn default() -> Self {
        Self::new(1_000) // 1 second windows.
    }
}

// ============================================================================
// RecoveryCircuitBreaker — circuit breaker for detection pipeline
// ============================================================================

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed — requests pass through normally.
    Closed,
    /// Circuit is open — requests are rejected immediately.
    Open,
    /// Circuit is half-open — testing if the system has recovered.
    HalfOpen,
}

/// Circuit breaker configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit.
    pub failure_threshold: u32,
    /// Number of successes in half-open before closing.
    pub success_threshold: u32,
    /// Timeout before attempting half-open (milliseconds).
    pub timeout_ms: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 10_000,
        }
    }
}

/// Circuit breaker for the detection pipeline — prevents cascading failures
/// by opening the circuit when detection repeatedly fails.
pub struct RecoveryCircuitBreaker {
    /// Configuration.
    pub config: CircuitBreakerConfig,
    /// Current state.
    pub state: CircuitBreakerState,
    /// Failure count (resets on success or state change).
    pub failure_count: u32,
    /// Success count in half-open state.
    pub success_count: u32,
    /// Time when the circuit was opened.
    pub opened_at: Option<Instant>,
    /// Total times the circuit has opened.
    pub open_count: u64,
    /// Total times the circuit has closed.
    pub close_count: u64,
    /// Total requests rejected by open circuit.
    pub rejected_count: u64,
    /// Total requests allowed through.
    pub allowed_count: u64,
}

impl RecoveryCircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new() -> Self {
        Self {
            config: CircuitBreakerConfig::default(),
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            opened_at: None,
            open_count: 0,
            close_count: 0,
            rejected_count: 0,
            allowed_count: 0,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            opened_at: None,
            open_count: 0,
            close_count: 0,
            rejected_count: 0,
            allowed_count: 0,
        }
    }

    /// Attempt to execute a request through the circuit breaker.
    /// Returns true if the request is allowed, false if rejected.
    pub fn attempt_request(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => {
                self.allowed_count += 1;
                true
            }
            CircuitBreakerState::Open => {
                // Check if timeout has elapsed.
                if let Some(opened) = self.opened_at {
                    if opened.elapsed().as_millis() as u64 >= self.config.timeout_ms {
                        self.transition_to_half_open();
                        self.allowed_count += 1;
                        true
                    } else {
                        self.rejected_count += 1;
                        false
                    }
                } else {
                    self.rejected_count += 1;
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.allowed_count += 1;
                true
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.transition_to_closed();
                }
            }
            CircuitBreakerState::Open => {
                // Shouldn't happen (requests are rejected in open state).
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.config.failure_threshold {
                    self.transition_to_open();
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Any failure in half-open re-opens the circuit.
                self.transition_to_open();
            }
            CircuitBreakerState::Open => {}
        }
    }

    /// Transition to open state.
    fn transition_to_open(&mut self) {
        self.state = CircuitBreakerState::Open;
        self.opened_at = Some(Instant::now());
        self.failure_count = 0;
        self.success_count = 0;
        self.open_count += 1;
    }

    /// Transition to half-open state.
    fn transition_to_half_open(&mut self) {
        self.state = CircuitBreakerState::HalfOpen;
        self.success_count = 0;
        self.opened_at = None;
    }

    /// Transition to closed state.
    fn transition_to_closed(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.opened_at = None;
        self.close_count += 1;
    }

    /// Returns true if the circuit is open (requests rejected).
    pub fn is_open(&self) -> bool {
        self.state == CircuitBreakerState::Open
    }

    /// Returns true if the circuit is closed (requests allowed).
    pub fn is_closed(&self) -> bool {
        self.state == CircuitBreakerState::Closed
    }

    /// Returns true if the circuit is half-open.
    pub fn is_half_open(&self) -> bool {
        self.state == CircuitBreakerState::HalfOpen
    }

    /// Reset the circuit breaker.
    pub fn reset(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.opened_at = None;
    }

    /// Set the circuit breaker configuration.
    pub fn set_config(&mut self, config: CircuitBreakerConfig) {
        self.config = config;
    }

    /// Returns the rejection rate.
    pub fn rejection_rate(&self) -> f64 {
        let total = self.allowed_count + self.rejected_count;
        if total == 0 {
            return 0.0;
        }
        self.rejected_count as f64 / total as f64
    }

    /// Returns the allowance rate.
    pub fn allowance_rate(&self) -> f64 {
        let total = self.allowed_count + self.rejected_count;
        if total == 0 {
            return 0.0;
        }
        self.allowed_count as f64 / total as f64
    }
}

impl Default for RecoveryCircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ComprehensiveRecoveryManager — combines all recovery subsystems
// ============================================================================

/// Result of comprehensive recovery management.
#[derive(Debug, Clone, PartialEq)]
pub struct ComprehensiveRecoveryResult {
    /// Overall recovery outcome.
    pub outcome: RecoveryOutcome,
    /// Recovery strategy used.
    pub strategy: RecoveryStrategy,
    /// Escalation level (if any).
    pub escalation_level: EscalationLevel,
    /// Whether circuit breaker is open.
    pub circuit_open: bool,
    /// Whether in degraded mode.
    pub degraded: bool,
    /// Recovery time (milliseconds).
    pub recovery_time_ms: f64,
    /// Recommended action.
    pub action: RecoveryAction,
}

/// Manages all recovery subsystems together: engine, escalation, backoff,
/// quality monitoring, and circuit breaker.
pub struct ComprehensiveRecoveryManager {
    /// Recovery engine.
    pub engine: DetectionRecovery,
    /// Escalation manager.
    pub escalation: RecoveryEscalationManager,
    /// Backoff scheduler.
    pub backoff: RecoveryBackoffScheduler,
    /// Quality monitor.
    pub quality: DetectionQualityMonitor,
    /// Circuit breaker.
    pub circuit_breaker: RecoveryCircuitBreaker,
    /// Strategy analyzer.
    pub analyzer: RecoveryStrategyAnalyzer,
    /// Fallback detector.
    pub fallback: FallbackDetector,
    /// Sync loss detector.
    pub sync_loss: SyncLossDetector,
    /// Manager statistics.
    pub stats: ComprehensiveRecoveryStats,
}

/// Statistics for comprehensive recovery management.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ComprehensiveRecoveryStats {
    /// Total recovery requests.
    pub total_requests: u64,
    /// Successful recoveries.
    pub successful: u64,
    /// Failed recoveries.
    pub failed: u64,
    /// Circuit breaker rejections.
    pub circuit_rejections: u64,
    /// Escalations triggered.
    pub escalations: u64,
    /// Fallback activations.
    pub fallback_activations: u64,
    /// Average recovery time (milliseconds).
    pub avg_recovery_time_ms: f64,
}

impl ComprehensiveRecoveryManager {
    /// Create a new comprehensive recovery manager.
    pub fn new() -> Self {
        Self {
            engine: DetectionRecovery::new(),
            escalation: RecoveryEscalationManager::new(),
            backoff: RecoveryBackoffScheduler::new(),
            quality: DetectionQualityMonitor::new(1_000),
            circuit_breaker: RecoveryCircuitBreaker::new(),
            analyzer: RecoveryStrategyAnalyzer::new(),
            fallback: FallbackDetector::new(),
            sync_loss: SyncLossDetector::new(),
            stats: ComprehensiveRecoveryStats::default(),
        }
    }

    /// Process a detection result through the comprehensive recovery system.
    pub fn process_detection_result(
        &mut self,
        success: bool,
        confidence: f64,
        snr: f64,
        latency_us: f64,
        error: Option<String>,
    ) -> ComprehensiveRecoveryResult {
        let start = Instant::now();
        self.stats.total_requests += 1;

        // Record quality.
        self.quality.record_detection(confidence, snr, success, latency_us);

        // Check circuit breaker.
        if !self.circuit_breaker.attempt_request() {
            self.stats.circuit_rejections += 1;
            return ComprehensiveRecoveryResult {
                outcome: RecoveryOutcome::Failed,
                strategy: RecoveryStrategy::Abort,
                escalation_level: self.escalation.current_level,
                circuit_open: true,
                degraded: self.engine.in_degraded_mode,
                recovery_time_ms: start.elapsed().as_millis() as f64,
                action: RecoveryAction::Abort,
            };
        }

        if success {
            // Record success across subsystems.
            self.engine.on_recovery_success(RecoveryOutcome::Success);
            self.escalation.record_success();
            self.circuit_breaker.record_success();
            self.backoff.reset();
            self.sync_loss.record_success();
            self.stats.successful += 1;

            ComprehensiveRecoveryResult {
                outcome: RecoveryOutcome::Success,
                strategy: RecoveryStrategy::ImmediateRetry,
                escalation_level: self.escalation.current_level,
                circuit_open: false,
                degraded: self.engine.in_degraded_mode,
                recovery_time_ms: start.elapsed().as_millis() as f64,
                action: RecoveryAction::ContinueDetection,
            }
        } else {
            // Record failure.
            self.circuit_breaker.record_failure();
            let context = RecoveryContext {
                error: error.clone(),
                pre_failure_quality: confidence,
                time_since_last_success_ms: 0,
                consecutive_failures: self.engine.consecutive_failures,
                signal_strength: confidence,
                noise_level: 1.0 - snr.clamp(0.0, 1.0),
                channel_unstable: self.quality.is_degrading(),
                hardware_healthy: true,
            };

            // Analyze and choose strategy.
            let strategy = self.analyzer.analyze(&context);

            // Record failure with escalation manager.
            let escalation_level = self.escalation.record_failure(
                error.clone().unwrap_or_default(),
            );

            // Check if we should escalate or recover.
            if escalation_level.requires_manual() {
                self.stats.failed += 1;
                self.circuit_breaker.record_failure();
                return ComprehensiveRecoveryResult {
                    outcome: RecoveryOutcome::Failed,
                    strategy: RecoveryStrategy::Escalate,
                    escalation_level,
                    circuit_open: self.circuit_breaker.is_open(),
                    degraded: true,
                    recovery_time_ms: start.elapsed().as_millis() as f64,
                    action: RecoveryAction::Escalate,
                };
            }

            // Attempt recovery.
            let (recovery_strategy, should_proceed) =
                self.engine.on_detection_failure(error.clone());

            if !should_proceed || !self.analyzer.should_recover(&context) {
                self.stats.failed += 1;
                return ComprehensiveRecoveryResult {
                    outcome: RecoveryOutcome::Failed,
                    strategy: recovery_strategy,
                    escalation_level,
                    circuit_open: self.circuit_breaker.is_open(),
                    degraded: true,
                    recovery_time_ms: start.elapsed().as_millis() as f64,
                    action: RecoveryAction::Escalate,
                };
            }

            // Try fallback if available.
            let mut degraded = false;
            let mut outcome = RecoveryOutcome::Success;

            if matches!(
                recovery_strategy,
                RecoveryStrategy::FallbackMode
                    | RecoveryStrategy::GracefulDegradation
            ) {
                degraded = true;
                // Use fallback detection with a synthetic measurement.
                let fallback_result = self.fallback.detect(confidence);
                if fallback_result.reliable {
                    self.stats.successful += 1;
                    outcome = RecoveryOutcome::Partial;
                } else {
                    self.stats.failed += 1;
                    outcome = RecoveryOutcome::Failed;
                }
            } else {
                self.stats.successful += 1;
            }

            // Update backoff.
            let _backoff_delay = self.backoff.next_delay();

            ComprehensiveRecoveryResult {
                outcome,
                strategy: recovery_strategy,
                escalation_level,
                circuit_open: self.circuit_breaker.is_open(),
                degraded,
                recovery_time_ms: start.elapsed().as_millis() as f64,
                action: if degraded {
                    RecoveryAction::ContinueFallback
                } else {
                    RecoveryAction::ContinueDetection
                },
            }
        }
    }

    /// Record a sync loss event.
    pub fn record_sync_loss(&mut self, gap_ms: u64) {
        self.sync_loss.record_gap(gap_ms);
    }

    /// Record sync loss end.
    pub fn end_sync_loss(&mut self) {
        self.sync_loss.end_loss();
    }

    /// Returns true if the system is in degraded mode.
    pub fn is_degraded(&self) -> bool {
        self.engine.in_degraded_mode || self.quality.is_degrading()
    }

    /// Returns true if the circuit breaker is open.
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_breaker.is_open()
    }

    /// Returns true if manual intervention is required.
    pub fn requires_manual_intervention(&self) -> bool {
        self.escalation.requires_manual_intervention()
    }

    /// Reset the comprehensive manager.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.escalation.reset();
        self.backoff.reset();
        self.quality.reset();
        self.circuit_breaker.reset();
        self.fallback.reset();
        self.sync_loss.reset();
        self.stats = ComprehensiveRecoveryStats::default();
    }

    /// Returns the recovery engine.
    pub fn engine(&self) -> &DetectionRecovery {
        &self.engine
    }

    /// Returns the escalation manager.
    pub fn escalation(&self) -> &RecoveryEscalationManager {
        &self.escalation
    }

    /// Returns the backoff scheduler.
    pub fn backoff(&self) -> &RecoveryBackoffScheduler {
        &self.backoff
    }

    /// Returns the quality monitor.
    pub fn quality(&self) -> &DetectionQualityMonitor {
        &self.quality
    }

    /// Returns the circuit breaker.
    pub fn circuit_breaker(&self) -> &RecoveryCircuitBreaker {
        &self.circuit_breaker
    }

    /// Returns the manager statistics.
    pub fn stats(&self) -> &ComprehensiveRecoveryStats {
        &self.stats
    }
}

impl Default for ComprehensiveRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // RecoveryOutcome tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_outcome_variants() {
        assert_eq!(RecoveryOutcome::Success as u32, 0);
        assert_eq!(RecoveryOutcome::Partial as u32, 1);
        assert_eq!(RecoveryOutcome::Failed as u32, 2);
        assert_eq!(RecoveryOutcome::InProgress as u32, 3);
        assert_eq!(RecoveryOutcome::Skipped as u32, 4);
    }

    #[test]
    fn recovery_outcome_is_success() {
        assert!(RecoveryOutcome::Success.is_success());
        assert!(RecoveryOutcome::Partial.is_success());
        assert!(!RecoveryOutcome::Failed.is_success());
        assert!(!RecoveryOutcome::InProgress.is_success());
        assert!(!RecoveryOutcome::Skipped.is_success());
    }

    #[test]
    fn recovery_outcome_is_failure() {
        assert!(RecoveryOutcome::Failed.is_failure());
        assert!(!RecoveryOutcome::Success.is_failure());
    }

    #[test]
    fn recovery_outcome_is_ongoing() {
        assert!(RecoveryOutcome::InProgress.is_ongoing());
        assert!(!RecoveryOutcome::Success.is_ongoing());
    }

    #[test]
    fn recovery_outcome_default() {
        assert_eq!(RecoveryOutcome::default(), RecoveryOutcome::Skipped);
    }

    // ---------------------------------------------------------------------------
    // RecoveryStrategy tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_strategy_variants() {
        assert_eq!(RecoveryStrategy::ImmediateRetry as u32, 0);
        assert_eq!(RecoveryStrategy::RetryWithBackoff as u32, 1);
        assert_eq!(RecoveryStrategy::FallbackMode as u32, 2);
        assert_eq!(RecoveryStrategy::ReSynchronize as u32, 3);
        assert_eq!(RecoveryStrategy::ResetState as u32, 4);
        assert_eq!(RecoveryStrategy::GracefulDegradation as u32, 5);
        assert_eq!(RecoveryStrategy::Abort as u32, 6);
        assert_eq!(RecoveryStrategy::Escalate as u32, 7);
    }

    #[test]
    fn recovery_strategy_is_retry() {
        assert!(RecoveryStrategy::ImmediateRetry.is_retry());
        assert!(RecoveryStrategy::RetryWithBackoff.is_retry());
        assert!(!RecoveryStrategy::FallbackMode.is_retry());
    }

    #[test]
    fn recovery_strategy_is_resync() {
        assert!(RecoveryStrategy::ReSynchronize.is_resync());
        assert!(!RecoveryStrategy::ImmediateRetry.is_resync());
    }

    #[test]
    fn recovery_strategy_is_reset() {
        assert!(RecoveryStrategy::ResetState.is_reset());
        assert!(!RecoveryStrategy::ImmediateRetry.is_reset());
    }

    #[test]
    fn recovery_strategy_default_backoff_ms() {
        assert_eq!(RecoveryStrategy::ImmediateRetry.default_backoff_ms(), 0);
        assert_eq!(RecoveryStrategy::RetryWithBackoff.default_backoff_ms(), 100);
        assert_eq!(RecoveryStrategy::ReSynchronize.default_backoff_ms(), 500);
    }

    #[test]
    fn recovery_strategy_max_attempts() {
        assert_eq!(RecoveryStrategy::ImmediateRetry.max_attempts(), 3);
        assert_eq!(RecoveryStrategy::RetryWithBackoff.max_attempts(), 5);
        assert_eq!(RecoveryStrategy::Abort.max_attempts(), 1);
    }

    #[test]
    fn recovery_strategy_default() {
        assert_eq!(
            RecoveryStrategy::default(),
            RecoveryStrategy::ImmediateRetry
        );
    }

    // ---------------------------------------------------------------------------
    // DetectionRecoveryConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn detection_recovery_config_default() {
        let config = DetectionRecoveryConfig::default();
        assert_eq!(config.default_strategy, RecoveryStrategy::ImmediateRetry);
        assert_eq!(config.max_recovery_attempts, 5);
        assert_eq!(config.base_backoff_ms, 100);
        assert_eq!(config.max_backoff_ms, 5000);
        assert!((config.backoff_multiplier - 2.0).abs() < 0.001);
        assert_eq!(config.attempt_timeout_ms, 1000);
        assert!(config.auto_recover);
        assert!(config.enable_fallback);
    }

    // ---------------------------------------------------------------------------
    // RecoveryStatistics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_statistics_default() {
        let stats = RecoveryStatistics::default();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful_recoveries, 0);
        assert_eq!(stats.failed_recoveries, 0);
    }

    // ---------------------------------------------------------------------------
    // RecoveryState tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_state_variants() {
        assert_eq!(RecoveryState::Normal as u32, 0);
        assert_eq!(RecoveryState::Recovering as u32, 1);
        assert_eq!(RecoveryState::Degraded as u32, 2);
        assert_eq!(RecoveryState::Exhausted as u32, 3);
        assert_eq!(RecoveryState::ReSynchronizing as u32, 4);
    }

    #[test]
    fn recovery_state_default() {
        assert_eq!(RecoveryState::default(), RecoveryState::Normal);
    }

    // ---------------------------------------------------------------------------
    // DetectionRecovery tests
    // ---------------------------------------------------------------------------

    #[test]
    fn detection_recovery_new() {
        let recovery = DetectionRecovery::new();
        assert_eq!(recovery.state, RecoveryState::Normal);
        assert_eq!(recovery.stats.total_attempts, 0);
        assert!(!recovery.in_degraded_mode);
    }

    #[test]
    fn detection_recovery_with_config() {
        let mut config = DetectionRecoveryConfig::default();
        config.max_recovery_attempts = 10;
        config.auto_recover = false;
        let recovery = DetectionRecovery::with_config(config);
        assert_eq!(recovery.config.max_recovery_attempts, 10);
        assert!(!recovery.config.auto_recover);
    }

    #[test]
    fn detection_recovery_on_detection_failure() {
        let mut recovery = DetectionRecovery::new();
        let (strategy, proceed) = recovery.on_detection_failure(Some("test error".to_string()));
        assert_eq!(strategy, RecoveryStrategy::ImmediateRetry);
        assert!(proceed);
        assert_eq!(recovery.state, RecoveryState::Recovering);
        assert_eq!(recovery.consecutive_failures, 1);
        assert_eq!(recovery.stats.total_attempts, 1);
    }

    #[test]
    fn detection_recovery_on_recovery_success() {
        let mut recovery = DetectionRecovery::new();
        recovery.on_detection_failure(None);
        recovery.on_recovery_success(RecoveryOutcome::Success);
        assert_eq!(recovery.state, RecoveryState::Normal);
        assert_eq!(recovery.stats.successful_recoveries, 1);
        assert_eq!(recovery.consecutive_failures, 0);
        assert_eq!(recovery.consecutive_successes, 1);
    }

    #[test]
    fn detection_recovery_on_recovery_partial() {
        let mut recovery = DetectionRecovery::new();
        recovery.on_detection_failure(None);
        recovery.on_recovery_success(RecoveryOutcome::Partial);
        assert_eq!(recovery.state, RecoveryState::Degraded);
        assert!(recovery.in_degraded_mode);
        assert_eq!(recovery.stats.partial_recoveries, 1);
    }

    #[test]
    fn detection_recovery_on_recovery_failure() {
        let mut recovery = DetectionRecovery::new();
        recovery.on_detection_failure(None);
        recovery.on_recovery_failure();
        assert_eq!(recovery.state, RecoveryState::Exhausted);
        assert_eq!(recovery.stats.failed_recoveries, 1);
    }

    #[test]
    fn detection_recovery_start_attempt() {
        let mut recovery = DetectionRecovery::new();
        recovery.start_recovery_attempt();
        assert!(recovery.current_attempt_start.is_some());
    }

    #[test]
    fn detection_recovery_enter_fallback_mode() {
        let mut recovery = DetectionRecovery::new();
        recovery.enter_fallback_mode(0.6);
        assert!(recovery.in_degraded_mode);
        assert_eq!(recovery.state, RecoveryState::Degraded);
        assert!((recovery.fallback_quality - 0.6).abs() < 0.001);
        assert_eq!(recovery.stats.fallback_activations, 1);
    }

    #[test]
    fn detection_recovery_reset_state() {
        let mut recovery = DetectionRecovery::new();
        recovery.on_detection_failure(None);
        recovery.consecutive_failures = 5;
        recovery.reset_state();
        assert_eq!(recovery.state, RecoveryState::Normal);
        assert_eq!(recovery.consecutive_failures, 0);
        assert!(!recovery.in_degraded_mode);
    }

    #[test]
    fn detection_recovery_resynchronize() {
        let mut recovery = DetectionRecovery::new();
        recovery.consecutive_failures = 5;
        recovery.current_backoff_ms = 1000;
        let outcome = recovery.resynchronize();
        assert_eq!(outcome, RecoveryOutcome::Success);
        assert_eq!(recovery.state, RecoveryState::Normal);
        assert_eq!(recovery.consecutive_failures, 0);
        assert_eq!(recovery.current_backoff_ms, 0);
    }

    #[test]
    fn detection_recovery_degraded_mode_expired() {
        let mut recovery = DetectionRecovery::new();
        // Config max_degraded_ms = 10000 by default.
        recovery.enter_fallback_mode(0.5);
        assert!(!recovery.degraded_mode_expired());
        // Can't easily test expiration without waiting, but structure is correct.
    }

    #[test]
    fn detection_recovery_exit_degraded_mode() {
        let mut recovery = DetectionRecovery::new();
        recovery.enter_fallback_mode(0.5);
        assert!(recovery.in_degraded_mode);
        recovery.exit_degraded_mode();
        assert!(!recovery.in_degraded_mode);
        assert_eq!(recovery.state, RecoveryState::Normal);
    }

    #[test]
    fn detection_recovery_history() {
        let mut recovery = DetectionRecovery::new();
        recovery.on_detection_failure(Some("err".to_string()));
        assert!(!recovery.history().is_empty());
    }

    #[test]
    fn detection_recovery_backoff_ms() {
        let mut recovery = DetectionRecovery::new();
        assert_eq!(recovery.backoff_ms(), 0);
        recovery.on_detection_failure(None);
        // After ImmediateRetry failure, backoff should be 0.
        assert_eq!(recovery.backoff_ms(), 0);
    }

    #[test]
    fn detection_recovery_is_states() {
        let recovery = DetectionRecovery::new();
        assert!(recovery.is_normal());
        assert!(!recovery.is_recovering());
        assert!(!recovery.is_exhausted());
        assert!(!recovery.is_resyncing());
    }

    #[test]
    fn detection_recovery_recovery_rate() {
        let mut recovery = DetectionRecovery::new();
        assert!((recovery.recovery_rate() - 0.0).abs() < 0.001);
        recovery.on_detection_failure(None);
        recovery.on_recovery_success(RecoveryOutcome::Success);
        assert!((recovery.recovery_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn detection_recovery_failure_rate() {
        let mut recovery = DetectionRecovery::new();
        assert!((recovery.failure_rate() - 0.0).abs() < 0.001);
        recovery.on_detection_failure(None);
        recovery.on_recovery_failure();
        assert!((recovery.failure_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn detection_recovery_reset_stats() {
        let mut recovery = DetectionRecovery::new();
        recovery.on_detection_failure(None);
        recovery.reset_stats();
        assert_eq!(recovery.stats.total_attempts, 0);
        assert_eq!(recovery.history().len(), 0);
    }

    #[test]
    fn detection_recovery_set_config() {
        let mut recovery = DetectionRecovery::new();
        let mut new_config = DetectionRecoveryConfig::default();
        new_config.max_recovery_attempts = 20;
        recovery.set_config(new_config);
        assert_eq!(recovery.config.max_recovery_attempts, 20);
    }

    #[test]
    fn detection_recovery_consecutive_counts() {
        let mut recovery = DetectionRecovery::new();
        assert_eq!(recovery.consecutive_failures(), 0);
        assert_eq!(recovery.consecutive_successes(), 0);
        recovery.on_detection_failure(None);
        assert_eq!(recovery.consecutive_failures(), 1);
        recovery.on_recovery_success(RecoveryOutcome::Success);
        assert_eq!(recovery.consecutive_successes(), 1);
    }

    #[test]
    fn detection_recovery_default() {
        let recovery = DetectionRecovery::default();
        assert_eq!(recovery.state, RecoveryState::Normal);
    }

    // ---------------------------------------------------------------------------
    // RecoveryContext tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_context_creation() {
        let context = RecoveryContext {
            error: Some("test".to_string()),
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 100,
            consecutive_failures: 2,
            signal_strength: 0.7,
            noise_level: 0.3,
            channel_unstable: false,
            hardware_healthy: true,
        };
        assert_eq!(context.pre_failure_quality, 0.8);
        assert!(context.hardware_healthy);
    }

    // ---------------------------------------------------------------------------
    // RecoveryStrategyAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn strategy_analyzer_new() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        assert!((analyzer.min_retry_quality - 0.3).abs() < 0.001);
        assert_eq!(analyzer.max_failures_before_escalate, 5);
    }

    #[test]
    fn strategy_analyzer_escalate_on_bad_hardware() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.9,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: false,
        };
        assert_eq!(
            analyzer.analyze(&context),
            RecoveryStrategy::Escalate
        );
    }

    #[test]
    fn strategy_analyzer_escalate_on_too_many_failures() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.9,
            time_since_last_success_ms: 10,
            consecutive_failures: 10,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: true,
        };
        assert_eq!(
            analyzer.analyze(&context),
            RecoveryStrategy::Escalate
        );
    }

    #[test]
    fn strategy_analyzer_fallback_on_high_noise() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.9,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.9,
            channel_unstable: false,
            hardware_healthy: true,
        };
        assert_eq!(
            analyzer.analyze(&context),
            RecoveryStrategy::FallbackMode
        );
    }

    #[test]
    fn strategy_analyzer_resync_on_weak_signal() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.9,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.1,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: true,
        };
        assert_eq!(
            analyzer.analyze(&context),
            RecoveryStrategy::ReSynchronize
        );
    }

    #[test]
    fn strategy_analyzer_backoff_on_unstable() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.9,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: true,
            hardware_healthy: true,
        };
        assert_eq!(
            analyzer.analyze(&context),
            RecoveryStrategy::RetryWithBackoff
        );
    }

    #[test]
    fn strategy_analyzer_immediate_retry_on_good_quality() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: true,
        };
        assert_eq!(
            analyzer.analyze(&context),
            RecoveryStrategy::ImmediateRetry
        );
    }

    #[test]
    fn strategy_analyzer_should_recover() {
        let analyzer = RecoveryStrategyAnalyzer::new();
        let good_context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: true,
        };
        assert!(analyzer.should_recover(&good_context));

        let bad_context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 10,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: false,
        };
        assert!(!analyzer.should_recover(&bad_context));
    }

    #[test]
    fn strategy_analyzer_default() {
        let analyzer = RecoveryStrategyAnalyzer::default();
        assert!((analyzer.resync_signal_threshold - 0.2).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // FallbackMode tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fallback_mode_variants() {
        assert_eq!(FallbackMode::ThresholdBased as u32, 0);
        assert_eq!(FallbackMode::LowPrecision as u32, 1);
        assert_eq!(FallbackMode::HistoricalAverage as u32, 2);
        assert_eq!(FallbackMode::None as u32, 3);
    }

    #[test]
    fn fallback_mode_default() {
        assert_eq!(FallbackMode::default(), FallbackMode::None);
    }

    // ---------------------------------------------------------------------------
    // FallbackDetectionResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fallback_detection_result_creation() {
        let tuple = SkyrmionTuple::new(&[1]).unwrap();
        let result = FallbackDetectionResult {
            tuple: Some(tuple.clone()),
            confidence: 0.8,
            reliable: true,
            mode: FallbackMode::ThresholdBased,
        };
        assert!(result.reliable);
        assert_eq!(result.mode, FallbackMode::ThresholdBased);
    }

    // ---------------------------------------------------------------------------
    // FallbackDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fallback_detector_new() {
        let detector = FallbackDetector::new();
        assert_eq!(detector.mode, FallbackMode::ThresholdBased);
        assert!((detector.min_confidence - 0.4).abs() < 0.001);
        assert_eq!(detector.detection_count, 0);
    }

    #[test]
    fn fallback_detect_threshold_based() {
        let mut detector = FallbackDetector::new();
        let result = detector.detect(1.5);
        assert!(result.tuple.is_some());
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn fallback_detect_low_precision() {
        let mut detector = FallbackDetector::new();
        detector.set_mode(FallbackMode::LowPrecision);
        let result = detector.detect(1.5);
        assert!(result.tuple.is_some());
    }

    #[test]
    fn fallback_detect_historical_average_empty() {
        let mut detector = FallbackDetector::new();
        detector.set_mode(FallbackMode::HistoricalAverage);
        let result = detector.detect(1.5);
        assert!(result.tuple.is_some());
    }

    #[test]
    fn fallback_detect_none() {
        let mut detector = FallbackDetector::new();
        detector.set_mode(FallbackMode::None);
        let result = detector.detect(1.5);
        assert!(result.tuple.is_none());
        assert!((result.confidence - 0.0).abs() < 0.001);
        assert!(!result.reliable);
    }

    #[test]
    fn fallback_detect_reliability_rate() {
        let mut detector = FallbackDetector::new();
        detector.detect(1.5); // High confidence.
        detector.detect(0.1); // Low confidence.
        let rate = detector.reliability_rate();
        assert!(rate >= 0.0);
        assert!(rate <= 1.0);
    }

    #[test]
    fn fallback_detector_history() {
        let mut detector = FallbackDetector::new();
        detector.detect(1.5);
        assert!(!detector.history().is_empty());
    }

    #[test]
    fn fallback_detector_reset() {
        let mut detector = FallbackDetector::new();
        detector.detect(1.5);
        assert!(!detector.history().is_empty());
        detector.reset();
        assert_eq!(detector.history().len(), 0);
        assert_eq!(detector.detection_count, 0);
    }

    #[test]
    fn fallback_detector_set_mode() {
        let mut detector = FallbackDetector::new();
        detector.set_mode(FallbackMode::LowPrecision);
        assert_eq!(detector.mode, FallbackMode::LowPrecision);
    }

    #[test]
    fn fallback_detector_set_min_confidence() {
        let mut detector = FallbackDetector::new();
        detector.set_min_confidence(0.8);
        assert!((detector.min_confidence() - 0.8).abs() < 0.001);
    }

    #[test]
    fn fallback_detector_default() {
        let detector = FallbackDetector::default();
        assert_eq!(detector.mode, FallbackMode::ThresholdBased);
    }

    // ---------------------------------------------------------------------------
    // SyncLossEvent tests
    // ---------------------------------------------------------------------------

    #[test]
    fn sync_loss_event_creation() {
        let event = SyncLossEvent {
            timestamp: Instant::now(),
            duration_ms: 100.0,
            severity: 0.5,
            auto_recovered: true,
        };
        assert!((event.duration_ms - 100.0).abs() < 0.001);
        assert!((event.severity - 0.5).abs() < 0.001);
        assert!(event.auto_recovered);
    }

    // ---------------------------------------------------------------------------
    // SyncLossDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn sync_loss_detector_new() {
        let detector = SyncLossDetector::new();
        assert_eq!(detector.max_gap_ms, 500);
        assert!((detector.min_severity - 0.3).abs() < 0.001);
        assert_eq!(detector.total_losses, 0);
    }

    #[test]
    fn sync_loss_detector_record_success() {
        let mut detector = SyncLossDetector::new();
        detector.loss_start = Some(Instant::now());
        detector.record_success();
        assert!(detector.loss_start.is_none());
        assert!(detector.last_success.is_some());
    }

    #[test]
    fn sync_loss_detector_record_gap_large() {
        let mut detector = SyncLossDetector::new();
        detector.record_gap(1000); // Above max_gap_ms (500).
        assert!(detector.in_loss());
    }

    #[test]
    fn sync_loss_detector_record_gap_small() {
        let mut detector = SyncLossDetector::new();
        detector.record_gap(100); // Below max_gap_ms.
        assert!(!detector.in_loss());
    }

    #[test]
    fn sync_loss_detector_end_loss() {
        let mut detector = SyncLossDetector::new();
        detector.record_gap(1000);
        assert!(detector.in_loss());
        detector.end_loss();
        assert!(!detector.in_loss());
        assert_eq!(detector.total_losses, 1);
    }

    #[test]
    fn sync_loss_detector_in_loss() {
        let detector = SyncLossDetector::new();
        assert!(!detector.in_loss());
    }

    #[test]
    fn sync_loss_detector_current_loss_duration() {
        let mut detector = SyncLossDetector::new();
        assert!((detector.current_loss_duration_ms() - 0.0).abs() < 0.001);
        detector.record_gap(1000);
        assert!(detector.current_loss_duration_ms() >= 0.0);
    }

    #[test]
    fn sync_loss_detector_events() {
        let mut detector = SyncLossDetector::new();
        detector.record_gap(1000);
        detector.end_loss();
        assert!(!detector.events().is_empty());
    }

    #[test]
    fn sync_loss_detector_avg_loss_duration() {
        let detector = SyncLossDetector::new();
        assert!((detector.avg_loss_duration_ms() - 0.0).abs() < 0.001);
    }

    #[test]
    fn sync_loss_detector_reset() {
        let mut detector = SyncLossDetector::new();
        detector.record_gap(1000);
        detector.end_loss();
        assert!(!detector.events().is_empty());
        detector.reset();
        assert_eq!(detector.events().len(), 0);
        assert_eq!(detector.total_losses, 0);
    }

    #[test]
    fn sync_loss_detector_set_max_gap_ms() {
        let mut detector = SyncLossDetector::new();
        detector.set_max_gap_ms(1000);
        assert_eq!(detector.max_gap_ms, 1000);
    }

    #[test]
    fn sync_loss_detector_set_min_severity() {
        let mut detector = SyncLossDetector::new();
        detector.set_min_severity(0.5);
        assert!((detector.min_severity - 0.5).abs() < 0.001);
    }

    #[test]
    fn sync_loss_detector_default() {
        let detector = SyncLossDetector::default();
        assert_eq!(detector.max_gap_ms, 500);
    }

    // ---------------------------------------------------------------------------
    // RecoveryAction tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_action_variants() {
        assert_eq!(RecoveryAction::ContinueDetection as u32, 0);
        assert_eq!(RecoveryAction::ContinueFallback as u32, 1);
        assert_eq!(RecoveryAction::RetryDetection as u32, 2);
        assert_eq!(RecoveryAction::Escalate as u32, 3);
        assert_eq!(RecoveryAction::Abort as u32, 4);
        assert_eq!(RecoveryAction::WaitBackoff as u32, 5);
    }

    // ---------------------------------------------------------------------------
    // PipelineRecoveryStats tests
    // ---------------------------------------------------------------------------

    #[test]
    fn pipeline_recovery_stats_default() {
        let stats = PipelineRecoveryStats::default();
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }

    // ---------------------------------------------------------------------------
    // RecoveryPipelineResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_pipeline_result_creation() {
        let result = RecoveryPipelineResult {
            outcome: RecoveryOutcome::Success,
            strategy: RecoveryStrategy::ImmediateRetry,
            fallback_result: None,
            degraded: false,
            recovery_time_ms: 10.0,
            next_action: RecoveryAction::ContinueDetection,
        };
        assert!(result.outcome.is_success());
        assert!(!result.degraded);
    }

    // ---------------------------------------------------------------------------
    // RecoveryPipeline tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_pipeline_new() {
        let pipeline = RecoveryPipeline::new();
        assert_eq!(pipeline.stats().total_runs, 0);
        assert!(!pipeline.is_degraded());
    }

    #[test]
    fn recovery_pipeline_recover_immediate_retry() {
        let mut pipeline = RecoveryPipeline::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: true,
        };
        let result = pipeline.recover(&context, None);
        assert!(result.outcome.is_success());
        assert_eq!(result.next_action, RecoveryAction::RetryDetection);
    }

    #[test]
    fn recovery_pipeline_recover_with_fallback() {
        let mut pipeline = RecoveryPipeline::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.9, // High noise triggers fallback.
            channel_unstable: false,
            hardware_healthy: true,
        };
        let result = pipeline.recover(&context, Some(1.5));
        assert!(result.degraded);
        assert_eq!(result.next_action, RecoveryAction::ContinueFallback);
    }

    #[test]
    fn recovery_pipeline_recover_escalate() {
        let mut pipeline = RecoveryPipeline::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 10, // Too many failures.
            signal_strength: 0.9,
            noise_level: 0.1,
            channel_unstable: false,
            hardware_healthy: false, // Bad hardware triggers escalate.
        };
        let result = pipeline.recover(&context, None);
        assert_eq!(result.outcome, RecoveryOutcome::Failed);
        assert_eq!(result.next_action, RecoveryAction::Escalate);
    }

    #[test]
    fn recovery_pipeline_on_success() {
        let mut pipeline = RecoveryPipeline::new();
        pipeline.recover(
            &RecoveryContext {
                error: None,
                pre_failure_quality: 0.8,
                time_since_last_success_ms: 10,
                consecutive_failures: 1,
                signal_strength: 0.9,
                noise_level: 0.1,
                channel_unstable: false,
                hardware_healthy: true,
            },
            None,
        );
        pipeline.on_success();
        assert!(!pipeline.is_degraded());
    }

    #[test]
    fn recovery_pipeline_is_degraded() {
        let mut pipeline = RecoveryPipeline::new();
        assert!(!pipeline.is_degraded());
        pipeline.recover(
            &RecoveryContext {
                error: None,
                pre_failure_quality: 0.8,
                time_since_last_success_ms: 10,
                consecutive_failures: 1,
                signal_strength: 0.9,
                noise_level: 0.9,
                channel_unstable: false,
                hardware_healthy: true,
            },
            Some(1.5),
        );
        assert!(pipeline.is_degraded());
    }

    #[test]
    fn recovery_pipeline_stats() {
        let mut pipeline = RecoveryPipeline::new();
        pipeline.recover(
            &RecoveryContext {
                error: None,
                pre_failure_quality: 0.8,
                time_since_last_success_ms: 10,
                consecutive_failures: 1,
                signal_strength: 0.9,
                noise_level: 0.1,
                channel_unstable: false,
                hardware_healthy: true,
            },
            None,
        );
        assert_eq!(pipeline.stats().total_runs, 1);
        assert_eq!(pipeline.stats().successful, 1);
    }

    #[test]
    fn recovery_pipeline_reset() {
        let mut pipeline = RecoveryPipeline::new();
        pipeline.recover(
            &RecoveryContext {
                error: None,
                pre_failure_quality: 0.8,
                time_since_last_success_ms: 10,
                consecutive_failures: 1,
                signal_strength: 0.9,
                noise_level: 0.1,
                channel_unstable: false,
                hardware_healthy: true,
            },
            None,
        );
        assert_eq!(pipeline.stats().total_runs, 1);
        pipeline.reset();
        assert_eq!(pipeline.stats().total_runs, 0);
    }

    #[test]
    fn recovery_pipeline_default() {
        let pipeline = RecoveryPipeline::default();
        assert_eq!(pipeline.stats().total_runs, 0);
    }

    // ---------------------------------------------------------------------------
    // Edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recovery_retry_with_backoff_delay() {
        let mut recovery = DetectionRecovery::new();
        // Force backoff strategy.
        recovery.config.default_strategy = RecoveryStrategy::RetryWithBackoff;
        recovery.on_detection_failure(None);
        assert!(recovery.backoff_ms() > 0);
    }

    #[test]
    fn recovery_fallback_quality_bounds() {
        let mut recovery = DetectionRecovery::new();
        recovery.enter_fallback_mode(-0.5); // Below 0.
        assert!((recovery.fallback_quality - 0.0).abs() < 0.001);
        recovery.enter_fallback_mode(1.5); // Above 1.
        assert!((recovery.fallback_quality - 1.0).abs() < 0.001);
    }

    #[test]
    fn recovery_pipeline_recover_no_measurement() {
        let mut pipeline = RecoveryPipeline::new();
        let context = RecoveryContext {
            error: None,
            pre_failure_quality: 0.8,
            time_since_last_success_ms: 10,
            consecutive_failures: 1,
            signal_strength: 0.9,
            noise_level: 0.9,
            channel_unstable: false,
            hardware_healthy: true,
        };
        let result = pipeline.recover(&context, None); // No measurement for fallback.
        assert!(result.degraded);
        assert_eq!(result.fallback_result, None);
    }

    #[test]
    fn fallback_detector_reliability_rate_empty() {
        let detector = FallbackDetector::new();
        assert!((detector.reliability_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn sync_loss_detector_avg_duration_empty() {
        let detector = SyncLossDetector::new();
        assert!((detector.avg_loss_duration_ms() - 0.0).abs() < 0.001);
    }
}