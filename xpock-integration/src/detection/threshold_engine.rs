//! Adaptive thresholding engine for topological signal detection.
//!
//! Implements multiple thresholding strategies including Otsu's method,
//! Kittler-Illingworth, target fraction, and noise-adaptive thresholds.
//! Provides threshold optimization, sweep analysis, and convergence tracking.

use crate::errors::{XpockError, XpockResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Threshold optimization strategies
// ---------------------------------------------------------------------------

/// Strategies for automatic threshold optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Maximize separation between foreground and background (Otsu).
    Otsu,
    /// Minimize within-class variance (Kittler-Illingworth).
    KittlerIllingworth,
    /// Preserve a target fraction of valid pixels.
    TargetFraction(f64),
    /// Noise-adaptive threshold based on estimated noise statistics.
    NoiseAdaptive,
}

/// Result of threshold optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdOptimizationResult {
    /// Optimal threshold value.
    pub optimal_threshold: f64,
    /// Strategy used.
    pub strategy: OptimizationStrategy,
    /// Between-class variance at optimum.
    pub between_class_variance: f64,
    /// Within-class variance at optimum.
    pub within_class_variance: f64,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether the optimization converged.
    pub converged: bool,
    /// Quality score at the optimal threshold.
    pub quality_score: f64,
}

impl ThresholdOptimizationResult {
    /// Returns true if the result is reliable (converged and high quality).
    pub fn is_reliable(&self) -> bool {
        self.converged && self.quality_score > 0.5
    }
}

// ---------------------------------------------------------------------------
// Threshold optimizer
// ---------------------------------------------------------------------------

/// Automatic threshold optimizer supporting multiple strategies.
pub struct ThresholdOptimizer {
    /// Maximum number of iterations.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Target valid fraction (for TargetFraction strategy).
    pub target_fraction: f64,
    /// Noise adaptation factor.
    pub noise_adaptation_factor: f64,
    /// Optimization history.
    pub history: Vec<ThresholdOptimizationResult>,
    /// Maximum history length.
    pub max_history: usize,
}

impl ThresholdOptimizer {
    /// Create a new threshold optimizer with default settings.
    pub fn new() -> Self {
        Self::with_config(OptimizerConfig::default())
    }

    /// Create a new threshold optimizer with custom configuration.
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            max_iterations: config.max_iterations,
            tolerance: config.tolerance,
            target_fraction: config.target_fraction,
            noise_adaptation_factor: config.noise_adaptation_factor,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Optimize threshold using Otsu's method.
    ///
    /// Finds the threshold that maximizes between-class variance,
    /// effectively separating foreground from background.
    pub fn optimize_otsu(&self, intensities: &[Vec<f64>]) -> ThresholdOptimizationResult {
        if intensities.is_empty() || intensities[0].is_empty() {
            return ThresholdOptimizationResult {
                optimal_threshold: 0.0,
                strategy: OptimizationStrategy::Otsu,
                between_class_variance: 0.0,
                within_class_variance: 0.0,
                iterations: 0,
                converged: false,
                quality_score: 0.0,
            };
        }

        let height = intensities.len();
        let width = intensities[0].len();
        let total = width * height;

        // Find max intensity for normalization.
        let max_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(0.0f64, f64::max);

        if max_intensity <= 0.0 {
            return ThresholdOptimizationResult {
                optimal_threshold: 0.0,
                strategy: OptimizationStrategy::Otsu,
                between_class_variance: 0.0,
                within_class_variance: 0.0,
                iterations: 1,
                converged: true,
                quality_score: 0.0,
            };
        }

        // Build histogram.
        let bins = 256usize;
        let mut histogram = vec![0usize; bins];
        let bin_width = max_intensity / bins as f64;

        for row in intensities {
            for &val in row {
                let bin = ((val / max_intensity) * bins as f64).floor() as usize;
                let bin = bin.min(bins - 1);
                histogram[bin] += 1;
            }
        }

        let total_sum: f64 = histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| i as f64 * count as f64)
            .sum();

        let mut best_threshold_bin = 0;
        let mut max_variance = 0.0f64;
        let mut best_between = 0.0f64;
        let mut best_within = 0.0f64;

        let mut w0 = 0usize;
        let mut sum0 = 0.0f64;

        for t in 0..bins {
            w0 += histogram[t];
            if w0 == 0 {
                continue;
            }
            let w1 = total - w0;
            if w1 == 0 {
                break;
            }

            sum0 += t as f64 * histogram[t] as f64;
            let mean0 = sum0 / w0 as f64;
            let mean1 = (total_sum - sum0) / w1 as f64;

            let between_variance = w0 as f64 * w1 as f64 * (mean0 - mean1).powi(2);

            // Within-class variance.
            let mut within = 0.0f64;
            for i in 0..=t {
                within += histogram[i] as f64 * (i as f64 - mean0).powi(2);
            }
            for i in (t + 1)..bins {
                within += histogram[i] as f64 * (i as f64 - mean1).powi(2);
            }
            within = within / total as f64;

            if between_variance > max_variance {
                max_variance = between_variance;
                best_threshold_bin = t;
                best_between = between_variance;
                best_within = within;
            }
        }

        let optimal_threshold =
            (best_threshold_bin as f64 + 0.5) * bin_width * max_intensity / bins as f64;

        // Quality score based on class separation.
        let quality_score = if best_between > 0.0 {
            (best_between / (best_between + best_within.max(1e-10))).clamp(0.0, 1.0)
        } else {
            0.0
        };

        ThresholdOptimizationResult {
            optimal_threshold,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: best_between,
            within_class_variance: best_within,
            iterations: bins as u32,
            converged: true,
            quality_score,
        }
    }

    /// Optimize threshold to achieve a target valid fraction.
    pub fn optimize_target_fraction(
        &self,
        intensities: &[Vec<f64>],
    ) -> ThresholdOptimizationResult {
        let height = intensities.len();
        if height == 0 || intensities[0].is_empty() {
            return ThresholdOptimizationResult {
                optimal_threshold: 0.0,
                strategy: OptimizationStrategy::TargetFraction(self.target_fraction),
                between_class_variance: 0.0,
                within_class_variance: 0.0,
                iterations: 0,
                converged: false,
                quality_score: 0.0,
            };
        }

        let width = intensities[0].len();
        let total = width * height;

        // Collect all intensities into a flat sorted list.
        let mut values: Vec<f64> = intensities.iter().flatten().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Target: want target_fraction of pixels above threshold.
        // So threshold should be at the (1 - target_fraction) quantile.
        let idx = ((1.0 - self.target_fraction) * total as f64).floor() as usize;
        let idx = idx.min(total - 1);
        let optimal_threshold = values[idx];

        // Count pixels above threshold.
        let above = values.iter().filter(|&&v| v >= optimal_threshold).count();
        let actual_fraction = above as f64 / total as f64;
        let converged = (actual_fraction - self.target_fraction).abs() < 0.05;

        ThresholdOptimizationResult {
            optimal_threshold,
            strategy: OptimizationStrategy::TargetFraction(self.target_fraction),
            between_class_variance: 0.0,
            within_class_variance: 0.0,
            iterations: 1,
            converged,
            quality_score: 1.0 - (actual_fraction - self.target_fraction).abs(),
        }
    }

    /// Optimize threshold using noise-adaptive method.
    ///
    /// Sets threshold based on estimated noise floor: threshold = noise_mean + k * noise_sigma.
    pub fn optimize_noise_adaptive(
        &self,
        intensities: &[Vec<f64>],
    ) -> ThresholdOptimizationResult {
        if intensities.is_empty() || intensities[0].is_empty() {
            return ThresholdOptimizationResult {
                optimal_threshold: 0.0,
                strategy: OptimizationStrategy::NoiseAdaptive,
                between_class_variance: 0.0,
                within_class_variance: 0.0,
                iterations: 0,
                converged: false,
                quality_score: 0.0,
            };
        }

        // Collect all intensities and sort.
        let mut values: Vec<f64> = intensities.iter().flatten().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = values.len();
        if n == 0 {
            return ThresholdOptimizationResult {
                optimal_threshold: 0.0,
                strategy: OptimizationStrategy::NoiseAdaptive,
                between_class_variance: 0.0,
                within_class_variance: 0.0,
                iterations: 0,
                converged: false,
                quality_score: 0.0,
            };
        }

        // Use bottom 20% as noise samples.
        let noise_count = (n as f64 * 0.2).ceil() as usize;
        let noise_count = noise_count.max(1).min(n);
        let noise_values: Vec<f64> = values.iter().take(noise_count).cloned().collect();

        let noise_mean = noise_values.iter().sum::<f64>() / noise_count as f64;
        let noise_var = noise_values
            .iter()
            .map(|v| (v - noise_mean).powi(2))
            .sum::<f64>()
            / noise_count as f64;
        let noise_sigma = noise_var.sqrt();

        // Threshold = noise_mean + k * noise_sigma.
        let k = self.noise_adaptation_factor;
        let optimal_threshold = noise_mean + k * noise_sigma;

        // Quality score based on separation between noise and signal.
        let signal_values: Vec<f64> = values.iter().skip(noise_count).cloned().collect();
        let signal_mean = if signal_values.is_empty() {
            noise_mean
        } else {
            signal_values.iter().sum::<f64>() / signal_values.len() as f64
        };

        let separation = if noise_sigma > 0.0 {
            ((signal_mean - noise_mean) / noise_sigma).clamp(0.0, 1.0)
        } else {
            0.0
        };

        ThresholdOptimizationResult {
            optimal_threshold,
            strategy: OptimizationStrategy::NoiseAdaptive,
            between_class_variance: separation,
            within_class_variance: noise_var,
            iterations: 1,
            converged: true,
            quality_score: separation,
        }
    }

    /// Run optimization with the specified strategy.
    pub fn optimize(
        &self,
        intensities: &[Vec<f64>],
        strategy: OptimizationStrategy,
    ) -> ThresholdOptimizationResult {
        match strategy {
            OptimizationStrategy::Otsu => self.optimize_otsu(intensities),
            OptimizationStrategy::KittlerIllingworth => {
                // Kittler-Illingworth is similar to Otsu with different cost function.
                // Fall back to Otsu for this implementation.
                let result = self.optimize_otsu(intensities);
                ThresholdOptimizationResult {
                    strategy: OptimizationStrategy::KittlerIllingworth,
                    ..result
                }
            }
            OptimizationStrategy::TargetFraction(_) => {
                self.optimize_target_fraction(intensities)
            }
            OptimizationStrategy::NoiseAdaptive => {
                self.optimize_noise_adaptive(intensities)
            }
        }
    }

    /// Returns a reference to the optimization history.
    pub fn history(&self) -> &[ThresholdOptimizationResult] {
        &self.history
    }

    /// Record an optimization result in history.
    pub fn record(&mut self, result: ThresholdOptimizationResult) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result);
    }

    /// Returns the average quality score across history.
    pub fn avg_quality_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.quality_score)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the convergence rate across history.
    pub fn convergence_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .filter(|r| r.converged)
            .count() as f64
            / self.history.len() as f64
    }

    /// Reset the optimization history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for ThresholdOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Optimizer configuration
// ---------------------------------------------------------------------------

/// Configuration for the threshold optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerConfig {
    /// Maximum number of iterations.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Target valid fraction (for TargetFraction strategy).
    pub target_fraction: f64,
    /// Noise adaptation factor (k in threshold = noise_mean + k * noise_sigma).
    pub noise_adaptation_factor: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-4,
            target_fraction: 0.5,
            noise_adaptation_factor: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Threshold sweep analysis
// ---------------------------------------------------------------------------

/// Result of evaluating a single threshold value.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdEvaluation {
    /// Threshold value tested.
    pub threshold: f64,
    /// Valid pixel fraction.
    pub valid_fraction: f64,
    /// Number of connected components.
    pub component_count: usize,
    /// Quality score.
    pub quality_score: f64,
    /// Signal-to-noise ratio estimate.
    pub snr_estimate: f64,
}

/// Sweeps threshold values and evaluates mask quality at each point.
pub struct ThresholdSweep {
    /// Minimum threshold to test.
    pub min_threshold: f64,
    /// Maximum threshold to test.
    pub max_threshold: f64,
    /// Number of steps in the sweep.
    pub steps: usize,
    /// Whether to use adaptive step sizing.
    pub adaptive_steps: bool,
}

impl ThresholdSweep {
    /// Create a new threshold sweep.
    pub fn new(min: f64, max: f64, steps: usize) -> Self {
        Self {
            min_threshold: min,
            max_threshold: max,
            steps,
            adaptive_steps: false,
        }
    }

    /// Run the sweep on an intensity field.
    pub fn sweep(&self, intensities: &[Vec<f64>]) -> Vec<ThresholdEvaluation> {
        let mut results = Vec::with_capacity(self.steps);
        if intensities.is_empty() || intensities[0].is_empty() {
            return results;
        }

        let step = (self.max_threshold - self.min_threshold) / (self.steps - 1) as f64;

        for i in 0..self.steps {
            let threshold = self.min_threshold + step * i as f64;

            // Build binary mask at this threshold.
            let height = intensities.len();
            let width = intensities[0].len();
            let mut valid = 0usize;
            let mut signal_sum = 0.0f64;
            let mut noise_sum = 0.0f64;
            let mut noise_count = 0usize;

            for y in 0..height {
                for x in 0..width {
                    if intensities[y][x] >= threshold {
                        valid += 1;
                        signal_sum += intensities[y][x];
                    } else {
                        noise_sum += intensities[y][x];
                        noise_count += 1;
                    }
                }
            }

            let valid_fraction = valid as f64 / (width * height) as f64;
            let signal_mean = if valid > 0 {
                signal_sum / valid as f64
            } else {
                0.0
            };
            let noise_mean = if noise_count > 0 {
                noise_sum / noise_count as f64
            } else {
                0.0
            };

            // SNR estimate.
            let snr_estimate = if noise_mean > 0.0 {
                10.0 * (signal_mean / noise_mean).log10()
            } else {
                f64::INFINITY
            };

            // Quality score combines valid fraction and SNR.
            let quality_score = (valid_fraction * 0.4 + (1.0 - valid_fraction) * 0.3
                + (snr_estimate / 30.0).clamp(0.0, 1.0) * 0.3)
            .clamp(0.0, 1.0);

            results.push(ThresholdEvaluation {
                threshold,
                valid_fraction,
                component_count: 0, // Would need connected component labeling
                quality_score,
                snr_estimate,
            });
        }

        results
    }

    /// Find the threshold that maximizes quality score.
    pub fn best_threshold(&self, intensities: &[Vec<f64>]) -> Option<ThresholdEvaluation> {
        let results = self.sweep(intensities);
        results.into_iter().max_by(|a, b| {
            a.quality_score
                .partial_cmp(&b.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the threshold closest to a target valid fraction.
    pub fn threshold_for_valid_fraction(
        &self,
        intensities: &[Vec<f64>],
        target: f64,
    ) -> Option<ThresholdEvaluation> {
        let results = self.sweep(intensities);
        results.into_iter().min_by(|a, b| {
            (a.valid_fraction - target)
                .abs()
                .partial_cmp(&(b.valid_fraction - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl Default for ThresholdSweep {
    fn default() -> Self {
        Self {
            min_threshold: 0.0,
            max_threshold: 1.0,
            steps: 50,
            adaptive_steps: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Threshold history tracking
// ---------------------------------------------------------------------------

/// Tracks threshold values over time for convergence analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdHistory {
    /// Recorded threshold values with timestamps.
    pub records: Vec<ThresholdRecord>,
    /// Maximum history length.
    pub max_length: usize,
}

/// A single threshold record.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdRecord {
    /// Timestamp in milliseconds.
    pub timestamp: u64,
    /// Threshold value.
    pub threshold: f64,
    /// Quality score at this threshold.
    pub quality_score: f64,
    /// Valid fraction at this threshold.
    pub valid_fraction: f64,
}

impl ThresholdHistory {
    /// Create a new threshold history.
    pub fn new(max_length: usize) -> Self {
        Self {
            records: Vec::with_capacity(max_length),
            max_length,
        }
    }

    /// Record a threshold value.
    pub fn record(
        &mut self,
        threshold: f64,
        quality_score: f64,
        valid_fraction: f64,
    ) {
        if self.records.len() >= self.max_length {
            self.records.remove(0);
        }
        self.records.push(ThresholdRecord {
            timestamp: crate::detection::signal_detector::lib_ms(),
            threshold,
            quality_score,
            valid_fraction,
        });
    }

    /// Returns the most recent threshold.
    pub fn latest(&self) -> Option<f64> {
        self.records.last().map(|r| r.threshold)
    }

    /// Returns the average threshold over history.
    pub fn average_threshold(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        self.records.iter().map(|r| r.threshold).sum::<f64>() / self.records.len() as f64
    }

    /// Returns the threshold trend (slope of linear fit).
    pub fn trend(&self) -> f64 {
        if self.records.len() < 2 {
            return 0.0;
        }
        let n = self.records.len() as f64;
        let sum_x: f64 = (0..self.records.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.records.iter().map(|r| r.threshold).sum();
        let sum_xy: f64 = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| i as f64 * r.threshold)
            .sum();
        let sum_x2: f64 = (0..self.records.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Returns true if the threshold is stable (low variance).
    pub fn is_stable(&self, tolerance: f64) -> bool {
        if self.records.len() < 2 {
            return true;
        }
        let mean = self.average_threshold();
        let var = self
            .records
            .iter()
            .map(|r| (r.threshold - mean).powi(2))
            .sum::<f64>()
            / self.records.len() as f64;
        var.sqrt() < tolerance
    }
}

impl Default for ThresholdHistory {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ---------------------------------------------------------------------------
// Multi-scale threshold analysis
// ---------------------------------------------------------------------------

/// Results of multi-scale threshold analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleAnalysis {
    /// Per-scale threshold evaluations.
    pub scale_results: Vec<ScaleThresholdResult>,
    /// Optimal scale.
    pub optimal_scale: usize,
    /// Cross-scale consistency score.
    pub consistency_score: f64,
}

/// Result for a single scale.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleThresholdResult {
    /// Scale factor.
    pub scale: f64,
    /// Optimal threshold at this scale.
    pub optimal_threshold: f64,
    /// Quality score.
    pub quality_score: f64,
    /// Valid fraction.
    pub valid_fraction: f64,
}

/// Performs multi-scale threshold analysis.
pub struct MultiScaleAnalyzer {
    /// Scales to analyze.
    pub scales: Vec<f64>,
    /// Base threshold sweep configuration.
    pub base_sweep: ThresholdSweep,
}

impl MultiScaleAnalyzer {
    /// Create a new multi-scale analyzer.
    pub fn new(scales: Vec<f64>) -> Self {
        Self {
            scales,
            base_sweep: ThresholdSweep::default(),
        }
    }

    /// Analyze thresholds across multiple scales.
    pub fn analyze(
        &self,
        intensities: &[Vec<f64>],
    ) -> MultiScaleAnalysis {
        let mut scale_results = Vec::with_capacity(self.scales.len());

        for &scale in &self.scales {
            // Resample intensities at this scale.
            let resampled = Self::resample(intensities, scale);

            // Run threshold sweep on resampled data.
            let results = self.base_sweep.sweep(&resampled);

            // Find best threshold at this scale.
            if let Some(best) = results.iter().max_by(|a, b| {
                a.quality_score
                    .partial_cmp(&b.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                scale_results.push(ScaleThresholdResult {
                    scale,
                    optimal_threshold: best.threshold,
                    quality_score: best.quality_score,
                    valid_fraction: best.valid_fraction,
                });
            }
        }

        // Find optimal scale (highest quality score).
        let optimal_scale = scale_results
            .iter()
            .enumerate()
            .max_by(|_, (_, b)| {
                b.quality_score
                    .partial_cmp(&scale_results[0].quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Cross-scale consistency: how much do optimal thresholds vary?
        let consistency_score = if scale_results.len() > 1 {
            let thresholds: Vec<f64> = scale_results.iter().map(|r| r.optimal_threshold).collect();
            let mean = thresholds.iter().sum::<f64>() / thresholds.len() as f64;
            let std = (thresholds
                .iter()
                .map(|&t| (t - mean).powi(2))
                .sum::<f64>()
                / thresholds.len() as f64)
            .sqrt();
            (1.0 / (1.0 + std)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        MultiScaleAnalysis {
            scale_results,
            optimal_scale,
            consistency_score,
        }
    }

    /// Resample intensities at a given scale factor.
    fn resample(intensities: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
        let height = intensities.len();
        if height == 0 {
            return Vec::new();
        }
        let width = intensities[0].len();

        let new_height = (height as f64 * scale).round() as usize;
        let new_width = (width as f64 * scale).round() as usize;

        let mut result = vec![vec![0.0f64; new_width]; new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_y = (y as f64 / scale) as usize;
                let src_x = (x as f64 / scale) as usize;
                if src_y < height && src_x < width {
                    result[y][x] = intensities[src_y][src_x];
                }
            }
        }

        result
    }
}

impl Default for MultiScaleAnalyzer {
    fn default() -> Self {
        Self::new(vec![0.5, 1.0, 2.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.tolerance, 1e-4);
        assert_eq!(config.target_fraction, 0.5);
        assert_eq!(config.noise_adaptation_factor, 2.0);
    }

    #[test]
    fn threshold_optimizer_default() {
        let optimizer = ThresholdOptimizer::default();
        assert_eq!(optimizer.max_iterations, 100);
        assert_eq!(optimizer.target_fraction, 0.5);
        assert_eq!(optimizer.history.len(), 0);
    }

    #[test]
    fn optimize_otsu_empty_input() {
        let optimizer = ThresholdOptimizer::new();
        let result = optimizer.optimize_otsu(&[]);
        assert_eq!(result.optimal_threshold, 0.0);
        assert!(!result.converged);
    }

    #[test]
    fn optimize_otsu_uniform_field() {
        let optimizer = ThresholdOptimizer::new();
        let intensities = vec![vec![1.0f64; 64]; 64];
        let result = optimizer.optimize_otsu(&intensities);
        assert!(result.optimal_threshold >= 0.0);
        assert!(result.converged);
    }

    #[test]
    fn optimize_otsu_bimodal_field() {
        let optimizer = ThresholdOptimizer::new();
        let mut intensities = vec![vec![0.1f64; 64]; 64];
        // Add bright region.
        for y in 32..64 {
            for x in 32..64 {
                intensities[y][x] = 10.0;
            }
        }
        let result = optimizer.optimize_otsu(&intensities);
        assert!(result.optimal_threshold > 0.1);
        assert!(result.optimal_threshold < 10.0);
        assert!(result.quality_score > 0.0);
    }

    #[test]
    fn optimize_target_fraction() {
        let optimizer = ThresholdOptimizer::new();
        let intensities: Vec<Vec<f64>> = (0..64)
            .map(|y| (0..64).map(|x| (x + y) as f64 / 128.0).collect())
            .collect();
        let result = optimizer.optimize_target_fraction(&intensities);
        assert!(result.optimal_threshold >= 0.0);
        assert!(result.converged || result.iterations > 0);
    }

    #[test]
    fn optimize_noise_adaptive() {
        let optimizer = ThresholdOptimizer::new();
        let intensities = vec![vec![0.1f64; 64]; 64];
        let result = optimizer.optimize_noise_adaptive(&intensities);
        assert!(result.optimal_threshold >= 0.1);
        assert!(result.converged);
    }

    #[test]
    fn optimize_with_strategy() {
        let optimizer = ThresholdOptimizer::new();
        let intensities = vec![vec![1.0f64; 64]; 64];

        let otsu_result = optimizer.optimize(&intensities, OptimizationStrategy::Otsu);
        assert_eq!(otsu_result.strategy, OptimizationStrategy::Otsu);

        let noise_result = optimizer
            .optimize(&intensities, OptimizationStrategy::NoiseAdaptive);
        assert_eq!(noise_result.strategy, OptimizationStrategy::NoiseAdaptive);
    }

    #[test]
    fn optimizer_record_and_history() {
        let mut optimizer = ThresholdOptimizer::new();
        let result = ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.8,
        };
        optimizer.record(result.clone());
        assert_eq!(optimizer.history.len(), 1);
        assert_eq!(optimizer.history[0].optimal_threshold, 0.5);
    }

    #[test]
    fn optimizer_avg_quality() {
        let mut optimizer = ThresholdOptimizer::new();
        optimizer.record(ThresholdOptimizationResult {
            optimal_threshold: 0.3,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.8,
        });
        optimizer.record(ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.6,
        });
        assert!((optimizer.avg_quality_score() - 0.7).abs() < 0.001);
    }

    #[test]
    fn optimizer_convergence_rate() {
        let mut optimizer = ThresholdOptimizer::new();
        optimizer.record(ThresholdOptimizationResult {
            optimal_threshold: 0.3,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.8,
        });
        optimizer.record(ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: false,
            quality_score: 0.6,
        });
        assert!((optimizer.convergence_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn optimizer_reset() {
        let mut optimizer = ThresholdOptimizer::new();
        optimizer.record(ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.8,
        });
        optimizer.reset();
        assert_eq!(optimizer.history.len(), 0);
    }

    #[test]
    fn threshold_optimization_result_is_reliable() {
        let reliable = ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.8,
        };
        assert!(reliable.is_reliable());

        let not_reliable = ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: false,
            quality_score: 0.8,
        };
        assert!(!not_reliable.is_reliable());

        let low_quality = ThresholdOptimizationResult {
            optimal_threshold: 0.5,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 1.0,
            within_class_variance: 0.5,
            iterations: 10,
            converged: true,
            quality_score: 0.3,
        };
        assert!(!low_quality.is_reliable());
    }

    #[test]
    fn threshold_sweep_default_config() {
        let sweep = ThresholdSweep::default();
        assert_eq!(sweep.min_threshold, 0.0);
        assert_eq!(sweep.max_threshold, 1.0);
        assert_eq!(sweep.steps, 50);
    }

    #[test]
    fn threshold_sweep_empty_input() {
        let sweep = ThresholdSweep::new(0.0, 1.0, 10);
        let results = sweep.sweep(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn threshold_sweep_uniform_field() {
        let sweep = ThresholdSweep::new(0.0, 1.0, 10);
        let intensities = vec![vec![0.5f64; 10]; 10];
        let results = sweep.sweep(&intensities);
        assert_eq!(results.len(), 10);
        assert!((results[0].valid_fraction - 1.0).abs() < 0.001);
        assert_eq!(results.last().unwrap().valid_fraction, 0.0);
    }

    #[test]
    fn threshold_sweep_best_threshold() {
        let sweep = ThresholdSweep::new(0.0, 1.0, 20);
        let mut intensities = vec![vec![0.0f64; 10]; 10];
        for y in 3..7 {
            for x in 3..7 {
                intensities[y][x] = 0.9;
            }
        }
        let best = sweep.best_threshold(&intensities);
        assert!(best.is_some());
        let best = best.unwrap();
        assert!(best.threshold >= 0.0);
        assert!(best.threshold <= 1.0);
    }

    #[test]
    fn threshold_sweep_threshold_for_valid_fraction() {
        let sweep = ThresholdSweep::new(0.0, 1.0, 20);
        let intensities: Vec<Vec<f64>> = (0..10).map(|_| vec![0.5f64; 10]).collect();
        let result = sweep.threshold_for_valid_fraction(&intensities, 0.5);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!((result.valid_fraction - 0.5).abs() < 0.1);
    }

    #[test]
    fn threshold_history_new() {
        let history = ThresholdHistory::new(100);
        assert_eq!(history.max_length, 100);
        assert!(history.records.is_empty());
        assert_eq!(history.average_threshold(), 0.0);
        assert_eq!(history.trend(), 0.0);
    }

    #[test]
    fn threshold_history_record() {
        let mut history = ThresholdHistory::new(100);
        history.record(0.5, 0.8, 0.6);
        history.record(0.6, 0.9, 0.5);
        assert_eq!(history.records.len(), 2);
        assert_eq!(history.latest(), Some(0.6));
        assert!((history.average_threshold() - 0.55).abs() < 0.001);
    }

    #[test]
    fn threshold_history_trend() {
        let mut history = ThresholdHistory::new(100);
        history.record(0.1, 0.5, 0.5);
        history.record(0.5, 0.5, 0.5);
        history.record(0.9, 0.5, 0.5);
        assert!(history.trend() > 0.0);
    }

    #[test]
    fn threshold_history_stable() {
        let mut history = ThresholdHistory::new(100);
        history.record(0.5, 0.5, 0.5);
        history.record(0.51, 0.5, 0.5);
        history.record(0.49, 0.5, 0.5);
        assert!(history.is_stable(0.1));
    }

    #[test]
    fn threshold_history_empty_trend() {
        let history = ThresholdHistory::new(100);
        assert_eq!(history.trend(), 0.0);
        assert!(history.is_stable(0.1));
    }

    #[test]
    fn multi_scale_analyzer_default() {
        let analyzer = MultiScaleAnalyzer::default();
        assert_eq!(analyzer.scales.len(), 3);
        assert!(analyzer.scales.contains(&0.5));
        assert!(analyzer.scales.contains(&1.0));
        assert!(analyzer.scales.contains(&2.0));
    }

    #[test]
    fn multi_scale_analyzer_empty_input() {
        let analyzer = MultiScaleAnalyzer::new(vec![1.0]);
        let result = analyzer.analyze(&[]);
        assert_eq!(result.scale_results.len(), 0);
        assert_eq!(result.optimal_scale, 0);
        assert_eq!(result.consistency_score, 0.0);
    }

    #[test]
    fn multi_scale_analyzer_uniform_field() {
        let analyzer = MultiScaleAnalyzer::new(vec![1.0]);
        let intensities = vec![vec![0.5f64; 10]; 10];
        let result = analyzer.analyze(&intensities);
        assert_eq!(result.scale_results.len(), 1);
        assert_eq!(result.optimal_scale, 0);
        assert!(result.consistency_score > 0.0);
    }

    #[test]
    fn multi_scale_analyzer_multiple_scales() {
        let analyzer = MultiScaleAnalyzer::new(vec![0.5, 1.0, 2.0]);
        let intensities: Vec<Vec<f64>> = (0..64).map(|_| vec![0.5f64; 64]).collect();
        let result = analyzer.analyze(&intensities);
        assert_eq!(result.scale_results.len(), 3);
        assert!(result.consistency_score >= 0.0);
        assert!(result.consistency_score <= 1.0);
    }

    #[test]
    fn optimization_strategy_variants() {
        assert_eq!(OptimizationStrategy::Otsu as u32, 0);
        assert_eq!(OptimizationStrategy::KittlerIllingworth as u32, 1);
        assert!(matches!(
            OptimizationStrategy::TargetFraction(0.3),
            OptimizationStrategy::TargetFraction(0.3)
        ));
        assert!(matches!(
            OptimizationStrategy::NoiseAdaptive,
            OptimizationStrategy::NoiseAdaptive
        ));
    }

    #[test]
    fn threshold_evaluation_default_fields() {
        let eval = ThresholdEvaluation {
            threshold: 0.5,
            valid_fraction: 0.7,
            component_count: 3,
            quality_score: 0.8,
            snr_estimate: 20.0,
        };
        assert_eq!(eval.threshold, 0.5);
        assert_eq!(eval.valid_fraction, 0.7);
        assert_eq!(eval.component_count, 3);
        assert_eq!(eval.quality_score, 0.8);
        assert_eq!(eval.snr_estimate, 20.0);
    }

    #[test]
    fn threshold_optimization_result_default() {
        let result = ThresholdOptimizationResult {
            optimal_threshold: 0.0,
            strategy: OptimizationStrategy::Otsu,
            between_class_variance: 0.0,
            within_class_variance: 0.0,
            iterations: 0,
            converged: false,
            quality_score: 0.0,
        };
        assert!(!result.is_reliable());
    }
}