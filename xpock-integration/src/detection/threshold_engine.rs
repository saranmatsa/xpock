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

// ---------------------------------------------------------------------------
// Kittler-Illingworth minimum error thresholding
// ---------------------------------------------------------------------------

/// Result of Kittler-Illingworth thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct KittlerIllingworthResult {
    /// Optimal threshold.
    pub threshold: f64,
    /// Minimum error probability.
    pub min_error: f64,
    /// Number of iterations.
    pub iterations: u32,
    /// Whether converged.
    pub converged: bool,
    /// Class means.
    pub class_means: (f64, f64),
    /// Class standard deviations.
    pub class_stds: (f64, f64),
}

/// Implements Kittler-Illingworth minimum error thresholding.
/// Assumes bimodal Gaussian distribution and finds threshold minimizing
/// the total probability of misclassification.
pub struct KittlerIllingworth {
    /// Maximum iterations for iterative refinement.
    pub max_iterations: u32,
    /// Convergence tolerance for threshold change.
    pub tolerance: f64,
    /// Number of histogram bins.
    pub num_bins: usize,
}

impl KittlerIllingworth {
    /// Create a new Kittler-Illingworth thresholder.
    pub fn new() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-4,
            num_bins: 256,
        }
    }

    /// Compute the Kittler-Illingworth threshold for an intensity field.
    pub fn compute(&self, intensities: &[Vec<f64>]) -> KittlerIllingworthResult {
        if intensities.is_empty() || intensities[0].is_empty() {
            return KittlerIllingworthResult {
                threshold: 0.0,
                min_error: 1.0,
                iterations: 0,
                converged: false,
                class_means: (0.0, 0.0),
                class_stds: (0.0, 0.0),
            };
        }

        let height = intensities.len();
        let width = intensities[0].len();
        let total = width * height;

        // Find min and max intensity.
        let min_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_intensity <= min_intensity {
            return KittlerIllingworthResult {
                threshold: min_intensity,
                min_error: 0.0,
                iterations: 1,
                converged: true,
                class_means: (min_intensity, min_intensity),
                class_stds: (0.0, 0.0),
            };
        }

        // Build histogram.
        let bin_width = (max_intensity - min_intensity) / self.num_bins as f64;
        let mut histogram = vec![0usize; self.num_bins];
        for row in intensities {
            for &val in row {
                let bin = (((val - min_intensity) / bin_width).floor() as usize).min(self.num_bins - 1);
                histogram[bin] += 1;
            }
        }

        // Compute cumulative sums for efficient statistics.
        let mut cum_count: Vec<f64> = vec![0.0; self.num_bins + 1];
        let mut cum_sum: Vec<f64> = vec![0.0; self.num_bins + 1];
        let mut cum_sq_sum: Vec<f64> = vec![0.0; self.num_bins + 1];

        for i in 0..self.num_bins {
            let center = min_intensity + (i as f64 + 0.5) * bin_width;
            let count = histogram[i] as f64;
            cum_count[i + 1] = cum_count[i] + count;
            cum_sum[i + 1] = cum_sum[i] + count * center;
            cum_sq_sum[i + 1] = cum_sq_sum[i] + count * center * center;
        }

        // Iterative Kittler-Illingworth: start with Otsu-like initial guess.
        let mut threshold = min_intensity + (max_intensity - min_intensity) * 0.5;
        let mut converged = false;
        let mut iterations = 0u32;

        for _ in 0..self.max_iterations {
            iterations += 1;
            let t_bin = (((threshold - min_intensity) / bin_width).round() as usize).min(self.num_bins - 1);
            let t_bin = t_bin.max(1).min(self.num_bins - 1);

            // Class 0 (background): bins [0, t_bin)
            let w0 = cum_count[t_bin];
            let w1 = cum_count[self.num_bins] - cum_count[t_bin];

            if w0 < 1.0 || w1 < 1.0 {
                break;
            }

            let mean0 = cum_sum[t_bin] / w0;
            let mean1 = (cum_sum[self.num_bins] - cum_sum[t_bin]) / w1;

            let var0 = if w0 > 1.0 {
                (cum_sq_sum[t_bin] - cum_sum[t_bin] * mean0) / w0
            } else {
                0.0
            };
            let var1 = if w1 > 1.0 {
                (cum_sq_sum[self.num_bins] - cum_sum[self.num_bins]) / w1 - mean1 * mean1
                    + (cum_sq_sum[t_bin] - cum_sum[t_bin] * mean0) / w0
            } else {
                0.0
            };

            let std0 = var0.max(0.0).sqrt();
            let std1 = var1.max(0.0).sqrt();

            // Kittler-Illingworth new threshold:
            // t = (mean0*std1 + mean1*std0) / (std0 + std1) when priors equal
            let new_threshold = if (std0 + std1) > 1e-10 {
                (mean0 * std1 + mean1 * std0) / (std0 + std1)
            } else {
                (mean0 + mean1) / 2.0
            };

            if (new_threshold - threshold).abs() < self.tolerance {
                threshold = new_threshold;
                converged = true;
                break;
            }
            threshold = new_threshold;
        }

        // Compute final class statistics at converged threshold.
        let t_bin = (((threshold - min_intensity) / bin_width).round() as usize).min(self.num_bins - 1);
        let t_bin = t_bin.max(1).min(self.num_bins - 1);

        let w0 = cum_count[t_bin];
        let w1 = cum_count[self.num_bins] - cum_count[t_bin];
        let mean0 = if w0 > 0.0 { cum_sum[t_bin] / w0 } else { 0.0 };
        let mean1 = if w1 > 0.0 { (cum_sum[self.num_bins] - cum_sum[t_bin]) / w1 } else { 0.0 };
        let var0 = if w0 > 1.0 {
            (cum_sq_sum[t_bin] - cum_sum[t_bin] * mean0) / w0
        } else {
            0.0
        };
        let var1 = if w1 > 1.0 {
            (cum_sq_sum[self.num_bins] - cum_sum[t_bin] * mean1) / w1
        } else {
            0.0
        };
        let std0 = var0.max(0.0).sqrt();
        let std1 = var1.max(0.0).sqrt();

        // Minimum error probability approximation.
        let min_error = if (std0 + std1) > 1e-10 {
            let overlap = 2.0 * std0 * std1 / (std0 + std1).powi(2);
            let separation = (mean0 - mean1).abs() / (std0 + std1);
            (0.5 * overlap * (-separation.powi(2)).exp()).clamp(0.0, 1.0)
        } else {
            0.0
        };

        KittlerIllingworthResult {
            threshold,
            min_error,
            iterations,
            converged,
            class_means: (mean0, mean1),
            class_stds: (std0, std1),
        }
    }

    /// Returns true if the result indicates a clean bimodal separation.
    pub fn is_clean_separation(&self, result: &KittlerIllingworthResult) -> bool {
        result.converged && result.min_error < 0.1 && result.class_stds.0 + result.class_stds.1 > 0.0
    }

    /// Update parameters.
    pub fn update_params(&mut self, max_iterations: u32, tolerance: f64, num_bins: usize) {
        self.max_iterations = max_iterations.max(10);
        self.tolerance = tolerance.max(1e-8);
        self.num_bins = num_bins.max(16);
    }
}

impl Default for KittlerIllingworth {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Entropy-based thresholding (Kapur's method)
// ---------------------------------------------------------------------------

/// Result of entropy-based thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropyThresholdResult {
    /// Optimal threshold.
    pub threshold: f64,
    /// Total entropy at optimum.
    pub total_entropy: f64,
    /// Background entropy.
    pub background_entropy: f64,
    /// Foreground entropy.
    pub foreground_entropy: f64,
    /// Number of iterations.
    pub iterations: u32,
}

/// Implements Kapur's maximum entropy thresholding.
/// Selects threshold that maximizes sum of background and foreground entropy.
pub struct EntropyThreshold {
    /// Number of histogram bins.
    pub num_bins: usize,
}

impl EntropyThreshold {
    /// Create a new entropy thresholder.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Compute entropy-based threshold for an intensity field.
    pub fn compute(&self, intensities: &[Vec<f64>]) -> EntropyThresholdResult {
        if intensities.is_empty() || intensities[0].is_empty() {
            return EntropyThresholdResult {
                threshold: 0.0,
                total_entropy: 0.0,
                background_entropy: 0.0,
                foreground_entropy: 0.0,
                iterations: 0,
            };
        }

        let height = intensities.len();
        let width = intensities[0].len();
        let total = width * height;

        let max_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(0.0f64, f64::max);

        if max_intensity <= 0.0 {
            return EntropyThresholdResult {
                threshold: 0.0,
                total_entropy: 0.0,
                background_entropy: 0.0,
                foreground_entropy: 0.0,
                iterations: 1,
            };
        }

        // Build histogram.
        let mut histogram = vec![0usize; self.num_bins];
        let bin_width = max_intensity / self.num_bins as f64;

        for row in intensities {
            for &val in row {
                let bin = ((val / max_intensity) * self.num_bins as f64).floor() as usize;
                let bin = bin.min(self.num_bins - 1);
                histogram[bin] += 1;
            }
        }

        // Compute cumulative probabilities.
        let mut cum_prob: Vec<f64> = vec![0.0; self.num_bins + 1];
        for i in 0..self.num_bins {
            cum_prob[i + 1] = cum_prob[i] + histogram[i] as f64 / total as f64;
        }

        // Find threshold maximizing total entropy.
        let mut best_threshold = 0.0;
        let mut max_total_entropy = f64::NEG_INFINITY;
        let mut best_bg_entropy = 0.0;
        let mut best_fg_entropy = 0.0;
        let mut iterations = 0u32;

        for t in 0..self.num_bins.saturating_sub(1) {
            iterations += 1;

            let p0 = cum_prob[t + 1];
            let p1 = 1.0 - p0;

            if p0 < 1e-10 || p1 < 1e-10 {
                continue;
            }

            // Background entropy.
            let mut h0 = 0.0f64;
            for i in 0..=t {
                let p = histogram[i] as f64 / total as f64;
                if p > 0.0 {
                    h0 -= (p / p0) * (p / p0).ln();
                }
            }

            // Foreground entropy.
            let mut h1 = 0.0f64;
            for i in (t + 1)..self.num_bins {
                let p = histogram[i] as f64 / total as f64;
                if p > 0.0 {
                    h1 -= (p / p1) * (p / p1).ln();
                }
            }

            let total_entropy = h0 + h1;
            if total_entropy > max_total_entropy {
                max_total_entropy = total_entropy;
                best_threshold = min_intensity_from_bins(t, self.num_bins, max_intensity);
                best_bg_entropy = h0;
                best_fg_entropy = h1;
            }
        }

        EntropyThresholdResult {
            threshold: best_threshold,
            total_entropy: max_total_entropy,
            background_entropy: best_bg_entropy,
            foreground_entropy: best_fg_entropy,
            iterations,
        }
    }

    /// Update parameters.
    pub fn update_params(&mut self, num_bins: usize) {
        self.num_bins = num_bins.max(16);
    }
}

fn min_intensity_from_bins(bin: usize, num_bins: usize, max_intensity: f64) -> f64 {
    (bin as f64 + 0.5) / num_bins as f64 * max_intensity
}

impl Default for EntropyThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Moment-preserving thresholding
// ---------------------------------------------------------------------------

/// Result of moment-preserving thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentPreservingResult {
    /// Optimal threshold.
    pub threshold: f64,
    /// Zeroth moment preserved.
    pub m0: f64,
    /// First moment preserved.
    pub m1: f64,
    /// Second moment preserved.
    pub m2: f64,
    /// Third moment preserved.
    pub m3: f64,
    /// Moment preservation error.
    pub preservation_error: f64,
}

/// Implements moment-preserving thresholding (Tsai's method).
/// Preserves the first three moments of the histogram in the binary image.
pub struct MomentPreserving {
    /// Number of histogram bins.
    pub num_bins: usize,
}

impl MomentPreserving {
    /// Create a new moment-preserving thresholder.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Compute moment-preserving threshold for an intensity field.
    pub fn compute(&self, intensities: &[Vec<f64>]) -> MomentPreservingResult {
        if intensities.is_empty() || intensities[0].is_empty() {
            return MomentPreservingResult {
                threshold: 0.0,
                m0: 0.0,
                m1: 0.0,
                m2: 0.0,
                m3: 0.0,
                preservation_error: 1.0,
            };
        }

        let height = intensities.len();
        let width = intensities[0].len();
        let total = width * height as f64;

        let min_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_intensity <= min_intensity {
            return MomentPreservingResult {
                threshold: min_intensity,
                m0: 1.0,
                m1: min_intensity,
                m2: min_intensity * min_intensity,
                m3: min_intensity.powi(3),
                preservation_error: 0.0,
            };
        }

        // Normalize intensities to [0, 1] for moment computation.
        let range = max_intensity - min_intensity;
        let mut values: Vec<f64> = intensities
            .iter()
            .flatten()
            .map(|v| (v - min_intensity) / range)
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute global moments of normalized histogram.
        let m0_global = 1.0;
        let m1_global = values.iter().sum::<f64>() / total;
        let m2_global = values.iter().map(|v| v * v).sum::<f64>() / total;
        let m3_global = values.iter().map(|v| v.powi(3)).sum::<f64>() / total;

        // Search for threshold preserving moments.
        let mut best_threshold = 0.5;
        let mut best_error = f64::INFINITY;

        for i in 1..values.len() {
            let t = (values[i - 1] + values[i]) / 2.0;

            let n0 = i as f64;
            let n1 = total - n0;

            if n0 < 1.0 || n1 < 1.0 {
                continue;
            }

            let m0_bg = n0 / total;
            let m0_fg = n1 / total;
            let m1_bg = values.iter().take(i).sum::<f64>() / total;
            let m1_fg = (values.iter().skip(i).sum::<f64>()) / total;
            let m2_bg = values.iter().take(i).map(|v| v * v).sum::<f64>() / total;
            let m2_fg = values.iter().skip(i).map(|v| v * v).sum::<f64>() / total;
            let m3_bg = values.iter().take(i).map(|v| v.powi(3)).sum::<f64>() / total;
            let m3_fg = values.iter().skip(i).map(|v| v.powi(3)).sum::<f64>() / total;

            // Preserved moments for two-level approximation.
            let m0_preserved = m0_bg + m0_fg;
            let m1_preserved = m1_bg + m1_fg;
            let m2_preserved = m2_bg + m2_fg;
            let m3_preserved = m3_bg + m3_fg;

            let error = (m0_preserved - m0_global).abs()
                + (m1_preserved - m1_global).abs()
                + (m2_preserved - m2_global).abs()
                + (m3_preserved - m3_global).abs();

            if error < best_error {
                best_error = error;
                best_threshold = min_intensity + t * range;
            }
        }

        MomentPreservingResult {
            threshold: best_threshold,
            m0: m0_global,
            m1: m1_global,
            m2: m2_global,
            m3: m3_global,
            preservation_error: best_error,
        }
    }

    /// Update parameters.
    pub fn update_params(&mut self, num_bins: usize) {
        self.num_bins = num_bins.max(16);
    }
}

impl Default for MomentPreserving {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fuzzy thresholding
// ---------------------------------------------------------------------------

/// Result of fuzzy thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyThresholdResult {
    /// Optimal threshold.
    pub threshold: f64,
    /// Fuzzy entropy measure.
    pub fuzzy_entropy: f64,
    /// Ambiguity measure at threshold.
    pub ambiguity: f64,
    /// Number of iterations.
    pub iterations: u32,
}

/// Implements fuzzy thresholding using fuzzy entropy minimization.
/// Minimizes the ambiguity in classification at the threshold.
pub struct FuzzyThreshold {
    /// Number of histogram bins.
    pub num_bins: usize,
    /// Fuzziness parameter (lambda in fuzzy entropy).
    pub lambda: f64,
}

impl FuzzyThreshold {
    /// Create a new fuzzy thresholder.
    pub fn new() -> Self {
        Self {
            num_bins: 256,
            lambda: 1.0,
        }
    }

    /// Compute fuzzy threshold for an intensity field.
    pub fn compute(&self, intensities: &[Vec<f64>]) -> FuzzyThresholdResult {
        if intensities.is_empty() || intensities[0].is_empty() {
            return FuzzyThresholdResult {
                threshold: 0.0,
                fuzzy_entropy: 0.0,
                ambiguity: 1.0,
                iterations: 0,
            };
        }

        let max_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(0.0f64, f64::max);

        if max_intensity <= 0.0 {
            return FuzzyThresholdResult {
                threshold: 0.0,
                fuzzy_entropy: 0.0,
                ambiguity: 1.0,
                iterations: 1,
            };
        }

        // Build histogram.
        let mut histogram = vec![0usize; self.num_bins];
        let bin_width = max_intensity / self.num_bins as f64;

        for row in intensities {
            for &val in row {
                let bin = ((val / max_intensity) * self.num_bins as f64).floor() as usize;
                let bin = bin.min(self.num_bins - 1);
                histogram[bin] += 1;
            }
        }

        let total = intensities.iter().map(|r| r.len()).sum::<usize>() as f64;

        // Find threshold minimizing fuzzy entropy (maximizing clarity).
        let mut best_threshold = 0.0;
        let mut min_entropy = f64::INFINITY;
        let mut min_ambiguity = 1.0;
        let mut iterations = 0u32;

        for t in 1..self.num_bins {
            iterations += 1;

            let mut entropy = 0.0f64;
            let mut ambiguity = 0.0f64;

            for i in 0..self.num_bins {
                let p = histogram[i] as f64 / total;
                if p == 0.0 {
                    continue;
                }

                // Membership function: sigmoid centered at threshold t.
                let x = i as f64 / self.num_bins as f64;
                let t_norm = t as f64 / self.num_bins as f64;
                let mu = 1.0 / (1.0 + (-self.lambda * (x - t_norm)).exp());

                // Fuzzy entropy term.
                if mu > 0.0 && mu < 1.0 {
                    entropy -= p * (mu * mu.ln() + (1.0 - mu) * (1.0 - mu).ln());
                }

                // Ambiguity: how close mu is to 0.5.
                ambiguity += p * (1.0 - (2.0 * mu - 1.0).abs());
            }

            if entropy < min_entropy {
                min_entropy = entropy;
                min_ambiguity = ambiguity;
                best_threshold = (t as f64 + 0.5) / self.num_bins as f64 * max_intensity;
            }
        }

        FuzzyThresholdResult {
            threshold: best_threshold,
            fuzzy_entropy: min_entropy,
            ambiguity: min_ambiguity,
            iterations,
        }
    }

    /// Update parameters.
    pub fn update_params(&mut self, num_bins: usize, lambda: f64) {
        self.num_bins = num_bins.max(16);
        self.lambda = lambda.max(0.1);
    }
}

impl Default for FuzzyThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Local / adaptive thresholding
// ---------------------------------------------------------------------------

/// Result of local thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalThresholdResult {
    /// Per-pixel threshold values.
    pub threshold_grid: Vec<Vec<f64>>,
    /// Local window size used.
    pub window_size: usize,
    /// Mean local threshold.
    pub mean_threshold: f64,
    /// Standard deviation of local thresholds.
    pub threshold_std: f64,
}

/// Implements local (adaptive) thresholding using local statistics.
/// Each pixel gets a threshold based on its neighborhood mean and std.
pub struct LocalThreshold {
    /// Window size for local statistics (must be odd).
    pub window_size: usize,
    /// Constant subtracted from local mean (C in mean - C).
    pub c: f64,
    /// Multiplier for local standard deviation.
    pub std_multiplier: f64,
}

impl LocalThreshold {
    /// Create a new local thresholder.
    pub fn new() -> Self {
        Self {
            window_size: 15,
            c: 0.0,
            std_multiplier: 0.0,
        }
    }

    /// Compute local thresholds for an intensity field.
    pub fn compute(&self, intensities: &[Vec<f64>]) -> LocalThresholdResult {
        let height = intensities.len();
        if height == 0 || intensities[0].is_empty() {
            return LocalThresholdResult {
                threshold_grid: Vec::new(),
                window_size: self.window_size,
                mean_threshold: 0.0,
                threshold_std: 0.0,
            };
        }

        let width = intensities[0].len();
        let half = self.window_size / 2;

        let mut threshold_grid = vec![vec![0.0f64; width]; height];
        let mut all_thresholds: Vec<f64> = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                let mut count = 0usize;

                let y_start = y.saturating_sub(half);
                let y_end = (y + half + 1).min(height);
                let x_start = x.saturating_sub(half);
                let x_end = (x + half + 1).min(width);

                for yy in y_start..y_end {
                    for xx in x_start..x_end {
                        let v = intensities[yy][xx];
                        sum += v;
                        sum_sq += v * v;
                        count += 1;
                    }
                }

                if count > 0 {
                    let mean = sum / count as f64;
                    let variance = (sum_sq - sum * mean) / count as f64;
                    let std = variance.max(0.0).sqrt();
                    let local_threshold = mean - self.c + self.std_multiplier * std;
                    threshold_grid[y][x] = local_threshold;
                    all_thresholds.push(local_threshold);
                }
            }
        }

        let mean_threshold = if all_thresholds.is_empty() {
            0.0
        } else {
            all_thresholds.iter().sum::<f64>() / all_thresholds.len() as f64
        };

        let threshold_std = if all_thresholds.len() > 1 {
            let mean = mean_threshold;
            let var: f64 = all_thresholds.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
                / all_thresholds.len() as f64;
            var.sqrt()
        } else {
            0.0
        };

        LocalThresholdResult {
            threshold_grid,
            window_size: self.window_size,
            mean_threshold,
            threshold_std,
        }
    }

    /// Apply local threshold to produce binary mask.
    pub fn apply(&self, intensities: &[Vec<f64>]) -> Vec<Vec<bool>> {
        let result = self.compute(intensities);
        let height = intensities.len();
        if height == 0 {
            return Vec::new();
        }
        let width = intensities[0].len();
        let mut mask = vec![vec![false; width]; height];

        for y in 0..height {
            for x in 0..width {
                mask[y][x] = intensities[y][x] >= result.threshold_grid[y][x];
            }
        }

        mask
    }

    /// Update parameters.
    pub fn update_params(&mut self, window_size: usize, c: f64, std_multiplier: f64) {
        self.window_size = window_size.max(3);
        self.c = c;
        self.std_multiplier = std_multiplier;
    }
}

impl Default for LocalThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hysteresis thresholding
// ---------------------------------------------------------------------------

/// Result of hysteresis thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct HysteresisResult {
    /// Binary mask after hysteresis.
    pub mask: Vec<Vec<bool>>,
    /// Low threshold used.
    pub low_threshold: f64,
    /// High threshold used.
    pub high_threshold: f64,
    /// Number of strong pixels.
    pub strong_count: usize,
    /// Number of weak pixels connected to strong.
    pub connected_weak_count: usize,
}

/// Implements hysteresis thresholding (Canny-style).
/// Strong pixels (above high threshold) are kept. Weak pixels (between low
/// and high) are kept only if connected to a strong pixel.
pub struct HysteresisThreshold {
    /// Low threshold.
    pub low_threshold: f64,
    /// High threshold.
    pub high_threshold: f64,
    /// Connectivity (4 or 8).
    pub connectivity: usize,
}

impl HysteresisThreshold {
    /// Create a new hysteresis thresholder.
    pub fn new(low: f64, high: f64) -> Self {
        Self {
            low_threshold: low,
            high_threshold: high,
            connectivity: 8,
        }
    }

    /// Apply hysteresis thresholding to an intensity field.
    pub fn apply(&self, intensities: &[Vec<f64>]) -> HysteresisResult {
        let height = intensities.len();
        if height == 0 || intensities[0].is_empty() {
            return HysteresisResult {
                mask: Vec::new(),
                low_threshold: self.low_threshold,
                high_threshold: self.high_threshold,
                strong_count: 0,
                connected_weak_count: 0,
            };
        }

        let width = intensities[0].len();
        let mut mask = vec![vec![false; width]; height];
        let mut visited = vec![vec![false; width]; height];

        let mut strong_count = 0usize;
        let mut connected_weak_count = 0usize;

        // First pass: mark strong pixels.
        for y in 0..height {
            for x in 0..width {
                if intensities[y][x] >= self.high_threshold {
                    mask[y][x] = true;
                    strong_count += 1;
                }
            }
        }

        // Second pass: flood fill from strong pixels through weak pixels.
        let neighbors: Vec<(isize, isize)> = if self.connectivity >= 8 {
            vec![
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        } else {
            vec![(-1, 0), (0, -1), (0, 1), (1, 0)]
        };

        for y in 0..height {
            for x in 0..width {
                if mask[y][x] && !visited[y][x] {
                    self.flood_fill(intensities, &mut mask, &mut visited, x, y, &neighbors);
                }
            }
        }

        // Count connected weak pixels.
        for y in 0..height {
            for x in 0..width {
                if mask[y][x] && intensities[y][x] < self.high_threshold {
                    connected_weak_count += 1;
                }
            }
        }

        HysteresisResult {
            mask,
            low_threshold: self.low_threshold,
            high_threshold: self.high_threshold,
            strong_count,
            connected_weak_count,
        }
    }

    /// Flood fill from a strong pixel through connected weak pixels.
    fn flood_fill(
        &self,
        intensities: &[Vec<f64>],
        mask: &mut Vec<Vec<bool>>,
        visited: &mut Vec<Vec<bool>>,
        start_x: usize,
        start_y: usize,
        neighbors: &[(isize, isize)],
    ) {
        let height = intensities.len();
        let width = intensities[0].len();
        let mut stack = vec![(start_x, start_y)];

        while let Some((x, y)) = stack.pop() {
            if visited[y][x] {
                continue;
            }
            visited[y][x] = true;

            for &(dx, dy) in neighbors {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if !visited[ny][nx]
                        && intensities[ny][nx] >= self.low_threshold
                        && intensities[ny][nx] < self.high_threshold
                    {
                        mask[ny][nx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }

    /// Update thresholds.
    pub fn update_thresholds(&mut self, low: f64, high: f64) {
        self.low_threshold = low;
        self.high_threshold = high.max(low);
    }

    /// Update connectivity.
    pub fn set_connectivity(&mut self, connectivity: usize) {
        self.connectivity = connectivity.min(8);
    }
}

impl Default for HysteresisThreshold {
    fn default() -> Self {
        Self::new(0.2, 0.5)
    }
}

// ---------------------------------------------------------------------------
// Threshold quality assessment
// ---------------------------------------------------------------------------

/// Quality metrics for a threshold value.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdQuality {
    /// Separation measure (between-class variance / total variance).
    pub separation: f64,
    /// Compactness measure (how compact each class is).
    pub compactness: f64,
    /// Stability measure (sensitivity to small threshold changes).
    pub stability: f64,
    /// Overall quality score (0.0 to 1.0).
    pub overall: f64,
}

/// Assesses the quality of a threshold value.
pub struct ThresholdQualityAssessor {
    /// Number of perturbation trials for stability.
    pub stability_trials: usize,
    /// Perturbation magnitude for stability analysis.
    pub stability_perturbation: f64,
}

impl ThresholdQualityAssessor {
    /// Create a new quality assessor.
    pub fn new() -> Self {
        Self {
            stability_trials: 20,
            stability_perturbation: 0.05,
        }
    }

    /// Assess threshold quality for an intensity field.
    pub fn assess(&self, intensities: &[Vec<f64>], threshold: f64) -> ThresholdQuality {
        if intensities.is_empty() || intensities[0].is_empty() {
            return ThresholdQuality {
                separation: 0.0,
                compactness: 0.0,
                stability: 0.0,
                overall: 0.0,
            };
        }

        let height = intensities.len();
        let width = intensities[0].len();
        let total = width * height;

        // Separation: between-class variance.
        let mean_all: f64 = intensities.iter().flatten().cloned().sum::<f64>() / total as f64;

        let (mut w0, mut sum0, mut sum_sq0) = (0usize, 0.0f64, 0.0f64);
        let (mut w1, mut sum1, mut sum_sq1) = (0usize, 0.0f64, 0.0f64);

        for row in intensities {
            for &val in row {
                if val >= threshold {
                    w1 += 1;
                    sum1 += val;
                    sum_sq1 += val * val;
                } else {
                    w0 += 1;
                    sum0 += val;
                    sum_sq0 += val * val;
                }
            }
        }

        let separation = if w0 > 0 && w1 > 0 {
            let mean0 = sum0 / w0 as f64;
            let mean1 = sum1 / w1 as f64;
            let between = w0 as f64 * w1 as f64 * (mean0 - mean1).powi(2) / (total as f64 * total as f64);
            let total_var = intensities.iter().flatten().map(|v| (v - mean_all).powi(2)).sum::<f64>() / total as f64;
            (between / total_var.max(1e-10)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Compactness: inverse of within-class variance.
        let var0 = if w0 > 1 {
            (sum_sq0 - sum0 * sum0 / w0 as f64) / (w0 - 1) as f64
        } else {
            0.0
        };
        let var1 = if w1 > 1 {
            (sum_sq1 - sum1 * sum1 / w1 as f64) / (w1 - 1) as f64
        } else {
            0.0
        };
        let compactness = (1.0 / (1.0 + var0 + var1)).clamp(0.0, 1.0);

        // Stability: how much does classification change with small threshold perturbations.
        let mut stable_count = 0usize;
        for _ in 0..self.stability_trials {
            let perturb = (self.rand_stable() - 0.5) * 2.0 * self.stability_perturbation;
            let perturbed = threshold + perturb;
            let mut changes = 0usize;
            for row in intensities {
                for &val in row {
                    let orig = val >= threshold;
                    let new = val >= perturbed;
                    if orig != new {
                        changes += 1;
                    }
                }
            }
            if (changes as f64) / (total as f64) < 0.1 {
                stable_count += 1;
            }
        }
        let stability = stable_count as f64 / self.stability_trials as f64;

        let overall = (separation * 0.4 + compactness * 0.3 + stability * 0.3).clamp(0.0, 1.0);

        ThresholdQuality {
            separation,
            compactness,
            stability,
            overall,
        }
    }

    /// Deterministic pseudo-random for stability analysis.
    fn rand_stable(&self) -> f64 {
        static mut SEED2: u64 = 987654321;
        unsafe {
            SEED2 = SEED2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (SEED2 >> 11) as f64 / u64::MAX as f64
        }
    }

    /// Update parameters.
    pub fn update_params(&mut self, stability_trials: usize, stability_perturbation: f64) {
        self.stability_trials = stability_trials.max(5);
        self.stability_perturbation = stability_perturbation.clamp(0.0, 1.0);
    }
}

impl Default for ThresholdQualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-level (Otsu multi-threshold) thresholding
// ---------------------------------------------------------------------------

/// Result of multi-level thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiLevelResult {
    /// Thresholds in ascending order.
    pub thresholds: Vec<f64>,
    /// Between-class variance at optimum.
    pub between_class_variance: f64,
    /// Number of levels (thresholds + 1).
    pub num_levels: usize,
}

/// Implements multi-level Otsu thresholding.
/// Finds multiple thresholds that divide the histogram into K classes.
pub struct MultiLevelThreshold {
    /// Number of thresholds to find.
    pub num_thresholds: usize,
    /// Number of histogram bins.
    pub num_bins: usize,
}

impl MultiLevelThreshold {
    /// Create a new multi-level thresholder.
    pub fn new(num_thresholds: usize) -> Self {
        Self {
            num_thresholds,
            num_bins: 256,
        }
    }

    /// Compute multi-level thresholds for an intensity field.
    pub fn compute(&self, intensities: &[Vec<f64>]) -> MultiLevelResult {
        if intensities.is_empty() || intensities[0].is_empty() || self.num_thresholds == 0 {
            return MultiLevelResult {
                thresholds: Vec::new(),
                between_class_variance: 0.0,
                num_levels: 1,
            };
        }

        let max_intensity = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(0.0f64, f64::max);

        if max_intensity <= 0.0 {
            return MultiLevelResult {
                thresholds: vec![0.0; self.num_thresholds],
                between_class_variance: 0.0,
                num_levels: self.num_thresholds + 1,
            };
        }

        // Build histogram.
        let mut histogram = vec![0usize; self.num_bins];
        let bin_width = max_intensity / self.num_bins as f64;

        for row in intensities {
            for &val in row {
                let bin = ((val / max_intensity) * self.num_bins as f64).floor() as usize;
                let bin = bin.min(self.num_bins - 1);
                histogram[bin] += 1;
            }
        }

        let total = intensities.iter().map(|r| r.len()).sum::<usize>() as f64;

        // Compute cumulative sum and cumulative mean.
        let mut cum_count: Vec<f64> = vec![0.0; self.num_bins + 1];
        let mut cum_sum: Vec<f64> = vec![0.0; self.num_bins + 1];

        for i in 0..self.num_bins {
            let center = (i as f64 + 0.5) * bin_width;
            let count = histogram[i] as f64;
            cum_count[i + 1] = cum_count[i] + count;
            cum_sum[i + 1] = cum_sum[i] + count * center;
        }

        let total_sum = cum_sum[self.num_bins];

        // Exhaustive search for 2 thresholds, greedy for more.
        let thresholds = if self.num_thresholds == 1 {
            self.find_single_threshold(&histogram, &cum_count, &cum_sum, total, total_sum, max_intensity)
        } else if self.num_thresholds == 2 {
            self.find_two_thresholds(&histogram, &cum_count, &cum_sum, total, total_sum, max_intensity)
        } else {
            self.find_multi_thresholds(&histogram, &cum_count, &cum_sum, total, total_sum, max_intensity)
        };

        let between_class_variance = self.compute_bcv(
            &thresholds,
            &histogram,
            &cum_count,
            &cum_sum,
            total,
            total_sum,
        );

        MultiLevelResult {
            thresholds,
            between_class_variance,
            num_levels: self.num_thresholds + 1,
        }
    }

    fn find_single_threshold(
        &self,
        histogram: &[usize],
        cum_count: &[f64],
        cum_sum: &[f64],
        total: f64,
        total_sum: f64,
        max_intensity: f64,
    ) -> Vec<f64> {
        let mut best_t = 0;
        let mut max_bcv = f64::NEG_INFINITY;

        for t in 1..self.num_bins {
            let w0 = cum_count[t];
            let w1 = total - w0;
            if w0 < 1.0 || w1 < 1.0 {
                continue;
            }
            let mean0 = cum_sum[t] / w0;
            let mean1 = (total_sum - cum_sum[t]) / w1;
            let bcv = w0 * w1 * (mean0 - mean1).powi(2) / (total * total);
            if bcv > max_bcv {
                max_bcv = bcv;
                best_t = t;
            }
        }

        vec![(best_t as f64 + 0.5) / self.num_bins as f64 * max_intensity]
    }

    fn find_two_thresholds(
        &self,
        histogram: &[usize],
        cum_count: &[f64],
        cum_sum: &[f64],
        total: f64,
        total_sum: f64,
        max_intensity: f64,
    ) -> Vec<f64> {
        let mut best = (0, 0);
        let mut max_bcv = f64::NEG_INFINITY;

        for t1 in 1..self.num_bins {
            for t2 in (t1 + 1)..self.num_bins {
                let w0 = cum_count[t1];
                let w1 = cum_count[t2] - cum_count[t1];
                let w2 = total - cum_count[t2];
                if w0 < 1.0 || w1 < 1.0 || w2 < 1.0 {
                    continue;
                }
                let mean0 = cum_sum[t1] / w0;
                let mean1 = (cum_sum[t2] - cum_sum[t1]) / w1;
                let mean2 = (total_sum - cum_sum[t2]) / w2;
                let mean_all = total_sum / total;
                let bcv = (w0 * (mean0 - mean_all).powi(2)
                    + w1 * (mean1 - mean_all).powi(2)
                    + w2 * (mean2 - mean_all).powi(2))
                    / total;
                if bcv > max_bcv {
                    max_bcv = bcv;
                    best = (t1, t2);
                }
            }
        }

        vec![
            (best.0 as f64 + 0.5) / self.num_bins as f64 * max_intensity,
            (best.1 as f64 + 0.5) / self.num_bins as f64 * max_intensity,
        ]
    }

    fn find_multi_thresholds(
        &self,
        histogram: &[usize],
        cum_count: &[f64],
        cum_sum: &[f64],
        total: f64,
        total_sum: f64,
        max_intensity: f64,
    ) -> Vec<f64> {
        // Greedy approach: add thresholds one at a time.
        let mut thresholds: Vec<usize> = Vec::new();
        let mut remaining = self.num_thresholds;

        while remaining > 0 {
            let mut best_t = 0;
            let mut max_bcv = f64::NEG_INFINITY;

            for t in 1..self.num_bins {
                if thresholds.contains(&t) {
                    continue;
                }
                let trial: Vec<usize> = thresholds.iter().chain(std::iter::once(&t)).cloned().collect();
                let bcv = self.compute_bcv_from_bins(
                    &trial,
                    cum_count,
                    cum_sum,
                    total,
                    total_sum,
                );
                if bcv > max_bcv {
                    max_bcv = bcv;
                    best_t = t;
                }
            }

            thresholds.push(best_t);
            thresholds.sort();
            remaining -= 1;
        }

        thresholds
            .iter()
            .map(|&t| (t as f64 + 0.5) / self.num_bins as f64 * max_intensity)
            .collect()
    }

    fn compute_bcv(
        &self,
        thresholds: &[f64],
        histogram: &[usize],
        cum_count: &[f64],
        cum_sum: &[f64],
        total: f64,
        total_sum: f64,
    ) -> f64 {
        let bins: Vec<usize> = thresholds
            .iter()
            .map(|&t| ((t / (1.0 / self.num_bins as f64)).round() as usize).min(self.num_bins - 1))
            .collect();
        self.compute_bcv_from_bins(&bins, cum_count, cum_sum, total, total_sum)
    }

    fn compute_bcv_from_bins(
        &self,
        bins: &[usize],
        cum_count: &[f64],
        cum_sum: &[f64],
        total: f64,
        total_sum: f64,
    ) -> f64 {
        let mut boundaries: Vec<usize> = bins.to_vec();
        boundaries.push(self.num_bins);

        let mut prev = 0usize;
        let mean_all = total_sum / total;
        let mut bcv = 0.0f64;

        for &bound in &boundaries {
            if bound <= prev {
                continue;
            }
            let w = cum_count[bound] - cum_count[prev];
            if w < 1.0 {
                prev = bound;
                continue;
            }
            let mean = (cum_sum[bound] - cum_sum[prev]) / w;
            bcv += w * (mean - mean_all).powi(2);
            prev = bound;
        }

        bcv / total
    }

    /// Update parameters.
    pub fn update_params(&mut self, num_thresholds: usize, num_bins: usize) {
        self.num_thresholds = num_thresholds;
        self.num_bins = num_bins.max(16);
    }
}

impl Default for MultiLevelThreshold {
    fn default() -> Self {
        Self::new(1)
    }
}

// ---------------------------------------------------------------------------
// Threshold adaptation controller
// ---------------------------------------------------------------------------

/// Adaptive threshold controller that adjusts thresholds based on feedback.
pub struct ThresholdAdaptationController {
    /// Current threshold.
    pub current_threshold: f64,
    /// Target false positive rate.
    pub target_fpr: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Minimum threshold.
    pub min_threshold: f64,
    /// Maximum threshold.
    pub max_threshold: f64,
    /// Adaptation history.
    pub history: Vec<f64>,
    /// Maximum history length.
    pub max_history: usize,
}

impl ThresholdAdaptationController {
    /// Create a new adaptation controller.
    pub fn new(initial_threshold: f64) -> Self {
        Self {
            current_threshold: initial_threshold,
            target_fpr: 0.1,
            learning_rate: 0.1,
            min_threshold: 0.0,
            max_threshold: 1.0,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Adapt threshold based on observed false positive rate.
    pub fn adapt(&mut self, observed_fpr: f64) -> f64 {
        let error = observed_fpr - self.target_fpr;
        let adjustment = self.learning_rate * error;
        self.current_threshold = (self.current_threshold + adjustment)
            .clamp(self.min_threshold, self.max_threshold);

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(self.current_threshold);

        self.current_threshold
    }

    /// Returns the average threshold over history.
    pub fn avg_threshold(&self) -> f64 {
        if self.history.is_empty() {
            return self.current_threshold;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    /// Returns the threshold variance over history.
    pub fn threshold_variance(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_threshold();
        let var: f64 = self.history.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / self.history.len() as f64;
        var
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update parameters.
    pub fn update_params(&mut self, target_fpr: f64, learning_rate: f64, min_threshold: f64, max_threshold: f64) {
        self.target_fpr = target_fpr.clamp(0.0, 1.0);
        self.learning_rate = learning_rate.clamp(0.0, 1.0);
        self.min_threshold = min_threshold;
        self.max_threshold = max_threshold.max(min_threshold);
        self.current_threshold = self.current_threshold.clamp(min_threshold, max_threshold);
    }
}

impl Default for ThresholdAdaptationController {
    fn default() -> Self {
        Self::new(0.5)
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

    // ---------------------------------------------------------------------------
    // KittlerIllingworth tests
    // ---------------------------------------------------------------------------

    #[test]
    fn kittler_illingworth_default() {
        let ki = KittlerIllingworth::default();
        assert_eq!(ki.max_iterations, 50);
        assert!((ki.tolerance - 1e-4).abs() < 1e-12);
        assert_eq!(ki.num_bins, 256);
    }

    #[test]
    fn kittler_illingworth_empty_input() {
        let ki = KittlerIllingworth::new();
        let result = ki.compute(&[]);
        assert_eq!(result.threshold, 0.0);
        assert!(!result.converged);
    }

    #[test]
    fn kittler_illingworth_uniform_field() {
        let ki = KittlerIllingworth::new();
        let intensities = vec![vec![5.0f64; 32]; 32];
        let result = ki.compute(&intensities);
        assert!(result.converged || result.iterations > 0);
        assert!(result.threshold >= 0.0);
    }

    #[test]
    fn kittler_illingworth_bimodal_field() {
        let ki = KittlerIllingworth::new();
        let mut intensities = vec![vec![0.1f64; 32]; 32];
        for y in 16..32 {
            for x in 16..32 {
                intensities[y][x] = 9.0;
            }
        }
        let result = ki.compute(&intensities);
        assert!(result.threshold > 0.1);
        assert!(result.threshold < 9.0);
        assert!(result.class_means.0 >= 0.0);
        assert!(result.class_means.1 > 0.0);
    }

    #[test]
    fn kittler_illingworth_is_clean_separation() {
        let ki = KittlerIllingworth::new();
        let result = KittlerIllingworthResult {
            threshold: 0.5,
            min_error: 0.05,
            iterations: 10,
            converged: true,
            class_means: (0.1, 0.9),
            class_stds: (0.1, 0.1),
        };
        assert!(ki.is_clean_separation(&result));

        let bad = KittlerIllingworthResult {
            threshold: 0.5,
            min_error: 0.5,
            iterations: 10,
            converged: true,
            class_means: (0.1, 0.9),
            class_stds: (0.1, 0.1),
        };
        assert!(!ki.is_clean_separation(&bad));
    }

    #[test]
    fn kittler_illingworth_update_params() {
        let mut ki = KittlerIllingworth::new();
        ki.update_params(200, 1e-6, 512);
        assert_eq!(ki.max_iterations, 200);
        assert!((ki.tolerance - 1e-6).abs() < 1e-12);
        assert_eq!(ki.num_bins, 512);
    }

    // ---------------------------------------------------------------------------
    // EntropyThreshold tests
    // ---------------------------------------------------------------------------

    #[test]
    fn entropy_threshold_default() {
        let et = EntropyThreshold::default();
        assert_eq!(et.num_bins, 256);
    }

    #[test]
    fn entropy_threshold_empty_input() {
        let et = EntropyThreshold::new();
        let result = et.compute(&[]);
        assert_eq!(result.threshold, 0.0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn entropy_threshold_uniform_field() {
        let et = EntropyThreshold::new();
        let intensities = vec![vec![1.0f64; 32]; 32];
        let result = et.compute(&intensities);
        assert!(result.threshold >= 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn entropy_threshold_bimodal_field() {
        let et = EntropyThreshold::new();
        let mut intensities = vec![vec![0.1f64; 32]; 32];
        for y in 16..32 {
            for x in 16..32 {
                intensities[y][x] = 8.0;
            }
        }
        let result = et.compute(&intensities);
        assert!(result.threshold > 0.1);
        assert!(result.threshold < 8.0);
        assert!(result.total_entropy.is_finite());
    }

    #[test]
    fn entropy_threshold_entropy_values() {
        let et = EntropyThreshold::new();
        let intensities = vec![vec![0.5f64; 16]; 16];
        let result = et.compute(&intensities);
        assert!(result.background_entropy >= 0.0);
        assert!(result.foreground_entropy >= 0.0);
    }

    #[test]
    fn entropy_threshold_update_params() {
        let mut et = EntropyThreshold::new();
        et.update_params(128);
        assert_eq!(et.num_bins, 128);
    }

    // ---------------------------------------------------------------------------
    // MomentPreserving tests
    // ---------------------------------------------------------------------------

    #[test]
    fn moment_preserving_default() {
        let mp = MomentPreserving::default();
        assert_eq!(mp.num_bins, 256);
    }

    #[test]
    fn moment_preserving_empty_input() {
        let mp = MomentPreserving::new();
        let result = mp.compute(&[]);
        assert_eq!(result.threshold, 0.0);
        assert_eq!(result.preservation_error, 1.0);
    }

    #[test]
    fn moment_preserving_uniform_field() {
        let mp = MomentPreserving::new();
        let intensities = vec![vec![3.0f64; 32]; 32];
        let result = mp.compute(&intensities);
        assert!((result.threshold - 3.0).abs() < 0.1);
        assert!((result.preservation_error - 0.0).abs() < 0.001);
    }

    #[test]
    fn moment_preserving_moments() {
        let mp = MomentPreserving::new();
        let intensities: Vec<Vec<f64>> = (0..32).map(|_| vec![0.5f64; 32]).collect();
        let result = mp.compute(&intensities);
        assert!((result.m0 - 1.0).abs() < 0.001);
        assert!(result.m1 >= 0.0);
        assert!(result.m2 >= 0.0);
        assert!(result.m3 >= 0.0);
    }

    #[test]
    fn moment_preserving_update_params() {
        let mut mp = MomentPreserving::new();
        mp.update_params(512);
        assert_eq!(mp.num_bins, 512);
    }

    // ---------------------------------------------------------------------------
    // FuzzyThreshold tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fuzzy_threshold_default() {
        let ft = FuzzyThreshold::default();
        assert_eq!(ft.num_bins, 256);
        assert!((ft.lambda - 1.0).abs() < 0.001);
    }

    #[test]
    fn fuzzy_threshold_empty_input() {
        let ft = FuzzyThreshold::new();
        let result = ft.compute(&[]);
        assert_eq!(result.threshold, 0.0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn fuzzy_threshold_uniform_field() {
        let ft = FuzzyThreshold::new();
        let intensities = vec![vec![0.7f64; 32]; 32];
        let result = ft.compute(&intensities);
        assert!(result.threshold >= 0.0);
        assert!(result.fuzzy_entropy.is_finite());
        assert!(result.ambiguity >= 0.0);
        assert!(result.ambiguity <= 1.0);
    }

    #[test]
    fn fuzzy_threshold_bimodal_field() {
        let ft = FuzzyThreshold::new();
        let mut intensities = vec![vec![0.1f64; 32]; 32];
        for y in 16..32 {
            for x in 16..32 {
                intensities[y][x] = 7.0;
            }
        }
        let result = ft.compute(&intensities);
        assert!(result.threshold > 0.1);
        assert!(result.threshold < 7.0);
    }

    #[test]
    fn fuzzy_threshold_update_params() {
        let mut ft = FuzzyThreshold::new();
        ft.update_params(128, 2.0);
        assert_eq!(ft.num_bins, 128);
        assert!((ft.lambda - 2.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // LocalThreshold tests
    // ---------------------------------------------------------------------------

    #[test]
    fn local_threshold_default() {
        let lt = LocalThreshold::default();
        assert_eq!(lt.window_size, 15);
        assert_eq!(lt.c, 0.0);
        assert_eq!(lt.std_multiplier, 0.0);
    }

    #[test]
    fn local_threshold_empty_input() {
        let lt = LocalThreshold::new();
        let result = lt.compute(&[]);
        assert!(result.threshold_grid.is_empty());
        assert_eq!(result.mean_threshold, 0.0);
    }

    #[test]
    fn local_threshold_uniform_field() {
        let lt = LocalThreshold::new();
        let intensities = vec![vec![1.0f64; 16]; 16];
        let result = lt.compute(&intensities);
        assert_eq!(result.threshold_grid.len(), 16);
        assert_eq!(result.threshold_grid[0].len(), 16);
        assert!((result.mean_threshold - 1.0).abs() < 0.001);
        assert_eq!(result.threshold_std, 0.0);
    }

    #[test]
    fn local_threshold_varying_field() {
        let lt = LocalThreshold::new();
        let mut intensities = vec![vec![0.0f64; 16]; 16];
        for y in 8..16 {
            for x in 8..16 {
                intensities[y][x] = 5.0;
            }
        }
        let result = lt.compute(&intensities);
        assert!(!result.threshold_grid.is_empty());
        assert!(result.threshold_std > 0.0);
    }

    #[test]
    fn local_threshold_apply() {
        let lt = LocalThreshold::new();
        let intensities = vec![vec![0.5f64; 8]; 8];
        let mask = lt.apply(&intensities);
        assert_eq!(mask.len(), 8);
        assert_eq!(mask[0].len(), 8);
        assert!(mask[0][0]); // 0.5 >= local mean 0.5
    }

    #[test]
    fn local_threshold_update_params() {
        let mut lt = LocalThreshold::new();
        lt.update_params(7, 0.5, 1.0);
        assert_eq!(lt.window_size, 7);
        assert!((lt.c - 0.5).abs() < 0.001);
        assert!((lt.std_multiplier - 1.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // HysteresisThreshold tests
    // ---------------------------------------------------------------------------

    #[test]
    fn hysteresis_threshold_default() {
        let ht = HysteresisThreshold::default();
        assert!((ht.low_threshold - 0.2).abs() < 0.001);
        assert!((ht.high_threshold - 0.5).abs() < 0.001);
        assert_eq!(ht.connectivity, 8);
    }

    #[test]
    fn hysteresis_threshold_empty_input() {
        let ht = HysteresisThreshold::new(0.1, 0.5);
        let result = ht.apply(&[]);
        assert!(result.mask.is_empty());
        assert_eq!(result.strong_count, 0);
    }

    #[test]
    fn hysteresis_threshold_basic() {
        let ht = HysteresisThreshold::new(0.2, 0.8);
        let intensities = vec![vec![0.9f64; 8]; 8];
        let result = ht.apply(&intensities);
        assert_eq!(result.strong_count, 64);
        assert!(result.connected_weak_count >= 0);
    }

    #[test]
    fn hysteresis_threshold_mixed() {
        let ht = HysteresisThreshold::new(0.3, 0.7);
        let mut intensities = vec![vec![0.1f64; 8]; 8];
        for y in 3..5 {
            for x in 3..5 {
                intensities[y][x] = 0.9;
            }
        }
        let result = ht.apply(&intensities);
        assert_eq!(result.strong_count, 4);
        assert!(result.connected_weak_count >= 0);
    }

    #[test]
    fn hysteresis_threshold_update_thresholds() {
        let mut ht = HysteresisThreshold::new(0.1, 0.4);
        ht.update_thresholds(0.2, 0.9);
        assert!((ht.low_threshold - 0.2).abs() < 0.001);
        assert!((ht.high_threshold - 0.9).abs() < 0.001);
    }

    #[test]
    fn hysteresis_threshold_connectivity() {
        let mut ht = HysteresisThreshold::new(0.1, 0.5);
        ht.set_connectivity(4);
        assert_eq!(ht.connectivity, 4);
        ht.set_connectivity(8);
        assert_eq!(ht.connectivity, 8);
    }

    // ---------------------------------------------------------------------------
    // ThresholdQualityAssessor tests
    // ---------------------------------------------------------------------------

    #[test]
    fn threshold_quality_assessor_default() {
        let tqa = ThresholdQualityAssessor::default();
        assert_eq!(tqa.stability_trials, 20);
        assert!((tqa.stability_perturbation - 0.05).abs() < 0.001);
    }

    #[test]
    fn threshold_quality_assess_empty_input() {
        let tqa = ThresholdQualityAssessor::new();
        let result = tqa.assess(&[], 0.5);
        assert_eq!(result.separation, 0.0);
        assert_eq!(result.compactness, 0.0);
        assert_eq!(result.stability, 0.0);
        assert_eq!(result.overall, 0.0);
    }

    #[test]
    fn threshold_quality_assess_uniform_field() {
        let tqa = ThresholdQualityAssessor::new();
        let intensities = vec![vec![1.0f64; 16]; 16];
        let result = tqa.assess(&intensities, 0.5);
        // All pixels above threshold -> no separation.
        assert_eq!(result.separation, 0.0);
        assert!(result.overall >= 0.0);
        assert!(result.overall <= 1.0);
    }

    #[test]
    fn threshold_quality_assess_bimodal_field() {
        let tqa = ThresholdQualityAssessor::new();
        let mut intensities = vec![vec![0.1f64; 16]; 16];
        for y in 8..16 {
            for x in 8..16 {
                intensities[y][x] = 9.0;
            }
        }
        let result = tqa.assess(&intensities, 1.0);
        assert!(result.separation > 0.0);
        assert!(result.overall > 0.0);
    }

    #[test]
    fn threshold_quality_bounds() {
        let tqa = ThresholdQualityAssessor::new();
        let intensities = vec![vec![0.5f64; 8]; 8];
        let result = tqa.assess(&intensities, 0.5);
        assert!(result.separation >= 0.0 && result.separation <= 1.0);
        assert!(result.compactness >= 0.0 && result.compactness <= 1.0);
        assert!(result.stability >= 0.0 && result.stability <= 1.0);
        assert!(result.overall >= 0.0 && result.overall <= 1.0);
    }

    #[test]
    fn threshold_quality_update_params() {
        let mut tqa = ThresholdQualityAssessor::new();
        tqa.update_params(50, 0.1);
        assert_eq!(tqa.stability_trials, 50);
        assert!((tqa.stability_perturbation - 0.1).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // MultiLevelThreshold tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_level_threshold_default() {
        let mlt = MultiLevelThreshold::default();
        assert_eq!(mlt.num_thresholds, 1);
        assert_eq!(mlt.num_bins, 256);
    }

    #[test]
    fn multi_level_threshold_empty_input() {
        let mlt = MultiLevelThreshold::new(2);
        let result = mlt.compute(&[]);
        assert!(result.thresholds.is_empty());
        assert_eq!(result.num_levels, 1);
    }

    #[test]
    fn multi_level_threshold_single() {
        let mlt = MultiLevelThreshold::new(1);
        let intensities: Vec<Vec<f64>> = (0..32).map(|_| vec![0.5f64; 32]).collect();
        let result = mlt.compute(&intensities);
        assert_eq!(result.thresholds.len(), 1);
        assert_eq!(result.num_levels, 2);
    }

    #[test]
    fn multi_level_threshold_double() {
        let mlt = MultiLevelThreshold::new(2);
        let intensities: Vec<Vec<f64>> = (0..32).map(|_| vec![0.5f64; 32]).collect();
        let result = mlt.compute(&intensities);
        assert_eq!(result.thresholds.len(), 2);
        assert_eq!(result.num_levels, 3);
    }

    #[test]
    fn multi_level_threshold_ordered() {
        let mlt = MultiLevelThreshold::new(3);
        let intensities: Vec<Vec<f64>> = (0..32).map(|_| vec![0.5f64; 32]).collect();
        let result = mlt.compute(&intensities);
        assert_eq!(result.thresholds.len(), 3);
        // Thresholds should be in ascending order.
        for i in 1..result.thresholds.len() {
            assert!(result.thresholds[i] >= result.thresholds[i - 1]);
        }
    }

    #[test]
    fn multi_level_threshold_bimodal() {
        let mlt = MultiLevelThreshold::new(2);
        let mut intensities = vec![vec![0.1f64; 32]; 32];
        for y in 16..32 {
            for x in 16..32 {
                intensities[y][x] = 8.0;
            }
        }
        let result = mlt.compute(&intensities);
        assert_eq!(result.thresholds.len(), 2);
        assert!(result.between_class_variance >= 0.0);
    }

    #[test]
    fn multi_level_threshold_update_params() {
        let mut mlt = MultiLevelThreshold::new(1);
        mlt.update_params(3, 512);
        assert_eq!(mlt.num_thresholds, 3);
        assert_eq!(mlt.num_bins, 512);
    }

    // ---------------------------------------------------------------------------
    // ThresholdAdaptationController tests
    // ---------------------------------------------------------------------------

    #[test]
    fn threshold_adaptation_controller_default() {
        let tac = ThresholdAdaptationController::default();
        assert!((tac.current_threshold - 0.5).abs() < 0.001);
        assert!((tac.target_fpr - 0.1).abs() < 0.001);
        assert!((tac.learning_rate - 0.1).abs() < 0.001);
    }

    #[test]
    fn threshold_adaptation_adapt_increases() {
        let mut tac = ThresholdAdaptationController::new(0.5);
        // FPR above target -> threshold should increase.
        let new_t = tac.adapt(0.3);
        assert!(new_t > 0.5);
    }

    #[test]
    fn threshold_adaptation_adapt_decreases() {
        let mut tac = ThresholdAdaptationController::new(0.5);
        // FPR below target -> threshold should decrease.
        let new_t = tac.adapt(0.01);
        assert!(new_t < 0.5);
    }

    #[test]
    fn threshold_adaptation_clamp() {
        let mut tac = ThresholdAdaptationController::new(0.9);
        tac.update_params(0.1, 1.0, 0.0, 1.0);
        // Extreme error should be clamped to max.
        let new_t = tac.adapt(0.99);
        assert!(new_t <= 1.0);
    }

    #[test]
    fn threshold_adaptation_avg_threshold() {
        let mut tac = ThresholdAdaptationController::new(0.5);
        tac.adapt(0.3);
        tac.adapt(0.3);
        let avg = tac.avg_threshold();
        assert!(avg > 0.5);
    }

    #[test]
    fn threshold_adaptation_variance() {
        let mut tac = ThresholdAdaptationController::new(0.5);
        tac.adapt(0.3);
        tac.adapt(0.1);
        let var = tac.threshold_variance();
        assert!(var >= 0.0);
    }

    #[test]
    fn threshold_adaptation_reset() {
        let mut tac = ThresholdAdaptationController::new(0.5);
        tac.adapt(0.3);
        assert!(!tac.history.is_empty());
        tac.reset();
        assert!(tac.history.is_empty());
    }

    #[test]
    fn threshold_adaptation_update_params() {
        let mut tac = ThresholdAdaptationController::new(0.5);
        tac.update_params(0.2, 0.5, 0.1, 0.9);
        assert!((tac.target_fpr - 0.2).abs() < 0.001);
        assert!((tac.learning_rate - 0.5).abs() < 0.001);
        assert!((tac.min_threshold - 0.1).abs() < 0.001);
        assert!((tac.max_threshold - 0.9).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // ThresholdSweep additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn threshold_sweep_adaptive_steps() {
        let mut sweep = ThresholdSweep::new(0.0, 1.0, 10);
        sweep.adaptive_steps = true;
        assert!(sweep.adaptive_steps);
    }

    #[test]
    fn threshold_sweep_high_contrast() {
        let sweep = ThresholdSweep::new(0.0, 1.0, 20);
        let mut intensities = vec![vec![0.0f64; 10]; 10];
        for y in 3..7 {
            for x in 3..7 {
                intensities[y][x] = 1.0;
            }
        }
        let results = sweep.sweep(&intensities);
        assert_eq!(results.len(), 20);
        // At threshold 0, all valid.
        assert!((results[0].valid_fraction - 1.0).abs() < 0.001);
        // At threshold 1.0, none valid.
        assert_eq!(results.last().unwrap().valid_fraction, 0.0);
    }

    #[test]
    fn threshold_sweep_snr_estimate() {
        let sweep = ThresholdSweep::new(0.0, 1.0, 10);
        let intensities = vec![vec![0.5f64; 10]; 10];
        let results = sweep.sweep(&intensities);
        // All pixels same value -> SNR is infinite or very high.
        assert!(results[0].snr_estimate.is_infinite() || results[0].snr_estimate > 0.0);
    }

    // ---------------------------------------------------------------------------
    // OptimizationStrategy additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn optimization_strategy_target_fraction_value() {
        let s = OptimizationStrategy::TargetFraction(0.7);
        assert!(matches!(s, OptimizationStrategy::TargetFraction(0.7)));
    }

    #[test]
    fn optimization_strategy_all_variants() {
        let _ = OptimizationStrategy::Otsu;
        let _ = OptimizationStrategy::KittlerIllingworth;
        let _ = OptimizationStrategy::TargetFraction(0.5);
        let _ = OptimizationStrategy::NoiseAdaptive;
    }

    // ---------------------------------------------------------------------------
    // ThresholdOptimizer KittlerIllingworth path
    // ---------------------------------------------------------------------------

    #[test]
    fn optimizer_kittler_illingworth_strategy() {
        let optimizer = ThresholdOptimizer::new();
        let intensities = vec![vec![1.0f64; 32]; 32];
        let result = optimizer.optimize(&intensities, OptimizationStrategy::KittlerIllingworth);
        assert_eq!(result.strategy, OptimizationStrategy::KittlerIllingworth);
        assert!(result.converged);
    }

    // ---------------------------------------------------------------------------
    // Edge case: all zeros
    // ---------------------------------------------------------------------------

    #[test]
    fn all_methods_all_zeros() {
        let intensities = vec![vec![0.0f64; 16]; 16];

        let otsu = ThresholdOptimizer::new().optimize_otsu(&intensities);
        assert_eq!(otsu.optimal_threshold, 0.0);

        let ki = KittlerIllingworth::new().compute(&intensities);
        assert_eq!(ki.threshold, 0.0);

        let et = EntropyThreshold::new().compute(&intensities);
        assert_eq!(et.threshold, 0.0);

        let mp = MomentPreserving::new().compute(&intensities);
        assert_eq!(mp.threshold, 0.0);

        let ft = FuzzyThreshold::new().compute(&intensities);
        assert_eq!(ft.threshold, 0.0);
    }

    // ---------------------------------------------------------------------------
    // Edge case: single pixel
    // ---------------------------------------------------------------------------

    #[test]
    fn all_methods_single_pixel() {
        let intensities = vec![vec![42.0f64]];

        let otsu = ThresholdOptimizer::new().optimize_otsu(&intensities);
        assert!(otsu.optimal_threshold >= 0.0);

        let ki = KittlerIllingworth::new().compute(&intensities);
        assert!(ki.threshold >= 0.0);

        let lt = LocalThreshold::new().compute(&intensities);
        assert_eq!(lt.threshold_grid.len(), 1);
        assert_eq!(lt.threshold_grid[0].len(), 1);
    }
}