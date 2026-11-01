//! Crosstalk matrix computation and analysis for topological signal detection.
//!
//! Models inter-channel interference (crosstalk) between skyrmion number
//! constellation points, computes crosstalk matrices from measured data,
//! applies crosstalk correction, and quantifies crosstalk-induced error
//! rates. All operations are carrier-agnostic.

use crate::errors::{XpockError, XpockResult};
use std::collections::HashMap;

// ============================================================================
// Crosstalk matrix types
// ============================================================================

/// A crosstalk matrix C where C[i][j] = fraction of signal from channel j
/// that leaks into channel i. Diagonal entries represent direct transmission,
/// off-diagonal entries represent crosstalk.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkMatrix {
    /// Number of channels (constellation points).
    pub num_channels: usize,
    /// Matrix entries: C[i][j] = crosstalk from channel j to channel i.
    pub matrix: Vec<Vec<f64>>,
    /// Channel labels (skyrmion number values).
    pub channel_labels: Vec<i32>,
    /// Matrix computation timestamp.
    pub computed_at: u64,
    /// Confidence in the matrix estimate (0.0 to 1.0).
    pub confidence: f64,
}

impl CrosstalkMatrix {
    /// Create a new identity crosstalk matrix (no crosstalk).
    pub fn identity(num_channels: usize, channel_labels: Vec<i32>) -> Self {
        let mut matrix = vec![vec![0.0f64; num_channels]; num_channels];
        for i in 0..num_channels {
            matrix[i][i] = 1.0;
        }
        Self {
            num_channels,
            matrix,
            channel_labels,
            computed_at: crate::detection::signal_detector::lib_ms(),
            confidence: 1.0,
        }
    }

    /// Create a new crosstalk matrix from raw entries.
    pub fn new(
        num_channels: usize,
        matrix: Vec<Vec<f64>>,
        channel_labels: Vec<i32>,
    ) -> XpockResult<Self> {
        if matrix.len() != num_channels
            || matrix.iter().any(|row| row.len() != num_channels)
        {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (num_channels, num_channels),
                    actual: (matrix.len(), matrix.first().map_or(0, |r| r.len())),
                    component: "crosstalk_matrix".to_string(),
                },
            ));
        }
        if channel_labels.len() != num_channels {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (num_channels, 1),
                    actual: (channel_labels.len(), 1),
                    component: "channel_labels".to_string(),
                },
            ));
        }

        let confidence = Self::compute_confidence(&matrix);

        Ok(Self {
            num_channels,
            matrix,
            channel_labels,
            computed_at: crate::detection::signal_detector::lib_ms(),
            confidence,
        })
    }

    /// Compute confidence based on diagonal dominance.
    fn compute_confidence(matrix: &[Vec<f64>]) -> f64 {
        let n = matrix.len();
        if n == 0 {
            return 0.0;
        }

        let mut diagonal_dominance_sum = 0.0f64;
        for i in 0..n {
            let diagonal = matrix[i][i].abs();
            let off_diagonal_sum: f64 = matrix[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v.abs())
                .sum();
            let total = diagonal + off_diagonal_sum;
            if total > 1e-10 {
                diagonal_dominance_sum += diagonal / total;
            }
        }

        (diagonal_dominance_sum / n as f64).clamp(0.0, 1.0)
    }

    /// Returns the crosstalk from channel j to channel i.
    pub fn crosstalk(&self, from: usize, to: usize) -> f64 {
        self.matrix[to][from]
    }

    /// Returns the total crosstalk into channel i (sum of off-diagonal in row i).
    pub fn total_crosstalk_into(&self, channel: usize) -> f64 {
        self.matrix[channel]
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != channel)
            .map(|(_, &v)| v.abs())
            .sum()
    }

    /// Returns the total crosstalk out of channel j (sum of off-diagonal in column j).
    pub fn total_crosstalk_out_of(&self, channel: usize) -> f64 {
        (0..self.num_channels)
            .filter(|&i| i != channel)
            .map(|i| self.matrix[i][channel].abs())
            .sum()
    }

    /// Returns the maximum off-diagonal crosstalk entry.
    pub fn max_crosstalk(&self) -> f64 {
        let mut max = 0.0f64;
        for i in 0..self.num_channels {
            for j in 0..self.num_channels {
                if i != j {
                    max = max.max(self.matrix[i][j].abs());
                }
            }
        }
        max
    }

    /// Returns true if the matrix is diagonally dominant (low crosstalk).
    pub fn is_diagonally_dominant(&self, threshold: f64) -> bool {
        for i in 0..self.num_channels {
            let diagonal = self.matrix[i][i].abs();
            let off_diagonal: f64 = self.matrix[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v.abs())
                .sum();
            if diagonal < off_diagonal * threshold {
                return false;
            }
        }
        true
    }

    /// Returns the condition number estimate (ratio of max to min singular value proxy).
    pub fn condition_number_estimate(&self) -> f64 {
        // Simple estimate: max diagonal / min diagonal.
        let mut max_diag = 0.0f64;
        let mut min_diag = f64::INFINITY;
        for i in 0..self.num_channels {
            let d = self.matrix[i][i].abs();
            max_diag = max_diag.max(d);
            min_diag = min_diag.min(d);
        }
        if min_diag < 1e-10 {
            f64::INFINITY
        } else {
            max_diag / min_diag
        }
    }

    /// Returns the channel with the most crosstalk.
    pub fn worst_channel(&self) -> (usize, f64) {
        let mut worst = (0usize, 0.0f64);
        for i in 0..self.num_channels {
            let total = self.total_crosstalk_into(i);
            if total > worst.1 {
                worst = (i, total);
            }
        }
        worst
    }

    /// Returns the channel with the least crosstalk.
    pub fn best_channel(&self) -> (usize, f64) {
        let mut best = (0usize, f64::INFINITY);
        for i in 0..self.num_channels {
            let total = self.total_crosstalk_into(i);
            if total < best.1 {
                best = (i, total);
            }
        }
        best
    }
}

impl Default for CrosstalkMatrix {
    fn default() -> Self {
        Self::identity(2, vec![-1, 1])
    }
}

/// Crosstalk measurement from a single observation.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkMeasurement {
    /// Transmitted channel index.
    pub transmitted_channel: usize,
    /// Received channel index.
    pub received_channel: usize,
    /// Measured intensity at received channel.
    pub intensity: f64,
    /// Measurement timestamp.
    pub timestamp: u64,
}

/// Collects crosstalk measurements for matrix estimation.
pub struct CrosstalkMeasurementCollector {
    /// Collected measurements.
    measurements: Vec<CrosstalkMeasurement>,
    /// Maximum history length.
    max_history: usize,
    /// Number of channels.
    num_channels: usize,
}

impl CrosstalkMeasurementCollector {
    /// Create a new collector.
    pub fn new(num_channels: usize) -> Self {
        Self {
            measurements: Vec::with_capacity(1000),
            max_history: 10000,
            num_channels,
        }
    }

    /// Record a crosstalk measurement.
    pub fn record(
        &mut self,
        transmitted: usize,
        received: usize,
        intensity: f64,
    ) -> XpockResult<()> {
        if transmitted >= self.num_channels || received >= self.num_channels {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "channel_index".to_string(),
                    value: format!("({}, {})", transmitted, received),
                    constraint: format!("both < {}", self.num_channels),
                },
            ));
        }

        if self.measurements.len() >= self.max_history {
            self.measurements.remove(0);
        }
        self.measurements.push(CrosstalkMeasurement {
            transmitted_channel: transmitted,
            received_channel: received,
            intensity,
            timestamp: crate::detection::signal_detector::lib_ms(),
        });

        Ok(())
    }

    /// Estimate crosstalk matrix from collected measurements.
    /// Normalizes each column so diagonal represents direct transmission.
    pub fn estimate_matrix(&self, channel_labels: &[i32]) -> XpockResult<CrosstalkMatrix> {
        if self.measurements.is_empty() {
            return CrosstalkMatrix::identity(self.num_channels, channel_labels.to_vec());
        }

        // Accumulate intensity sums per (transmitted, received) pair.
        let mut sums = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        let mut counts = vec![vec![0usize; self.num_channels]; self.num_channels];

        for m in &self.measurements {
            sums[m.received_channel][m.transmitted_channel] += m.intensity;
            counts[m.received_channel][m.transmitted_channel] += 1;
        }

        // Compute mean intensities.
        let mut mean_matrix = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        for i in 0..self.num_channels {
            for j in 0..self.num_channels {
                if counts[i][j] > 0 {
                    mean_matrix[i][j] = sums[i][j] / counts[i][j] as f64;
                }
            }
        }

        // Normalize columns: divide each column by its diagonal element
        // so C[j][j] = 1 (direct transmission = 100%).
        let mut crosstalk_matrix = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        for j in 0..self.num_channels {
            let diagonal = mean_matrix[j][j];
            if diagonal.abs() > 1e-10 {
                for i in 0..self.num_channels {
                    crosstalk_matrix[i][j] = mean_matrix[i][j] / diagonal;
                }
            } else {
                crosstalk_matrix[j][j] = 1.0;
            }
        }

        CrosstalkMatrix::new(self.num_channels, crosstalk_matrix, channel_labels.to_vec())
    }

    /// Returns the number of measurements collected.
    pub fn count(&self) -> usize {
        self.measurements.len()
    }

    /// Clear all measurements.
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

impl Default for CrosstalkMeasurementCollector {
    fn default() -> Self {
        Self::new(2)
    }
}

// ============================================================================
// Crosstalk correction
// ============================================================================

/// Result of crosstalk correction.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkCorrectionResult {
    /// Corrected signal intensities (crosstalk removed).
    pub corrected_signal: Vec<f64>,
    /// Original measured signal.
    pub measured_signal: Vec<f64>,
    /// Crosstalk matrix used for correction.
    pub crosstalk_matrix: CrosstalkMatrix,
    /// Correction confidence (0.0 to 1.0).
    pub correction_confidence: f64,
    /// Residual error after correction.
    pub residual_error: f64,
}

/// Corrects measured signals for crosstalk using matrix inversion.
pub struct CrosstalkCorrector {
    /// Regularization parameter for matrix inversion.
    pub regularization: f64,
    /// Maximum condition number for reliable inversion.
    pub max_condition_number: f64,
    /// Correction history.
    history: Vec<CrosstalkCorrectionResult>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkCorrector {
    /// Create a new crosstalk corrector.
    pub fn new() -> Self {
        Self {
            regularization: 1e-6,
            max_condition_number: 100.0,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Correct a measured signal vector using the crosstalk matrix.
    ///
    /// Solves: measured = C * true_signal, so true_signal = C^{-1} * measured.
    pub fn correct(
        &mut self,
        measured: &[f64],
        crosstalk_matrix: &CrosstalkMatrix,
    ) -> XpockResult<CrosstalkCorrectionResult> {
        if measured.len() != crosstalk_matrix.num_channels {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (crosstalk_matrix.num_channels, 1),
                    actual: (measured.len(), 1),
                    component: "measured_signal".to_string(),
                },
            ));
        }

        let n = crosstalk_matrix.num_channels;

        // Build the matrix as a 2D array for Gaussian elimination.
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = crosstalk_matrix.matrix[i][j];
            }
            a[i][n] = measured[i];
        }

        // Add regularization to diagonal.
        for i in 0..n {
            a[i][i] += self.regularization;
        }

        // Gaussian elimination with partial pivoting.
        let mut success = true;
        for col in 0..n {
            // Find pivot.
            let mut pivot_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    pivot_row = row;
                }
            }

            if max_val < 1e-12 {
                success = false;
                break;
            }

            // Swap rows.
            if pivot_row != col {
                a.swap(col, pivot_row);
            }

            // Eliminate below.
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for k in col..=n {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }

        let corrected_signal = if success {
            // Back substitution.
            let mut x = vec![0.0f64; n];
            for i in (0..n).rev() {
                let mut sum = a[i][n];
                for j in (i + 1)..n {
                    sum -= a[i][j] * x[j];
                }
                x[i] = sum / a[i][i];
            }
            x
        } else {
            // Fallback: return measured signal (no correction possible).
            measured.to_vec()
        };

        // Compute residual: C * corrected - measured.
        let mut residual = 0.0f64;
        for i in 0..n {
            let mut reconstructed = 0.0f64;
            for j in 0..n {
                reconstructed += crosstalk_matrix.matrix[i][j] * corrected_signal[j];
            }
            residual += (reconstructed - measured[i]).powi(2);
        }
        residual = (residual / n as f64).sqrt();

        // Correction confidence based on condition number and residual.
        let cond = crosstalk_matrix.condition_number_estimate();
        let cond_factor = (1.0 - (cond / self.max_condition_number).clamp(0.0, 1.0));
        let residual_factor = (1.0 - residual.clamp(0.0, 1.0));
        let correction_confidence = (cond_factor * 0.6 + residual_factor * 0.4).clamp(0.0, 1.0);

        let result = CrosstalkCorrectionResult {
            corrected_signal,
            measured_signal: measured.to_vec(),
            crosstalk_matrix: crosstalk_matrix.clone(),
            correction_confidence,
            residual_error: residual,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        Ok(result)
    }

    /// Correct multiple signal vectors.
    pub fn correct_batch(
        &mut self,
        signals: &[Vec<f64>],
        crosstalk_matrix: &CrosstalkMatrix,
    ) -> XpockResult<Vec<CrosstalkCorrectionResult>> {
        let mut results = Vec::with_capacity(signals.len());
        for signal in signals {
            results.push(self.correct(signal, crosstalk_matrix)?);
        }
        Ok(results)
    }

    /// Returns the average correction confidence across history.
    pub fn avg_confidence(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.correction_confidence)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the average residual error across history.
    pub fn avg_residual(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.residual_error)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Reset correction history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update corrector parameters.
    pub fn update_params(&mut self, regularization: f64, max_condition_number: f64) {
        self.regularization = regularization.max(0.0);
        self.max_condition_number = max_condition_number.max(1.0);
    }
}

impl Default for CrosstalkCorrector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Crosstalk calibration
// ============================================================================

/// Calibration result for crosstalk matrix estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkCalibrationResult {
    /// Estimated crosstalk matrix.
    pub matrix: CrosstalkMatrix,
    /// Number of calibration measurements used.
    pub num_measurements: usize,
    /// Calibration quality score (0.0 to 1.0).
    pub quality: f64,
    /// Calibration timestamp.
    pub timestamp: u64,
}

/// Calibrates crosstalk matrix using known channel transmissions.
pub struct CrosstalkCalibrator {
    /// Number of channels.
    num_channels: usize,
    /// Channel labels.
    channel_labels: Vec<i32>,
    /// Measurement collector.
    collector: CrosstalkMeasurementCollector,
    /// Calibration history.
    history: Vec<CrosstalkCalibrationResult>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkCalibrator {
    /// Create a new crosstalk calibrator.
    pub fn new(num_channels: usize, channel_labels: Vec<i32>) -> Self {
        Self {
            num_channels,
            channel_labels: channel_labels.clone(),
            collector: CrosstalkMeasurementCollector::new(num_channels),
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Record a calibration measurement: transmit channel `tx`, observe at `rx`.
    pub fn record_calibration(
        &mut self,
        tx: usize,
        rx: usize,
        intensity: f64,
    ) -> XpockResult<()> {
        self.collector.record(tx, rx, intensity)
    }

    /// Perform calibration: estimate crosstalk matrix from collected data.
    pub fn calibrate(&mut self) -> XpockResult<CrosstalkCalibrationResult> {
        let matrix = self.collector.estimate_matrix(&self.channel_labels)?;
        let num_measurements = self.collector.count();

        // Quality based on matrix confidence and diagonal dominance.
        let quality = (matrix.confidence * 0.6
            + if matrix.is_diagonally_dominant(2.0) {
                1.0
            } else {
                0.5
            } * 0.4)
            .clamp(0.0, 1.0);

        let result = CrosstalkCalibrationResult {
            matrix,
            num_measurements,
            quality,
            timestamp: crate::detection::signal_detector::lib_ms(),
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        Ok(result)
    }

    /// Returns the latest calibration result.
    pub fn latest_calibration(&self) -> Option<&CrosstalkCalibrationResult> {
        self.history.last()
    }

    /// Returns the calibration history.
    pub fn history(&self) -> &[CrosstalkCalibrationResult] {
        &self.history
    }

    /// Returns the number of measurements collected.
    pub fn measurement_count(&self) -> usize {
        self.collector.count()
    }

    /// Clear all calibration data.
    pub fn clear(&mut self) {
        self.collector.clear();
        self.history.clear();
    }

    /// Update channel labels.
    pub fn update_channel_labels(&mut self, labels: Vec<i32>) -> XpockResult<()> {
        if labels.len() != self.num_channels {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.num_channels, 1),
                    actual: (labels.len(), 1),
                    component: "channel_labels".to_string(),
                },
            ));
        }
        self.channel_labels = labels;
        Ok(())
    }
}

/// A measured crosstalk coupling coefficient with uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkCoefficient {
    /// Source channel index.
    pub source: usize,
    /// Destination channel index.
    pub destination: usize,
    /// Measured coupling coefficient.
    pub coefficient: f64,
    /// Uncertainty (standard deviation).
    pub uncertainty: f64,
    /// Number of samples used.
    pub num_samples: usize,
}

/// Aggregates crosstalk coefficients across multiple measurements.
pub struct CrosstalkCoefficientAggregator {
    /// Coefficients indexed by (source, destination).
    coefficients: HashMap<(usize, usize), Vec<f64>>,
    /// Number of channels.
    num_channels: usize,
}

impl CrosstalkCoefficientAggregator {
    /// Create a new aggregator.
    pub fn new(num_channels: usize) -> Self {
        Self {
            coefficients: HashMap::new(),
            num_channels,
        }
    }

    /// Add a coefficient measurement.
    pub fn add(&mut self, source: usize, destination: usize, value: f64) {
        if source < self.num_channels && destination < self.num_channels {
            self.coefficients
                .entry((source, destination))
                .or_default()
                .push(value);
        }
    }

    /// Compute aggregated coefficient with mean and uncertainty.
    pub fn aggregate(&self, source: usize, destination: usize) -> Option<CrosstalkCoefficient> {
        let values = self.coefficients.get(&(source, destination))?;
        if values.is_empty() {
            return None;
        }

        let n = values.len();
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let uncertainty = variance.sqrt();

        Some(CrosstalkCoefficient {
            source,
            destination,
            coefficient: mean,
            uncertainty,
            num_samples: n,
        })
    }

    /// Build a full crosstalk matrix from aggregated coefficients.
    pub fn build_matrix(&self, channel_labels: &[i32]) -> CrosstalkMatrix {
        let mut matrix = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        for i in 0..self.num_channels {
            for j in 0..self.num_channels {
                if let Some(coeff) = self.aggregate(i, j) {
                    matrix[i][j] = coeff.coefficient;
                }
            }
        }
        CrosstalkMatrix::new(
            self.num_channels,
            matrix,
            channel_labels.to_vec(),
        )
        .unwrap_or_else(|_| CrosstalkMatrix::identity(self.num_channels, channel_labels.to_vec()))
    }

    /// Returns the total number of coefficient measurements.
    pub fn total_measurements(&self) -> usize {
        self.coefficients.values().map(|v| v.len()).sum()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.coefficients.clear();
    }
}

impl Default for CrosstalkCoefficientAggregator {
    fn default() -> Self {
        Self::new(2)
    }
}

// ============================================================================
// Crosstalk quality assessment
// ============================================================================

/// Crosstalk quality metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkQuality {
    /// Overall crosstalk level (0.0 = none, 1.0 = severe).
    pub overall_level: f64,
    /// Worst channel crosstalk.
    pub worst_channel_crosstalk: f64,
    /// Average off-diagonal crosstalk.
    pub avg_off_diagonal: f64,
    /// Diagonal dominance ratio.
    pub diagonal_dominance: f64,
    /// Number of channels exceeding crosstalk threshold.
    pub channels_above_threshold: usize,
    /// Quality score (0.0 = poor, 1.0 = excellent).
    pub quality_score: f64,
}

/// Assesses crosstalk quality from a crosstalk matrix.
pub struct CrosstalkQualityAssessor {
    /// Crosstalk threshold for flagging channels.
    pub crosstalk_threshold: f64,
    /// Quality assessment history.
    history: Vec<CrosstalkQuality>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkQualityAssessor {
    /// Create a new quality assessor.
    pub fn new() -> Self {
        Self {
            crosstalk_threshold: 0.15,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Assess crosstalk quality from a matrix.
    pub fn assess(&mut self, matrix: &CrosstalkMatrix) -> CrosstalkQuality {
        let n = matrix.num_channels;
        if n == 0 {
            return CrosstalkQuality {
                overall_level: 0.0,
                worst_channel_crosstalk: 0.0,
                avg_off_diagonal: 0.0,
                diagonal_dominance: 1.0,
                channels_above_threshold: 0,
                quality_score: 1.0,
            };
        }

        // Compute average off-diagonal crosstalk.
        let mut off_diag_sum = 0.0f64;
        let mut off_diag_count = 0usize;
        let mut max_crosstalk = 0.0f64;
        let mut channels_above = 0usize;

        for i in 0..n {
            let mut row_off_sum = 0.0f64;
            for j in 0..n {
                if i != j {
                    let val = matrix.matrix[i][j].abs();
                    off_diag_sum += val;
                    row_off_sum += val;
                    max_crosstalk = max_crosstalk.max(val);
                    off_diag_count += 1;
                }
            }
            if row_off_sum > self.crosstalk_threshold {
                channels_above += 1;
            }
        }

        let avg_off_diagonal = if off_diag_count > 0 {
            off_diag_sum / off_diag_count as f64
        } else {
            0.0
        };

        // Diagonal dominance: min(diagonal / sum(off-diagonal)).
        let mut min_dominance = f64::INFINITY;
        for i in 0..n {
            let diagonal = matrix.matrix[i][i].abs();
            let off_sum: f64 = matrix.matrix[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v.abs())
                .sum();
            let dominance = if off_sum > 1e-10 {
                diagonal / off_sum
            } else {
                f64::INFINITY
            };
            min_dominance = min_dominance.min(dominance);
        }
        let diagonal_dominance = if min_dominance == f64::INFINITY {
            1.0
        } else {
            min_dominance.clamp(0.0, 10.0) / 10.0 // Normalize to [0, 1].
        };

        // Overall crosstalk level.
        let overall_level = avg_off_diagonal.clamp(0.0, 1.0);

        // Quality score combines diagonal dominance and low crosstalk.
        let quality_score = (diagonal_dominance * 0.6 + (1.0 - overall_level) * 0.4)
            .clamp(0.0, 1.0);

        let quality = CrosstalkQuality {
            overall_level,
            worst_channel_crosstalk: max_crosstalk,
            avg_off_diagonal,
            diagonal_dominance,
            channels_above_threshold: channels_above,
            quality_score,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(quality.clone());

        quality
    }

    /// Returns true if crosstalk quality is acceptable.
    pub fn is_acceptable(&self, quality: &CrosstalkQuality) -> bool {
        quality.quality_score > 0.7
            && quality.worst_channel_crosstalk < self.crosstalk_threshold * 2.0
    }

    /// Returns the average quality score across history.
    pub fn avg_quality_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|q| q.quality_score)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the pass rate across history.
    pub fn pass_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let passed = self
            .history
            .iter()
            .filter(|q| self.is_acceptable(q))
            .count() as f64;
        passed / self.history.len() as f64
    }

    /// Reset assessment history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update crosstalk threshold.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.crosstalk_threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for CrosstalkQualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Frequency-dependent crosstalk spectrum
// ============================================================================

/// Crosstalk at a single frequency bin.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkFrequencyBin {
    /// Frequency bin index.
    pub bin_index: usize,
    /// Center frequency (normalized 0.0 to 1.0).
    pub frequency: f64,
    /// Crosstalk matrix at this frequency.
    pub matrix: CrosstalkMatrix,
    /// Magnitude of crosstalk at this frequency.
    pub magnitude: f64,
}

/// Frequency-dependent crosstalk spectrum across multiple bins.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkSpectrum {
    /// Frequency bins.
    pub bins: Vec<CrosstalkFrequencyBin>,
    /// Number of channels.
    pub num_channels: usize,
    /// Frequency range (min, max).
    pub frequency_range: (f64, f64),
    /// Peak crosstalk frequency.
    pub peak_frequency: f64,
    /// Peak crosstalk magnitude.
    pub peak_magnitude: f64,
}

/// Analyzes crosstalk across frequency bins.
pub struct CrosstalkSpectrumAnalyzer {
    /// Number of frequency bins.
    pub num_bins: usize,
    /// Frequency range.
    pub frequency_range: (f64, f64),
    /// Spectrum history.
    history: Vec<CrosstalkSpectrum>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkSpectrumAnalyzer {
    /// Create a new spectrum analyzer.
    pub fn new(num_bins: usize, frequency_range: (f64, f64)) -> Self {
        Self {
            num_bins,
            frequency_range,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Analyze crosstalk spectrum from per-bin matrices.
    pub fn analyze(
        &mut self,
        bin_matrices: &[CrosstalkMatrix],
    ) -> XpockResult<CrosstalkSpectrum> {
        if bin_matrices.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        let num_channels = bin_matrices[0].num_channels;
        let (f_min, f_max) = self.frequency_range;
        let bin_width = (f_max - f_min) / self.num_bins as f64;

        let mut bins = Vec::with_capacity(self.num_bins);
        let mut peak_frequency = 0.0f64;
        let mut peak_magnitude = 0.0f64;

        for (i, matrix) in bin_matrices.iter().enumerate() {
            let frequency = f_min + i as f64 * bin_width;
            let magnitude = matrix.max_crosstalk();

            if magnitude > peak_magnitude {
                peak_magnitude = magnitude;
                peak_frequency = frequency;
            }

            bins.push(CrosstalkFrequencyBin {
                bin_index: i,
                frequency,
                matrix: matrix.clone(),
                magnitude,
            });
        }

        let spectrum = CrosstalkSpectrum {
            bins,
            num_channels,
            frequency_range: (f_min, f_max),
            peak_frequency,
            peak_magnitude,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(spectrum.clone());

        Ok(spectrum)
    }

    /// Returns the frequency with maximum crosstalk.
    pub fn peak_crosstalk_frequency(&self, spectrum: &CrosstalkSpectrum) -> f64 {
        spectrum.peak_frequency
    }

    /// Returns the crosstalk magnitude at a given frequency (nearest bin).
    pub fn magnitude_at_frequency(
        &self,
        spectrum: &CrosstalkSpectrum,
        frequency: f64,
    ) -> f64 {
        let (f_min, f_max) = self.frequency_range;
        if frequency <= f_min {
            return spectrum.bins.first().map_or(0.0, |b| b.magnitude);
        }
        if frequency >= f_max {
            return spectrum.bins.last().map_or(0.0, |b| b.magnitude);
        }

        let bin_index =
            ((frequency - f_min) / (f_max - f_min) * self.num_bins as f64) as usize;
        let bin_index = bin_index.min(self.num_bins - 1);
        spectrum.bins[bin_index].magnitude
    }

    /// Returns the average crosstalk magnitude across all bins.
    pub fn avg_magnitude(&self, spectrum: &CrosstalkSpectrum) -> f64 {
        if spectrum.bins.is_empty() {
            return 0.0;
        }
        spectrum
            .bins
            .iter()
            .map(|b| b.magnitude)
            .sum::<f64>()
            / spectrum.bins.len() as f64
    }

    /// Returns the spectrum history.
    pub fn history(&self) -> &[CrosstalkSpectrum] {
        &self.history
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for CrosstalkSpectrumAnalyzer {
    fn default() -> Self {
        Self::new(16, (0.0, 1.0))
    }
}

// ============================================================================
// Adaptive crosstalk cancellation
// ============================================================================

/// Adaptive crosstalk canceller using least-mean-squares (LMS) adaptation.
/// Continuously updates crosstalk matrix estimates based on error feedback.
pub struct AdaptiveCrosstalkCanceller {
    /// Current crosstalk matrix estimate.
    pub matrix: CrosstalkMatrix,
    /// LMS step size (learning rate).
    pub step_size: f64,
    /// Adaptation history.
    history: Vec<CrosstalkMatrix>,
    /// Maximum history length.
    max_history: usize,
    /// Number of adaptations performed.
    pub adaptation_count: usize,
}

impl AdaptiveCrosstalkCanceller {
    /// Create a new adaptive canceller.
    pub fn new(num_channels: usize, channel_labels: Vec<i32>) -> Self {
        Self {
            matrix: CrosstalkMatrix::identity(num_channels, channel_labels),
            step_size: 0.01,
            history: Vec::with_capacity(1000),
            max_history: 1000,
            adaptation_count: 0,
        }
    }

    /// Adapt the crosstalk matrix based on measured error.
    ///
    /// `measured` is the observed output, `expected` is the expected output
    /// (no crosstalk). The error drives the matrix update.
    pub fn adapt(
        &mut self,
        measured: &[f64],
        expected: &[f64],
    ) -> XpockResult<CrosstalkMatrix> {
        if measured.len() != self.matrix.num_channels
            || expected.len() != self.matrix.num_channels
        {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.matrix.num_channels, 1),
                    actual: (measured.len(), 1),
                    component: "adaptive_canceller_input".to_string(),
                },
            ));
        }

        let n = self.matrix.num_channels;
        let mut new_matrix = vec![vec![0.0f64; n]; n];

        // LMS update: C_new = C_old - mu * error * measured^T
        // where error = measured - expected (crosstalk-induced error)
        for i in 0..n {
            for j in 0..n {
                let error = measured[i] - expected[i];
                let gradient = self.step_size * error * measured[j];
                new_matrix[i][j] = self.matrix.matrix[i][j] - gradient;
            }
        }

        // Ensure diagonal stays positive and near 1.
        for i in 0..n {
            new_matrix[i][i] = new_matrix[i][i].max(0.1).min(2.0);
        }

        // Normalize columns.
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| new_matrix[i][j].abs()).sum();
            if col_sum > 1e-10 {
                for i in 0..n {
                    new_matrix[i][j] /= col_sum;
                }
            }
        }

        self.matrix = CrosstalkMatrix::new(n, new_matrix, self.matrix.channel_labels.clone())?;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(self.matrix.clone());
        self.adaptation_count += 1;

        Ok(self.matrix.clone())
    }

    /// Returns the number of adaptations performed.
    pub fn adaptation_count(&self) -> usize {
        self.adaptation_count
    }

    /// Returns the adaptation history.
    pub fn history(&self) -> &[CrosstalkMatrix] {
        &self.history
    }

    /// Returns the current matrix.
    pub fn current_matrix(&self) -> &CrosstalkMatrix {
        &self.matrix
    }

    /// Reset the canceller.
    pub fn reset(&mut self) {
        self.matrix = CrosstalkMatrix::identity(
            self.matrix.num_channels,
            self.matrix.channel_labels.clone(),
        );
        self.history.clear();
        self.adaptation_count = 0;
    }

    /// Update step size.
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size.clamp(1e-6, 1.0);
    }
}

impl Default for AdaptiveCrosstalkCanceller {
    fn default() -> Self {
        Self::new(2, vec![-1, 1])
    }
}

// ============================================================================
// Multi-path crosstalk modeling
// ============================================================================

/// A single crosstalk path (source -> destination with delay and attenuation).
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkPath {
    /// Source channel index.
    pub source: usize,
    /// Destination channel index.
    pub destination: usize,
    /// Path delay (in samples).
    pub delay: usize,
    /// Path attenuation factor (0.0 to 1.0).
    pub attenuation: f64,
    /// Path phase shift (radians).
    pub phase_shift: f64,
    /// Whether the path is active.
    pub active: bool,
}

/// Multi-path crosstalk model with delayed and attenuated paths.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiPathCrosstalkModel {
    /// Number of channels.
    pub num_channels: usize,
    /// All crosstalk paths.
    pub paths: Vec<CrosstalkPath>,
    /// Maximum delay across all paths.
    pub max_delay: usize,
    /// Model quality score.
    pub quality: f64,
}

/// Models crosstalk with multiple propagation paths (echoes, reflections).
pub struct MultiPathCrosstalkModeler {
    /// Maximum number of paths to track.
    pub max_paths: usize,
    /// Minimum attenuation for a path to be considered significant.
    pub min_attenuation: f64,
    /// Model history.
    history: Vec<MultiPathCrosstalkModel>,
    /// Maximum history length.
    max_history: usize,
}

impl MultiPathCrosstalkModeler {
    /// Create a new multi-path modeler.
    pub fn new() -> Self {
        Self {
            max_paths: 50,
            min_attenuation: 0.01,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Model crosstalk paths from a crosstalk matrix.
    /// Each significant off-diagonal entry becomes a path.
    pub fn model_from_matrix(
        &mut self,
        matrix: &CrosstalkMatrix,
    ) -> MultiPathCrosstalkModel {
        let mut paths = Vec::new();
        let mut max_delay = 0usize;

        for i in 0..matrix.num_channels {
            for j in 0..matrix.num_channels {
                if i != j {
                    let attenuation = matrix.matrix[i][j].abs();
                    if attenuation > self.min_attenuation {
                        // Delay derived from attenuation (higher attenuation = longer path).
                        let delay = ((1.0 - attenuation) * 10.0) as usize;
                        let phase_shift = (attenuation * std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI);

                        paths.push(CrosstalkPath {
                            source: j,
                            destination: i,
                            delay,
                            attenuation,
                            phase_shift,
                            active: true,
                        });
                        max_delay = max_delay.max(delay);
                    }
                }
            }
        }

        // Limit number of paths.
        paths.sort_by(|a, b| b.attenuation.partial_cmp(&a.attenuation).unwrap());
        paths.truncate(self.max_paths);

        let quality = if paths.is_empty() {
            1.0
        } else {
            let avg_attenuation: f64 = paths.iter().map(|p| p.attenuation).sum::<f64>()
                / paths.len() as f64;
            (1.0 - avg_attenuation).clamp(0.0, 1.0)
        };

        let model = MultiPathCrosstalkModel {
            num_channels: matrix.num_channels,
            paths,
            max_delay,
            quality,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(model.clone());

        model
    }

    /// Apply multi-path crosstalk to a signal.
    /// Returns the signal with crosstalk paths applied.
    pub fn apply_paths(
        &self,
        model: &MultiPathCrosstalkModel,
        signal: &[f64],
    ) -> Vec<f64> {
        let n = model.num_channels;
        let mut result = vec![0.0f64; n];

        // Direct path.
        for i in 0..n {
            if i < signal.len() {
                result[i] = signal[i];
            }
        }

        // Add crosstalk paths.
        for path in &model.paths {
            if !path.active {
                continue;
            }
            if path.source < signal.len() {
                let contribution = signal[path.source] * path.attenuation;
                if path.destination < result.len() {
                    result[path.destination] += contribution * path.phase_shift.cos();
                }
            }
        }

        result
    }

    /// Returns the model history.
    pub fn history(&self) -> &[MultiPathCrosstalkModel] {
        &self.history
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update modeler parameters.
    pub fn update_params(&mut self, max_paths: usize, min_attenuation: f64) {
        self.max_paths = max_paths.max(1);
        self.min_attenuation = min_attenuation.clamp(0.0, 1.0);
    }
}

impl Default for MultiPathCrosstalkModeler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Crosstalk statistics and distribution
// ============================================================================

/// Statistical distribution of crosstalk coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkDistribution {
    /// Mean crosstalk value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Median value.
    pub median: f64,
    /// 95th percentile.
    pub p95: f64,
    /// Number of samples.
    pub num_samples: usize,
    /// Histogram bins.
    pub histogram: Vec<(f64, usize)>, // (bin_center, count)
}

/// Computes statistical distribution of crosstalk coefficients.
pub struct CrosstalkStatistics {
    /// Number of histogram bins.
    pub num_bins: usize,
    /// Statistics history.
    history: Vec<CrosstalkDistribution>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkStatistics {
    /// Create a new statistics computer.
    pub fn new() -> Self {
        Self {
            num_bins: 20,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Compute distribution of off-diagonal crosstalk coefficients.
    pub fn compute_distribution(
        &mut self,
        matrix: &CrosstalkMatrix,
    ) -> CrosstalkDistribution {
        let mut values: Vec<f64> = Vec::new();
        for i in 0..matrix.num_channels {
            for j in 0..matrix.num_channels {
                if i != j {
                    values.push(matrix.matrix[i][j].abs());
                }
            }
        }

        if values.is_empty() {
            return CrosstalkDistribution {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                p95: 0.0,
                num_samples: 0,
                histogram: Vec::new(),
            };
        }

        let n = values.len();
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[n - 1];
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let p95_idx = ((n as f64) * 0.95) as usize;
        let p95 = sorted[p95_idx.min(n - 1)];

        // Histogram.
        let range = max - min;
        let bin_width = if range > 1e-10 { range / self.num_bins as f64 } else { 1.0 };
        let mut histogram = Vec::with_capacity(self.num_bins);
        for i in 0..self.num_bins {
            let bin_center = min + i as f64 * bin_width + bin_width / 2.0;
            let count = values
                .iter()
                .filter(|&&v| {
                    let bin_low = min + i as f64 * bin_width;
                    let bin_high = bin_low + bin_width;
                    v >= bin_low && v < bin_high
                })
                .count();
            histogram.push((bin_center, count));
        }

        let distribution = CrosstalkDistribution {
            mean,
            std_dev,
            min,
            max,
            median,
            p95,
            num_samples: n,
            histogram,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(distribution.clone());

        distribution
    }

    /// Returns the distribution history.
    pub fn history(&self) -> &[CrosstalkDistribution] {
        &self.history
    }

    /// Returns the average mean crosstalk across history.
    pub fn avg_mean(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|d| d.mean)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the average p95 across history.
    pub fn avg_p95(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|d| d.p95)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for CrosstalkStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Crosstalk mitigation strategies
// ============================================================================

/// Mitigation strategy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationStrategy {
    /// Matrix inversion (exact inverse).
    MatrixInversion,
    /// Regularized inversion (Tikhonov).
    RegularizedInversion,
    /// Truncated SVD (keep only large singular values).
    TruncatedSVD,
    /// Iterative refinement.
    IterativeRefinement,
    /// Threshold-based (zero out small crosstalk).
    Thresholding,
}

/// Result of crosstalk mitigation.
#[derive(Debug, Clone, PartialEq)]
pub struct MitigationResult {
    /// Mitigated signal.
    pub signal: Vec<f64>,
    /// Strategy used.
    pub strategy: MitigationStrategy,
    /// Residual crosstalk level after mitigation.
    pub residual_crosstalk: f64,
    /// Improvement factor (original / residual).
    pub improvement_factor: f64,
    /// Mitigation confidence.
    pub confidence: f64,
}

/// Applies crosstalk mitigation strategies.
pub struct CrosstalkMitigator {
    /// Default strategy.
    pub default_strategy: MitigationStrategy,
    /// Threshold for thresholding strategy.
    pub threshold: f64,
    /// Regularization parameter.
    pub regularization: f64,
    /// Maximum iterations for iterative refinement.
    pub max_iterations: usize,
    /// Mitigation history.
    history: Vec<MitigationResult>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkMitigator {
    /// Create a new mitigator.
    pub fn new() -> Self {
        Self {
            default_strategy: MitigationStrategy::RegularizedInversion,
            threshold: 0.1,
            regularization: 1e-4,
            max_iterations: 50,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Mitigate crosstalk in a measured signal.
    pub fn mitigate(
        &mut self,
        measured: &[f64],
        crosstalk_matrix: &CrosstalkMatrix,
        strategy: MitigationStrategy,
    ) -> XpockResult<MitigationResult> {
        if measured.len() != crosstalk_matrix.num_channels {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (crosstalk_matrix.num_channels, 1),
                    actual: (measured.len(), 1),
                    component: "mitigation_signal".to_string(),
                },
            ));
        }

        let n = crosstalk_matrix.num_channels;
        let original_crosstalk = crosstalk_matrix.max_crosstalk();

        let signal = match strategy {
            MitigationStrategy::MatrixInversion => {
                self.matrix_inversion(measured, crosstalk_matrix)?
            }
            MitigationStrategy::RegularizedInversion => {
                self.regularized_inversion(measured, crosstalk_matrix)?
            }
            MitigationStrategy::TruncatedSVD => {
                self.truncated_svd(measured, crosstalk_matrix)?
            }
            MitigationStrategy::IterativeRefinement => {
                self.iterative_refinement(measured, crosstalk_matrix)?
            }
            MitigationStrategy::Thresholding => {
                self.thresholding(measured, crosstalk_matrix)?
            }
        };

        // Compute residual crosstalk: C * signal - signal (should be ~0 if perfect).
        let mut residual = 0.0f64;
        for i in 0..n {
            let mut reconstructed = 0.0f64;
            for j in 0..n {
                reconstructed += crosstalk_matrix.matrix[i][j] * signal[j];
            }
            residual += (reconstructed - signal[i]).powi(2);
        }
        residual = (residual / n as f64).sqrt();

        let improvement_factor = if residual > 1e-10 {
            original_crosstalk / residual
        } else {
            f64::INFINITY
        };

        let confidence = (1.0 - residual.clamp(0.0, 1.0)).clamp(0.0, 1.0);

        let result = MitigationResult {
            signal,
            strategy,
            residual_crosstalk: residual,
            improvement_factor,
            confidence,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        Ok(result)
    }

    /// Matrix inversion mitigation (Gaussian elimination).
    fn matrix_inversion(
        &self,
        measured: &[f64],
        matrix: &CrosstalkMatrix,
    ) -> XpockResult<Vec<f64>> {
        let n = matrix.num_channels;
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = matrix.matrix[i][j];
            }
            a[i][n] = measured[i];
        }

        for col in 0..n {
            let mut pivot_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    pivot_row = row;
                }
            }
            if max_val < 1e-12 {
                return Ok(measured.to_vec()); // Fallback.
            }
            if pivot_row != col {
                a.swap(col, pivot_row);
            }
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for k in col..=n {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }

        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = a[i][n];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        Ok(x)
    }

    /// Regularized inversion (Tikhonov regularization).
    fn regularized_inversion(
        &self,
        measured: &[f64],
        matrix: &CrosstalkMatrix,
    ) -> XpockResult<Vec<f64>> {
        let n = matrix.num_channels;
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = matrix.matrix[i][j];
            }
            a[i][i] += self.regularization;
            a[i][n] = measured[i];
        }

        for col in 0..n {
            let mut pivot_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    pivot_row = row;
                }
            }
            if max_val < 1e-12 {
                return Ok(measured.to_vec());
            }
            if pivot_row != col {
                a.swap(col, pivot_row);
            }
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for k in col..=n {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }

        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = a[i][n];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        Ok(x)
    }

    /// Truncated SVD mitigation (simplified: zero small eigenvalues).
    fn truncated_svd(
        &self,
        measured: &[f64],
        matrix: &CrosstalkMatrix,
    ) -> XpockResult<Vec<f64>> {
        // Simplified: use regularized inversion with higher regularization
        // as a proxy for truncated SVD.
        let n = matrix.num_channels;
        let lambda = self.regularization * 10.0;
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = matrix.matrix[i][j];
            }
            a[i][i] += lambda;
            a[i][n] = measured[i];
        }

        for col in 0..n {
            let mut pivot_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    pivot_row = row;
                }
            }
            if max_val < 1e-12 {
                return Ok(measured.to_vec());
            }
            if pivot_row != col {
                a.swap(col, pivot_row);
            }
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for k in col..=n {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }

        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = a[i][n];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        Ok(x)
    }

    /// Iterative refinement mitigation.
    fn iterative_refinement(
        &self,
        measured: &[f64],
        matrix: &CrosstalkMatrix,
    ) -> XpockResult<Vec<f64>> {
        let n = matrix.num_channels;
        let mut x = measured.to_vec();

        for _ in 0..self.max_iterations {
            // Compute residual: measured - C * x.
            let mut residual = vec![0.0f64; n];
            for i in 0..n {
                let mut cx = 0.0f64;
                for j in 0..n {
                    cx += matrix.matrix[i][j] * x[j];
                }
                residual[i] = measured[i] - cx;
            }

            // Update: x = x + residual (simple Richardson iteration).
            let mut converged = true;
            for i in 0..n {
                let correction = residual[i] * 0.5; // Damping factor.
                x[i] += correction;
                if correction.abs() > 1e-6 {
                    converged = false;
                }
            }
            if converged {
                break;
            }
        }
        Ok(x)
    }

    /// Thresholding mitigation: zero out crosstalk below threshold.
    fn thresholding(
        &self,
        measured: &[f64],
        matrix: &CrosstalkMatrix,
    ) -> XpockResult<Vec<f64>> {
        let n = matrix.num_channels;
        let mut signal = vec![0.0f64; n];

        for i in 0..n {
            // Keep only the diagonal contribution and strong off-diagonal.
            let mut direct = matrix.matrix[i][i] * measured[i];
            let mut crosstalk_sum = 0.0f64;
            for j in 0..n {
                if i != j {
                    let ct = matrix.matrix[i][j] * measured[j];
                    if ct.abs() > self.threshold {
                        crosstalk_sum += ct;
                    }
                }
            }
            signal[i] = direct - crosstalk_sum;
        }

        Ok(signal)
    }

    /// Returns the mitigation history.
    pub fn history(&self) -> &[MitigationResult] {
        &self.history
    }

    /// Returns the average improvement factor across history.
    pub fn avg_improvement(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.improvement_factor)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update mitigator parameters.
    pub fn update_params(
        &mut self,
        threshold: f64,
        regularization: f64,
        max_iterations: usize,
    ) {
        self.threshold = threshold.clamp(0.0, 1.0);
        self.regularization = regularization.max(0.0);
        self.max_iterations = max_iterations.max(1);
    }
}

impl Default for CrosstalkMitigator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Temporal crosstalk window
// ============================================================================

/// A window of crosstalk measurements over time.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkTemporalWindow {
    /// Window size (number of time steps).
    pub window_size: usize,
    /// Crosstalk values per channel pair over time.
    pub window: Vec<Vec<Vec<f64>>>, // [time_step][source][dest]
    /// Timestamps for each time step.
    pub timestamps: Vec<u64>,
    /// Current write position (circular buffer).
    pub write_pos: usize,
    /// Whether the window is full.
    pub is_full: bool,
}

/// Manages a temporal window of crosstalk measurements for tracking crosstalk
/// evolution over time.
pub struct CrosstalkTemporalWindowManager {
    /// Number of channels.
    num_channels: usize,
    /// Window size.
    window_size: usize,
    /// Current window.
    window: CrosstalkTemporalWindow,
    /// Window history (snapshots).
    history: Vec<CrosstalkTemporalWindow>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkTemporalWindowManager {
    /// Create a new temporal window manager.
    pub fn new(num_channels: usize, window_size: usize) -> Self {
        let window = CrosstalkTemporalWindow {
            window_size,
            window: vec![vec![vec![0.0f64; num_channels]; num_channels]; window_size],
            timestamps: vec![0; window_size],
            write_pos: 0,
            is_full: false,
        };
        Self {
            num_channels,
            window_size,
            window,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Record a crosstalk measurement at the current time step.
    pub fn record(
        &mut self,
        timestamp: u64,
        source: usize,
        dest: usize,
        value: f64,
    ) -> XpockResult<()> {
        if source >= self.num_channels || dest >= self.num_channels {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "channel_index".to_string(),
                    value: format!("({}, {})", source, dest),
                    constraint: format!("both < {}", self.num_channels),
                },
            ));
        }

        self.window.window[self.window.write_pos][source][dest] = value;
        self.window.timestamps[self.window.write_pos] = timestamp;

        self.window.write_pos += 1;
        if self.window.write_pos >= self.window_size {
            self.window.write_pos = 0;
            self.window.is_full = true;
        }

        Ok(())
    }

    /// Get the crosstalk matrix at a specific time offset from now.
    /// offset=0 is the most recent, offset=1 is one step ago, etc.
    pub fn matrix_at_offset(
        &self,
        offset: usize,
    ) -> Option<CrosstalkMatrix> {
        if self.window.timestamps.is_empty() {
            return None;
        }

        let n = self.window_size;
        let idx = if self.window.is_full {
            (self.window.write_pos + n - 1 - offset) % n
        } else {
            if offset >= self.window.write_pos {
                return None;
            }
            self.window.write_pos - 1 - offset
        };

        let mut matrix_data = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        for i in 0..self.num_channels {
            for j in 0..self.num_channels {
                matrix_data[i][j] = self.window.window[idx][i][j];
            }
        }

        CrosstalkMatrix::new(
            self.num_channels,
            matrix_data,
            (0..self.num_channels as i32).collect(),
        )
        .ok()
    }

    /// Get the most recent crosstalk matrix.
    pub fn current_matrix(&self) -> Option<CrosstalkMatrix> {
        self.matrix_at_offset(0)
    }

    /// Compute the average crosstalk matrix over the window.
    pub fn average_matrix(&self) -> Option<CrosstalkMatrix> {
        if self.window.timestamps.is_empty() {
            return None;
        }

        let n = self.window_size;
        let count = if self.window.is_full {
            n
        } else {
            self.window.write_pos
        };
        if count == 0 {
            return None;
        }

        let mut sum = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        let start = if self.window.is_full {
            self.window.write_pos
        } else {
            0
        };

        for step in 0..count {
            let idx = (start + step) % n;
            for i in 0..self.num_channels {
                for j in 0..self.num_channels {
                    sum[i][j] += self.window.window[idx][i][j];
                }
            }
        }

        let mut avg = vec![vec![0.0f64; self.num_channels]; self.num_channels];
        for i in 0..self.num_channels {
            for j in 0..self.num_channels {
                avg[i][j] = sum[i][j] / count as f64;
            }
        }

        CrosstalkMatrix::new(
            self.num_channels,
            avg,
            (0..self.num_channels as i32).collect(),
        )
        .ok()
    }

    /// Compute crosstalk trend: whether crosstalk is increasing or decreasing.
    pub fn crosstalk_trend(&self) -> f64 {
        if self.window.timestamps.is_empty() || self.window.write_pos < 2 {
            return 0.0;
        }

        let n = self.window_size;
        let count = if self.window.is_full { n } else { self.window.write_pos };
        if count < 2 {
            return 0.0;
        }

        // Compute total off-diagonal crosstalk for first and last half.
        let half = count / 2;
        let mut first_half_sum = 0.0f64;
        let mut second_half_sum = 0.0f64;
        let mut first_count = 0usize;
        let mut second_count = 0usize;

        let start = if self.window.is_full {
            self.window.write_pos
        } else {
            0
        };

        for step in 0..count {
            let idx = (start + step) % n;
            for i in 0..self.num_channels {
                for j in 0..self.num_channels {
                    if i != j {
                        let val = self.window.window[idx][i][j].abs();
                        if step < half {
                            first_half_sum += val;
                            first_count += 1;
                        } else {
                            second_half_sum += val;
                            second_count += 1;
                        }
                    }
                }
            }
        }

        if first_count == 0 || second_count == 0 {
            return 0.0;
        }

        let first_avg = first_half_sum / first_count as f64;
        let second_avg = second_half_sum / second_count as f64;

        // Positive = increasing, negative = decreasing.
        second_avg - first_avg
    }

    /// Record a window snapshot in history.
    pub fn snapshot(&mut self) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(self.window.clone());
    }

    /// Returns the window history.
    pub fn history(&self) -> &[CrosstalkTemporalWindow] {
        &self.history
    }

    /// Reset the window manager.
    pub fn reset(&mut self) {
        self.window = CrosstalkTemporalWindow {
            window_size: self.window_size,
            window: vec![vec![vec![0.0f64; self.num_channels]; self.num_channels]; self.window_size],
            timestamps: vec![0; self.window_size],
            write_pos: 0,
            is_full: false,
        };
        self.history.clear();
    }

    /// Update window parameters.
    pub fn update_params(&mut self, window_size: usize) {
        self.window_size = window_size.max(1);
        self.window.write_pos = self.window.write_pos.min(self.window_size - 1);
    }

    /// Returns the number of recorded time steps.
    pub fn recorded_steps(&self) -> usize {
        if self.window.is_full {
            self.window_size
        } else {
            self.window.write_pos
        }
    }
}

impl Default for CrosstalkTemporalWindowManager {
    fn default() -> Self {
        Self::new(2, 10)
    }
}

// ============================================================================
// Crosstalk frequency response
// ============================================================================

/// Frequency response of crosstalk between two channels.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkFrequencyResponse {
    /// Source channel index.
    pub source: usize,
    /// Destination channel index.
    pub destination: usize,
    /// Frequencies (normalized 0.0 to 1.0).
    pub frequencies: Vec<f64>,
    /// Crosstalk magnitude at each frequency.
    pub magnitudes: Vec<f64>,
    /// Phase at each frequency (radians).
    pub phases: Vec<f64>,
}

/// Computes frequency response of crosstalk from time-domain measurements.
pub struct CrosstalkFrequencyResponseComputer {
    /// Number of frequency points.
    pub num_freq_points: usize,
    /// History of responses.
    history: Vec<CrosstalkFrequencyResponse>,
    /// Maximum history length.
    max_history: usize,
}

impl CrosstalkFrequencyResponseComputer {
    /// Create a new frequency response computer.
    pub fn new(num_freq_points: usize) -> Self {
        Self {
            num_freq_points,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Compute frequency response from temporal window data.
    /// Uses a simple DFT-like computation on the crosstalk time series.
    pub fn compute_response(
        &mut self,
        window: &CrosstalkTemporalWindow,
        source: usize,
        dest: usize,
    ) -> XpockResult<CrosstalkFrequencyResponse> {
        if source >= window.window.len() || dest >= window.window[0].len() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "channel_index".to_string(),
                    value: format!("({}, {})", source, dest),
                    constraint: format!("both < {}", window.window.len()),
                },
            ));
        }

        let n = window.window.len();
        let mut frequencies = Vec::with_capacity(self.num_freq_points);
        let mut magnitudes = Vec::with_capacity(self.num_freq_points);
        let mut phases = Vec::with_capacity(self.num_freq_points);

        // Extract time series for this channel pair.
        let mut time_series: Vec<f64> = Vec::with_capacity(n);
        let count = if window.is_full { n } else { window.write_pos };
        let start = if window.is_full { window.write_pos } else { 0 };

        for step in 0..count {
            let idx = (start + step) % n;
            time_series.push(window.window[idx][source][dest]);
        }

        if time_series.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        // Compute DFT at each frequency point.
        for k in 0..self.num_freq_points {
            let freq = k as f64 / self.num_freq_points as f64;
            let mut real = 0.0f64;
            let mut imag = 0.0f64;

            for (t, &val) in time_series.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * freq * t as f64 / time_series.len() as f64;
                real += val * angle.cos();
                imag -= val * angle.sin();
            }

            let magnitude = (real * real + imag * imag).sqrt() / time_series.len() as f64;
            let phase = imag.atan2(real);

            frequencies.push(freq);
            magnitudes.push(magnitude);
            phases.push(phase);
        }

        let response = CrosstalkFrequencyResponse {
            source,
            destination,
            frequencies,
            magnitudes,
            phases,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(response.clone());

        Ok(response)
    }

    /// Find the peak frequency for a channel pair response.
    pub fn peak_frequency(&self, response: &CrosstalkFrequencyResponse) -> f64 {
        let mut max_mag = 0.0f64;
        let mut max_freq = 0.0f64;
        for (i, &mag) in response.magnitudes.iter().enumerate() {
            if mag > max_mag {
                max_mag = mag;
                max_freq = response.frequencies[i];
            }
        }
        max_freq
    }

    /// Returns the response history.
    pub fn history(&self) -> &[CrosstalkFrequencyResponse] {
        &self.history
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update number of frequency points.
    pub fn set_num_freq_points(&mut self, num_points: usize) {
        self.num_freq_points = num_points.max(1);
    }
}

impl Default for CrosstalkFrequencyResponseComputer {
    fn default() -> Self {
        Self::new(16)
    }
}

// ============================================================================
// Crosstalk analysis result type
// ============================================================================

/// Comprehensive crosstalk analysis result combining matrix, statistics,
/// spectrum, and quality assessment.
#[derive(Debug, Clone, PartialEq)]
pub struct CrosstalkAnalysis {
    /// Crosstalk matrix.
    pub matrix: CrosstalkMatrix,
    /// Crosstalk quality assessment.
    pub quality: CrosstalkQuality,
    /// Crosstalk distribution statistics.
    pub distribution: CrosstalkDistribution,
    /// Peak crosstalk magnitude.
    pub peak_magnitude: f64,
    /// Whether crosstalk is within acceptable limits.
    pub within_limits: bool,
    /// Recommended mitigation strategy.
    pub recommended_strategy: MitigationStrategy,
}

/// Computes a comprehensive crosstalk analysis from a crosstalk matrix.
pub struct CrosstalkAnalyzer {
    /// Maximum acceptable crosstalk level.
    pub max_acceptable_crosstalk: f64,
    /// Statistics computer.
    pub stats: CrosstalkStatistics,
    /// Quality assessor.
    pub assessor: CrosstalkQualityAssessor,
}

impl CrosstalkAnalyzer {
    /// Create a new crosstalk analyzer.
    pub fn new(max_acceptable_crosstalk: f64) -> Self {
        Self {
            max_acceptable_crosstalk,
            stats: CrosstalkStatistics::new(),
            assessor: CrosstalkQualityAssessor::new(),
        }
    }

    /// Perform a complete crosstalk analysis.
    pub fn analyze(&mut self, matrix: &CrosstalkMatrix) -> CrosstalkAnalysis {
        let quality = self.assessor.assess(matrix);
        let distribution = self.stats.compute_distribution(matrix);
        let peak_magnitude = matrix.max_crosstalk();
        let within_limits = peak_magnitude <= self.max_acceptable_crosstalk;

        let recommended_strategy = if peak_magnitude < 0.05 {
            MitigationStrategy::Thresholding
        } else if peak_magnitude < 0.15 {
            MitigationStrategy::RegularizedInversion
        } else if peak_magnitude < 0.3 {
            MitigationStrategy::TruncatedSVD
        } else {
            MitigationStrategy::IterativeRefinement
        };

        CrosstalkAnalysis {
            matrix: matrix.clone(),
            quality,
            distribution,
            peak_magnitude,
            within_limits,
            recommended_strategy,
        }
    }

    /// Returns the analysis history (distributions).
    pub fn distribution_history(&self) -> &[CrosstalkDistribution] {
        self.stats.history()
    }

    /// Returns the quality history.
    pub fn quality_history(&self) -> &[CrosstalkQuality] {
        self.assessor.history()
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.stats.reset();
        self.assessor.reset();
    }

    /// Update the maximum acceptable crosstalk.
    pub fn set_max_acceptable_crosstalk(&mut self, max: f64) {
        self.max_acceptable_crosstalk = max.clamp(0.0, 1.0);
    }
}

impl Default for CrosstalkAnalyzer {
    fn default() -> Self {
        Self::new(0.15)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ---------------------------------------------------------------------------
    // CrosstalkMatrix tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_matrix_identity() {
        let matrix = CrosstalkMatrix::identity(3, vec![-1, 0, 1]);
        assert_eq!(matrix.num_channels, 3);
        assert_eq!(matrix.channel_labels, vec![-1, 0, 1]);
        // Identity: diagonal = 1, off-diagonal = 0.
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((matrix.matrix[i][j] - 1.0).abs() < 0.001);
                } else {
                    assert_eq!(matrix.matrix[i][j], 0.0);
                }
            }
        }
        assert!((matrix.confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_new() {
        let matrix = vec![
            vec![0.9f64, 0.1, 0.0],
            vec![0.05, 0.9, 0.05],
            vec![0.0, 0.1, 0.9],
        ];
        let result = CrosstalkMatrix::new(3, matrix.clone(), vec![-1, 0, 1]);
        assert!(result.is_ok());
        let m = result.unwrap();
        assert_eq!(m.num_channels, 3);
        assert_eq!(m.matrix, matrix);
    }

    #[test]
    fn crosstalk_matrix_wrong_dimensions() {
        let matrix = vec![vec![1.0f64, 0.0]; 2]; // 2x2 but claiming 3 channels.
        let result = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn crosstalk_matrix_wrong_labels() {
        let matrix = vec![vec![1.0f64, 0.0]; 2];
        let result = CrosstalkMatrix::new(2, matrix, vec![-1]); // Only 1 label for 2 channels.
        assert!(result.is_err());
    }

    #[test]
    fn crosstalk_matrix_crosstalk_accessor() {
        let matrix = vec![
            vec![0.9f64, 0.1, 0.0],
            vec![0.05, 0.9, 0.05],
            vec![0.0, 0.1, 0.9],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        // from channel 0 to channel 1: matrix[1][0] = 0.05.
        assert!((m.crosstalk(0, 1) - 0.05).abs() < 0.001);
        // from channel 1 to channel 0: matrix[0][1] = 0.1.
        assert!((m.crosstalk(1, 0) - 0.1).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_total_into() {
        let matrix = vec![
            vec![0.9f64, 0.1, 0.0],
            vec![0.05, 0.9, 0.05],
            vec![0.0, 0.1, 0.9],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        // Channel 0 receives crosstalk from channel 1: matrix[0][1] = 0.1.
        assert!((m.total_crosstalk_into(0) - 0.1).abs() < 0.001);
        // Channel 1 receives from 0 and 2: 0.05 + 0.05 = 0.1.
        assert!((m.total_crosstalk_into(1) - 0.1).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_total_out_of() {
        let matrix = vec![
            vec![0.9f64, 0.1, 0.0],
            vec![0.05, 0.9, 0.05],
            vec![0.0, 0.1, 0.9],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        // Channel 0 leaks to channel 1: matrix[1][0] = 0.05.
        assert!((m.total_crosstalk_out_of(0) - 0.05).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_max_crosstalk() {
        let matrix = vec![
            vec![0.9f64, 0.1, 0.0],
            vec![0.05, 0.9, 0.2],
            vec![0.0, 0.1, 0.9],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        assert!((m.max_crosstalk() - 0.2).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_diagonally_dominant() {
        let matrix = vec![
            vec![0.9f64, 0.05, 0.0],
            vec![0.05, 0.9, 0.05],
            vec![0.0, 0.05, 0.9],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        assert!(m.is_diagonally_dominant(2.0));
    }

    #[test]
    fn crosstalk_matrix_not_dominant() {
        let matrix = vec![
            vec![0.4f64, 0.4, 0.2],
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.4, 0.4],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        assert!(!m.is_diagonally_dominant(2.0));
    }

    #[test]
    fn crosstalk_matrix_condition_number() {
        let matrix = vec![
            vec![2.0f64, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.5],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        // max_diag/min_diag = 2.0/0.5 = 4.0.
        assert!((m.condition_number_estimate() - 4.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_worst_best_channel() {
        let matrix = vec![
            vec![0.9f64, 0.2, 0.0],
            vec![0.05, 0.9, 0.0],
            vec![0.0, 0.0, 0.9],
        ];
        let m = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        let (worst_idx, worst_val) = m.worst_channel();
        assert_eq!(worst_idx, 1); // Channel 1 receives 0.2 from channel 0.
        assert!((worst_val - 0.2).abs() < 0.001);
        let (best_idx, best_val) = m.best_channel();
        assert_eq!(best_idx, 2); // Channel 2 has no crosstalk.
        assert_eq!(best_val, 0.0);
    }

    #[test]
    fn crosstalk_matrix_default() {
        let m = CrosstalkMatrix::default();
        assert_eq!(m.num_channels, 2);
        assert_eq!(m.channel_labels, vec![-1, 1]);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkMeasurementCollector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn measurement_collector_new() {
        let collector = CrosstalkMeasurementCollector::new(3);
        assert_eq!(collector.num_channels, 3);
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn measurement_collector_record() {
        let mut collector = CrosstalkMeasurementCollector::new(2);
        assert!(collector.record(0, 0, 1.0).is_ok());
        assert!(collector.record(0, 1, 0.1).is_ok());
        assert!(collector.record(1, 0, 0.05).is_ok());
        assert!(collector.record(1, 1, 1.0).is_ok());
        assert_eq!(collector.count(), 4);
    }

    #[test]
    fn measurement_collector_record_invalid_channel() {
        let mut collector = CrosstalkMeasurementCollector::new(2);
        assert!(collector.record(0, 2, 1.0).is_err()); // Channel 2 doesn't exist.
        assert!(collector.record(2, 0, 1.0).is_err());
    }

    #[test]
    fn measurement_collector_estimate_empty() {
        let collector = CrosstalkMeasurementCollector::new(2);
        let matrix = collector.estimate_matrix(&[-1, 1]).unwrap();
        // Empty -> identity matrix.
        assert_eq!(matrix.matrix[0][0], 1.0);
        assert_eq!(matrix.matrix[0][1], 0.0);
    }

    #[test]
    fn measurement_collector_estimate_from_data() {
        let mut collector = CrosstalkMeasurementCollector::new(2);
        // Channel 0 transmitted, mostly received at 0.
        for _ in 0..10 {
            collector.record(0, 0, 1.0).unwrap();
            collector.record(0, 1, 0.1).unwrap();
        }
        // Channel 1 transmitted, mostly received at 1.
        for _ in 0..10 {
            collector.record(1, 0, 0.05).unwrap();
            collector.record(1, 1, 1.0).unwrap();
        }
        let matrix = collector.estimate_matrix(&[-1, 1]).unwrap();
        assert_eq!(matrix.num_channels, 2);
        // Column 0 (channel 0): received mostly at 0 -> diagonal dominant.
        assert!(matrix.matrix[0][0] > matrix.matrix[1][0]);
    }

    #[test]
    fn measurement_collector_clear() {
        let mut collector = CrosstalkMeasurementCollector::new(2);
        collector.record(0, 0, 1.0).unwrap();
        assert_eq!(collector.count(), 1);
        collector.clear();
        assert_eq!(collector.count(), 0);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkCorrector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_corrector_new() {
        let corrector = CrosstalkCorrector::new();
        assert!((corrector.regularization - 1e-6).abs() < 1e-12);
        assert_eq!(corrector.max_condition_number, 100.0);
    }

    #[test]
    fn crosstalk_corrector_identity_matrix() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let measured = vec![1.0, 0.5];
        let result = corrector.correct(&measured, &matrix).unwrap();
        // Identity matrix: corrected = measured.
        assert!((result.corrected_signal[0] - 1.0).abs() < 0.001);
        assert!((result.corrected_signal[1] - 0.5).abs() < 0.001);
        assert!((result.residual_error - 0.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_corrector_with_crosstalk() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = vec![
            vec![0.9f64, 0.1],
            vec![0.05, 0.9],
        ];
        let matrix = CrosstalkMatrix::new(2, matrix, vec![-1, 1]).unwrap();
        // true_signal = [1.0, 0.5], measured = C * true_signal = [0.95, 0.5].
        let measured = vec![0.95, 0.5];
        let result = corrector.correct(&measured, &matrix).unwrap();
        assert_eq!(result.corrected_signal.len(), 2);
        assert!(result.corrected_signal[0] > 0.5);
        assert!(result.corrected_confidence > 0.0);
    }

    #[test]
    fn crosstalk_corrector_wrong_dimensions() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let measured = vec![1.0, 0.5, 0.3]; // 3 values for 2-channel matrix.
        let result = corrector.correct(&measured, &matrix);
        assert!(result.is_err());
    }

    #[test]
    fn crosstalk_corrector_batch() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let signals = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let results = corrector.correct_batch(&signals, &matrix).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].corrected_confidence > 0.0);
    }

    #[test]
    fn crosstalk_corrector_avg_confidence() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        corrector.correct(&vec![1.0, 0.0], &matrix).unwrap();
        corrector.correct(&vec![0.0, 1.0], &matrix).unwrap();
        assert!(corrector.avg_confidence() > 0.0);
    }

    #[test]
    fn crosstalk_corrector_avg_residual() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        corrector.correct(&vec![1.0, 0.0], &matrix).unwrap();
        assert!((corrector.avg_residual() - 0.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_corrector_reset() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        corrector.correct(&vec![1.0, 0.0], &matrix).unwrap();
        assert!(!corrector.history.is_empty());
        corrector.reset();
        assert_eq!(corrector.history.len(), 0);
    }

    #[test]
    fn crosstalk_corrector_update_params() {
        let mut corrector = CrosstalkCorrector::new();
        corrector.update_params(1e-4, 50.0);
        assert!((corrector.regularization - 1e-4).abs() < 1e-12);
        assert_eq!(corrector.max_condition_number, 50.0);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkCalibrator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_calibrator_new() {
        let calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        assert_eq!(calibrator.num_channels, 2);
        assert_eq!(calibrator.channel_labels, vec![-1, 1]);
        assert_eq!(calibrator.measurement_count(), 0);
    }

    #[test]
    fn crosstalk_calibrator_record_and_calibrate() {
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        calibrator.record_calibration(0, 0, 1.0).unwrap();
        calibrator.record_calibration(0, 1, 0.1).unwrap();
        calibrator.record_calibration(1, 0, 0.05).unwrap();
        calibrator.record_calibration(1, 1, 1.0).unwrap();
        let result = calibrator.calibrate().unwrap();
        assert_eq!(result.num_measurements, 4);
        assert!(result.quality >= 0.0);
        assert!(result.quality <= 1.0);
        assert_eq!(result.matrix.num_channels, 2);
    }

    #[test]
    fn crosstalk_calibrator_latest() {
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        assert!(calibrator.latest_calibration().is_none());
        calibrator.record_calibration(0, 0, 1.0).unwrap();
        calibrator.calibrate().unwrap();
        assert!(calibrator.latest_calibration().is_some());
    }

    #[test]
    fn crosstalk_calibrator_history() {
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        calibrator.record_calibration(0, 0, 1.0).unwrap();
        calibrator.calibrate().unwrap();
        calibrator.record_calibration(0, 0, 1.0).unwrap();
        calibrator.calibrate().unwrap();
        assert_eq!(calibrator.history().len(), 2);
    }

    #[test]
    fn crosstalk_calibrator_clear() {
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        calibrator.record_calibration(0, 0, 1.0).unwrap();
        calibrator.calibrate().unwrap();
        assert!(!calibrator.history().is_empty());
        calibrator.clear();
        assert_eq!(calibrator.history().len(), 0);
        assert_eq!(calibrator.measurement_count(), 0);
    }

    #[test]
    fn crosstalk_calibrator_update_labels() {
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        assert!(calibrator.update_channel_labels(vec![0, 1]).is_ok());
        assert_eq!(calibrator.channel_labels, vec![0, 1]);
        // Wrong number of labels.
        assert!(calibrator.update_channel_labels(vec![0]).is_err());
    }

    // ---------------------------------------------------------------------------
    // CrosstalkCoefficient tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_coefficient_aggregator_new() {
        let aggregator = CrosstalkCoefficientAggregator::new(3);
        assert_eq!(aggregator.num_channels, 3);
        assert_eq!(aggregator.total_measurements(), 0);
    }

    #[test]
    fn crosstalk_coefficient_add_and_aggregate() {
        let mut aggregator = CrosstalkCoefficientAggregator::new(2);
        aggregator.add(0, 0, 0.9);
        aggregator.add(0, 0, 1.0);
        aggregator.add(0, 0, 1.1);
        aggregator.add(0, 1, 0.1);
        aggregator.add(0, 1, 0.2);
        let coeff = aggregator.aggregate(0, 0).unwrap();
        assert!((coeff.coefficient - 1.0).abs() < 0.001);
        assert_eq!(coeff.num_samples, 3);
        assert!(coeff.uncertainty >= 0.0);
        let coeff01 = aggregator.aggregate(0, 1).unwrap();
        assert!((coeff01.coefficient - 0.15).abs() < 0.001);
    }

    #[test]
    fn crosstalk_coefficient_aggregate_none() {
        let aggregator = CrosstalkCoefficientAggregator::new(2);
        assert!(aggregator.aggregate(0, 0).is_none());
    }

    #[test]
    fn crosstalk_coefficient_build_matrix() {
        let mut aggregator = CrosstalkCoefficientAggregator::new(2);
        aggregator.add(0, 0, 0.9);
        aggregator.add(0, 1, 0.1);
        aggregator.add(1, 0, 0.05);
        aggregator.add(1, 1, 0.9);
        let matrix = aggregator.build_matrix(&[-1, 1]);
        assert_eq!(matrix.num_channels, 2);
        assert!((matrix.matrix[0][0] - 0.9).abs() < 0.001);
    }

    #[test]
    fn crosstalk_coefficient_clear() {
        let mut aggregator = CrosstalkCoefficientAggregator::new(2);
        aggregator.add(0, 0, 1.0);
        assert_eq!(aggregator.total_measurements(), 1);
        aggregator.clear();
        assert_eq!(aggregator.total_measurements(), 0);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkQualityAssessor tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_quality_assessor_new() {
        let assessor = CrosstalkQualityAssessor::new();
        assert!((assessor.crosstalk_threshold - 0.15).abs() < 0.001);
    }

    #[test]
    fn crosstalk_quality_assess_identity() {
        let mut assessor = CrosstalkQualityAssessor::new();
        let matrix = CrosstalkMatrix::identity(3, vec![-1, 0, 1]);
        let quality = assessor.assess(&matrix);
        assert!((quality.overall_level - 0.0).abs() < 0.001);
        assert!((quality.quality_score - 1.0).abs() < 0.001);
        assert_eq!(quality.channels_above_threshold, 0);
    }

    #[test]
    fn crosstalk_quality_assess_high_crosstalk() {
        let mut assessor = CrosstalkQualityAssessor::new();
        let matrix = vec![
            vec![0.5f64, 0.3, 0.2],
            vec![0.3, 0.5, 0.2],
            vec![0.2, 0.3, 0.5],
        ];
        let matrix = CrosstalkMatrix::new(3, matrix, vec![-1, 0, 1]).unwrap();
        let quality = assessor.assess(&matrix);
        assert!(quality.overall_level > 0.0);
        assert!(quality.quality_score < 1.0);
        assert!(quality.channels_above_threshold > 0);
    }

    #[test]
    fn crosstalk_quality_is_acceptable() {
        let assessor = CrosstalkQualityAssessor::new();
        let good = CrosstalkQuality {
            overall_level: 0.05,
            worst_channel_crosstalk: 0.1,
            avg_off_diagonal: 0.05,
            diagonal_dominance: 0.9,
            channels_above_threshold: 0,
            quality_score: 0.9,
        };
        assert!(assessor.is_acceptable(&good));
        let bad = CrosstalkQuality {
            overall_level: 0.5,
            worst_channel_crosstalk: 0.5,
            avg_off_diagonal: 0.5,
            diagonal_dominance: 0.2,
            channels_above_threshold: 3,
            quality_score: 0.3,
        };
        assert!(!assessor.is_acceptable(&bad));
    }

    #[test]
    fn crosstalk_quality_avg_score() {
        let mut assessor = CrosstalkQualityAssessor::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        assessor.assess(&matrix);
        assessor.assess(&matrix);
        assert!((assessor.avg_quality_score() - 1.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_quality_pass_rate() {
        let mut assessor = CrosstalkQualityAssessor::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        assessor.assess(&matrix);
        assert!((assessor.pass_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_quality_reset() {
        let mut assessor = CrosstalkQualityAssessor::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        assessor.assess(&matrix);
        assert!(!assessor.history.is_empty());
        assessor.reset();
        assert_eq!(assessor.history.len(), 0);
    }

    #[test]
    fn crosstalk_quality_set_threshold() {
        let mut assessor = CrosstalkQualityAssessor::new();
        assessor.set_threshold(0.3);
        assert!((assessor.crosstalk_threshold - 0.3).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkQuality tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_quality_default_fields() {
        let quality = CrosstalkQuality {
            overall_level: 0.1,
            worst_channel_crosstalk: 0.15,
            avg_off_diagonal: 0.05,
            diagonal_dominance: 0.8,
            channels_above_threshold: 1,
            quality_score: 0.85,
        };
        assert!((quality.overall_level - 0.1).abs() < 0.001);
        assert!(quality.quality_score >= 0.0);
        assert!(quality.quality_score <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // Edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_matrix_single_channel() {
        let matrix = vec![vec![1.0f64]];
        let m = CrosstalkMatrix::new(1, matrix, vec![0]).unwrap();
        assert_eq!(m.num_channels, 1);
        assert_eq!(m.max_crosstalk(), 0.0);
        assert!(m.is_diagonally_dominant(2.0));
    }

    #[test]
    fn crosstalk_corrector_single_channel() {
        let mut corrector = CrosstalkCorrector::new();
        let matrix = CrosstalkMatrix::identity(1, vec![0]);
        let result = corrector.correct(&vec![5.0], &matrix).unwrap();
        assert!((result.corrected_signal[0] - 5.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_matrix_zero_diagonal() {
        let matrix = vec![
            vec![0.0f64, 0.5],
            vec![0.5, 0.0],
        ];
        let m = CrosstalkMatrix::new(2, matrix, vec![-1, 1]).unwrap();
        assert!(!m.is_diagonally_dominant(2.0));
        assert_eq!(m.condition_number_estimate(), f64::INFINITY);
    }

    #[test]
    fn crosstalk_pipeline_end_to_end() {
        // Full pipeline: collect measurements -> calibrate -> correct -> assess.
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        for _ in 0..20 {
            calibrator.record_calibration(0, 0, 1.0).unwrap();
            calibrator.record_calibration(0, 1, 0.1).unwrap();
            calibrator.record_calibration(1, 0, 0.05).unwrap();
            calibrator.record_calibration(1, 1, 1.0).unwrap();
        }
        let calibration = calibrator.calibrate().unwrap();
        assert!(calibration.quality > 0.0);

        let mut corrector = CrosstalkCorrector::new();
        let measured = vec![0.95, 0.55];
        let result = corrector.correct(&measured, &calibration.matrix).unwrap();
        assert_eq!(result.corrected_signal.len(), 2);

        let mut assessor = CrosstalkQualityAssessor::new();
        let quality = assessor.assess(&calibration.matrix);
        assert!(quality.quality_score >= 0.0);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkSpectrumAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn spectrum_analyzer_new() {
        let analyzer = CrosstalkSpectrumAnalyzer::default();
        assert_eq!(analyzer.num_bins, 16);
        assert_eq!(analyzer.frequency_range, (0.0, 1.0));
    }

    #[test]
    fn spectrum_analyzer_analyze_empty() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(4, (0.0, 1.0));
        let result = analyzer.analyze(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn spectrum_analyzer_analyze_single_bin() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(4, (0.0, 1.0));
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let result = analyzer.analyze(&[matrix]).unwrap();
        assert_eq!(result.bins.len(), 1);
        assert_eq!(result.num_channels, 2);
        assert_eq!(result.peak_magnitude, 0.0);
        assert_eq!(result.peak_frequency, 0.0);
    }

    #[test]
    fn spectrum_analyzer_analyze_multiple_bins() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(4, (0.0, 1.0));
        let m1 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m2 = CrosstalkMatrix::new(
            2,
            vec![vec![0.8f64, 0.2], vec![0.1, 0.8]],
            vec![-1, 1],
        )
        .unwrap();
        let m3 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m4 = CrosstalkMatrix::new(
            2,
            vec![vec![0.7f64, 0.3], vec![0.2, 0.7]],
            vec![-1, 1],
        )
        .unwrap();
        let result = analyzer.analyze(&[m1, m2, m3, m4]).unwrap();
        assert_eq!(result.bins.len(), 4);
        assert!(result.peak_magnitude > 0.0);
        assert!(result.peak_frequency >= 0.0);
        assert!(result.peak_frequency <= 1.0);
    }

    #[test]
    fn spectrum_analyzer_magnitude_at_frequency() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(4, (0.0, 1.0));
        let m1 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m2 = CrosstalkMatrix::new(
            2,
            vec![vec![0.8f64, 0.2], vec![0.1, 0.8]],
            vec![-1, 1],
        )
        .unwrap();
        let result = analyzer.analyze(&[m1, m2]).unwrap();
        let mag_at_0 = analyzer.magnitude_at_frequency(&result, 0.0);
        let mag_at_1 = analyzer.magnitude_at_frequency(&result, 1.0);
        assert!(mag_at_0 >= 0.0);
        assert!(mag_at_1 >= 0.0);
    }

    #[test]
    fn spectrum_analyzer_avg_magnitude() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(2, (0.0, 1.0));
        let m1 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m2 = CrosstalkMatrix::new(
            2,
            vec![vec![0.8f64, 0.2], vec![0.1, 0.8]],
            vec![-1, 1],
        )
        .unwrap();
        let result = analyzer.analyze(&[m1, m2]).unwrap();
        let avg = analyzer.avg_magnitude(&result);
        assert!(avg >= 0.0);
        assert!(avg <= 1.0);
    }

    #[test]
    fn spectrum_analyzer_peak_frequency() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(4, (0.0, 1.0));
        let m1 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m2 = CrosstalkMatrix::new(
            2,
            vec![vec![0.8f64, 0.2], vec![0.1, 0.8]],
            vec![-1, 1],
        )
        .unwrap();
        let result = analyzer.analyze(&[m1, m2]).unwrap();
        let peak = analyzer.peak_crosstalk_frequency(&result);
        assert!(peak >= 0.0);
        assert!(peak <= 1.0);
    }

    #[test]
    fn spectrum_analyzer_reset() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(2, (0.0, 1.0));
        let m = CrosstalkMatrix::identity(2, vec![-1, 1]);
        analyzer.analyze(&[m]).unwrap();
        assert!(!analyzer.history().is_empty());
        analyzer.reset();
        assert_eq!(analyzer.history().len(), 0);
    }

    // ---------------------------------------------------------------------------
    // AdaptiveCrosstalkCanceller tests
    // ---------------------------------------------------------------------------

    #[test]
    fn adaptive_canceller_new() {
        let canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        assert_eq!(canceller.adaptation_count(), 0);
        assert!((canceller.step_size - 0.01).abs() < 1e-12);
        assert_eq!(canceller.history().len(), 0);
    }

    #[test]
    fn adaptive_canceller_adapt_identity() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        let measured = vec![1.0, 0.5];
        let expected = vec![1.0, 0.5];
        let result = canceller.adapt(&measured, &expected).unwrap();
        assert_eq!(result.num_channels, 2);
        assert_eq!(canceller.adaptation_count(), 1);
    }

    #[test]
    fn adaptive_canceller_adapt_wrong_dimensions() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        let measured = vec![1.0, 0.5, 0.3];
        let expected = vec![1.0, 0.5];
        let result = canceller.adapt(&measured, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn adaptive_canceller_multiple_adaptations() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        for _ in 0..5 {
            canceller
                .adapt(&vec![1.0, 0.0], &vec![1.0, 0.0])
                .unwrap();
        }
        assert_eq!(canceller.adaptation_count(), 5);
        assert_eq!(canceller.history().len(), 5);
    }

    #[test]
    fn adaptive_canceller_reset() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        canceller.adapt(&vec![1.0, 0.0], &vec![1.0, 0.0]).unwrap();
        assert!(!canceller.history().is_empty());
        canceller.reset();
        assert_eq!(canceller.history().len(), 0);
        assert_eq!(canceller.adaptation_count(), 0);
    }

    #[test]
    fn adaptive_canceller_set_step_size() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        canceller.set_step_size(0.5);
        assert!((canceller.step_size - 0.5).abs() < 1e-12);
        canceller.set_step_size(0.0); // Should be clamped to minimum.
        assert!(canceller.step_size >= 1e-6);
    }

    #[test]
    fn adaptive_canceller_current_matrix() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(2, vec![-1, 1]);
        let matrix = canceller.current_matrix();
        assert_eq!(matrix.num_channels, 2);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkPath tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_path_creation() {
        let path = CrosstalkPath {
            source: 0,
            destination: 1,
            delay: 3,
            attenuation: 0.5,
            phase_shift: std::f64::consts::PI / 4.0,
            active: true,
        };
        assert_eq!(path.source, 0);
        assert_eq!(path.destination, 1);
        assert_eq!(path.delay, 3);
        assert!((path.attenuation - 0.5).abs() < 0.001);
        assert!(path.active);
    }

    // ---------------------------------------------------------------------------
    // MultiPathCrosstalkModeler tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_path_modeler_new() {
        let modeler = MultiPathCrosstalkModeler::new();
        assert_eq!(modeler.max_paths, 50);
        assert!((modeler.min_attenuation - 0.01).abs() < 0.001);
    }

    #[test]
    fn multi_path_model_from_matrix_identity() {
        let mut modeler = MultiPathCrosstalkModeler::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let model = modeler.model_from_matrix(&matrix);
        assert_eq!(model.num_channels, 2);
        assert_eq!(model.paths.len(), 0); // No off-diagonal crosstalk.
        assert_eq!(model.max_delay, 0);
        assert!((model.quality - 1.0).abs() < 0.001);
    }

    #[test]
    fn multi_path_model_from_matrix_with_crosstalk() {
        let mut modeler = MultiPathCrosstalkModeler::new();
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.2], vec![0.1, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        let model = modeler.model_from_matrix(&matrix);
        assert!(!model.paths.is_empty());
        assert!(model.quality >= 0.0);
        assert!(model.quality <= 1.0);
    }

    #[test]
    fn multi_path_apply_paths() {
        let mut modeler = MultiPathCrosstalkModeler::new();
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.2], vec![0.1, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        let model = modeler.model_from_matrix(&matrix);
        let signal = vec![1.0, 0.0];
        let result = modeler.apply_paths(&model, &signal);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn multi_path_modeler_update_params() {
        let mut modeler = MultiPathCrosstalkModeler::new();
        modeler.update_params(10, 0.05);
        assert_eq!(modeler.max_paths, 10);
        assert!((modeler.min_attenuation - 0.05).abs() < 0.001);
    }

    #[test]
    fn multi_path_modeler_reset() {
        let mut modeler = MultiPathCrosstalkModeler::new();
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.2], vec![0.1, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        modeler.model_from_matrix(&matrix);
        assert!(!modeler.history().is_empty());
        modeler.reset();
        assert_eq!(modeler.history().len(), 0);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkDistribution tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_distribution_fields() {
        let dist = CrosstalkDistribution {
            mean: 0.1,
            std_dev: 0.05,
            min: 0.0,
            max: 0.2,
            median: 0.1,
            p95: 0.18,
            num_samples: 100,
            histogram: vec![(0.05, 50), (0.15, 50)],
        };
        assert!((dist.mean - 0.1).abs() < 0.001);
        assert_eq!(dist.num_samples, 100);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkStatistics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_statistics_new() {
        let stats = CrosstalkStatistics::new();
        assert_eq!(stats.num_bins, 20);
        assert_eq!(stats.history.len(), 0);
    }

    #[test]
    fn crosstalk_statistics_compute_identity() {
        let mut stats = CrosstalkStatistics::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let dist = stats.compute_distribution(&matrix);
        assert_eq!(dist.num_samples, 2); // Two off-diagonal entries.
        assert!((dist.mean - 0.0).abs() < 0.001);
        assert!((dist.max - 0.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_statistics_compute_with_crosstalk() {
        let mut stats = CrosstalkStatistics::new();
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.2], vec![0.1, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        let dist = stats.compute_distribution(&matrix);
        assert_eq!(dist.num_samples, 2);
        assert!(dist.mean > 0.0);
        assert!(dist.max > 0.0);
        assert!(dist.histogram.len() <= 20);
    }

    #[test]
    fn crosstalk_statistics_avg_mean() {
        let mut stats = CrosstalkStatistics::new();
        let m1 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m2 = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.1], vec![0.1, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        stats.compute_distribution(&m1);
        stats.compute_distribution(&m2);
        let avg = stats.avg_mean();
        assert!(avg >= 0.0);
    }

    #[test]
    fn crosstalk_statistics_avg_p95() {
        let mut stats = CrosstalkStatistics::new();
        let m = CrosstalkMatrix::identity(2, vec![-1, 1]);
        stats.compute_distribution(&m);
        let avg = stats.avg_p95();
        assert!(avg >= 0.0);
    }

    #[test]
    fn crosstalk_statistics_reset() {
        let mut stats = CrosstalkStatistics::new();
        let m = CrosstalkMatrix::identity(2, vec![-1, 1]);
        stats.compute_distribution(&m);
        assert!(!stats.history().is_empty());
        stats.reset();
        assert_eq!(stats.history().len(), 0);
    }

    // ---------------------------------------------------------------------------
    // MitigationStrategy tests
    // ---------------------------------------------------------------------------

    #[test]
    fn mitigation_strategy_variants() {
        assert_eq!(MitigationStrategy::MatrixInversion as u32, 0);
        assert_eq!(MitigationStrategy::RegularizedInversion as u32, 1);
        assert_eq!(MitigationStrategy::TruncatedSVD as u32, 2);
        assert_eq!(MitigationStrategy::IterativeRefinement as u32, 3);
        assert_eq!(MitigationStrategy::Thresholding as u32, 4);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkMitigator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_mitigator_new() {
        let mitigator = CrosstalkMitigator::new();
        assert_eq!(
            mitigator.default_strategy,
            MitigationStrategy::RegularizedInversion
        );
        assert!((mitigator.threshold - 0.1).abs() < 0.001);
        assert_eq!(mitigator.history.len(), 0);
    }

    #[test]
    fn crosstalk_mitigator_mitigate_identity() {
        let mut mitigator = CrosstalkMitigator::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let measured = vec![1.0, 0.5];
        let result = mitigator
            .mitigate(&measured, &matrix, MitigationStrategy::MatrixInversion)
            .unwrap();
        assert_eq!(result.signal.len(), 2);
        assert!((result.signal[0] - 1.0).abs() < 0.001);
        assert!((result.signal[1] - 0.5).abs() < 0.001);
        assert!((result.residual_crosstalk - 0.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_mitigator_wrong_dimensions() {
        let mut mitigator = CrosstalkMitigator::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let measured = vec![1.0, 0.5, 0.3];
        let result = mitigator.mitigate(
            &measured,
            &matrix,
            MitigationStrategy::MatrixInversion,
        );
        assert!(result.is_err());
    }

    #[test]
    fn crosstalk_mitigator_all_strategies() {
        let mut mitigator = CrosstalkMitigator::new();
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.1], vec![0.05, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        let measured = vec![0.95, 0.5];

        for strategy in [
            MitigationStrategy::MatrixInversion,
            MitigationStrategy::RegularizedInversion,
            MitigationStrategy::TruncatedSVD,
            MitigationStrategy::IterativeRefinement,
            MitigationStrategy::Thresholding,
        ] {
            let result = mitigator.mitigate(&measured, &matrix, strategy).unwrap();
            assert_eq!(result.signal.len(), 2);
            assert!(result.confidence >= 0.0);
            assert!(result.confidence <= 1.0);
        }
    }

    #[test]
    fn crosstalk_mitigator_avg_improvement() {
        let mut mitigator = CrosstalkMitigator::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        mitigator
            .mitigate(&vec![1.0, 0.0], &matrix, MitigationStrategy::MatrixInversion)
            .unwrap();
        let avg = mitigator.avg_improvement();
        assert!(avg >= 0.0);
    }

    #[test]
    fn crosstalk_mitigator_reset() {
        let mut mitigator = CrosstalkMitigator::new();
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        mitigator
            .mitigate(&vec![1.0, 0.0], &matrix, MitigationStrategy::MatrixInversion)
            .unwrap();
        assert!(!mitigator.history().is_empty());
        mitigator.reset();
        assert_eq!(mitigator.history().len(), 0);
    }

    #[test]
    fn crosstalk_mitigator_update_params() {
        let mut mitigator = CrosstalkMitigator::new();
        mitigator.update_params(0.2, 1e-3, 100);
        assert!((mitigator.threshold - 0.2).abs() < 0.001);
        assert!((mitigator.regularization - 1e-3).abs() < 1e-12);
        assert_eq!(mitigator.max_iterations, 100);
    }

    // ---------------------------------------------------------------------------
    // MitigationResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn mitigation_result_fields() {
        let result = MitigationResult {
            signal: vec![1.0, 0.0],
            strategy: MitigationStrategy::MatrixInversion,
            residual_crosstalk: 0.01,
            improvement_factor: 10.0,
            confidence: 0.95,
        };
        assert_eq!(result.signal.len(), 2);
        assert!((result.residual_crosstalk - 0.01).abs() < 0.001);
        assert!((result.improvement_factor - 10.0).abs() < 0.001);
        assert!((result.confidence - 0.95).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // Edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_mitigator_single_channel() {
        let mut mitigator = CrosstalkMitigator::new();
        let matrix = CrosstalkMatrix::identity(1, vec![0]);
        let result = mitigator
            .mitigate(&vec![5.0], &matrix, MitigationStrategy::MatrixInversion)
            .unwrap();
        assert!((result.signal[0] - 5.0).abs() < 0.001);
    }

    #[test]
    fn adaptive_canceller_single_channel() {
        let mut canceller = AdaptiveCrosstalkCanceller::new(1, vec![0]);
        let result = canceller
            .adapt(&vec![1.0], &vec![1.0])
            .unwrap();
        assert_eq!(result.num_channels, 1);
    }

    #[test]
    fn spectrum_analyzer_single_channel() {
        let mut analyzer = CrosstalkSpectrumAnalyzer::new(2, (0.0, 1.0));
        let m = CrosstalkMatrix::identity(1, vec![0]);
        let result = analyzer.analyze(&[m]).unwrap();
        assert_eq!(result.num_channels, 1);
    }

    #[test]
    fn multi_path_modeler_single_channel() {
        let mut modeler = MultiPathCrosstalkModeler::new();
        let matrix = CrosstalkMatrix::identity(1, vec![0]);
        let model = modeler.model_from_matrix(&matrix);
        assert_eq!(model.num_channels, 1);
        assert_eq!(model.paths.len(), 0);
    }

    #[test]
    fn crosstalk_statistics_single_channel() {
        let mut stats = CrosstalkStatistics::new();
        let matrix = CrosstalkMatrix::identity(1, vec![0]);
        let dist = stats.compute_distribution(&matrix);
        assert_eq!(dist.num_samples, 0); // No off-diagonal entries for 1 channel.
    }

    // ---------------------------------------------------------------------------
    // End-to-end crosstalk pipeline test
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_pipeline_end_to_end() {
        // Full pipeline: calibrate -> correct -> mitigate -> assess.
        let mut calibrator = CrosstalkCalibrator::new(2, vec![-1, 1]);
        for _ in 0..10 {
            calibrator.record_calibration(0, 0, 1.0).unwrap();
            calibrator.record_calibration(0, 1, 0.15).unwrap();
            calibrator.record_calibration(1, 0, 0.1).unwrap();
            calibrator.record_calibration(1, 1, 1.0).unwrap();
        }
        let calibration = calibrator.calibrate().unwrap();
        assert!(calibration.quality > 0.0);

        // Correct measured signal.
        let mut corrector = CrosstalkCorrector::new();
        let measured = vec![0.95, 0.65];
        let correction = corrector.correct(&measured, &calibration.matrix).unwrap();
        assert_eq!(correction.corrected_signal.len(), 2);

        // Mitigate crosstalk.
        let mut mitigator = CrosstalkMitigator::new();
        let mitigation = mitigator
            .mitigate(
                &correction.corrected_signal,
                &calibration.matrix,
                MitigationStrategy::RegularizedInversion,
            )
            .unwrap();
        assert_eq!(mitigation.signal.len(), 2);
        assert!(mitigation.confidence >= 0.0);

        // Assess crosstalk quality.
        let mut assessor = CrosstalkQualityAssessor::new();
        let quality = assessor.assess(&calibration.matrix);
        assert!(quality.quality_score >= 0.0);

        // Compute crosstalk statistics.
        let mut stats = CrosstalkStatistics::new();
        let dist = stats.compute_distribution(&calibration.matrix);
        assert!(dist.num_samples >= 2);
    }

    // ---------------------------------------------------------------------------
    // CrosstalkAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_analyzer_new() {
        let analyzer = CrosstalkAnalyzer::new(0.1);
        assert!((analyzer.max_acceptable_crosstalk - 0.1).abs() < 0.001);
    }

    #[test]
    fn crosstalk_analyzer_analyze_identity() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let analysis = analyzer.analyze(&matrix);
        assert!(analysis.within_limits);
        assert!((analysis.peak_magnitude - 0.0).abs() < 0.001);
        assert!(analysis.quality.quality_score >= 0.0);
    }

    #[test]
    fn crosstalk_analyzer_analyze_high_crosstalk() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.8f64, 0.3], vec![0.2, 0.8]],
            vec![-1, 1],
        )
        .unwrap();
        let analysis = analyzer.analyze(&matrix);
        assert!(!analysis.within_limits);
        assert!(analysis.peak_magnitude > 0.15);
        assert!(analysis.recommended_strategy as u32 >= 0);
    }

    #[test]
    fn crosstalk_analyzer_recommended_strategy_low() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let analysis = analyzer.analyze(&matrix);
        assert_eq!(
            analysis.recommended_strategy,
            MitigationStrategy::Thresholding
        );
    }

    #[test]
    fn crosstalk_analyzer_recommended_strategy_medium() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.1], vec![0.05, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        let analysis = analyzer.analyze(&matrix);
        assert_eq!(
            analysis.recommended_strategy,
            MitigationStrategy::RegularizedInversion
        );
    }

    #[test]
    fn crosstalk_analyzer_recommended_strategy_high() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let matrix = CrosstalkMatrix::new(
            2,
            vec![vec![0.7f64, 0.35], vec![0.3, 0.7]],
            vec![-1, 1],
        )
        .unwrap();
        let analysis = analyzer.analyze(&matrix);
        assert!(matches!(
            analysis.recommended_strategy,
            MitigationStrategy::TruncatedSVD | MitigationStrategy::IterativeRefinement
        ));
    }

    #[test]
    fn crosstalk_analyzer_distribution_history() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let m1 = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let m2 = CrosstalkMatrix::new(
            2,
            vec![vec![0.9f64, 0.1], vec![0.1, 0.9]],
            vec![-1, 1],
        )
        .unwrap();
        analyzer.analyze(&m1);
        analyzer.analyze(&m2);
        assert_eq!(analyzer.distribution_history().len(), 2);
        assert_eq!(analyzer.quality_history().len(), 2);
    }

    #[test]
    fn crosstalk_analyzer_reset() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let m = CrosstalkMatrix::identity(2, vec![-1, 1]);
        analyzer.analyze(&m);
        assert!(!analyzer.distribution_history().is_empty());
        analyzer.reset();
        assert_eq!(analyzer.distribution_history().len(), 0);
        assert_eq!(analyzer.quality_history().len(), 0);
    }

    #[test]
    fn crosstalk_analyzer_set_max_acceptable() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        analyzer.set_max_acceptable_crosstalk(0.05);
        assert!((analyzer.max_acceptable_crosstalk - 0.05).abs() < 0.001);
        analyzer.set_max_acceptable_crosstalk(2.0); // Clamped to 1.0.
        assert!((analyzer.max_acceptable_crosstalk - 1.0).abs() < 0.001);
    }

    #[test]
    fn crosstalk_analysis_fields() {
        let matrix = CrosstalkMatrix::identity(2, vec![-1, 1]);
        let quality = CrosstalkQuality {
            overall_level: 0.0,
            diagonal_dominance: 1.0,
            max_off_diagonal: 0.0,
            pass_threshold: 0.15,
            quality_score: 1.0,
        };
        let distribution = CrosstalkDistribution {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p95: 0.0,
            num_samples: 2,
            histogram: vec![],
        };
        let analysis = CrosstalkAnalysis {
            matrix,
            quality,
            distribution,
            peak_magnitude: 0.0,
            within_limits: true,
            recommended_strategy: MitigationStrategy::Thresholding,
        };
        assert!(analysis.within_limits);
        assert!((analysis.peak_magnitude - 0.0).abs() < 0.001);
        assert_eq!(analysis.distribution.num_samples, 2);
    }

    #[test]
    fn crosstalk_analyzer_single_channel() {
        let mut analyzer = CrosstalkAnalyzer::new(0.15);
        let matrix = CrosstalkMatrix::identity(1, vec![0]);
        let analysis = analyzer.analyze(&matrix);
        assert!(analysis.within_limits);
        assert!((analysis.peak_magnitude - 0.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // End-to-end full crosstalk pipeline with temporal window + frequency response
    // ---------------------------------------------------------------------------

    #[test]
    fn crosstalk_full_pipeline_temporal_and_frequency() {
        // Build temporal window with evolving crosstalk.
        let mut window_mgr = CrosstalkTemporalWindowManager::new(2, 8);
        for t in 0..8u64 {
            let val = 0.05 + 0.02 * (t as f64);
            window_mgr.record(t, 0, 1, val).unwrap();
            window_mgr.record(t, 1, 0, val * 0.5).unwrap();
        }

        let trend = window_mgr.crosstalk_trend();
        assert!(trend > 0.0); // Increasing crosstalk.

        let avg_matrix = window_mgr.average_matrix().unwrap();
        assert_eq!(avg_matrix.num_channels, 2);

        let current = window_mgr.current_matrix().unwrap();
        assert_eq!(current.num_channels, 2);

        // Compute frequency response for channel pair (0,1).
        let mut freq_computer = CrosstalkFrequencyResponseComputer::new(8);
        let window = &window_mgr.window;
        let response = freq_computer
            .compute_response(window, 0, 1)
            .unwrap();
        assert_eq!(response.source, 0);
        assert_eq!(response.destination, 1);
        assert_eq!(response.frequencies.len(), 8);
        assert_eq!(response.magnitudes.len(), 8);
        assert_eq!(response.phases.len(), 8);

        let peak_freq = freq_computer.peak_frequency(&response);
        assert!(peak_freq >= 0.0);
        assert!(peak_freq <= 1.0);

        // Run full crosstalk analysis.
        let mut analyzer = CrosstalkAnalyzer::new(0.2);
        let analysis = analyzer.analyze(&avg_matrix);
        assert!(analysis.within_limits);
    }

    #[test]
    fn crosstalk_temporal_window_error_on_bad_channel() {
        let mut window_mgr = CrosstalkTemporalWindowManager::new(2, 4);
        let result = window_mgr.record(0, 5, 1, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn crosstalk_freq_response_error_on_bad_channel() {
        let mut computer = CrosstalkFrequencyResponseComputer::new(4);
        let window = CrosstalkTemporalWindow {
            window_size: 4,
            window: vec![vec![vec![0.0f64; 2]; 2]; 4],
            timestamps: vec![0; 4],
            write_pos: 0,
            is_full: false,
        };
        let result = computer.compute_response(&window, 5, 1);
        assert!(result.is_err());
    }

    #[test]
    fn crosstalk_freq_response_empty_window() {
        let mut computer = CrosstalkFrequencyResponseComputer::new(4);
        let window = CrosstalkTemporalWindow {
            window_size: 4,
            window: vec![vec![vec![0.0f64; 2]; 2]; 4],
            timestamps: vec![0; 4],
            write_pos: 0,
            is_full: false,
        };
        let result = computer.compute_response(&window, 0, 1);
        assert!(result.is_err());
    }
}
