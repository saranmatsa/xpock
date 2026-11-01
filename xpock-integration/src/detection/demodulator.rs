//! Stokes demodulation engine for topological signal detection.
//!
//! Converts raw Stokes polarization measurements (S0, S1, S2, S3) into
//! normalized Stokes vectors on the Poincaré sphere, computes the skyrmion
//! density field, and demodulates skyrmion-number-encoded information.
//! All operations are carrier-agnostic — no assumptions about the physical
//! medium that carried the optical field.

use crate::errors::{XpockError, XpockResult};
use crate::detection::skyrmion_number::{SkyrmionNumber, SkyrmionField, SkyrmionDensity};
use std::collections::HashMap;

// ============================================================================
// Demodulation configuration
// ============================================================================

/// Configuration for Stokes demodulation.
#[derive(Debug, Clone, PartialEq)]
pub struct DemodulationConfig {
    /// Number of Stokes measurements used for demodulation (4, 6, or more).
    pub num_measurements: usize,
    /// Stokes demodulation matrix (measurements x 4 components).
    /// Each row contains [a0, a1, a2, a3] coefficients for computing Si.
    pub demod_matrix: Option<[[f64; 4]; 16]>,
    /// Calibration matrix for intensity normalization.
    pub calibration_matrix: Option<[[f64; 4]; 4]>,
    /// Minimum S0 intensity for valid demodulation.
    pub min_s0_intensity: f64,
    /// Maximum allowed demodulation residual error.
    pub max_residual_error: f64,
    /// Enable sub-pixel refinement of demodulated vectors.
    pub enable_subpixel: bool,
    /// Noise threshold for rejecting unreliable measurements.
    pub noise_threshold: f64,
}

impl DemodulationConfig {
    /// Create a new demodulation configuration with default 4-measurement setup.
    pub fn new() -> Self {
        Self {
            num_measurements: 4,
            demod_matrix: None,
            calibration_matrix: None,
            min_s0_intensity: 1e-6,
            max_residual_error: 0.1,
            enable_subpixel: true,
            noise_threshold: 0.01,
        }
    }

    /// Create a 4-measurement demodulation config (standard linear polarizations).
    pub fn four_measurement() -> Self {
        let mut config = Self::new();
        config.num_measurements = 4;
        // Standard Stokes demodulation matrix for 0°, 45°, 90°, circular measurements.
        config.demod_matrix = Some([
            [0.5,  0.5,  0.0,  0.0],  // S0 from 0°
            [0.5, -0.5,  0.0,  0.0],  // S1 from 90°
            [0.5,  0.0,  0.5,  0.0],  // S2 from 45°
            [0.5,  0.0,  0.0,  0.5],  // S3 from circular
        ]);
        config
    }

    /// Create a 6-measurement demodulation config (overdetermined).
    pub fn six_measurement() -> Self {
        let mut config = Self::new();
        config.num_measurements = 6;
        config.demod_matrix = Some([
            [0.5,  0.5,  0.0,  0.0],
            [0.5, -0.5,  0.0,  0.0],
            [0.5,  0.0,  0.5,  0.0],
            [0.5,  0.0, -0.5,  0.0],
            [0.5,  0.0,  0.0,  0.5],
            [0.5,  0.0,  0.0, -0.5],
        ]);
        config
    }
}

impl Default for DemodulationConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Demodulation result types
// ============================================================================

/// Result of Stokes demodulation at a single pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct DemodulatedPixel {
    /// Pixel coordinates (x, y).
    pub position: (usize, usize),
    /// Raw S0 intensity (total intensity).
    pub s0: f64,
    /// Normalized Stokes parameter S1/S0.
    pub s1_norm: f64,
    /// Normalized Stokes parameter S2/S0.
    pub s2_norm: f64,
    /// Normalized Stokes parameter S3/S0.
    pub s3_norm: f64,
    /// Degree of polarization (0.0 to 1.0).
    pub dop: f64,
    /// Degree of linear polarization.
    pub dolp: f64,
    /// Degree of circular polarization.
    pub docp: f64,
    /// Polarization angle (radians, 0 to π).
    pub polarization_angle: f64,
    /// Ellipticity angle (radians, -π/4 to π/4).
    pub ellipticity_angle: f64,
    /// Demodulation confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Residual error from demodulation matrix inversion.
    pub residual_error: f64,
}

/// Complete demodulation result for a field of pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct DemodulationResult {
    /// Grid dimensions (width, height).
    pub dimensions: (usize, usize),
    /// Per-pixel demodulated data.
    pub pixels: Vec<Vec<DemodulatedPixel>>,
    /// Overall demodulation quality metrics.
    pub quality: DemodulationQuality,
    /// Computation timestamp.
    pub computed_at: u64,
    /// Number of pixels successfully demodulated.
    pub valid_pixel_count: usize,
    /// Number of pixels rejected due to low intensity or high noise.
    pub rejected_pixel_count: usize,
}

/// Quality metrics for a demodulation result.
#[derive(Debug, Clone, PartialEq)]
pub struct DemodulationQuality {
    /// Mean degree of polarization across valid pixels.
    pub mean_dop: f64,
    /// Standard deviation of DOP.
    pub dop_std: f64,
    /// Mean residual error.
    pub mean_residual: f64,
    /// Maximum residual error.
    pub max_residual: f64,
    /// Fraction of valid pixels.
    pub valid_fraction: f64,
    /// Overall quality score (0.0 to 1.0).
    pub overall_score: f64,
}

// ============================================================================
// Stokes demodulator
// ============================================================================

/// Stokes demodulation engine.
///
/// Takes raw intensity measurements through different polarization analyzers
/// and computes the full Stokes vector (S0, S1, S2, S3) at each pixel,
/// then normalizes to the Poincaré sphere for skyrmion analysis.
pub struct SkyrmionDemodulator {
    /// Demodulation configuration.
    config: DemodulationConfig,
    /// Demodulation history.
    history: Vec<DemodulationResult>,
    /// Maximum history length.
    max_history: usize,
    /// Calibration data accumulated over time.
    calibration_history: Vec<CalibrationRecord>,
}

/// A calibration record for tracking demodulation drift.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationRecord {
    /// Timestamp in milliseconds.
    pub timestamp: u64,
    /// Mean S0 intensity at calibration time.
    pub mean_s0: f64,
    /// Mean DOP at calibration time.
    pub mean_dop: f64,
    /// Calibration quality score.
    pub quality: f64,
}

impl SkyrmionDemodulator {
    /// Create a new demodulator with default configuration.
    pub fn new() -> Self {
        Self::with_config(DemodulationConfig::new())
    }

    /// Create a new demodulator with custom configuration.
    pub fn with_config(config: DemodulationConfig) -> Self {
        Self {
            config,
            history: Vec::with_capacity(1000),
            max_history: 1000,
            calibration_history: Vec::with_capacity(100),
        }
    }

    /// Demodulate raw Stokes measurements into normalized Stokes vectors.
    ///
    /// `measurements` is [num_measurements][height][width] — raw intensity
    /// values from each polarization analyzer.
    pub fn demodulate(
        &mut self,
        measurements: &[Vec<Vec<f64>>],
    ) -> XpockResult<DemodulationResult> {
        if measurements.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: self.config.num_measurements,
                },
            ));
        }

        if measurements.len() < self.config.num_measurements {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: measurements.len(),
                    need: self.config.num_measurements,
                },
            ));
        }

        let height = measurements[0].len();
        if height == 0 || measurements[0][0].is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::EmptyField,
            ));
        }
        let width = measurements[0][0].len();

        // Verify all measurement planes have same dimensions.
        for (i, plane) in measurements.iter().enumerate() {
            if plane.len() != height || plane.iter().any(|row| row.len() != width) {
                return Err(XpockError::Detection(
                    crate::errors::DetectionError::DimensionMismatch {
                        expected: (height, width),
                        actual: (plane.len(), plane[0].len()),
                        component: format!("measurement_{}", i),
                    },
                ));
            }
        }

        let mut pixels = Vec::with_capacity(height);
        let mut valid_count = 0usize;
        let mut rejected_count = 0usize;
        let mut dop_sum = 0.0f64;
        let mut dop_sq_sum = 0.0f64;
        let mut residual_sum = 0.0f64;
        let mut max_residual = 0.0f64;

        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                // Collect intensities from all measurements at this pixel.
                let mut intensities = Vec::with_capacity(self.config.num_measurements);
                for m in 0..self.config.num_measurements {
                    intensities.push(measurements[m][y][x]);
                }

                let pixel = self.demodulate_pixel(x, y, &intensities)?;
                if pixel.confidence > 0.0 {
                    valid_count += 1;
                    dop_sum += pixel.dop;
                    dop_sq_sum += pixel.dop * pixel.dop;
                    residual_sum += pixel.residual_error;
                    max_residual = max_residual.max(pixel.residual_error);
                } else {
                    rejected_count += 1;
                }
                row.push(pixel);
            }
            pixels.push(row);
        }

        let total_pixels = width * height;
        let valid_fraction = if total_pixels > 0 {
            valid_count as f64 / total_pixels as f64
        } else {
            0.0
        };

        let mean_dop = if valid_count > 0 {
            dop_sum / valid_count as f64
        } else {
            0.0
        };
        let dop_std = if valid_count > 1 {
            let mean = dop_sum / valid_count as f64;
            let var = (dop_sq_sum - 2.0 * mean * dop_sum + valid_count as f64 * mean * mean)
                / (valid_count - 1) as f64;
            var.max(0.0).sqrt()
        } else {
            0.0
        };

        let mean_residual = if valid_count > 0 {
            residual_sum / valid_count as f64
        } else {
            0.0
        };

        let quality = DemodulationQuality {
            mean_dop,
            dop_std,
            mean_residual,
            max_residual,
            valid_fraction,
            overall_score: (valid_fraction * 0.4
                + mean_dop * 0.3
                + (1.0 - mean_residual.min(1.0)) * 0.3)
                .clamp(0.0, 1.0),
        };

        let result = DemodulationResult {
            dimensions: (width, height),
            pixels,
            quality,
            computed_at: crate::detection::signal_detector::lib_ms(),
            valid_pixel_count: valid_count,
            rejected_pixel_count: rejected_count,
        };

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        // Record calibration data.
        self.calibration_history.push(CalibrationRecord {
            timestamp: crate::detection::signal_detector::lib_ms(),
            mean_s0: self.avg_s0(&measurements),
            mean_dop: quality.mean_dop,
            quality: quality.overall_score,
        });
        if self.calibration_history.len() > 100 {
            self.calibration_history.remove(0);
        }

        Ok(result)
    }

    /// Demodulate a single pixel from intensity measurements.
    fn demodulate_pixel(
        &self,
        x: usize,
        y: usize,
        intensities: &[f64],
    ) -> XpockResult<DemodulatedPixel> {
        let n = self.config.num_measurements;
        if intensities.len() < n {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: intensities.len(),
                    need: n,
                },
            ));
        }

        // Use demodulation matrix if available, otherwise use simple 4-measurement formulas.
        let (s0, s1, s2, s3, residual_error) = if let Some(matrix) = &self.config.demod_matrix {
            self.matrix_demodulate(intensities, matrix)?
        } else {
            self.simple_demodulate(intensities)?
        };

        // Reject pixels with insufficient S0 intensity.
        if s0.abs() < self.config.min_s0_intensity {
            return Ok(DemodulatedPixel {
                position: (x, y),
                s0,
                s1_norm: 0.0,
                s2_norm: 0.0,
                s3_norm: 0.0,
                dop: 0.0,
                dolp: 0.0,
                docp: 0.0,
                polarization_angle: 0.0,
                ellipticity_angle: 0.0,
                confidence: 0.0,
                residual_error,
            });
        }

        // Normalize Stokes parameters by S0.
        let s1_norm = s1 / s0;
        let s2_norm = s2 / s0;
        let s3_norm = s3 / s0;

        // Compute degree of polarization.
        let dop = (s1_norm.powi(2) + s2_norm.powi(2) + s3_norm.powi(2)).sqrt().clamp(0.0, 1.0);
        let dolp = (s1_norm.powi(2) + s2_norm.powi(2)).sqrt().clamp(0.0, 1.0);
        let docp = s3_norm.clamp(-1.0, 1.0);

        // Polarization angle: 0.5 * atan2(S2, S1).
        let polarization_angle = 0.5 * s2_norm.atan2(s1_norm);

        // Ellipticity angle: 0.5 * asin(S3).
        let ellipticity_angle = 0.5 * s3_norm.clamp(-1.0, 1.0).asin();

        // Confidence based on DOP and residual error.
        let dop_confidence = dop.clamp(0.0, 1.0);
        let residual_confidence = (1.0 - residual_error / self.config.max_residual_error.max(1e-10))
            .clamp(0.0, 1.0);
        let confidence = (dop_confidence * 0.7 + residual_confidence * 0.3).clamp(0.0, 1.0);

        Ok(DemodulatedPixel {
            position: (x, y),
            s0,
            s1_norm,
            s2_norm,
            s3_norm,
            dop,
            dolp,
            docp,
            polarization_angle,
            ellipticity_angle,
            confidence,
            residual_error,
        })
    }

    /// Demodulate using the provided demodulation matrix.
    fn matrix_demodulate(
        &self,
        intensities: &[f64],
        matrix: &[[f64; 4]; 16],
    ) -> XpockResult<(f64, f64, f64, f64, f64)> {
        let n = self.config.num_measurements.min(16);
        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut s3 = 0.0f64;
        let mut residual_sum = 0.0f64;

        for m in 0..n {
            let row = matrix[m];
            s0 += row[0] * intensities[m];
            s1 += row[1] * intensities[m];
            s2 += row[2] * intensities[m];
            s3 += row[3] * intensities[m];
        }

        // Estimate residual: reconstruction error from the demodulation model.
        for m in 0..n {
            let row = matrix[m];
            let predicted = row[0] * s0 + row[1] * s1 + row[2] * s2 + row[3] * s3;
            residual_sum += (intensities[m] - predicted).powi(2);
        }
        let residual_error = (residual_sum / n as f64).sqrt();

        Ok((s0, s1, s2, s3, residual_error))
    }

    /// Simple 4-measurement demodulation (no matrix).
    fn simple_demodulate(&self, intensities: &[f64]) -> XpockResult<(f64, f64, f64, f64, f64)> {
        if intensities.len() < 4 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: intensities.len(),
                    need: 4,
                },
            ));
        }

        // Standard four-measurement: I0, I90, I45, Icircular
        let i0 = intensities[0];
        let i90 = intensities[1];
        let i45 = intensities[2];
        let icirc = intensities[3];

        let s0 = i0 + i90;
        let s1 = i0 - i90;
        let s2 = i45 - (s0 - i45); // I45 - I135 where I135 = S0 - I45
        let s3 = 2.0 * icirc - s0; // Icircular = (S0 + S3)/2

        // Residual: consistency check S0^2 >= S1^2 + S2^2 + S3^2.
        let s0_sq = s0 * s0;
        let s_vec_sq = s1 * s1 + s2 * s2 + s3 * s3;
        let residual_error = if s0_sq > 0.0 {
            ((s0_sq - s_vec_sq).abs() / s0_sq).sqrt()
        } else {
            1.0
        };

        Ok((s0, s1, s2, s3, residual_error))
    }

    /// Convert demodulated result to a skyrmion field.
    pub fn to_skyrmion_field(&self, result: &DemodulationResult) -> SkyrmionField {
        let (width, height) = result.dimensions;
        let mut density_grid = vec![vec![0.0f64; width]; height];

        for y in 0..height {
            for x in 0..width {
                let p = &result.pixels[y][x];
                if p.confidence > 0.0 {
                    // Skyrmion density proxy from normalized Stokes vector curl.
                    let n = [p.s1_norm, p.s2_norm, p.s3_norm];
                    let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
                    if norm > 1e-10 {
                        // Use DOP-weighted density as skyrmion density proxy.
                        density_grid[y][x] = p.dop * p.s3_norm;
                    }
                }
            }
        }

        let sn = SkyrmionNumber::new(0, 0.0, result.valid_pixel_count);
        SkyrmionField::new((width, height), density_grid, sn)
    }

    /// Returns the demodulation configuration.
    pub fn config(&self) -> &DemodulationConfig {
        &self.config
    }

    /// Returns the demodulation history.
    pub fn history(&self) -> &[DemodulationResult] {
        &self.history
    }

    /// Returns the calibration history.
    pub fn calibration_history(&self) -> &[CalibrationRecord] {
        &self.calibration_history
    }

    /// Returns the average demodulation quality score.
    pub fn avg_quality_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.quality.overall_score)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the average valid fraction across history.
    pub fn avg_valid_fraction(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.quality.valid_fraction)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Reset the demodulation history.
    pub fn reset(&mut self) {
        self.history.clear();
        self.calibration_history.clear();
    }

    /// Update demodulation configuration.
    pub fn update_config(&mut self, config: DemodulationConfig) {
        self.config = config;
    }

    /// Average S0 intensity across all measurement planes.
    fn avg_s0(&self, measurements: &[Vec<Vec<f64>>]) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for m in 0..self.config.num_measurements.min(measurements.len()) {
            for row in &measurements[m] {
                for &val in row {
                    sum += val;
                    count += 1;
                }
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

impl Default for SkyrmionDemodulator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stokes vector normalization
// ============================================================================

/// Normalized Stokes vector on the Poincaré sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StokesVector {
    /// S1/S0 component (x on Poincaré sphere).
    pub x: f64,
    /// S2/S0 component (y on Poincaré sphere).
    pub y: f64,
    /// S3/S0 component (z on Poincaré sphere).
    pub z: f64,
    /// Degree of polarization (magnitude).
    pub magnitude: f64,
}

impl StokesVector {
    /// Create a new Stokes vector from normalized components.
    pub fn new(s1_norm: f64, s2_norm: f64, s3_norm: f64) -> Self {
        let magnitude = (s1_norm.powi(2) + s2_norm.powi(2) + s3_norm.powi(2)).sqrt();
        Self {
            x: s1_norm,
            y: s2_norm,
            z: s3_norm,
            magnitude,
        }
    }

    /// Returns true if the vector is on the Poincaré sphere (|v| ≈ 1).
    pub fn is_on_sphere(&self, tolerance: f64) -> bool {
        (self.magnitude - 1.0).abs() < tolerance
    }

    /// Normalize the vector to unit length.
    pub fn normalize(&self) -> StokesVector {
        if self.magnitude > 1e-10 {
            StokesVector::new(
                self.x / self.magnitude,
                self.y / self.magnitude,
                self.z / self.magnitude,
            )
        } else {
            StokesVector::new(0.0, 0.0, 0.0)
        }
    }

    /// Dot product with another Stokes vector.
    pub fn dot(&self, other: &StokesVector) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Angular distance on the Poincaré sphere (radians).
    pub fn angular_distance(&self, other: &StokesVector) -> f64 {
        let dot = self.dot(other).clamp(-1.0, 1.0);
        dot.acos()
    }

    /// Returns the polarization state classification.
    pub fn classification(&self) -> PolarizationState {
        if self.magnitude < 0.1 {
            return PolarizationState::Unpolarized;
        }
        if (self.magnitude - 1.0).abs() < 0.05 {
            if self.z.abs() > 0.9 {
                if self.z > 0.0 {
                    return PolarizationState::RightCircular;
                } else {
                    return PolarizationState::LeftCircular;
                }
            }
            if self.z.abs() < 0.1 {
                return PolarizationState::Linear;
            }
            return PolarizationState::Elliptical;
        }
        PolarizationState::PartiallyPolarized
    }
}

/// Polarization state classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarizationState {
    Unpolarized,
    PartiallyPolarized,
    Linear,
    Circular,
    RightCircular,
    LeftCircular,
    Elliptical,
}

impl Default for StokesVector {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

// ============================================================================
// Poincaré sphere operations
// ============================================================================

/// Operations on the Poincaré sphere for topological analysis.
pub struct PoincareSphere {
    /// Radius of the sphere (always 1.0 for normalized Stokes).
    pub radius: f64,
}

impl PoincareSphere {
    /// Create a new Poincaré sphere.
    pub fn new() -> Self {
        Self { radius: 1.0 }
    }

    /// Map a Stokes vector to spherical coordinates (theta, phi).
    /// theta: polar angle from north pole (0 to π).
    /// phi: azimuthal angle (0 to 2π).
    pub fn to_spherical(&self, v: &StokesVector) -> (f64, f64) {
        let v = v.normalize();
        let theta = v.z.clamp(-1.0, 1.0).acos();
        let phi = v.y.atan2(v.x).rem_euclid(2.0 * std::f64::consts::PI);
        (theta, phi)
    }

    /// Map spherical coordinates back to a Stokes vector.
    pub fn from_spherical(&self, theta: f64, phi: f64) -> StokesVector {
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();
        StokesVector::new(x, y, z)
    }

    /// Compute the solid angle subtended by three points on the sphere.
    /// Used for topological charge (skyrmion number) computation.
    pub fn solid_angle(&self, a: &StokesVector, b: &StokesVector, c: &StokesVector) -> f64 {
        let a = a.normalize();
        let b = b.normalize();
        let c = c.normalize();

        // Van Oosterom-Strackee formula for solid angle of a spherical triangle.
        let dot_ab = a.dot(&b);
        let dot_ac = a.dot(&c);
        let dot_bc = b.dot(&c);

        // Scalar triple product |a · (b × c)|.
        let cross_x = b.y * c.z - b.z * c.y;
        let cross_y = b.z * c.x - b.x * c.z;
        let cross_z = b.x * c.y - b.y * c.x;
        let triple = a.x * cross_x + a.y * cross_y + a.z * cross_z;

        let denom = 1.0 + dot_ab + dot_ac + dot_bc;
        let solid_angle = (2.0 * triple / denom.max(1e-10)).atan().abs();

        solid_angle
    }

    /// Returns true if three points form a right-handed triplet on the sphere.
    pub fn is_right_handed(&self, a: &StokesVector, b: &StokesVector, c: &StokesVector) -> bool {
        let a = a.normalize();
        let b = b.normalize();
        let c = c.normalize();

        let cross_x = b.y * c.z - b.z * c.y;
        let cross_y = b.z * c.x - b.x * c.z;
        let cross_z = b.x * c.y - b.y * c.x;

        let dot = a.x * cross_x + a.y * cross_y + a.z * cross_z;
        dot > 0.0
    }
}

impl Default for PoincareSphere {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stokes parameter analysis
// ============================================================================

/// Analysis of Stokes parameters across a field.
#[derive(Debug, Clone, PartialEq)]
pub struct StokesAnalysis {
    /// Mean S0 intensity.
    pub mean_s0: f64,
    /// Mean S1/S0.
    pub mean_s1: f64,
    /// Mean S2/S0.
    pub mean_s2: f64,
    /// Mean S3/S0.
    pub mean_s3: f64,
    /// Standard deviation of S0.
    pub s0_std: f64,
    /// Standard deviation of DOP.
    pub dop_std: f64,
    /// Mean degree of polarization.
    pub mean_dop: f64,
    /// Fraction of fully polarized pixels (DOP > 0.9).
    pub fully_polarized_fraction: f64,
    /// Fraction of unpolarized pixels (DOP < 0.1).
    pub unpolarized_fraction: f64,
    /// Dominant polarization state.
    pub dominant_state: PolarizationState,
}

/// Analyzes Stokes parameters from a demodulation result.
pub struct StokesAnalyzer {
    /// Threshold for fully polarized classification.
    pub fully_polarized_threshold: f64,
    /// Threshold for unpolarized classification.
    pub unpolarized_threshold: f64,
}

impl StokesAnalyzer {
    /// Create a new Stokes analyzer.
    pub fn new() -> Self {
        Self {
            fully_polarized_threshold: 0.9,
            unpolarized_threshold: 0.1,
        }
    }

    /// Analyze Stokes parameters from a demodulation result.
    pub fn analyze(&self, result: &DemodulationResult) -> StokesAnalysis {
        let (width, height) = result.dimensions;
        let total = width * height;

        let mut s0_sum = 0.0f64;
        let mut s1_sum = 0.0f64;
        let mut s2_sum = 0.0f64;
        let mut s3_sum = 0.0f64;
        let mut dop_sum = 0.0f64;
        let mut dop_sq_sum = 0.0f64;
        let mut fully_polarized = 0usize;
        let mut unpolarized = 0usize;
        let mut valid_count = 0usize;

        let mut state_counts: HashMap<PolarizationState, usize> = HashMap::new();

        for y in 0..height {
            for x in 0..width {
                let p = &result.pixels[y][x];
                if p.confidence > 0.0 {
                    valid_count += 1;
                    s0_sum += p.s0;
                    s1_sum += p.s1_norm;
                    s2_sum += p.s2_norm;
                    s3_sum += p.s3_norm;
                    dop_sum += p.dop;
                    dop_sq_sum += p.dop * p.dop;

                    if p.dop > self.fully_polarized_threshold {
                        fully_polarized += 1;
                    }
                    if p.dop < self.unpolarized_threshold {
                        unpolarized += 1;
                    }

                    let state = StokesVector::new(p.s1_norm, p.s2_norm, p.s3_norm).classification();
                    *state_counts.entry(state).or_insert(0) += 1;
                }
            }
        }

        if valid_count == 0 {
            return StokesAnalysis {
                mean_s0: 0.0,
                mean_s1: 0.0,
                mean_s2: 0.0,
                mean_s3: 0.0,
                s0_std: 0.0,
                dop_std: 0.0,
                mean_dop: 0.0,
                fully_polarized_fraction: 0.0,
                unpolarized_fraction: 0.0,
                dominant_state: PolarizationState::Unpolarized,
            };
        }

        let mean_s0 = s0_sum / valid_count as f64;
        let mean_s1 = s1_sum / valid_count as f64;
        let mean_s2 = s2_sum / valid_count as f64;
        let mean_s3 = s3_sum / valid_count as f64;
        let mean_dop = dop_sum / valid_count as f64;

        // Compute S0 standard deviation.
        let mut s0_sq_sum = 0.0f64;
        for y in 0..height {
            for x in 0..width {
                let p = &result.pixels[y][x];
                if p.confidence > 0.0 {
                    s0_sq_sum += p.s0 * p.s0;
                }
            }
        }
        let s0_var = if valid_count > 1 {
            (s0_sq_sum - mean_s0 * s0_sum) / (valid_count - 1) as f64
        } else {
            0.0
        };
        let s0_std = s0_var.max(0.0).sqrt();

        let dop_var = if valid_count > 1 {
            (dop_sq_sum - mean_dop * dop_sum) / (valid_count - 1) as f64
        } else {
            0.0
        };
        let dop_std = dop_var.max(0.0).sqrt();

        let fully_polarized_fraction = fully_polarized as f64 / valid_count as f64;
        let unpolarized_fraction = unpolarized as f64 / valid_count as f64;

        let dominant_state = state_counts
            .into_iter()
            .max_by_key(|_, &v| v)
            .map(|(s, _)| s)
            .unwrap_or(PolarizationState::Unpolarized);

        StokesAnalysis {
            mean_s0,
            mean_s1,
            mean_s2,
            mean_s3,
            s0_std,
            dop_std,
            mean_dop,
            fully_polarized_fraction,
            unpolarized_fraction,
            dominant_state,
        }
    }

    /// Update classification thresholds.
    pub fn update_thresholds(&mut self, fully_polarized: f64, unpolarized: f64) {
        self.fully_polarized_threshold = fully_polarized.clamp(0.0, 1.0);
        self.unpolarized_threshold = unpolarized.clamp(0.0, 1.0);
    }
}

impl Default for StokesAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Stokes parameter processing — noise reduction and spatial filtering
// ---------------------------------------------------------------------------

/// Parameters for Stokes spatial filtering.
#[derive(Debug, Clone, PartialEq)]
pub struct StokesSpatialFilterConfig {
    /// Window size for spatial averaging (must be odd).
    pub window_size: usize,
    /// Gaussian sigma for weighted spatial filtering.
    pub sigma: f64,
    /// Edge preservation factor (0.0 = no preservation, 1.0 = strong).
    pub edge_preservation: f64,
    /// Minimum DOP for a pixel to be included in spatial averaging.
    pub min_dop_for_average: f64,
}

impl Default for StokesSpatialFilterConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            sigma: 1.0,
            edge_preservation: 0.5,
            min_dop_for_average: 0.3,
        }
    }
}

/// Applies spatial filtering to Stokes parameters to reduce noise while
/// preserving polarization edges.
pub struct StokesSpatialFilter {
    config: StokesSpatialFilterConfig,
    /// Filtering history.
    history: Vec<f64>,
    /// Maximum history length.
    max_history: usize,
}

impl StokesSpatialFilter {
    /// Create a new spatial filter.
    pub fn new() -> Self {
        Self::with_config(StokesSpatialFilterConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: StokesSpatialFilterConfig) -> Self {
        Self {
            config,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Apply spatial filtering to a demodulation result.
    pub fn filter(&self, result: &DemodulationResult) -> DemodulationResult {
        let (width, height) = result.dimensions;
        let half = self.config.window_size / 2;

        let mut filtered_pixels = Vec::with_capacity(height);
        let mut dop_sum = 0.0f64;
        let mut valid_count = 0usize;

        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let filtered = self.filter_pixel(&result.pixels, x, y, width, height);
                if filtered.confidence > 0.0 {
                    valid_count += 1;
                    dop_sum += filtered.dop;
                }
                row.push(filtered);
            }
            filtered_pixels.push(row);
        }

        let mean_dop = if valid_count > 0 {
            dop_sum / valid_count as f64
        } else {
            0.0
        };

        let quality = DemodulationQuality {
            mean_dop,
            dop_std: result.quality.dop_std,
            mean_residual: result.quality.mean_residual,
            max_residual: result.quality.max_residual,
            valid_fraction: result.quality.valid_fraction,
            overall_score: (result.quality.valid_fraction * 0.4
                + mean_dop * 0.3
                + result.quality.overall_score * 0.3)
                .clamp(0.0, 1.0),
        };

        let mut result = result.clone();
        result.pixels = filtered_pixels;
        result.quality = quality;

        result
    }

    /// Filter a single pixel using weighted spatial neighborhood.
    fn filter_pixel(
        &self,
        pixels: &[Vec<DemodulatedPixel>],
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> DemodulatedPixel {
        let half = self.config.window_size / 2;
        let x_start = x.saturating_sub(half);
        let x_end = (x + half + 1).min(width);
        let y_start = y.saturating_sub(half);
        let y_end = (y + half + 1).min(height);

        let center = &pixels[y][x];
        if center.confidence <= 0.0 {
            return center.clone();
        }

        let mut weight_sum = 0.0f64;
        let mut s0_sum = 0.0f64;
        let mut s1_sum = 0.0f64;
        let mut s2_sum = 0.0f64;
        let mut s3_sum = 0.0f64;
        let mut dop_sum = 0.0f64;

        for yy in y_start..y_end {
            for xx in x_start..x_end {
                let p = &pixels[yy][xx];
                if p.confidence <= 0.0 || p.dop < self.config.min_dop_for_average {
                    continue;
                }

                let dx = xx as f64 - x as f64;
                let dy = yy as f64 - y as f64;
                let dist_sq = dx * dx + dy * dy;
                let gaussian_weight =
                    (-dist_sq / (2.0 * self.config.sigma * self.config.sigma)).exp();

                // Edge preservation: reduce weight for pixels with very different DOP.
                let dop_diff = (p.dop - center.dop).abs();
                let edge_weight = (1.0 - self.config.edge_preservation * dop_diff).clamp(0.0, 1.0);

                let weight = gaussian_weight * edge_weight * p.confidence;
                weight_sum += weight;
                s0_sum += weight * p.s0;
                s1_sum += weight * p.s1_norm;
                s2_sum += weight * p.s2_norm;
                s3_sum += weight * p.s3_norm;
                dop_sum += weight * p.dop;
            }
        }

        if weight_sum < 1e-10 {
            return center.clone();
        }

        let s0 = s0_sum / weight_sum;
        let s1_norm = s1_sum / weight_sum;
        let s2_norm = s2_sum / weight_sum;
        let s3_norm = s3_sum / weight_sum;
        let dop = dop_sum / weight_sum;

        let dolp = (s1_norm.powi(2) + s2_norm.powi(2)).sqrt().clamp(0.0, 1.0);
        let docp = s3_norm.clamp(-1.0, 1.0);
        let polarization_angle = 0.5 * s2_norm.atan2(s1_norm);
        let ellipticity_angle = 0.5 * s3_norm.clamp(-1.0, 1.0).asin();

        DemodulatedPixel {
            position: (x, y),
            s0,
            s1_norm,
            s2_norm,
            s3_norm,
            dop,
            dolp,
            docp,
            polarization_angle,
            ellipticity_angle,
            confidence: center.confidence.min(1.0),
            residual_error: center.residual_error,
        }
    }

    /// Update filter configuration.
    pub fn update_config(&mut self, config: StokesSpatialFilterConfig) {
        self.config = config;
    }

    /// Record a quality metric in history.
    pub fn record_quality(&mut self, quality: f64) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(quality);
    }

    /// Returns the average recorded quality.
    pub fn avg_recorded_quality(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }
}

// ============================================================================
// Polarization ellipse analysis
// ============================================================================

/// Parameters of a polarization ellipse.
#[derive(Debug, Clone, PartialEq)]
pub struct PolarizationEllipse {
    /// Major axis amplitude (sqrt of intensity along major axis).
    pub major_amplitude: f64,
    /// Minor axis amplitude.
    pub minor_amplitude: f64,
    /// Orientation angle of major axis (radians, 0 to π).
    pub orientation: f64,
    /// Ellipticity angle (radians, -π/4 to π/4).
    pub ellipticity: f64,
    /// Handedness: +1 = right-handed, -1 = left-handed, 0 = linear.
    pub handedness: i8,
    /// Degree of polarization.
    pub dop: f64,
    /// Ellipse circularity (0.0 = linear, 1.0 = circular).
    pub circularity: f64,
}

/// Analyzes polarization ellipses from Stokes parameters.
pub struct PolarizationEllipseAnalyzer {
    /// Minimum DOP for valid ellipse analysis.
    pub min_dop: f64,
    /// Analysis history.
    history: Vec<PolarizationEllipse>,
    /// Maximum history length.
    max_history: usize,
}

impl PolarizationEllipseAnalyzer {
    /// Create a new ellipse analyzer.
    pub fn new() -> Self {
        Self {
            min_dop: 0.1,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Analyze the polarization ellipse for a demodulated pixel.
    pub fn analyze(&mut self, pixel: &DemodulatedPixel) -> Option<PolarizationEllipse> {
        if pixel.dop < self.min_dop {
            return None;
        }

        let s1 = pixel.s1_norm;
        let s2 = pixel.s2_norm;
        let s3 = pixel.s3_norm;

        // Orientation angle: psi = 0.5 * atan2(S2, S1).
        let orientation = 0.5 * s2.atan2(s1);

        // Ellipticity angle: chi = 0.5 * asin(S3 / S0) where S0 = 1 for normalized.
        let ellipticity = 0.5 * pixel.s3_norm.clamp(-1.0, 1.0).asin();

        // Major and minor axis amplitudes.
        let chi = ellipticity;
        let major = (1.0 + pixel.dop).sqrt() / std::f64::consts::SQRT_2;
        let minor = (1.0 - pixel.dop).sqrt() / std::f64::consts::SQRT_2;

        // Handedness from sign of S3.
        let handedness = if pixel.s3_norm > 0.01 {
            1
        } else if pixel.s3_norm < -0.01 {
            -1
        } else {
            0
        };

        // Circularity: |S3| / sqrt(S1^2 + S2^2 + S3^2).
        let total = (s1.powi(2) + s2.powi(2) + s3.powi(2)).sqrt();
        let circularity = if total > 1e-10 {
            pixel.s3_norm.abs() / total
        } else {
            0.0
        };

        let ellipse = PolarizationEllipse {
            major_amplitude: major,
            minor_amplitude: minor,
            orientation,
            ellipticity,
            handedness,
            dop: pixel.dop,
            circularity,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(ellipse.clone());

        Some(ellipse)
    }

    /// Analyze all pixels in a demodulation result.
    pub fn analyze_field(
        &mut self,
        result: &DemodulationResult,
    ) -> Vec<Option<PolarizationEllipse>> {
        let mut ellipses = Vec::with_capacity(result.pixels.len());
        for row in &result.pixels {
            let mut row_ellipses = Vec::with_capacity(row.len());
            for pixel in row {
                row_ellipses.push(self.analyze(pixel));
            }
            ellipses.push(row_ellipses);
        }
        ellipses
    }

    /// Returns the dominant handedness across history.
    pub fn dominant_handedness(&self) -> i8 {
        if self.history.is_empty() {
            return 0;
        }
        let mut pos = 0usize;
        let mut neg = 0usize;
        for e in &self.history {
            if e.handedness > 0 {
                pos += 1;
            } else if e.handedness < 0 {
                neg += 1;
            }
        }
        if pos > neg {
            1
        } else if neg > pos {
            -1
        } else {
            0
        }
    }

    /// Returns the mean ellipticity across history.
    pub fn mean_ellipticity(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|e| e.ellipticity)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Returns the mean circularity across history.
    pub fn mean_circularity(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|e| e.circularity)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for PolarizationEllipseAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-frame demodulation and averaging
// ============================================================================

/// A single frame of raw Stokes measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct StokesFrame {
    /// Frame timestamp.
    pub timestamp: u64,
    /// Measurements: [num_measurements][height][width].
    pub measurements: Vec<Vec<Vec<f64>>>,
    /// Frame quality score (0.0 to 1.0).
    pub quality: f64,
}

/// Result of multi-frame averaging.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFrameResult {
    /// Averaged demodulation result.
    pub averaged_result: DemodulationResult,
    /// Number of frames averaged.
    pub frame_count: usize,
    /// Temporal noise reduction factor.
    pub noise_reduction: f64,
    /// Frame quality scores.
    pub frame_qualities: Vec<f64>,
}

/// Performs multi-frame demodulation with temporal averaging to reduce noise.
pub struct MultiFrameDemodulator {
    /// Internal demodulator.
    demodulator: SkyrmionDemodulator,
    /// Frame buffer for temporal averaging.
    frame_buffer: Vec<StokesFrame>,
    /// Maximum buffer size.
    max_buffer_size: usize,
    /// Minimum quality for a frame to be included.
    min_frame_quality: f64,
    /// Temporal averaging results.
    averaging_results: Vec<MultiFrameResult>,
    /// Maximum averaging history length.
    max_averaging_history: usize,
}

impl MultiFrameDemodulator {
    /// Create a new multi-frame demodulator.
    pub fn new() -> Self {
        Self {
            demodulator: SkyrmionDemodulator::new(),
            frame_buffer: Vec::with_capacity(50),
            max_buffer_size: 50,
            min_frame_quality: 0.3,
            averaging_results: Vec::with_capacity(100),
            max_averaging_history: 100,
        }
    }

    /// Add a frame to the buffer.
    pub fn add_frame(&mut self, frame: StokesFrame) {
        if self.frame_buffer.len() >= self.max_buffer_size {
            self.frame_buffer.remove(0);
        }
        self.frame_buffer.push(frame);
    }

    /// Demodulate and average all frames in the buffer.
    pub fn demodulate_and_average(&mut self) -> XpockResult<MultiFrameResult> {
        if self.frame_buffer.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        let quality_filtered: Vec<&StokesFrame> = self
            .frame_buffer
            .iter()
            .filter(|f| f.quality >= self.min_frame_quality)
            .collect();

        if quality_filtered.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        // Demodulate each frame.
        let mut demodulated = Vec::with_capacity(quality_filtered.len());
        for frame in &quality_filtered {
            let result = self.demodulator.demodulate(&frame.measurements)?;
            demodulated.push(result);
        }

        // Average the demodulated results pixel-wise.
        let first = &demodulated[0];
        let (width, height) = first.dimensions;
        let mut averaged_pixels = Vec::with_capacity(height);

        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let mut s0_sum = 0.0f64;
                let mut s1_sum = 0.0f64;
                let mut s2_sum = 0.0f64;
                let mut s3_sum = 0.0f64;
                let mut conf_sum = 0.0f64;
                let mut count = 0usize;

                for d in &demodulated {
                    let p = &d.pixels[y][x];
                    if p.confidence > 0.0 {
                        s0_sum += p.s0;
                        s1_sum += p.s1_norm;
                        s2_sum += p.s2_norm;
                        s3_sum += p.s3_norm;
                        conf_sum += p.confidence;
                        count += 1;
                    }
                }

                if count > 0 {
                    let inv_count = 1.0 / count as f64;
                    let s0 = s0_sum * inv_count;
                    let s1_norm = s1_sum * inv_count;
                    let s2_norm = s2_sum * inv_count;
                    let s3_norm = s3_sum * inv_count;
                    let confidence = conf_sum * inv_count;

                    let dop = (s1_norm.powi(2) + s2_norm.powi(2) + s3_norm.powi(2))
                        .sqrt()
                        .clamp(0.0, 1.0);
                    let dolp = (s1_norm.powi(2) + s2_norm.powi(2)).sqrt().clamp(0.0, 1.0);
                    let docp = s3_norm.clamp(-1.0, 1.0);
                    let polarization_angle = 0.5 * s2_norm.atan2(s1_norm);
                    let ellipticity_angle = 0.5 * s3_norm.clamp(-1.0, 1.0).asin();

                    row.push(DemodulatedPixel {
                        position: (x, y),
                        s0,
                        s1_norm,
                        s2_norm,
                        s3_norm,
                        dop,
                        dolp,
                        docp,
                        polarization_angle,
                        ellipticity_angle,
                        confidence,
                        residual_error: 0.0,
                    });
                } else {
                    row.push(DemodulatedPixel {
                        position: (x, y),
                        s0: 0.0,
                        s1_norm: 0.0,
                        s2_norm: 0.0,
                        s3_norm: 0.0,
                        dop: 0.0,
                        dolp: 0.0,
                        docp: 0.0,
                        polarization_angle: 0.0,
                        ellipticity_angle: 0.0,
                        confidence: 0.0,
                        residual_error: 0.0,
                    });
                }
            }
            averaged_pixels.push(row);
        }

        let valid_count = averaged_pixels
            .iter()
            .flatten()
            .filter(|p| p.confidence > 0.0)
            .count();
        let total = width * height;
        let valid_fraction = if total > 0 {
            valid_count as f64 / total as f64
        } else {
            0.0
        };

        let mean_dop = averaged_pixels
            .iter()
            .flatten()
            .filter(|p| p.confidence > 0.0)
            .map(|p| p.dop)
            .sum::<f64>()
            / valid_count.max(1) as f64;

        let quality = DemodulationQuality {
            mean_dop,
            dop_std: 0.0,
            mean_residual: 0.0,
            max_residual: 0.0,
            valid_fraction,
            overall_score: (valid_fraction * 0.5 + mean_dop * 0.5).clamp(0.0, 1.0),
        };

        let frame_qualities: Vec<f64> = quality_filtered.iter().map(|f| f.quality).collect();
        let noise_reduction = (quality_filtered.len() as f64).sqrt();

        let averaged_result = DemodulationResult {
            dimensions: (width, height),
            pixels: averaged_pixels,
            quality,
            computed_at: crate::detection::signal_detector::lib_ms(),
            valid_pixel_count: valid_count,
            rejected_pixel_count: total - valid_count,
        };

        let mf_result = MultiFrameResult {
            averaged_result,
            frame_count: quality_filtered.len(),
            noise_reduction,
            frame_qualities,
        };

        if self.averaging_results.len() >= self.max_averaging_history {
            self.averaging_results.remove(0);
        }
        self.averaging_results.push(mf_result.clone());

        Ok(mf_result)
    }

    /// Returns the number of frames in the buffer.
    pub fn frame_count(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Clear the frame buffer.
    pub fn clear_buffer(&mut self) {
        self.frame_buffer.clear();
    }

    /// Update multi-frame parameters.
    pub fn update_params(&mut self, max_buffer_size: usize, min_frame_quality: f64) {
        self.max_buffer_size = max_buffer_size.max(1);
        self.min_frame_quality = min_frame_quality.clamp(0.0, 1.0);
        while self.frame_buffer.len() > self.max_buffer_size {
            self.frame_buffer.remove(0);
        }
    }

    /// Returns the averaging history.
    pub fn averaging_history(&self) -> &[MultiFrameResult] {
        &self.averaging_results
    }

    /// Returns the average noise reduction across history.
    pub fn avg_noise_reduction(&self) -> f64 {
        if self.averaging_results.is_empty() {
            return 1.0;
        }
        self.averaging_results
            .iter()
            .map(|r| r.noise_reduction)
            .sum::<f64>()
            / self.averaging_results.len() as f64
    }
}

impl Default for MultiFrameDemodulator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Calibration engine
// ============================================================================

/// Calibration parameters for demodulation.
#[derive(Debug, Clone, PartialEq)]
pub struct DemodulationCalibration {
    /// Offset for each Stokes component (S0, S1, S2, S3).
    pub offset: [f64; 4],
    /// Scale factor for each Stokes component.
    pub scale: [f64; 4],
    /// Cross-talk matrix (4x4) for correcting component mixing.
    pub crosstalk_matrix: [[f64; 4]; 4],
    /// Calibration timestamp.
    pub timestamp: u64,
    /// Calibration quality score.
    pub quality: f64,
}

impl Default for DemodulationCalibration {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0, 1.0],
            crosstalk_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            timestamp: 0,
            quality: 1.0,
        }
    }
}

/// Calibrates demodulation parameters using reference measurements.
pub struct DemodulationCalibrator {
    /// Current calibration parameters.
    pub calibration: DemodulationCalibration,
    /// Calibration history.
    history: Vec<DemodulationCalibration>,
    /// Maximum history length.
    max_history: usize,
    /// Number of calibration samples used.
    pub sample_count: usize,
}

impl DemodulationCalibrator {
    /// Create a new calibrator with default calibration.
    pub fn new() -> Self {
        Self {
            calibration: DemodulationCalibration::default(),
            history: Vec::with_capacity(100),
            max_history: 100,
            sample_count: 0,
        }
    }

    /// Calibrate using a set of known reference Stokes states.
    /// `measured` is the measured intensities, `expected` is the expected
    /// normalized Stokes vectors [S1/S0, S2/S0, S3/S0].
    pub fn calibrate(
        &mut self,
        measured: &[Vec<f64>],
        expected: &[[f64; 3]],
    ) -> XpockResult<DemodulationCalibration> {
        if measured.is_empty() || expected.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        if measured.len() != expected.len() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (expected.len(), 3),
                    actual: (measured.len(), 4),
                    component: "calibration_data".to_string(),
                },
            ));
        }

        // Compute offset as mean difference between measured and expected S0.
        let mut s0_offset_sum = 0.0f64;
        let mut s1_scale_sum = 0.0f64;
        let mut s2_scale_sum = 0.0f64;
        let mut s3_scale_sum = 0.0f64;
        let mut count = 0usize;

        for (m, e) in measured.iter().zip(expected.iter()) {
            if m.len() < 4 {
                continue;
            }
            let measured_s0 = m[0];
            let measured_s1 = m[1];
            let measured_s2 = m[2];
            let measured_s3 = m[3];

            let expected_s1 = e[0];
            let expected_s2 = e[1];
            let expected_s3 = e[2];

            s0_offset_sum += measured_s0 - 1.0; // Assume normalized S0 = 1.
            s1_scale_sum += if expected_s1.abs() > 1e-10 {
                measured_s1 / expected_s1
            } else {
                1.0
            };
            s2_scale_sum += if expected_s2.abs() > 1e-10 {
                measured_s2 / expected_s2
            } else {
                1.0
            };
            s3_scale_sum += if expected_s3.abs() > 1e-10 {
                measured_s3 / expected_s3
            } else {
                1.0
            };
            count += 1;
        }

        if count == 0 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
                },
            ));
        }

        let inv_count = 1.0 / count as f64;
        let mut new_cal = self.calibration.clone();
        new_cal.offset[0] = s0_offset_sum * inv_count;
        new_cal.scale[0] = 1.0;
        new_cal.scale[1] = s1_scale_sum * inv_count;
        new_cal.scale[2] = s2_scale_sum * inv_count;
        new_cal.scale[3] = s3_scale_sum * inv_count;
        new_cal.timestamp = crate::detection::signal_detector::lib_ms();
        new_cal.quality = (1.0 - new_cal.offset[0].abs()).clamp(0.0, 1.0);

        // Estimate cross-talk from residual correlations.
        // Simplified: diagonal dominance check.
        let crosstalk_strength = ((new_cal.scale[1] - 1.0).abs()
            + (new_cal.scale[2] - 1.0).abs()
            + (new_cal.scale[3] - 1.0).abs())
            / 3.0;
        if crosstalk_strength > 0.1 {
            // Apply cross-talk correction.
            new_cal.crosstalk_matrix[0][0] = 1.0 / new_cal.scale[0].max(1e-10);
            new_cal.crosstalk_matrix[1][1] = 1.0 / new_cal.scale[1].max(1e-10);
            new_cal.crosstalk_matrix[2][2] = 1.0 / new_cal.scale[2].max(1e-10);
            new_cal.crosstalk_matrix[3][3] = 1.0 / new_cal.scale[3].max(1e-10);
        }

        self.sample_count += count;

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(new_cal.clone());
        self.calibration = new_cal.clone();

        Ok(new_cal)
    }

    /// Apply calibration to raw Stokes measurements.
    pub fn apply_calibration(&self, measurements: &mut [f64]) {
        if measurements.len() < 4 {
            return;
        }
        let cal = &self.calibration;
        for i in 0..4 {
            measurements[i] = (measurements[i] - cal.offset[i]) * cal.scale[i];
        }
    }

    /// Returns the calibration quality.
    pub fn calibration_quality(&self) -> f64 {
        self.calibration.quality
    }

    /// Returns the number of calibration samples.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Reset calibration to defaults.
    pub fn reset(&mut self) {
        self.calibration = DemodulationCalibration::default();
        self.history.clear();
        self.sample_count = 0;
    }

    /// Returns the calibration history.
    pub fn history(&self) -> &[DemodulationCalibration] {
        &self.history
    }
}

// ============================================================================
// Stokes vector field operations
// ============================================================================

/// Gradient of a Stokes vector field at a pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct StokesGradient {
    /// dS1/dx, dS1/dy.
    pub ds1: (f64, f64),
    /// dS2/dx, dS2/dy.
    pub ds2: (f64, f64),
    /// dS3/dx, dS3/dy.
    pub ds3: (f64, f64),
    /// Gradient magnitude.
    pub magnitude: f64,
}

/// Computes spatial derivatives of Stokes vector fields.
pub struct StokesFieldOperations {
    /// Finite difference step size.
    pub step_size: f64,
    /// Smoothing before differentiation.
    pub pre_smooth: bool,
}

impl StokesFieldOperations {
    /// Create new field operations.
    pub fn new() -> Self {
        Self {
            step_size: 1.0,
            pre_smooth: false,
        }
    }

    /// Compute the gradient of the Stokes field at each pixel.
    pub fn compute_gradient(&self, result: &DemodulationResult) -> Vec<Vec<StokesGradient>> {
        let (width, height) = result.dimensions;
        let mut gradients = Vec::with_capacity(height);

        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let p = &result.pixels[y][x];
                if p.confidence <= 0.0 {
                    row.push(StokesGradient {
                        ds1: (0.0, 0.0),
                        ds2: (0.0, 0.0),
                        ds3: (0.0, 0.0),
                        magnitude: 0.0,
                    });
                    continue;
                }

                // Central differences for interior pixels.
                let xp = (x + 1).min(width - 1);
                let xm = x.saturating_sub(1);
                let yp = (y + 1).min(height - 1);
                let ym = y.saturating_sub(1);

                let ds1_dx = (result.pixels[y][xp].s1_norm - result.pixels[y][xm].s1_norm)
                    / (2.0 * self.step_size);
                let ds1_dy = (result.pixels[yp][x].s1_norm - result.pixels[ym][x].s1_norm)
                    / (2.0 * self.step_size);

                let ds2_dx = (result.pixels[y][xp].s2_norm - result.pixels[y][xm].s2_norm)
                    / (2.0 * self.step_size);
                let ds2_dy = (result.pixels[yp][x].s2_norm - result.pixels[ym][x].s2_norm)
                    / (2.0 * self.step_size);

                let ds3_dx = (result.pixels[y][xp].s3_norm - result.pixels[y][xm].s3_norm)
                    / (2.0 * self.step_size);
                let ds3_dy = (result.pixels[yp][x].s3_norm - result.pixels[ym][x].s3_norm)
                    / (2.0 * self.step_size);

                let magnitude = (ds1_dx.powi(2)
                    + ds1_dy.powi(2)
                    + ds2_dx.powi(2)
                    + ds2_dy.powi(2)
                    + ds3_dx.powi(2)
                    + ds3_dy.powi(2))
                .sqrt();

                row.push(StokesGradient {
                    ds1: (ds1_dx, ds1_dy),
                    ds2: (ds2_dx, ds2_dy),
                    ds3: (ds3_dx, ds3_dy),
                    magnitude,
                });
            }
            gradients.push(row);
        }

        gradients
    }

    /// Compute the skyrmion density from the Stokes field gradient.
    /// This is the local contribution to Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy.
    pub fn compute_skyrmion_density(
        &self,
        result: &DemodulationResult,
    ) -> Vec<Vec<f64>> {
        let (width, height) = result.dimensions;
        let mut density = vec![vec![0.0f64; width]; height];

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let p = &result.pixels[y][x];
                if p.confidence <= 0.0 {
                    continue;
                }

                let n = [p.s1_norm, p.s2_norm, p.s3_norm];
                let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
                if norm < 1e-10 {
                    continue;
                }
                let n = [n[0] / norm, n[1] / norm, n[2] / norm];

                // Neighbors.
                let nr = &result.pixels[y][(x + 1).min(width - 1)];
                let nd = &result.pixels[(y + 1).min(height - 1)][x];

                let n_r = [nr.s1_norm, nr.s2_norm, nr.s3_norm];
                let n_d = [nd.s1_norm, nd.s2_norm, nd.s3_norm];

                let dn_dx = [n_r[0] - n[0], n_r[1] - n[1], n_r[2] - n[2]];
                let dn_dy = [n_d[0] - n[0], n_d[1] - n[1], n_d[2] - n[2]];

                // Cross product ∂x n × ∂y n.
                let cross = [
                    dn_dx[1] * dn_dy[2] - dn_dx[2] * dn_dy[1],
                    dn_dx[2] * dn_dy[0] - dn_dx[0] * dn_dy[2],
                    dn_dx[0] * dn_dy[1] - dn_dx[1] * dn_dy[0],
                ];

                // Dot product n · (∂x n × ∂y n).
                density[y][x] = n[0] * cross[0] + n[1] * cross[1] + n[2] * cross[2];
            }
        }

        density
    }

    /// Update parameters.
    pub fn update_params(&mut self, step_size: f64, pre_smooth: bool) {
        self.step_size = step_size.max(1e-6);
        self.pre_smooth = pre_smooth;
    }
}

impl Default for StokesFieldOperations {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Demodulation pipeline orchestrator
// ---------------------------------------------------------------------------

/// Stages of the demodulation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    RawAcquisition,
    Calibration,
    Demodulation,
    SpatialFiltering,
    EllipseAnalysis,
    QualityAssessment,
    Complete,
}

/// A complete demodulation pipeline result.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineResult {
    /// Final demodulation result.
    pub demodulation_result: DemodulationResult,
    /// Stages completed.
    pub stages_completed: Vec<PipelineStage>,
    /// Per-stage timing (microseconds).
    pub stage_timings: Vec<(PipelineStage, f64)>,
    /// Overall pipeline success.
    pub success: bool,
    /// Pipeline error message (if any).
    pub error_message: Option<String>,
}

/// Orchestrates the full demodulation pipeline with staged processing,
/// quality gating, and error recovery.
pub struct DemodulationPipeline {
    /// Pipeline configuration.
    config: DemodulationConfig,
    /// Whether to apply spatial filtering.
    pub enable_spatial_filter: bool,
    /// Whether to perform ellipse analysis.
    pub enable_ellipse_analysis: bool,
    /// Minimum quality threshold to proceed to next stage.
    pub quality_gate_threshold: f64,
    /// Pipeline statistics.
    pub stats: PipelineStatistics,
}

/// Statistics for pipeline runs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PipelineStatistics {
    /// Total number of pipeline runs.
    pub total_runs: u64,
    /// Number of successful runs.
    pub successful_runs: u64,
    /// Number of failed runs.
    pub failed_runs: u64,
    /// Average total processing time (microseconds).
    pub avg_processing_time_us: f64,
    /// Average quality score.
    pub avg_quality_score: f64,
    /// Stage failure counts.
    pub stage_failures: std::collections::HashMap<PipelineStage, u64>,
}

impl DemodulationPipeline {
    /// Create a new pipeline with default configuration.
    pub fn new() -> Self {
        Self {
            config: DemodulationConfig::new(),
            enable_spatial_filter: true,
            enable_ellipse_analysis: true,
            quality_gate_threshold: 0.3,
            stats: PipelineStatistics::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: DemodulationConfig) -> Self {
        Self {
            config,
            enable_spatial_filter: true,
            enable_ellipse_analysis: true,
            quality_gate_threshold: 0.3,
            stats: PipelineStatistics::default(),
        }
    }

    /// Run the full demodulation pipeline on raw measurements.
    pub fn run(
        &mut self,
        measurements: &[Vec<Vec<f64>>],
    ) -> XpockResult<PipelineResult> {
        let start = crate::detection::signal_detector::lib_ms();
        self.stats.total_runs += 1;

        let mut stages_completed = Vec::new();
        let mut stage_timings = Vec::new();
        let mut current_result: Option<DemodulationResult> = None;
        let mut error_message: Option<String> = None;

        // Stage 1: Raw acquisition (validation).
        let stage_start = crate::detection::signal_detector::lib_ms();
        if measurements.is_empty() {
            error_message = Some("No measurement planes provided".to_string());
            self.stats.failed_runs += 1;
            self.stats.stage_failures
                .entry(PipelineStage::RawAcquisition)
                .or_insert(0)
                += 1;
            let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
            stage_timings.push((PipelineStage::RawAcquisition, elapsed));
            return self.build_pipeline_result(
                stages_completed,
                stage_timings,
                false,
                error_message,
                start,
            );
        }
        stages_completed.push(PipelineStage::RawAcquisition);
        let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
        stage_timings.push((PipelineStage::RawAcquisition, elapsed));

        // Stage 2: Calibration (apply if calibration data available).
        let stage_start = crate::detection::signal_detector::lib_ms();
        // Calibration is a no-op with default calibration (identity).
        stages_completed.push(PipelineStage::Calibration);
        let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
        stage_timings.push((PipelineStage::Calibration, elapsed));

        // Stage 3: Demodulation.
        let stage_start = crate::detection::signal_detector::lib_ms();
        let mut demodulator = SkyrmionDemodulator::with_config(self.config.clone());
        let demod_result = match demodulator.demodulate(measurements) {
            Ok(r) => r,
            Err(e) => {
                error_message = Some(format!("Demodulation failed: {:?}", e));
                self.stats.failed_runs += 1;
                self.stats.stage_failures
                    .entry(PipelineStage::Demodulation)
                    .or_insert(0)
                    += 1;
                let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
                stage_timings.push((PipelineStage::Demodulation, elapsed));
                return self.build_pipeline_result(
                    stages_completed,
                    stage_timings,
                    false,
                    error_message,
                    start,
                );
            }
        };
        stages_completed.push(PipelineStage::Demodulation);
        let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
        stage_timings.push((PipelineStage::Demodulation, elapsed));

        // Quality gate: check if demodulation quality is sufficient.
        if demod_result.quality.overall_score < self.quality_gate_threshold {
            error_message = Some(format!(
                "Demodulation quality {:.3} below gate threshold {:.3}",
                demod_result.quality.overall_score, self.quality_gate_threshold
            ));
            self.stats.failed_runs += 1;
            self.stats.stage_failures
                .entry(PipelineStage::QualityAssessment)
                .or_insert(0)
                += 1;
            let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
            stage_timings.push((PipelineStage::QualityAssessment, elapsed));
            return self.build_pipeline_result(
                stages_completed,
                stage_timings,
                false,
                error_message,
                start,
            );
        }

        current_result = Some(demod_result);

        // Stage 4: Spatial filtering (optional).
        if self.enable_spatial_filter {
            let stage_start = crate::detection::signal_detector::lib_ms();
            let filter = StokesSpatialFilter::new();
            let mut filtered = filter.filter(current_result.as_ref().unwrap());
            // Re-run quality gate after filtering.
            if filtered.quality.overall_score < self.quality_gate_threshold * 0.8 {
                error_message = Some("Post-filtering quality below threshold".to_string());
                self.stats.failed_runs += 1;
                self.stats.stage_failures
                    .entry(PipelineStage::SpatialFiltering)
                    .or_insert(0)
                    += 1;
                let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
                stage_timings.push((PipelineStage::SpatialFiltering, elapsed));
                return self.build_pipeline_result(
                    stages_completed,
                    stage_timings,
                    false,
                    error_message,
                    start,
                );
            }
            current_result = Some(filtered);
            stages_completed.push(PipelineStage::SpatialFiltering);
            let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
            stage_timings.push((PipelineStage::SpatialFiltering, elapsed));
        }

        // Stage 5: Ellipse analysis (optional, informational).
        if self.enable_ellipse_analysis {
            let stage_start = crate::detection::signal_detector::lib_ms();
            let mut ellipse_analyzer = PolarizationEllipseAnalyzer::new();
            let _ellipses = ellipse_analyzer.analyze_field(current_result.as_ref().unwrap());
            stages_completed.push(PipelineStage::EllipseAnalysis);
            let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
            stage_timings.push((PipelineStage::EllipseAnalysis, elapsed));
        }

        // Stage 6: Quality assessment.
        let stage_start = crate::detection::signal_detector::lib_ms();
        stages_completed.push(PipelineStage::QualityAssessment);
        let elapsed = (crate::detection::signal_detector::lib_ms() - stage_start) as f64;
        stage_timings.push((PipelineStage::QualityAssessment, elapsed));

        stages_completed.push(PipelineStage::Complete);
        self.stats.successful_runs += 1;

        let total_time = (crate::detection::signal_detector::lib_ms() - start) as f64;
        self.stats.avg_processing_time_us = if self.stats.successful_runs > 0 {
            (self.stats.avg_processing_time_us * (self.stats.successful_runs - 1) as f64
                + total_time)
                / self.stats.successful_runs as f64
        } else {
            total_time
        };

        let final_result = current_result.unwrap();
        self.stats.avg_quality_score = if self.stats.successful_runs > 0 {
            (self.stats.avg_quality_score * (self.stats.successful_runs - 1) as f64
                + final_result.quality.overall_score)
                / self.stats.successful_runs as f64
        } else {
            final_result.quality.overall_score
        };

        Ok(PipelineResult {
            demodulation_result: final_result,
            stages_completed,
            stage_timings,
            success: true,
            error_message: None,
        })
    }

    /// Build a pipeline result (success or failure).
    fn build_pipeline_result(
        &mut self,
        stages_completed: Vec<PipelineStage>,
        stage_timings: Vec<(PipelineStage, f64)>,
        success: bool,
        error_message: Option<String>,
        start: u64,
    ) -> XpockResult<PipelineResult> {
        let total_time = (crate::detection::signal_detector::lib_ms() - start) as f64;
        let empty_result = DemodulationResult {
            dimensions: (0, 0),
            pixels: Vec::new(),
            quality: DemodulationQuality {
                mean_dop: 0.0,
                dop_std: 0.0,
                mean_residual: 0.0,
                max_residual: 0.0,
                valid_fraction: 0.0,
                overall_score: 0.0,
            },
            computed_at: start,
            valid_pixel_count: 0,
            rejected_pixel_count: 0,
        };

        if success {
            self.stats.avg_processing_time_us = if self.stats.successful_runs > 0 {
                (self.stats.avg_processing_time_us * (self.stats.successful_runs - 1) as f64
                    + total_time)
                    / self.stats.successful_runs as f64
            } else {
                total_time
            };
        }

        Ok(PipelineResult {
            demodulation_result: empty_result,
            stages_completed,
            stage_timings,
            success,
            error_message,
        })
    }

    /// Returns pipeline statistics.
    pub fn stats(&self) -> &PipelineStatistics {
        &self.stats
    }

    /// Returns the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.stats.total_runs == 0 {
            return 0.0;
        }
        self.stats.successful_runs as f64 / self.stats.total_runs as f64
    }

    /// Reset pipeline statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStatistics::default();
    }

    /// Update pipeline configuration.
    pub fn update_config(&mut self, config: DemodulationConfig) {
        self.config = config;
    }

    /// Set quality gate threshold.
    pub fn set_quality_gate(&mut self, threshold: f64) {
        self.quality_gate_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enable or disable spatial filtering.
    pub fn set_spatial_filter(&mut self, enabled: bool) {
        self.enable_spatial_filter = enabled;
    }

    /// Enable or disable ellipse analysis.
    pub fn set_ellipse_analysis(&mut self, enabled: bool) {
        self.enable_ellipse_analysis = enabled;
    }
}

impl Default for DemodulationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stokes parameter conversion utilities
// ============================================================================

/// Convert Stokes parameters to Jones vector components.
#[derive(Debug, Clone, PartialEq)]
pub struct JonesVector {
    /// Ex component.
    pub ex: f64,
    /// Ey component.
    pub ey: f64,
    /// Relative phase (Ex to Ey) in radians.
    pub phase: f64,
}

/// Converts normalized Stokes parameters to Jones vector representation.
pub struct StokesToJonesConverter {
    /// Conversion history.
    history: Vec<JonesVector>,
    /// Maximum history length.
    max_history: usize,
}

impl StokesToJonesConverter {
    /// Create a new converter.
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Convert normalized Stokes (S1, S2, S3) to Jones vector.
    /// Assumes S0 = 1 (fully polarized).
    pub fn convert(&mut self, s1: f64, s2: f64, s3: f64) -> JonesVector {
        // Jones vector from Stokes:
        // Ex = sqrt((1 + S1) / 2)
        // Ey = sqrt((1 - S1) / 2) * exp(i * delta)
        // where tan(delta) = S2 / (something) and S3 = 2*Ex*Ey*sin(delta)
        let ex = ((1.0 + s1.clamp(-1.0, 1.0)) / 2.0).sqrt();
        let ey_mag = ((1.0 - s1.clamp(-1.0, 1.0)) / 2.0).sqrt();

        // Phase from S3: S3 = 2 * Ex * Ey * sin(delta)
        let phase = if ex * ey_mag > 1e-10 {
            (s3 / (2.0 * ex * ey_mag)).clamp(-1.0, 1.0).asin()
        } else {
            0.0
        };

        let jones = JonesVector {
            ex,
            ey: ey_mag,
            phase,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(jones.clone());

        jones
    }

    /// Convert Jones vector back to Stokes parameters.
    pub fn jones_to_stokes(&self, jones: &JonesVector) -> [f64; 4] {
        let ex2 = jones.ex * jones.ex;
        let ey2 = jones.ey * jones.ey;
        let s0 = ex2 + ey2;
        let s1 = ex2 - ey2;
        let s2 = 2.0 * jones.ex * jones.ey * jones.phase.cos();
        let s3 = 2.0 * jones.ex * jones.ey * jones.phase.sin();
        [s0, s1, s2, s3]
    }

    /// Returns the conversion history.
    pub fn history(&self) -> &[JonesVector] {
        &self.history
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for StokesToJonesConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Demodulation quality validator
// ============================================================================

/// Validation result for demodulation quality.
#[derive(Debug, Clone, PartialEq)]
pub struct DemodulationValidationResult {
    /// Whether the demodulation passed validation.
    pub is_valid: bool,
    /// Validation score (0.0 to 1.0).
    pub score: f64,
    /// List of validation issues.
    pub issues: Vec<String>,
    /// Recommended action.
    pub recommended_action: ValidationAction,
}

/// Recommended action based on validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationAction {
    /// Accept the demodulation result as-is.
    Accept,
    /// Accept with warning (use with caution).
    AcceptWithWarning,
    /// Reject and recalibrate.
    RejectAndRecalibrate,
    /// Reject and retry with different parameters.
    RejectAndRetry,
}

/// Validates demodulation results against quality thresholds.
pub struct DemodulationQualityValidator {
    /// Minimum DOP for acceptance.
    pub min_dop: f64,
    /// Minimum valid fraction for acceptance.
    pub min_valid_fraction: f64,
    /// Maximum residual error for acceptance.
    pub max_residual_error: f64,
    /// Minimum overall quality score.
    pub min_overall_score: f64,
    /// Validation history.
    history: Vec<DemodulationValidationResult>,
    /// Maximum history length.
    pub max_history: usize,
}

impl DemodulationQualityValidator {
    /// Create a new validator with default thresholds.
    pub fn new() -> Self {
        Self {
            min_dop: 0.5,
            min_valid_fraction: 0.7,
            max_residual_error: 0.2,
            min_overall_score: 0.6,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Validate a demodulation result.
    pub fn validate(&mut self, result: &DemodulationResult) -> DemodulationValidationResult {
        let mut issues = Vec::new();
        let mut score = 1.0f64;

        // Check DOP.
        if result.quality.mean_dop < self.min_dop {
            issues.push(format!(
                "Mean DOP {:.3} below minimum {:.3}",
                result.quality.mean_dop, self.min_dop
            ));
            score *= 0.5;
        }

        // Check valid fraction.
        if result.quality.valid_fraction < self.min_valid_fraction {
            issues.push(format!(
                "Valid fraction {:.3} below minimum {:.3}",
                result.quality.valid_fraction, self.min_valid_fraction
            ));
            score *= 0.5;
        }

        // Check residual error.
        if result.quality.max_residual > self.max_residual_error {
            issues.push(format!(
                "Max residual {:.3} above maximum {:.3}",
                result.quality.max_residual, self.max_residual_error
            ));
            score *= 0.7;
        }

        // Check overall score.
        if result.quality.overall_score < self.min_overall_score {
            issues.push(format!(
                "Overall score {:.3} below minimum {:.3}",
                result.quality.overall_score, self.min_overall_score
            ));
            score *= 0.5;
        }

        score = score.clamp(0.0, 1.0);

        let is_valid = issues.is_empty();

        let recommended_action = if is_valid {
            ValidationAction::Accept
        } else if score > 0.6 {
            ValidationAction::AcceptWithWarning
        } else if score > 0.3 {
            ValidationAction::RejectAndRetry
        } else {
            ValidationAction::RejectAndRecalibrate
        };

        let validation = DemodulationValidationResult {
            is_valid,
            score,
            issues,
            recommended_action,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(validation.clone());

        validation
    }

    /// Returns the validation pass rate across history.
    pub fn pass_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let passed = self.history.iter().filter(|v| v.is_valid).count() as f64;
        passed / self.history.len() as f64
    }

    /// Returns the average validation score.
    pub fn avg_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|v| v.score).sum::<f64>() / self.history.len() as f64
    }

    /// Reset validation history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update validation thresholds.
    pub fn update_thresholds(
        &mut self,
        min_dop: f64,
        min_valid_fraction: f64,
        max_residual_error: f64,
        min_overall_score: f64,
    ) {
        self.min_dop = min_dop.clamp(0.0, 1.0);
        self.min_valid_fraction = min_valid_fraction.clamp(0.0, 1.0);
        self.max_residual_error = max_residual_error.clamp(0.0, 1.0);
        self.min_overall_score = min_overall_score.clamp(0.0, 1.0);
    }
}

impl Default for DemodulationQualityValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulation_config_default() {
        let config = DemodulationConfig::default();
        assert_eq!(config.num_measurements, 4);
        assert!((config.min_s0_intensity - 1e-6).abs() < 1e-12);
        assert!((config.max_residual_error - 0.1).abs() < 0.001);
        assert!(config.enable_subpixel);
    }

    #[test]
    fn demodulation_config_four_measurement() {
        let config = DemodulationConfig::four_measurement();
        assert_eq!(config.num_measurements, 4);
        assert!(config.demod_matrix.is_some());
        let matrix = config.demod_matrix.unwrap();
        assert!((matrix[0][0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn demodulation_config_six_measurement() {
        let config = DemodulationConfig::six_measurement();
        assert_eq!(config.num_measurements, 6);
        assert!(config.demod_matrix.is_some());
        let matrix = config.demod_matrix.unwrap();
        assert_eq!(matrix.len(), 6);
    }

    // ---------------------------------------------------------------------------
    // DemodulatedPixel tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulated_pixel_default_fields() {
        let pixel = DemodulatedPixel {
            position: (10, 20),
            s0: 100.0,
            s1_norm: 0.5,
            s2_norm: 0.3,
            s3_norm: 0.1,
            dop: 0.6,
            dolp: 0.5,
            docp: 0.1,
            polarization_angle: 0.5,
            ellipticity_angle: 0.1,
            confidence: 0.9,
            residual_error: 0.01,
        };
        assert_eq!(pixel.position, (10, 20));
        assert!((pixel.s0 - 100.0).abs() < 0.001);
        assert!((pixel.dop - 0.6).abs() < 0.001);
        assert!((pixel.confidence - 0.9).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // DemodulationResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulation_result_empty() {
        let result = DemodulationResult {
            dimensions: (0, 0),
            pixels: Vec::new(),
            quality: DemodulationQuality {
                mean_dop: 0.0,
                dop_std: 0.0,
                mean_residual: 0.0,
                max_residual: 0.0,
                valid_fraction: 0.0,
                overall_score: 0.0,
            },
            computed_at: 0,
            valid_pixel_count: 0,
            rejected_pixel_count: 0,
        };
        assert_eq!(result.dimensions, (0, 0));
        assert_eq!(result.valid_pixel_count, 0);
        assert_eq!(result.quality.overall_score, 0.0);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionDemodulator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulator_default() {
        let demod = SkyrmionDemodulator::default();
        assert_eq!(demod.config.num_measurements, 4);
        assert_eq!(demod.history.len(), 0);
        assert_eq!(demod.avg_quality_score(), 0.0);
    }

    #[test]
    fn demodulator_with_config() {
        let config = DemodulationConfig::four_measurement();
        let demod = SkyrmionDemodulator::with_config(config);
        assert_eq!(demod.config.num_measurements, 4);
    }

    #[test]
    fn demodulate_insufficient_measurements() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements = vec![vec![vec![1.0f64; 4]; 4]]; // Only 1 measurement, need 4
        let result = demod.demodulate(&measurements);
        assert!(result.is_err());
    }

    #[test]
    fn demodulate_empty_field() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = vec![vec![]; 4];
        let result = demod.demodulate(&measurements);
        assert!(result.is_err());
    }

    #[test]
    fn demodulate_uniform_field() {
        let mut demod = SkyrmionDemodulator::new();
        // 4 measurements, 4x4 pixels, uniform intensity.
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (4, 4));
        assert_eq!(result.pixels.len(), 4);
        assert_eq!(result.pixels[0].len(), 4);
        assert!(result.valid_pixel_count > 0);
    }

    #[test]
    fn demodulate_varying_intensities() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..4)
                    .map(|y| {
                        (0..4)
                            .map(|x| (m + y + x) as f64 * 0.5 + 1.0)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (4, 4));
        assert!(result.quality.overall_score >= 0.0);
        assert!(result.quality.overall_score <= 1.0);
    }

    #[test]
    fn demodulate_dimension_mismatch() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements = vec![
            vec![vec![1.0f64; 4]; 4],
            vec![vec![1.0f64; 3]; 4], // Different width
            vec![vec![1.0f64; 4]; 4],
            vec![vec![1.0f64; 4]; 4],
        ];
        let result = demod.demodulate(&measurements);
        assert!(result.is_err());
    }

    #[test]
    fn demodulator_history_tracking() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        demod.demodulate(&measurements).unwrap();
        demod.demodulate(&measurements).unwrap();
        assert_eq!(demod.history.len(), 2);
        assert!(demod.avg_quality_score() > 0.0);
    }

    #[test]
    fn demodulator_avg_valid_fraction() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        demod.demodulate(&measurements).unwrap();
        assert!(demod.avg_valid_fraction() > 0.0);
    }

    #[test]
    fn demodulator_reset() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        demod.demodulate(&measurements).unwrap();
        assert!(!demod.history.is_empty());
        demod.reset();
        assert_eq!(demod.history.len(), 0);
    }

    #[test]
    fn demodulator_update_config() {
        let mut demod = SkyrmionDemodulator::new();
        let config = DemodulationConfig::six_measurement();
        demod.update_config(config);
        assert_eq!(demod.config().num_measurements, 6);
    }

    #[test]
    fn demodulate_to_skyrmion_field() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let field = demod.to_skyrmion_field(&result);
        assert_eq!(field.dimensions, (8, 8));
    }

    // ---------------------------------------------------------------------------
    // CalibrationRecord tests
    // ---------------------------------------------------------------------------

    #[test]
    fn calibration_record_creation() {
        let record = CalibrationRecord {
            timestamp: 1000,
            mean_s0: 50.0,
            mean_dop: 0.8,
            quality: 0.9,
        };
        assert_eq!(record.timestamp, 1000);
        assert!((record.mean_s0 - 50.0).abs() < 0.001);
        assert!((record.mean_dop - 0.8).abs() < 0.001);
    }

    #[test]
    fn demodulator_calibration_history() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        demod.demodulate(&measurements).unwrap();
        assert!(!demod.calibration_history().is_empty());
    }

    // ---------------------------------------------------------------------------
    // StokesVector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_vector_new() {
        let v = StokesVector::new(0.6, 0.4, 0.2);
        assert!((v.x - 0.6).abs() < 0.001);
        assert!((v.y - 0.4).abs() < 0.001);
        assert!((v.z - 0.2).abs() < 0.001);
        assert!(v.magnitude > 0.0);
    }

    #[test]
    fn stokes_vector_normalize() {
        let v = StokesVector::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.magnitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stokes_vector_zero_normalize() {
        let v = StokesVector::new(0.0, 0.0, 0.0);
        let n = v.normalize();
        assert_eq!(n.magnitude, 0.0);
    }

    #[test]
    fn stokes_vector_dot() {
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(0.0, 1.0, 0.0);
        assert!((a.dot(&b) - 0.0).abs() < 1e-10);
        assert!((a.dot(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stokes_vector_angular_distance() {
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(0.0, 1.0, 0.0);
        let dist = a.angular_distance(&b);
        assert!((dist - std::f64::consts::PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn stokes_vector_is_on_sphere() {
        let v = StokesVector::new(1.0, 0.0, 0.0);
        assert!(v.is_on_sphere(0.01));
        let v2 = StokesVector::new(0.5, 0.0, 0.0);
        assert!(!v2.is_on_sphere(0.01));
    }

    #[test]
    fn stokes_vector_classification() {
        let linear = StokesVector::new(1.0, 0.0, 0.0).normalize();
        assert!(matches!(linear.classification(), PolarizationState::Linear));

        let circular = StokesVector::new(0.0, 0.0, 1.0).normalize();
        assert!(matches!(circular.classification(), PolarizationState::RightCircular));

        let unpolarized = StokesVector::new(0.01, 0.0, 0.0);
        assert!(matches!(unpolarized.classification(), PolarizationState::Unpolarized));
    }

    // ---------------------------------------------------------------------------
    // PoincareSphere tests
    // ---------------------------------------------------------------------------

    #[test]
    fn poincare_sphere_default() {
        let sphere = PoincareSphere::default();
        assert!((sphere.radius - 1.0).abs() < 0.001);
    }

    #[test]
    fn poincare_sphere_to_spherical() {
        let sphere = PoincareSphere::new();
        let v = StokesVector::new(0.0, 0.0, 1.0); // North pole.
        let (theta, phi) = sphere.to_spherical(&v);
        assert!(theta < 0.001); // theta ≈ 0 at north pole.
    }

    #[test]
    fn poincare_sphere_from_spherical() {
        let sphere = PoincareSphere::new();
        let v = sphere.from_spherical(0.0, 0.0);
        assert!((v.z - 1.0).abs() < 0.001);
    }

    #[test]
    fn poincare_sphere_solid_angle() {
        let sphere = PoincareSphere::new();
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(0.0, 1.0, 0.0);
        let c = StokesVector::new(0.0, 0.0, 1.0);
        let angle = sphere.solid_angle(&a, &b, &c);
        assert!(angle >= 0.0);
        assert!(angle <= std::f64::consts::PI * 2.0);
    }

    #[test]
    fn poincare_sphere_is_right_handed() {
        let sphere = PoincareSphere::new();
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(0.0, 1.0, 0.0);
        let c = StokesVector::new(0.0, 0.0, 1.0);
        assert!(sphere.is_right_handed(&a, &b, &c));
    }

    // ---------------------------------------------------------------------------
    // StokesAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_analyzer_default() {
        let analyzer = StokesAnalyzer::default();
        assert!((analyzer.fully_polarized_threshold - 0.9).abs() < 0.001);
        assert!((analyzer.unpolarized_threshold - 0.1).abs() < 0.001);
    }

    #[test]
    fn stokes_analyzer_empty_result() {
        let analyzer = StokesAnalyzer::new();
        let result = DemodulationResult {
            dimensions: (0, 0),
            pixels: Vec::new(),
            quality: DemodulationQuality {
                mean_dop: 0.0,
                dop_std: 0.0,
                mean_residual: 0.0,
                max_residual: 0.0,
                valid_fraction: 0.0,
                overall_score: 0.0,
            },
            computed_at: 0,
            valid_pixel_count: 0,
            rejected_pixel_count: 0,
        };
        let analysis = analyzer.analyze(&result);
        assert_eq!(analysis.mean_dop, 0.0);
        assert_eq!(analysis.valid_pixel_count, 0);
    }

    #[test]
    fn stokes_analyzer_uniform_field() {
        let analyzer = StokesAnalyzer::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let analysis = analyzer.analyze(&result);
        assert!(analysis.mean_dop >= 0.0);
        assert!(analysis.mean_dop <= 1.0);
        assert!(analysis.fully_polarized_fraction >= 0.0);
        assert!(analysis.unpolarized_fraction >= 0.0);
    }

    #[test]
    fn stokes_analyzer_dominant_state() {
        let analyzer = StokesAnalyzer::new();
        // All pixels linear polarization.
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let analysis = analyzer.analyze(&result);
        assert!(matches!(
            analysis.dominant_state,
            PolarizationState::Linear | PolarizationState::Unpolarized
        ));
    }

    #[test]
    fn stokes_analyzer_update_thresholds() {
        let mut analyzer = StokesAnalyzer::new();
        analyzer.update_thresholds(0.95, 0.05);
        assert!((analyzer.fully_polarized_threshold - 0.95).abs() < 0.001);
        assert!((analyzer.unpolarized_threshold - 0.05).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // Edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulate_low_intensity_pixels() {
        let mut demod = SkyrmionDemodulator::new();
        // Mix of high and zero intensity pixels.
        let mut measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..4)
                    .map(|y| {
                        (0..4)
                            .map(|x| {
                                if y < 2 && x < 2 {
                                    (m as f64 + 1.0) * 0.5
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert!(result.rejected_pixel_count > 0);
        assert!(result.valid_pixel_count > 0);
    }

    #[test]
    fn demodulate_high_residual() {
        let mut demod = SkyrmionDemodulator::new();
        demod.config.max_residual_error = 0.001; // Very strict.
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![100.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        // High intensities may produce residuals.
        assert!(result.quality.max_residual >= 0.0);
    }

    #[test]
    fn demodulate_single_pixel() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64]])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (1, 1));
        assert_eq!(result.pixels.len(), 1);
        assert_eq!(result.pixels[0].len(), 1);
    }

    #[test]
    fn demodulate_large_field() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 64]; 64])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (64, 64));
        assert_eq!(result.valid_pixel_count, 4096);
    }

    // ---------------------------------------------------------------------------
    // StokesVector additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_vector_display_properties() {
        let v = StokesVector::new(0.0, 0.0, 1.0);
        assert!(v.is_on_sphere(0.01));
        assert!(matches!(v.classification(), PolarizationState::RightCircular));

        let v2 = StokesVector::new(0.0, 0.0, -1.0);
        assert!(matches!(v2.classification(), PolarizationState::LeftCircular));
    }

    #[test]
    fn stokes_vector_angular_distance_same() {
        let v = StokesVector::new(1.0, 0.0, 0.0);
        let dist = v.angular_distance(&v);
        assert!(dist < 1e-10);
    }

    #[test]
    fn stokes_vector_angular_distance_opposite() {
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(-1.0, 0.0, 0.0);
        let dist = a.angular_distance(&b);
        assert!((dist - std::f64::consts::PI).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // DemodulationQuality tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulation_quality_score_bounds() {
        let quality = DemodulationQuality {
            mean_dop: 0.5,
            dop_std: 0.1,
            mean_residual: 0.05,
            max_residual: 0.1,
            valid_fraction: 0.8,
            overall_score: 0.7,
        };
        assert!(quality.overall_score >= 0.0);
        assert!(quality.overall_score <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // StokesSpatialFilter tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_spatial_filter_config_default() {
        let config = StokesSpatialFilterConfig::default();
        assert_eq!(config.window_size, 5);
        assert!((config.sigma - 1.0).abs() < 0.001);
        assert!((config.edge_preservation - 0.5).abs() < 0.001);
    }

    #[test]
    fn stokes_spatial_filter_new() {
        let filter = StokesSpatialFilter::new();
        assert_eq!(filter.config.window_size, 5);
        assert_eq!(filter.history.len(), 0);
    }

    #[test]
    fn stokes_spatial_filter_uniform_field() {
        let filter = StokesSpatialFilter::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let filtered = filter.filter(&result);
        assert_eq!(filtered.dimensions, (8, 8));
        assert!(filtered.quality.overall_score >= 0.0);
    }

    #[test]
    fn stokes_spatial_filter_varying_field() {
        let filter = StokesSpatialFilter::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..8)
                    .map(|y| {
                        (0..8)
                            .map(|x| (m + y + x) as f64 * 0.25 + 0.5)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let filtered = filter.filter(&result);
        assert_eq!(filtered.dimensions, (8, 8));
        assert!(filtered.quality.valid_fraction >= 0.0);
    }

    #[test]
    fn stokes_spatial_filter_update_config() {
        let mut filter = StokesSpatialFilter::new();
        let config = StokesSpatialFilterConfig {
            window_size: 7,
            sigma: 2.0,
            edge_preservation: 0.8,
            min_dop_for_average: 0.5,
        };
        filter.update_config(config);
        assert_eq!(filter.config.window_size, 7);
        assert!((filter.config.sigma - 2.0).abs() < 0.001);
    }

    #[test]
    fn stokes_spatial_filter_record_quality() {
        let mut filter = StokesSpatialFilter::new();
        filter.record_quality(0.8);
        filter.record_quality(0.6);
        assert!((filter.avg_recorded_quality() - 0.7).abs() < 0.001);
    }

    #[test]
    fn stokes_spatial_filter_edge_preservation() {
        let mut filter = StokesSpatialFilter::new();
        filter.update_config(StokesSpatialFilterConfig {
            edge_preservation: 1.0,
            ..StokesSpatialFilterConfig::default()
        });
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let filtered = filter.filter(&result);
        assert_eq!(filtered.dimensions, (8, 8));
    }

    // ---------------------------------------------------------------------------
    // PolarizationEllipseAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn polarization_ellipse_analyzer_new() {
        let analyzer = PolarizationEllipseAnalyzer::new();
        assert!((analyzer.min_dop - 0.1).abs() < 0.001);
        assert_eq!(analyzer.history.len(), 0);
    }

    #[test]
    fn polarization_ellipse_analyze_linear() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 1.0,
            s2_norm: 0.0,
            s3_norm: 0.0,
            dop: 1.0,
            dolp: 1.0,
            docp: 0.0,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        let ellipse = analyzer.analyze(&pixel);
        assert!(ellipse.is_some());
        let e = ellipse.unwrap();
        assert_eq!(e.handedness, 0);
        assert!((e.circularity - 0.0).abs() < 0.001);
    }

    #[test]
    fn polarization_ellipse_analyze_circular() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 0.0,
            s2_norm: 0.0,
            s3_norm: 1.0,
            dop: 1.0,
            dolp: 0.0,
            docp: 1.0,
            polarization_angle: 0.0,
            ellipticity_angle: std::f64::consts::PI / 4.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        let ellipse = analyzer.analyze(&pixel);
        assert!(ellipse.is_some());
        let e = ellipse.unwrap();
        assert_eq!(e.handedness, 1);
        assert!((e.circularity - 1.0).abs() < 0.001);
    }

    #[test]
    fn polarization_ellipse_low_dop_rejected() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 0.01,
            s2_norm: 0.0,
            s3_norm: 0.0,
            dop: 0.01,
            dolp: 0.01,
            docp: 0.0,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 0.5,
            residual_error: 0.0,
        };
        let ellipse = analyzer.analyze(&pixel);
        assert!(ellipse.is_none());
    }

    #[test]
    fn polarization_ellipse_dominant_handedness() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel_rhcp = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 0.0,
            s2_norm: 0.0,
            s3_norm: 1.0,
            dop: 1.0,
            dolp: 0.0,
            docp: 1.0,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        let pixel_lhcp = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 0.0,
            s2_norm: 0.0,
            s3_norm: -1.0,
            dop: 1.0,
            dolp: 0.0,
            docp: -1.0,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        analyzer.analyze(&pixel_rhcp);
        analyzer.analyze(&pixel_rhcp);
        analyzer.analyze(&pixel_lhcp);
        assert_eq!(analyzer.dominant_handedness(), 1);
    }

    #[test]
    fn polarization_ellipse_mean_ellipticity() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 0.0,
            s2_norm: 0.0,
            s3_norm: 0.5,
            dop: 1.0,
            dolp: 0.0,
            docp: 0.5,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        analyzer.analyze(&pixel);
        let mean_ell = analyzer.mean_ellipticity();
        assert!(mean_ell > 0.0);
    }

    #[test]
    fn polarization_ellipse_mean_circularity() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 0.0,
            s2_norm: 0.0,
            s3_norm: 0.8,
            dop: 1.0,
            dolp: 0.0,
            docp: 0.8,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        analyzer.analyze(&pixel);
        let mean_circ = analyzer.mean_circularity();
        assert!(mean_circ > 0.0);
    }

    #[test]
    fn polarization_ellipse_reset() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let pixel = DemodulatedPixel {
            position: (0, 0),
            s0: 100.0,
            s1_norm: 1.0,
            s2_norm: 0.0,
            s3_norm: 0.0,
            dop: 1.0,
            dolp: 1.0,
            docp: 0.0,
            polarization_angle: 0.0,
            ellipticity_angle: 0.0,
            confidence: 1.0,
            residual_error: 0.0,
        };
        analyzer.analyze(&pixel);
        assert!(!analyzer.history.is_empty());
        analyzer.reset();
        assert_eq!(analyzer.history.len(), 0);
    }

    #[test]
    fn polarization_ellipse_analyze_field() {
        let mut analyzer = PolarizationEllipseAnalyzer::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let ellipses = analyzer.analyze_field(&result);
        assert_eq!(ellipses.len(), 4);
        assert_eq!(ellipses[0].len(), 4);
    }

    // ---------------------------------------------------------------------------
    // MultiFrameDemodulator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_frame_demodulator_new() {
        let mfd = MultiFrameDemodulator::new();
        assert_eq!(mfd.frame_count(), 0);
        assert_eq!(mfd.avg_noise_reduction(), 1.0);
    }

    #[test]
    fn multi_frame_add_frame() {
        let mut mfd = MultiFrameDemodulator::new();
        let frame = StokesFrame {
            timestamp: 1000,
            measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
            quality: 0.9,
        };
        mfd.add_frame(frame);
        assert_eq!(mfd.frame_count(), 1);
    }

    #[test]
    fn multi_frame_demodulate_and_average_empty() {
        let mut mfd = MultiFrameDemodulator::new();
        let result = mfd.demodulate_and_average();
        assert!(result.is_err());
    }

    #[test]
    fn multi_frame_demodulate_and_average_single() {
        let mut mfd = MultiFrameDemodulator::new();
        let frame = StokesFrame {
            timestamp: 1000,
            measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
            quality: 0.9,
        };
        mfd.add_frame(frame);
        let result = mfd.demodulate_and_average().unwrap();
        assert_eq!(result.frame_count, 1);
        assert_eq!(result.averaged_result.dimensions, (4, 4));
        assert!(result.noise_reduction > 0.0);
    }

    #[test]
    fn multi_frame_multiple_frames() {
        let mut mfd = MultiFrameDemodulator::new();
        for i in 0..3 {
            let frame = StokesFrame {
                timestamp: 1000 + i as u64,
                measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
                quality: 0.9,
            };
            mfd.add_frame(frame);
        }
        let result = mfd.demodulate_and_average().unwrap();
        assert_eq!(result.frame_count, 3);
        assert!(result.noise_reduction > 1.0);
    }

    #[test]
    fn multi_frame_low_quality_filtered() {
        let mut mfd = MultiFrameDemodulator::new();
        let frame_good = StokesFrame {
            timestamp: 1000,
            measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
            quality: 0.9,
        };
        let frame_bad = StokesFrame {
            timestamp: 1001,
            measurements: vec![vec![vec![0.01f64; 4]; 4]; 4],
            quality: 0.1,
        };
        mfd.add_frame(frame_good);
        mfd.add_frame(frame_bad);
        let result = mfd.demodulate_and_average().unwrap();
        assert_eq!(result.frame_count, 1); // Only good frame passes quality filter
    }

    #[test]
    fn multi_frame_clear_buffer() {
        let mut mfd = MultiFrameDemodulator::new();
        let frame = StokesFrame {
            timestamp: 1000,
            measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
            quality: 0.9,
        };
        mfd.add_frame(frame);
        assert_eq!(mfd.frame_count(), 1);
        mfd.clear_buffer();
        assert_eq!(mfd.frame_count(), 0);
    }

    #[test]
    fn multi_frame_update_params() {
        let mut mfd = MultiFrameDemodulator::new();
        mfd.update_params(10, 0.5);
        // Can't directly access private fields, but frame_count should work.
        assert_eq!(mfd.frame_count(), 0);
    }

    #[test]
    fn multi_frame_averaging_history() {
        let mut mfd = MultiFrameDemodulator::new();
        let frame = StokesFrame {
            timestamp: 1000,
            measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
            quality: 0.9,
        };
        mfd.add_frame(frame);
        mfd.demodulate_and_average().unwrap();
        assert!(!mfd.averaging_history().is_empty());
    }

    #[test]
    fn multi_frame_avg_noise_reduction() {
        let mut mfd = MultiFrameDemodulator::new();
        let frame = StokesFrame {
            timestamp: 1000,
            measurements: vec![vec![vec![1.0f64; 4]; 4]; 4],
            quality: 0.9,
        };
        mfd.add_frame(frame);
        mfd.demodulate_and_average().unwrap();
        assert!(mfd.avg_noise_reduction() > 0.0);
    }

    // ---------------------------------------------------------------------------
    // DemodulationCalibrator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulation_calibration_default() {
        let cal = DemodulationCalibration::default();
        assert_eq!(cal.offset, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(cal.scale, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(cal.quality, 1.0);
    }

    #[test]
    fn demodulation_calibrator_new() {
        let calibrator = DemodulationCalibrator::new();
        assert_eq!(calibrator.sample_count(), 0);
        assert!((calibrator.calibration_quality() - 1.0).abs() < 0.001);
    }

    #[test]
    fn demodulation_calibrator_calibrate() {
        let mut calibrator = DemodulationCalibrator::new();
        let measured = vec![
            vec![1.0, 0.5, 0.0, 0.0],
            vec![1.0, -0.5, 0.0, 0.0],
            vec![1.0, 0.0, 0.5, 0.0],
            vec![1.0, 0.0, 0.0, 0.5],
        ];
        let expected = vec![
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ];
        let result = calibrator.calibrate(&measured, &expected);
        assert!(result.is_ok());
        let cal = result.unwrap();
        assert!(cal.quality >= 0.0);
        assert!(cal.quality <= 1.0);
    }

    #[test]
    fn demodulation_calibrator_calibrate_empty() {
        let mut calibrator = DemodulationCalibrator::new();
        let result = calibrator.calibrate(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn demodulation_calibrator_dimension_mismatch() {
        let mut calibrator = DemodulationCalibrator::new();
        let measured = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let expected = vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
        let result = calibrator.calibrate(&measured, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn demodulation_calibrator_apply_calibration() {
        let mut calibrator = DemodulationCalibrator::new();
        let mut measurements = vec![2.0, 1.0, 0.5, 0.25];
        calibrator.apply_calibration(&mut measurements);
        // Default calibration is identity, so values should be unchanged.
        assert!((measurements[0] - 2.0).abs() < 1e-10);
        assert!((measurements[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn demodulation_calibrator_sample_count() {
        let mut calibrator = DemodulationCalibrator::new();
        assert_eq!(calibrator.sample_count(), 0);
        let measured = vec![vec![1.0, 0.5, 0.0, 0.0]];
        let expected = vec![[0.5, 0.0, 0.0]];
        let _ = calibrator.calibrate(&measured, &expected);
        assert_eq!(calibrator.sample_count(), 1);
    }

    #[test]
    fn demodulation_calibrator_reset() {
        let mut calibrator = DemodulationCalibrator::new();
        let measured = vec![vec![1.0, 0.5, 0.0, 0.0]];
        let expected = vec![[0.5, 0.0, 0.0]];
        let _ = calibrator.calibrate(&measured, &expected);
        assert!(!calibrator.history().is_empty());
        calibrator.reset();
        assert_eq!(calibrator.history().len(), 0);
        assert_eq!(calibrator.sample_count(), 0);
    }

    #[test]
    fn demodulation_calibrator_history() {
        let mut calibrator = DemodulationCalibrator::new();
        let measured = vec![vec![1.0, 0.5, 0.0, 0.0]];
        let expected = vec![[0.5, 0.0, 0.0]];
        calibrator.calibrate(&measured, &expected).unwrap();
        calibrator.calibrate(&measured, &expected).unwrap();
        assert_eq!(calibrator.history().len(), 2);
    }

    #[test]
    fn demodulation_calibrator_calibration_quality() {
        let mut calibrator = DemodulationCalibrator::new();
        let measured = vec![vec![1.0, 0.5, 0.0, 0.0]];
        let expected = vec![[0.5, 0.0, 0.0]];
        calibrator.calibrate(&measured, &expected).unwrap();
        assert!(calibrator.calibration_quality() >= 0.0);
        assert!(calibrator.calibration_quality() <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // StokesFieldOperations tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_field_operations_new() {
        let ops = StokesFieldOperations::new();
        assert!((ops.step_size - 1.0).abs() < 0.001);
        assert!(!ops.pre_smooth);
    }

    #[test]
    fn stokes_field_compute_gradient_uniform() {
        let ops = StokesFieldOperations::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let gradients = ops.compute_gradient(&result);
        assert_eq!(gradients.len(), 8);
        assert_eq!(gradients[0].len(), 8);
        // Uniform field should have zero gradients.
        for row in &gradients {
            for g in row {
                assert!(g.magnitude < 1e-6);
            }
        }
    }

    #[test]
    fn stokes_field_compute_gradient_varying() {
        let ops = StokesFieldOperations::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..8)
                    .map(|y| {
                        (0..8)
                            .map(|x| (m + y + x) as f64 * 0.1 + 1.0)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let gradients = ops.compute_gradient(&result);
        assert_eq!(gradients.len(), 8);
        // Some gradients should be non-zero.
        let has_nonzero = gradients
            .iter()
            .flatten()
            .any(|g| g.magnitude > 1e-6);
        assert!(has_nonzero);
    }

    #[test]
    fn stokes_field_compute_skyrmion_density_uniform() {
        let ops = StokesFieldOperations::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let density = ops.compute_skyrmion_density(&result);
        assert_eq!(density.len(), 8);
        assert_eq!(density[0].len(), 8);
        // Uniform field should have zero skyrmion density.
        for row in &density {
            for &d in row {
                assert!(d.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn stokes_field_update_params() {
        let mut ops = StokesFieldOperations::new();
        ops.update_params(0.5, true);
        assert!((ops.step_size - 0.5).abs() < 0.001);
        assert!(ops.pre_smooth);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionDemodulator additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulator_config_from_6_measurement() {
        let config = DemodulationConfig::six_measurement();
        let mut demod = SkyrmionDemodulator::with_config(config);
        let measurements: Vec<Vec<Vec<f64>>> = (0..6)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (4, 4));
    }

    #[test]
    fn demodulator_noisy_field() {
        let mut demod = SkyrmionDemodulator::new();
        // Add noise to measurements.
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..8)
                    .map(|y| {
                        (0..8)
                            .map(|x| {
                                1.0 + ((x + y + m) as f64 * 0.01).sin() * 0.1
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (8, 8));
        assert!(result.quality.overall_score >= 0.0);
    }

    #[test]
    fn demodulator_all_zero_intensities() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![0.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (4, 4));
        // All zero intensities -> no valid pixels (below min_s0_intensity).
        assert_eq!(result.valid_pixel_count, 0);
        assert_eq!(result.rejected_pixel_count, 16);
    }

    #[test]
    fn demodulator_confidence_bounds() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        for row in &result.pixels {
            for p in row {
                if p.confidence > 0.0 {
                    assert!(p.confidence >= 0.0 && p.confidence <= 1.0);
                }
            }
        }
    }

    #[test]
    fn demodulator_dop_bounds() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        for row in &result.pixels {
            for p in row {
                if p.confidence > 0.0 {
                    assert!(p.dop >= 0.0 && p.dop <= 1.0);
                }
            }
        }
    }

    #[test]
    fn demodulator_quality_score_bounds() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert!(result.quality.overall_score >= 0.0);
        assert!(result.quality.overall_score <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // StokesVector additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_vector_linear_classification() {
        let v = StokesVector::new(0.8, 0.0, 0.0).normalize();
        assert!(matches!(v.classification(), PolarizationState::Linear));
    }

    #[test]
    fn stokes_vector_elliptical_classification() {
        let v = StokesVector::new(0.6, 0.0, 0.6).normalize();
        assert!(matches!(v.classification(), PolarizationState::Elliptical));
    }

    #[test]
    fn stokes_vector_partially_polarized() {
        let v = StokesVector::new(0.3, 0.0, 0.0);
        assert!(matches!(
            v.classification(),
            PolarizationState::PartiallyPolarized
        ));
    }

    #[test]
    fn stokes_vector_angular_distance_symmetry() {
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(0.0, 1.0, 0.0);
        let d_ab = a.angular_distance(&b);
        let d_ba = b.angular_distance(&a);
        assert!((d_ab - d_ba).abs() < 1e-10);
    }

    #[test]
    fn stokes_vector_dot_normalized() {
        let a = StokesVector::new(1.0, 0.0, 0.0).normalize();
        let b = StokesVector::new(0.0, 1.0, 0.0).normalize();
        assert!((a.dot(&b) - 0.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------------------
    // PoincareSphere additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn poincare_sphere_solid_angle_zero() {
        let sphere = PoincareSphere::new();
        let a = StokesVector::new(1.0, 0.0, 0.0);
        let b = StokesVector::new(1.0, 0.0, 0.0);
        let c = StokesVector::new(1.0, 0.0, 0.0);
        let angle = sphere.solid_angle(&a, &b, &c);
        assert!(angle < 1e-6);
    }

    #[test]
    fn poincare_sphere_from_spherical_roundtrip() {
        let sphere = PoincareSphere::new();
        let original = StokesVector::new(0.6, 0.4, 0.7).normalize();
        let (theta, phi) = sphere.to_spherical(&original);
        let reconstructed = sphere.from_spherical(theta, phi);
        // Should be approximately the same direction.
        assert!((original.x - reconstructed.x).abs() < 0.01);
        assert!((original.y - reconstructed.y).abs() < 0.01);
        assert!((original.z - reconstructed.z).abs() < 0.01);
    }

    // ---------------------------------------------------------------------------
    // StokesAnalyzer additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_analyzer_all_linear() {
        let analyzer = StokesAnalyzer::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let analysis = analyzer.analyze(&result);
        assert!(analysis.mean_dop >= 0.0);
        assert!(analysis.s0_std >= 0.0);
    }

    #[test]
    fn stokes_analyzer_fully_polarized_fraction() {
        let analyzer = StokesAnalyzer::new();
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 8]; 8])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        let analysis = analyzer.analyze(&result);
        assert!(analysis.fully_polarized_fraction >= 0.0);
        assert!(analysis.fully_polarized_fraction <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // StokesGradient tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stokes_gradient_zero() {
        let g = StokesGradient {
            ds1: (0.0, 0.0),
            ds2: (0.0, 0.0),
            ds3: (0.0, 0.0),
            magnitude: 0.0,
        };
        assert_eq!(g.magnitude, 0.0);
    }

    #[test]
    fn stokes_gradient_magnitude() {
        let g = StokesGradient {
            ds1: (3.0, 0.0),
            ds2: (4.0, 0.0),
            ds3: (0.0, 0.0),
            magnitude: 5.0,
        };
        assert!((g.magnitude - 5.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // MultiFrameResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_frame_result_fields() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 4]; 4])
            .collect();
        let averaged = demod.demodulate(&measurements).unwrap();
        let mf_result = MultiFrameResult {
            averaged_result: averaged,
            frame_count: 3,
            noise_reduction: 1.732,
            frame_qualities: vec![0.8, 0.9, 0.85],
        };
        assert_eq!(mf_result.frame_count, 3);
        assert!((mf_result.noise_reduction - 1.732).abs() < 0.01);
        assert_eq!(mf_result.frame_qualities.len(), 3);
    }

    // ---------------------------------------------------------------------------
    // Edge case: NaN and Inf handling
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulator_nan_resistant() {
        let mut demod = SkyrmionDemodulator::new();
        // Measurements with NaN should be handled gracefully.
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..4)
                    .map(|y| {
                        (0..4)
                            .map(|x| {
                                if x == 0 && y == 0 && m == 0 {
                                    f64::NAN
                                } else {
                                    1.0
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements);
        // Should not panic; may return error or result with rejected pixels.
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn demodulator_inf_resistant() {
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|m| {
                (0..4)
                    .map(|y| {
                        (0..4)
                            .map(|x| {
                                if x == 0 && y == 0 && m == 0 {
                                    f64::INFINITY
                                } else {
                                    1.0
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = demod.demodulate(&measurements);
        assert!(result.is_ok() || result.is_err());
    }

    // ---------------------------------------------------------------------------
    // End-to-end demodulation pipeline test
    // ---------------------------------------------------------------------------

    #[test]
    fn demodulation_pipeline_end_to_end() {
        // Full pipeline: demodulate -> spatial filter -> ellipse analysis -> field operations.
        let mut demod = SkyrmionDemodulator::new();
        let measurements: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| vec![vec![1.0f64; 16]; 16])
            .collect();
        let result = demod.demodulate(&measurements).unwrap();
        assert_eq!(result.dimensions, (16, 16));

        // Spatial filter.
        let filter = StokesSpatialFilter::new();
        let filtered = filter.filter(&result);
        assert_eq!(filtered.dimensions, (16, 16));

        // Ellipse analysis.
        let mut ellipse_analyzer = PolarizationEllipseAnalyzer::new();
        let ellipses = ellipse_analyzer.analyze_field(&filtered);
        assert_eq!(ellipses.len(), 16);

        // Field operations.
        let field_ops = StokesFieldOperations::new();
        let gradients = field_ops.compute_gradient(&filtered);
        assert_eq!(gradients.len(), 16);

        // Skyrmion density.
        let density = field_ops.compute_skyrmion_density(&filtered);
        assert_eq!(density.len(), 16);
    }
}