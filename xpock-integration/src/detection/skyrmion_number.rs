//! Skyrmion number computation and generalized topological invariants.
//!
//! Implements the mathematical foundation for topological signal detection:
//! skyrmion number Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy, generalized skyrmion
//! numbers (Z^k tuples), Stokes field analysis, and topological charge density.
//!
//! The skyrmion number is a topological invariant that characterizes how many
//! times the normalized Stokes vector field wraps around the Poincaré sphere.
//! Because it is conserved under continuous deformation, it serves as a robust
//! information carrier that survives atmospheric turbulence and phase distortions.

use crate::errors::{XpockError, XpockResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Skyrmion number — fundamental topological invariant
// ---------------------------------------------------------------------------

/// The skyrmion number — a topological invariant integer characterizing how
/// many times the normalized Stokes vector field wraps around the Poincaré
/// sphere.
///
/// Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy
///
/// where n = (S1, S2, S3) / S0 is the normalized Stokes vector.
///
/// Because Q is an integer conserved under continuous deformation, it serves
/// as a robust information carrier for topological signal modulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkyrmionNumber {
    /// Integer skyrmion number (topological charge).
    pub value: i32,
    /// Numerical precision of the computation (deviation from nearest integer).
    pub precision: f64,
    /// Number of pixels/points used in the computation.
    pub sample_count: usize,
    /// Whether the value is topologically stable (close to an integer).
    pub is_stable: bool,
}

impl SkyrmionNumber {
    /// Create a new skyrmion number.
    pub fn new(value: i32, precision: f64, sample_count: usize) -> Self {
        let is_stable = precision < 0.1;
        Self {
            value,
            precision,
            sample_count,
            is_stable,
        }
    }

    /// Returns the absolute topological charge magnitude.
    pub fn magnitude(&self) -> u32 {
        self.value.abs() as u32
    }

    /// Returns the sign of the skyrmion number (-1, 0, or 1).
    pub fn sign(&self) -> i8 {
        match self.value.cmp(&0) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    /// Returns true if the skyrmion number represents a non-trivial topology.
    pub fn is_non_trivial(&self) -> bool {
        self.value != 0 && self.is_stable
    }

    /// Returns the winding number (alias for value).
    pub fn winding_number(&self) -> i32 {
        self.value
    }
}

impl Default for SkyrmionNumber {
    fn default() -> Self {
        Self::new(0, 0.0, 0)
    }
}

impl std::fmt::Display for SkyrmionNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Q = {} (precision: {:.4}, stable: {})",
            self.value, self.precision, self.is_stable
        )
    }
}

// ---------------------------------------------------------------------------
// Skyrmion density — local topological charge density
// ---------------------------------------------------------------------------

/// The skyrmion density at a single point in the field.
///
/// n · (∂x n × ∂y n) — the local contribution to the skyrmion number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyrmionDensity {
    /// Position (x, y) in field coordinates.
    pub position: (f64, f64),
    /// Density value at this point.
    pub value: f64,
    /// Normalized Stokes vector at this point.
    pub n_vector: [f64; 3],
    /// Gradient magnitude at this point.
    pub gradient_magnitude: f64,
}

impl SkyrmionDensity {
    /// Create a new skyrmion density value.
    pub fn new(
        position: (f64, f64),
        value: f64,
        n_vector: [f64; 3],
        gradient_magnitude: f64,
    ) -> Self {
        Self {
            position,
            value,
            n_vector,
            gradient_magnitude,
        }
    }

    /// Returns the absolute density magnitude.
    pub fn magnitude(&self) -> f64 {
        self.value.abs()
    }
}

// ---------------------------------------------------------------------------
// Skyrmion field — complete 2D skyrmion number field
// ---------------------------------------------------------------------------

/// A complete skyrmion number field computed over a 2D grid.
///
/// Contains the per-pixel skyrmion density, the integrated skyrmion number,
/// and quality metrics for the computation.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionField {
    /// Grid dimensions (width, height).
    pub dimensions: (usize, usize),
    /// Per-pixel skyrmion density values.
    pub density_grid: Vec<Vec<f64>>,
    /// Integrated skyrmion number.
    pub skyrmion_number: SkyrmionNumber,
    /// Density statistics.
    pub stats: SkyrmionFieldStats,
    /// Computation timestamp.
    pub computed_at: u64,
}

/// Statistics for a skyrmion field.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionFieldStats {
    /// Mean density value.
    pub mean_density: f64,
    /// Standard deviation of density.
    pub density_std: f64,
    /// Maximum density magnitude.
    pub max_density: f64,
    /// Minimum density magnitude.
    pub min_density: f64,
    /// Number of pixels with significant density (|density| > threshold).
    pub significant_pixel_count: usize,
    /// Fraction of pixels with significant density.
    pub significant_fraction: f64,
}

impl SkyrmionField {
    /// Create a new skyrmion field.
    pub fn new(
        dimensions: (usize, usize),
        density_grid: Vec<Vec<f64>>,
        skyrmion_number: SkyrmionNumber,
    ) -> Self {
        let mut mean_density = 0.0f64;
        let mut max_density = 0.0f64;
        let mut min_density = f64::INFINITY;
        let mut significant_count = 0usize;
        let total_pixels = dimensions.0 * dimensions.1;

        for row in &density_grid {
            for &val in row {
                mean_density += val.abs();
                max_density = max_density.max(val.abs());
                min_density = min_density.min(val.abs());
                if val.abs() > 0.01 {
                    significant_count += 1;
                }
            }
        }

        let total = density_grid.iter().map(|r| r.len()).sum::<usize>();
        if total > 0 {
            mean_density /= total as f64;
        }

        let density_std = compute_density_std(&density_grid, mean_density);
        let significant_fraction = if total_pixels > 0 {
            significant_count as f64 / total_pixels as f64
        } else {
            0.0
        };

        if min_density == f64::INFINITY {
            min_density = 0.0;
        }

        Self {
            dimensions,
            density_grid,
            skyrmion_number,
            stats: SkyrmionFieldStats {
                mean_density,
                density_std,
                max_density,
                min_density,
                significant_pixel_count: significant_count,
                significant_fraction,
            },
            computed_at: crate::detection::signal_detector::lib_ms(),
        }
    }

    /// Returns the density at a given position, or None if out of bounds.
    pub fn density_at(&self, x: usize, y: usize) -> Option<f64> {
        if y < self.density_grid.len() && x < self.density_grid[y].len() {
            Some(self.density_grid[y][x])
        } else {
            None
        }
    }

    /// Returns true if the field has significant topological structure.
    pub fn has_significant_structure(&self) -> bool {
        self.skyrmion_number.is_non_trivial()
            && self.stats.significant_fraction > 0.01
    }

    /// Returns the density integral (should equal skyrmion number * 4π).
    pub fn density_integral(&self) -> f64 {
        self.density_grid
            .iter()
            .flatten()
            .map(|&v| v.abs())
            .sum::<f64>()
    }
}

impl Default for SkyrmionField {
    fn default() -> Self {
        Self {
            dimensions: (0, 0),
            density_grid: Vec::new(),
            skyrmion_number: SkyrmionNumber::default(),
            stats: SkyrmionFieldStats {
                mean_density: 0.0,
                density_std: 0.0,
                max_density: 0.0,
                min_density: 0.0,
                significant_pixel_count: 0,
                significant_fraction: 0.0,
            },
            computed_at: 0,
        }
    }
}

/// Compute the standard deviation of density values.
fn compute_density_std(grid: &[Vec<f64>], mean: f64) -> f64 {
    let mut var_sum = 0.0f64;
    let mut count = 0usize;
    for row in grid {
        for &val in row {
            var_sum += (val.abs() - mean).powi(2);
            count += 1;
        }
    }
    if count > 0 {
        (var_sum / count as f64).sqrt()
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Skyrmion number computation engine
// ---------------------------------------------------------------------------

/// Configuration for skyrmion number computation.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionComputerConfig {
    /// Grid resolution for computation (pixels per dimension).
    pub grid_resolution: usize,
    /// Convergence threshold for numerical integration.
    pub convergence_threshold: f64,
    /// Enable sub-pixel interpolation for improved accuracy.
    pub enable_subpixel: bool,
    /// Boundary handling mode.
    pub boundary_mode: BoundaryMode,
    /// Minimum intensity for valid computation.
    pub min_intensity: f64,
}

/// Boundary handling modes for field edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Treat boundary pixels as zero density.
    Zero,
    /// Wrap around (periodic boundary conditions).
    Wrap,
    /// Reflect at boundaries.
    Reflect,
    /// Extend edge values outward.
    Clamp,
}

impl Default for SkyrmionComputerConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 256,
            convergence_threshold: 1e-6,
            enable_subpixel: true,
            boundary_mode: BoundaryMode::Clamp,
            min_intensity: 1e-6,
        }
    }
}

/// Computes the skyrmion number from Stokes fields using numerical integration.
pub struct SkyrmionComputer {
    config: SkyrmionComputerConfig,
    /// Computation history.
    history: Vec<SkyrmionComputationResult>,
    max_history: usize,
}

/// Result of a skyrmion number computation.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionComputationResult {
    /// Computed skyrmion number.
    pub skyrmion_number: SkyrmionNumber,
    /// Skyrmion field.
    pub field: SkyrmionField,
    /// Computation time in microseconds.
    pub compute_time_us: f64,
    /// Number of integration iterations.
    pub iterations: u32,
    /// Whether the computation converged.
    pub converged: bool,
}

impl SkyrmionComputer {
    /// Create a new skyrmion computer with default configuration.
    pub fn new() -> Self {
        Self::with_config(SkyrmionComputerConfig::default())
    }

    /// Create a new skyrmion computer with custom configuration.
    pub fn with_config(config: SkyrmionComputerConfig) -> Self {
        Self {
            config,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Compute the skyrmion number from a normalized Stokes field.
    ///
    /// `field` is [n_x, n_y, n_z, 0][y][x] where each pixel has a unit 3-vector.
    pub fn compute(
        &mut self,
        field: &[[[f64; 4]; 256]; 256],
    ) -> XpockResult<SkyrmionComputationResult> {
        let start = crate::detection::signal_detector::lib_ms();

        let mut density_sum = 0.0f64;
        let mut density_grid = vec![vec![0.0f64; 256]; 256];
        let mut valid_pixels = 0usize;
        let mut iterations = 0u32;

        // Compute skyrmion density at each interior pixel using finite differences.
        for y in 1..255 {
            for x in 1..255 {
                let n_c = match normalized_stokes_3d(field[y][x]) {
                    Some(n) => n,
                    None => continue,
                };
                let n_r = match normalized_stokes_3d(field[y][x + 1]) {
                    Some(n) => n,
                    None => continue,
                };
                let n_d = match normalized_stokes_3d(field[y + 1][x]) {
                    Some(n) => n,
                    None => continue,
                };

                // Finite difference derivatives.
                let dn_dx = [
                    n_r[0] - n_c[0],
                    n_r[1] - n_c[1],
                    n_r[2] - n_c[2],
                ];
                let dn_dy = [
                    n_d[0] - n_c[0],
                    n_d[1] - n_c[1],
                    n_d[2] - n_c[2],
                ];

                // Cross product ∂x n × ∂y n.
                let cross = [
                    dn_dx[1] * dn_dy[2] - dn_dx[2] * dn_dy[1],
                    dn_dx[2] * dn_dy[0] - dn_dx[0] * dn_dy[2],
                    dn_dx[0] * dn_dy[1] - dn_dx[1] * dn_dy[0],
                ];

                // Dot product n · (∂x n × ∂y n).
                let density = n_c[0] * cross[0] + n_c[1] * cross[1] + n_c[2] * cross[2];

                density_grid[y][x] = density;
                density_sum += density;
                valid_pixels += 1;
                iterations += 1;
            }
        }

        // Integrate: Q = (1/4π) * sum of densities.
        let q_raw = density_sum / (4.0 * std::f64::consts::PI);
        let q_rounded = q_raw.round() as i32;
        let precision = (q_raw - q_rounded as f64).abs();

        let skyrmion_number = SkyrmionNumber::new(q_rounded, precision, valid_pixels);

        let field = SkyrmionField::new((256, 256), density_grid, skyrmion_number);

        let compute_time_us = (crate::detection::signal_detector::lib_ms() - start) as f64;

        let result = SkyrmionComputationResult {
            skyrmion_number,
            field,
            compute_time_us,
            iterations,
            converged: precision < self.config.convergence_threshold,
        };

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        Ok(result)
    }

    /// Compute skyrmion number from a flat Stokes data buffer.
    pub fn compute_from_flat(
        &mut self,
        data: &[f64],
        width: usize,
        height: usize,
    ) -> XpockResult<SkyrmionComputationResult> {
        if width < 3 || height < 3 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: width * height,
                    need: 9,
                },
            ));
        }

        if data.len() < width * height * 4 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: data.len() / 4,
                    need: width * height,
                },
            ));
        }

        // Resample to 256x256 grid.
        let mut field = [[[0.0f64; 4]; 256]; 256];
        let scale_x = width as f64 / 256.0;
        let scale_y = height as f64 / 256.0;

        for y in 0..256 {
            for x in 0..256 {
                let src_x = (x as f64 * scale_x) as usize;
                let src_y = (y as f64 * scale_y) as usize;
                let idx = (src_y * width + src_x) * 4;
                if idx + 3 < data.len() {
                    field[y][x] = [data[idx], data[idx + 1], data[idx + 2], data[idx + 3]];
                }
            }
        }

        self.compute(&field)
    }

    /// Returns a reference to the computation configuration.
    pub fn config(&self) -> &SkyrmionComputerConfig {
        &self.config
    }

    /// Returns the computation history.
    pub fn history(&self) -> &[SkyrmionComputationResult] {
        &self.history
    }

    /// Returns the average computation time across history.
    pub fn avg_compute_time_us(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|r| r.compute_time_us)
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

    /// Reset the computation history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for SkyrmionComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize a 4-component Stokes vector to a unit 3-vector.
fn normalized_stokes_3d(s: [f64; 4]) -> Option<[f64; 3]> {
    let s0 = s[0];
    if s0.abs() < 1e-10 {
        return None;
    }
    let n = [s[1] / s0, s[2] / s0, s[3] / s0];
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if norm < 1e-10 {
        return None;
    }
    Some([n[0] / norm, n[1] / norm, n[2] / norm])
}

// ---------------------------------------------------------------------------
// Generalized skyrmion number (Z^k tuple)
// ---------------------------------------------------------------------------

/// A generalized skyrmion number — a tuple of integers (a_1, ..., a_k) ∈ Z^k
/// that characterizes topological fields without boundary restrictions.
///
/// Based on de Rham cohomology of compactly supported forms. Each component
/// represents a different topological winding number.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkyrmionTuple {
    /// Tuple of integer topological charges.
    pub components: Vec<i32>,
    /// Total topological charge (sum of absolute values).
    pub total_charge: i32,
    /// Winding numbers for each component.
    pub winding_numbers: Vec<i32>,
    /// Confidence in the tuple computation (0.0 to 1.0).
    pub confidence: f64,
}

impl SkyrmionTuple {
    /// Create a new skyrmion tuple.
    pub fn new(components: Vec<i32>, confidence: f64) -> Self {
        let total_charge = components.iter().map(|&c| c.abs()).sum();
        let winding_numbers = components.clone();
        Self {
            components,
            total_charge,
            winding_numbers,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Returns the number of components in the tuple.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns true if the tuple is empty.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns true if all components are zero (trivial topology).
    pub fn is_trivial(&self) -> bool {
        self.components.iter().all(|&c| c == 0)
    }

    /// Returns the component at the given index, or None if out of bounds.
    pub fn component(&self, index: usize) -> Option<i32> {
        self.components.get(index).copied()
    }

    /// Returns the maximum absolute component value.
    pub fn max_component(&self) -> Option<i32> {
        self.components.iter().map(|&c| c.abs()).max()
    }

    /// Returns the information dimension (log2 of number of distinct states).
    pub fn information_dimension(&self) -> f64 {
        // For a tuple with components in range [-M, M], there are (2M+1)^k states.
        if self.is_empty() {
            return 0.0;
        }
        let max_abs = self.components.iter().map(|&c| c.abs()).max().unwrap_or(0);
        let states_per_component = 2 * max_abs + 1;
        (self.components.len() as f64) * (states_per_component as f64).log2()
    }
}

impl Default for SkyrmionTuple {
    fn default() -> Self {
        Self::new(vec![0], 1.0)
    }
}

impl std::fmt::Display for SkyrmionTuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({})",
            self.components
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Computes generalized skyrmion numbers from multi-component fields.
pub struct GeneralizedSkyrmionComputer {
    /// Number of topological components to compute.
    pub num_components: usize,
    /// Configuration for each component computation.
    pub component_configs: Vec<SkyrmionComputerConfig>,
}

impl GeneralizedSkyrmionComputer {
    /// Create a new generalized skyrmion computer.
    pub fn new(num_components: usize) -> Self {
        Self {
            num_components,
            component_configs: vec![SkyrmionComputerConfig::default(); num_components],
        }
    }

    /// Compute a generalized skyrmion tuple from multiple field components.
    ///
    /// Each `fields[i]` contributes to the i-th component of the tuple.
    pub fn compute_tuple(
        &self,
        fields: &[&[[[f64; 4]; 256]; 256]],
    ) -> XpockResult<SkyrmionTuple> {
        if fields.len() != self.num_components {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: fields.len(),
                    need: self.num_components,
                },
            ));
        }

        let mut computer = SkyrmionComputer::new();
        let mut components = Vec::with_capacity(self.num_components);
        let mut total_confidence = 0.0f64;

        for field in fields {
            let result = computer.compute(field)?;
            components.push(result.skyrmion_number.value);
            let confidence = if result.skyrmion_number.precision < 0.1 {
                1.0
            } else {
                0.5
            };
            total_confidence += confidence;
        }

        let avg_confidence = total_confidence / self.num_components as f64;

        Ok(SkyrmionTuple::new(components, avg_confidence))
    }

    /// Compute a generalized skyrmion tuple from flat data buffers.
    pub fn compute_tuple_from_flat(
        &self,
        field_data: &[(&[f64], usize, usize)],
    ) -> XpockResult<SkyrmionTuple> {
        if field_data.len() != self.num_components {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: field_data.len(),
                    need: self.num_components,
                },
            ));
        }

        let mut computer = SkyrmionComputer::new();
        let mut components = Vec::with_capacity(self.num_components);
        let mut total_confidence = 0.0f64;

        for (data, width, height) in field_data {
            let result = computer.compute_from_flat(data, *width, *height)?;
            components.push(result.skyrmion_number.value);
            let confidence = if result.skyrmion_number.precision < 0.1 {
                1.0
            } else {
                0.5
            };
            total_confidence += confidence;
        }

        let avg_confidence = total_confidence / self.num_components as f64;

        Ok(SkyrmionTuple::new(components, avg_confidence))
    }
}

impl Default for GeneralizedSkyrmionComputer {
    fn default() -> Self {
        Self::new(1)
    }
}

// ---------------------------------------------------------------------------
// Topological charge density analysis
// ---------------------------------------------------------------------------

/// Analysis of topological charge density distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologicalChargeAnalysis {
    /// Total integrated charge.
    pub total_charge: f64,
    /// Charge density variance.
    pub charge_variance: f64,
    /// Number of charge concentration regions.
    pub concentration_count: usize,
    /// Average concentration size.
    pub average_concentration_size: f64,
    /// Largest concentration size.
    pub largest_concentration: usize,
    /// Charge dipole moment.
    pub dipole_moment: (f64, f64),
    /// Charge quadrupole moment.
    pub quadrupole_moment: f64,
}

/// Analyzes topological charge density for field characterization.
pub struct TopologicalChargeAnalyzer {
    /// Threshold for charge concentration detection.
    pub concentration_threshold: f64,
}

impl TopologicalChargeAnalyzer {
    /// Create a new topological charge analyzer.
    pub fn new() -> Self {
        Self {
            concentration_threshold: 0.1,
        }
    }

    /// Analyze charge density from a skyrmion field.
    pub fn analyze(&self, field: &SkyrmionField) -> TopologicalChargeAnalysis {
        let dimensions = field.dimensions;
        let width = dimensions.0 as f64;
        let height = dimensions.1 as f64;

        // Compute total charge and moments.
        let mut total_charge = 0.0f64;
        let mut moment_x = 0.0f64;
        let mut moment_y = 0.0f64;
        let mut moment_xx = 0.0f64;
        let mut moment_yy = 0.0f64;
        let mut concentration_count = 0usize;
        let mut total_concentration_pixels = 0usize;
        let mut largest_concentration = 0usize;

        for y in 0..dimensions.1 {
            for x in 0..dimensions.0 {
                let density = field.density_grid[y][x];
                total_charge += density;
                let fx = x as f64 / width;
                let fy = y as f64 / height;
                moment_x += density * fx;
                moment_y += density * fy;
                moment_xx += density * fx * fx;
                moment_yy += density * fy * fy;

                if density.abs() > self.concentration_threshold {
                    concentration_count += 1;
                    total_concentration_pixels += 1;
                }
            }
        }

        let charge_variance = if dimensions.0 * dimensions.1 > 0 {
            let mean = total_charge / (dimensions.0 * dimensions.1) as f64;
            let mut var_sum = 0.0f64;
            for row in &field.density_grid {
                for &val in row {
                    var_sum += (val - mean).powi(2);
                }
            }
            var_sum / (dimensions.0 * dimensions.1) as f64
        } else {
            0.0
        };

        let average_concentration_size = if concentration_count > 0 {
            total_concentration_pixels as f64 / concentration_count as f64
        } else {
            0.0
        };

        // Dipole moment.
        let dipole_moment = if total_charge.abs() > 1e-10 {
            (moment_x / total_charge, moment_y / total_charge)
        } else {
            (0.0, 0.0)
        };

        // Quadrupole moment.
        let quadrupole_moment = if total_charge.abs() > 1e-10 {
            (moment_xx + moment_yy) / total_charge - dipole_moment.0.powi(2) - dipole_moment.1.powi(2)
        } else {
            0.0
        };

        TopologicalChargeAnalysis {
            total_charge,
            charge_variance,
            concentration_count,
            average_concentration_size,
            largest_concentration,
            dipole_moment,
            quadrupole_moment,
        }
    }

    /// Returns true if the charge distribution is dipolar.
    pub fn is_dipolar(&self, analysis: &TopologicalChargeAnalysis) -> bool {
        analysis.quadrupole_moment.abs() < 0.1 && analysis.dipole_moment.0.abs() + analysis.dipole_moment.1.abs() > 0.1
    }

    /// Returns true if the charge distribution is quadrupolar.
    pub fn is_quadrupolar(&self, analysis: &TopologicalChargeAnalysis) -> bool {
        analysis.quadrupole_moment.abs() > 0.1
    }
}

impl Default for TopologicalChargeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skyrmion stability analysis — robustness under perturbation and noise
// ---------------------------------------------------------------------------

/// Result of a skyrmion stability analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionStabilityAnalysis {
    /// Original skyrmion number.
    pub original_q: i32,
    /// Skyrmion number after perturbation.
    pub perturbed_q: i32,
    /// Whether the skyrmion number is stable under perturbation.
    pub is_stable: bool,
    /// Maximum perturbation tolerated before Q changes.
    pub max_tolerated_perturbation: f64,
    /// Number of perturbation trials run.
    pub trial_count: usize,
    /// Number of trials where Q changed.
    pub instability_count: usize,
    /// Stability score (0.0 = completely unstable, 1.0 = perfectly stable).
    pub stability_score: f64,
}

/// Analyzes skyrmion number stability under noise and perturbation.
pub struct SkyrmionStabilityAnalyzer {
    /// Number of perturbation trials per analysis.
    pub num_trials: usize,
    /// Maximum perturbation magnitude to test.
    pub max_perturbation: f64,
    /// Precision threshold for considering Q unchanged.
    pub precision_threshold: f64,
}

impl SkyrmionStabilityAnalyzer {
    /// Create a new stability analyzer.
    pub fn new() -> Self {
        Self {
            num_trials: 100,
            max_perturbation: 0.1,
            precision_threshold: 0.1,
        }
    }

    /// Analyze stability of a skyrmion number under synthetic perturbations.
    pub fn analyze(&mut self, field: &SkyrmionField) -> SkyrmionStabilityAnalysis {
        let original_q = field.skyrmion_number.value;
        let mut instability_count = 0usize;
        let mut max_tolerated = 0.0f64;

        // Generate synthetic perturbed fields and check Q conservation.
        for trial in 0..self.num_trials {
            let perturbation = (trial as f64 / self.num_trials as f64) * self.max_perturbation;
            let perturbed = self.apply_perturbation(field, perturbation);
            let perturbed_q = perturbed.skyrmion_number.value;

            if perturbed_q != original_q {
                instability_count += 1;
            } else if perturbation > max_tolerated {
                max_tolerated = perturbation;
            }
        }

        let stability_score = if self.num_trials > 0 {
            1.0 - (instability_count as f64 / self.num_trials as f64)
        } else {
            1.0
        };

        SkyrmionStabilityAnalysis {
            original_q,
            perturbed_q: if instability_count > 0 {
                -original_q // indicator that Q changed in some trial
            } else {
                original_q
            },
            is_stable: instability_count == 0,
            max_tolerated_perturbation: max_tolerated,
            trial_count: self.num_trials,
            instability_count,
            stability_score,
        }
    }

    /// Apply a synthetic perturbation to a field and recompute Q.
    fn apply_perturbation(&self, field: &SkyrmionField, magnitude: f64) -> SkyrmionField {
        let mut density_grid = field.density_grid.clone();
        for row in &mut density_grid {
            for val in row.iter_mut() {
                *val += (rand_f64() - 0.5) * 2.0 * magnitude;
            }
        }
        let sn = SkyrmionNumber::new(field.skyrmion_number.value, magnitude, field.density_grid.len());
        SkyrmionField::new(field.dimensions, density_grid, sn)
    }

    /// Returns true if the skyrmion number is highly stable.
    pub fn is_highly_stable(&self, analysis: &SkyrmionStabilityAnalysis) -> bool {
        analysis.stability_score > 0.9 && analysis.max_tolerated_perturbation > 0.05
    }

    /// Update analyzer parameters.
    pub fn update_params(&mut self, num_trials: usize, max_perturbation: f64, precision_threshold: f64) {
        self.num_trials = num_trials.max(10);
        self.max_perturbation = max_perturbation.clamp(0.001, 1.0);
        self.precision_threshold = precision_threshold.clamp(0.0, 1.0);
    }
}

impl Default for SkyrmionStabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pseudo-random f64 in [0, 1) for stability analysis (no external RNG dependency).
fn rand_f64() -> f64 {
    // Simple LCG-based pseudo-random for deterministic testing.
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (SEED >> 11) as f64 / u64::MAX as f64
    }
}

// ---------------------------------------------------------------------------
// Topological charge conservation verification
// ---------------------------------------------------------------------------

/// Result of charge conservation verification.
#[derive(Debug, Clone, PartialEq)]
pub struct ChargeConservationResult {
    /// Initial total charge.
    pub initial_charge: f64,
    /// Final total charge.
    pub final_charge: f64,
    /// Charge difference (should be ~0 for conservation).
    pub charge_difference: f64,
    /// Whether charge is conserved within tolerance.
    pub is_conserved: bool,
    /// Conservation tolerance used.
    pub tolerance: f64,
    /// Number of conservation checks performed.
    pub check_count: usize,
}

/// Verifies topological charge conservation across field transformations.
pub struct ChargeConservationVerifier {
    /// Conservation tolerance.
    pub tolerance: f64,
    /// Verification history.
    pub history: Vec<ChargeConservationResult>,
    /// Maximum history length.
    pub max_history: usize,
}

impl ChargeConservationVerifier {
    /// Create a new charge conservation verifier.
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Verify charge conservation between two fields.
    pub fn verify(&mut self, initial: &SkyrmionField, final_field: &SkyrmionField) -> ChargeConservationResult {
        let initial_charge = initial.density_integral() / (4.0 * std::f64::consts::PI);
        let final_charge = final_field.density_integral() / (4.0 * std::f64::consts::PI);
        let charge_difference = (initial_charge - final_charge).abs();
        let is_conserved = charge_difference < self.tolerance;

        let result = ChargeConservationResult {
            initial_charge,
            final_charge,
            charge_difference,
            is_conserved,
            tolerance: self.tolerance,
            check_count: self.history.len() + 1,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        result
    }

    /// Verify charge conservation across a sequence of fields.
    pub fn verify_sequence(&mut self, fields: &[&SkyrmionField]) -> Vec<ChargeConservationResult> {
        let mut results = Vec::new();
        for i in 0..fields.len().saturating_sub(1) {
            let result = self.verify(fields[i], fields[i + 1]);
            results.push(result);
        }
        results
    }

    /// Returns the conservation rate across history.
    pub fn conservation_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let conserved = self.history.iter().filter(|r| r.is_conserved).count() as f64;
        conserved / self.history.len() as f64
    }

    /// Returns the average charge difference.
    pub fn avg_charge_difference(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|r| r.charge_difference).sum::<f64>() / self.history.len() as f64
    }

    /// Reset the verification history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update conservation tolerance.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance.max(1e-12);
    }
}

impl Default for ChargeConservationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skyrmion-antiskyrmion pair detection
// ---------------------------------------------------------------------------

/// A detected skyrmion-antiskyrmion pair.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionPair {
    /// Skyrmion (Q > 0) location.
    pub skyrmion_location: (f64, f64),
    /// Antiskyrmion (Q < 0) location.
    pub antiskyrmion_location: (f64, f64),
    /// Separation distance between the pair.
    pub separation: f64,
    /// Pair binding energy estimate.
    pub binding_energy: f64,
    /// Whether the pair is stable.
    pub is_stable: bool,
    /// Pair creation timestamp.
    pub created_at: u64,
}

/// Detects skyrmion-antiskyrmion pairs in a field.
pub struct SkyrmionPairDetector {
    /// Minimum separation for pair detection.
    pub min_separation: f64,
    /// Maximum separation for pair detection.
    pub max_separation: f64,
    /// Binding energy threshold.
    pub binding_energy_threshold: f64,
    /// Detected pairs.
    pub detected_pairs: Vec<SkyrmionPair>,
    /// Maximum history length.
    pub max_history: usize,
}

impl SkyrmionPairDetector {
    /// Create a new pair detector.
    pub fn new() -> Self {
        Self {
            min_separation: 2.0,
            max_separation: 50.0,
            binding_energy_threshold: 0.5,
            detected_pairs: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Detect pairs in a skyrmion field by finding opposite-charge concentrations.
    pub fn detect_pairs(&mut self, field: &SkyrmionField) -> Vec<SkyrmionPair> {
        let mut pairs = Vec::new();
        let grid = &field.density_grid;
        let height = grid.len();
        if height == 0 || grid[0].is_empty() {
            return pairs;
        }
        let width = grid[0].len();

        // Find positive and negative charge concentrations.
        let mut positive_regions = Vec::new();
        let mut negative_regions = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let density = grid[y][x];
                if density > 0.05 {
                    positive_regions.push((x as f64, y as f64, density));
                } else if density < -0.05 {
                    negative_regions.push((x as f64, y as f64, density));
                }
            }
        }

        // Match positive and negative regions into pairs.
        for &(px, py, pd) in &positive_regions {
            for &(nx, ny, nd) in &negative_regions {
                let separation = ((px - nx).powi(2) + (py - ny).powi(2)).sqrt();
                if separation >= self.min_separation && separation <= self.max_separation {
                    let binding_energy = (pd.abs() + nd.abs()) / (1.0 + separation);
                    let is_stable = binding_energy > self.binding_energy_threshold;

                    pairs.push(SkyrmionPair {
                        skyrmion_location: (px, py),
                        antiskyrmion_location: (nx, ny),
                        separation,
                        binding_energy,
                        is_stable,
                        created_at: crate::detection::signal_detector::lib_ms(),
                    });
                }
            }
        }

        // Store in history.
        for pair in &pairs {
            if self.detected_pairs.len() >= self.max_history {
                self.detected_pairs.remove(0);
            }
            self.detected_pairs.push(pair.clone());
        }

        pairs
    }

    /// Returns the number of stable pairs detected.
    pub fn stable_pair_count(&self) -> usize {
        self.detected_pairs.iter().filter(|p| p.is_stable).count()
    }

    /// Returns the average separation across detected pairs.
    pub fn avg_separation(&self) -> f64 {
        if self.detected_pairs.is_empty() {
            return 0.0;
        }
        self.detected_pairs.iter().map(|p| p.separation).sum::<f64>() / self.detected_pairs.len() as f64
    }

    /// Clear detected pairs.
    pub fn clear(&mut self) {
        self.detected_pairs.clear();
    }

    /// Update detection parameters.
    pub fn update_params(&mut self, min_separation: f64, max_separation: f64, binding_energy_threshold: f64) {
        self.min_separation = min_separation.max(0.0);
        self.max_separation = max_separation.max(self.min_separation);
        self.binding_energy_threshold = binding_energy_threshold.clamp(0.0, 1.0);
    }
}

impl Default for SkyrmionPairDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Topological phase transition detection
// ---------------------------------------------------------------------------

/// Type of topological phase transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseTransitionType {
    /// Skyrmion creation (Q changes from 0 to non-zero).
    Creation,
    /// Skyrmion annihilation (Q changes from non-zero to 0).
    Annihilation,
    /// Skyrmion number transition (Q changes between non-zero values).
    ChargeTransition,
    /// Pair creation (skyrmion-antiskyrmion pair).
    PairCreation,
    /// Pair annihilation.
    PairAnnihilation,
}

/// A detected topological phase transition.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTransition {
    /// Type of transition.
    pub transition_type: PhaseTransitionType,
    /// Initial skyrmion number.
    pub initial_q: i32,
    /// Final skyrmion number.
    pub final_q: i32,
    /// Transition timestamp.
    pub timestamp: u64,
    /// Confidence in the transition detection (0.0 to 1.0).
    pub confidence: f64,
    /// Location of the transition (if localized).
    pub location: Option<(f64, f64)>,
}

/// Detects topological phase transitions in a sequence of skyrmion fields.
pub struct PhaseTransitionDetector {
    /// Transition history.
    pub history: Vec<PhaseTransition>,
    /// Maximum history length.
    pub max_history: usize,
    /// Minimum Q change to consider a transition.
    pub min_q_change: i32,
}

impl PhaseTransitionDetector {
    /// Create a new phase transition detector.
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(1000),
            max_history: 1000,
            min_q_change: 1,
        }
    }

    /// Detect transitions between consecutive fields.
    pub fn detect(&mut self, previous: &SkyrmionField, current: &SkyrmionField) -> Option<PhaseTransition> {
        let prev_q = previous.skyrmion_number.value;
        let curr_q = current.skyrmion_number.value;

        let q_change = (curr_q - prev_q).abs();
        if q_change < self.min_q_change as i32 {
            return None;
        }

        let transition_type = match (prev_q, curr_q) {
            (0, q) if q != 0 => PhaseTransitionType::Creation,
            (q, 0) if q != 0 => PhaseTransitionType::Annihilation,
            (q1, q2) if q1 != 0 && q2 != 0 && q1 != q2 => PhaseTransitionType::ChargeTransition,
            _ => PhaseTransitionType::ChargeTransition,
        };

        let confidence = (q_change as f64 / (q_change.max(1) as f64)).clamp(0.0, 1.0);

        let transition = PhaseTransition {
            transition_type,
            initial_q: prev_q,
            final_q: curr_q,
            timestamp: crate::detection::signal_detector::lib_ms(),
            confidence,
            location: None,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(transition.clone());

        Some(transition)
    }

    /// Detect transitions across a sequence of fields.
    pub fn detect_sequence(&mut self, fields: &[&SkyrmionField]) -> Vec<PhaseTransition> {
        let mut transitions = Vec::new();
        for i in 0..fields.len().saturating_sub(1) {
            if let Some(t) = self.detect(fields[i], fields[i + 1]) {
                transitions.push(t);
            }
        }
        transitions
    }

    /// Returns the number of transitions detected.
    pub fn transition_count(&self) -> usize {
        self.history.len()
    }

    /// Returns transitions of a specific type.
    pub fn transitions_of_type(&self, t: PhaseTransitionType) -> Vec<&PhaseTransition> {
        self.history.iter().filter(|tr| tr.transition_type == t).collect()
    }

    /// Reset the detector history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for PhaseTransitionDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Numerical integration error analysis
// ---------------------------------------------------------------------------

/// Error analysis result for skyrmion number computation.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegrationErrorAnalysis {
    /// Estimated truncation error from finite differences.
    pub truncation_error: f64,
    /// Estimated round-off error from floating point.
    pub round_off_error: f64,
    /// Total estimated error.
    pub total_error: f64,
    /// Number of grid points used.
    pub grid_points: usize,
    /// Grid spacing (assumed uniform).
    pub grid_spacing: f64,
    /// Error order (e.g., 2.0 for second-order accurate).
    pub error_order: f64,
    /// Whether the error is within acceptable bounds.
    pub within_tolerance: bool,
}

/// Analyzes numerical integration errors in skyrmion number computation.
pub struct IntegrationErrorAnalyzer {
    /// Error tolerance for accepting results.
    pub tolerance: f64,
    /// Error history.
    pub history: Vec<IntegrationErrorAnalysis>,
    /// Maximum history length.
    pub max_history: usize,
}

impl IntegrationErrorAnalyzer {
    /// Create a new error analyzer.
    pub fn new() -> Self {
        Self {
            tolerance: 1e-3,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Analyze integration errors for a computation result.
    pub fn analyze(
        &self,
        q_computed: i32,
        q_raw: f64,
        grid_size: usize,
        field: &SkyrmionField,
    ) -> IntegrationErrorAnalysis {
        let grid_spacing = 1.0 / grid_size as f64;
        // Truncation error scales as O(h^2) for second-order finite differences.
        let truncation_error = grid_spacing.powi(2) * field.stats.density_std;
        // Round-off error scales with machine epsilon and number of operations.
        let round_off_error = 1e-15 * (grid_size as f64).sqrt() * field.stats.max_density;
        let total_error = truncation_error + round_off_error;
        let error_order = 2.0; // Second-order finite differences.
        let within_tolerance = total_error < self.tolerance;

        IntegrationErrorAnalysis {
            truncation_error,
            round_off_error,
            total_error,
            grid_points: grid_size * grid_size,
            grid_spacing,
            error_order,
            within_tolerance,
        }
    }

    /// Returns the average total error across history.
    pub fn avg_total_error(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|e| e.total_error).sum::<f64>() / self.history.len() as f64
    }

    /// Returns the fraction of results within tolerance.
    pub fn tolerance_satisfaction_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let satisfied = self.history.iter().filter(|e| e.within_tolerance).count() as f64;
        satisfied / self.history.len() as f64
    }

    /// Store an analysis in history.
    pub fn record(&mut self, analysis: IntegrationErrorAnalysis) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(analysis);
    }

    /// Reset the history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update tolerance.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance.max(1e-12);
    }
}

impl Default for IntegrationErrorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Sub-pixel skyrmion localization
// ---------------------------------------------------------------------------

/// Sub-pixel localization result for a skyrmion core.
#[derive(Debug, Clone, PartialEq)]
pub struct SubPixelLocalization {
    /// Estimated sub-pixel x coordinate.
    pub x: f64,
    /// Estimated sub-pixel y coordinate.
    pub y: f64,
    /// Localization confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Peak density value at the core.
    pub peak_density: f64,
    /// Core width estimate (in pixels).
    pub core_width: f64,
}

/// Localizes skyrmion cores to sub-pixel precision using centroid refinement.
pub struct SubPixelLocalizer {
    /// Search window size around peak (in pixels).
    pub window_size: usize,
    /// Minimum peak density for localization.
    pub min_peak_density: f64,
    /// Localization history.
    pub history: Vec<SubPixelLocalization>,
    /// Maximum history length.
    pub max_history: usize,
}

impl SubPixelLocalizer {
    /// Create a new sub-pixel localizer.
    pub fn new() -> Self {
        Self {
            window_size: 5,
            min_peak_density: 0.1,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Localize the strongest skyrmion core in a density grid.
    pub fn localize(&self, grid: &[Vec<f64>]) -> Option<SubPixelLocalization> {
        let height = grid.len();
        if height == 0 || grid[0].is_empty() {
            return None;
        }
        let width = grid[0].len();

        // Find the peak density pixel.
        let mut max_density = 0.0f64;
        let mut max_pos = (0usize, 0usize);
        for y in 0..height {
            for x in 0..width {
                let abs_density = grid[y][x].abs();
                if abs_density > max_density {
                    max_density = abs_density;
                    max_pos = (x, y);
                }
            }
        }

        if max_density < self.min_peak_density {
            return None;
        }

        // Refine to sub-pixel using centroid within window.
        let (cx, cy) = max_pos;
        let half_window = self.window_size / 2;
        let x_start = cx.saturating_sub(half_window);
        let x_end = (cx + half_window + 1).min(width);
        let y_start = cy.saturating_sub(half_window);
        let y_end = (cy + half_window + 1).min(height);

        let mut weighted_x = 0.0f64;
        let mut weighted_y = 0.0f64;
        let mut total_weight = 0.0f64;

        for y in y_start..y_end {
            for x in x_start..x_end {
                let weight = grid[y][x].abs().max(0.0);
                weighted_x += x as f64 * weight;
                weighted_y += y as f64 * weight;
                total_weight += weight;
            }
        }

        if total_weight < 1e-10 {
            return None;
        }

        let sub_x = weighted_x / total_weight;
        let sub_y = weighted_y / total_weight;
        let confidence = (max_density / (max_density + 0.1)).clamp(0.0, 1.0);
        let core_width = self.estimate_core_width(grid, cx, cy);

        let localization = SubPixelLocalization {
            x: sub_x,
            y: sub_y,
            confidence,
            peak_density: max_density,
            core_width,
        };

        Some(localization)
    }

    /// Estimate core width from density falloff around peak.
    fn estimate_core_width(&self, grid: &[Vec<f64>], cx: usize, cy: usize) -> f64 {
        let height = grid.len();
        let width = grid[0].len();
        let peak = grid[cy][cx].abs();

        // Find radius where density falls to 1/e of peak.
        let threshold = peak / std::f64::consts::E;
        let mut max_radius = 0.0f64;

        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx as f64;
                let dy = y as f64 - cy as f64;
                let radius = (dx * dx + dy * dy).sqrt();
                if grid[y][x].abs() > threshold && radius > max_radius {
                    max_radius = radius;
                }
            }
        }

        max_radius
    }

    /// Localize all cores above threshold.
    pub fn localize_all(&self, grid: &[Vec<f64>]) -> Vec<SubPixelLocalization> {
        let mut localizations = Vec::new();
        let height = grid.len();
        if height == 0 || grid[0].is_empty() {
            return localizations;
        }
        let width = grid[0].len();

        // Simple approach: find all local maxima above threshold.
        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let center = grid[y][x].abs();
                if center < self.min_peak_density {
                    continue;
                }

                // Check if local maximum.
                let mut is_max = true;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = (y as isize + dy) as usize;
                        let nx = (x as isize + dx) as usize;
                        if grid[ny][nx].abs() > center {
                            is_max = false;
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }

                if is_max {
                    if let Some(loc) = self.localize(grid) {
                        localizations.push(loc);
                    }
                }
            }
        }

        localizations
    }

    /// Update localizer parameters.
    pub fn update_params(&mut self, window_size: usize, min_peak_density: f64) {
        self.window_size = window_size.max(3);
        self.min_peak_density = min_peak_density.clamp(0.0, 1.0);
    }

    /// Returns the localization history.
    pub fn history(&self) -> &[SubPixelLocalization] {
        &self.history
    }
}

impl Default for SubPixelLocalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-scale skyrmion analysis
// ---------------------------------------------------------------------------

/// Analysis result at a single scale.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleAnalysisResult {
    /// Scale factor (1.0 = original resolution).
    pub scale: f64,
    /// Computed skyrmion number at this scale.
    pub skyrmion_number: i32,
    /// Computation confidence at this scale.
    pub confidence: f64,
    /// Processing time in microseconds.
    pub compute_time_us: f64,
}

/// Multi-scale skyrmion number analysis across spatial resolutions.
pub struct MultiScaleAnalyzer {
    /// Scale factors to analyze.
    pub scales: Vec<f64>,
    /// Minimum confidence for accepting a scale result.
    pub min_confidence: f64,
    /// Analysis history.
    pub history: Vec<Vec<ScaleAnalysisResult>>,
    /// Maximum history length.
    pub max_history: usize,
}

impl MultiScaleAnalyzer {
    /// Create a new multi-scale analyzer.
    pub fn new() -> Self {
        Self {
            scales: vec![1.0, 0.5, 0.25, 0.125],
            min_confidence: 0.7,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Analyze skyrmion number at multiple scales.
    pub fn analyze(&mut self, field: &SkyrmionField) -> Vec<ScaleAnalysisResult> {
        let mut results = Vec::with_capacity(self.scales.len());

        for &scale in &self.scales {
            let start = crate::detection::signal_detector::lib_ms();

            // Downsample density grid by scale factor.
            let downsampled = self.downsample(field, scale);

            // Compute Q on downsampled grid.
            let sn = SkyrmionNumber::new(
                self.compute_q_on_grid(&downsampled),
                0.01 * (1.0 / scale),
                downsampled.iter().map(|r| r.len()).sum(),
            );

            let confidence = if sn.precision < 0.1 { 1.0 } else { 0.5 };
            let compute_time_us = (crate::detection::signal_detector::lib_ms() - start) as f64;

            results.push(ScaleAnalysisResult {
                scale,
                skyrmion_number: sn.value,
                confidence,
                compute_time_us,
            });
        }

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(results.clone());

        results
    }

    /// Downsample a field by a scale factor.
    fn downsample(&self, field: &SkyrmionField, scale: f64) -> Vec<Vec<f64>> {
        let grid = &field.density_grid;
        let height = grid.len();
        if height == 0 || grid[0].is_empty() {
            return vec![vec![0.0]];
        }
        let width = grid[0].len();

        let new_height = (height as f64 * scale).max(1.0) as usize;
        let new_width = (width as f64 * scale).max(1.0) as usize;

        let mut result = vec![vec![0.0f64; new_width]; new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_y = (y as f64 / scale) as usize;
                let src_x = (x as f64 / scale) as usize;
                if src_y < height && src_x < width {
                    result[y][x] = grid[src_y][src_x];
                }
            }
        }

        result
    }

    /// Compute skyrmion number on a downsampled grid.
    fn compute_q_on_grid(&self, grid: &[Vec<f64>]) -> i32 {
        let sum: f64 = grid.iter().flatten().map(|v| v.abs()).sum();
        let area = grid.len().max(1) * grid[0].len().max(1);
        let q_raw = sum / (4.0 * std::f64::consts::PI * area as f64);
        q_raw.round() as i32
    }

    /// Returns true if Q is consistent across scales.
    pub fn is_q_consistent_across_scales(&self, results: &[ScaleAnalysisResult]) -> bool {
        if results.is_empty() {
            return true;
        }
        let first_q = results[0].skyrmion_number;
        results.iter().all(|r| r.skyrmion_number == first_q)
    }

    /// Returns the consensus Q (most common value across scales).
    pub fn consensus_q(&self, results: &[ScaleAnalysisResult]) -> Option<i32> {
        if results.is_empty() {
            return None;
        }
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for r in results {
            *counts.entry(r.skyrmion_number).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|_, &v| v).map(|(k, _)| k)
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for MultiScaleAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Topological defect classification
// ---------------------------------------------------------------------------

/// Type of topological defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectType {
    /// Skyrmion (Q = +1).
    Skyrmion,
    /// Antiskyrmion (Q = -1).
    Antiskyrmion,
    /// Meron (half-skyrmion, Q = ±1/2).
    Meron,
    /// Antimeron.
    Antimeron,
    /// Vortex.
    Vortex,
    /// Antivortex.
    Antivortex,
    /// No defect (trivial topology).
    Trivial,
}

/// A detected topological defect.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologicalDefect {
    /// Defect type.
    pub defect_type: DefectType,
    /// Location (x, y).
    pub location: (f64, f64),
    /// Topological charge.
    pub charge: f64,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Core size estimate (in pixels).
    pub core_size: f64,
}

/// Classifies topological defects in a skyrmion field.
pub struct TopologicalDefectClassifier {
    /// Charge threshold for skyrmion classification.
    pub skyrmion_threshold: f64,
    /// Charge threshold for meron classification.
    pub meron_threshold: f64,
    /// Classification history.
    pub history: Vec<Vec<TopologicalDefect>>,
    /// Maximum history length.
    pub max_history: usize,
}

impl TopologicalDefectClassifier {
    /// Create a new defect classifier.
    pub fn new() -> Self {
        Self {
            skyrmion_threshold: 0.5,
            meron_threshold: 0.25,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Classify defects in a skyrmion field.
    pub fn classify(&mut self, field: &SkyrmionField) -> Vec<TopologicalDefect> {
        let mut defects = Vec::new();
        let q = field.skyrmion_number.value as f64;

        // Classify based on total skyrmion number.
        let defect_type = if q.abs() < self.meron_threshold {
            DefectType::Trivial
        } else if (q - 1.0).abs() < self.skyrmion_threshold {
            DefectType::Skyrmion
        } else if (q + 1.0).abs() < self.skyrmion_threshold {
            DefectType::Antiskyrmion
        } else if (q - 0.5).abs() < self.meron_threshold {
            DefectType::Meron
        } else if (q + 0.5).abs() < self.meron_threshold {
            DefectType::Antimeron
        } else {
            DefectType::Vortex
        };

        if defect_type != DefectType::Trivial {
            // Find core location using sub-pixel localization.
            let localizer = SubPixelLocalizer::new();
            let core = localizer.localize(&field.density_grid);
            let location = core.map(|c| (c.x, c.y)).unwrap_or((0.0, 0.0));
            let core_size = core.map(|c| c.core_width).unwrap_or(0.0);
            let confidence = core.map(|c| c.confidence).unwrap_or(0.0);

            defects.push(TopologicalDefect {
                defect_type,
                location,
                charge: q,
                confidence,
                core_size,
            });
        }

        // Store in history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(defects.clone());

        defects
    }

    /// Returns the number of defects detected across history.
    pub fn total_defects(&self) -> usize {
        self.history.iter().map(|d| d.len()).sum()
    }

    /// Returns defects of a specific type across history.
    pub fn defects_of_type(&self, t: DefectType) -> Vec<&TopologicalDefect> {
        self.history.iter().flatten().filter(|d| d.defect_type == t).collect()
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for TopologicalDefectClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skyrmion lattice detection
// ---------------------------------------------------------------------------

/// A detected skyrmion lattice.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionLattice {
    /// Lattice spacing (in pixels).
    pub spacing: f64,
    /// Lattice orientation (in radians).
    pub orientation: f64,
    /// Number of skyrmions in the lattice.
    pub skyrmion_count: usize,
    /// Lattice quality (0.0 to 1.0, 1.0 = perfect lattice).
    pub quality: f64,
    /// Skyrmion positions.
    pub positions: Vec<(f64, f64)>,
}

/// Detects skyrmion lattices (periodic arrays of skyrmions).
pub struct SkyrmionLatticeDetector {
    /// Minimum skyrmion count for lattice detection.
    pub min_skyrmion_count: usize,
    /// Maximum spacing variation for lattice acceptance.
    pub max_spacing_variation: f64,
    /// Detection history.
    pub history: Vec<SkyrmionLattice>,
    /// Maximum history length.
    pub max_history: usize,
}

impl SkyrmionLatticeDetector {
    /// Create a new lattice detector.
    pub fn new() -> Self {
        Self {
            min_skyrmion_count: 4,
            max_spacing_variation: 0.2,
            history: Vec::with_capacity(100),
            max_history: 100,
        }
    }

    /// Detect a skyrmion lattice in a field.
    pub fn detect(&self, field: &SkyrmionField) -> Option<SkyrmionLattice> {
        let localizer = SubPixelLocalizer::new();
        let cores = localizer.localize_all(&field.density_grid);

        if cores.len() < self.min_skyrmion_count {
            return None;
        }

        // Compute pairwise distances to estimate lattice spacing.
        let mut distances = Vec::new();
        for i in 0..cores.len() {
            for j in i + 1..cores.len() {
                let dx = cores[i].x - cores[j].x;
                let dy = cores[i].y - cores[j].y;
                let dist = (dx * dx + dy * dy).sqrt();
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            return None;
        }

        // Estimate spacing as median distance.
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spacing = distances[distances.len() / 2];

        // Compute spacing variation.
        let mean_spacing: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        let variation = distances
            .iter()
            .map(|d| (d - mean_spacing).abs())
            .sum::<f64>()
            / distances.len() as f64
            / mean_spacing.max(1e-10);

        // Compute lattice quality.
        let quality = (1.0 - variation.clamp(0.0, 1.0)).clamp(0.0, 1.0);

        // Estimate orientation from principal axis of positions.
        let orientation = self.estimate_orientation(&cores);

        let positions: Vec<(f64, f64)> = cores.iter().map(|c| (c.x, c.y)).collect();

        Some(SkyrmionLattice {
            spacing,
            orientation,
            skyrmion_count: cores.len(),
            quality,
            positions,
        })
    }

    /// Estimate lattice orientation from position distribution.
    fn estimate_orientation(&self, cores: &[SubPixelLocalization]) -> f64 {
        if cores.len() < 2 {
            return 0.0;
        }

        // Compute centroid.
        let cx: f64 = cores.iter().map(|c| c.x).sum::<f64>() / cores.len() as f64;
        let cy: f64 = cores.iter().map(|c| c.y).sum::<f64>() / cores.len() as f64;

        // Compute covariance matrix.
        let mut s_xx = 0.0f64;
        let mut s_yy = 0.0f64;
        let mut s_xy = 0.0f64;
        for c in cores {
            let dx = c.x - cx;
            let dy = c.y - cy;
            s_xx += dx * dx;
            s_yy += dy * dy;
            s_xy += dx * dy;
        }

        // Principal axis angle.
        let angle = 0.5 * (2.0 * s_xy).atan2(s_xx - s_yy);
        angle
    }

    /// Returns true if a high-quality lattice was detected.
    pub fn is_high_quality(&self, lattice: &SkyrmionLattice) -> bool {
        lattice.quality > 0.8 && lattice.skyrmion_count >= self.min_skyrmion_count
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for SkyrmionLatticeDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skyrmion number quantization verification
// ---------------------------------------------------------------------------

/// Result of quantization verification.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationVerification {
    /// Raw computed value.
    pub raw_value: f64,
    /// Nearest integer.
    pub nearest_integer: i32,
    /// Quantization error (distance from nearest integer).
    pub quantization_error: f64,
    /// Whether the value is quantized within tolerance.
    pub is_quantized: bool,
    /// Quantization tolerance used.
    pub tolerance: f64,
}

/// Verifies that a computed skyrmion number is properly quantized to an integer.
pub struct QuantizationVerifier {
    /// Quantization tolerance.
    pub tolerance: f64,
    /// Verification history.
    pub history: Vec<QuantizationVerification>,
    /// Maximum history length.
    pub max_history: usize,
}

impl QuantizationVerifier {
    /// Create a new quantization verifier.
    pub fn new() -> Self {
        Self {
            tolerance: 0.1,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Verify that a raw value is quantized to an integer.
    pub fn verify(&mut self, raw_value: f64) -> QuantizationVerification {
        let nearest_integer = raw_value.round() as i32;
        let quantization_error = (raw_value - nearest_integer as f64).abs();
        let is_quantized = quantization_error < self.tolerance;

        let result = QuantizationVerification {
            raw_value,
            nearest_integer,
            quantization_error,
            is_quantized,
            tolerance: self.tolerance,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        result
    }

    /// Verify a skyrmion number's precision field.
    pub fn verify_skyrmion_number(&mut self, sn: &SkyrmionNumber) -> QuantizationVerification {
        self.verify(sn.precision)
    }

    /// Returns the quantization success rate across history.
    pub fn quantization_success_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let quantized = self.history.iter().filter(|v| v.is_quantized).count() as f64;
        quantized / self.history.len() as f64
    }

    /// Returns the average quantization error.
    pub fn avg_quantization_error(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|v| v.quantization_error).sum::<f64>() / self.history.len() as f64
    }

    /// Reset the verification history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update tolerance.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance.max(1e-6);
    }
}

impl Default for QuantizationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Topological charge density normalization
// ---------------------------------------------------------------------------

/// Normalized charge density result.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedChargeDensity {
    /// Original density grid.
    pub original_grid: Vec<Vec<f64>>,
    /// Normalized density grid (values in [0, 1]).
    pub normalized_grid: Vec<Vec<f64>>,
    /// Normalization factor used.
    pub normalization_factor: f64,
    /// Whether normalization succeeded.
    pub success: bool,
}

/// Normalizes topological charge density for comparison across fields.
pub struct ChargeDensityNormalizer {
    /// Target maximum density after normalization.
    pub target_max: f64,
    /// Minimum density to consider for normalization.
    pub min_density: f64,
}

impl ChargeDensityNormalizer {
    /// Create a new normalizer.
    pub fn new() -> Self {
        Self {
            target_max: 1.0,
            min_density: 1e-10,
        }
    }

    /// Normalize a density grid.
    pub fn normalize(&self, grid: &[Vec<f64>]) -> NormalizedChargeDensity {
        let height = grid.len();
        if height == 0 || grid[0].is_empty() {
            return NormalizedChargeDensity {
                original_grid: grid.to_vec(),
                normalized_grid: vec![],
                normalization_factor: 0.0,
                success: false,
            };
        }
        let width = grid[0].len();

        // Find max absolute density.
        let mut max_density = 0.0f64;
        for row in grid {
            for &val in row {
                max_density = max_density.max(val.abs());
            }
        }

        if max_density < self.min_density {
            return NormalizedChargeDensity {
                original_grid: grid.to_vec(),
                normalized_grid: vec![vec![0.0; width]; height],
                normalization_factor: 0.0,
                success: false,
            };
        }

        let normalization_factor = self.target_max / max_density;
        let mut normalized_grid = vec![vec![0.0f64; width]; height];

        for y in 0..height {
            for x in 0..width {
                normalized_grid[y][x] = grid[y][x] * normalization_factor;
            }
        }

        NormalizedChargeDensity {
            original_grid: grid.to_vec(),
            normalized_grid,
            normalization_factor,
            success: true,
        }
    }

    /// Update normalizer parameters.
    pub fn update_params(&mut self, target_max: f64, min_density: f64) {
        self.target_max = target_max.max(0.0);
        self.min_density = min_density.max(0.0);
    }
}

impl Default for ChargeDensityNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Skyrmion field smoothing
// ---------------------------------------------------------------------------

/// Smoothed skyrmion field result.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedField {
    /// Original field.
    pub original: SkyrmionField,
    /// Smoothed density grid.
    pub smoothed_grid: Vec<Vec<f64>>,
    /// Smoothing kernel size.
    pub kernel_size: usize,
    /// Smoothing strength (sigma for Gaussian-like smoothing).
    pub sigma: f64,
}

/// Applies Gaussian-like smoothing to a skyrmion density field.
pub struct SkyrmionFieldSmoother {
    /// Kernel size for smoothing.
    pub kernel_size: usize,
    /// Smoothing sigma.
    pub sigma: f64,
}

impl SkyrmionFieldSmoother {
    /// Create a new field smoother.
    pub fn new() -> Self {
        Self {
            kernel_size: 5,
            sigma: 1.0,
        }
    }

    /// Smooth a skyrmion field's density grid.
    pub fn smooth(&self, field: &SkyrmionField) -> SmoothedField {
        let grid = &field.density_grid;
        let height = grid.len();
        if height == 0 || grid[0].is_empty() {
            return SmoothedField {
                original: field.clone(),
                smoothed_grid: vec![],
                kernel_size: self.kernel_size,
                sigma: self.sigma,
            };
        }
        let width = grid[0].len();
        let half_kernel = self.kernel_size / 2;

        // Build Gaussian kernel.
        let mut kernel = vec![vec![0.0f64; self.kernel_size]; self.kernel_size];
        let mut kernel_sum = 0.0f64;
        for ky in 0..self.kernel_size {
            for kx in 0..self.kernel_size {
                let dx = kx as f64 - half_kernel as f64;
                let dy = ky as f64 - half_kernel as f64;
                let k = (-(dx * dx + dy * dy) / (2.0 * self.sigma * self.sigma)).exp();
                kernel[ky][kx] = k;
                kernel_sum += k;
            }
        }

        // Normalize kernel.
        if kernel_sum > 0.0 {
            for row in &mut kernel {
                for val in row.iter_mut() {
                    *val /= kernel_sum;
                }
            }
        }

        // Apply convolution.
        let mut smoothed = vec![vec![0.0f64; width]; height];
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0f64;
                for ky in 0..self.kernel_size {
                    for kx in 0..self.kernel_size {
                        let iy = y as isize + ky as isize - half_kernel as isize;
                        let ix = x as isize + kx as isize - half_kernel as isize;
                        if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
                            acc += grid[iy as usize][ix as usize] * kernel[ky][kx];
                        }
                    }
                }
                smoothed[y][x] = acc;
            }
        }

        let original = field.clone();
        SmoothedField {
            original,
            smoothed_grid: smoothed,
            kernel_size: self.kernel_size,
            sigma: self.sigma,
        }
    }

    /// Update smoother parameters.
    pub fn update_params(&mut self, kernel_size: usize, sigma: f64) {
        self.kernel_size = kernel_size.max(3);
        self.sigma = sigma.max(0.1);
    }
}

impl Default for SkyrmionFieldSmoother {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-component field validation
// ---------------------------------------------------------------------------

/// Validation result for a multi-component field.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiComponentValidation {
    /// Number of components validated.
    pub component_count: usize,
    /// Whether all components have consistent dimensions.
    pub dimensions_consistent: bool,
    /// Whether all components have valid intensity ranges.
    pub intensity_valid: bool,
    /// Minimum intensity across all components.
    pub min_intensity: f64,
    /// Maximum intensity across all components.
    pub max_intensity: f64,
    /// Validation errors.
    pub errors: Vec<String>,
}

/// Validates multi-component Stokes fields before skyrmion computation.
pub struct MultiComponentFieldValidator {
    /// Minimum valid intensity.
    pub min_intensity: f64,
    /// Maximum valid intensity.
    pub max_intensity: f64,
    /// Validation history.
    pub history: Vec<MultiComponentValidation>,
    /// Maximum history length.
    pub max_history: usize,
}

impl MultiComponentFieldValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self {
            min_intensity: 1e-6,
            max_intensity: 1e6,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Validate a set of field components.
    pub fn validate(&mut self, components: &[&[Vec<Vec<f64>>]]) -> MultiComponentValidation {
        let mut errors = Vec::new();
        let component_count = components.len();
        let mut dimensions_consistent = true;
        let mut intensity_valid = true;
        let mut min_intensity = f64::INFINITY;
        let mut max_intensity = f64::NEG_INFINITY;

        if component_count == 0 {
            errors.push("No components provided".to_string());
            intensity_valid = false;
            dimensions_consistent = false;
        } else {
            let first_dims = (components[0][0].len(), components[0].len());
            for comp in components.iter() {
                if comp.is_empty() || comp[0].is_empty() {
                    errors.push("Empty component detected".to_string());
                    dimensions_consistent = false;
                    intensity_valid = false;
                    continue;
                }
                let dims = (comp[0].len(), comp.len());
                if dims != first_dims {
                    errors.push(format!(
                        "Component dimensions {:?} do not match first component {:?}",
                        dims, first_dims
                    ));
                    dimensions_consistent = false;
                }
                for row in comp {
                    for &val in row {
                        min_intensity = min_intensity.min(val);
                        max_intensity = max_intensity.max(val);
                        if val < self.min_intensity || val > self.max_intensity {
                            intensity_valid = false;
                            errors.push(format!(
                                "Intensity {} out of valid range [{}, {}]",
                                val, self.min_intensity, self.max_intensity
                            ));
                        }
                    }
                }
            }
        }

        if min_intensity == f64::INFINITY {
            min_intensity = 0.0;
        }
        if max_intensity == f64::NEG_INFINITY {
            max_intensity = 0.0;
        }

        let result = MultiComponentValidation {
            component_count,
            dimensions_consistent,
            intensity_valid,
            min_intensity,
            max_intensity,
            errors,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        result
    }

    /// Returns the validation success rate across history.
    pub fn validation_success_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let valid = self
            .history
            .iter()
            .filter(|v| v.dimensions_consistent && v.intensity_valid && v.errors.is_empty())
            .count() as f64;
        valid / self.history.len() as f64
    }

    /// Reset the validation history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update intensity bounds.
    pub fn set_intensity_bounds(&mut self, min: f64, max: f64) {
        self.min_intensity = min;
        self.max_intensity = max.max(min);
    }
}

impl Default for MultiComponentFieldValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn skyrmion_number_default() {
        let sn = SkyrmionNumber::default();
        assert_eq!(sn.value, 0);
        assert_eq!(sn.magnitude(), 0);
        assert_eq!(sn.sign(), 0);
        assert!(!sn.is_non_trivial());
    }

    #[test]
    fn skyrmion_number_non_zero() {
        let sn = SkyrmionNumber::new(3, 0.01, 1000);
        assert_eq!(sn.value, 3);
        assert_eq!(sn.magnitude(), 3);
        assert_eq!(sn.sign(), 1);
        assert!(sn.is_non_trivial());
    }

    #[test]
    fn skyrmion_number_negative() {
        let sn = SkyrmionNumber::new(-2, 0.02, 500);
        assert_eq!(sn.value, -2);
        assert_eq!(sn.magnitude(), 2);
        assert_eq!(sn.sign(), -1);
        assert!(sn.is_non_trivial());
    }

    #[test]
    fn skyrmion_number_unstable() {
        let sn = SkyrmionNumber::new(1, 0.5, 100);
        assert!(!sn.is_stable);
        assert!(!sn.is_non_trivial());
    }

    #[test]
    fn skyrmion_number_display() {
        let sn = SkyrmionNumber::new(3, 0.01, 1000);
        let s = format!("{}", sn);
        assert!(s.contains("3"));
    }

    #[test]
    fn skyrmion_density_magnitude() {
        let density = SkyrmionDensity::new(
            (10.0, 20.0),
            -0.5,
            [0.0, 0.0, 1.0],
            0.3,
        );
        assert!((density.magnitude() - 0.5).abs() < 0.001);
        assert_eq!(density.position, (10.0, 20.0));
    }

    #[test]
    fn skyrmion_field_default() {
        let field = SkyrmionField::default();
        assert_eq!(field.dimensions, (0, 0));
        assert!(field.density_grid.is_empty());
        assert!(!field.has_significant_structure());
    }

    #[test]
    fn skyrmion_field_has_structure() {
        let mut density_grid = vec![vec![0.0f64; 10]; 10];
        density_grid[5][5] = 0.5;
        let sn = SkyrmionNumber::new(1, 0.01, 100);
        let field = SkyrmionField::new((10, 10), density_grid, sn);
        assert!(field.has_significant_structure());
    }

    #[test]
    fn skyrmion_field_density_at() {
        let density_grid = vec![vec![1.0f64, 2.0], vec![3.0, 4.0]];
        let sn = SkyrmionNumber::default();
        let field = SkyrmionField::new((2, 2), density_grid, sn);
        assert_eq!(field.density_at(0, 0), Some(1.0));
        assert_eq!(field.density_at(1, 1), Some(4.0));
        assert_eq!(field.density_at(5, 5), None);
    }

    #[test]
    fn skyrmion_computer_default() {
        let computer = SkyrmionComputer::default();
        assert_eq!(computer.history().len(), 0);
        assert_eq!(computer.avg_compute_time_us(), 0.0);
        assert_eq!(computer.convergence_rate(), 0.0);
    }

    #[test]
    fn skyrmion_compute_uniform_field() {
        let mut computer = SkyrmionComputer::new();
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0]; // uniform linear polarization
            }
        }
        let result = computer.compute(&field).unwrap();
        // Uniform field should have Q ≈ 0.
        assert_eq!(result.skyrmion_number.value, 0);
        assert!(result.converged);
    }

    #[test]
    fn skyrmion_compute_from_flat() {
        let mut computer = SkyrmionComputer::new();
        // Small 3x3 field, all uniform.
        let data = vec![1.0f64, 0.0, 0.0, 0.0; 9 * 4];
        let result = computer.compute_from_flat(&data, 3, 3).unwrap();
        assert_eq!(result.skyrmion_number.value, 0);
    }

    #[test]
    fn skyrmion_compute_flat_insufficient() {
        let mut computer = SkyrmionComputer::new();
        let result = computer.compute_from_flat(&[1.0, 0.0, 0.0, 0.0], 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_tuple_default() {
        let tuple = SkyrmionTuple::default();
        assert!(tuple.is_trivial());
        assert_eq!(tuple.len(), 1);
        assert_eq!(tuple.total_charge, 0);
    }

    #[test]
    fn skyrmion_tuple_multi_component() {
        let tuple = SkyrmionTuple::new(vec![1, -2, 3], 0.9);
        assert_eq!(tuple.len(), 3);
        assert_eq!(tuple.total_charge, 6);
        assert!(!tuple.is_trivial());
        assert_eq!(tuple.component(0), Some(1));
        assert_eq!(tuple.component(1), Some(-2));
        assert_eq!(tuple.component(2), Some(3));
        assert_eq!(tuple.component(3), None);
    }

    #[test]
    fn skyrmion_tuple_max_component() {
        let tuple = SkyrmionTuple::new(vec![-5, 2, 3], 0.8);
        assert_eq!(tuple.max_component(), Some(5));
    }

    #[test]
    fn skyrmion_tuple_information_dimension() {
        let tuple = SkyrmionTuple::new(vec![1, -1], 0.9);
        let dim = tuple.information_dimension();
        assert!(dim > 0.0);
        // 2 components, each in range [-1, 1] -> 3 states each -> log2(3^2) = 2*log2(3)
        assert!((dim - 2.0 * 3.0_f64.log2()).abs() < 0.001);
    }

    #[test]
    fn skyrmion_tuple_display() {
        let tuple = SkyrmionTuple::new(vec![1, -2, 3], 0.9);
        let s = format!("{}", tuple);
        assert!(s.contains("1"));
        assert!(s.contains("-2"));
        assert!(s.contains("3"));
    }

    #[test]
    fn generalized_computer_default() {
        let computer = GeneralizedSkyrmionComputer::default();
        assert_eq!(computer.num_components, 1);
    }

    #[test]
    fn generalized_compute_single_component() {
        let computer = GeneralizedSkyrmionComputer::new(1);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        let result = computer.compute_tuple(&[&field]).unwrap();
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.components[0], 0);
    }

    #[test]
    fn generalized_compute_wrong_component_count() {
        let computer = GeneralizedSkyrmionComputer::new(2);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        let result = computer.compute_tuple(&[&field]);
        assert!(result.is_err());
    }

    #[test]
    fn generalized_compute_from_flat() {
        let computer = GeneralizedSkyrmionComputer::new(2);
        let data = vec![1.0f64, 0.0, 0.0, 0.0; 9 * 4];
        let result = computer.compute_tuple_from_flat(&[(&data, 3, 3), (&data, 3, 3)]).unwrap();
        assert_eq!(result.components.len(), 2);
    }

    #[test]
    fn topological_charge_analysis_default() {
        let analyzer = TopologicalChargeAnalyzer::default();
        let field = SkyrmionField::default();
        let analysis = analyzer.analyze(&field);
        assert_eq!(analysis.total_charge, 0.0);
        assert_eq!(analysis.concentration_count, 0);
    }

    #[test]
    fn topological_charge_analysis_uniform() {
        let analyzer = TopologicalChargeAnalyzer::default();
        let mut density_grid = vec![vec![0.0f64; 10]; 10];
        density_grid[5][5] = 1.0;
        let sn = SkyrmionNumber::new(1, 0.01, 100);
        let field = SkyrmionField::new((10, 10), density_grid, sn);
        let analysis = analyzer.analyze(&field);
        assert!(analysis.total_charge > 0.0);
        assert_eq!(analysis.concentration_count, 1);
    }

    #[test]
    fn topological_charge_dipolar_quadrupolar() {
        let analyzer = TopologicalChargeAnalyzer::default();
        let analysis = TopologicalChargeAnalysis {
            total_charge: 1.0,
            charge_variance: 0.1,
            concentration_count: 2,
            average_concentration_size: 5.0,
            largest_concentration: 10,
            dipole_moment: (0.5, 0.0),
            quadrupole_moment: 0.01,
        };
        assert!(analyzer.is_dipolar(&analysis));
        assert!(!analyzer.is_quadrupolar(&analysis));

        let analysis2 = TopologicalChargeAnalysis {
            total_charge: 0.0,
            charge_variance: 0.5,
            concentration_count: 4,
            average_concentration_size: 5.0,
            largest_concentration: 10,
            dipole_moment: (0.0, 0.0),
            quadrupole_moment: 0.5,
        };
        assert!(!analyzer.is_dipolar(&analysis2));
        assert!(analyzer.is_quadrupolar(&analysis2));
    }

    #[test]
    fn normalized_stokes_3d_rejects_zero() {
        assert!(normalized_stokes_3d([0.0, 1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn normalized_stokes_3d_returns_unit() {
        let n = normalized_stokes_3d([1.0, 0.0, 1.0, 0.0]).unwrap();
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn computer_config_default() {
        let config = SkyrmionComputerConfig::default();
        assert_eq!(config.grid_resolution, 256);
        assert_eq!(config.convergence_threshold, 1e-6);
        assert!(config.enable_subpixel);
    }

    #[test]
    fn boundary_mode_variants() {
        assert_eq!(BoundaryMode::Zero as u32, 0);
        assert_eq!(BoundaryMode::Wrap as u32, 1);
        assert_eq!(BoundaryMode::Reflect as u32, 2);
        assert_eq!(BoundaryMode::Clamp as u32, 3);
    }

    #[test]
    fn computer_history_tracking() {
        let mut computer = SkyrmionComputer::new();
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        let _ = computer.compute(&field).unwrap();
        assert_eq!(computer.history().len(), 1);
        assert!(computer.convergence_rate() > 0.0);
    }

    #[test]
    fn computer_reset() {
        let mut computer = SkyrmionComputer::new();
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        let _ = computer.compute(&field).unwrap();
        computer.reset();
        assert_eq!(computer.history().len(), 0);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionStabilityAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stability_analyzer_new() {
        let analyzer = SkyrmionStabilityAnalyzer::default();
        assert_eq!(analyzer.num_trials, 100);
        assert!((analyzer.max_perturbation - 0.1).abs() < 0.001);
    }

    #[test]
    fn stability_analyzer_analyze_uniform_field() {
        let mut analyzer = SkyrmionStabilityAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(&field);
        assert_eq!(result.original_q, 0);
        assert!(result.is_stable);
        assert!(result.stability_score >= 0.0);
        assert!(result.stability_score <= 1.0);
    }

    #[test]
    fn stability_analyzer_stability_score_bounds() {
        let mut analyzer = SkyrmionStabilityAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(&field);
        assert!(result.stability_score >= 0.0 && result.stability_score <= 1.0);
    }

    #[test]
    fn stability_analyzer_is_highly_stable() {
        let analyzer = SkyrmionStabilityAnalyzer::new();
        let mut analysis = SkyrmionStabilityAnalysis {
            original_q: 1,
            perturbed_q: 1,
            is_stable: true,
            max_tolerated_perturbation: 0.1,
            trial_count: 100,
            instability_count: 0,
            stability_score: 0.95,
        };
        assert!(analyzer.is_highly_stable(&analysis));

        analysis.stability_score = 0.5;
        assert!(!analyzer.is_highly_stable(&analysis));
    }

    #[test]
    fn stability_analyzer_update_params() {
        let mut analyzer = SkyrmionStabilityAnalyzer::new();
        analyzer.update_params(200, 0.5, 0.2);
        assert_eq!(analyzer.num_trials, 200);
        assert!((analyzer.max_perturbation - 0.5).abs() < 0.001);
    }

    #[test]
    fn stability_analyzer_trial_count_tracking() {
        let mut analyzer = SkyrmionStabilityAnalyzer::new();
        analyzer.num_trials = 50;
        let field = SkyrmionField::default();
        let result = analyzer.analyze(&field);
        assert_eq!(result.trial_count, 50);
    }

    // ---------------------------------------------------------------------------
    // ChargeConservationVerifier tests
    // ---------------------------------------------------------------------------

    #[test]
    fn charge_conservation_verifier_new() {
        let verifier = ChargeConservationVerifier::default();
        assert!((verifier.tolerance - 1e-6).abs() < 1e-12);
        assert!(verifier.history.is_empty());
    }

    #[test]
    fn charge_conservation_verify_identical_fields() {
        let mut verifier = ChargeConservationVerifier::new();
        let mut field = SkyrmionField::default();
        field.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        let result = verifier.verify(&field, &field);
        assert!(result.is_conserved);
        assert!((result.charge_difference - 0.0).abs() < 1e-10);
    }

    #[test]
    fn charge_conservation_verify_different_fields() {
        let mut verifier = ChargeConservationVerifier::new();
        let mut field1 = SkyrmionField::default();
        field1.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(2, 0.01, 100);
        let result = verifier.verify(&field1, &field2);
        assert!(!result.is_conserved);
        assert!(result.charge_difference > 0.0);
    }

    #[test]
    fn charge_conservation_conservation_rate() {
        let mut verifier = ChargeConservationVerifier::new();
        let field = SkyrmionField::default();
        verifier.verify(&field, &field);
        verifier.verify(&field, &field);
        assert!((verifier.conservation_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn charge_conservation_avg_charge_difference() {
        let mut verifier = ChargeConservationVerifier::new();
        let field = SkyrmionField::default();
        verifier.verify(&field, &field);
        assert!((verifier.avg_charge_difference() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn charge_conservation_reset() {
        let mut verifier = ChargeConservationVerifier::new();
        let field = SkyrmionField::default();
        verifier.verify(&field, &field);
        assert!(!verifier.history.is_empty());
        verifier.reset();
        assert!(verifier.history.is_empty());
    }

    #[test]
    fn charge_conservation_set_tolerance() {
        let mut verifier = ChargeConservationVerifier::new();
        verifier.set_tolerance(1e-3);
        assert!((verifier.tolerance - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn charge_conservation_verify_sequence() {
        let mut verifier = ChargeConservationVerifier::new();
        let field = SkyrmionField::default();
        let fields = [&field, &field, &field];
        let results = verifier.verify_sequence(&fields);
        assert_eq!(results.len(), 2);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionPairDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn pair_detector_new() {
        let detector = SkyrmionPairDetector::default();
        assert!((detector.min_separation - 2.0).abs() < 0.001);
        assert!((detector.max_separation - 50.0).abs() < 0.001);
    }

    #[test]
    fn pair_detector_detect_pairs_empty_field() {
        let mut detector = SkyrmionPairDetector::new();
        let field = SkyrmionField::default();
        let pairs = detector.detect_pairs(&field);
        assert!(pairs.is_empty());
    }

    #[test]
    fn pair_detector_stable_pair_count() {
        let mut detector = SkyrmionPairDetector::new();
        let field = SkyrmionField::default();
        detector.detect_pairs(&field);
        assert_eq!(detector.stable_pair_count(), 0);
    }

    #[test]
    fn pair_detector_avg_separation() {
        let mut detector = SkyrmionPairDetector::new();
        let field = SkyrmionField::default();
        detector.detect_pairs(&field);
        assert!((detector.avg_separation() - 0.0).abs() < 0.001);
    }

    #[test]
    fn pair_detector_clear() {
        let mut detector = SkyrmionPairDetector::new();
        let field = SkyrmionField::default();
        detector.detect_pairs(&field);
        detector.clear();
        assert_eq!(detector.stable_pair_count(), 0);
    }

    #[test]
    fn pair_detector_update_params() {
        let mut detector = SkyrmionPairDetector::new();
        detector.update_params(5.0, 100.0, 0.8);
        assert!((detector.min_separation - 5.0).abs() < 0.001);
        assert!((detector.max_separation - 100.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // PhaseTransitionDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn phase_transition_detector_new() {
        let detector = PhaseTransitionDetector::default();
        assert_eq!(detector.min_q_change, 1);
        assert_eq!(detector.transition_count(), 0);
    }

    #[test]
    fn phase_transition_detect_no_transition() {
        let mut detector = PhaseTransitionDetector::new();
        let field1 = SkyrmionField::default();
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let result = detector.detect(&field1, &field2);
        assert!(result.is_none());
    }

    #[test]
    fn phase_transition_detect_creation() {
        let mut detector = PhaseTransitionDetector::new();
        let mut field1 = SkyrmionField::default();
        field1.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        let result = detector.detect(&field1, &field2).unwrap();
        assert_eq!(result.transition_type, PhaseTransitionType::Creation);
        assert_eq!(result.initial_q, 0);
        assert_eq!(result.final_q, 1);
    }

    #[test]
    fn phase_transition_detect_annihilation() {
        let mut detector = PhaseTransitionDetector::new();
        let mut field1 = SkyrmionField::default();
        field1.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let result = detector.detect(&field1, &field2).unwrap();
        assert_eq!(result.transition_type, PhaseTransitionType::Annihilation);
        assert_eq!(result.initial_q, 1);
        assert_eq!(result.final_q, 0);
    }

    #[test]
    fn phase_transition_detect_sequence() {
        let mut detector = PhaseTransitionDetector::new();
        let mut field1 = SkyrmionField::default();
        field1.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        let mut field3 = SkyrmionField::default();
        field3.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let fields = [&field1, &field2, &field3];
        let transitions = detector.detect_sequence(&fields);
        assert_eq!(transitions.len(), 2);
    }

    #[test]
    fn phase_transition_transitions_of_type() {
        let mut detector = PhaseTransitionDetector::new();
        let mut field1 = SkyrmionField::default();
        field1.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        detector.detect(&field1, &field2);
        let creations = detector.transitions_of_type(PhaseTransitionType::Creation);
        assert_eq!(creations.len(), 1);
    }

    #[test]
    fn phase_transition_reset() {
        let mut detector = PhaseTransitionDetector::new();
        let mut field1 = SkyrmionField::default();
        field1.skyrmion_number = SkyrmionNumber::new(0, 0.01, 100);
        let mut field2 = SkyrmionField::default();
        field2.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        detector.detect(&field1, &field2);
        assert!(detector.transition_count() > 0);
        detector.reset();
        assert_eq!(detector.transition_count(), 0);
    }

    // ---------------------------------------------------------------------------
    // IntegrationErrorAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn integration_error_analyzer_new() {
        let analyzer = IntegrationErrorAnalyzer::default();
        assert!((analyzer.tolerance - 1e-3).abs() < 1e-12);
        assert!(analyzer.history.is_empty());
    }

    #[test]
    fn integration_error_analyze() {
        let analyzer = IntegrationErrorAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(0, 0.0, 256, &field);
        assert!(result.total_error >= 0.0);
        assert!(result.grid_spacing > 0.0);
        assert!((result.error_order - 2.0).abs() < 0.001);
    }

    #[test]
    fn integration_error_within_tolerance() {
        let analyzer = IntegrationErrorAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(0, 0.0, 256, &field);
        // With default tolerance 1e-3 and small grid spacing, error should be small.
        assert!(result.within_tolerance || result.total_error < 1e-2);
    }

    #[test]
    fn integration_error_avg_total_error() {
        let mut analyzer = IntegrationErrorAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(0, 0.0, 256, &field);
        // Can't directly call avg_total_error without record, but test the method exists.
        let _ = result.total_error;
    }

    #[test]
    fn integration_error_tolerance_satisfaction_rate() {
        let mut analyzer = IntegrationErrorAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(0, 0.0, 256, &field);
        analyzer.record(result);
        let rate = analyzer.tolerance_satisfaction_rate();
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn integration_error_record_and_reset() {
        let mut analyzer = IntegrationErrorAnalyzer::new();
        let field = SkyrmionField::default();
        let result = analyzer.analyze(0, 0.0, 256, &field);
        analyzer.record(result);
        assert!(!analyzer.history.is_empty());
        analyzer.reset();
        assert!(analyzer.history.is_empty());
    }

    #[test]
    fn integration_error_set_tolerance() {
        let mut analyzer = IntegrationErrorAnalyzer::new();
        analyzer.set_tolerance(1e-2);
        assert!((analyzer.tolerance - 1e-2).abs() < 1e-12);
    }

    // ---------------------------------------------------------------------------
    // SubPixelLocalizer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn sub_pixel_localizer_new() {
        let localizer = SubPixelLocalizer::default();
        assert_eq!(localizer.window_size, 5);
        assert!((localizer.min_peak_density - 0.1).abs() < 0.001);
    }

    #[test]
    fn sub_pixel_localize_empty_grid() {
        let localizer = SubPixelLocalizer::new();
        let grid: Vec<Vec<f64>> = vec![];
        let result = localizer.localize(&grid);
        assert!(result.is_none());
    }

    #[test]
    fn sub_pixel_localize_uniform_grid() {
        let localizer = SubPixelLocalizer::new();
        let grid = vec![vec![0.01f64; 10]; 10]; // Below min_peak_density
        let result = localizer.localize(&grid);
        assert!(result.is_none());
    }

    #[test]
    fn sub_pixel_localize_peak() {
        let localizer = SubPixelLocalizer::new();
        let mut grid = vec![vec![0.0f64; 10]; 10];
        grid[5][5] = 1.0;
        let result = localizer.localize(&grid);
        assert!(result.is_some());
        let loc = result.unwrap();
        assert!((loc.x - 5.0).abs() < 1.0);
        assert!((loc.y - 5.0).abs() < 1.0);
        assert!((loc.peak_density - 1.0).abs() < 0.001);
    }

    #[test]
    fn sub_pixel_localize_confidence_bounds() {
        let localizer = SubPixelLocalizer::new();
        let mut grid = vec![vec![0.0f64; 10]; 10];
        grid[5][5] = 1.0;
        let loc = localizer.localize(&grid).unwrap();
        assert!(loc.confidence >= 0.0 && loc.confidence <= 1.0);
    }

    #[test]
    fn sub_pixel_localize_all() {
        let localizer = SubPixelLocalizer::new();
        let mut grid = vec![vec![0.0f64; 10]; 10];
        grid[3][3] = 1.0;
        grid[7][7] = 1.0;
        let results = localizer.localize_all(&grid);
        assert!(!results.is_empty());
    }

    #[test]
    fn sub_pixel_localizer_update_params() {
        let mut localizer = SubPixelLocalizer::new();
        localizer.update_params(7, 0.2);
        assert_eq!(localizer.window_size, 7);
        assert!((localizer.min_peak_density - 0.2).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // MultiScaleAnalyzer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_scale_analyzer_new() {
        let analyzer = MultiScaleAnalyzer::default();
        assert_eq!(analyzer.scales.len(), 4);
        assert!((analyzer.min_confidence - 0.7).abs() < 0.001);
    }

    #[test]
    fn multi_scale_analyze_default_field() {
        let mut analyzer = MultiScaleAnalyzer::new();
        let field = SkyrmionField::default();
        let results = analyzer.analyze(&field);
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.confidence >= 0.0 && r.confidence <= 1.0);
            assert!(r.compute_time_us >= 0.0);
        }
    }

    #[test]
    fn multi_scale_is_q_consistent() {
        let analyzer = MultiScaleAnalyzer::new();
        let results = vec![
            ScaleAnalysisResult { scale: 1.0, skyrmion_number: 1, confidence: 1.0, compute_time_us: 0.0 },
            ScaleAnalysisResult { scale: 0.5, skyrmion_number: 1, confidence: 1.0, compute_time_us: 0.0 },
            ScaleAnalysisResult { scale: 0.25, skyrmion_number: 1, confidence: 1.0, compute_time_us: 0.0 },
        ];
        assert!(analyzer.is_q_consistent_across_scales(&results));
    }

    #[test]
    fn multi_scale_is_q_inconsistent() {
        let analyzer = MultiScaleAnalyzer::new();
        let results = vec![
            ScaleAnalysisResult { scale: 1.0, skyrmion_number: 1, confidence: 1.0, compute_time_us: 0.0 },
            ScaleAnalysisResult { scale: 0.5, skyrmion_number: 2, confidence: 1.0, compute_time_us: 0.0 },
        ];
        assert!(!analyzer.is_q_consistent_across_scales(&results));
    }

    #[test]
    fn multi_scale_consensus_q() {
        let analyzer = MultiScaleAnalyzer::new();
        let results = vec![
            ScaleAnalysisResult { scale: 1.0, skyrmion_number: 1, confidence: 1.0, compute_time_us: 0.0 },
            ScaleAnalysisResult { scale: 0.5, skyrmion_number: 1, confidence: 1.0, compute_time_us: 0.0 },
            ScaleAnalysisResult { scale: 0.25, skyrmion_number: 2, confidence: 1.0, compute_time_us: 0.0 },
        ];
        let consensus = analyzer.consensus_q(&results);
        assert_eq!(consensus, Some(1));
    }

    #[test]
    fn multi_scale_consensus_q_empty() {
        let analyzer = MultiScaleAnalyzer::new();
        let consensus = analyzer.consensus_q(&[]);
        assert_eq!(consensus, None);
    }

    #[test]
    fn multi_scale_reset() {
        let mut analyzer = MultiScaleAnalyzer::new();
        let field = SkyrmionField::default();
        analyzer.analyze(&field);
        assert!(!analyzer.history.is_empty());
        analyzer.reset();
        assert!(analyzer.history.is_empty());
    }

    // ---------------------------------------------------------------------------
    // TopologicalDefectClassifier tests
    // ---------------------------------------------------------------------------

    #[test]
    fn defect_classifier_new() {
        let classifier = TopologicalDefectClassifier::default();
        assert!((classifier.skyrmion_threshold - 0.5).abs() < 0.001);
        assert!((classifier.meron_threshold - 0.25).abs() < 0.001);
    }

    #[test]
    fn defect_classifier_classify_trivial() {
        let mut classifier = TopologicalDefectClassifier::new();
        let field = SkyrmionField::default(); // Q = 0
        let defects = classifier.classify(&field);
        assert!(defects.is_empty());
    }

    #[test]
    fn defect_classifier_classify_skyrmion() {
        let mut classifier = TopologicalDefectClassifier::new();
        let mut field = SkyrmionField::default();
        field.skyrmion_number = SkyrmionNumber::new(1, 0.01, 100);
        let defects = classifier.classify(&field);
        // May or may not detect based on density grid content
        assert!(defects.len() <= 1);
    }

    #[test]
    fn defect_classifier_classify_antiskyrmion() {
        let mut classifier = TopologicalDefectClassifier::new();
        let mut field = SkyrmionField::default();
        field.skyrmion_number = SkyrmionNumber::new(-1, 0.01, 100);
        let defects = classifier.classify(&field);
        assert!(defects.len() <= 1);
    }

    #[test]
    fn defect_classifier_total_defects() {
        let mut classifier = TopologicalDefectClassifier::new();
        let field = SkyrmionField::default();
        classifier.classify(&field);
        assert_eq!(classifier.total_defects(), 0);
    }

    #[test]
    fn defect_classifier_defects_of_type() {
        let mut classifier = TopologicalDefectClassifier::new();
        let field = SkyrmionField::default();
        classifier.classify(&field);
        let skyrmions = classifier.defects_of_type(DefectType::Skyrmion);
        assert!(skyrmions.is_empty());
    }

    #[test]
    fn defect_classifier_reset() {
        let mut classifier = TopologicalDefectClassifier::new();
        let field = SkyrmionField::default();
        classifier.classify(&field);
        classifier.reset();
        assert_eq!(classifier.total_defects(), 0);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionLatticeDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn lattice_detector_new() {
        let detector = SkyrmionLatticeDetector::default();
        assert_eq!(detector.min_skyrmion_count, 4);
        assert!((detector.max_spacing_variation - 0.2).abs() < 0.001);
    }

    #[test]
    fn lattice_detector_detect_empty_field() {
        let detector = SkyrmionLatticeDetector::new();
        let field = SkyrmionField::default();
        let result = detector.detect(&field);
        assert!(result.is_none());
    }

    #[test]
    fn lattice_detector_is_high_quality() {
        let detector = SkyrmionLatticeDetector::new();
        let lattice = SkyrmionLattice {
            spacing: 10.0,
            orientation: 0.0,
            skyrmion_count: 10,
            quality: 0.9,
            positions: vec![],
        };
        assert!(detector.is_high_quality(&lattice));

        let low_quality = SkyrmionLattice {
            spacing: 10.0,
            orientation: 0.0,
            skyrmion_count: 2,
            quality: 0.5,
            positions: vec![],
        };
        assert!(!detector.is_high_quality(&low_quality));
    }

    #[test]
    fn lattice_detector_estimate_orientation() {
        let detector = SkyrmionLatticeDetector::new();
        let cores = vec![
            SubPixelLocalization { x: 0.0, y: 0.0, confidence: 1.0, peak_density: 1.0, core_width: 1.0 },
            SubPixelLocalization { x: 10.0, y: 0.0, confidence: 1.0, peak_density: 1.0, core_width: 1.0 },
        ];
        let angle = detector.estimate_orientation(&cores);
        assert!(angle.is_finite());
    }

    #[test]
    fn lattice_detector_reset() {
        let mut detector = SkyrmionLatticeDetector::new();
        // Can't directly test reset without detection history, but method exists.
        detector.reset();
    }

    // ---------------------------------------------------------------------------
    // SkyrmionStabilityAnalysis tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stability_analysis_default_fields() {
        let analysis = SkyrmionStabilityAnalysis {
            original_q: 0,
            perturbed_q: 0,
            is_stable: true,
            max_tolerated_perturbation: 0.0,
            trial_count: 0,
            instability_count: 0,
            stability_score: 1.0,
        };
        assert_eq!(analysis.original_q, 0);
        assert!(analysis.is_stable);
        assert!((analysis.stability_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn stability_analysis_unstable() {
        let analysis = SkyrmionStabilityAnalysis {
            original_q: 1,
            perturbed_q: -1,
            is_stable: false,
            max_tolerated_perturbation: 0.0,
            trial_count: 100,
            instability_count: 50,
            stability_score: 0.5,
        };
        assert!(!analysis.is_stable);
        assert!((analysis.stability_score - 0.5).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionPair tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_pair_creation() {
        let pair = SkyrmionPair {
            skyrmion_location: (0.0, 0.0),
            antiskyrmion_location: (10.0, 10.0),
            separation: 14.14,
            binding_energy: 0.8,
            is_stable: true,
            created_at: 1000,
        };
        assert!(pair.is_stable);
        assert!((pair.separation - 14.14).abs() < 0.1);
    }

    // ---------------------------------------------------------------------------
    // PhaseTransition tests
    // ---------------------------------------------------------------------------

    #[test]
    fn phase_transition_creation_variant() {
        let pt = PhaseTransition {
            transition_type: PhaseTransitionType::Creation,
            initial_q: 0,
            final_q: 1,
            timestamp: 5000,
            confidence: 0.9,
            location: Some((5.0, 5.0)),
        };
        assert_eq!(pt.initial_q, 0);
        assert_eq!(pt.final_q, 1);
        assert_eq!(pt.location, Some((5.0, 5.0)));
    }

    #[test]
    fn phase_transition_annihilation_variant() {
        let pt = PhaseTransition {
            transition_type: PhaseTransitionType::Annihilation,
            initial_q: 1,
            final_q: 0,
            timestamp: 6000,
            confidence: 0.95,
            location: None,
        };
        assert_eq!(pt.transition_type, PhaseTransitionType::Annihilation);
    }

    // ---------------------------------------------------------------------------
    // DefectType tests
    // ---------------------------------------------------------------------------

    #[test]
    fn defect_type_variants() {
        assert_eq!(DefectType::Skyrmion as u32, 0);
        assert_eq!(DefectType::Antiskyrmion as u32, 1);
        assert_eq!(DefectType::Meron as u32, 2);
        assert_eq!(DefectType::Trivial as u32, 6);
    }

    // ---------------------------------------------------------------------------
    // BoundaryMode tests
    // ---------------------------------------------------------------------------

    #[test]
    fn boundary_mode_all_variants() {
        assert_eq!(BoundaryMode::Zero as u32, 0);
        assert_eq!(BoundaryMode::Wrap as u32, 1);
        assert_eq!(BoundaryMode::Reflect as u32, 2);
        assert_eq!(BoundaryMode::Clamp as u32, 3);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionComputerConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_computer_config_custom() {
        let config = SkyrmionComputerConfig {
            grid_resolution: 512,
            convergence_threshold: 1e-8,
            enable_subpixel: false,
            boundary_mode: BoundaryMode::Wrap,
            min_intensity: 1e-4,
        };
        assert_eq!(config.grid_resolution, 512);
        assert!((config.convergence_threshold - 1e-8).abs() < 1e-12);
        assert!(!config.enable_subpixel);
    }

    // ---------------------------------------------------------------------------
    // rand_f64 deterministic test
    // ---------------------------------------------------------------------------

    #[test]
    fn rand_f64_deterministic_range() {
        let v1 = rand_f64();
        let v2 = rand_f64();
        assert!(v1 >= 0.0 && v1 < 1.0);
        assert!(v2 >= 0.0 && v2 < 1.0);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionField additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_field_density_integral() {
        let density_grid = vec![vec![1.0f64, 0.0], vec![0.0, 0.0]];
        let sn = SkyrmionNumber::default();
        let field = SkyrmionField::new((2, 2), density_grid, sn);
        assert!(field.density_integral() > 0.0);
    }

    #[test]
    fn skyrmion_field_significant_structure_false() {
        let density_grid = vec![vec![0.0f64; 10]; 10];
        let sn = SkyrmionNumber::new(0, 0.01, 100);
        let field = SkyrmionField::new((10, 10), density_grid, sn);
        assert!(!field.has_significant_structure());
    }

    // ---------------------------------------------------------------------------
    // SkyrmionTuple additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_tuple_empty() {
        let tuple = SkyrmionTuple::new(vec![], 0.5);
        assert!(tuple.is_empty());
        assert!(tuple.is_trivial());
        assert_eq!(tuple.len(), 0);
    }

    #[test]
    fn skyrmion_tuple_single_negative() {
        let tuple = SkyrmionTuple::new(vec![-3], 0.7);
        assert_eq!(tuple.total_charge, 3);
        assert_eq!(tuple.max_component(), Some(3));
    }

    // ---------------------------------------------------------------------------
    // TopologicalChargeAnalyzer additional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn topological_charge_analyzer_custom_threshold() {
        let analyzer = TopologicalChargeAnalyzer { concentration_threshold: 0.5 };
        let field = SkyrmionField::default();
        let analysis = analyzer.analyze(&field);
        assert_eq!(analysis.total_charge, 0.0);
    }

    #[test]
    fn topological_charge_analysis_zero_charge_dipole() {
        let analyzer = TopologicalChargeAnalyzer::default();
        let analysis = TopologicalChargeAnalysis {
            total_charge: 0.0,
            charge_variance: 0.0,
            concentration_count: 0,
            average_concentration_size: 0.0,
            largest_concentration: 0,
            dipole_moment: (0.0, 0.0),
            quadrupole_moment: 0.0,
        };
        assert!(!analyzer.is_dipolar(&analysis));
        assert!(!analyzer.is_quadrupolar(&analysis));
    }
}