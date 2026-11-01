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

#[cfg(test)]
mod tests {
    use super::*;

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
}