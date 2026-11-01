//! Generalized skyrmion number (Z^k tuple) detection.
//!
//! Generalizes the scalar skyrmion number Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy
//! to a k-component integer tuple Q = (Q_1, ..., Q_k) where each component
//! corresponds to a distinct topological charge sector. This enables
//! higher-dimensional constellation encoding where each generalized skyrmion
//! number tuple maps to a unique symbol.
//!
//! All types are carrier-agnostic — no assumptions about physical medium,
//! transport, or modulation scheme.

use crate::errors::XpockError;
use crate::types::SkyrmionNumber;
use std::f64::consts::PI;

// ============================================================================
// SkyrmionTuple — k-component integer skyrmion number tuple
// ============================================================================

/// A k-component generalized skyrmion number tuple.
///
/// Each component Q_i is an integer topological charge measured in a distinct
/// sector/subspace of the field configuration space. The tuple uniquely
/// identifies a symbol in a generalized skyrmion constellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkyrmionTuple {
    /// Number of components k.
    pub k: usize,
    /// Integer charge components Q_1, ..., Q_k.
    pub components: [i64; 8], // Fixed max of 8 components for stack storage.
    /// Whether all components are valid (finite, within range).
    pub valid: bool,
}

impl SkyrmionTuple {
    /// Create a new SkyrmionTuple from a slice of components.
    ///
    /// Returns an error if k > 8 or any component is out of valid range.
    pub fn new(components: &[i64]) -> Result<Self, XpockError> {
        let k = components.len();
        if k == 0 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "k".to_string(),
                    value: "0".to_string(),
                    constraint: "k >= 1".to_string(),
                },
            ));
        }
        if k > 8 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "k".to_string(),
                    value: k.to_string(),
                    constraint: "k <= 8".to_string(),
                },
            ));
        }

        let mut arr = [0i64; 8];
        for (i, &c) in components.iter().enumerate() {
            arr[i] = c;
        }

        Ok(Self {
            k,
            components: arr,
            valid: true,
        })
    }

    /// Create a zero tuple of dimension k.
    pub fn zero(k: usize) -> Result<Self, XpockError> {
        if k == 0 || k > 8 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "k".to_string(),
                    value: k.to_string(),
                    constraint: "1 <= k <= 8".to_string(),
                },
            ));
        }
        Ok(Self {
            k,
            components: [0i64; 8],
            valid: true,
        })
    }

    /// Returns the component at index i.
    pub fn component(&self, i: usize) -> Option<i64> {
        if i < self.k {
            Some(self.components[i])
        } else {
            None
        }
    }

    /// Sets the component at index i.
    pub fn set_component(&mut self, i: usize, value: i64) -> Result<(), XpockError> {
        if i >= self.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "i".to_string(),
                    value: i.to_string(),
                    constraint: format!("i < {}", self.k),
                },
            ));
        }
        self.components[i] = value;
        Ok(())
    }

    /// Returns the L1 norm (sum of absolute values).
    pub fn l1_norm(&self) -> i64 {
        self.components[..self.k].iter().map(|c| c.abs()).sum()
    }

    /// Returns the L2 norm (Euclidean norm).
    pub fn l2_norm(&self) -> f64 {
        self.components[..self.k]
            .iter()
            .map(|c| (*c as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Returns the maximum absolute component value.
    pub fn max_abs(&self) -> i64 {
        self.components[..self.k].iter().map(|c| c.abs()).max().unwrap_or(0)
    }

    /// Returns true if all components are zero.
    pub fn is_zero(&self) -> bool {
        self.components[..self.k].iter().all(|&c| c == 0)
    }

    /// Returns true if all components are non-negative.
    pub fn is_non_negative(&self) -> bool {
        self.components[..self.k].iter().all(|&c| c >= 0)
    }

    /// Returns true if all components are non-positive.
    pub fn is_non_positive(&self) -> bool {
        self.components[..self.k].iter().all(|&c| c <= 0)
    }

    /// Returns the sign pattern as a tuple of i8 (-1, 0, or 1).
    pub fn sign_pattern(&self) -> [i8; 8] {
        let mut pattern = [0i8; 8];
        for (i, &c) in self.components[..self.k].iter().enumerate() {
            pattern[i] = c.signum();
        }
        pattern
    }

    /// Returns the hash-compatible encoding of the tuple.
    pub fn encode(&self) -> u64 {
        let mut hash: u64 = 1469598103934665603; // FNV offset basis.
        for i in 0..self.k {
            let c = self.components[i] as u64;
            hash ^= c;
            hash = hash.wrapping_mul(1099511628211); // FNV prime.
            hash ^= (i as u64) << (i % 8);
        }
        hash
    }
}

impl Default for SkyrmionTuple {
    fn default() -> Self {
        Self {
            k: 1,
            components: [0i64; 8],
            valid: true,
        }
    }
}

impl std::fmt::Display for SkyrmionTuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q(")?;
        for i in 0..self.k {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.components[i])?;
        }
        write!(f, ")")
    }
}

// Arithmetic operations on SkyrmionTuple.

impl std::ops::Add for SkyrmionTuple {
    type Output = Result<Self, XpockError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.k != rhs.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.k, 1),
                    actual: (rhs.k, 1),
                    component: "skyrmion_tuple_add".to_string(),
                },
            ));
        }
        let mut components = [0i64; 8];
        for i in 0..self.k {
            components[i] = self.components[i] + rhs.components[i];
        }
        Ok(Self {
            k: self.k,
            components,
            valid: true,
        })
    }
}

impl std::ops::Sub for SkyrmionTuple {
    type Output = Result<Self, XpockError>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.k != rhs.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.k, 1),
                    actual: (rhs.k, 1),
                    component: "skyrmion_tuple_sub".to_string(),
                },
            ));
        }
        let mut components = [0i64; 8];
        for i in 0..self.k {
            components[i] = self.components[i] - rhs.components[i];
        }
        Ok(Self {
            k: self.k,
            components,
            valid: true,
        })
    }
}

impl std::ops::Mul<i64> for SkyrmionTuple {
    type Output = Result<Self, XpockError>;

    fn mul(self, scalar: i64) -> Self::Output {
        let mut components = [0i64; 8];
        for i in 0..self.k {
            components[i] = self.components[i] * scalar;
        }
        Ok(Self {
            k: self.k,
            components,
            valid: true,
        })
    }
}

// ============================================================================
// GeneralizedSkyrmionNumber — k-component skyrmion number computation
// ============================================================================

/// Result of generalized skyrmion number computation.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedSkyrmionNumber {
    /// The computed k-component tuple.
    pub tuple: SkyrmionNumber,
    /// Number of components k.
    pub k: usize,
    /// Per-component charge values (floating point before rounding).
    pub raw_charges: Vec<f64>,
    /// Per-component quantization errors (|raw - rounded|).
    pub quantization_errors: Vec<f64>,
    /// Overall quantization quality (0.0 to 1.0, higher = better quantized).
    pub quantization_quality: f64,
    /// Computation grid resolution.
    pub grid_resolution: usize,
    /// Whether computation succeeded.
    pub success: bool,
}

impl GeneralizedSkyrmionNumber {
    /// Create a new generalized skyrmion number result.
    pub fn new(k: usize, grid_resolution: usize) -> Self {
        Self {
            tuple: SkyrmionNumber::default(),
            k,
            raw_charges: vec![0.0f64; k],
            quantization_errors: vec![0.0f64; k],
            quantization_quality: 0.0,
            grid_resolution,
            success: false,
        }
    }

    /// Set the tuple from computed charges (rounds to nearest integer).
    pub fn set_from_charges(&mut self, charges: &[f64]) -> Result<(), XpockError> {
        if charges.len() != self.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.k, 1),
                    actual: (charges.len(), 1),
                    component: "generalized_skyrmion_charges".to_string(),
                },
            ));
        }

        let mut int_components = [0i64; 8];
        let mut total_error = 0.0f64;

        for (i, &charge) in charges.iter().enumerate() {
            let rounded = charge.round() as i64;
            int_components[i] = rounded;
            self.raw_charges[i] = charge;
            self.quantization_errors[i] = (charge - rounded as f64).abs();
            total_error += self.quantization_errors[i];
        }

        self.tuple = SkyrmionNumber::default();
        self.tuple.components = int_components;
        self.tuple.k = self.k;
        self.quantization_quality = (1.0 - total_error / self.k as f64).clamp(0.0, 1.0);
        self.success = true;

        Ok(())
    }

    /// Returns the topological charge magnitude (L1 norm of integer tuple).
    pub fn charge_magnitude(&self) -> i64 {
        let mut sum = 0i64;
        for i in 0..self.k {
            sum += self.tuple.components[i].abs();
        }
        sum
    }

    /// Returns true if all components are well-quantized (error < 0.1).
    pub fn is_well_quantized(&self, tolerance: f64) -> bool {
        self.quantization_errors
            .iter()
            .all(|&e| e < tolerance)
    }

    /// Returns the component with maximum quantization error.
    pub fn max_quantization_error(&self) -> f64 {
        self.quantization_errors
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
    }

    /// Returns the dominant component (largest absolute charge).
    pub fn dominant_component(&self) -> Option<(usize, i64)> {
        let mut best: Option<(usize, i64)> = None;
        for i in 0..self.k {
            let abs_val = self.tuple.components[i].abs();
            match best {
                None => best = Some((i, self.tuple.components[i])),
                Some((_, best_abs)) if abs_val > best_abs.abs() => {
                    best = Some((i, self.tuple.components[i]));
                }
                _ => {}
            }
        }
        best
    }

    /// Reset the result to initial state.
    pub fn reset(&mut self) {
        self.tuple = SkyrmionNumber::default();
        self.raw_charges = vec![0.0f64; self.k];
        self.quantization_errors = vec![0.0f64; self.k];
        self.quantization_quality = 0.0;
        self.success = false;
    }
}

impl Default for GeneralizedSkyrmionNumber {
    fn default() -> Self {
        Self::new(1, 256)
    }
}

// ============================================================================
// Generalized skyrmion number computer
// ============================================================================

/// Configuration for generalized skyrmion number computation.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedSkyrmionConfig {
    /// Number of components k (1 to 8).
    pub k: usize,
    /// Grid resolution (pixels per dimension).
    pub grid_resolution: usize,
    /// Quantization tolerance (max allowed error for valid quantization).
    pub quantization_tolerance: f64,
    /// Field smoothing radius (pixels, 0 = no smoothing).
    pub smoothing_radius: usize,
    /// Boundary condition mode.
    pub boundary_mode: BoundaryMode,
}

/// Boundary handling modes for field edge treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Treat field as periodic (wrap around edges).
    Periodic,
    /// Treat field as having zero values outside boundary.
    Zero,
    /// Treat field as having constant value outside boundary.
    Constant,
    /// Reflect field at boundary.
    Reflect,
}

impl Default for GeneralizedSkyrmionConfig {
    fn default() -> Self {
        Self {
            k: 1,
            grid_resolution: 256,
            quantization_tolerance: 0.1,
            smoothing_radius: 1,
            boundary_mode: BoundaryMode::Periodic,
        }
    }
}

/// Computes generalized skyrmion numbers from a vector field.
///
/// The field is represented as a grid of 3D unit vectors n(x,y) = (nx, ny, nz).
/// For k > 1, each component Q_i is computed by projecting the field onto a
/// distinct subspace and computing the skyrmion number in that subspace.
pub struct GeneralizedSkyrmionComputer {
    /// Configuration.
    pub config: GeneralizedSkyrmionConfig,
    /// Computation history.
    history: Vec<GeneralizedSkyrmionNumber>,
    /// Maximum history length.
    max_history: usize,
}

impl GeneralizedSkyrmionComputer {
    /// Create a new computer with the given configuration.
    pub fn new(config: GeneralizedSkyrmionConfig) -> Self {
        Self {
            config,
            history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Compute generalized skyrmion number from a vector field grid.
    ///
    /// `field` is a grid of shape [grid_resolution][grid_resolution][3]
    /// containing unit vectors (nx, ny, nz) at each grid point.
    pub fn compute(
        &mut self,
        field: &[[[f64; 3]]],
    ) -> Result<GeneralizedSkyrmionNumber, XpockError> {
        let n = field.len();
        if n == 0 || field[0].len() != n {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "field".to_string(),
                    value: format!("{}x{}", n, field.get(0).map_or(0, |r| r.len())),
                    constraint: "square grid".to_string(),
                },
            ));
        }

        let mut result = GeneralizedSkyrmionNumber::new(self.config.k, n);

        // For each component i, compute skyrmion number using projected field.
        // Component i projects onto subspace rotated by angle 2π*i/k.
        let mut charges = vec![0.0f64; self.config.k];

        for comp in 0..self.config.k {
            let angle = 2.0 * PI * comp as f64 / self.config.k as f64;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let mut q = 0.0f64;
            for i in 0..n - 1 {
                for j in 0..n - 1 {
                    // Get four corner vectors.
                    let p00 = field[i][j];
                    let p10 = field[i + 1][j];
                    let p01 = field[i][j + 1];
                    let p11 = field[i + 1][j + 1];

                    // Project each vector onto the component subspace.
                    let proj = |v: &[f64; 3]| -> [f64; 3] {
                        [
                            v[0] * cos_a - v[1] * sin_a,
                            v[0] * sin_a + v[1] * cos_a,
                            v[2],
                        ]
                    };

                    let q00 = proj(&p00);
                    let q10 = proj(&p10);
                    let q01 = proj(&p01);
                    let q11 = proj(&p11);

                    // Compute solid angle contribution using the standard formula.
                    // Q = (1/4π) Σ Ω where Ω is the solid angle of the spherical
                    // triangle formed by the four projected vectors.
                    let solid_angle = spherical_triangle_solid_angle(&q00, &q10, &q01, &q11);
                    q += solid_angle;
                }
            }

            charges[comp] = q / (4.0 * PI);
        }

        result.set_from_charges(&charges)?;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(result.clone());

        Ok(result)
    }

    /// Compute generalized skyrmion number from separate component grids.
    /// `nx_grid`, `ny_grid`, `nz_grid` are each [grid_resolution][grid_resolution].
    pub fn compute_from_components(
        &mut self,
        nx_grid: &[[f64]],
        ny_grid: &[[f64]],
        nz_grid: &[[f64]],
    ) -> Result<GeneralizedSkyrmionNumber, XpockError> {
        let n = nx_grid.len();
        if n == 0 || nx_grid[0].len() != n {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "nx_grid".to_string(),
                    value: format!("{}x{}", n, nx_grid.get(0).map_or(0, |r| r.len())),
                    constraint: "square grid".to_string(),
                },
            ));
        }
        if ny_grid.len() != n || nz_grid.len() != n {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (n, n),
                    actual: (ny_grid.len(), nz_grid.len()),
                    component: "generalized_skyrmion_component_grids".to_string(),
                },
            ));
        }

        // Build combined field grid.
        let mut field: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                row.push([nx_grid[i][j], ny_grid[i][j], nz_grid[i][j]]);
            }
            field.push(row);
        }

        self.compute(&field)
    }

    /// Returns the computation history.
    pub fn history(&self) -> &[GeneralizedSkyrmionNumber] {
        &self.history
    }

    /// Returns the most recent computation result.
    pub fn latest(&self) -> Option<&GeneralizedSkyrmionNumber> {
        self.history.last()
    }

    /// Reset history and computer state.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: GeneralizedSkyrmionConfig) {
        self.config = config;
    }

    /// Returns the number of components k.
    pub fn k(&self) -> usize {
        self.config.k
    }

    /// Returns the grid resolution.
    pub fn grid_resolution(&self) -> usize {
        self.config.grid_resolution
    }
}

impl Default for GeneralizedSkyrmionComputer {
    fn default() -> Self {
        Self::new(GeneralizedSkyrmionConfig::default())
    }
}

// ============================================================================
// Solid angle computation helpers
// ============================================================================

/// Compute the solid angle of a spherical quadrilateral formed by four
/// unit vectors using Van Warden's formula.
fn spherical_triangle_solid_angle(
    v00: &[f64; 3],
    v10: &[f64; 3],
    v01: &[f64; 3],
    v11: &[f64; 3],
) -> f64 {
    // Use the standard solid angle formula for a pair of adjacent triangles.
    // Triangle 1: (v00, v10, v01)
    // Triangle 2: (v10, v11, v01)
    // Solid angle of triangle (a, b, c):
    //   tan(Ω/2) = |a · (b × c)| / (1 + a·b + b·c + c·a)

    let solid_triangle = |a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]| -> f64 {
        let cross = cross_product(b, c);
        let numerator = dot_product(a, &cross).abs();
        let denominator = 1.0 + dot_product(a, b) + dot_product(b, c) + dot_product(c, a);
        if denominator.abs() < 1e-12 {
            return 0.0;
        }
        2.0 * (numerator / denominator).atan()
    };

    let omega1 = solid_triangle(v00, v10, v01);
    let omega2 = solid_triangle(v10, v11, v01);

    omega1 + omega2
}

/// Compute the dot product of two 3D vectors.
fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the cross product of two 3D vectors.
fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3D vector to unit length.
fn normalize_vector(v: &[f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-12 {
        return [0.0, 0.0, 1.0]; // Default to north pole.
    }
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

// ============================================================================
// Constellation point from generalized skyrmion number
// ============================================================================

/// A constellation point defined by a generalized skyrmion number tuple.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyrmionConstellationPoint {
    /// The skyrmion number tuple defining this point.
    pub tuple: SkyrmionTuple,
    /// Symbol index (0 to 2^k - 1 for binary encoding).
    pub symbol_index: usize,
    /// Bit pattern encoded by this point (k bits per component sign).
    pub bit_pattern: Vec<bool>,
    /// Minimum distance to other constellation points (in tuple space).
    pub min_distance: f64,
}

/// Builds a constellation from generalized skyrmion number tuples.
pub struct SkyrmionConstellationBuilder {
    /// Number of components k.
    pub k: usize,
    /// Maximum charge magnitude per component.
    pub max_charge: i64,
    /// Constellation points.
    points: Vec<SkyrmionConstellationPoint>,
}

impl SkyrmionConstellationBuilder {
    /// Create a new constellation builder.
    pub fn new(k: usize, max_charge: i64) -> Result<Self, XpockError> {
        if k == 0 || k > 8 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "k".to_string(),
                    value: k.to_string(),
                    constraint: "1 <= k <= 8".to_string(),
                },
            ));
        }
        if max_charge == 0 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "max_charge".to_string(),
                    value: "0".to_string(),
                    constraint: "max_charge > 0".to_string(),
                },
            ));
        }

        Ok(Self {
            k,
            max_charge,
            points: Vec::new(),
        })
    }

    /// Add a constellation point.
    pub fn add_point(
        &mut self,
        tuple: SkyrmionTuple,
        symbol_index: usize,
    ) -> Result<(), XpockError> {
        if tuple.k != self.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.k, 1),
                    actual: (tuple.k, 1),
                    component: "constellation_point_tuple".to_string(),
                },
            ));
        }

        // Build bit pattern from sign of each component.
        let bit_pattern: Vec<bool> = (0..self.k)
            .map(|i| tuple.components[i] >= 0)
            .collect();

        let min_distance = self
            .points
            .iter()
            .map(|p| {
                let mut dist = 0.0f64;
                for i in 0..self.k {
                    let diff = (tuple.components[i] - p.tuple.components[i]) as f64;
                    dist += diff * diff;
                }
                dist.sqrt()
            })
            .fold(f64::INFINITY, f64::min);

        let point = SkyrmionConstellationPoint {
            tuple,
            symbol_index,
            bit_pattern,
            min_distance: if min_distance.is_infinite() {
                f64::INFINITY
            } else {
                min_distance
            },
        };

        self.points.push(point);
        Ok(())
    }

    /// Find the nearest constellation point to a given tuple.
    pub fn nearest_point(&self, tuple: &SkyrmionTuple) -> Option<&SkyrmionConstellationPoint> {
        self.points.iter().min_by(|a, b| {
            let dist_a = tuple_distance(tuple, &a.tuple);
            let dist_b = tuple_distance(tuple, &b.tuple);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns all constellation points.
    pub fn points(&self) -> &[SkyrmionConstellationPoint] {
        &self.points
    }

    /// Returns the number of points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Clear all points.
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

/// Compute distance between two skyrmion tuples.
fn tuple_distance(a: &SkyrmionTuple, b: &SkyrmionTuple) -> f64 {
    let k = a.k.min(b.k);
    let mut dist = 0.0f64;
    for i in 0..k {
        let diff = (a.components[i] - b.components[i]) as f64;
        dist += diff * diff;
    }
    dist.sqrt()
}

impl Default for SkyrmionConstellationBuilder {
    fn default() -> Self {
        Self::new(1, 1).unwrap()
    }
}

// ============================================================================
// SkyrmionTuple iteration and serialization helpers
// ============================================================================

/// Iterator over the components of a SkyrmionTuple.
pub struct SkyrmionTupleIter<'a> {
    tuple: &'a SkyrmionTuple,
    index: usize,
}

impl<'a> Iterator for SkyrmionTupleIter<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.tuple.k {
            let val = self.tuple.components[self.index];
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tuple.k - self.index;
        (remaining, Some(remaining))
    }
}

impl SkyrmionTuple {
    /// Returns an iterator over the components.
    pub fn iter(&self) -> SkyrmionTupleIter<'_> {
        SkyrmionTupleIter {
            tuple: self,
            index: 0,
        }
    }

    /// Returns the number of components.
    pub fn len(&self) -> usize {
        self.k
    }

    /// Returns true if the tuple has no components.
    pub fn is_empty(&self) -> bool {
        self.k == 0
    }

    /// Serializes the tuple to a compact byte representation.
    /// Format: [k: u8][components: i64 × k] (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1 + 8 * self.k);
        bytes.push(self.k as u8);
        for i in 0..self.k {
            bytes.extend_from_slice(&self.components[i].to_le_bytes());
        }
        bytes
    }

    /// Deserializes a tuple from bytes produced by `to_bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, XpockError> {
        if bytes.is_empty() {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "bytes".to_string(),
                    value: "empty".to_string(),
                    constraint: "non-empty byte buffer".to_string(),
                },
            ));
        }

        let k = bytes[0] as usize;
        if k == 0 || k > 8 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "k".to_string(),
                    value: k.to_string(),
                    constraint: "1 <= k <= 8".to_string(),
                },
            ));
        }

        let expected_len = 1 + 8 * k;
        if bytes.len() < expected_len {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "bytes".to_string(),
                    value: bytes.len().to_string(),
                    constraint: format!("at least {} bytes", expected_len),
                },
            ));
        }

        let mut components = [0i64; 8];
        for i in 0..k {
            let offset = 1 + i * 8;
            components[i] = i64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
        }

        Ok(Self {
            k,
            components,
            valid: true,
        })
    }

    /// Returns the serialized size in bytes.
    pub fn serialized_size(&self) -> usize {
        1 + 8 * self.k
    }

    /// Returns the components as a slice.
    pub fn as_slice(&self) -> &[i64] {
        &self.components[..self.k]
    }

    /// Clamps all components to the range [-max_charge, max_charge].
    pub fn clamp_components(&mut self, max_charge: i64) {
        for i in 0..self.k {
            self.components[i] = self.components[i].clamp(-max_charge, max_charge);
        }
    }

    /// Returns true if any component exceeds max_charge in magnitude.
    pub fn exceeds(&self, max_charge: i64) -> bool {
        self.components[..self.k]
            .iter()
            .any(|&c| c.abs() > max_charge)
    }

    /// Creates a tuple where each component is the absolute value.
    pub fn abs(&self) -> Self {
        let mut components = [0i64; 8];
        for i in 0..self.k {
            components[i] = self.components[i].abs();
        }
        Self {
            k: self.k,
            components,
            valid: true,
        }
    }

    /// Creates a tuple where each component is negated.
    pub fn neg(&self) -> Self {
        let mut components = [0i64; 8];
        for i in 0..self.k {
            components[i] = -self.components[i];
        }
        Self {
            k: self.k,
            components,
            valid: true,
        }
    }

    /// Returns the dot product with another tuple (component-wise).
    pub fn dot(&self, other: &SkyrmionTuple) -> Result<i64, XpockError> {
        if self.k != other.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.k, 1),
                    actual: (other.k, 1),
                    component: "skyrmion_tuple_dot".to_string(),
                },
            ));
        }
        let mut sum = 0i64;
        for i in 0..self.k {
            sum += self.components[i] * other.components[i];
        }
        Ok(sum)
    }

    /// Returns the cosine similarity with another tuple.
    pub fn cosine_similarity(&self, other: &SkyrmionTuple) -> Result<f64, XpockError> {
        if self.k != other.k {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::DimensionMismatch {
                    expected: (self.k, 1),
                    actual: (other.k, 1),
                    component: "skyrmion_tuple_cosine".to_string(),
                },
            ));
        }
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();
        if norm_a < 1e-12 || norm_b < 1e-12 {
            return Ok(0.0);
        }
        self.dot(other).map(|d| d as f64 / (norm_a * norm_b))
    }

    /// Returns the angular distance (in radians) between two tuples.
    pub fn angular_distance(&self, other: &SkyrmionTuple) -> Result<f64, XpockError> {
        let cos = self.cosine_similarity(other)?;
        let cos_clamped = cos.clamp(-1.0, 1.0);
        Ok(cos_clamped.acos())
    }
}

impl<'a> IntoIterator for &'a SkyrmionTuple {
    type Item = i64;
    type IntoIter = SkyrmionTupleIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ============================================================================
// GeneralizedSkyrmionNumber field analysis extensions
// ============================================================================

/// Field smoothing result for a component of the vector field.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSmoothingResult {
    /// Smoothed field grid.
    pub smoothed_field: Vec<Vec<[f64; 3]>>,
    /// Smoothing radius used.
    pub radius: usize,
    /// Total variation reduction (0.0 to 1.0).
    pub variation_reduction: f64,
}

/// Defect detected in the vector field.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDefect {
    /// Grid position (i, j).
    pub position: (usize, usize),
    /// Defect type.
    pub defect_type: DefectType,
    /// Defect strength (0.0 to 1.0).
    pub strength: f64,
    /// Topological charge contribution.
    pub charge: f64,
}

/// Types of field defects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectType {
    /// Skyrmion (point-like topological defect).
    Skyrmion,
    /// Antiskyrmion (opposite charge).
    Antiskyrmion,
    /// Meron (half-charge defect).
    Meron,
    /// Vortex (in-plane circulation).
    Vortex,
    /// No defect.
    None,
}

impl GeneralizedSkyrmionNumber {
    /// Computes a field smoothing result from the raw charges.
    /// This is a diagnostic that estimates how much smoothing would be
    /// needed to achieve clean quantization.
    pub fn estimated_smoothing_radius(&self) -> usize {
        let avg_error = self
            .quantization_errors
            .iter()
            .sum::<f64>()
            / self.quantization_errors.len() as f64;
        if avg_error < 0.05 {
            1
        } else if avg_error < 0.15 {
            2
        } else if avg_error < 0.3 {
            3
        } else {
            5
        }
    }

    /// Returns true if the result represents a valid topological signal.
    pub fn is_valid_signal(&self, min_quality: f64) -> bool {
        self.success
            && self.quantization_quality >= min_quality
            && !self.tuple.is_zero()
    }

    /// Returns the number of non-zero components.
    pub fn num_nonzero_components(&self) -> usize {
        self.tuple
            .components[..self.k]
            .iter()
            .filter(|&&c| c != 0)
            .count()
    }

    /// Returns the entropy of the charge distribution across components.
    pub fn charge_entropy(&self) -> f64 {
        let total: f64 = self
            .tuple
            .components[..self.k]
            .iter()
            .map(|&c| c.abs() as f64)
            .sum();
        if total < 1e-12 {
            return 0.0;
        }
        let mut entropy = 0.0f64;
        for i in 0..self.k {
            let p = self.tuple.components[i].abs() as f64 / total;
            if p > 1e-12 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Returns true if charge is concentrated in a single component.
    pub fn is_charge_concentrated(&self, threshold: f64) -> bool {
        let max_abs = self.tuple.components[..self.k]
            .iter()
            .map(|&c| c.abs() as f64)
            .fold(0.0f64, f64::max);
        let total: f64 = self
            .tuple
            .components[..self.k]
            .iter()
            .map(|&c| c.abs() as f64)
            .sum();
        if total < 1e-12 {
            return false;
        }
        max_abs / total >= threshold
    }
}

/// Smooths a vector field grid using a Gaussian-like kernel.
pub struct FieldSmoother {
    /// Smoothing radius.
    pub radius: usize,
}

impl FieldSmoother {
    /// Create a new field smoother.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Smooth the field grid.
    pub fn smooth(&self, field: &[Vec<[f64; 3]>]) -> FieldSmoothingResult {
        let n = field.len();
        if n == 0 {
            return FieldSmoothingResult {
                smoothed_field: field.to_vec(),
                radius: self.radius,
                variation_reduction: 0.0,
            };
        }

        let mut smoothed = vec![vec![[0.0f64; 3]; n]; n];
        let mut total_original_var = 0.0f64;
        let mut total_smoothed_var = 0.0f64;

        let r = self.radius as isize;

        for i in 0..n {
            for j in 0..n {
                let mut sum = [0.0f64; 3];
                let mut count = 0usize;
                for di in -r..=r {
                    for dj in -r..=r {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        if ni >= 0 && ni < n as isize && nj >= 0 && nj < n as isize {
                            let v = field[ni as usize][nj as usize];
                            sum[0] += v[0];
                            sum[1] += v[1];
                            sum[2] += v[2];
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    smoothed[i][j] = [
                        sum[0] / count as f64,
                        sum[1] / count as f64,
                        sum[2] / count as f64,
                    ];
                }
            }
        }

        // Compute total variation (sum of gradient magnitudes).
        for i in 0..n.saturating_sub(1) {
            for j in 0..n.saturating_sub(1) {
                let dx_o = (field[i + 1][j][0] - field[i][j][0]).abs()
                    + (field[i + 1][j][1] - field[i][j][1]).abs()
                    + (field[i + 1][j][2] - field[i][j][2]).abs();
                let dy_o = (field[i][j + 1][0] - field[i][j][0]).abs()
                    + (field[i][j + 1][1] - field[i][j][1]).abs()
                    + (field[i][j + 1][2] - field[i][j][2]).abs();
                total_original_var += dx_o + dy_o;

                let dx_s = (smoothed[i + 1][j][0] - smoothed[i][j][0]).abs()
                    + (smoothed[i + 1][j][1] - smoothed[i][j][1]).abs()
                    + (smoothed[i + 1][j][2] - smoothed[i][j][2]).abs();
                let dy_s = (smoothed[i][j + 1][0] - smoothed[i][j][0]).abs()
                    + (smoothed[i][j + 1][1] - smoothed[i][j][1]).abs()
                    + (smoothed[i][j + 1][2] - smoothed[i][j][2]).abs();
                total_smoothed_var += dx_s + dy_s;
            }
        }

        let variation_reduction = if total_original_var > 1e-12 {
            (1.0 - total_smoothed_var / total_original_var).clamp(0.0, 1.0)
        } else {
            0.0
        };

        FieldSmoothingResult {
            smoothed_field: smoothed,
            radius: self.radius,
            variation_reduction,
        }
    }
}

impl Default for FieldSmoother {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Detects topological defects in a vector field.
pub struct FieldDefectDetector {
    /// Minimum defect strength to report.
    pub min_strength: f64,
    /// Grid cell size for analysis.
    pub cell_size: usize,
}

impl FieldDefectDetector {
    /// Create a new defect detector.
    pub fn new() -> Self {
        Self {
            min_strength: 0.1,
            cell_size: 2,
        }
    }

    /// Detect defects in a field grid.
    pub fn detect(&self, field: &[Vec<[f64; 3]>]) -> Vec<FieldDefect> {
        let mut defects = Vec::new();
        let n = field.len();
        if n < self.cell_size {
            return defects;
        }

        for i in (0..n.saturating_sub(self.cell_size)).step_by(self.cell_size) {
            for j in (0..n.saturating_sub(self.cell_size)).step_by(self.cell_size) {
                // Compute skyrmion number in this cell.
                let mut q = 0.0f64;
                for ii in i..i + self.cell_size.saturating_sub(1) {
                    for jj in j..j + self.cell_size.saturating_sub(1) {
                        if ii + 1 < n && jj + 1 < n {
                            let solid = spherical_triangle_solid_angle(
                                &field[ii][jj],
                                &field[ii + 1][jj],
                                &field[ii][jj + 1],
                                &field[ii + 1][jj + 1],
                            );
                            q += solid / (4.0 * PI);
                        }
                    }
                }

                let strength = q.abs();
                if strength > self.min_strength {
                    let defect_type = if q > 0.0 {
                        DefectType::Skyrmion
                    } else if q < -0.5 {
                        DefectType::Antiskyrmion
                    } else if q.abs() > 0.25 {
                        DefectType::Meron
                    } else {
                        DefectType::Vortex
                    };

                    defects.push(FieldDefect {
                        position: (i, j),
                        defect_type,
                        strength,
                        charge: q,
                    });
                }
            }
        }

        defects
    }

    /// Returns the total topological charge of all detected defects.
    pub fn total_charge(defects: &[FieldDefect]) -> f64 {
        defects.iter().map(|d| d.charge).sum()
    }

    /// Returns defects of a specific type.
    pub fn by_type(defects: &[FieldDefect], dtype: DefectType) -> Vec<&FieldDefect> {
        defects.iter().filter(|d| d.defect_type == dtype).collect()
    }
}

impl Default for FieldDefectDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GeneralizedSkyrmionComputer multi-scale and adaptive computation
// ============================================================================

/// Result of multi-scale skyrmion computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleResult {
    /// Results at each scale.
    pub scale_results: Vec<GeneralizedSkyrmionNumber>,
    /// Scale factors used.
    pub scale_factors: Vec<f64>,
    /// Converged skyrmion number (most stable across scales).
    pub converged: Option<SkyrmionTuple>,
    /// Convergence quality (0.0 to 1.0).
    pub convergence_quality: f64,
}

/// Computes skyrmion numbers at multiple spatial scales.
pub struct MultiScaleSkyrmionComputer {
    /// Base computer.
    pub computer: GeneralizedSkyrmionComputer,
    /// Number of scales.
    pub num_scales: usize,
    /// Scale factor between consecutive scales.
    pub scale_step: f64,
}

impl MultiScaleSkyrmionComputer {
    /// Create a new multi-scale computer.
    pub fn new(
        computer: GeneralizedSkyrmionComputer,
        num_scales: usize,
        scale_step: f64,
    ) -> Self {
        Self {
            computer,
            num_scales: num_scales.max(1),
            scale_step: scale_step.max(1.01),
        }
    }

    /// Compute skyrmion numbers at multiple scales.
    /// The field is progressively coarsened (subsampled) at each scale.
    pub fn compute_multi_scale(
        &mut self,
        field: &[Vec<[f64; 3]>],
    ) -> Result<MultiScaleResult, XpockError> {
        let n = field.len();
        if n == 0 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "field".to_string(),
                    value: "empty".to_string(),
                    constraint: "non-empty field".to_string(),
                },
            ));
        }

        let mut scale_results = Vec::with_capacity(self.num_scales);
        let mut scale_factors = Vec::with_capacity(self.num_scales);

        for s in 0..self.num_scales {
            let scale = self.scale_step.powi(s as i32) as usize;
            scale_factors.push(scale as f64);

            // Subsample field by factor `scale`.
            if scale >= n {
                break;
            }
            let subsampled_n = n / scale;
            if subsampled_n < 2 {
                break;
            }

            let mut subsampled = vec![vec![[0.0f64; 3]; subsampled_n]; subsampled_n];
            for i in 0..subsampled_n {
                for j in 0..subsampled_n {
                    subsampled[i][j] = field[i * scale][j * scale];
                }
            }

            let result = self.computer.compute(&subsampled)?;
            scale_results.push(result);
        }

        // Find converged result (most frequent tuple across scales).
        let converged = Self::find_converged(&scale_results);
        let convergence_quality = if scale_results.len() > 1 {
            let matches = scale_results
                .iter()
                .filter(|r| {
                    converged.as_ref().map_or(false, |c| r.tuple == *c)
                })
                .count();
            matches as f64 / scale_results.len() as f64
        } else {
            1.0
        };

        Ok(MultiScaleResult {
            scale_results,
            scale_factors,
            converged,
            convergence_quality,
        })
    }

    /// Find the most common tuple across scale results.
    fn find_converged(results: &[GeneralizedSkyrmionNumber]) -> Option<SkyrmionTuple> {
        if results.is_empty() {
            return None;
        }
        // Simple approach: return the tuple from the result with best quantization.
        let best = results
            .iter()
            .max_by(|a, b| a.quantization_quality.partial_cmp(&b.quantization_quality).unwrap());
        best.map(|r| r.tuple)
    }

    /// Returns the number of scales.
    pub fn num_scales(&self) -> usize {
        self.num_scales
    }

    /// Returns the scale step.
    pub fn scale_step(&self) -> f64 {
        self.scale_step
    }
}

/// Adaptive resolution skyrmion computer that adjusts grid resolution
/// based on field complexity.
pub struct AdaptiveSkyrmionComputer {
    /// Base computer.
    pub computer: GeneralizedSkyrmionComputer,
    /// Minimum resolution.
    pub min_resolution: usize,
    /// Maximum resolution.
    pub max_resolution: usize,
    /// Complexity threshold for upscaling.
    pub complexity_threshold: f64,
}

impl AdaptiveSkyrmionComputer {
    /// Create a new adaptive computer.
    pub fn new(
        computer: GeneralizedSkyrmionComputer,
        min_resolution: usize,
        max_resolution: usize,
    ) -> Self {
        Self {
            computer,
            min_resolution: min_resolution.max(2),
            max_resolution: max_resolution.max(2),
            complexity_threshold: 0.3,
        }
    }

    /// Compute with adaptive resolution.
    /// First computes at low resolution, then upscales if complexity is high.
    pub fn compute_adaptive(
        &mut self,
        field: &[Vec<[f64; 3]>],
    ) -> Result<(GeneralizedSkyrmionNumber, usize), XpockError> {
        let n = field.len();
        if n < self.min_resolution {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InvalidParameter {
                    parameter: "field".to_string(),
                    value: n.to_string(),
                    constraint: format!("at least {}", self.min_resolution),
                },
            ));
        }

        // Compute at min resolution first.
        let resolution = self.min_resolution.min(n);
        let result = self.computer.compute(field)?;

        // Estimate complexity from quantization quality.
        let complexity = 1.0 - result.quantization_quality;

        let final_resolution = if complexity > self.complexity_threshold
            && resolution < self.max_resolution
            && resolution < n
        {
            // Upscale.
            (resolution * 2).min(self.max_resolution).min(n)
        } else {
            resolution
        };

        Ok((result, final_resolution))
    }

    /// Returns the min resolution.
    pub fn min_resolution(&self) -> usize {
        self.min_resolution
    }

    /// Returns the max resolution.
    pub fn max_resolution(&self) -> usize {
        self.max_resolution
    }

    /// Set complexity threshold.
    pub fn set_complexity_threshold(&mut self, threshold: f64) {
        self.complexity_threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for MultiScaleSkyrmionComputer {
    fn default() -> Self {
        Self::new(
            GeneralizedSkyrmionComputer::default(),
            3,
            1.5,
        )
    }
}

impl Default for AdaptiveSkyrmionComputer {
    fn default() -> Self {
        Self::new(
            GeneralizedSkyrmionComputer::default(),
            32,
            256,
        )
    }
}

// ============================================================================
// Additional solid angle formulations
// ============================================================================

/// Compute solid angle using the l'Huilier's theorem formulation
/// (more numerically stable for small triangles).
fn solid_angle_lhuilier(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    // Compute side lengths of the spherical triangle.
    let ab = (dot_product(a, b).clamp(-1.0, 1.0)).acos();
    let bc = (dot_product(b, c).clamp(-1.0, 1.0)).acos();
    let ca = (dot_product(c, a).clamp(-1.0, 1.0)).acos();

    let s = (ab + bc + ca) / 2.0;
    let val = (s / 2.0)
        * ((s - ab) / 2.0)
        * ((s - bc) / 2.0)
        * ((s - ca) / 2.0);
    if val < 0.0 {
        return 0.0;
    }
    4.0 * val.sqrt().atan()
}

/// Compute solid angle using the vector triple product formulation.
fn solid_angle_triple_product(
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
) -> f64 {
    let cross = cross_product(b, c);
    let numerator = dot_product(a, &cross).abs();
    let denom = 1.0 + dot_product(a, b) + dot_product(b, c) + dot_product(c, a);
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    2.0 * (numerator / denom).atan()
}

/// Estimate numerical error bound for solid angle computation.
fn solid_angle_error_bound(
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
) -> f64 {
    // Error scales with the sine of the solid angle divided by grid resolution.
    // Conservative bound: machine epsilon times condition number.
    let eps = 1e-15;
    let norm_product = (dot_product(a, a) * dot_product(b, b) * dot_product(c, c)).sqrt();
    if norm_product < 1e-12 {
        return 0.0;
    }
    eps * 100.0 / norm_product // Conservative factor of 100.
}

// ============================================================================
// SkyrmionConstellationBuilder spacing and capacity analysis
// ============================================================================

/// Constellation spacing analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationSpacingAnalysis {
    /// Minimum pairwise distance.
    pub min_distance: f64,
    /// Maximum pairwise distance.
    pub max_distance: f64,
    /// Mean pairwise distance.
    pub mean_distance: f64,
    /// Standard deviation of pairwise distances.
    pub std_distance: f64,
    /// Number of point pairs analyzed.
    pub num_pairs: usize,
    /// Uniformity score (0.0 to 1.0, higher = more uniform spacing).
    pub uniformity: f64,
}

/// Constellation capacity analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationCapacityAnalysis {
    /// Number of components k.
    pub k: usize,
    /// Maximum charge per component.
    pub max_charge: i64,
    /// Theoretical capacity (number of possible tuples).
    pub theoretical_capacity: usize,
    /// Actual number of points.
    pub actual_points: usize,
    /// Utilization ratio (actual / theoretical).
    pub utilization: f64,
    /// Bits per symbol (log2 of actual points).
    pub bits_per_symbol: f64,
}

/// Error correction capability of a constellation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationErrorCorrection {
    /// Minimum distance between any two points.
    pub min_distance: f64,
    /// Number of correctable errors (floor of (d_min - 1) / 2).
    pub correctable_errors: usize,
    /// Number of detectable errors (d_min - 1).
    pub detectable_errors: usize,
    /// Error correction strength (0.0 to 1.0).
    pub strength: f64,
}

impl SkyrmionConstellationBuilder {
    /// Perform spacing analysis on all constellation points.
    pub fn spacing_analysis(&self) -> ConstellationSpacingAnalysis {
        let points = &self.points;
        let n = points.len();
        if n < 2 {
            return ConstellationSpacingAnalysis {
                min_distance: 0.0,
                max_distance: 0.0,
                mean_distance: 0.0,
                std_distance: 0.0,
                num_pairs: 0,
                uniformity: 1.0,
            };
        }

        let mut distances = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = tuple_distance(&points[i].tuple, &points[j].tuple);
                distances.push(dist);
            }
        }

        let min_dist = distances.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_dist = distances.iter().cloned().fold(0.0f64, f64::max);
        let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance =
            distances
                .iter()
                .map(|&d| (d - mean_dist).powi(2))
                .sum::<f64>()
                / distances.len() as f64;
        let std_dist = variance.sqrt();

        // Uniformity: how close distances are to the mean.
        let uniformity = if mean_dist > 1e-12 {
            (1.0 - std_dist / mean_dist).clamp(0.0, 1.0)
        } else {
            0.0
        };

        ConstellationSpacingAnalysis {
            min_distance: min_dist,
            max_distance: max_dist,
            mean_distance: mean_dist,
            std_distance: std_dist,
            num_pairs: distances.len(),
            uniformity,
        }
    }

    /// Perform capacity analysis.
    pub fn capacity_analysis(&self) -> ConstellationCapacityAnalysis {
        let theoretical = ((2 * self.max_charge + 1) as usize).pow(self.k as u32);
        let actual = self.points.len();
        let utilization = if theoretical > 0 {
            actual as f64 / theoretical as f64
        } else {
            0.0
        };
        let bits_per_symbol = if actual > 0 {
            (actual as f64).log2()
        } else {
            0.0
        };

        ConstellationCapacityAnalysis {
            k: self.k,
            max_charge: self.max_charge,
            theoretical_capacity: theoretical,
            actual_points: actual,
            utilization,
            bits_per_symbol,
        }
    }

    /// Compute error correction capability.
    pub fn error_correction(&self) -> ConstellationErrorCorrection {
        let spacing = self.spacing_analysis();
        let d_min = spacing.min_distance;
        let correctable = if d_min >= 1.0 {
            ((d_min - 1.0) / 2.0).floor() as usize
        } else {
            0
        };
        let detectable = if d_min >= 1.0 {
            (d_min - 1.0).floor() as usize
        } else {
            0
        };
        let strength = (d_min / (2.0 * self.max_charge as f64)).clamp(0.0, 1.0);

        ConstellationErrorCorrection {
            min_distance: d_min,
            correctable_errors: correctable,
            detectable_errors: detectable,
            strength,
        }
    }

    /// Returns true if all points are well-separated (min distance >= threshold).
    pub fn is_well_separated(&self, threshold: f64) -> bool {
        let spacing = self.spacing_analysis();
        spacing.num_pairs == 0 || spacing.min_distance >= threshold
    }

    /// Returns the point at a given index.
    pub fn point_at(&self, index: usize) -> Option<&SkyrmionConstellationPoint> {
        self.points.get(index)
    }

    /// Returns true if a tuple exists in the constellation.
    pub fn contains_tuple(&self, tuple: &SkyrmionTuple) -> bool {
        self.points.iter().any(|p| p.tuple == *tuple)
    }
}

// ============================================================================
// SkyrmionTuple comparison utilities
// ============================================================================

/// Comparison result between two skyrmion tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TupleComparison {
    /// Tuples are identical.
    Equal,
    /// First tuple is lexicographically less than second.
    Less,
    /// First tuple is lexicographically greater than second.
    Greater,
    /// Tuples have different dimensions.
    DimensionMismatch,
}

/// Compares skyrmion tuples lexicographically by component.
pub struct SkyrmionTupleComparator {
    /// Comparison mode.
    pub mode: ComparisonMode,
}

/// Comparison mode for tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    /// Compare component by component (lexicographic).
    Lexicographic,
    /// Compare by L1 norm.
    L1Norm,
    /// Compare by L2 norm.
    L2Norm,
    /// Compare by maximum absolute component.
    MaxAbs,
}

impl SkyrmionTupleComparator {
    /// Create a new comparator.
    pub fn new(mode: ComparisonMode) -> Self {
        Self { mode }
    }

    /// Compare two tuples.
    pub fn compare(&self, a: &SkyrmionTuple, b: &SkyrmionTuple) -> TupleComparison {
        if a.k != b.k {
            return TupleComparison::DimensionMismatch;
        }

        match self.mode {
            ComparisonMode::Lexicographic => {
                for i in 0..a.k {
                    if a.components[i] < b.components[i] {
                        return TupleComparison::Less;
                    }
                    if a.components[i] > b.components[i] {
                        return TupleComparison::Greater;
                    }
                }
                TupleComparison::Equal
            }
            ComparisonMode::L1Norm => {
                let na = a.l1_norm();
                let nb = b.l1_norm();
                if na < nb {
                    TupleComparison::Less
                } else if na > nb {
                    TupleComparison::Greater
                } else {
                    TupleComparison::Equal
                }
            }
            ComparisonMode::L2Norm => {
                let na = a.l2_norm();
                let nb = b.l2_norm();
                if na < nb {
                    TupleComparison::Less
                } else if na > nb {
                    TupleComparison::Greater
                } else {
                    TupleComparison::Equal
                }
            }
            ComparisonMode::MaxAbs => {
                let ma = a.max_abs();
                let mb = b.max_abs();
                if ma < mb {
                    TupleComparison::Less
                } else if ma > mb {
                    TupleComparison::Greater
                } else {
                    TupleComparison::Equal
                }
            }
        }
    }

    /// Returns true if a is less than b.
    pub fn less_than(&self, a: &SkyrmionTuple, b: &SkyrmionTuple) -> bool {
        matches!(self.compare(a, b), TupleComparison::Less)
    }

    /// Returns true if a is greater than b.
    pub fn greater_than(&self, a: &SkyrmionTuple, b: &SkyrmionTuple) -> bool {
        matches!(self.compare(a, b), TupleComparison::Greater)
    }

    /// Returns true if a is equal to b.
    pub fn equal_to(&self, a: &SkyrmionTuple, b: &SkyrmionTuple) -> bool {
        matches!(self.compare(a, b), TupleComparison::Equal)
    }
}

impl Default for SkyrmionTupleComparator {
    fn default() -> Self {
        Self::new(ComparisonMode::Lexicographic)
    }
}

// ============================================================================
// Field gradient computation
// ============================================================================

/// Gradient of a vector field at a grid point.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldGradient {
    /// Grid position (i, j).
    pub position: (usize, usize),
    /// Gradient vector (d/dx, d/dy) for each field component.
    pub grad_x: [f64; 3],
    pub grad_y: [f64; 3],
    /// Gradient magnitude.
    pub magnitude: f64,
}

/// Computes spatial gradients of a vector field.
pub struct FieldGradientComputer {
    /// Gradient stencil radius.
    pub radius: usize,
}

impl FieldGradientComputer {
    /// Create a new gradient computer.
    pub fn new(radius: usize) -> Self {
        Self { radius: radius.max(1) }
    }

    /// Compute gradient at a specific grid point using central differences.
    pub fn compute_at(
        &self,
        field: &[Vec<[f64; 3]>],
        i: usize,
        j: usize,
    ) -> Option<FieldGradient> {
        let n = field.len();
        if n == 0 || i >= n || j >= n {
            return None;
        }

        let r = self.radius as isize;
        let ni = i as isize;
        let nj = j as isize;

        // Central difference for d/dx (varying i).
        let ip = ((ni + r) as usize).min(n - 1);
        let im = ((ni - r).max(0)) as usize;
        let jp = ((nj + r) as usize).min(n - 1);
        let jm = ((nj - r).max(0)) as usize;

        let mut grad_x = [0.0f64; 3];
        let mut grad_y = [0.0f64; 3];

        for c in 0..3 {
            grad_x[c] = (field[ip][j][c] - field[im][j][c]) / (2.0 * r as f64);
            grad_y[c] = (field[i][jp][c] - field[i][jm][c]) / (2.0 * r as f64);
        }

        let magnitude = (grad_x.iter().map(|&v| v * v).sum::<f64>()
            + grad_y.iter().map(|&v| v * v).sum::<f64>())
        .sqrt();

        Some(FieldGradient {
            position: (i, j),
            grad_x,
            grad_y,
            magnitude,
        })
    }

    /// Compute gradients for the entire field.
    pub fn compute_all(&self, field: &[Vec<[f64; 3]>]) -> Vec<FieldGradient> {
        let mut gradients = Vec::new();
        let n = field.len();
        for i in 0..n {
            for j in 0..n {
                if let Some(g) = self.compute_at(field, i, j) {
                    gradients.push(g);
                }
            }
        }
        gradients
    }

    /// Returns the maximum gradient magnitude in the field.
    pub fn max_gradient(&self, field: &[Vec<[f64; 3]>]) -> f64 {
        self.compute_all(field)
            .iter()
            .map(|g| g.magnitude)
            .fold(0.0f64, f64::max)
    }

    /// Returns the average gradient magnitude.
    pub fn avg_gradient(&self, field: &[Vec<[f64; 3]>]) -> f64 {
        let gradients = self.compute_all(field);
        if gradients.is_empty() {
            return 0.0;
        }
        gradients.iter().map(|g| g.magnitude).sum::<f64>() / gradients.len() as f64
    }
}

impl Default for FieldGradientComputer {
    fn default() -> Self {
        Self::new(1)
    }
}

// ============================================================================
// Topological charge density computation
// ============================================================================

/// Topological charge density at a grid cell.
#[derive(Debug, Clone, PartialEq)]
pub struct ChargeDensity {
    /// Grid cell position (i, j).
    pub position: (usize, usize),
    /// Charge density value (charge per unit area).
    pub density: f64,
    /// Integrated charge over the cell.
    pub integrated_charge: f64,
}

/// Computes topological charge density from a vector field.
pub struct ChargeDensityComputer {
    /// Grid spacing (assumed uniform).
    pub spacing: f64,
}

impl ChargeDensityComputer {
    /// Create a new charge density computer.
    pub fn new(spacing: f64) -> Self {
        Self {
            spacing: spacing.max(1e-12),
        }
    }

    /// Compute charge density at each grid cell.
    pub fn compute(&self, field: &[Vec<[f64; 3]>]) -> Vec<ChargeDensity> {
        let mut densities = Vec::new();
        let n = field.len();
        if n < 2 {
            return densities;
        }

        let cell_area = self.spacing * self.spacing;

        for i in 0..n - 1 {
            for j in 0..n - 1 {
                let solid = spherical_triangle_solid_angle(
                    &field[i][j],
                    &field[i + 1][j],
                    &field[i][j + 1],
                );
                let solid2 = spherical_triangle_solid_angle(
                    &field[i + 1][j],
                    &field[i + 1][j + 1],
                    &field[i][j + 1],
                );
                let total_solid = solid + solid2;
                let charge = total_solid / (4.0 * PI);
                let density = charge / cell_area;

                densities.push(ChargeDensity {
                    position: (i, j),
                    density,
                    integrated_charge: charge,
                });
            }
        }

        densities
    }

    /// Returns the total integrated charge over all cells.
    pub fn total_charge(densities: &[ChargeDensity]) -> f64 {
        densities.iter().map(|d| d.integrated_charge).sum()
    }

    /// Returns the maximum charge density magnitude.
    pub fn max_density(densities: &[ChargeDensity]) -> f64 {
        densities
            .iter()
            .map(|d| d.density.abs())
            .fold(0.0f64, f64::max)
    }

    /// Returns cells with charge density above a threshold.
    pub fn above_threshold(
        densities: &[ChargeDensity],
        threshold: f64,
    ) -> Vec<&ChargeDensity> {
        densities
            .iter()
            .filter(|d| d.density.abs() > threshold)
            .collect()
    }
}

impl Default for ChargeDensityComputer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ============================================================================
// GeneralizedSkyrmionComputer additional analysis methods
// ============================================================================

impl GeneralizedSkyrmionComputer {
    /// Compute skyrmion number with error bounds.
    pub fn compute_with_error_bounds(
        &mut self,
        field: &[Vec<[f64; 3]>],
    ) -> Result<(GeneralizedSkyrmionNumber, f64), XpockError> {
        let result = self.compute(field)?;

        // Estimate numerical error from grid resolution and field variation.
        let n = field.len();
        let gradient_computer = FieldGradientComputer::default();
        let avg_grad = gradient_computer.avg_gradient(field);

        // Error scales with gradient magnitude / resolution.
        let error_estimate = (avg_grad / n as f64).clamp(0.0, 1.0);

        Ok((result, error_estimate))
    }

    /// Returns the quantization tolerance.
    pub fn quantization_tolerance(&self) -> f64 {
        self.config.quantization_tolerance
    }

    /// Set quantization tolerance.
    pub fn set_quantization_tolerance(&mut self, tolerance: f64) {
        self.config.quantization_tolerance = tolerance.clamp(0.0, 1.0);
    }
}

// ============================================================================
// Utility: validate skyrmion tuple against constellation
// ============================================================================

/// Validation result for a tuple against a constellation.
#[derive(Debug, Clone, PartialEq)]
pub struct TupleValidationResult {
    /// The validated tuple.
    pub tuple: SkyrmionTuple,
    /// Whether the tuple is in the constellation.
    pub in_constellation: bool,
    /// Nearest constellation point (if any).
    pub nearest_point: Option<usize>,
    /// Distance to nearest point.
    pub distance_to_nearest: f64,
    /// Validation confidence (0.0 to 1.0).
    pub confidence: f64,
}

/// Validates skyrmion tuples against a constellation.
pub struct TupleValidator {
    /// Constellation to validate against.
    pub constellation: SkyrmionConstellationBuilder,
    /// Maximum distance for a tuple to be considered "near" a point.
    pub max_distance: f64,
}

impl TupleValidator {
    /// Create a new tuple validator.
    pub fn new(constellation: SkyrmionConstellationBuilder, max_distance: f64) -> Self {
        Self {
            constellation,
            max_distance: max_distance.max(0.0),
        }
    }

    /// Validate a tuple against the constellation.
    pub fn validate(&self, tuple: &SkyrmionTuple) -> TupleValidationResult {
        let in_constellation = self.constellation.contains_tuple(tuple);
        let nearest = self.constellation.nearest_point(tuple);

        let (nearest_index, distance, confidence) = match nearest {
            Some(point) => {
                let dist = tuple_distance(tuple, &point.tuple);
                let idx = self
                    .constellation
                    .points
                    .iter()
                    .position(|p| p.symbol_index == point.symbol_index)
                    .unwrap_or(0);
                let conf = if dist <= self.max_distance {
                    (1.0 - dist / self.max_distance.max(1e-12)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (Some(idx), dist, conf)
            }
            None => (None, f64::INFINITY, 0.0),
        };

        TupleValidationResult {
            tuple: *tuple,
            in_constellation,
            nearest_point: nearest_index,
            distance_to_nearest: distance,
            confidence,
        }
    }

    /// Returns the constellation.
    pub fn constellation(&self) -> &SkyrmionConstellationBuilder {
        &self.constellation
    }

    /// Returns the max distance threshold.
    pub fn max_distance(&self) -> f64 {
        self.max_distance
    }
}

impl Default for TupleValidator {
    fn default() -> Self {
        Self::new(
            SkyrmionConstellationBuilder::new(1, 1).unwrap(),
            1.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // SkyrmionTuple tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_tuple_new_valid() {
        let tuple = SkyrmionTuple::new(&[1, -1, 0]).unwrap();
        assert_eq!(tuple.k, 3);
        assert_eq!(tuple.components[0], 1);
        assert_eq!(tuple.components[1], -1);
        assert_eq!(tuple.components[2], 0);
        assert!(tuple.valid);
    }

    #[test]
    fn skyrmion_tuple_new_empty() {
        let result = SkyrmionTuple::new(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_tuple_new_too_large() {
        let components: Vec<i64> = (0..9).map(|_| 0).collect();
        let result = SkyrmionTuple::new(&components);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_tuple_zero() {
        let tuple = SkyrmionTuple::zero(4).unwrap();
        assert_eq!(tuple.k, 4);
        assert!(tuple.is_zero());
    }

    #[test]
    fn skyrmion_tuple_zero_invalid_k() {
        let result = SkyrmionTuple::zero(0);
        assert!(result.is_err());
        let result = SkyrmionTuple::zero(9);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_tuple_component_get_set() {
        let mut tuple = SkyrmionTuple::new(&[1, 2, 3]).unwrap();
        assert_eq!(tuple.component(0), Some(1));
        assert_eq!(tuple.component(2), Some(3));
        assert_eq!(tuple.component(5), None);

        tuple.set_component(1, 10).unwrap();
        assert_eq!(tuple.component(1), Some(10));

        let err = tuple.set_component(5, 1);
        assert!(err.is_err());
    }

    #[test]
    fn skyrmion_tuple_l1_norm() {
        let tuple = SkyrmionTuple::new(&[3, -4, 0]).unwrap();
        assert_eq!(tuple.l1_norm(), 7);
    }

    #[test]
    fn skyrmion_tuple_l2_norm() {
        let tuple = SkyrmionTuple::new(&[3, 4]).unwrap();
        assert!((tuple.l2_norm() - 5.0).abs() < 0.001);
    }

    #[test]
    fn skyrmion_tuple_max_abs() {
        let tuple = SkyrmionTuple::new(&[-3, 2, -7]).unwrap();
        assert_eq!(tuple.max_abs(), 7);
    }

    #[test]
    fn skyrmion_tuple_is_zero() {
        let t1 = SkyrmionTuple::new(&[0, 0]).unwrap();
        assert!(t1.is_zero());
        let t2 = SkyrmionTuple::new(&[0, 1]).unwrap();
        assert!(!t2.is_zero());
    }

    #[test]
    fn skyrmion_tuple_sign_pattern() {
        let tuple = SkyrmionTuple::new(&[-1, 0, 3]).unwrap();
        let pattern = tuple.sign_pattern();
        assert_eq!(pattern[0], -1);
        assert_eq!(pattern[1], 0);
        assert_eq!(pattern[2], 1);
    }

    #[test]
    fn skyrmion_tuple_encode() {
        let tuple = SkyrmionTuple::new(&[1, 2]).unwrap();
        let hash = tuple.encode();
        assert_ne!(hash, 0);
    }

    #[test]
    fn skyrmion_tuple_display() {
        let tuple = SkyrmionTuple::new(&[1, -2]).unwrap();
        let s = format!("{}", tuple);
        assert!(s.contains("1"));
        assert!(s.contains("-2"));
    }

    #[test]
    fn skyrmion_tuple_add() {
        let a = SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = SkyrmionTuple::new(&[3, 4]).unwrap();
        let result = (a + b).unwrap();
        assert_eq!(result.components[0], 4);
        assert_eq!(result.components[1], 6);
    }

    #[test]
    fn skyrmion_tuple_add_dimension_mismatch() {
        let a = SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = SkyrmionTuple::new(&[3]).unwrap();
        let result = a + b;
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_tuple_sub() {
        let a = SkyrmionTuple::new(&[5, 5]).unwrap();
        let b = SkyrmionTuple::new(&[2, 3]).unwrap();
        let result = (a - b).unwrap();
        assert_eq!(result.components[0], 3);
        assert_eq!(result.components[1], 2);
    }

    #[test]
    fn skyrmion_tuple_mul() {
        let a = SkyrmionTuple::new(&[1, -2]).unwrap();
        let result = (a * 3).unwrap();
        assert_eq!(result.components[0], 3);
        assert_eq!(result.components[1], -6);
    }

    #[test]
    fn skyrmion_tuple_is_non_negative() {
        let t1 = SkyrmionTuple::new(&[0, 1, 2]).unwrap();
        assert!(t1.is_non_negative());
        let t2 = SkyrmionTuple::new(&[-1, 0]).unwrap();
        assert!(!t2.is_non_negative());
    }

    #[test]
    fn skyrmion_tuple_is_non_positive() {
        let t1 = SkyrmionTuple::new(&[0, -1]).unwrap();
        assert!(t1.is_non_positive());
        let t2 = SkyrmionTuple::new(&[1, 0]).unwrap();
        assert!(!t2.is_non_positive());
    }

    #[test]
    fn skyrmion_tuple_default() {
        let tuple = SkyrmionTuple::default();
        assert_eq!(tuple.k, 1);
        assert!(tuple.is_zero());
    }

    // ---------------------------------------------------------------------------
    // GeneralizedSkyrmionNumber tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generalized_skyrmion_number_new() {
        let result = GeneralizedSkyrmionNumber::new(3, 128);
        assert_eq!(result.k, 3);
        assert_eq!(result.grid_resolution, 128);
        assert!(!result.success);
    }

    #[test]
    fn generalized_skyrmion_number_set_from_charges() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result
            .set_from_charges(&[1.02, -0.98])
            .unwrap();
        assert!(result.success);
        assert_eq!(result.tuple.components[0], 1);
        assert_eq!(result.tuple.components[1], -1);
        assert!((result.quantization_errors[0] - 0.02).abs() < 0.001);
        assert!((result.quantization_errors[1] - 0.02).abs() < 0.001);
    }

    #[test]
    fn generalized_skyrmion_number_set_from_charges_wrong_dim() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        let err = result.set_from_charges(&[1.0]);
        assert!(err.is_err());
    }

    #[test]
    fn generalized_skyrmion_number_charge_magnitude() {
        let mut result = GeneralizedSkyrmionNumber::new(3, 256);
        result
            .set_from_charges(&[3.0, -4.0, 0.0])
            .unwrap();
        assert_eq!(result.charge_magnitude(), 7);
    }

    #[test]
    fn generalized_skyrmion_number_is_well_quantized() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result
            .set_from_charges(&[1.01, -0.99])
            .unwrap();
        assert!(result.is_well_quantized(0.1));
        assert!(!result.is_well_quantized(0.005));
    }

    #[test]
    fn generalized_skyrmion_number_max_quantization_error() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result
            .set_from_charges(&[1.05, -0.02])
            .unwrap();
        assert!((result.max_quantization_error() - 0.05).abs() < 0.001);
    }

    #[test]
    fn generalized_skyrmion_number_dominant_component() {
        let mut result = GeneralizedSkyrmionNumber::new(3, 256);
        result
            .set_from_charges(&[1.0, -5.0, 2.0])
            .unwrap();
        let dominant = result.dominant_component().unwrap();
        assert_eq!(dominant.0, 1);
        assert_eq!(dominant.1, -5);
    }

    #[test]
    fn generalized_skyrmion_number_reset() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result.set_from_charges(&[1.0, 2.0]).unwrap();
        assert!(result.success);
        result.reset();
        assert!(!result.success);
    }

    #[test]
    fn generalized_skyrmion_number_default() {
        let result = GeneralizedSkyrmionNumber::default();
        assert_eq!(result.k, 1);
        assert_eq!(result.grid_resolution, 256);
    }

    // ---------------------------------------------------------------------------
    // BoundaryMode tests
    // ---------------------------------------------------------------------------

    #[test]
    fn boundary_mode_variants() {
        assert_eq!(BoundaryMode::Periodic as u32, 0);
        assert_eq!(BoundaryMode::Zero as u32, 1);
        assert_eq!(BoundaryMode::Constant as u32, 2);
        assert_eq!(BoundaryMode::Reflect as u32, 3);
    }

    // ---------------------------------------------------------------------------
    // GeneralizedSkyrmionConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generalized_skyrmion_config_default() {
        let config = GeneralizedSkyrmionConfig::default();
        assert_eq!(config.k, 1);
        assert_eq!(config.grid_resolution, 256);
        assert!((config.quantization_tolerance - 0.1).abs() < 0.001);
        assert_eq!(config.smoothing_radius, 1);
        assert_eq!(config.boundary_mode, BoundaryMode::Periodic);
    }

    // ---------------------------------------------------------------------------
    // GeneralizedSkyrmionComputer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generalized_skyrmion_computer_new() {
        let config = GeneralizedSkyrmionConfig::default();
        let computer = GeneralizedSkyrmionComputer::new(config);
        assert_eq!(computer.k(), 1);
        assert_eq!(computer.grid_resolution(), 256);
        assert_eq!(computer.history().len(), 0);
    }

    #[test]
    fn generalized_skyrmion_compute_uniform_field() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let n = 4;
        let field = [[[1.0f64, 0.0, 0.0]; 4]; 4];
        let field: Vec<Vec<[f64; 3]>> = field.iter().map(|row| row.to_vec()).collect();
        let field: Vec<Vec<[f64; 3]>> = field;
        let result = computer.compute(&field).unwrap();
        assert!(result.success);
        assert_eq!(result.k, 1);
        // Uniform field has zero skyrmion number.
        assert_eq!(result.tuple.components[0], 0);
    }

    #[test]
    fn generalized_skyrmion_compute_wrong_shape() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let field: Vec<Vec<[f64; 3]>> = vec![vec![[0.0; 3]; 3], vec![[0.0; 3]; 2]];
        let result = computer.compute(&field);
        assert!(result.is_err());
    }

    #[test]
    fn generalized_skyrmion_compute_empty_field() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let field: Vec<Vec<[f64; 3]>> = vec![];
        let result = computer.compute(&field);
        assert!(result.is_err());
    }

    #[test]
    fn generalized_skyrmion_compute_k3() {
        let mut config = GeneralizedSkyrmionConfig::default();
        config.k = 3;
        let mut computer = GeneralizedSkyrmionComputer::new(config);
        let n = 4;
        let field = [[[1.0f64, 0.0, 0.0]; 4]; 4];
        let field: Vec<Vec<[f64; 3]>> = field.iter().map(|row| row.to_vec()).collect();
        let result = computer.compute(&field).unwrap();
        assert_eq!(result.k, 3);
        assert!(result.success);
    }

    #[test]
    fn generalized_skyrmion_compute_from_components() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let n = 3;
        let nx = vec![vec![1.0f64; n]; n];
        let ny = vec![vec![0.0f64; n]; n];
        let nz = vec![vec![0.0f64; n]; n];
        let result = computer
            .compute_from_components(&nx, &ny, &nz)
            .unwrap();
        assert!(result.success);
        assert_eq!(result.tuple.components[0], 0);
    }

    #[test]
    fn generalized_skyrmion_compute_from_components_dim_mismatch() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let nx = vec![vec![1.0f64; 3]; 3];
        let ny = vec![vec![0.0f64; 2]; 2];
        let nz = vec![vec![0.0f64; 3]; 3];
        let result = computer.compute_from_components(&nx, &ny, &nz);
        assert!(result.is_err());
    }

    #[test]
    fn generalized_skyrmion_computer_history() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let field = [[[1.0f64, 0.0, 0.0]; 2]; 2];
        let field: Vec<Vec<[f64; 3]>> = field.iter().map(|row| row.to_vec()).collect();
        computer.compute(&field).unwrap();
        assert_eq!(computer.history().len(), 1);
        assert!(computer.latest().is_some());
    }

    #[test]
    fn generalized_skyrmion_computer_reset() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let field = [[[1.0f64, 0.0, 0.0]; 2]; 2];
        let field: Vec<Vec<[f64; 3]>> = field.iter().map(|row| row.to_vec()).collect();
        computer.compute(&field).unwrap();
        assert!(!computer.history().is_empty());
        computer.reset();
        assert_eq!(computer.history().len(), 0);
    }

    #[test]
    fn generalized_skyrmion_computer_set_config() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig::default(),
        );
        let mut new_config = GeneralizedSkyrmionConfig::default();
        new_config.k = 2;
        new_config.grid_resolution = 512;
        computer.set_config(new_config.clone());
        assert_eq!(computer.k(), 2);
        assert_eq!(computer.grid_resolution(), 512);
    }

    // ---------------------------------------------------------------------------
    // Solid angle helper tests
    // ---------------------------------------------------------------------------

    #[test]
    fn dot_product_test() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!((dot_product(&a, &b) - 0.0).abs() < 0.001);
        let c = [1.0, 2.0, 3.0];
        let d = [4.0, 5.0, 6.0];
        assert!((dot_product(&c, &d) - 32.0).abs() < 0.001);
    }

    #[test]
    fn cross_product_test() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross_product(&a, &b);
        assert!((c[0] - 0.0).abs() < 0.001);
        assert!((c[1] - 0.0).abs() < 0.001);
        assert!((c[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn normalize_vector_test() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize_vector(&v);
        assert!((n[0] - 0.6).abs() < 0.001);
        assert!((n[1] - 0.8).abs() < 0.001);
        assert!((n[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn normalize_vector_zero() {
        let v = [0.0, 0.0, 0.0];
        let n = normalize_vector(&v);
        assert_eq!(n, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn spherical_triangle_solid_angle_identical() {
        let v = [1.0, 0.0, 0.0];
        let omega = spherical_triangle_solid_angle(&v, &v, &v, &v);
        assert!((omega - 0.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionConstellationPoint tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_constellation_point_creation() {
        let tuple = SkyrmionTuple::new(&[1, 0]).unwrap();
        let point = SkyrmionConstellationPoint {
            tuple,
            symbol_index: 5,
            bit_pattern: vec![true, false],
            min_distance: 2.0,
        };
        assert_eq!(point.symbol_index, 5);
        assert_eq!(point.bit_pattern, vec![true, false]);
        assert!((point.min_distance - 2.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionConstellationBuilder tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_constellation_builder_new() {
        let builder = SkyrmionConstellationBuilder::new(2, 3).unwrap();
        assert_eq!(builder.k, 2);
        assert_eq!(builder.num_points(), 0);
    }

    #[test]
    fn skyrmion_constellation_builder_invalid_k() {
        let result = SkyrmionConstellationBuilder::new(0, 3);
        assert!(result.is_err());
        let result = SkyrmionConstellationBuilder::new(9, 3);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_constellation_builder_invalid_max_charge() {
        let result = SkyrmionConstellationBuilder::new(1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_constellation_builder_add_point() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        let tuple = SkyrmionTuple::new(&[1]).unwrap();
        builder.add_point(tuple, 0).unwrap();
        assert_eq!(builder.num_points(), 1);
    }

    #[test]
    fn skyrmion_constellation_builder_add_point_wrong_k() {
        let mut builder = SkyrmionConstellationBuilder::new(2, 3).unwrap();
        let tuple = SkyrmionTuple::new(&[1]).unwrap();
        let result = builder.add_point(tuple, 0);
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_constellation_builder_nearest_point() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[0]).unwrap(), 0).unwrap();
        builder.add_point(SkyrmionTuple::new(&[5]).unwrap(), 1).unwrap();

        let query = SkyrmionTuple::new(&[2]).unwrap();
        let nearest = builder.nearest_point(&query).unwrap();
        assert_eq!(nearest.symbol_index, 0); // Closer to 0 than to 5.
    }

    #[test]
    fn skyrmion_constellation_builder_nearest_point_empty() {
        let builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        let query = SkyrmionTuple::new(&[0]).unwrap();
        assert!(builder.nearest_point(&query).is_none());
    }

    #[test]
    fn skyrmion_constellation_builder_clear() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[1]).unwrap(), 0).unwrap();
        builder.clear();
        assert_eq!(builder.num_points(), 0);
    }

    #[test]
    fn skyrmion_constellation_builder_default() {
        let builder = SkyrmionConstellationBuilder::default();
        assert_eq!(builder.k, 1);
        assert_eq!(builder.num_points(), 0);
    }

    // ---------------------------------------------------------------------------
    // Tuple distance tests
    // ---------------------------------------------------------------------------

    #[test]
    fn tuple_distance_test() {
        let a = SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = SkyrmionTuple::new(&[4, 6]).unwrap();
        let dist = tuple_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001); // sqrt(3^2 + 4^2) = 5.
    }

    #[test]
    fn tuple_distance_same() {
        let a = SkyrmionTuple::new(&[3, 3]).unwrap();
        let b = SkyrmionTuple::new(&[3, 3]).unwrap();
        assert!((tuple_distance(&a, &b) - 0.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // Edge case tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generalized_skyrmion_single_component() {
        let mut computer = GeneralizedSkyrmionComputer::new(
            GeneralizedSkyrmionConfig {
                k: 1,
                ..Default::default()
            },
        );
        let field = [[[0.0f64, 0.0, 1.0]; 2]; 2];
        let field: Vec<Vec<[f64; 3]>> = field.iter().map(|row| row.to_vec()).collect();
        let result = computer.compute(&field).unwrap();
        assert!(result.success);
    }

    #[test]
    fn generalized_skyrmion_max_components() {
        let mut config = GeneralizedSkyrmionConfig::default();
        config.k = 8;
        let mut computer = GeneralizedSkyrmionComputer::new(config);
        let field = [[[1.0f64, 0.0, 0.0]; 2]; 2];
        let field: Vec<Vec<[f64; 3]>> = field.iter().map(|row| row.to_vec()).collect();
        let result = computer.compute(&field).unwrap();
        assert_eq!(result.k, 8);
        assert!(result.success);
    }

    #[test]
    fn skyrmion_tuple_large_values() {
        let tuple = SkyrmionTuple::new(&[1000, -2000]).unwrap();
        assert_eq!(tuple.l1_norm(), 3000);
        assert_eq!(tuple.max_abs(), 2000);
    }

    #[test]
    fn generalized_skyrmion_quantization_quality_bounds() {
        let mut result = GeneralizedSkyrmionNumber::new(1, 256);
        result.set_from_charges(&[0.5]).unwrap();
        assert!(result.quantization_quality >= 0.0);
        assert!(result.quantization_quality <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionTuple iteration and serialization tests
    // ---------------------------------------------------------------------------

    #[test]
    fn skyrmion_tuple_iter() {
        let tuple = SkyrmionTuple::new(&[1, 2, 3]).unwrap();
        let vals: Vec<i64> = tuple.iter().collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn skyrmion_tuple_iter_into_iterator() {
        let tuple = SkyrmionTuple::new(&[5, 6]).unwrap();
        let vals: Vec<i64> = (&tuple).into_iter().collect();
        assert_eq!(vals, vec![5, 6]);
    }

    #[test]
    fn skyrmion_tuple_len_is_empty() {
        let t1 = SkyrmionTuple::new(&[1]).unwrap();
        assert_eq!(t1.len(), 1);
        assert!(!t1.is_empty());
    }

    #[test]
    fn skyrmion_tuple_to_from_bytes_roundtrip() {
        let original = SkyrmionTuple::new(&[3, -1, 7]).unwrap();
        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), original.serialized_size());
        let restored = SkyrmionTuple::from_bytes(&bytes).unwrap();
        assert_eq!(restored.k, original.k);
        assert_eq!(restored.components[0], 3);
        assert_eq!(restored.components[1], -1);
        assert_eq!(restored.components[2], 7);
    }

    #[test]
    fn skyrmion_tuple_from_bytes_errors() {
        let result = SkyrmionTuple::from_bytes(&[]);
        assert!(result.is_err());

        let result = SkyrmionTuple::from_bytes(&[0]);
        assert!(result.is_err());

        let result = SkyrmionTuple::from_bytes(&[9]);
        assert!(result.is_err());

        let result = SkyrmionTuple::from_bytes(&[1, 0, 0, 0, 0, 0, 0, 0, 0]); // too short
        assert!(result.is_err());
    }

    #[test]
    fn skyrmion_tuple_clamp_components() {
        let mut tuple = SkyrmionTuple::new(&[10, -5, 100]).unwrap();
        tuple.clamp_components(10);
        assert_eq!(tuple.components[0], 10);
        assert_eq!(tuple.components[1], -5);
        assert_eq!(tuple.components[2], 10);
    }

    #[test]
    fn skyrmion_tuple_exceeds() {
        let tuple = SkyrmionTuple::new(&[3, -7]).unwrap();
        assert!(!tuple.exceeds(7));
        assert!(tuple.exceeds(6));
    }

    #[test]
    fn skyrmion_tuple_abs_neg() {
        let tuple = SkyrmionTuple::new(&[-3, 5]).unwrap();
        let abs_t = tuple.abs();
        assert_eq!(abs_t.components[0], 3);
        assert_eq!(abs_t.components[1], 5);

        let neg_t = tuple.neg();
        assert_eq!(neg_t.components[0], 3);
        assert_eq!(neg_t.components[1], -5);
    }

    #[test]
    fn skyrmion_tuple_dot() {
        let a = SkyrmionTuple::new(&[1, 2, 3]).unwrap();
        let b = SkyrmionTuple::new(&[4, 5, 6]).unwrap();
        assert_eq!(a.dot(&b).unwrap(), 32); // 1*4 + 2*5 + 3*6 = 32.
    }

    #[test]
    fn skyrmion_tuple_dot_dimension_mismatch() {
        let a = SkyrmionTuple::new(&[1, 2]).unwrap();
        let b = SkyrmionTuple::new(&[1]).unwrap();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn skyrmion_tuple_cosine_similarity() {
        let a = SkyrmionTuple::new(&[1, 0]).unwrap();
        let b = SkyrmionTuple::new(&[0, 1]).unwrap();
        assert!((a.cosine_similarity(&b).unwrap() - 0.0).abs() < 0.001);

        let c = SkyrmionTuple::new(&[1, 1]).unwrap();
        assert!((a.cosine_similarity(&c).unwrap() - 0.7071).abs() < 0.01);
    }

    #[test]
    fn skyrmion_tuple_cosine_zero_vector() {
        let a = SkyrmionTuple::new(&[0, 0]).unwrap();
        let b = SkyrmionTuple::new(&[1, 1]).unwrap();
        assert!((a.cosine_similarity(&b).unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn skyrmion_tuple_angular_distance() {
        let a = SkyrmionTuple::new(&[1, 0]).unwrap();
        let b = SkyrmionTuple::new(&[0, 1]).unwrap();
        let dist = a.angular_distance(&b).unwrap();
        assert!((dist - std::f64::consts::PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn skyrmion_tuple_as_slice() {
        let tuple = SkyrmionTuple::new(&[1, 2, 3]).unwrap();
        assert_eq!(tuple.as_slice(), &[1, 2, 3]);
    }

    // ---------------------------------------------------------------------------
    // GeneralizedSkyrmionNumber analysis tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generalized_skyrmion_number_estimated_smoothing_radius() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result.set_from_charges(&[1.01, -0.99]).unwrap();
        assert_eq!(result.estimated_smoothing_radius(), 1);

        let mut result2 = GeneralizedSkyrmionNumber::new(2, 256);
        result2.set_from_charges(&[1.5, -0.5]).unwrap();
        assert!(result2.estimated_smoothing_radius() >= 2);
    }

    #[test]
    fn generalized_skyrmion_number_is_valid_signal() {
        let mut result = GeneralizedSkyrmionNumber::new(1, 256);
        result.set_from_charges(&[1.0]).unwrap();
        assert!(result.is_valid_signal(0.5));

        let mut result2 = GeneralizedSkyrmionNumber::new(1, 256);
        result2.set_from_charges(&[0.0]).unwrap();
        assert!(!result2.is_valid_signal(0.5)); // Zero tuple.

        let mut result3 = GeneralizedSkyrmionNumber::new(1, 256);
        result3.set_from_charges(&[1.45]).unwrap(); // Poor quantization.
        assert!(!result3.is_valid_signal(0.9));
    }

    #[test]
    fn generalized_skyrmion_number_num_nonzero_components() {
        let mut result = GeneralizedSkyrmionNumber::new(3, 256);
        result.set_from_charges(&[3.0, 0.0, -2.0]).unwrap();
        assert_eq!(result.num_nonzero_components(), 2);
    }

    #[test]
    fn generalized_skyrmion_number_charge_entropy() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result.set_from_charges(&[1.0, 1.0]).unwrap();
        let entropy = result.charge_entropy();
        assert!((entropy - std::f64::consts::LN_2).abs() < 0.01); // ln(2) for equal split.

        let mut result2 = GeneralizedSkyrmionNumber::new(2, 256);
        result2.set_from_charges(&[5.0, 0.0]).unwrap();
        assert!((result2.charge_entropy() - 0.0).abs() < 0.001); // All in one component.
    }

    #[test]
    fn generalized_skyrmion_number_is_charge_concentrated() {
        let mut result = GeneralizedSkyrmionNumber::new(2, 256);
        result.set_from_charges(&[5.0, 0.0]).unwrap();
        assert!(result.is_charge_concentrated(0.9));

        let mut result2 = GeneralizedSkyrmionNumber::new(2, 256);
        result2.set_from_charges(&[3.0, 3.0]).unwrap();
        assert!(!result2.is_charge_concentrated(0.9));
    }

    // ---------------------------------------------------------------------------
    // FieldSmoother tests
    // ---------------------------------------------------------------------------

    #[test]
    fn field_smoother_new() {
        let smoother = FieldSmoother::new(2);
        assert_eq!(smoother.radius, 2);
    }

    #[test]
    fn field_smoother_smooth_uniform() {
        let smoother = FieldSmoother::new(1);
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 4]; 4];
        let result = smoother.smooth(&field);
        assert_eq!(result.radius, 1);
        // Uniform field should not change (after normalization, approximately).
        assert!((result.smoothed_field[0][0][0] - 1.0).abs() < 0.001);
        assert!(result.variation_reduction >= 0.0);
        assert!(result.variation_reduction <= 1.0);
    }

    #[test]
    fn field_smoother_smooth_empty() {
        let smoother = FieldSmoother::new(1);
        let field: Vec<Vec<[f64; 3]>> = vec![];
        let result = smoother.smooth(&field);
        assert_eq!(result.smoothed_field.len(), 0);
        assert_eq!(result.variation_reduction, 0.0);
    }

    #[test]
    fn field_smoother_reduces_variation() {
        let smoother = FieldSmoother::new(1);
        let mut field = vec![vec![[0.0f64, 0.0, 0.0]; 4]; 4];
        field[0][0] = [1.0, 0.0, 0.0];
        field[3][3] = [-1.0, 0.0, 0.0];
        let result = smoother.smooth(&field);
        assert!(result.variation_reduction >= 0.0);
    }

    #[test]
    fn field_smoother_default() {
        let smoother = FieldSmoother::default();
        assert_eq!(smoother.radius, 1);
    }

    // ---------------------------------------------------------------------------
    // FieldDefectDetector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn field_defect_detector_new() {
        let detector = FieldDefectDetector::new();
        assert!((detector.min_strength - 0.1).abs() < 0.001);
        assert_eq!(detector.cell_size, 2);
    }

    #[test]
    fn field_defect_detector_uniform_field() {
        let detector = FieldDefectDetector::new();
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 4]; 4];
        let defects = detector.detect(&field);
        assert!(defects.is_empty()); // Uniform field has no defects.
    }

    #[test]
    fn field_defect_detector_small_field() {
        let detector = FieldDefectDetector::new();
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 1]; 1];
        let defects = detector.detect(&field);
        assert!(defects.is_empty());
    }

    #[test]
    fn field_defect_detector_total_charge() {
        let defects = vec![
            FieldDefect {
                position: (0, 0),
                defect_type: DefectType::Skyrmion,
                strength: 1.0,
                charge: 1.0,
            },
            FieldDefect {
                position: (1, 1),
                defect_type: DefectType::Antiskyrmion,
                strength: 1.0,
                charge: -1.0,
            },
        ];
        assert!((FieldDefectDetector::total_charge(&defects) - 0.0).abs() < 0.001);
    }

    #[test]
    fn field_defect_detector_by_type() {
        let defects = vec![
            FieldDefect {
                position: (0, 0),
                defect_type: DefectType::Skyrmion,
                strength: 1.0,
                charge: 1.0,
            },
            FieldDefect {
                position: (1, 1),
                defect_type: DefectType::Antiskyrmion,
                strength: 1.0,
                charge: -1.0,
            },
        ];
        let skyrmions = FieldDefectDetector::by_type(&defects, DefectType::Skyrmion);
        assert_eq!(skyrmions.len(), 1);
        let antiskyrmions = FieldDefectDetector::by_type(&defects, DefectType::Antiskyrmion);
        assert_eq!(antiskyrmions.len(), 1);
    }

    #[test]
    fn field_defect_detector_default() {
        let detector = FieldDefectDetector::default();
        assert!((detector.min_strength - 0.1).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // DefectType tests
    // ---------------------------------------------------------------------------

    #[test]
    fn defect_type_variants() {
        assert_eq!(DefectType::Skyrmion as u32, 0);
        assert_eq!(DefectType::Antiskyrmion as u32, 1);
        assert_eq!(DefectType::Meron as u32, 2);
        assert_eq!(DefectType::Vortex as u32, 3);
        assert_eq!(DefectType::None as u32, 4);
    }

    // ---------------------------------------------------------------------------
    // MultiScaleSkyrmionComputer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_scale_computer_new() {
        let computer = GeneralizedSkyrmionComputer::default();
        let ms = MultiScaleSkyrmionComputer::new(computer, 3, 1.5);
        assert_eq!(ms.num_scales(), 3);
        assert!((ms.scale_step() - 1.5).abs() < 0.001);
    }

    #[test]
    fn multi_scale_compute_uniform_field() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ms = MultiScaleSkyrmionComputer::new(computer, 3, 1.5);
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 8]; 8];
        let result = ms.compute_multi_scale(&field).unwrap();
        assert!(!result.scale_results.is_empty());
        assert_eq!(result.scale_factors.len(), result.scale_results.len());
        assert!(result.convergence_quality >= 0.0);
        assert!(result.convergence_quality <= 1.0);
    }

    #[test]
    fn multi_scale_empty_field() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ms = MultiScaleSkyrmionComputer::new(computer, 3, 1.5);
        let field: Vec<Vec<[f64; 3]>> = vec![];
        let result = ms.compute_multi_scale(&field);
        assert!(result.is_err());
    }

    #[test]
    fn multi_scale_num_scales_min_one() {
        let computer = GeneralizedSkyrmionComputer::default();
        let ms = MultiScaleSkyrmionComputer::new(computer, 0, 1.5);
        assert_eq!(ms.num_scales(), 1);
    }

    #[test]
    fn multi_scale_scale_step_min() {
        let computer = GeneralizedSkyrmionComputer::default();
        let ms = MultiScaleSkyrmionComputer::new(computer, 3, 0.5); // Should clamp to 1.01.
        assert!((ms.scale_step() - 1.01).abs() < 0.001);
    }

    #[test]
    fn multi_scale_default() {
        let ms = MultiScaleSkyrmionComputer::default();
        assert_eq!(ms.num_scales(), 3);
    }

    // ---------------------------------------------------------------------------
    // AdaptiveSkyrmionComputer tests
    // ---------------------------------------------------------------------------

    #[test]
    fn adaptive_computer_new() {
        let computer = GeneralizedSkyrmionComputer::default();
        let ac = AdaptiveSkyrmionComputer::new(computer, 16, 128);
        assert_eq!(ac.min_resolution(), 16);
        assert_eq!(ac.max_resolution(), 128);
    }

    #[test]
    fn adaptive_compute_uniform_field() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ac = AdaptiveSkyrmionComputer::new(computer, 8, 64);
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 16]; 16];
        let (result, resolution) = ac.compute_adaptive(&field).unwrap();
        assert!(result.success);
        assert!(resolution >= 8);
    }

    #[test]
    fn adaptive_compute_field_too_small() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ac = AdaptiveSkyrmionComputer::new(computer, 32, 64);
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 8]; 8];
        let result = ac.compute_adaptive(&field);
        assert!(result.is_err());
    }

    #[test]
    fn adaptive_set_complexity_threshold() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ac = AdaptiveSkyrmionComputer::new(computer, 8, 64);
        ac.set_complexity_threshold(0.5);
        assert!((ac.computer.config.quantization_tolerance - 0.1).abs() < 0.001);
    }

    #[test]
    fn adaptive_default() {
        let ac = AdaptiveSkyrmionComputer::default();
        assert_eq!(ac.min_resolution(), 32);
        assert_eq!(ac.max_resolution(), 256);
    }

    // ---------------------------------------------------------------------------
    // Solid angle formulation tests
    // ---------------------------------------------------------------------------

    #[test]
    fn solid_angle_lhuilier_identical() {
        let v = [1.0, 0.0, 0.0];
        let omega = solid_angle_lhuilier(&v, &v, &v);
        assert!((omega - 0.0).abs() < 0.001);
    }

    #[test]
    fn solid_angle_triple_product_identical() {
        let v = [1.0, 0.0, 0.0];
        let omega = solid_angle_triple_product(&v, &v, &v);
        assert!((omega - 0.0).abs() < 0.001);
    }

    #[test]
    fn solid_angle_triple_product_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        let omega = solid_angle_triple_product(&a, &b, &c);
        // Solid angle of octant = 4π/8 = π/2.
        assert!((omega - std::f64::consts::PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn solid_angle_error_bound() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        let err = solid_angle_error_bound(&a, &b, &c);
        assert!(err >= 0.0);
        assert!(err < 1.0);
    }

    #[test]
    fn solid_angle_error_bound_zero_vector() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let err = solid_angle_error_bound(&a, &b, &c);
        assert_eq!(err, 0.0);
    }

    // ---------------------------------------------------------------------------
    // SkyrmionConstellationBuilder spacing/capacity/error correction tests
    // ---------------------------------------------------------------------------

    #[test]
    fn constellation_spacing_analysis_single_point() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[0]).unwrap(), 0).unwrap();
        let analysis = builder.spacing_analysis();
        assert_eq!(analysis.num_pairs, 0);
        assert_eq!(analysis.uniformity, 1.0);
    }

    #[test]
    fn constellation_spacing_analysis_two_points() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[0]).unwrap(), 0).unwrap();
        builder.add_point(SkyrmionTuple::new(&[3]).unwrap(), 1).unwrap();
        let analysis = builder.spacing_analysis();
        assert_eq!(analysis.num_pairs, 1);
        assert!((analysis.min_distance - 3.0).abs() < 0.001);
        assert!((analysis.max_distance - 3.0).abs() < 0.001);
        assert!((analysis.mean_distance - 3.0).abs() < 0.001);
        assert_eq!(analysis.std_distance, 0.0);
    }

    #[test]
    fn constellation_capacity_analysis() {
        let mut builder = SkyrmionConstellationBuilder::new(2, 1).unwrap();
        builder.add_point(SkyrmionTuple::new(&[0, 0]).unwrap(), 0).unwrap();
        builder.add_point(SkyrmionTuple::new(&[1, 0]).unwrap(), 1).unwrap();
        let cap = builder.capacity_analysis();
        assert_eq!(cap.k, 2);
        assert_eq!(cap.max_charge, 1);
        assert_eq!(cap.theoretical_capacity, 9); // (2*1+1)^2 = 9.
        assert_eq!(cap.actual_points, 2);
        assert!((cap.utilization - 2.0 / 9.0).abs() < 0.001);
        assert!((cap.bits_per_symbol - 1.0).abs() < 0.001); // log2(2) = 1.
    }

    #[test]
    fn constellation_capacity_empty() {
        let builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        let cap = builder.capacity_analysis();
        assert_eq!(cap.actual_points, 0);
        assert_eq!(cap.bits_per_symbol, 0.0);
    }

    #[test]
    fn constellation_error_correction_single_point() {
        let builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        let ec = builder.error_correction();
        assert_eq!(ec.correctable_errors, 0);
        assert_eq!(ec.detectable_errors, 0);
    }

    #[test]
    fn constellation_error_correction_two_points() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[0]).unwrap(), 0).unwrap();
        builder.add_point(SkyrmionTuple::new(&[5]).unwrap(), 1).unwrap();
        let ec = builder.error_correction();
        assert!((ec.min_distance - 5.0).abs() < 0.001);
        assert!(ec.correctable_errors > 0);
        assert!(ec.strength >= 0.0);
        assert!(ec.strength <= 1.0);
    }

    #[test]
    fn constellation_is_well_separated() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[0]).unwrap(), 0).unwrap();
        builder.add_point(SkyrmionTuple::new(&[5]).unwrap(), 1).unwrap();
        assert!(builder.is_well_separated(1.0));
        assert!(!builder.is_well_separated(10.0));
    }

    #[test]
    fn constellation_point_at() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[1]).unwrap(), 0).unwrap();
        let point = builder.point_at(0).unwrap();
        assert_eq!(point.symbol_index, 0);
        assert!(builder.point_at(5).is_none());
    }

    #[test]
    fn constellation_contains_tuple() {
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(SkyrmionTuple::new(&[2]).unwrap(), 0).unwrap();
        assert!(builder.contains_tuple(&SkyrmionTuple::new(&[2]).unwrap()));
        assert!(!builder.contains_tuple(&SkyrmionTuple::new(&[3]).unwrap()));
    }

    // ---------------------------------------------------------------------------
    // Multi-scale result tests
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_scale_result_fields() {
        let result = MultiScaleResult {
            scale_results: Vec::new(),
            scale_factors: Vec::new(),
            converged: None,
            convergence_quality: 0.0,
        };
        assert!(result.scale_results.is_empty());
        assert!(result.converged.is_none());
    }

    // ---------------------------------------------------------------------------
    // FieldSmoothingResult and FieldDefect tests
    // ---------------------------------------------------------------------------

    #[test]
    fn field_smoothing_result_creation() {
        let result = FieldSmoothingResult {
            smoothed_field: vec![vec![[1.0f64, 0.0, 0.0]; 2]; 2],
            radius: 1,
            variation_reduction: 0.5,
        };
        assert_eq!(result.radius, 1);
        assert!((result.variation_reduction - 0.5).abs() < 0.001);
    }

    #[test]
    fn field_defect_creation() {
        let defect = FieldDefect {
            position: (3, 5),
            defect_type: DefectType::Meron,
            strength: 0.75,
            charge: 0.5,
        };
        assert_eq!(defect.position, (3, 5));
        assert_eq!(defect.defect_type, DefectType::Meron);
        assert!((defect.strength - 0.75).abs() < 0.001);
        assert!((defect.charge - 0.5).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // End-to-end generalized skyrmion pipeline test
    // ---------------------------------------------------------------------------

    #[test]
    fn generalized_skyrmion_end_to_end_pipeline() {
        // 1. Build a field.
        let mut field = vec![vec![[0.0f64, 0.0, 1.0]; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                field[i][j] = [1.0, 0.0, 0.0];
            }
        }

        // 2. Smooth the field.
        let smoother = FieldSmoother::new(1);
        let smoothed = smoother.smooth(&field);
        assert!(smoothed.variation_reduction >= 0.0);

        // 3. Compute skyrmion number.
        let mut computer = GeneralizedSkyrmionComputer::default();
        let result = computer.compute(&smoothed.smoothed_field).unwrap();
        assert!(result.success);

        // 4. Detect defects.
        let detector = FieldDefectDetector::new();
        let defects = detector.detect(&smoothed.smoothed_field);
        assert!(defects.len() >= 0);

        // 5. Build constellation.
        let mut builder = SkyrmionConstellationBuilder::new(1, 3).unwrap();
        builder.add_point(result.tuple, 0).unwrap();
        assert_eq!(builder.num_points(), 1);

        // 6. Analyze constellation.
        let cap = builder.capacity_analysis();
        assert!(cap.bits_per_symbol >= 0.0);
        let ec = builder.error_correction();
        assert!(ec.strength >= 0.0);

        // 7. Test tuple serialization roundtrip.
        let bytes = result.tuple.to_bytes();
        let restored = SkyrmionTuple::from_bytes(&bytes).unwrap();
        assert_eq!(restored, result.tuple);
    }

    #[test]
    fn generalized_skyrmion_multi_scale_pipeline() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ms = MultiScaleSkyrmionComputer::new(computer, 3, 1.5);
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 8]; 8];
        let result = ms.compute_multi_scale(&field).unwrap();
        assert!(result.convergence_quality >= 0.0);
        assert!(!result.scale_factors.is_empty());
    }

    #[test]
    fn generalized_skyrmion_adaptive_pipeline() {
        let mut computer = GeneralizedSkyrmionComputer::default();
        let mut ac = AdaptiveSkyrmionComputer::new(computer, 4, 16);
        let field = vec![vec![[1.0f64, 0.0, 0.0]; 8]; 8];
        let (result, resolution) = ac.compute_adaptive(&field).unwrap();
        assert!(result.success);
        assert!(resolution >= 4);
    }
}