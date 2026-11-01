//! Intensity-based masking for skyrmion number computation.
//!
//! Implements the intensity-based masking technique from Hara et al. (arXiv:2603.11548)
//! "Exploiting Skyrmions in Free-Space Optical Communication". The masking suppresses
//! low-intensity regions where skyrmion number calculation is unreliable due to noise,
//! improving robustness of topological signal detection under turbulence.
//!
//! Masking strategy:
//! 1. Compute per-pixel intensity S0 = |Ex|² + |Ey|² (first Stokes parameter)
//! 2. Determine threshold as fraction of maximum intensity
//! 3. Mark pixels below threshold as masked (unreliable)
//! 4. Exclude masked pixels from skyrmion number integration
//! 5. Track masked-region statistics for adaptive threshold adjustment

use crate::errors::{XpockError, XpockResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Masking configuration
// ---------------------------------------------------------------------------

/// Configuration for intensity-based masking.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskingConfig {
    /// Threshold as fraction of maximum intensity (0.0 to 1.0).
    /// Pixels with intensity below threshold * max_intensity are masked.
    pub intensity_threshold: f64,
    /// Minimum fraction of pixels that must remain unmasked.
    /// If more is masked, detection returns InsufficientData.
    pub min_valid_fraction: f64,
    /// Enable adaptive threshold adjustment based on noise floor estimation.
    pub enable_adaptive: bool,
    /// Number of histogram bins for adaptive threshold estimation.
    pub histogram_bins: usize,
    /// Minimum absolute intensity threshold (prevents masking everything in dark fields).
    pub min_absolute_threshold: f64,
}

impl Default for MaskingConfig {
    fn default() -> Self {
        Self {
            intensity_threshold: 0.1,
            min_valid_fraction: 0.1,
            enable_adaptive: true,
            histogram_bins: 256,
            min_absolute_threshold: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Masked region — contiguous area of masked pixels
// ---------------------------------------------------------------------------

/// A contiguous region of masked (excluded) pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskedRegion {
    /// Top-left corner (x, y).
    pub origin: (usize, usize),
    /// Region dimensions (width, height).
    pub dimensions: (usize, usize),
    /// Number of masked pixels in this region.
    pub pixel_count: usize,
    /// Average intensity within the region before masking.
    pub avg_intensity: f64,
}

impl MaskedRegion {
    /// Returns the bounding box as ((x_min, y_min), (x_max, y_max)).
    pub fn bounding_box(&self) -> ((usize, usize), (usize, usize)) {
        (
            self.origin,
            (
                self.origin.0 + self.dimensions.0,
                self.origin.1 + self.dimensions.1,
            ),
        )
    }

    /// Returns the fraction of the total field this region occupies.
    pub fn field_fraction(&self, field_width: usize, field_height: usize) -> f64 {
        let total = field_width * field_height;
        if total == 0 {
            return 0.0;
        }
        self.pixel_count as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Masking statistics
// ---------------------------------------------------------------------------

/// Statistics from a masking operation.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskingStats {
    /// Total pixels in the field.
    pub total_pixels: usize,
    /// Number of valid (unmasked) pixels.
    pub valid_pixels: usize,
    /// Number of masked pixels.
    pub masked_pixels: usize,
    /// Fraction of pixels that are valid.
    pub valid_fraction: f64,
    /// Maximum intensity in the field.
    pub max_intensity: f64,
    /// Mean intensity of valid pixels.
    pub mean_valid_intensity: f64,
    /// Applied threshold value.
    pub threshold: f64,
    /// Number of contiguous masked regions detected.
    pub masked_region_count: usize,
    /// Adaptive threshold used (if adaptive enabled).
    pub adaptive_threshold_used: Option<f64>,
}

impl MaskingStats {
    /// Returns the fraction of pixels masked.
    pub fn masked_fraction(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        self.masked_pixels as f64 / self.total_pixels as f64
    }

    /// Returns true if the valid fraction meets the minimum requirement.
    pub fn meets_minimum(&self, min_fraction: f64) -> bool {
        self.valid_fraction >= min_fraction
    }
}

impl Default for MaskingStats {
    fn default() -> Self {
        Self {
            total_pixels: 0,
            valid_pixels: 0,
            masked_pixels: 0,
            valid_fraction: 0.0,
            max_intensity: 0.0,
            mean_valid_intensity: 0.0,
            threshold: 0.0,
            masked_region_count: 0,
            adaptive_threshold_used: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Intensity mask — pixel-wise masking of Stokes field
// ---------------------------------------------------------------------------

/// Pixel-wise intensity mask for a Stokes field.
///
/// Tracks which pixels are valid (unmasked) vs masked (excluded from
/// skyrmion number computation). Supports both fixed-threshold and
/// adaptive thresholding based on noise floor estimation.
pub struct IntensityMask {
    /// Boolean mask: true = valid pixel, false = masked.
    valid: Vec<Vec<bool>>,
    /// Per-pixel intensity values (S0 Stokes parameter).
    intensity: Vec<Vec<f64>>,
    /// Field dimensions.
    width: usize,
    height: usize,
    /// Applied threshold value.
    threshold: f64,
    /// Masking configuration used.
    config: MaskingConfig,
    /// Statistics about the masking operation.
    stats: MaskingStats,
}

impl IntensityMask {
    /// Create a new intensity mask with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self::with_config(MaskingConfig {
            intensity_threshold: threshold,
            ..Default::default()
        })
    }

    /// Create a new intensity mask with full configuration.
    pub fn with_config(config: MaskingConfig) -> Self {
        Self {
            valid: Vec::new(),
            intensity: Vec::new(),
            width: 0,
            height: 0,
            threshold: config.intensity_threshold,
            config,
            stats: MaskingStats::default(),
        }
    }

    /// Apply masking to a Stokes field [S0, S1, S2, S3][y][x].
    ///
    /// Pixels with S0 intensity below threshold * max_intensity are masked.
    /// Returns the mask ready for use in skyrmion number computation.
    pub fn apply(&mut self, stokes_field: &[[[f64; 4]; 256]; 256]) -> &IntensityMask {
        self.width = 256;
        self.height = 256;

        // Step 1: Extract S0 intensities and find max.
        let mut intensities = vec![vec![0.0f64; 256]; 256];
        let mut max_intensity = 0.0f64;

        for y in 0..256 {
            for x in 0..256 {
                let s0 = stokes_field[y][x][0];
                intensities[y][x] = s0.abs();
                if s0.abs() > max_intensity {
                    max_intensity = s0.abs();
                }
            }
        }

        // Step 2: Determine threshold.
        let threshold = if self.config.enable_adaptive && max_intensity > 0.0 {
            self.compute_adaptive_threshold(&intensities, max_intensity)
        } else {
            self.config.intensity_threshold * max_intensity
        };

        let effective_threshold = threshold.max(self.config.min_absolute_threshold);

        // Step 3: Build valid/invalid mask.
        let mut valid = vec![vec![false; 256]; 256];
        let mut valid_pixels = 0usize;
        let mut valid_intensity_sum = 0.0f64;

        for y in 0..256 {
            for x in 0..256 {
                if intensities[y][x] >= effective_threshold {
                    valid[y][x] = true;
                    valid_pixels += 1;
                    valid_intensity_sum += intensities[y][x];
                }
            }
        }

        // Step 4: Detect contiguous masked regions (simple row-run analysis).
        let masked_regions = self.detect_masked_regions(&valid);

        // Step 5: Record statistics.
        let total_pixels = 256 * 256;
        let masked_pixels = total_pixels - valid_pixels;
        let valid_fraction = valid_pixels as f64 / total_pixels as f64;
        let mean_valid_intensity = if valid_pixels > 0 {
            valid_intensity_sum / valid_pixels as f64
        } else {
            0.0
        };

        self.stats = MaskingStats {
            total_pixels,
            valid_pixels,
            masked_pixels,
            valid_fraction,
            max_intensity,
            mean_valid_intensity,
            threshold: effective_threshold,
            masked_region_count: masked_regions.len(),
            adaptive_threshold_used: if self.config.enable_adaptive {
                Some(effective_threshold)
            } else {
                None
            },
        };

        self.valid = valid;
        self.intensity = intensities;
        self.threshold = effective_threshold;

        self
    }

    /// Apply masking to a flat Stokes data buffer with custom dimensions.
    ///
    /// `data` is [S0, S1, S2, S3] in row-major order. `width` and `height`
    /// define field dimensions. Data is resampled to the internal 256x256 grid.
    pub fn apply_flat(
        &mut self,
        data: &[f64],
        width: usize,
        height: usize,
    ) -> Result<&IntensityMask, XpockError> {
        if width == 0 || height == 0 {
            return Err(XpockError::Detection(
                crate::errors::DetectionError::InsufficientStokesMeasurements {
                    have: 0,
                    need: 1,
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

        self.width = width.min(256);
        self.height = height.min(256);

        let scale_x = width as f64 / 256.0;
        let scale_y = height as f64 / 256.0;

        // Resample to 256x256 grid using nearest-neighbor.
        let mut intensities = vec![vec![0.0f64; 256]; 256];
        let mut max_intensity = 0.0f64;

        for y in 0..256 {
            for x in 0..256 {
                let src_x = (x as f64 * scale_x) as usize;
                let src_y = (y as f64 * scale_y) as usize;
                let idx = (src_y * width + src_x) * 4;
                if idx + 3 < data.len() {
                    let s0 = data[idx].abs();
                    intensities[y][x] = s0;
                    if s0 > max_intensity {
                        max_intensity = s0;
                    }
                }
            }
        }

        let threshold = if self.config.enable_adaptive && max_intensity > 0.0 {
            self.compute_adaptive_threshold(&intensities, max_intensity)
        } else {
            self.config.intensity_threshold * max_intensity
        };

        let effective_threshold = threshold.max(self.config.min_absolute_threshold);

        let mut valid = vec![vec![false; 256]; 256];
        let mut valid_pixels = 0usize;
        let mut valid_intensity_sum = 0.0f64;

        for y in 0..256 {
            for x in 0..256 {
                if intensities[y][x] >= effective_threshold {
                    valid[y][x] = true;
                    valid_pixels += 1;
                    valid_intensity_sum += intensities[y][x];
                }
            }
        }

        let masked_regions = self.detect_masked_regions(&valid);

        let total_pixels = 256 * 256;
        let masked_pixels = total_pixels - valid_pixels;
        let valid_fraction = valid_pixels as f64 / total_pixels as f64;
        let mean_valid_intensity = if valid_pixels > 0 {
            valid_intensity_sum / valid_pixels as f64
        } else {
            0.0
        };

        self.stats = MaskingStats {
            total_pixels,
            valid_pixels,
            masked_pixels,
            valid_fraction,
            max_intensity,
            mean_valid_intensity,
            threshold: effective_threshold,
            masked_region_count: masked_regions.len(),
            adaptive_threshold_used: if self.config.enable_adaptive {
                Some(effective_threshold)
            } else {
                None
            },
        };

        self.valid = valid;
        self.intensity = intensities;
        self.threshold = effective_threshold;

        Ok(self)
    }

    /// Check if a pixel at (x, y) is valid (unmasked).
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        if x >= 256 || y >= 256 {
            return false;
        }
        self.valid[y][x]
    }

    /// Get the intensity at pixel (x, y).
    pub fn intensity_at(&self, x: usize, y: usize) -> Option<f64> {
        if x >= 256 || y >= 256 {
            return None;
        }
        Some(self.intensity[y][x])
    }

    /// Returns the ratio of valid (unmasked) pixels.
    pub fn valid_pixel_ratio(&self) -> f64 {
        self.stats.valid_fraction
    }

    /// Returns the mean intensity of valid pixels.
    pub fn mean_intensity(&self) -> f64 {
        self.stats.mean_valid_intensity
    }

    /// Returns the maximum intensity in the field.
    pub fn max_intensity(&self) -> f64 {
        self.stats.max_intensity
    }

    /// Returns the applied threshold value.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns the detected masked regions.
    pub fn masked_regions(&self) -> Vec<MaskedRegion> {
        self.detect_masked_regions(&self.valid)
    }

    /// Returns the masking statistics.
    pub fn stats(&self) -> &MaskingStats {
        &self.stats
    }

    /// Returns true if enough pixels are valid for reliable detection.
    pub fn sufficient_valid_pixels(&self) -> bool {
        self.stats.meets_minimum(self.config.min_valid_fraction)
    }

    /// Compute adaptive threshold using histogram-based noise floor estimation.
    ///
    /// Uses Otsu-like thresholding adapted for intensity masking: finds the
    /// intensity value that best separates signal pixels from noise floor.
    fn compute_adaptive_threshold(&self, intensities: &[Vec<f64>], max_intensity: f64) -> f64 {
        if max_intensity <= 0.0 {
            return 0.0;
        }

        // Build histogram of intensities.
        let bins = self.config.histogram_bins;
        let mut histogram = vec![0usize; bins];
        let bin_width = max_intensity / bins as f64;

        for row in intensities {
            for &val in row {
                let bin = ((val / max_intensity) * bins as f64).floor() as usize;
                let bin = bin.min(bins - 1);
                histogram[bin] += 1;
            }
        }

        let total_pixels: usize = histogram.iter().sum();
        if total_pixels == 0 {
            return self.config.intensity_threshold * max_intensity;
        }

        // Otsu's method: find threshold maximizing between-class variance.
        let mut best_threshold_bin = 0;
        let mut max_variance = 0.0f64;

        let total_sum: f64 = histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| i as f64 * count as f64)
            .sum();

        let mut w0 = 0usize;
        let mut sum0 = 0.0f64;

        for t in 0..bins {
            w0 += histogram[t];
            if w0 == 0 {
                continue;
            }
            let w1 = total_pixels - w0;
            if w1 == 0 {
                break;
            }

            sum0 += t as f64 * histogram[t] as f64;
            let mean0 = sum0 / w0 as f64;
            let mean1 = (total_sum - sum0) / w1 as f64;

            let between_variance = w0 as f64 * w1 as f64 * (mean0 - mean1).powi(2);

            if between_variance > max_variance {
                max_variance = between_variance;
                best_threshold_bin = t;
            }
        }

        // Convert bin back to intensity, then apply configured threshold fraction.
        let otsu_intensity = (best_threshold_bin as f64 + 0.5) * bin_width;

        // Blend Otsu threshold with configured fraction of max intensity.
        let configured = self.config.intensity_threshold * max_intensity;
        let adaptive = (otsu_intensity + configured) / 2.0;

        adaptive.max(self.config.min_absolute_threshold)
    }

    /// Detect contiguous masked regions using simple row-run analysis.
    ///
    /// Returns regions of consecutive masked pixels. This is a lightweight
    /// approximation suitable for statistics; exact connected-component
    /// labeling can be added if needed.
    fn detect_masked_regions(&self, valid: &[Vec<bool>]) -> Vec<MaskedRegion> {
        let mut regions = Vec::new();

        for y in 0..256 {
            let mut x = 0;
            while x < 256 {
                if !valid[y][x] {
                    // Start of a masked run in this row.
                    let run_start = x;
                    let mut run_end = x;
                    while run_end + 1 < 256 && !valid[y][run_end + 1] {
                        run_end += 1;
                    }

                    // Compute average intensity of this region.
                    let mut intensity_sum = 0.0f64;
                    let mut count = 0usize;
                    for cx in run_start..=run_end {
                        intensity_sum += self.intensity[y][cx];
                        count += 1;
                    }
                    let avg_intensity = if count > 0 {
                        intensity_sum / count as f64
                    } else {
                        0.0
                    };

                    regions.push(MaskedRegion {
                        origin: (run_start, y),
                        dimensions: (run_end - run_start + 1, 1),
                        pixel_count: count,
                        avg_intensity,
                    });

                    x = run_end + 1;
                } else {
                    x += 1;
                }
            }
        }

        regions
    }

    /// Merge adjacent row runs into larger regions (optional refinement).
    pub fn refine_regions(&mut self) {
        // Simple vertical merging: if regions in adjacent rows overlap in x,
        // merge them into a single region.
        let mut regions = self.detect_masked_regions(&self.valid);
        if regions.is_empty() {
            return;
        }

        regions.sort_by_key(|r| (r.origin.1, r.origin.0));

        let mut merged: Vec<MaskedRegion> = Vec::new();
        let mut current = regions[0].clone();

        for region in regions.iter().skip(1) {
            let same_row_or_adjacent =
                (region.origin.1 as i64 - current.origin.1 as i64).abs() <= 1;
            let x_overlap = region.origin.0 <= current.origin.0 + current.dimensions.0
                && current.origin.0 <= region.origin.0 + region.dimensions.0;

            if same_row_or_adjacent && x_overlap {
                // Merge regions.
                let new_x_min = current.origin.0.min(region.origin.0);
                let new_x_max = (current.origin.0 + current.dimensions.0)
                    .max(region.origin.0 + region.dimensions.0);
                let new_y_max = current.origin.1.max(region.origin.1) + 1;
                current = MaskedRegion {
                    origin: (new_x_min, current.origin.1),
                    dimensions: (new_x_max - new_x_min, new_y_max - current.origin.1),
                    pixel_count: current.pixel_count + region.pixel_count,
                    avg_intensity: (current.avg_intensity * current.pixel_count as f64
                        + region.avg_intensity * region.pixel_count as f64)
                        / (current.pixel_count + region.pixel_count) as f64,
                };
            } else {
                merged.push(current.clone());
                current = region.clone();
            }
        }
        merged.push(current);

        // Update stats with refined region count.
        self.stats.masked_region_count = merged.len();
    }
}

// ---------------------------------------------------------------------------
// Multi-scale thresholding
// ---------------------------------------------------------------------------

/// Multi-scale adaptive thresholding for non-uniform illumination fields.
///
/// Divides the field into overlapping tiles, computes a local threshold for
/// each tile based on its intensity distribution, and blends thresholds
/// across tile boundaries to avoid blocking artifacts.
pub struct MultiScaleThreshold {
    /// Tile size in pixels (square tiles).
    pub tile_size: usize,
    /// Overlap between adjacent tiles in pixels.
    pub overlap: usize,
    /// Threshold fraction applied to each tile's local maximum.
    pub local_fraction: f64,
    /// Blend weight for boundary smoothing (0.0 = no blending, 1.0 = full).
    pub blend_weight: f64,
}

impl Default for MultiScaleThreshold {
    fn default() -> Self {
        Self {
            tile_size: 64,
            overlap: 16,
            local_fraction: 0.2,
            blend_weight: 0.5,
        }
    }
}

impl MultiScaleThreshold {
    /// Compute a per-pixel threshold map using multi-scale local thresholding.
    ///
    /// Returns a grid of threshold values matching the input intensity dimensions.
    pub fn compute_threshold_map(&self, intensities: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let height = intensities.len();
        if height == 0 {
            return Vec::new();
        }
        let width = intensities[0].len();
        let mut threshold_map = vec![vec![0.0f64; width]; height];
        let mut weight_map = vec![vec![0.0f64; width]; height];

        let tile = self.tile_size;
        let overlap = self.overlap;
        let fraction = self.local_fraction;

        let mut ty = 0usize;
        while ty < height {
            let mut tx = 0usize;
            while tx < width {
                let ty_end = (ty + tile).min(height);
                let tx_end = (tx + tile).min(width);

                // Compute local max and mean within this tile.
                let mut local_max = 0.0f64;
                let mut local_sum = 0.0f64;
                let mut count = 0usize;
                for y in ty..ty_end {
                    for x in tx..tx_end {
                        let v = intensities[y][x];
                        if v > local_max {
                            local_max = v;
                        }
                        local_sum += v;
                        count += 1;
                    }
                }
                let local_mean = if count > 0 {
                    local_sum / count as f64
                } else {
                    0.0
                };
                let local_threshold = (local_max * fraction).max(local_mean * 0.5);

                // Apply this threshold to the tile region with blending.
                let blend = self.blend_weight;
                for y in ty..ty_end {
                    for x in tx..tx_end {
                        let dx =
                            (x as i64 - tx as i64).abs().min((tx_end as i64 - x as i64 - 1).abs())
                                as f64;
                        let dy =
                            (y as i64 - ty as i64).abs().min((ty_end as i64 - y as i64 - 1).abs())
                                as f64;
                        let edge_dist = dx.min(dy);
                        let w = (edge_dist / overlap as f64).clamp(0.0, 1.0);

                        let existing = threshold_map[y][x];
                        let existing_w = weight_map[y][x];
                        if existing_w == 0.0 {
                            threshold_map[y][x] = local_threshold;
                            weight_map[y][x] = 1.0;
                        } else {
                            let total_w = existing_w + w;
                            threshold_map[y][x] =
                                (existing * existing_w + local_threshold * w) / total_w;
                            weight_map[y][x] = total_w;
                        }
                    }
                }

                tx = tx_end.saturating_sub(overlap);
                if tx_end >= width {
                    break;
                }
            }
            ty = ty_end.saturating_sub(overlap);
            if ty_end >= height {
                break;
            }
        }

        // Fill any uncovered pixels with global median threshold.
        let global = Self::global_threshold(intensities);
        for y in 0..height {
            for x in 0..width {
                if weight_map[y][x] == 0.0 {
                    threshold_map[y][x] = global;
                }
            }
        }

        threshold_map
    }

    /// Global fallback threshold — median of all intensities.
    fn global_threshold(intensities: &[Vec<f64>]) -> f64 {
        let mut values: Vec<f64> = intensities
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Apply multi-scale thresholding to produce a binary mask.
    pub fn apply_mask(&self, intensities: &[Vec<f64>]) -> Vec<Vec<bool>> {
        let threshold_map = self.compute_threshold_map(intensities);
        let height = intensities.len();
        if height == 0 {
            return Vec::new();
        }
        let width = intensities[0].len();
        let mut mask = vec![vec![false; width]; height];
        for y in 0..height {
            for x in 0..width {
                mask[y][x] = intensities[y][x] >= threshold_map[y][x];
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// Connected component labeling (8-connectivity, union-find)
// ---------------------------------------------------------------------------

/// A labeled connected component in a binary mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledComponent {
    /// Component label (unique ID).
    pub label: u32,
    /// Pixels belonging to this component.
    pub pixels: Vec<(usize, usize)>,
    /// Bounding box ((x_min, y_min), (x_max, y_max)).
    pub bounding_box: ((usize, usize), (usize, usize)),
    /// Component area (pixel count).
    pub area: usize,
    /// Average intensity within the component.
    pub mean_intensity: f64,
    /// Centroid (x, y) in pixel coordinates.
    pub centroid: (f64, f64),
}

impl LabeledComponent {
    /// Returns the component's width.
    pub fn width(&self) -> usize {
        self.bounding_box.1 .0 - self.bounding_box.0 .0
    }

    /// Returns the component's height.
    pub fn height(&self) -> usize {
        self.bounding_box.1 .1 - self.bounding_box.0 .1
    }

    /// Returns the component's aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f64 {
        let h = self.height();
        if h == 0 {
            return 0.0;
        }
        self.width() as f64 / h as f64
    }

    /// Returns the component's solidity (area / bounding box area).
    pub fn solidity(&self) -> f64 {
        let w = self.width();
        let h = self.height();
        if w == 0 || h == 0 {
            return 0.0;
        }
        self.area as f64 / (w * h) as f64
    }
}

/// Union-find data structure for connected component labeling.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

/// Connected component labeling with 8-connectivity.
pub struct ConnectedComponentLabeler {
    /// Minimum component size in pixels (smaller components are filtered).
    pub min_size: usize,
    /// Maximum number of components to return (sorted by area descending).
    pub max_components: Option<usize>,
}

impl Default for ConnectedComponentLabeler {
    fn default() -> Self {
        Self {
            min_size: 4,
            max_components: None,
        }
    }
}

impl ConnectedComponentLabeler {
    /// Label all connected components in a binary mask.
    ///
    /// Uses 8-connectivity (diagonal neighbors are connected).
    /// Returns components sorted by area descending.
    pub fn label(&self, mask: &[Vec<bool>]) -> Vec<LabeledComponent> {
        let height = mask.len();
        if height == 0 {
            return Vec::new();
        }
        let width = mask[0].len();

        // Assign a unique index to each foreground pixel.
        let mut pixel_index = vec![vec![None; width]; height];
        let mut foreground_pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if mask[y][x] {
                    pixel_index[y][x] = Some(foreground_pixels.len());
                    foreground_pixels.push((x, y));
                }
            }
        }

        let n = foreground_pixels.len();
        if n == 0 {
            return Vec::new();
        }

        let mut uf = UnionFind::new(n);

        // Union 8-connected neighbors.
        for (idx, &(x, y)) in foreground_pixels.iter().enumerate() {
            for dy in 0..=1i64 {
                for dx in 0..=1i64 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || nx >= width as i64 || ny < 0 || ny >= height as i64 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if mask[ny][nx] {
                        let neighbor_idx = pixel_index[ny][nx].unwrap();
                        uf.union(idx, neighbor_idx);
                    }
                }
            }
        }

        // Group pixels by root.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            groups.entry(root).or_default().push(i);
        }

        // Build components.
        let mut components: Vec<LabeledComponent> = groups
            .into_iter()
            .enumerate()
            .filter_map(|(label, member_indices)| {
                if member_indices.len() < self.min_size {
                    return None;
                }
                let mut pixels: Vec<(usize, usize)> =
                    member_indices.iter().map(|&i| foreground_pixels[i]).collect();
                pixels.sort();

                let x_min = pixels.iter().map(|p| p.0).min().unwrap_or(0);
                let x_max = pixels.iter().map(|p| p.0).max().unwrap_or(0);
                let y_min = pixels.iter().map(|p| p.1).min().unwrap_or(0);
                let y_max = pixels.iter().map(|p| p.1).max().unwrap_or(0);

                let area = pixels.len();
                let centroid_x =
                    pixels.iter().map(|p| p.0 as f64).sum::<f64>() / area as f64;
                let centroid_y =
                    pixels.iter().map(|p| p.1 as f64).sum::<f64>() / area as f64;

                Some(LabeledComponent {
                    label: label as u32,
                    pixels,
                    bounding_box: ((x_min, y_min), (x_max + 1, y_max + 1)),
                    area,
                    mean_intensity: 0.0, // filled below if intensities available
                    centroid: (centroid_x, centroid_y),
                })
            })
            .collect();

        // Sort by area descending.
        components.sort_by(|a, b| b.area.cmp(&a.area));

        // Apply max_components limit.
        if let Some(max) = self.max_components {
            components.truncate(max);
        }

        components
    }

    /// Label components and compute mean intensity for each using an intensity field.
    pub fn label_with_intensity(
        &self,
        mask: &[Vec<bool>],
        intensities: &[Vec<f64>],
    ) -> Vec<LabeledComponent> {
        let mut components = self.label(mask);
        for comp in components.iter_mut() {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for &(x, y) in comp.pixels.iter() {
                if y < intensities.len() && x < intensities[y].len() {
                    sum += intensities[y][x];
                    count += 1;
                }
            }
            comp.mean_intensity = if count > 0 {
                sum / count as f64
            } else {
                0.0
            };
        }
        components
    }
}

// ---------------------------------------------------------------------------
// Morphological operations on binary masks
// ---------------------------------------------------------------------------

/// Morphological operations for binary mask processing.
pub struct MorphologicalOps;

impl MorphologicalOps {
    /// Dilate a binary mask by a square structuring element of the given radius.
    ///
    /// Expands foreground regions outward by `radius` pixels.
    pub fn dilate(mask: &[Vec<bool>], radius: usize) -> Vec<Vec<bool>> {
        let height = mask.len();
        if height == 0 {
            return Vec::new();
        }
        let width = mask[0].len();
        let mut result = vec![vec![false; width]; height];

        for y in 0..height {
            for x in 0..width {
                if !mask[y][x] {
                    continue;
                }
                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius).min(height - 1) + 1;
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius).min(width - 1) + 1;
                for yy in y_start..y_end {
                    for xx in x_start..x_end {
                        result[yy][xx] = true;
                    }
                }
            }
        }
        result
    }

    /// Erode a binary mask by a square structuring element of the given radius.
    ///
    /// Shrinks foreground regions; a pixel remains foreground only if all
    /// pixels within `radius` are foreground.
    pub fn erode(mask: &[Vec<bool>], radius: usize) -> Vec<Vec<bool>> {
        let height = mask.len();
        if height == 0 {
            return Vec::new();
        }
        let width = mask[0].len();
        let mut result = vec![vec![false; width]; height];

        for y in 0..height {
            for x in 0..width {
                if !mask[y][x] {
                    continue;
                }
                let mut all_foreground = true;
                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius).min(height - 1) + 1;
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius).min(width - 1) + 1;
                for yy in y_start..y_end {
                    for xx in x_start..x_end {
                        if !mask[yy][xx] {
                            all_foreground = false;
                            break;
                        }
                    }
                    if !all_foreground {
                        break;
                    }
                }
                if all_foreground {
                    result[y][x] = true;
                }
            }
        }
        result
    }

    /// Opening: erosion followed by dilation. Removes small noise.
    pub fn open(mask: &[Vec<bool>], radius: usize) -> Vec<Vec<bool>> {
        let eroded = Self::erode(mask, radius);
        Self::dilate(&eroded, radius)
    }

    /// Closing: dilation followed by erosion. Fills small holes.
    pub fn close(mask: &[Vec<bool>], radius: usize) -> Vec<Vec<bool>> {
        let dilated = Self::dilate(mask, radius);
        Self::erode(&dilated, radius)
    }
}

// ---------------------------------------------------------------------------
// Mask quality metrics
// ---------------------------------------------------------------------------

/// Quality metrics for evaluating a binary mask.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskQualityMetrics {
    /// Fraction of pixels classified as foreground (valid).
    pub foreground_fraction: f64,
    /// Number of connected components.
    pub component_count: usize,
    /// Largest component area.
    pub largest_component_area: usize,
    /// Mean component area.
    pub mean_component_area: f64,
    /// Component area standard deviation.
    pub component_area_std: f64,
    /// Mask entropy (Shannon entropy of foreground/background distribution).
    pub entropy: f64,
    /// Spatial uniformity (1.0 = perfectly uniform, 0.0 = highly clustered).
    pub spatial_uniformity: f64,
    /// Boundary complexity (perimeter / area ratio of foreground).
    pub boundary_complexity: f64,
}

impl MaskQualityMetrics {
    /// Returns an overall quality score (0.0 = poor, 1.0 = excellent).
    pub fn quality_score(&self) -> f64 {
        // Balanced score: good foreground fraction, low fragmentation,
        // high spatial uniformity, reasonable boundary complexity.
        let fg_score = if self.foreground_fraction > 0.05 && self.foreground_fraction < 0.95 {
            1.0 - (self.foreground_fraction - 0.5).abs() * 2.0
        } else {
            0.2
        };
        let fragmentation_score = if self.component_count > 0 {
            1.0 / (1.0
                + self.component_area_std / self.mean_component_area.max(1.0))
        } else {
            1.0
        };
        (fg_score * 0.4 + fragmentation_score * 0.3 + self.spatial_uniformity * 0.3)
            .clamp(0.0, 1.0)
    }
}

/// Computes quality metrics for a binary mask.
pub fn compute_mask_quality(
    mask: &[Vec<bool>],
    _intensities: Option<&[Vec<f64>]>,
) -> MaskQualityMetrics {
    let height = mask.len();
    let width = if height > 0 { mask[0].len() } else { 0 };
    let total = width * height;

    if total == 0 {
        return MaskQualityMetrics {
            foreground_fraction: 0.0,
            component_count: 0,
            largest_component_area: 0,
            mean_component_area: 0.0,
            component_area_std: 0.0,
            entropy: 0.0,
            spatial_uniformity: 0.0,
            boundary_complexity: 0.0,
        };
    }

    let mut foreground = 0usize;
    for row in mask.iter() {
        for &v in row.iter() {
            if v {
                foreground += 1;
            }
        }
    }
    let foreground_fraction = foreground as f64 / total as f64;

    // Entropy of foreground/background distribution.
    let p_fg = foreground_fraction;
    let p_bg = 1.0 - p_fg;
    let entropy = if p_fg > 0.0 && p_bg > 0.0 {
        -(p_fg * p_fg.ln() + p_bg * p_bg.ln())
    } else {
        0.0
    };

    // Connected components.
    let labeler = ConnectedComponentLabeler::default();
    let components = labeler.label(mask);
    let component_count = components.len();
    let largest_component_area = components.iter().map(|c| c.area).max().unwrap_or(0);
    let mean_component_area = if component_count > 0 {
        components.iter().map(|c| c.area).sum::<usize>() as f64 / component_count as f64
    } else {
        0.0
    };
    let component_area_std = if component_count > 1 {
        let mean = mean_component_area;
        let var = components
            .iter()
            .map(|c| (c.area as f64 - mean).powi(2))
            .sum::<f64>()
            / (component_count - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    // Spatial uniformity: measure clustering via nearest-neighbor distance variance.
    let spatial_uniformity = if component_count > 1 {
        let centroids: Vec<(f64, f64)> = components.iter().map(|c| c.centroid).collect();
        let mut distances = Vec::new();
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let dx = centroids[i].0 - centroids[j].0;
                let dy = centroids[i].1 - centroids[j].1;
                distances.push((dx * dx + dy * dy).sqrt());
            }
        }
        if distances.len() > 1 {
            let mean_d = distances.iter().sum::<f64>() / distances.len() as f64;
            let var_d = distances
                .iter()
                .map(|d| (d - mean_d).powi(2))
                .sum::<f64>()
                / (distances.len() - 1) as f64;
            let cv = if mean_d > 0.0 { var_d.sqrt() / mean_d } else { 0.0 };
            (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Boundary complexity: approximate perimeter / area.
    let boundary_complexity = if foreground > 0 {
        let mut perimeter = 0usize;
        for y in 0..height {
            for x in 0..width {
                if !mask[y][x] {
                    continue;
                }
                // Count edges adjacent to background or image boundary.
                if x == 0 || !mask[y][x - 1] {
                    perimeter += 1;
                }
                if x + 1 >= width || !mask[y][x + 1] {
                    perimeter += 1;
                }
                if y == 0 || !mask[y - 1][x] {
                    perimeter += 1;
                }
                if y + 1 >= height || !mask[y + 1][x] {
                    perimeter += 1;
                }
            }
        }
        perimeter as f64 / foreground as f64
    } else {
        0.0
    };

    MaskQualityMetrics {
        foreground_fraction,
        component_count,
        largest_component_area,
        mean_component_area,
        component_area_std,
        entropy,
        spatial_uniformity,
        boundary_complexity,
    }
}

// ---------------------------------------------------------------------------
// Mask set operations (AND, OR, XOR, NOT)
// ---------------------------------------------------------------------------

/// Combines two binary masks using a boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// Logical AND — intersection of foreground regions.
    And,
    /// Logical OR — union of foreground regions.
    Or,
    /// Logical XOR — symmetric difference.
    Xor,
    /// Logical NOT — invert the first mask (second ignored).
    Not,
}

/// Apply a boolean operation between two masks of the same dimensions.
pub fn combine_masks(
    a: &[Vec<bool>],
    b: &[Vec<bool>],
    op: MaskOp,
) -> Vec<Vec<bool>> {
    let height = a.len();
    if height == 0 {
        return Vec::new();
    }
    let width = a[0].len();
    let mut result = vec![vec![false; width]; height];

    for y in 0..height {
        for x in 0..width {
            let av = if y < b.len() && x < b[y].len() { b[y][x] } else { false };
            let bv = a[y][x];
            result[y][x] = match op {
                MaskOp::And => av && bv,
                MaskOp::Or => av || bv,
                MaskOp::Xor => av ^ bv,
                MaskOp::Not => !av,
            };
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Regional intensity statistics
// ---------------------------------------------------------------------------

/// Statistics computed over a specific region of an intensity field.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionalStatistics {
    /// Region label.
    pub label: u32,
    /// Pixel count.
    pub count: usize,
    /// Mean intensity.
    pub mean: f64,
    /// Variance.
    pub variance: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum intensity.
    pub min: f64,
    /// Maximum intensity.
    pub max: f64,
    /// Median intensity.
    pub median: f64,
}

impl RegionalStatistics {
    /// Returns the coefficient of variation (std / mean).
    pub fn cv(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            return 0.0;
        }
        self.std_dev / self.mean.abs()
    }
}

/// Compute statistics for each labeled component over an intensity field.
pub fn compute_regional_statistics(
    components: &[LabeledComponent],
    intensities: &[Vec<f64>],
) -> Vec<RegionalStatistics> {
    let mut stats = Vec::with_capacity(components.len());
    for comp in components.iter() {
        let mut values: Vec<f64> = Vec::with_capacity(comp.pixels.len());
        for &(x, y) in comp.pixels.iter() {
            if y < intensities.len() && x < intensities[y].len() {
                values.push(intensities[y][x]);
            }
        }
        if values.is_empty() {
            stats.push(RegionalStatistics {
                label: comp.label,
                count: 0,
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
            });
            continue;
        }
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        stats.push(RegionalStatistics {
            label: comp.label,
            count,
            mean,
            variance,
            std_dev,
            min,
            max,
            median,
        });
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masking_config_default_is_valid() {
        let config = MaskingConfig::default();
        assert_eq!(config.intensity_threshold, 0.1);
        assert_eq!(config.min_valid_fraction, 0.1);
        assert!(config.enable_adaptive);
        assert_eq!(config.histogram_bins, 256);
    }

    #[test]
    fn mask_uniform_field_keeps_all_pixels() {
        let mut mask = IntensityMask::new(0.1);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        mask.apply(&field);
        assert!(mask.sufficient_valid_pixels());
        assert_eq!(mask.valid_pixel_ratio(), 1.0);
    }

    #[test]
    fn mask_dark_field_masks_everything() {
        let mut mask = IntensityMask::new(0.5);
        let field = [[[0.0f64; 4]; 256]; 256];
        mask.apply(&field);
        // All zero intensity, all masked (threshold = 0, min_absolute = 1e-6)
        assert!(!mask.sufficient_valid_pixels());
    }

    #[test]
    fn mask_gradient_field_partial_masking() {
        let mut mask = IntensityMask::new(0.5);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [x as f64 / 255.0, 0.0, 0.0, 0.0];
            }
        }
        mask.apply(&field);
        // Threshold = 0.5 * max = 0.5. Pixels with x/255 >= 0.5 are valid (~half).
        let ratio = mask.valid_pixel_ratio();
        assert!(ratio > 0.3 && ratio < 0.8);
    }

    #[test]
    fn mask_stats_consistent() {
        let mut mask = IntensityMask::new(0.1);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        mask.apply(&field);
        let stats = mask.stats();
        assert_eq!(stats.total_pixels, 65536);
        assert_eq!(stats.valid_pixels, 65536);
        assert_eq!(stats.masked_pixels, 0);
        assert_eq!(stats.valid_fraction, 1.0);
        assert_eq!(stats.max_intensity, 1.0);
    }

    #[test]
    fn is_valid_out_of_bounds_returns_false() {
        let mask = IntensityMask::new(0.1);
        assert!(!mask.is_valid(256, 100));
        assert!(!mask.is_valid(100, 256));
    }

    #[test]
    fn intensity_at_out_of_bounds_returns_none() {
        let mask = IntensityMask::new(0.1);
        assert!(mask.intensity_at(256, 100).is_none());
    }

    #[test]
    fn adaptive_threshold_enabled_by_default() {
        let config = MaskingConfig::default();
        assert!(config.enable_adaptive);
    }

    #[test]
    fn min_valid_fraction_check() {
        let mut mask = IntensityMask::new(0.9);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [0.5, 0.0, 0.0, 0.0];
            }
        }
        mask.apply(&field);
        // threshold = 0.9 * 0.5 = 0.45, all pixels have 0.5 >= 0.45, all valid
        assert!(mask.sufficient_valid_pixels());
    }

    #[test]
    fn masked_regions_empty_for_uniform_field() {
        let mut mask = IntensityMask::new(0.1);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        mask.apply(&field);
        assert!(mask.masked_regions().is_empty());
    }

    #[test]
    fn apply_flat_rejects_empty() {
        let mut mask = IntensityMask::new(0.1);
        let result = mask.apply_flat(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn apply_flat_rejects_insufficient() {
        let mut mask = IntensityMask::new(0.1);
        let result = mask.apply_flat(&[1.0, 0.0, 0.0, 0.0], 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn refine_regions_runs_without_panic() {
        let mut mask = IntensityMask::new(0.1);
        let mut field = [[[0.0f64; 4]; 256]; 256];
        for y in 0..256 {
            for x in 0..256 {
                field[y][x] = [1.0, 0.0, 0.0, 0.0];
            }
        }
        mask.apply(&field);
        mask.refine_regions(); // Should not panic
    }

    #[test]
    fn masked_region_bounding_box() {
        let region = MaskedRegion {
            origin: (10, 20),
            dimensions: (30, 40),
            pixel_count: 1200,
            avg_intensity: 0.5,
        };
        let ((x_min, y_min), (x_max, y_max)) = region.bounding_box();
        assert_eq!((x_min, y_min), (10, 20));
        assert_eq!((x_max, y_max), (40, 60));
    }

    #[test]
    fn masked_region_field_fraction() {
        let region = MaskedRegion {
            origin: (0, 0),
            dimensions: (128, 128),
            pixel_count: 16384,
            avg_intensity: 0.3,
        };
        let fraction = region.field_fraction(256, 256);
        assert!((fraction - 0.25).abs() < 0.01);
    }

    #[test]
    fn masking_stats_masked_fraction() {
        let stats = MaskingStats {
            total_pixels: 100,
            valid_pixels: 70,
            masked_pixels: 30,
            valid_fraction: 0.7,
            max_intensity: 1.0,
            mean_valid_intensity: 0.8,
            threshold: 0.1,
            masked_region_count: 3,
            adaptive_threshold_used: Some(0.15),
        };
        assert!((stats.masked_fraction() - 0.3).abs() < 0.001);
        assert!(stats.meets_minimum(0.5));
        assert!(!stats.meets_minimum(0.8));
    }

    #[test]
    fn default_masking_config_works() {
        let config = MaskingConfig::default();
        let _mask = IntensityMask::with_config(config);
    }
}