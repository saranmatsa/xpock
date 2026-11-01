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

impl IntensityMask {
    /// Create a new intensity mask with the given configuration.
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
    pub fn apply(
        &mut self,
        stokes_field: &[[[f64; 4]; 256]; 256],
    ) -> &IntensityMask {
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

        // Step 4: Detect contiguous masked regions (simple flood-fill lite — row runs).
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
        let mut visited = vec![vec![false; 256]; 256];

        for y in 0..256 {
            let mut x = 0;
            while x < 256 {
                if !valid[y][x] && !visited[y][x] {
                    // Start of a masked run in this row.
                    let run_start = x;
                    let mut run_end = x;
                    while run_end + 1 < 256 && !valid[y][run_end + 1] {
                        run_end += 1;
                    }

                    // Mark this run as visited.
                    for cx in run_start..=run_end {
                        visited[y][cx] = true;
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
            let same_row_or_adjacent = (region.origin.1 as i64 - current.origin.1 as i64).abs() <= 1;
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