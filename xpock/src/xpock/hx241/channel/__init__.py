"""
XPOCK — HX-241 Channel Capacity & Information Analysis

Information-theoretic analysis of wave structures for HX-241 communication.
Computes channel capacity, mutual information, and distinguishability
between wave structures using the ENGINES information theory engine.

These analyses answer the core HX-241 question: which wave structures
can reliably carry distinguishable information across physical carriers?

All computations are mathematical models — no claims about experimental
results. Designed to integrate with HX-241's carrier-agnostic architecture
without depending on any conventional networking stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum
from pathlib import Path
import numpy as np

from xpock.core import WaveEvent, WaveField
from xpock.analysis import WaveAnalyzer, MeasurableProperties
from xpock.engines_integration import (
    WaveInformationEngine,
    EngineResult,
)


class DistinguishabilityMetric(Enum):
    """Metrics for comparing wave structure distinguishability."""

    JS_DIVERGENCE = "js_divergence"       # Jensen-Shannon divergence (bounded [0,1])
    KL_DIVERGENCE = "kl_divergence"        # Kullback-Leibler divergence
    MUTUAL_INFORMATION = "mutual_information"  # I(X;Y) between parameter and measurement
    EUCLIDEAN_DISTANCE = "euclidean_distance"  # Feature space distance
    COSINE_SIMILARITY = "cosine_similarity"    # Cosine similarity of feature vectors


@dataclass
class WaveStructureSignature:
    """Feature signature of a wave structure for distinguishability analysis.

    Captures measurable properties that a receiver could potentially
    use to distinguish one wave structure from another.
    """

    event_id: str
    parameters: Dict[str, Any]

    # Temporal features
    duration_rms: Optional[float] = None
    duration_fwhm: Optional[float] = None
    bandwidth_rms: Optional[float] = None
    temporal_center: Optional[float] = None

    # Spatial features
    beam_waist_x: Optional[float] = None
    beam_waist_y: Optional[float] = None
    beam_waist_r: Optional[float] = None

    # Spectral features
    spectral_centroid: Optional[float] = None
    spectral_spread: Optional[float] = None

    # Topological features
    oam_per_photon: Optional[float] = None
    winding_number: Optional[float] = None

    # Polarization features
    degree_of_polarization: Optional[float] = None
    stokes_s1: Optional[float] = None
    stokes_s2: Optional[float] = None
    stokes_s3: Optional[float] = None

    # Nonseparability
    schmidt_number: Optional[float] = None

    # Power
    total_power: float = 0.0
    peak_intensity: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to a numeric feature vector for comparison.

        Missing values are filled with 0. All values are not normalized
        here — normalization should be applied before distance computation.
        """
        features = [
            self.duration_rms or 0.0,
            self.duration_fwhm or 0.0,
            self.bandwidth_rms or 0.0,
            self.beam_waist_x or 0.0,
            self.beam_waist_y or 0.0,
            self.beam_waist_r or 0.0,
            self.spectral_centroid or 0.0,
            self.spectral_spread or 0.0,
            self.oam_per_photon or 0.0,
            self.winding_number or 0.0,
            self.degree_of_polarization or 0.0,
            self.stokes_s1 or 0.0,
            self.stokes_s2 or 0.0,
            self.stokes_s3 or 0.0,
            self.schmidt_number or 1.0,
            self.total_power,
            self.peak_intensity,
        ]
        return np.array(features, dtype=np.float64)

    def to_probability_distribution(self, n_bins: int = 64) -> np.ndarray:
        """Convert feature vector to a probability distribution.

        Uses histogram binning of feature magnitudes. Useful for
        JS/KL divergence computation between wave structures.
        """
        vec = self.to_feature_vector()
        # Use absolute values for histogram
        abs_vec = np.abs(vec)
        # Add small epsilon to avoid zeros
        abs_vec = abs_vec + 1e-12
        # Normalize to sum to 1
        total = np.sum(abs_vec)
        if total > 0:
            return abs_vec / total
        return np.ones(n_bins) / n_bins


@dataclass
class CapacityAnalysisResult:
    """Result of channel capacity analysis."""

    config_name: str
    snr_db: float
    bandwidth_hz: float
    shannon_capacity_bps: float
    spectral_efficiency_bps_hz: float  # spectral efficiency (alias for shannon_capacity_bps_hz)
    usable_capacity_bps: float      # capacity after error correction overhead
    distinguishable_states: int     # number of distinguishable wave states
    bits_per_symbol: float          # log2(distinguishable_states)
    symbols_per_second: float       # baud rate estimate
    net_information_rate_bps: float # usable_capacity * coding_efficiency

    # Distinguishability breakdown
    min_js_divergence: float        # min JS divergence between any pair of states
    avg_js_divergence: float
    feature_correlation: np.ndarray  # correlation matrix of features

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "snr_db": self.snr_db,
            "bandwidth_hz": self.bandwidth_hz,
            "shannon_capacity_bps": self.shannon_capacity_bps,
            "spectral_efficiency_bps_hz": self.spectral_efficiency_bps_hz,
            "usable_capacity_bps": self.usable_capacity_bps,
            "distinguishable_states": self.distinguishable_states,
            "bits_per_symbol": self.bits_per_symbol,
            "symbols_per_second": self.symbols_per_second,
            "net_information_rate_bps": self.net_information_rate_bps,
            "min_js_divergence": self.min_js_divergence,
            "avg_js_divergence": self.avg_js_divergence,
            "notes": self.notes,
        }


@dataclass
class DistinguishabilityAnalysisResult:
    """Result of pairwise wave structure distinguishability analysis."""

    n_structures: int
    metric: DistinguishabilityMetric
    pairwise_matrix: np.ndarray      # (n, n) matrix of distances/divergences
    min_distance: float
    max_distance: float
    mean_distance: float
    most_confusable_pair: Tuple[int, int]  # indices of closest pair
    most_distinct_pair: Tuple[int, int]    # indices of most distant pair
    clusters: List[List[int]]        # groups of similar structures
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_structures": self.n_structures,
            "metric": self.metric.value,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "mean_distance": self.mean_distance,
            "most_confusable_pair": self.most_confusable_pair,
            "most_distinct_pair": self.most_distinct_pair,
            "n_clusters": len(self.clusters),
            "cluster_sizes": [len(c) for c in self.clusters],
            "notes": self.notes,
        }


class ChannelCapacityAnalyzer:
    """
    Computes channel capacity and information metrics for HX-241 wave structures.

    Uses the ENGINES information theory engine (Shannon entropy, mutual
    information, channel capacity via Blahut-Arimoto, Gaussian channel
    capacity) to analyze how much information can be reliably transmitted
    using different wave structures.

    Key analyses:
    1. Shannon capacity for a given SNR and bandwidth
    2. Number of distinguishable wave states (via JS divergence threshold)
    3. Bits per symbol achievable with wave structure modulation
    4. Feature correlation analysis (which wave properties are independent)
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./xpock_hx241_capacity")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._info_engine = None

    def _get_info(self) -> WaveInformationEngine:
        if self._info_engine is None:
            from xpock.engines_integration import WaveInformationEngine
            self._info_engine = WaveInformationEngine(workspace=self.workspace / "information")
        return self._info_engine

    def analyze_capacity(
        self,
        snr_db: float,
        bandwidth_hz: float,
        wave_events: List[WaveEvent],
        error_correction_overhead: float = 0.25,
    ) -> CapacityAnalysisResult:
        """
        Full capacity analysis for a set of wave events.

        Steps:
        1. Analyze each wave event to extract measurable features
        2. Compute pairwise distinguishability (JS divergence)
        3. Count distinguishable states (above JS threshold)
        4. Compute Shannon capacity
        5. Estimate net information rate
        """
        info_engine = self._get_info()
        analyzer = WaveAnalyzer()
        notes: List[str] = []

        # --- Analyze all events ---
        signatures: List[WaveStructureSignature] = []
        for ev in wave_events:
            props = analyzer.analyze(ev)
            sig = WaveStructureSignature(
                event_id=ev.event_id,
                parameters=ev.parameters,
                duration_rms=props.temporal_duration_rms,
                duration_fwhm=props.temporal_duration_fwhm,
                bandwidth_rms=props.bandwidth_rms,
                temporal_center=props.temporal_center,
                beam_waist_x=props.beam_waist_x,
                beam_waist_y=props.beam_waist_y,
                beam_waist_r=props.beam_waist_r,
                oam_per_photon=props.oam_per_photon,
                winding_number=props.winding_number,
                degree_of_polarization=props.degree_of_polarization,
                stokes_s1=props.stokes_s1,
                stokes_s2=props.stokes_s2,
                stokes_s3=props.stokes_s3,
                schmidt_number=props.schmidt_number,
                total_power=props.total_power,
                peak_intensity=props.peak_intensity,
            )
            signatures.append(sig)

        # --- Pairwise distinguishability ---
        n = len(signatures)
        js_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p_i = signatures[i].to_probability_distribution()
                p_j = signatures[j].to_probability_distribution()
                js_result = info_engine.state_distinguishability(p_i, p_j, base=2)
                if js_result.success and js_result.result:
                    js_div = js_result.result.get("js_divergence", 0.0)
                else:
                    js_div = 0.0
                js_matrix[i, j] = js_div
                js_matrix[j, i] = js_div

        # --- Count distinguishable states (JS > threshold) ---
        js_threshold = 0.05  # states are distinguishable if JS divergence > threshold
        distinguishable = 0
        for i in range(n):
            is_dist = True
            for j in range(n):
                if i != j and js_matrix[i, j] < js_threshold:
                    is_dist = False
                    break
            if is_dist:
                distinguishable += 1

        # If no state is distinguishable from all others, at least 1 is usable
        if distinguishable == 0 and n > 0:
            distinguishable = 1
            notes.append("No state fully distinguishable from all others; using 1 state")

        # --- Shannon capacity ---
        snr_linear = 10.0 ** (snr_db / 10.0)
        shannon_capacity = bandwidth_hz * np.log2(1.0 + snr_linear)
        spectral_efficiency = shannon_capacity / bandwidth_hz if bandwidth_hz > 0 else 0.0

        # --- Usable capacity (after FEC overhead) ---
        usable_capacity = shannon_capacity * (1.0 - error_correction_overhead)

        # --- Bits per symbol ---
        bits_per_symbol = np.log2(distinguishable) if distinguishable > 0 else 0.0

        # --- Symbol rate (Nyquist: 2*B for baseband, B for passband) ---
        symbols_per_second = bandwidth_hz  # passband assumption

        # --- Net information rate ---
        net_rate = usable_capacity * (bits_per_symbol / spectral_efficiency) if spectral_efficiency > 0 else 0.0
        net_rate = min(net_rate, usable_capacity)  # cap at usable capacity

        # --- Statistics ---
        off_diagonal = js_matrix[~np.eye(n, dtype=bool)] if n > 1 else np.array([0.0])
        min_js = float(np.min(off_diagonal)) if len(off_diagonal) > 0 else 0.0
        avg_js = float(np.mean(off_diagonal)) if len(off_diagonal) > 0 else 0.0

        # --- Feature correlation ---
        feature_matrix = np.stack([s.to_feature_vector() for s in signatures])
        if n > 1:
            feature_corr = np.corrcoef(feature_matrix, rowvar=False)
            # Replace NaN with 0
            feature_corr = np.nan_to_num(feature_corr, nan=0.0)
        else:
            feature_corr = np.eye(feature_matrix.shape[1])

        notes.append(f"SNR: {snr_db:.1f} dB, Bandwidth: {bandwidth_hz:.1f} Hz")
        notes.append(f"Shannon capacity: {shannon_capacity:.1f} bps ({spectral_efficiency:.3f} bps/Hz)")
        notes.append(f"Distinguishable wave states: {distinguishable}")
        notes.append(f"Bits per symbol: {bits_per_symbol:.3f}")
        notes.append(f"Min JS divergence: {min_js:.4f}, Avg: {avg_js:.4f}")

        return CapacityAnalysisResult(
            config_name="wave_structure_analysis",
            snr_db=snr_db,
            bandwidth_hz=bandwidth_hz,
            shannon_capacity_bps=float(shannon_capacity),
            spectral_efficiency_bps_hz=float(spectral_efficiency),
            usable_capacity_bps=float(usable_capacity),
            distinguishable_states=int(distinguishable),
            bits_per_symbol=float(bits_per_symbol),
            symbols_per_second=float(symbols_per_second),
            net_information_rate_bps=float(net_rate),
            min_js_divergence=min_js,
            avg_js_divergence=avg_js,
            feature_correlation=feature_corr,
            notes=notes,
        )

    def analyze_distinguishability(
        self,
        wave_events: List[WaveEvent],
        metric: DistinguishabilityMetric = DistinguishabilityMetric.JS_DIVERGENCE,
    ) -> DistinguishabilityAnalysisResult:
        """Pairwise distinguishability analysis of wave structures.

        Returns a distance matrix and clustering of similar structures.
        Useful for identifying which wave properties produce
        maximally distinguishable states.
        """
        info_engine = self._get_info()
        analyzer = WaveAnalyzer()
        notes: List[str] = []

        # --- Extract signatures ---
        signatures: List[WaveStructureSignature] = []
        for ev in wave_events:
            props = analyzer.analyze(ev)
            sig = WaveStructureSignature(
                event_id=ev.event_id,
                parameters=ev.parameters,
                duration_rms=props.temporal_duration_rms,
                duration_fwhm=props.temporal_duration_fwhm,
                bandwidth_rms=props.bandwidth_rms,
                beam_waist_x=props.beam_waist_x,
                beam_waist_y=props.beam_waist_y,
                beam_waist_r=props.beam_waist_r,
                oam_per_photon=props.oam_per_photon,
                winding_number=props.winding_number,
                degree_of_polarization=props.degree_of_polarization,
                stokes_s1=props.stokes_s1,
                stokes_s2=props.stokes_s2,
                stokes_s3=props.stokes_s3,
                schmidt_number=props.schmidt_number,
                total_power=props.total_power,
                peak_intensity=props.peak_intensity,
            )
            signatures.append(sig)

        n = len(signatures)
        dist_matrix = np.zeros((n, n))

        # --- Compute pairwise distances ---
        for i in range(n):
            for j in range(i + 1, n):
                if metric == DistinguishabilityMetric.JS_DIVERGENCE:
                    p_i = signatures[i].to_probability_distribution()
                    p_j = signatures[j].to_probability_distribution()
                    result = info_engine.state_distinguishability(p_i, p_j, base=2)
                    if result.success and result.result:
                        dist = result.result.get("js_divergence", 0.0)
                    else:
                        dist = 0.0
                elif metric == DistinguishabilityMetric.EUCLIDEAN_DISTANCE:
                    vec_i = signatures[i].to_feature_vector()
                    vec_j = signatures[j].to_feature_vector()
                    # Normalize before distance
                    vec_i = vec_i / (np.linalg.norm(vec_i) + 1e-12)
                    vec_j = vec_j / (np.linalg.norm(vec_j) + 1e-12)
                    dist = float(np.linalg.norm(vec_i - vec_j))
                elif metric == DistinguishabilityMetric.COSINE_SIMILARITY:
                    vec_i = signatures[i].to_feature_vector()
                    vec_j = signatures[j].to_feature_vector()
                    norm_i = np.linalg.norm(vec_i) + 1e-12
                    norm_j = np.linalg.norm(vec_j) + 1e-12
                    cosine = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
                    dist = 1.0 - cosine  # distance = 1 - similarity
                else:
                    # Default: Euclidean
                    vec_i = signatures[i].to_feature_vector()
                    vec_j = signatures[j].to_feature_vector()
                    dist = float(np.linalg.norm(vec_i - vec_j))

                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # --- Find most confusable and most distinct pairs ---
        off_diag = dist_matrix.copy()
        np.fill_diagonal(off_diag, -np.inf)  # Exclude diagonal from max
        max_idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)

        off_diag_min = dist_matrix.copy()
        np.fill_diagonal(off_diag_min, np.inf)  # Exclude diagonal from min
        min_idx = np.unravel_index(np.argmin(off_diag_min), off_diag_min.shape)

        min_dist = float(dist_matrix[min_idx])
        max_dist = float(dist_matrix[max_idx])
        mean_dist = float(np.mean(off_diag[np.isfinite(off_diag)]))

        # --- Simple clustering: group structures within threshold ---
        threshold = 0.1
        clusters: List[List[int]] = []
        assigned = set()
        for i in range(n):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, n):
                if j not in assigned and dist_matrix[i, j] < threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        notes.append(f"Metric: {metric.value}, N structures: {n}")
        notes.append(f"Min distance: {min_dist:.4f} (pair {min_idx})")
        notes.append(f"Max distance: {max_dist:.4f} (pair {max_idx})")
        notes.append(f"Mean distance: {mean_dist:.4f}")
        notes.append(f"Clusters (threshold={threshold}): {len(clusters)}")

        return DistinguishabilityAnalysisResult(
            n_structures=n,
            metric=metric,
            pairwise_matrix=dist_matrix,
            min_distance=min_dist,
            max_distance=max_dist,
            mean_distance=mean_dist,
            most_confusable_pair=min_idx,
            most_distinct_pair=max_idx,
            clusters=clusters,
            notes=notes,
        )

    def analyze_feature_independence(
        self,
        wave_events: List[WaveEvent],
    ) -> Dict[str, Any]:
        """
        Analyze which wave features are statistically independent.

        Returns correlation matrix and identifies feature groups that
        can be modulated independently — critical for determining
        how many bits can be encoded per wave structure.
        """
        analyzer = WaveAnalyzer()
        signatures: List[WaveStructureSignature] = []

        for ev in wave_events:
            props = analyzer.analyze(ev)
            sig = WaveStructureSignature(
                event_id=ev.event_id,
                parameters=ev.parameters,
                duration_rms=props.temporal_duration_rms,
                duration_fwhm=props.temporal_duration_fwhm,
                bandwidth_rms=props.bandwidth_rms,
                beam_waist_x=props.beam_waist_x,
                beam_waist_y=props.beam_waist_y,
                beam_waist_r=props.beam_waist_r,
                oam_per_photon=props.oam_per_photon,
                winding_number=props.winding_number,
                degree_of_polarization=props.degree_of_polarization,
                stokes_s1=props.stokes_s1,
                stokes_s2=props.stokes_s2,
                stokes_s3=props.stokes_s3,
                schmidt_number=props.schmidt_number,
                total_power=props.total_power,
                peak_intensity=props.peak_intensity,
            )
            signatures.append(sig)

        feature_names = [
            "duration_rms", "duration_fwhm", "bandwidth_rms",
            "beam_waist_x", "beam_waist_y", "beam_waist_r",
            "oam_per_photon", "winding_number",
            "degree_of_polarization", "stokes_s1", "stokes_s2", "stokes_s3",
            "schmidt_number", "total_power", "peak_intensity",
        ]

        feature_matrix = np.stack([s.to_feature_vector() for s in signatures])
        if len(signatures) > 1:
            corr_matrix = np.corrcoef(feature_matrix, rowvar=False)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        else:
            corr_matrix = np.eye(len(feature_names))

        # Find highly correlated feature pairs (|r| > 0.7)
        correlated_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                r = corr_matrix[i, j]
                if abs(r) > 0.7:
                    correlated_pairs.append({
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "correlation": float(r),
                    })

        # Find independent feature groups (clustering by correlation)
        # Simple approach: features with |r| < 0.3 are approximately independent
        independent_groups = []
        used = set()
        for i in range(len(feature_names)):
            if i in used:
                continue
            group = [feature_names[i]]
            used.add(i)
            for j in range(i + 1, len(feature_names)):
                if j not in used and abs(corr_matrix[i, j]) < 0.3:
                    group.append(feature_names[j])
                    used.add(j)
            independent_groups.append(group)

        return {
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "correlation_matrix": corr_matrix.tolist(),
            "highly_correlated_pairs": correlated_pairs,
            "approximate_independent_groups": independent_groups,
            "n_independent_groups": len(independent_groups),
            "notes": [
                f"Analyzed {len(signatures)} wave events",
                f"Found {len(correlated_pairs)} highly correlated feature pairs",
                f"Identified {len(independent_groups)} approximately independent feature groups",
            ],
        }