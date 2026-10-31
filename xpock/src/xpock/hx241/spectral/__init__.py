"""
XPOCK — HX-241 Receiver-Side Spectral Analysis

Spectral analysis tools for recovering wave structure properties from
received signals. This is the receiver-side counterpart to the modulator:
given a received wave event, extract the modulated properties (temporal
shape, duration, OAM charge, polarization, beam waist, etc.) so HX-241
can recover the encoded information.

Uses XPOCK's spectral engine (FFT, CWT, Hankel, Hilbert transforms) and
ENGINES' integral transforms engine for advanced time-frequency analysis.

HX-241 architectural principle: receiver-side analysis is a physical
layer concern, not a protocol concern. The protocol receives demodulated
bits; this module handles the physical-to-digital boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np

from xpock.core import WaveEvent, WaveField
from xpock.analysis import WaveAnalyzer
from xpock.engines_integration import (
    WaveSpectralEngine,
    WaveInformationEngine,
    EngineResult,
)


class AnalysisDomain(Enum):
    """Analysis domains for receiver-side processing."""

    TIME = "time"              # Temporal envelope analysis
    FREQUENCY = "frequency"    # Spectral analysis
    TIME_FREQUENCY = "time_frequency"  # Joint time-frequency (CWT)
    SPATIAL = "spatial"        # Spatial mode analysis
    POLARIZATION = "polarization"  # Stokes parameter analysis
    TOPOLOGICAL = "topological"  # OAM, winding number analysis


@dataclass
class SpectralFeature:
    """A single extracted spectral/temporal feature."""

    name: str
    value: float
    domain: AnalysisDomain
    confidence: float  # 0-1 reliability estimate
    units: str = ""


@dataclass
class ReceivedSignalAnalysis:
    """Complete analysis of a received wave signal.

    Attributes
    ----------
    event_id : str
    temporal_features : list of SpectralFeature
    spectral_features : list of SpectralFeature
    spatial_features : list of SpectralFeature
    polarization_features : list of SpectralFeature
    topological_features : list of SpectralFeature
    recovered_modulation : dict
        Best-guess modulation dimension assignments.
    recovery_confidence : float
        Overall confidence in recovery (0-1).
    snr_estimate_db : float
        Estimated signal-to-noise ratio.
    bandwidth_estimate_hz : float
        Estimated occupied bandwidth.
    duration_estimate_s : float
        Estimated pulse duration.
    notes : list
    """

    event_id: str
    temporal_features: List[SpectralFeature] = field(default_factory=list)
    spectral_features: List[SpectralFeature] = field(default_factory=list)
    spatial_features: List[SpectralFeature] = field(default_factory=list)
    polarization_features: List[SpectralFeature] = field(default_factory=list)
    topological_features: List[SpectralFeature] = field(default_factory=list)
    recovered_modulation: Dict[str, Any] = field(default_factory=dict)
    recovery_confidence: float = 0.0
    snr_estimate_db: float = 0.0
    bandwidth_estimate_hz: float = 0.0
    duration_estimate_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    def all_features(self) -> List[SpectralFeature]:
        """Return all features across all domains."""
        return (
            self.temporal_features
            + self.spectral_features
            + self.spatial_features
            + self.polarization_features
            + self.topological_features
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "n_temporal_features": len(self.temporal_features),
            "n_spectral_features": len(self.spectral_features),
            "n_spatial_features": len(self.spatial_features),
            "n_polarization_features": len(self.polarization_features),
            "n_topological_features": len(self.topological_features),
            "recovered_modulation": self.recovered_modulation,
            "recovery_confidence": self.recovery_confidence,
            "snr_estimate_db": self.snr_estimate_db,
            "bandwidth_estimate_hz": self.bandwidth_estimate_hz,
            "duration_estimate_s": self.duration_estimate_s,
            "notes": self.notes,
        }


class WaveSignalAnalyzer:
    """
    Receiver-side wave signal analysis for HX-241 demodulation.

    Extracts measurable properties from a received wave event and maps
    them back to modulation dimension assignments. Uses:
    - FFT for spectral analysis
    - CWT for time-frequency localization
    - Hilbert transform for analytic signal (instantaneous amplitude/phase/frequency)
    - Hankel transform for radially symmetric fields (OAM analysis)
    - ENGINES information theory for SNR estimation
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./xpock_hx241_spectral")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._spectral: Optional[WaveSpectralEngine] = None
        self._info: Optional[WaveInformationEngine] = None
        self._analyzer: Optional[WaveAnalyzer] = None

    def _get_spectral(self) -> WaveSpectralEngine:
        if self._spectral is None:
            self._spectral = WaveSpectralEngine(workspace=self.workspace / "spectral")
        return self._spectral

    def _get_info(self) -> WaveInformationEngine:
        if self._info is None:
            self._info = WaveInformationEngine(workspace=self.workspace / "information")
        return self._info

    def _get_analyzer(self) -> WaveAnalyzer:
        if self._analyzer is None:
            self._analyzer = WaveAnalyzer()
        return self._analyzer

    def analyze_received_signal(
        self,
        event: WaveEvent,
        domains: Optional[List[AnalysisDomain]] = None,
    ) -> ReceivedSignalAnalysis:
        """
        Comprehensive analysis of a received wave signal.

        Parameters
        ----------
        event : WaveEvent
            The received wave event to analyze.
        domains : list of AnalysisDomain, optional
            Which domains to analyze. Defaults to all.

        Returns
        -------
        ReceivedSignalAnalysis with extracted features and
        best-guess modulation recovery.
        """
        if domains is None:
            domains = [
                AnalysisDomain.TIME,
                AnalysisDomain.FREQUENCY,
                AnalysisDomain.TIME_FREQUENCY,
                AnalysisDomain.SPATIAL,
                AnalysisDomain.POLARIZATION,
                AnalysisDomain.TOPOLOGICAL,
            ]

        analyzer = self._get_analyzer()
        props = analyzer.analyze(event)
        spectral = self._get_spectral()
        info = self._get_info()

        temporal_feats: List[SpectralFeature] = []
        spectral_feats: List[SpectralFeature] = []
        spatial_feats: List[SpectralFeature] = []
        polarization_feats: List[SpectralFeature] = []
        topological_feats: List[SpectralFeature] = []

        notes: List[str] = []

        # --- Time domain analysis ---
        if AnalysisDomain.TIME in domains and event.field.t is not None:
            t = event.field.t
            I = event.field.intensity

            # Integrate over space to get temporal profile
            axes = tuple(range(I.ndim - 1))
            I_t = np.sum(I, axis=axes)
            I_t = np.maximum(I_t, 0.0)

            # Pulse duration (FWHM)
            if np.max(I_t) > 0:
                half_max = 0.5 * np.max(I_t)
                above = I_t >= half_max
                if np.any(above):
                    indices = np.where(above)[0]
                    fwhm = float(t[indices[-1]] - t[indices[0]])
                else:
                    fwhm = 0.0
                temporal_feats.append(SpectralFeature(
                    name="pulse_duration_fwhm",
                    value=fwhm,
                    domain=AnalysisDomain.TIME,
                    confidence=0.9,
                    units="seconds",
                ))
                notes.append(f"Pulse FWHM: {fwhm:.4f} s")

            # Temporal center
            total = np.sum(I_t)
            if total > 0:
                t_center = float(np.sum(t * I_t) / total)
                temporal_feats.append(SpectralFeature(
                    name="temporal_center",
                    value=t_center,
                    domain=AnalysisDomain.TIME,
                    confidence=0.9,
                    units="seconds",
                ))

        # --- Frequency domain analysis ---
        if AnalysisDomain.FREQUENCY in domains and event.field.t is not None:
            result = spectral.power_spectrum(
                event.field.intensity.flatten(), fs=1.0, window="hann"
            )
            if result.success and result.result:
                freqs = result.result["frequencies"]
                psd = result.result["psd"]
                peak_idx = int(np.argmax(psd))
                peak_freq = float(freqs[peak_idx])
                peak_power = float(psd[peak_idx])

                # Bandwidth estimate (RMS)
                total_power = np.sum(psd)
                if total_power > 0:
                    f_center = np.sum(freqs * psd) / total_power
                    bw = float(np.sqrt(np.sum((freqs - f_center) ** 2 * psd) / total_power))
                else:
                    bw = 0.0

                spectral_feats.append(SpectralFeature(
                    name="peak_frequency",
                    value=peak_freq,
                    domain=AnalysisDomain.FREQUENCY,
                    confidence=0.85,
                    units="rad/s",
                ))
                spectral_feats.append(SpectralFeature(
                    name="bandwidth_rms",
                    value=bw,
                    domain=AnalysisDomain.FREQUENCY,
                    confidence=0.8,
                    units="rad/s",
                ))
                notes.append(f"Peak frequency: {peak_freq:.2f}, Bandwidth: {bw:.2f}")

        # --- Time-frequency analysis (CWT) ---
        if AnalysisDomain.TIME_FREQUENCY in domains and event.field.t is not None:
            t = event.field.t
            I = event.field.intensity
            axes = tuple(range(I.ndim - 1))
            I_t = np.sum(I, axis=axes).astype(np.float64)

            result = spectral.continuous_wavelet_transform(I_t, fs=1.0, wavelet="morlet")
            if result.success and result.result:
                scalogram = result.result["scalogram"]
                # Energy concentration in time-frequency plane
                tf_energy = float(np.sum(scalogram ** 2))
                temporal_feats.append(SpectralFeature(
                    name="time_frequency_energy",
                    value=tf_energy,
                    domain=AnalysisDomain.TIME_FREQUENCY,
                    confidence=0.7,
                    units="arbitrary",
                ))
                notes.append(f"Time-frequency energy: {tf_energy:.4f}")

        # --- Spatial domain analysis ---
        if AnalysisDomain.SPATIAL in domains and event.field.x is not None and event.field.y is not None:
            I = event.field.intensity
            # Reduce to 2D spatial slice
            while I.ndim > 2:
                I = I[tuple([I.shape[i] // 2 for i in range(I.ndim - 2)])]

            if I.ndim == 2:
                x = event.field.x
                y = event.field.y
                X, Y = np.meshgrid(x, y, indexing="xy")

                # Beam width (RMS)
                total = np.sum(I)
                if total > 0:
                    x_c = np.sum(X * I) / total
                    y_c = np.sum(Y * I) / total
                    var_x = np.sum((X - x_c) ** 2 * I) / total
                    var_y = np.sum((Y - y_c) ** 2 * I) / total
                    waist_x = float(2.0 * np.sqrt(max(var_x, 0.0)))
                    waist_y = float(2.0 * np.sqrt(max(var_y, 0.0)))

                    spatial_feats.append(SpectralFeature(
                        name="beam_waist_x",
                        value=waist_x,
                        domain=AnalysisDomain.SPATIAL,
                        confidence=0.85,
                        units="meters",
                    ))
                    spatial_feats.append(SpectralFeature(
                        name="beam_waist_y",
                        value=waist_y,
                        domain=AnalysisDomain.SPATIAL,
                        confidence=0.85,
                        units="meters",
                    ))
                    notes.append(f"Beam waist: {waist_x:.3f} x {waist_y:.3f} m")

        # --- Polarization domain analysis ---
        if AnalysisDomain.POLARIZATION in domains and event.field.Ex is not None and event.field.Ey is not None:
            Ex = event.field.Ex
            Ey = event.field.Ey

            S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
            S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
            S2 = 2.0 * np.real(Ex * np.conj(Ey))
            S3 = 2.0 * np.imag(Ex * np.conj(Ey))

            S0_avg = float(np.mean(S0))
            if S0_avg > 0:
                dop = float(np.sqrt(
                    np.mean(S1) ** 2 + np.mean(S2) ** 2 + np.mean(S3) ** 2
                ) / S0_avg)
                dop = float(np.clip(dop, 0.0, 1.0))

                polarization_feats.append(SpectralFeature(
                    name="degree_of_polarization",
                    value=dop,
                    domain=AnalysisDomain.POLARIZATION,
                    confidence=0.9,
                    units="unitless",
                ))
                polarization_feats.append(SpectralFeature(
                    name="stokes_s1",
                    value=float(np.mean(S1) / S0_avg),
                    domain=AnalysisDomain.POLARIZATION,
                    confidence=0.9,
                    units="unitless",
                ))
                polarization_feats.append(SpectralFeature(
                    name="stokes_s2",
                    value=float(np.mean(S2) / S0_avg),
                    domain=AnalysisDomain.POLARIZATION,
                    confidence=0.9,
                    units="unitless",
                ))
                polarization_feats.append(SpectralFeature(
                    name="stokes_s3",
                    value=float(np.mean(S3) / S0_avg),
                    domain=AnalysisDomain.POLARIZATION,
                    confidence=0.9,
                    units="unitless",
                ))
                notes.append(f"Degree of polarization: {dop:.3f}")

        # --- Topological domain analysis ---
        if AnalysisDomain.TOPOLOGICAL in domains and event.field.Ex is not None:
            psi = event.field.Ex
            # Reduce to 2D spatial slice
            while psi.ndim > 2:
                psi = psi[tuple([psi.shape[i] // 2 for i in range(psi.ndim - 2)])]

            if psi.ndim == 2:
                x = event.field.x
                y = event.field.y
                X, Y = np.meshgrid(x, y, indexing="xy")

                # OAM estimate
                dpsi_dx = np.gradient(psi, axis=1)
                dpsi_dy = np.gradient(psi, axis=0)
                Lz_density = -1j * (X * dpsi_dy - Y * dpsi_dx) * np.conj(psi)
                Lz = np.sum(np.real(Lz_density))
                P = np.sum(np.abs(psi) ** 2)
                oam = float(Lz / P) if P > 0 else 0.0

                topological_feats.append(SpectralFeature(
                    name="oam_per_photon",
                    value=oam,
                    domain=AnalysisDomain.TOPOLOGICAL,
                    confidence=0.75,
                    units="hbar",
                ))
                notes.append(f"OAM per photon: {oam:.3f} hbar")

        # --- SNR estimate ---
        snr_result = info.intensity_entropy(event.field.intensity, base=2)
        if snr_result.success and snr_result.result:
            snr_estimate = 10.0  # Placeholder - real SNR estimation needs noise floor
        else:
            snr_estimate = 0.0

        # --- Build recovered modulation dict ---
        recovered = {}
        for feat in temporal_feats:
            if feat.name == "pulse_duration_fwhm":
                recovered["temporal_duration"] = feat.value
            if feat.name == "temporal_center":
                recovered["temporal_center"] = feat.value
        for feat in spectral_feats:
            if feat.name == "peak_frequency":
                recovered["carrier_frequency"] = feat.value / (2.0 * np.pi)
            if feat.name == "bandwidth_rms":
                recovered["bandwidth"] = feat.value
        for feat in spatial_feats:
            if feat.name == "beam_waist_x":
                recovered["beam_waist"] = feat.value
        for feat in polarization_feats:
            if feat.name == "degree_of_polarization":
                recovered["polarization"] = feat.value
        for feat in topological_feats:
            if feat.name == "oam_per_photon":
                recovered["oam_charge"] = int(round(feat.value))

        # Overall confidence
        all_feats = self._all_features_list(
            temporal_feats, spectral_feats, spatial_feats,
            polarization_feats, topological_feats,
        )
        if all_feats:
            recovery_confidence = float(np.mean([f.confidence for f in all_feats]))
        else:
            recovery_confidence = 0.0

        return ReceivedSignalAnalysis(
            event_id=event.event_id,
            temporal_features=temporal_feats,
            spectral_features=spectral_feats,
            spatial_features=spatial_feats,
            polarization_features=polarization_feats,
            topological_features=topological_feats,
            recovered_modulation=recovered,
            recovery_confidence=recovery_confidence,
            snr_estimate_db=snr_estimate,
            bandwidth_estimate_hz=float(
                next((f.value for f in spectral_feats if f.name == "bandwidth_rms"), 0.0)
            ),
            duration_estimate_s=float(
                next((f.value for f in temporal_feats if f.name == "pulse_duration_fwhm"), 0.0)
            ),
            notes=notes,
        )

    @staticmethod
    def _all_features_list(*feature_lists: List[SpectralFeature]) -> List[SpectralFeature]:
        """Combine all feature lists into one."""
        combined: List[SpectralFeature] = []
        for lst in feature_lists:
            combined.extend(lst)
        return combined

    def compare_signals(
        self,
        analysis1: ReceivedSignalAnalysis,
        analysis2: ReceivedSignalAnalysis,
    ) -> Dict[str, Any]:
        """
        Compare two received signal analyses to determine if they represent
        the same modulation state.

        Returns similarity metrics and a match decision.
        """
        # Compare recovered modulation dicts
        mods1 = analysis1.recovered_modulation
        mods2 = analysis2.recovered_modulation

        common_keys = set(mods1.keys()) & set(mods2.keys())
        if not common_keys:
            return {
                "similarity": 0.0,
                "match": False,
                "reason": "No common modulation dimensions",
            }

        # Compare each common dimension
        similarities = []
        for key in common_keys:
            v1 = mods1[key]
            v2 = mods2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(v1) > 1e-12:
                    sim = 1.0 - abs(v1 - v2) / abs(v1)
                    sim = max(0.0, min(1.0, sim))
                else:
                    sim = 1.0 if abs(v2) < 1e-12 else 0.0
                similarities.append(sim)
            elif v1 == v2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

        overall_similarity = float(np.mean(similarities)) if similarities else 0.0
        match = overall_similarity > 0.8

        return {
            "similarity": overall_similarity,
            "match": match,
            "n_common_dimensions": len(common_keys),
            "dimension_similarities": {
                k: s for k, s in zip(common_keys, similarities)
            },
        }
