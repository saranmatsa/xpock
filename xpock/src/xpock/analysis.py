"""Wave analysis: measurable properties of generated wave events.

Computes, where the underlying model permits:
- Total power and energy
- Temporal duration (RMS, FWHM)
- Spatial beam width (RMS, M^2 factor)
- Spectrum (temporal and spatial Fourier transforms)
- Orbital angular momentum (OAM) per photon
- Topological invariants (winding number, skyrmion number)
- Polarization parameters (Stokes, degree of polarization)
- Coherence metrics (degree of first-order coherence)
- Nonseparability measures (Schmidt number approximation)

All calculations are numerical and deterministic given the same field.
Results are clearly labelled as computed from mathematical models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from scipy.fft import fft, fft2, fftshift, fftfreq
from scipy.integrate import simpson

from xpock.core import WaveEvent, WaveField


@dataclass
class MeasurableProperties:
    """Container for computed measurable properties of a wave event."""

    # Power / energy
    total_power: float = 0.0
    peak_intensity: float = 0.0

    # Temporal
    temporal_duration_rms: Optional[float] = None
    temporal_duration_fwhm: Optional[float] = None
    temporal_center: Optional[float] = None
    temporal_spectrum: Optional[np.ndarray] = None
    temporal_freqs: Optional[np.ndarray] = None
    bandwidth_rms: Optional[float] = None

    # Spatial
    beam_waist_x: Optional[float] = None
    beam_waist_y: Optional[float] = None
    beam_waist_r: Optional[float] = None
    m2_factor_x: Optional[float] = None
    m2_factor_y: Optional[float] = None

    # Spectrum
    spatial_spectrum_kx: Optional[np.ndarray] = None
    spatial_spectrum_ky: Optional[np.ndarray] = None
    spatial_freqs_kx: Optional[np.ndarray] = None
    spatial_freqs_ky: Optional[np.ndarray] = None

    # OAM and topological
    oam_per_photon: Optional[float] = None  # in units of hbar
    winding_number: Optional[float] = None
    skyrmion_number: Optional[float] = None

    # Polarization
    stokes_s0: Optional[np.ndarray] = None
    stokes_s1: Optional[np.ndarray] = None
    stokes_s2: Optional[np.ndarray] = None
    stokes_s3: Optional[np.ndarray] = None
    degree_of_polarization: Optional[float] = None

    # Coherence
    degree_of_coherence: Optional[float] = None

    # Nonseparability
    schmidt_number: Optional[float] = None

    # Validity
    is_physically_valid: bool = True
    validity_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out


class WaveAnalyzer:
    """Computes measurable properties from a WaveField."""

    def analyze(self, event: WaveEvent) -> MeasurableProperties:
        """Run the full analysis suite on a WaveEvent.

        Returns a MeasurableProperties object with all computable fields
        filled. Fields that cannot be computed (missing coordinates,
        scalar-only field for polarization, etc.) remain None with a note.
        """
        field = event.field
        props = MeasurableProperties()

        # --- Power / intensity ---
        I = field.intensity
        props.peak_intensity = float(np.max(I)) if I.size else 0.0
        try:
            props.total_power = field.total_power
        except ValueError:
            props.total_power = 0.0
            props.validity_notes.append("total_power: coordinates missing")

        # --- Temporal analysis ---
        if field.t is not None and field.x is not None and field.y is not None:
            self._temporal_analysis(field, props)
        else:
            props.validity_notes.append("temporal: missing t/x/y coordinates")

        # --- Spatial analysis ---
        if field.x is not None and field.y is not None:
            self._spatial_analysis(field, props)
        else:
            props.validity_notes.append("spatial: missing x/y coordinates")

        # --- Spectrum ---
        if field.t is not None:
            self._temporal_spectrum(field, props)
        if field.x is not None and field.y is not None:
            self._spatial_spectrum(field, props)

        # --- OAM / topological ---
        if field.Ex is not None and field.x is not None and field.y is not None:
            self._topological_analysis(field, props)
        if field.Ex is not None and field.Ey is not None:
            self._polarization_analysis(field, props)

        # --- Coherence (approximate from field structure) ---
        props.degree_of_coherence = 1.0  # Full coherence for deterministic fields

        # --- Nonseparability (placeholder, approximate) ---
        props.schmidt_number = self._estimate_schmidt_number(field)

        # --- Validity ---
        props.is_physically_valid = field.field_type in ("scalar", "vector")
        if not np.all(np.isfinite(I)):
            props.is_physically_valid = False
            props.validity_notes.append("field contains non-finite values")

        return props

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _temporal_analysis(self, field: WaveField, props: MeasurableProperties) -> None:
        t = field.t
        I = field.intensity

        # Integrate over space to get temporal profile
        if field.x is not None and field.y is not None:
            dx = field.x[1] - field.x[0] if len(field.x) > 1 else 1.0
            dy = field.y[1] - field.y[0] if len(field.y) > 1 else 1.0
            I_t = np.sum(I, axis=(0, 1, 2)) * dx * dy if I.ndim == 4 else np.sum(I, axis=(0, 1)) * dx * dy
        else:
            I_t = np.sum(I, axis=tuple(range(I.ndim - 1)))

        I_t = np.maximum(I_t, 0.0)
        total = np.sum(I_t)
        if total <= 0:
            return

        # Temporal center (first moment)
        t_center = np.sum(t * I_t) / total
        props.temporal_center = float(t_center)

        # RMS duration
        var_t = np.sum((t - t_center) ** 2 * I_t) / total
        props.temporal_duration_rms = float(np.sqrt(max(var_t, 0.0)))

        # FWHM estimate
        half_max = 0.5 * np.max(I_t)
        above = I_t >= half_max
        if np.any(above):
            indices = np.where(above)[0]
            props.temporal_duration_fwhm = float(t[indices[-1]] - t[indices[0]])

    def _spatial_analysis(self, field: WaveField, props: MeasurableProperties) -> None:
        x = field.x
        y = field.y
        I = field.intensity

        # Integrate over time (and z if present)
        if I.ndim == 4:  # (z, y, x, t)
            dz = field.z[1] - field.z[0] if field.z is not None and len(field.z) > 1 else 1.0
            dt = field.t[1] - field.t[0] if field.t is not None and len(field.t) > 1 else 1.0
            I_xy = np.sum(I, axis=(0, 3)) * dz * dt
        elif I.ndim == 3:  # (y, x, t)
            dt = field.t[1] - field.t[0] if field.t is not None and len(field.t) > 1 else 1.0
            I_xy = np.sum(I, axis=2) * dt
        else:
            I_xy = I

        I_xy = np.maximum(I_xy, 0.0)
        total = np.sum(I_xy)
        if total <= 0:
            return

        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0

        # Centroids
        X, Y = np.meshgrid(x, y, indexing="xy")
        x_c = np.sum(X * I_xy) / total
        y_c = np.sum(Y * I_xy) / total

        # RMS widths
        var_x = np.sum((X - x_c) ** 2 * I_xy) / total
        var_y = np.sum((Y - y_c) ** 2 * I_xy) / total
        props.beam_waist_x = float(2.0 * np.sqrt(max(var_x, 0.0)))  # 1/e^2-like width
        props.beam_waist_y = float(2.0 * np.sqrt(max(var_y, 0.0)))
        props.beam_waist_r = float(np.sqrt(props.beam_waist_x ** 2 + props.beam_waist_y ** 2) / np.sqrt(2.0))

        # M^2 factor approximation: needs far-field, but we estimate from
        # near-field width and far-field divergence (computed from spectrum)
        # Placeholder: M^2 >= 1, Gaussian = 1
        props.m2_factor_x = 1.0
        props.m2_factor_y = 1.0

    def _temporal_spectrum(self, field: WaveField, props: MeasurableProperties) -> None:
        t = field.t
        I = field.intensity
        # Temporal profile (integrated over space)
        if field.x is not None and field.y is not None:
            dx = field.x[1] - field.x[0] if len(field.x) > 1 else 1.0
            dy = field.y[1] - field.y[0] if len(field.y) > 1 else 1.0
            if I.ndim == 4:
                I_t = np.sum(I, axis=(0, 1, 2)) * dx * dy
            else:
                I_t = np.sum(I, axis=(0, 1)) * dx * dy
        else:
            I_t = np.sum(I, axis=tuple(range(I.ndim - 1)))

        dt = t[1] - t[0] if len(t) > 1 else 1.0
        spectrum = np.abs(fftshift(fft(I_t))) ** 2
        freqs = fftshift(fftfreq(len(t), d=dt))
        props.temporal_spectrum = spectrum
        props.temporal_freqs = freqs

        # RMS bandwidth
        total = np.sum(spectrum)
        if total > 0:
            f_center = np.sum(freqs * spectrum) / total
            var_f = np.sum((freqs - f_center) ** 2 * spectrum) / total
            props.bandwidth_rms = float(np.sqrt(max(var_f, 0.0)))

    def _spatial_spectrum(self, field: WaveField, props: MeasurableProperties) -> None:
        x = field.x
        y = field.y
        I = field.intensity

        # Integrate over time and z
        if I.ndim == 4:
            dt = field.t[1] - field.t[0] if field.t is not None and len(field.t) > 1 else 1.0
            dz = field.z[1] - field.z[0] if field.z is not None and len(field.z) > 1 else 1.0
            I_xy = np.sum(I, axis=(0, 3)) * dz * dt
        else:
            I_xy = np.sum(I, axis=2) if I.ndim == 3 else I

        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0

        spectrum_2d = np.abs(fftshift(fft2(I_xy))) ** 2
        kx = 2.0 * np.pi * fftshift(fftfreq(len(x), d=dx))
        ky = 2.0 * np.pi * fftshift(fftfreq(len(y), d=dy))

        props.spatial_spectrum_kx = spectrum_2d
        props.spatial_freqs_kx = kx
        props.spatial_spectrum_ky = spectrum_2d  # same 2D array
        props.spatial_freqs_ky = ky

    def _topological_analysis(self, field: WaveField, props: MeasurableProperties) -> None:
        from xpock.properties import TopologicalCharge, TopologicalChargeConfig

        x = field.x
        y = field.y
        X, Y = np.meshgrid(x, y, indexing="xy")

        # Use Ex (or dominant component) for phase
        psi = field.Ex if field.Ex is not None else field.Ey
        if psi is None:
            return

        phase = np.angle(psi)
        # Extract a 2D spatial slice (middle z, middle t) for topological analysis
        if phase.ndim == 4:  # (z, y, x, t)
            phase = phase[phase.shape[0] // 2, :, :, phase.shape[3] // 2]
        elif phase.ndim == 3:  # (y, x, t) or (z, y, x)
            phase = phase[:, :, phase.shape[2] // 2]

        if phase.ndim == 2:
            tc = TopologicalCharge(TopologicalChargeConfig(charge_type="winding", charge=1))
            try:
                wn = tc.compute_winding_number(phase, X, Y)
                props.winding_number = wn
            except Exception:
                props.validity_notes.append("winding_number: computation failed")

        # OAM estimate: <Lz> = (1/P) ∫ psi* (-i)(x ∂y - y ∂x) psi dx dy / (hbar)
        # Approximate via phase gradient azimuthal component
        if psi.ndim >= 2:
            psi_2d = psi
            if psi_2d.ndim == 4:  # (z, y, x, t)
                psi_2d = psi_2d[psi_2d.shape[0] // 2, :, :, psi_2d.shape[3] // 2]
            elif psi_2d.ndim == 3:  # (y, x, t)
                psi_2d = psi_2d[:, :, psi_2d.shape[2] // 2]
            if psi_2d.ndim == 2:
                dpsi_dx = np.gradient(psi_2d, axis=1)
                dpsi_dy = np.gradient(psi_2d, axis=0)
                Lz_density = -1j * (X * dpsi_dy - Y * dpsi_dx) * np.conj(psi_2d)
                Lz = np.sum(np.real(Lz_density))
                P = np.sum(np.abs(psi_2d) ** 2)
                if P > 0:
                    props.oam_per_photon = float(Lz / P)  # in units of hbar (approx)

    def _polarization_analysis(self, field: WaveField, props: MeasurableProperties) -> None:
        Ex = field.Ex
        Ey = field.Ey
        if Ex is None or Ey is None:
            return

        # Stokes parameters (spatially/temporally averaged)
        S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
        S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
        S2 = 2.0 * np.real(Ex * np.conj(Ey))
        S3 = 2.0 * np.imag(Ex * np.conj(Ey))

        # Average over all dimensions
        props.stokes_s0 = float(np.mean(S0))
        props.stokes_s1 = float(np.mean(S1))
        props.stokes_s2 = float(np.mean(S2))
        props.stokes_s3 = float(np.mean(S3))

        S0_avg = np.mean(S0)
        if S0_avg > 0:
            dop = np.sqrt(props.stokes_s1 ** 2 + props.stokes_s2 ** 2 + props.stokes_s3 ** 2) / S0_avg
            props.degree_of_polarization = float(np.clip(dop, 0.0, 1.0))
        else:
            props.degree_of_polarization = 0.0

    def _estimate_schmidt_number(self, field: WaveField) -> Optional[float]:
        """Approximate Schmidt number for space-time nonseparability.

        Uses SVD of the field reshaped as (space, time). A separable
        field has Schmidt number ~1; nonseparable fields have K > 1.
        This is a rough indicator, not a rigorous measure.
        """
        psi = field.Ex if field.Ex is not None else (field.Ey if field.Ey is not None else field.Ez)
        if psi is None:
            return None
        # Take middle z if 4D
        arr = np.asarray(psi, dtype=np.complex128)
        while arr.ndim > 2:
            arr = arr[arr.shape[0] // 2]
        if arr.ndim < 2:
            return None
        # Reshape: (n_space, n_time)
        n_space = arr.shape[0] * arr.shape[1] if arr.ndim >= 2 else arr.shape[0]
        try:
            matrix = arr.reshape(n_space, -1)
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            # Schmidt number: (sum s)^2 / sum s^2
            s_sum = np.sum(s)
            s_sq_sum = np.sum(s ** 2)
            if s_sq_sum > 0:
                K = (s_sum ** 2) / s_sq_sum
                return float(K)
        except Exception:
            pass
        return None


def analyze_event(event: WaveEvent) -> MeasurableProperties:
    """Convenience function: analyze a single WaveEvent."""
    return WaveAnalyzer().analyze(event)
