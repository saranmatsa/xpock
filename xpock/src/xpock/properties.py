"""Physical wave property modules.

Each class in this module models a physically meaningful wave property:
- TemporalProfile: temporal pulse shapes (Gaussian, sech, super-Gaussian, etc.)
- SpatialMode: spatial mode families (HG, LG, Bessel, spherical, etc.)
- PolarizationTexture: polarization states including spatially varying and skyrmionic
- TopologicalCharge: topological invariants (winding numbers, skyrmion numbers)
- CoherenceModel: partial coherence and mutual coherence functions
- CoupledMode: coupled-mode interactions between field components

These modules produce complex field envelopes on (x, y, z, t) grids. They are
designed to be combined multiplicatively or superposed to create rich,
nonseparable wave structures.

All models assume scalar or paraxial vector wavefields unless noted. They are
mathematical models of finite wave events — not claims about experimentally
demonstrated physical states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Callable
from enum import Enum
import math
import numpy as np

from xpock.core import PolarizationState


class PulseShape(Enum):
    """Supported temporal pulse envelope shapes."""

    GAUSSIAN = "gaussian"
    SECH = "sech"
    SUPER_GAUSSIAN = "super_gaussian"
    RECTANGULAR = "rectangular"
    EXPONENTIAL = "exponential"
    CHIRPED_GAUSSIAN = "chirped_gaussian"
    BISECTORIAL = "bisectorial"  # Two-pulse burst
    CUSTOM = "custom"


@dataclass
class TemporalProfile:
    """Generate a 1D temporal envelope A(t) on a time grid.

    Parameters
    ----------
    shape : PulseShape
        Functional form of the envelope.
    amplitude : float
        Peak amplitude.
    center : float
        Temporal center of the pulse.
    duration : float
        Characteristic duration (FWHM-related for Gaussian/sech).
    chirp_rate : float
        Quadratic phase chirp (rad/s^2). Zero = transform-limited.
    carrier_freq : float
        Optical carrier frequency (rad/s). The envelope multiplies
        exp(-i * carrier_freq * t).
    custom_fn : callable, optional
        If shape == CUSTOM, callable(t) -> complex envelope.
    """

    shape: PulseShape = PulseShape.GAUSSIAN
    amplitude: float = 1.0
    center: float = 0.0
    duration: float = 1.0
    chirp_rate: float = 0.0
    carrier_freq: float = 0.0
    custom_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the temporal envelope on grid t.

        Returns complex envelope including carrier and chirp.
        """
        t = np.asarray(t, dtype=np.float64)
        tau = t - self.center

        if self.shape == PulseShape.GAUSSIAN:
            # Standard Gaussian: exp(-tau^2 / (2*sigma^2)), duration ~ FWHM
            sigma = self.duration / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            env = np.exp(-0.5 * (tau / sigma) ** 2)
        elif self.shape == PulseShape.SECH:
            # Sech envelope: sech(tau / duration)
            env = 1.0 / np.cosh(tau / self.duration)
        elif self.shape == PulseShape.SUPER_GAUSSIAN:
            # exp(-(tau/duration)^(2*order)), order=4 default
            order = 4
            env = np.exp(-(np.abs(tau) / self.duration) ** (2 * order))
        elif self.shape == PulseShape.RECTANGULAR:
            env = np.where(np.abs(tau) <= self.duration / 2.0, 1.0, 0.0)
        elif self.shape == PulseShape.EXPONENTIAL:
            # One-sided exponential decay starting at center
            env = np.where(tau >= 0, np.exp(-tau / self.duration), 0.0)
        elif self.shape == PulseShape.CHIRPED_GAUSSIAN:
            sigma = self.duration / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            env = np.exp(-0.5 * (tau / sigma) ** 2)
        elif self.shape == PulseShape.BISECTORIAL:
            # Two equal pulses separated by duration
            env = np.exp(-((tau - self.duration / 2.0) / (self.duration / 4.0)) ** 2) + np.exp(
                -((tau + self.duration / 2.0) / (self.duration / 4.0)) ** 2
            )
        elif self.shape == PulseShape.CUSTOM:
            if self.custom_fn is None:
                raise ValueError("custom_fn required when shape == CUSTOM")
            env = self.custom_fn(t)
        else:
            raise ValueError(f"Unknown pulse shape: {self.shape}")

        env = env.astype(np.complex128) * self.amplitude

        # Apply quadratic chirp: exp(i * 0.5 * chirp_rate * tau^2)
        if self.chirp_rate != 0.0:
            env = env * np.exp(0.5j * self.chirp_rate * tau ** 2)

        # Apply carrier: exp(-i * carrier_freq * t)
        if self.carrier_freq != 0.0:
            env = env * np.exp(-1j * self.carrier_freq * t)

        return env


@dataclass
class SpatialModeConfig:
    """Configuration for a spatial mode.

    Parameters
    ----------
    mode : str
        Mode family: "gaussian", "hermite_gaussian", "laguerre_gaussian",
        "bessel", "super_gaussian", "spherical", "plane_wave", "nonseparable".
    n, m : int
        Hermite-Gaussian orders (n along x, m along y).
    p, l : int
        Laguerre-Gaussian radial (p) and azimuthal (l) indices. l = OAM charge.
    waist : float
        Beam waist w0.
    rayleigh_range : float, optional
        Computed from waist and wavelength if not given.
    wavelength : float
        Central wavelength (for Gouy phase, k-vector).
    k : float
        Wavenumber 2*pi/wavelength. Overrides wavelength if set.
    position : tuple
        Beam center (x0, y0, z0).
    aperture_radius : float, optional
        Hard aperture applied to the mode.
    """

    mode: str = "gaussian"
    n: int = 0
    m: int = 0
    p: int = 0
    l: int = 0
    waist: float = 1.0
    rayleigh_range: Optional[float] = None
    wavelength: float = 1.0
    k: Optional[float] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    aperture_radius: Optional[float] = None

    def __post_init__(self) -> None:
        if self.k is None:
            self.k = 2.0 * np.pi / self.wavelength
        if self.rayleigh_range is None:
            self.rayleigh_range = np.pi * self.waist ** 2 / self.wavelength


class SpatialMode:
    """Generate spatial mode profiles psi(x, y, z) on a transverse grid.

    Implements scalar paraxial modes. These are the building blocks for
    constructing richer spatiotemporal structures.
    """

    def __init__(self, config: SpatialModeConfig):
        self.cfg = config

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: float = 0.0,
    ) -> np.ndarray:
        """Evaluate the spatial mode on meshgrid x, y at axial position z.

        x, y : 2D arrays from np.meshgrid (indexing='xy').
        Returns complex amplitude psi(x, y, z).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = float(z)

        # If 1D coordinate arrays are passed, build a 2D meshgrid.
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y, indexing="xy")
        else:
            X, Y = x, y

        x0, y0, z0 = self.cfg.position
        Xc = X - x0
        Yc = Y - y0
        Z = z - z0

        w0 = self.cfg.waist
        zR = self.cfg.rayleigh_range
        k = self.cfg.k

        # Beam radius and wavefront curvature at Z
        w = w0 * np.sqrt(1.0 + (Z / zR) ** 2)
        R = Z * (1.0 + (zR / Z) ** 2) if Z != 0.0 else np.inf
        gouy = (self.cfg.n + self.cfg.m + 1) * np.arctan(Z / zR)

        if self.cfg.mode == "gaussian":
            psi = self._gaussian(X, Y, w, Z, k, R, gouy)
        elif self.cfg.mode == "hermite_gaussian":
            psi = self._hermite_gaussian(X, Y, w, Z, k, R, gouy)
        elif self.cfg.mode == "laguerre_gaussian":
            psi = self._laguerre_gaussian(X, Y, w, Z, k, R, gouy)
        elif self.cfg.mode == "bessel":
            psi = self._bessel(X, Y, k, Z)
        elif self.cfg.mode == "super_gaussian":
            psi = self._super_gaussian(X, Y, w)
        elif self.cfg.mode == "spherical":
            psi = self._spherical(X, Y, Z, k)
        elif self.cfg.mode == "plane_wave":
            psi = self._plane_wave(X, Y, Z, k)
        elif self.cfg.mode == "nonseparable":
            psi = self._nonseparable(X, Y, Z, k, w0, zR)
        else:
            raise ValueError(f"Unknown spatial mode: {self.cfg.mode}")

        # Apply hard aperture if specified
        if self.cfg.aperture_radius is not None:
            r = np.sqrt(X ** 2 + Y ** 2)
            psi = np.where(r <= self.cfg.aperture_radius, psi, 0.0)

        return psi

    # ------------------------------------------------------------------
    # Individual mode constructors
    # ------------------------------------------------------------------

    def _gaussian(self, Xc, Yc, w, Z, k, R, gouy):
        r2 = Xc ** 2 + Yc ** 2
        amp = np.exp(-r2 / w ** 2)
        phase = k * r2 / (2.0 * R) if np.isfinite(R) else 0.0
        phase = phase - gouy
        # Normalization: 1/(w*sqrt(pi))
        norm = 1.0 / (w * np.sqrt(np.pi))
        return norm * amp * np.exp(1j * phase)

    def _hermite_gaussian(self, Xc, Yc, w, Z, k, R, gouy):
        from scipy.special import eval_hermite

        r2 = Xc ** 2 + Yc ** 2
        n, m = self.cfg.n, self.cfg.m
        Hn = eval_hermite(n, np.sqrt(2.0) * Xc / w)
        Hm = eval_hermite(m, np.sqrt(2.0) * Yc / w)
        amp = np.exp(-r2 / w ** 2) * Hn * Hm
        phase = k * r2 / (2.0 * R) - gouy
        norm = 1.0 / (w * np.sqrt(np.pi * 2.0 ** (n + m) * math.factorial(n) * math.factorial(m)))
        return norm * amp * np.exp(1j * phase)

    def _laguerre_gaussian(self, Xc, Yc, w, Z, k, R, gouy):
        from scipy.special import eval_genlaguerre, factorial

        r2 = Xc ** 2 + Yc ** 2
        r = np.sqrt(r2)
        phi = np.arctan2(Yc, Xc)
        p, l = self.cfg.p, self.cfg.l
        w_l = w * np.sqrt(2.0)  # LG scaling convention
        L = eval_genlaguerre(p, abs(l), 2.0 * r2 / w_l ** 2)
        amp = (np.sqrt(2.0) * r / w_l) ** abs(l) * np.exp(-r2 / w_l ** 2) * L
        zR_local = self.cfg.rayleigh_range
        phase = l * phi + k * r2 / (2.0 * R) - (2 * p + abs(l) + 1) * np.arctan(Z / zR_local)
        norm = 1.0 / (w_l * np.sqrt(np.pi)) * np.sqrt(factorial(p) / factorial(p + abs(l)))
        return norm * amp * np.exp(1j * phase)

    def _bessel(self, Xc, Yc, k, Z):
        from scipy.special import j0

        r = np.sqrt(Xc ** 2 + Yc ** 2)
        # Zero-order Bessel beam: J0(k_r * r) * exp(i * k_z * Z)
        k_r = k * 0.1  # small cone angle
        k_z = np.sqrt(k ** 2 - k_r ** 2)
        return j0(k_r * r) * np.exp(1j * k_z * Z)

    def _super_gaussian(self, Xc, Yc, w):
        order = 4
        r2 = Xc ** 2 + Yc ** 2
        return np.exp(-(r2 / w ** 2) ** order)

    def _spherical(self, Xc, Yc, Z, k):
        r = np.sqrt(Xc ** 2 + Yc ** 2 + Z ** 2)
        r = np.where(r == 0.0, 1e-12, r)
        return np.exp(1j * k * r) / r

    def _plane_wave(self, Xc, Yc, Z, k):
        return np.exp(1j * k * Z) * np.ones_like(Xc)

    def _nonseparable(self, Xc, Yc, Z, k, w0, zR):
        """A nonseparable space-time-like mode: spatial LG with an axial
        envelope that depends on transverse radius (simulating a
        spatiotemporal vortex precursor)."""
        r2 = Xc ** 2 + Yc ** 2
        r = np.sqrt(r2)
        phi = np.arctan2(Yc, Xc)
        l = self.cfg.l if self.cfg.l != 0 else 1
        # Radial-dependent axial phase creates nonseparability
        axial = np.exp(1j * k * Z * (1.0 + 0.1 * r2 / w0 ** 2))
        azimuthal = np.exp(1j * l * phi)
        radial = np.exp(-r2 / w0 ** 2)
        return radial * azimuthal * axial


# -----------------------------------------------------------------------
# Polarization textures
# ----------------------------------------------------------------------


@dataclass
class PolarizationConfig:
    """Configuration for polarization texture.

    state : PolarizationState
    angle : float  # linear polarization angle (rad) for LINEAR
    handedness : int  # +1 for LCP, -1 for RCP
    jones_fn : callable  # (x, y) -> (2,) complex Jones vector for CUSTOM_JONES
    skyrmion_radius : float
    skyrmion_charge : int  # topological charge of the skyrmion texture
    """

    state: PolarizationState = PolarizationState.LINEAR
    angle: float = 0.0
    handedness: int = +1
    jones_fn: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None
    skyrmion_radius: float = 1.0
    skyrmion_charge: int = 1


class PolarizationTexture:
    """Generate spatially varying polarization Jones vectors on (x, y).

    Returns Ex, Ey complex component fields (Ez = 0 for paraxial).
    Supports linear, circular, elliptical, radial, azimuthal, spatially
    varying Jones fields, and skyrmion textures.
    """

    def __init__(self, config: PolarizationConfig):
        self.cfg = config

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Ex, Ey) complex Jones component fields on mesh x, y."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.cfg.state == PolarizationState.LINEAR:
            return self._linear(x, y)
        if self.cfg.state == PolarizationState.CIRCULAR:
            return self._circular(x, y)
        if self.cfg.state == PolarizationState.ELLIPTICAL:
            return self._elliptical(x, y)
        if self.cfg.state == PolarizationState.RADIAL:
            return self._radial(x, y)
        if self.cfg.state == PolarizationState.AZIMUTHAL:
            return self._azimuthal(x, y)
        if self.cfg.state == PolarizationState.SPATIALLY_VARYING:
            return self._spatially_varying(x, y)
        if self.cfg.state == PolarizationState.SKYRMIONIC:
            return self._skyrmion(x, y)
        if self.cfg.state == PolarizationState.CUSTOM_JONES:
            return self._custom_jones(x, y)
        if self.cfg.state == PolarizationState.UNPOLARIZED:
            return self._unpolarized(x, y)
        raise ValueError(f"Unknown polarization state: {self.cfg.state}")

    def _linear(self, x, y):
        theta = self.cfg.angle
        Ex = np.cos(theta) * np.ones_like(x, dtype=np.complex128)
        Ey = np.sin(theta) * np.ones_like(x, dtype=np.complex128)
        return Ex, Ey

    def _circular(self, x, y):
        s = self.cfg.handedness
        Ex = (1.0 / np.sqrt(2.0)) * np.ones_like(x, dtype=np.complex128)
        Ey = (1.0j * s / np.sqrt(2.0)) * np.ones_like(x, dtype=np.complex128)
        return Ex, Ey

    def _elliptical(self, x, y):
        # General elliptical: angle theta, ellipticity epsilon
        theta = self.cfg.angle
        eps = 0.5  # fixed ellipticity for now; extendable
        Ex = np.cos(theta) * np.cos(eps) * np.ones_like(x, dtype=np.complex128)
        Ey = (np.sin(theta) * np.cos(eps) + 1j * np.sin(eps)) * np.ones_like(x, dtype=np.complex128)
        # Normalize
        norm = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
        norm = np.where(norm == 0, 1.0, norm)
        return Ex / norm, Ey / norm

    def _radial(self, x, y):
        phi = np.arctan2(y, x)
        Ex = np.cos(phi)
        Ey = np.sin(phi)
        return Ex.astype(np.complex128), Ey.astype(np.complex128)

    def _azimuthal(self, x, y):
        phi = np.arctan2(y, x)
        Ex = -np.sin(phi)
        Ey = np.cos(phi)
        return Ex.astype(np.complex128), Ey.astype(np.complex128)

    def _spatially_varying(self, x, y):
        # Linear polarization angle rotates linearly with azimuth: theta = m*phi
        m = 1  # vortex charge of polarization
        phi = np.arctan2(y, x)
        theta = m * phi
        Ex = np.cos(theta)
        Ey = np.sin(theta)
        return Ex.astype(np.complex128), Ey.astype(np.complex128)

    def _skyrmion(self, x, y):
        """Skyrmion polarization texture: Stokes vector wraps the Bloch sphere.

        The normalized Stokes vector is
        n = (sin(theta(r)) cos(q*phi), sin(theta(r)) sin(q*phi), cos(theta(r)))

        Construction notes:
        - theta(r) = pi * r / R for r < R and theta = pi for r >= R, so the texture
          is confined to the skyrmion radius and satisfies the proper boundary
          condition n -> constant (uniform vertical polarization) beyond R. Without
          this clamping, theta keeps growing past pi across the whole grid, the
          polarization wraps multiple times, and the computed skyrmion number is
          neither integer nor equal to the imposed charge.
        - The Jones vector is the pole-free representation
          Ex = cos(theta/2), Ey = sin(theta/2) exp(i q phi), which yields exactly
          S1 = cos(theta), S2 = sin(theta) cos(q phi), S3 = sin(theta) sin(q phi)
          with |Ex|^2 + |Ey|^2 = 1 everywhere. The previous Stokes->Jones mapping
          divided by Ex and was singular at the Bloch-sphere poles (Ex = 0),
          producing a zero field at r = R and a corrupted skyrmion number.
        """
        R = self.cfg.skyrmion_radius
        q = self.cfg.skyrmion_charge
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        # Confine the texture to the skyrmion radius: 0 at center, pi at edge,
        # constant pi beyond R (uniform boundary polarization).
        theta = np.pi * np.clip(r / R, 0.0, 1.0)
        half = theta / 2.0

        # Pole-free Jones representation of the normalized Stokes vector:
        #   S1 = cos(theta), S2 = sin(theta) cos(q phi), S3 = sin(theta) sin(q phi)
        #   |Ex|^2 + |Ey|^2 = 1 everywhere (no pole at Ex = 0).
        Ex = np.cos(half).astype(np.complex128)
        Ey = (np.sin(half) * np.exp(1j * q * phi)).astype(np.complex128)
        return Ex, Ey

    def _custom_jones(self, x, y):
        if self.cfg.jones_fn is None:
            raise ValueError("jones_fn required for CUSTOM_JONES")
        Ex, Ey = self.cfg.jones_fn(x, y)
        return np.asarray(Ex, dtype=np.complex128), np.asarray(Ey, dtype=np.complex128)

    def _unpolarized(self, x, y):
        # Model as incoherent sum: return equal-intensity random phase components
        # Represented as a coherent Jones vector with random phase (one realization)
        rng = np.random.default_rng(42)
        phase = rng.uniform(0, 2 * np.pi, x.shape)
        Ex = (1.0 / np.sqrt(2.0)) * np.exp(1j * phase)
        Ey = (1.0 / np.sqrt(2.0)) * np.exp(1j * (phase + np.pi / 2))
        return Ex, Ey


# -----------------------------------------------------------------------
# Topological charge / invariants
# ----------------------------------------------------------------------


@dataclass
class TopologicalChargeConfig:
    """Configuration for topological structures.

    charge_type : str  # "winding", "skyrmion", "oam", "polarization"
    charge : int  # integer topological charge
    core_radius : float  # vortex core radius
    """

    charge_type: str = "winding"
    charge: int = 1
    core_radius: float = 0.1


class TopologicalCharge:
    """Impose topological phase singularities on a field.

    Supports scalar vortices (exp(i*l*phi)) and polarization skyrmion
    charges. Computes winding numbers from phase gradients.
    """

    def __init__(self, config: TopologicalChargeConfig):
        self.cfg = config

    def apply_phase_singularity(
        self,
        field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Multiply field by exp(i * charge * phi) to imprint a vortex.

        phi is broadcast against the field, so the vortex can be applied to
        2D fields as well as (ny, nx, nt) or (nz, ny, nx, nt) volumes.
        """
        phi = np.arctan2(y, x)
        # Append trailing axes so phi broadcasts over z/t dimensions.
        while phi.ndim < field.ndim:
            phi = phi[..., np.newaxis]
        return field * np.exp(1j * self.cfg.charge * phi)

    def compute_winding_number(
        self,
        phase: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute the winding number of a phase field around the origin.

        Uses the contour integral: (1/2pi) * ∮ ∇phi · dl
        Approximated by summing phase differences around a closed loop.
        """
        phase = np.asarray(phase, dtype=np.float64)
        # Unwrap phase to avoid 2pi jumps
        unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1)
        # Compute gradient magnitude around a circular contour at r = core_radius
        r = np.sqrt(x ** 2 + y ** 2)
        # Find points near a contour
        contour_mask = np.abs(r - self.cfg.core_radius) < 0.1 * (x[0, 1] - x[0, 0]) if x.shape[1] > 1 else np.abs(r - self.cfg.core_radius) < 0.1
        if not np.any(contour_mask):
            return 0.0
        # Simple approach: count phase wraps around origin
        phi = np.arctan2(y, x)
        dphi = np.gradient(phi, axis=1)
        dphase = np.gradient(unwrapped, axis=1)
        # Winding = (1/2pi) * integral of dphase along azimuth
        # Approximate by mean azimuthal phase gradient * 2pi*r / 2pi
        avg_dphi_dphase = np.mean(dphase[contour_mask] / (dphi[contour_mask] + 1e-15))
        winding = avg_dphi_dphase / (2.0 * np.pi) * 2.0 * np.pi
        return float(round(winding))

    def compute_skyrmion_number(
        self,
        Ex: np.ndarray,
        Ey: np.ndarray,
    ) -> float:
        """Compute the skyrmion number from a polarization texture.

        Q = (1/4pi) ∫ n · (∂x n × ∂y n) dx dy
        where n is the normalized Stokes vector.
        """
        # Stokes parameters
        S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
        S0 = np.where(S0 == 0, 1e-15, S0)
        S1 = (np.abs(Ex) ** 2 - np.abs(Ey) ** 2) / S0
        S2 = (2.0 * np.real(Ex * np.conj(Ey))) / S0
        S3 = (2.0 * np.imag(Ex * np.conj(Ey))) / S0

        n = np.stack([S1, S2, S3], axis=0)
        # Normalize
        n_norm = np.linalg.norm(n, axis=0)
        n_norm = np.where(n_norm == 0, 1.0, n_norm)
        n = n / n_norm

        # Skyrmion number via the gauge-invariant lattice solid-angle
        # (plaquette) method. For each elementary plaquette the solid angle
        # subtended by the four n-vectors is summed; Q = (1/4pi) * sum Omega.
        # This is far more accurate than a finite-difference gradient product
        # (which under-resolves high topological charge and is sensitive to
        # branch cuts), and it returns an exactly integer charge for a
        # lattice skyrmion. Vectorized over all plaquettes.
        n00 = n[:, :-1, :-1]
        n01 = n[:, :-1, 1:]
        n11 = n[:, 1:, 1:]
        n10 = n[:, 1:, :-1]

        def _solid_angle(a, b, c):
            num = np.sum(a * np.cross(b, c, axis=0), axis=0)
            den = 1.0 + np.sum(a * b, axis=0) + np.sum(b * c, axis=0) + np.sum(c * a, axis=0)
            return 2.0 * np.arctan2(num, den)

        # Triangle (n00, n01, n11) + triangle (n00, n11, n10). The vertex
        # ordering below yields Q = +q for a skyrmion of charge q.
        omega = _solid_angle(n00, n11, n01) + _solid_angle(n00, n10, n11)
        Q = np.sum(omega) / (4.0 * np.pi)
        return float(Q)


# -----------------------------------------------------------------------
# Coherence model
# ----------------------------------------------------------------------


@dataclass
class CoherenceConfig:
    """Configuration for partial coherence.

    coherence_type : str  # "full", "partial", "incoherent"
    coherence_time : float  # temporal coherence time
    coherence_length : float  # spatial coherence length
    spectral_width : float  # bandwidth (rad/s)
    """

    coherence_type: str = "full"
    coherence_time: float = 1.0
    coherence_length: float = 1.0
    spectral_width: float = 0.1


class CoherenceModel:
    """Model partial coherence via the mutual coherence function.

    For partial coherence, the field is represented as an ensemble average
    over realizations with random phase fluctuations whose correlation
    is governed by coherence_time and coherence_length.
    """

    def __init__(self, config: CoherenceConfig):
        self.cfg = config

    def apply_coherence(
        self,
        field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        n_realizations: int = 4,
    ) -> np.ndarray:
        """Return a coherence-averaged intensity field.

        For "full" coherence returns |field|^2.
        For "partial", averages |sum of phase-shifted copies|^2.
        """
        if self.cfg.coherence_type == "full":
            return np.abs(field) ** 2

        field = np.asarray(field, dtype=np.complex128)
        rng = np.random.default_rng(12345)

        # Generate phase screens with spatial/temporal correlation
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        dx = x[0, 1] - x[0, 0] if x.shape[1] > 1 else 1.0
        dy = y[1, 0] - y[0, 0] if y.shape[0] > 1 else 1.0

        avg_intensity = np.zeros_like(field, dtype=np.float64)
        for _ in range(n_realizations):
            # Random phase screen with Gaussian correlation
            phase_screen = rng.standard_normal(field.shape)
            # Simple Gaussian low-pass filter in frequency domain
            from scipy.ndimage import gaussian_filter

            sigma_t = max(1, self.cfg.coherence_time / dt / 2.0)
            sigma_x = max(1, self.cfg.coherence_length / dx / 2.0)
            sigma_y = max(1, self.cfg.coherence_length / dy / 2.0)
            phase_screen = gaussian_filter(phase_screen, sigma=(sigma_x, sigma_y, sigma_t))
            phase_screen = phase_screen - np.mean(phase_screen)

            shifted = field * np.exp(1j * phase_screen)
            avg_intensity += np.abs(shifted) ** 2

        return avg_intensity / n_realizations


# -----------------------------------------------------------------------
# Coupled modes
# ----------------------------------------------------------------------


@dataclass
class CoupledModeConfig:
    """Configuration for coupled-mode interactions.

    coupling_coefficient : float  # kappa (1/length)
    interaction_length : float  # z over which coupling acts
    mode1_index : int
    mode2_index : int
    phase_mismatch : float  # Delta k
    """

    coupling_coefficient: float = 0.1
    interaction_length: float = 1.0
    mode1_index: int = 0
    mode2_index: int = 1
    phase_mismatch: float = 0.0


class CoupledMode:
    """Coupled-mode theory for two interacting field components.

    Solves:
    dA1/dz = i*kappa*A2*exp(i*Delta*k*z)
    dA2/dz = i*kappa*A1*exp(-i*Delta*k*z)

    Analytic solution used for the envelope evolution.
    """

    def __init__(self, config: CoupledModeConfig):
        self.cfg = config

    def evolve(
        self,
        A1: np.ndarray,
        A2: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve two coupled modes along z.

        A1, A2 : input complex envelopes (broadcastable to z shape).
        z : axial coordinate array.
        Returns (A1_out, A2_out) at each z.
        """
        z = np.asarray(z, dtype=np.float64)
        kappa = self.cfg.coupling_coefficient
        dk = self.cfg.phase_mismatch

        # General coupled-mode solution
        # Effective coupling: Omega = sqrt(kappa^2 + (dk/2)^2)
        Omega = np.sqrt(kappa ** 2 + (dk / 2.0) ** 2)

        if Omega == 0:
            # No coupling, no mismatch: trivial
            return A1.copy(), A2.copy()

        # Power transfer: sin^2(Omega*z) * kappa^2 / Omega^2
        transfer = (kappa ** 2 / Omega ** 2) * np.sin(Omega * z) ** 2
        # Phase factors
        phase = np.exp(1j * dk * z / 2.0)

        # Broadcast A1, A2 (any shape) against z (1D): append z axis
        A1 = np.asarray(A1, dtype=np.complex128)
        A2 = np.asarray(A2, dtype=np.complex128)
        # transfer, phase: shape (Nz,) -> (1, ..., 1, Nz)
        broadcast_shape = (1,) * A1.ndim + (len(z),)
        transfer_b = transfer.reshape(broadcast_shape)
        phase_b = phase.reshape(broadcast_shape)
        A1_b = A1[..., np.newaxis]
        A2_b = A2[..., np.newaxis]

        A1_out = A1_b * (1.0 - transfer_b) + A2_b * transfer_b * phase_b
        A2_out = A2_b * (1.0 - transfer_b) + A1_b * transfer_b * np.conj(phase_b)

        return A1_out, A2_out

