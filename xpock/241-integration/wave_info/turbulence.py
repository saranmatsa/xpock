"""
XPOCK — HX-241 Independent Research: Turbulence Propagation Model

Physically-based atmospheric turbulence model for simulating wave propagation
through free-space optical links. Uses Kolmogorov/von Karman spectrum for
refractive index fluctuations and split-step Fourier propagation.

This is XPOCK research only — no HX-241 involvement. All simulations are
mathematical models; results do not establish physical feasibility.

Design:
- Phase screen generation from von Karman spectrum Φₙ(κ) = 0.033·Cₙ²·κ^(-11/3)·exp(-κ²/κₘ²)
- Split-step Fourier propagation: alternate free-space (Fresnel) and phase screen
- Phase in radians: φ = 2π·OPD/λ where OPD is optical path difference
- Supports multi-screen propagation along z for accumulated turbulence

All arrays use (ny, nx) shape convention matching XPOCK WaveField.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


@dataclass
class TurbulenceConfig:
    """Configuration for atmospheric turbulence simulation.

    Attributes
    ----------
    cn2 : float
        Refractive index structure constant in m^(-2/3). Typical values:
        1e-17 (very weak), 1e-15 (weak), 1e-13 (moderate), 1e-12 (strong),
        1e-11 (very strong).
    wavelength : float
        Wavelength in meters.
    distance : float
        Propagation distance in meters.
    inner_scale : float
        Inner scale l0 in meters (typically 5-10 mm).
    outer_scale : float
        Outer scale L0 in meters (typically 10 m).
    n_screens : int
        Number of phase screens along propagation path. More screens =
        more accurate but slower.
    seed : int, optional
        Random seed for reproducibility.
    """

    cn2: float = 1e-14
    wavelength: float = 1.55e-6  # 1550 nm
    distance: float = 100.0  # meters
    inner_scale: float = 5e-3  # 5 mm
    outer_scale: float = 10.0  # 10 m
    n_screens: int = 10
    seed: Optional[int] = None


@dataclass
class TurbulenceResult:
    """Result of turbulence propagation."""

    input_field: np.ndarray
    output_field: np.ndarray
    phase_screens: list
    config: TurbulenceConfig
    rytov_variance: float
    scintillation_index: float


class TurbulenceModel:
    """Physically-based atmospheric turbulence model using Kolmogorov/von Karman
    spectrum and split-step Fourier propagation.

    Generates random phase screens from the von Karman refractive index
    spectrum and applies them alternately with free-space (Fresnel)
    propagation to simulate wave propagation through atmospheric turbulence.
    """

    def __init__(self, config: TurbulenceConfig):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)

    def generate_phase_screen(
        self, shape: Tuple[int, int], dx: float, dz: float
    ) -> np.ndarray:
        """Generate a single Kolmogorov turbulence phase screen.

        Parameters
        ----------
        shape : (ny, nx)
            Grid shape in (y, x) order.
        dx : float
            Spatial sampling interval in meters (assumed square pixels).
        dz : float
            Propagation distance for this screen in meters.

        Returns
        -------
        phase : (ny, nx) float array
            Phase screen in radians.
        """
        ny, nx = shape
        kx = 2.0 * np.pi * fftfreq(nx, dx)
        ky = 2.0 * np.pi * fftfreq(ny, dy if False else dx)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)

        # Von Karman spectrum: Φₙ(κ) = 0.033·Cₙ²·κ^(-11/3)·exp(-κ²/κₘ²)
        # Cut off at inner scale κₘ = 5.92/l₀ and outer scale κ₀ = 2π/L₀.
        #
        # The phase variance accumulated through a slice of thickness dz scales
        # linearly with dz (OPD variance ∝ Cₙ²·dz). Scaling the spectrum by dz
        # makes each phase screen represent its slice thickness, so the TOTAL
        # turbulence over n_screens screens is n_screens·dz = L, independent of
        # the screen count. Without this, every screen carries full-path strength
        # and turbulence is over-counted by a factor of n_screens.
        km = 5.92 / self.cfg.inner_scale
        k0 = 2.0 * np.pi / self.cfg.outer_scale

        # Avoid division by zero at K=0
        K_safe = np.where(K == 0, 1.0, K)
        phi_n = (
            0.033
            * self.cfg.cn2
            * dz
            * K_safe ** (-11.0 / 3.0)
            * np.exp(-(K_safe / km) ** 2)
        )
        # Zero out modes below outer scale
        phi_n = np.where(K < k0, 0.0, phi_n)

        # Generate random complex screen with this spectrum
        # Each Fourier component has random phase and amplitude sqrt(Phi * dkx * dky)
        dkx = kx[1] - kx[0] if nx > 1 else 1.0
        dky = ky[1] - ky[0] if ny > 1 else 1.0
        dk_area = dkx * dky

        rng = self._rng
        phase_random = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal(
            (ny, nx)
        )
        phase_screen_ft = phase_random * np.sqrt(phi_n * dk_area)

        # Inverse FFT to get OPD screen in meters
        opd_screen = np.real(ifft2(phase_screen_ft))

        # Convert OPD (meters) to phase (radians): phi = 2π·OPD/λ
        phase_screen = 2.0 * np.pi * opd_screen / self.cfg.wavelength

        # --- Normalize phase screen to theoretical von Karman variance ---
        # The ifft2 divides by N², drastically reducing amplitude.
        # Theoretical phase variance: σ_φ² = (2π/λ)² · ∫ Φₙ(κ) d²κ
        # Discrete: ∫ Φₙ d²κ ≈ Σ Φₙ[k] · Δκx · Δκy = Σ phi_n · dk_area
        k_area_integral = float(np.sum(phi_n) * dk_area)
        if k_area_integral > 0:
            sigma_phi_theoretical = (
                2.0 * np.pi / self.cfg.wavelength
            ) * np.sqrt(k_area_integral)
            sigma_phi_raw = float(np.std(phase_screen))
            if sigma_phi_raw > 1e-15:
                phase_screen = phase_screen * (
                    sigma_phi_theoretical / sigma_phi_raw
                )

        return phase_screen

    def propagate(
        self, field: np.ndarray, x: np.ndarray, y: np.ndarray,
        phase_screens: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Propagate complex field through turbulent atmosphere.

        Uses split-step Fourier method: alternate free-space (Fresnel)
        propagation over dz/2 with phase screen application, repeated
        n_screens times.

        Parameters
        ----------
        field : (ny, nx) complex array
            Input complex field at z=0.
        x : (nx,) or (ny, nx)
            x coordinates.
        y : (ny,) or (ny, nx)
            y coordinates.
        phase_screens : list of (ny, nx) arrays, optional
            Pre-generated phase screens, one per propagation step.
            If None, a new random screen is generated for each step.
            The same screen list must be used for Ex and Ey to preserve
            polarization relationships.

        Returns
        -------
        field_out : (ny, nx) complex array
            Complex field after propagation through turbulence.
        """
        # Ensure 2D arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y, indexing="xy")
        else:
            X, Y = x, y

        field = np.asarray(field, dtype=np.complex128)
        ny, nx = field.shape
        dx = x[0, 1] - x[0, 0] if x.ndim == 2 else x[1] - x[0]
        dy = y[1, 0] - y[0, 0] if y.ndim == 2 else y[1] - y[0]
        # Use average dx,dy for isotropic assumption
        d = (dx + dy) / 2.0

        dz = self.cfg.distance / self.cfg.n_screens
        k = 2.0 * np.pi / self.cfg.wavelength

        # Free-space propagation kernel (Fresnel approximation)
        # H(kx, ky) = exp(-i * (kx² + ky²) * dz / (2*k))
        kx_arr = 2.0 * np.pi * fftfreq(nx, d)
        ky_arr = 2.0 * np.pi * fftfreq(ny, d)
        KX, KY = np.meshgrid(kx_arr, ky_arr)
        K2 = KX**2 + KY**2
        H = np.exp(-1j * K2 * dz / (2.0 * k))

        E = field.copy()
        for step in range(self.cfg.n_screens):
            # Half-step free-space propagation
            E_hat = fft2(E)
            E_hat = E_hat * H
            E = ifft2(E_hat)

            # Apply phase screen
            if phase_screens is not None:
                phi = phase_screens[step]
            else:
                phi = self.generate_phase_screen((ny, nx), d, dz)
            E = E * np.exp(1j * phi)

            # Second half-step free-space propagation
            E_hat = fft2(E)
            E_hat = E_hat * H
            E = ifft2(E_hat)

        return E

    def compute_rytov_variance(self) -> float:
        """Compute Rytov variance σ_R² = 1.23·Cₙ²·k^(7/6)·L^(11/6).

        Weak turbulence: σ_R² < 1
        Strong turbulence: σ_R² > 1
        """
        k = 2.0 * np.pi / self.cfg.wavelength
        return 1.23 * self.cfg.cn2 * k ** (7.0 / 6.0) * self.cfg.distance ** (
            11.0 / 6.0
        )

    def propagate_with_metrics(
        self, field: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> TurbulenceResult:
        """Propagate and return full result with metrics."""
        cfg = self.cfg
        output = self.propagate(field, x, y)

        # Compute scintillation index: σ_I² = <I²>/<I>² - 1
        intensity_out = np.abs(output) ** 2
        mean_I = np.mean(intensity_out)
        if mean_I > 0:
            scintillation = np.mean(intensity_out**2) / (mean_I**2) - 1.0
        else:
            scintillation = 0.0

        return TurbulenceResult(
            input_field=field,
            output_field=output,
            phase_screens=[],  # Not stored to save memory
            config=cfg,
            rytov_variance=self.compute_rytov_variance(),
            scintillation_index=float(scintillation),
        )


def quick_turbulence_test():
    """Quick sanity check: turbulence should broaden and distort the field."""
    nx, ny = 64, 64
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Gaussian beam
    w0 = 1.0
    E_in = np.exp(-(X**2 + Y**2) / w0**2).astype(np.complex128)

    cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=100.0,
        cn2=1e-13,  # moderate turbulence
        n_screens=10,
        seed=42,
    )
    model = TurbulenceModel(cfg)
    result = model.propagate_with_metrics(E_in, X, Y)

    print(f"Rytov variance: {result.rytov_variance:.4f}")
    print(f"Scintillation index: {result.scintillation_index:.4f}")
    print(f"Input power: {np.sum(np.abs(E_in)**2):.4f}")
    print(f"Output power: {np.sum(np.abs(result.output_field)**2):.4f}")
    print(f"Power conserved: {np.isclose(np.sum(np.abs(E_in)**2), np.sum(np.abs(result.output_field)**2), rtol=1e-3)}")

    return result


if __name__ == "__main__":
    quick_turbulence_test()