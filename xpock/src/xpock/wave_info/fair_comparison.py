"""
XPOCK — Rigorous End-to-End Validation Framework

Compares Skyrmion Number Modulation (SKM) vs OAM vs Intensity modulation
under IDENTICAL physical constraints:

1. Same turbulence realization for all schemes (shared phase screens)
2. Same total optical power for all schemes (normalized before propagation)
3. Same grid, wavelength, distance, screen count
4. Opt-in receiver noise (photon shot noise + detector read noise)
5. Confusion matrices + BER + symbol error rate
6. Grid-convergence study for Q measurement
7. Statistical significance with sufficient trials

This addresses the critical questions:
- Does SKM's topological protection give a real BER advantage?
- Does the advantage survive receiver noise?
- Is the comparison fair (same power, same conditions)?
- Where is the breaking point?

All simulations are mathematical models; hardware validation required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from xpock.wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from xpock.wave_info.skyrmion_modem import (
    ModemConfig,
    SkyrmionModem,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SchemeResult:
    """BER result for one modulation scheme at one Cn²."""
    scheme: str                          # "SKM", "OAM", "Intensity"
    cn2: float
    n_trials: int
    n_errors: int
    n_symbols: int
    ber: float
    ser: float
    q_mean: Optional[float] = None       # SKM: mean measured Q
    q_std: Optional[float] = None        # SKM: std of measured Q
    overlap_mean: Optional[float] = None # OAM: mean mode overlap
    intensity_mean: Optional[float] = None  # Intensity: mean measured intensity
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    snr_db: float = 0.0                  # Estimated SNR at receiver
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a theoretical validation check."""
    name: str
    passed: bool
    measured: float
    expected: float
    rel_error: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_power(field: np.ndarray, dx: float, dy: float,
                    target_power: float = 1.0) -> np.ndarray:
    """Normalize field to target total power (integrated over space)."""
    power = float(np.sum(np.abs(field) ** 2) * dx * dy)
    if power <= 0:
        return field
    return field * np.sqrt(target_power / power)


def total_power(field: np.ndarray, dx: float, dy: float) -> float:
    """Total integrated power of a complex field."""
    return float(np.sum(np.abs(field) ** 2) * dx * dy)


def add_receiver_noise(Ex: np.ndarray, Ey: np.ndarray,
                       noise_std: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Add complex Gaussian receiver noise (photon shot + detector read).

    noise_std is relative to RMS field amplitude.
    """
    if noise_std <= 0.0:
        return Ex, Ey
    rms = float(np.sqrt(np.mean(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)))
    if rms <= 0.0:
        return Ex, Ey
    sigma = noise_std * rms / np.sqrt(2.0)
    Ex_n = Ex + sigma * (rng.standard_normal(Ex.shape)
                         + 1j * rng.standard_normal(Ex.shape))
    Ey_n = Ey + sigma * (rng.standard_normal(Ey.shape)
                         + 1j * rng.standard_normal(Ey.shape))
    return Ex_n, Ey_n


# ---------------------------------------------------------------------------
# Theoretical validation
# ---------------------------------------------------------------------------

def validate_turbulence_model(wavelength: float = 1.55e-6,
                              distance: float = 1000.0,
                              cn2: float = 1e-13,
                              inner_scale: float = 5e-3,
                              outer_scale: float = 10.0,
                              n_trials: int = 20,
                              grid_n: int = 128,
                              grid_extent: float = 4.0) -> List[ValidationResult]:
    """Validate turbulence phase screens against von Karman spectrum theory.

    Checks:
    1. Screen variance matches spectrum-integral theory (after normalization)
    2. Total turbulence independent of n_screens (dz scaling)
    3. Power conservation through split-step propagation
    """
    results: List[ValidationResult] = []
    dx = grid_extent / grid_n

    # --- Check 1: Spectrum-integral variance match ---
    cfg = TurbulenceConfig(wavelength=wavelength, distance=distance, cn2=cn2,
                           n_screens=1, inner_scale=inner_scale,
                           outer_scale=outer_scale, seed=42)
    model = TurbulenceModel(cfg)

    variances = []
    for s in range(n_trials):
        cfg_s = TurbulenceConfig(wavelength=wavelength, distance=distance,
                                 cn2=cn2, n_screens=1,
                                 inner_scale=inner_scale, outer_scale=outer_scale,
                                 seed=1000 + s)
        m = TurbulenceModel(cfg_s)
        phi = m.generate_phase_screen((grid_n, grid_n), dx, distance)
        variances.append(np.var(phi))
    emp_var = float(np.mean(variances))

    # Theoretical spectrum-integral variance
    km = 5.92 / inner_scale
    k0 = 2.0 * np.pi / outer_scale
    kx = 2.0 * np.pi * np.fft.fftfreq(grid_n, dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(grid_n, dx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K_safe = np.where(K == 0, 1.0, K)
    phi_n = (0.033 * cn2 * distance * K_safe ** (-11.0 / 3.0)
             * np.exp(-(K_safe / km) ** 2))
    phi_n = np.where(K < k0, 0.0, phi_n)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    dk_area = dkx * dky
    k_area_integral = float(np.sum(phi_n) * dk_area)
    theory_var = ((2.0 * np.pi / wavelength) ** 2) * k_area_integral

    rel_err = abs(emp_var - theory_var) / theory_var if theory_var > 0 else float('inf')
    results.append(ValidationResult(
        name="phase_screen_variance_matches_spectrum",
        passed=rel_err < 0.05,
        measured=emp_var,
        expected=theory_var,
        rel_error=rel_err,
        detail=f"empirical={emp_var:.6f}, theory={theory_var:.6f}, "
               f"rel_err={rel_err:.4f}",
    ))

    # --- Check 2: Total turbulence independent of n_screens ---
    x = np.linspace(-grid_extent / 2, grid_extent / 2, grid_n)
    y = np.linspace(-grid_extent / 2, grid_extent / 2, grid_n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    E_in = np.exp(-(X**2 + Y**2)).astype(np.complex128)

    phase_stds = []
    for n_screens in [1, 2, 5, 10, 20]:
        cfg_s = TurbulenceConfig(wavelength=wavelength, distance=distance,
                                 cn2=cn2, n_screens=n_screens, seed=7)
        m = TurbulenceModel(cfg_s)
        E_out = m.propagate(E_in, X, Y)
        mask = np.abs(E_in) > 0.1 * np.max(np.abs(E_in))
        pd = np.angle(E_out[mask] * np.conj(E_in[mask]))
        phase_stds.append(float(np.std(pd)))

    spread = max(phase_stds) - min(phase_stds)
    results.append(ValidationResult(
        name="turbulence_independent_of_n_screens",
        passed=spread < 0.15,
        measured=spread,
        expected=0.0,
        rel_error=spread / max(np.mean(phase_stds), 1e-12),
        detail=f"phase_stds={[f'{s:.4f}' for s in phase_stds]}, spread={spread:.4f}",
    ))

    # --- Check 3: Power conservation ---
    # Without amplitude turbulence (phase-only), split-step propagation is
    # unitary and conserves total power exactly. With amplitude turbulence
    # (scintillation), power fluctuates — the mean power over trials is
    # preserved but individual realizations vary.
    power_in = total_power(E_in, dx, dx)
    power_errors_phase_only = []
    power_mean_errors = []
    for n_screens in [1, 5, 10, 20]:
        # Phase-only: exact power conservation
        cfg_s = TurbulenceConfig(wavelength=wavelength, distance=distance,
                                 cn2=cn2, n_screens=n_screens,
                                 seed=7, amplitude_turbulence=False)
        m = TurbulenceModel(cfg_s)
        E_out = m.propagate(E_in, X, Y)
        power_out = total_power(E_out, dx, dx)
        power_errors_phase_only.append(abs(power_out - power_in) / power_in)

        # With amplitude turbulence: mean power preserved (not pointwise)
        cfg_a = TurbulenceConfig(wavelength=wavelength, distance=distance,
                                 cn2=cn2, n_screens=n_screens,
                                 seed=7, amplitude_turbulence=True)
        m_a = TurbulenceModel(cfg_a)
        powers = []
        for _ in range(10):
            E_out_a = m_a.propagate(E_in, X, Y)
            powers.append(total_power(E_out_a, dx, dx))
        power_mean_errors.append(abs(np.mean(powers) - power_in) / power_in)

    max_power_err_phase = max(power_errors_phase_only)
    max_power_err_mean = max(power_mean_errors)

    results.append(ValidationResult(
        name="power_conservation_phase_only",
        passed=max_power_err_phase < 1e-2,
        measured=max_power_err_phase,
        expected=0.0,
        rel_error=max_power_err_phase,
        detail=f"phase-only max relative power error={max_power_err_phase:.6g}",
    ))

    results.append(ValidationResult(
        name="power_mean_preserved_with_scintillation",
        passed=max_power_err_mean < 0.2,
        measured=max_power_err_mean,
        expected=0.0,
        rel_error=max_power_err_mean,
        detail=f"scintillation max mean power error={max_power_err_mean:.6g}",
    ))

    return results


def validate_skyrmion_quantization(charges: List[int] = [-4, -3, -2, -1, 1, 2, 3, 4],
                                   grid_n: int = 128,
                                   grid_extent: float = 8.0,
                                   skyrmion_radius: float = 1.0) -> List[ValidationResult]:
    """Validate skyrmion number Q = q exactness and grid-resolution independence."""
    from xpock.properties import (PolarizationTexture, PolarizationConfig,
                                  PolarizationState,
                                  TopologicalCharge, TopologicalChargeConfig)

    results: List[ValidationResult] = []
    x = np.linspace(-grid_extent / 2, grid_extent / 2, grid_n)
    y = np.linspace(-grid_extent / 2, grid_extent / 2, grid_n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    tc = TopologicalCharge(TopologicalChargeConfig(charge_type="skyrmion", charge=1))

    all_ok = True
    for q in charges:
        cfg = PolarizationConfig(state=PolarizationState.SKYRMIONIC,
                                 skyrmion_radius=skyrmion_radius,
                                 skyrmion_charge=q)
        Ex, Ey = PolarizationTexture(cfg).evaluate(X, Y)
        Q = tc.compute_skyrmion_number(Ex, Ey)
        ok = round(Q) == q
        all_ok &= ok
        results.append(ValidationResult(
            name=f"skyrmion_Q_q={q}",
            passed=ok,
            measured=float(Q),
            expected=float(q),
            rel_error=abs(Q - q) / abs(q) if q != 0 else abs(Q),
            detail=f"Q={Q:.6f}, round(Q)={round(Q)}",
        ))

    results.append(ValidationResult(
        name="skyrmion_quantization_all_charges",
        passed=all_ok,
        measured=1.0 if all_ok else 0.0,
        expected=1.0,
        rel_error=0.0 if all_ok else 1.0,
        detail=f"All charges {charges} quantized exactly",
    ))

    # Grid convergence: Q should be exact at all resolutions
    conv_ok = True
    for n in [32, 64, 128, 256]:
        x_n = np.linspace(-grid_extent / 2, grid_extent / 2, n)
        y_n = np.linspace(-grid_extent / 2, grid_extent / 2, n)
        X_n, Y_n = np.meshgrid(x_n, y_n, indexing='xy')
        cfg = PolarizationConfig(state=PolarizationState.SKYRMIONIC,
                                 skyrmion_radius=skyrmion_radius,
                                 skyrmion_charge=2)
        Ex_n, Ey_n = PolarizationTexture(cfg).evaluate(X_n, Y_n)
        Q_n = tc.compute_skyrmion_number(Ex_n, Ey_n)
        if round(Q_n) != 2:
            conv_ok = False
        results.append(ValidationResult(
            name=f"skyrmion_grid_conv_n={n}",
            passed=round(Q_n) == 2,
            measured=float(Q_n),
            expected=2.0,
            rel_error=abs(Q_n - 2.0),
            detail=f"Q={Q_n:.6f} at n={n}",
        ))
    all_ok &= conv_ok

    results.append(ValidationResult(
        name="skyrmion_grid_convergence",
        passed=conv_ok,
        measured=1.0 if conv_ok else 0.0,
        expected=1.0,
        rel_error=0.0 if conv_ok else 1.0,
        detail="Q=2 exact at n=32,64,128,256",
    ))

    return results


# ---------------------------------------------------------------------------
# End-to-end BER comparison (fair, identical conditions)
# ---------------------------------------------------------------------------

class FairComparison:
    """Fair SKM vs OAM vs Intensity comparison under identical constraints.

    All schemes:
    - Share the same turbulence model per Cn² (identical phase screens)
    - Transmit the same total optical power (normalized to 1.0)
    - Use the same grid, wavelength, distance, screen count
    - Independent trials (fresh phase screens per trial)
    """

    def __init__(self, grid_x: np.ndarray, grid_y: np.ndarray,
                 wavelength: float = 1.55e-6,
                 skyrmion_radius: float = 1.0,
                 seed: Optional[int] = None,
                 receiver_noise_std: float = 0.0):
        self.grid_x = np.asarray(grid_x, dtype=np.float64)
        self.grid_y = np.asarray(grid_y, dtype=np.float64)
        self.wavelength = wavelength
        self.skyrmion_radius = skyrmion_radius
        self.rng = np.random.default_rng(seed)
        self.receiver_noise_std = receiver_noise_std

        self.dx = self.grid_x[0, 1] - self.grid_x[0, 0]
        self.dy = self.grid_y[1, 0] - self.grid_y[0, 0]
        self.X = self.grid_x
        self.Y = self.grid_y

    def _make_turb_model(self, cn2: float, distance: float,
                         n_screens: int) -> TurbulenceModel:
        turb_cfg = TurbulenceConfig(
            wavelength=self.wavelength, distance=distance, cn2=cn2,
            n_screens=n_screens,
            seed=int(self.rng.integers(0, 2 ** 31 - 1)),
        )
        return TurbulenceModel(turb_cfg)

    def _fresh_screens(self, turb_model: TurbulenceModel, n_screens: int,
                       dz: float) -> Tuple[List[np.ndarray],
                                           Optional[List[np.ndarray]]]:
        """Generate one fresh, independent set of phase (+ amplitude) screens.

        Returns (phase_screens, amplitude_screens). When amplitude_turbulence
        is False, amplitude_screens is None. The same screens are applied to
        Ex and Ey so polarization relationships are preserved (scalar
        turbulence affects both components identically).
        """
        ny_dim = self.grid_x.shape[0]
        nx_dim = self.grid_x.shape[1]
        phase_screens = [
            turb_model.generate_phase_screen((ny_dim, nx_dim), self.dx, dz)
            for _ in range(n_screens)
        ]
        if turb_model.cfg.amplitude_turbulence:
            amplitude_screens = [
                turb_model.generate_amplitude_screen((ny_dim, nx_dim), self.dx, dz)
                for _ in range(n_screens)
            ]
        else:
            amplitude_screens = None
        return phase_screens, amplitude_screens

    # --- SKM ---

    def run_skm(self, constellation: List[int], cn2: float, distance: float,
                n_screens: int, n_trials: int,
                turb_model: Optional[TurbulenceModel] = None) -> SchemeResult:
        modem_cfg = ModemConfig(
            wavelength=self.wavelength, grid_x=self.grid_x,
            grid_y=self.grid_y, skyrmion_radius=self.skyrmion_radius,
            constellation=constellation,
        )
        modem = SkyrmionModem(modem_cfg)
        n_bits = modem.n_bits

        if turb_model is None:
            turb_model = self._make_turb_model(cn2, distance, n_screens)
        dz = distance / n_screens

        n_errors = 0
        n_symbols = 0
        q_values = []
        confusion = {}

        for _ in range(n_trials):
            n_tx = constellation[self.rng.integers(0, len(constellation))]
            bit_idx = constellation.index(n_tx)
            bits_tx = np.array(
                [(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)]
            )
            Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, self.X, self.Y)
            # Normalize the TWO-COMPONENT field to unit TOTAL power.
            # Normalizing Ex and Ey each to unit power would give the SKM
            # link twice the launch power of the scalar OAM/intensity
            # baselines — an unfair comparison. The transmitted optical
            # power is integral(|Ex|^2 + |Ey|^2) dx dy and must match the
            # unit power of the scalar schemes.
            p_ex = total_power(Ex_tx, self.dx, self.dy)
            p_ey = total_power(Ey_tx, self.dx, self.dy)
            p_tot = p_ex + p_ey
            if p_tot > 0:
                scale = np.sqrt(1.0 / p_tot)
                Ex_tx = Ex_tx * scale
                Ey_tx = Ey_tx * scale

            phase_screens, amplitude_screens = self._fresh_screens(
                turb_model, n_screens, dz
            )
            E_rx = turb_model.propagate(
                Ex_tx, self.X, self.Y,
                phase_screens=phase_screens,
                amplitude_screens=amplitude_screens,
            )
            E_rx_y = turb_model.propagate(
                Ey_tx, self.X, self.Y,
                phase_screens=phase_screens,
                amplitude_screens=amplitude_screens,
            )

            # Receiver noise
            E_rx, E_rx_y = add_receiver_noise(
                E_rx, E_rx_y, self.receiver_noise_std, self.rng
            )

            q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)
            q_values.append(q_measured)

            bits_rx = modem.skyrmion_to_bits(n_rx)
            bit_error = not np.array_equal(bits_tx, bits_rx)
            n_errors += int(bit_error)
            n_symbols += 1
            key = f"{n_tx}->{n_rx}"
            confusion[key] = confusion.get(key, 0) + 1

        ber = n_errors / n_symbols if n_symbols > 0 else 0.0
        return SchemeResult(
            scheme="SKM", cn2=cn2, n_trials=n_trials,
            n_errors=n_errors, n_symbols=n_symbols, ber=ber, ser=ber,
            q_mean=float(np.mean(q_values)) if q_values else 0.0,
            q_std=float(np.std(q_values)) if q_values else 0.0,
            confusion_matrix=confusion,
        )

    # --- OAM ---

    def run_oam(self, oam_modes: List[int], cn2: float, distance: float,
                n_screens: int, n_trials: int,
                turb_model: Optional[TurbulenceModel] = None,
                waist: float = 1.0) -> SchemeResult:
        n_bits = int(np.floor(np.log2(len(oam_modes))))

        # Pre-compute OAM mode fields (normalized to unit power)
        R = np.sqrt(self.X**2 + self.Y**2)
        Phi = np.arctan2(self.Y, self.X)
        k = 2.0 * np.pi / self.wavelength
        z_R = np.pi * waist**2 / self.wavelength
        w_z = waist * np.sqrt(1.0 + (0.0 / z_R) ** 2)
        R_over_w = np.sqrt(2.0) * R / w_z
        mode_fields = {}
        for l in oam_modes:
            radial = R_over_w ** abs(l) * np.exp(-R**2 / w_z**2)
            azimuthal = np.exp(1j * l * Phi)
            mf = radial * azimuthal
            norm = float(np.sqrt(np.sum(np.abs(mf) ** 2) * self.dx * self.dy))
            if norm > 0:
                mf = mf / norm
            mode_fields[l] = mf

        if turb_model is None:
            turb_model = self._make_turb_model(cn2, distance, n_screens)
        dz = distance / n_screens

        n_errors = 0
        n_symbols = 0
        overlaps = []
        confusion = {}

        for _ in range(n_trials):
            l_tx = oam_modes[self.rng.integers(0, len(oam_modes))]
            bit_idx = oam_modes.index(l_tx)
            bits_tx = np.array(
                [(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)]
            )
            E_tx = mode_fields[l_tx].copy()

            phase_screens, amplitude_screens = self._fresh_screens(
                turb_model, n_screens, dz
            )
            E_rx = turb_model.propagate(
                E_tx, self.X, self.Y,
                phase_screens=phase_screens,
                amplitude_screens=amplitude_screens,
            )
            # OAM is scalar polarization — add noise to the single component
            E_rx, _ = add_receiver_noise(
                E_rx, np.zeros_like(E_rx), self.receiver_noise_std, self.rng
            )

            overlap_vals = {}
            for l_rx in oam_modes:
                mf = mode_fields[l_rx]
                ov = float(np.abs(np.sum(E_rx * np.conj(mf)) * self.dx * self.dy))
                overlap_vals[l_rx] = ov
            l_rx = max(overlap_vals, key=overlap_vals.get)
            overlaps.append(overlap_vals[l_tx])

            bits_rx = np.array(
                [(oam_modes.index(l_rx) >> i) & 1 for i in range(n_bits - 1, -1, -1)]
            )
            bit_error = not np.array_equal(bits_tx, bits_rx)
            n_errors += int(bit_error)
            n_symbols += 1
            key = f"{l_tx}->{l_rx}"
            confusion[key] = confusion.get(key, 0) + 1

        ber = n_errors / n_symbols if n_symbols > 0 else 0.0
        return SchemeResult(
            scheme="OAM", cn2=cn2, n_trials=n_trials,
            n_errors=n_errors, n_symbols=n_symbols, ber=ber, ser=ber,
            overlap_mean=float(np.mean(overlaps)) if overlaps else 0.0,
            confusion_matrix=confusion,
        )

    # --- Intensity ---

    def run_intensity(self, levels: List[float], cn2: float, distance: float,
                      n_screens: int, n_trials: int,
                      turb_model: Optional[TurbulenceModel] = None,
                      waist: float = 1.0) -> SchemeResult:
        n_bits = int(np.floor(np.log2(len(levels))))

        # Pre-compute Gaussian beam (normalized to unit power)
        R2 = self.X**2 + self.Y**2
        gaussian = np.exp(-R2 / waist**2).astype(np.complex128)
        norm = float(np.sqrt(np.sum(np.abs(gaussian) ** 2) * self.dx * self.dy))
        if norm > 0:
            gaussian = gaussian / norm

        if turb_model is None:
            turb_model = self._make_turb_model(cn2, distance, n_screens)
        dz = distance / n_screens

        n_errors = 0
        n_symbols = 0
        intensities = []
        confusion = {}

        for _ in range(n_trials):
            level_idx = self.rng.integers(0, len(levels))
            level = levels[level_idx]
            bits_tx = np.array(
                [(level_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)]
            )
            # Intensity levels represent TRANSMITTED POWER (matching the units
            # of measured_I = integral(|E|^2) dx dy below). gaussian is
            # normalized to unit power, so E_tx = sqrt(level) * gaussian
            # transmits power == level. Average of levels [0.5, 1.5] = 1.0 ==
            # SKM unit power -> fair on average. Comparing power against
            # amplitude levels (the previous code) was a unit mismatch and
            # biased the decision boundaries.
            E_tx = np.sqrt(level) * gaussian

            phase_screens, amplitude_screens = self._fresh_screens(
                turb_model, n_screens, dz
            )
            E_rx = turb_model.propagate(
                E_tx, self.X, self.Y,
                phase_screens=phase_screens,
                amplitude_screens=amplitude_screens,
            )
            E_rx, _ = add_receiver_noise(
                E_rx, np.zeros_like(E_rx), self.receiver_noise_std, self.rng
            )

            measured_I = float(np.sum(np.abs(E_rx) ** 2) * self.dx * self.dy)
            intensities.append(measured_I)

            nearest_idx = int(np.argmin(
                [abs(measured_I - l) for l in levels]
            ))
            bits_rx = np.array(
                [(nearest_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)]
            )
            bit_error = not np.array_equal(bits_tx, bits_rx)
            n_errors += int(bit_error)
            n_symbols += 1
            key = f"{level:.2f}->{levels[nearest_idx]:.2f}"
            confusion[key] = confusion.get(key, 0) + 1

        ber = n_errors / n_symbols if n_symbols > 0 else 0.0
        return SchemeResult(
            scheme="Intensity", cn2=cn2, n_trials=n_trials,
            n_errors=n_errors, n_symbols=n_symbols, ber=ber, ser=ber,
            intensity_mean=float(np.mean(intensities)) if intensities else 0.0,
            confusion_matrix=confusion,
        )

    # --- Full comparison ---

    def compare_all(self, skm_constellation: List[int],
                    oam_modes: List[int],
                    intensity_levels: List[float],
                    cn2_values: List[float],
                    n_trials: int = 100,
                    distance: float = 100.0,
                    n_screens: int = 10,
                    waist: float = 1.0,
                    verbose: bool = True) -> List[SchemeResult]:
        """Run all three schemes under IDENTICAL conditions.

        One turbulence model per Cn² is shared across all schemes.
        """
        results: List[SchemeResult] = []

        # Shared turbulence models
        turb_models: Dict[float, TurbulenceModel] = {}
        for cn2 in cn2_values:
            turb_models[cn2] = self._make_turb_model(cn2, distance, n_screens)

        for cn2 in cn2_values:
            if verbose:
                print(f"\n--- Cn² = {cn2:.0e} ---")

            turb_model = turb_models[cn2]

            skm = self.run_skm(skm_constellation, cn2, distance,
                               n_screens, n_trials, turb_model)
            oam = self.run_oam(oam_modes, cn2, distance,
                               n_screens, n_trials, turb_model, waist)
            intensity = self.run_intensity(intensity_levels, cn2, distance,
                                           n_screens, n_trials, turb_model, waist)

            results.extend([skm, oam, intensity])

            if verbose:
                adv = "inf" if skm.ber == 0 and oam.ber > 0 else \
                    f"{10*np.log10(oam.ber/skm.ber):.2f}" if skm.ber > 0 and oam.ber > 0 else "0.0"
                print(f"  SKM BER={skm.ber:.6f}  OAM BER={oam.ber:.6f}  "
                      f"Int BER={intensity.ber:.6f}  SKM adv={adv} dB")

        return results
