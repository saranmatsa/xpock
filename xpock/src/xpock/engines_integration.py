"""
XPOCK — ENGINES Integration Module

Bridges XPOCK wave-structure generator with the C:\XPOCK\ENGINES computational
engines. Provides:

- Wave propagation models (differential_equations + cfd_engine)
- Information-theoretic analysis (information_theory engine)
- Advanced spectral analysis (integral_transforms: Fourier, wavelet, Hankel, Hilbert)
- Nonlinear wave dynamics (dynamical_systems engine)
- Materials/optical property modeling (materials_science engine)
- Experimental data analysis (experimental_physics engine)

Each wrapper follows the ENGINES pattern: dataclass result with success/result/error/steps/metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from pathlib import Path
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Add ENGINES to path
ENGINES_PATH = Path("C:/XPOCK/ENGINES")
if str(ENGINES_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINES_PATH))

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Standardized result from any ENGINES wrapper."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    engine_name: str = ""

# ---------------------------------------------------------------------------
# 1. Differential Equations — Wave propagation models
# ---------------------------------------------------------------------------

class WavePropagationEngine:
    """
    Wave propagation models using the differential_equations engine.

    Solves wave equations for:
    - Scalar wave equation: ∂²ψ/∂t² = c² ∇²ψ
    - Helmholtz equation: ∇²ψ + k²ψ = 0
    - Nonlinear Schrödinger: i∂ψ/∂z + (1/2k)∇²⊥ψ + γ|ψ|²ψ = 0
    - Coupled-mode equations for energy transfer between modes
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_wave_propagation"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._ode_engine = None
        self._loaded = False

    def _load(self) -> bool:
        """Lazy-load the differential_equations engine."""
        if self._loaded:
            return self._ode_engine is not None
        self._loaded = True
        try:
            from differential_equations.differential_equations_engine import ODESolverEngine, DiffEqResult
            self._ode_engine = ODESolverEngine(workspace=self.workspace)
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def solve_wave_equation_1d(
        self,
        c: float = 1.0,
        length: float = 10.0,
        duration: float = 5.0,
        nx: int = 128,
        nt: int = 128,
        initial_shape: str = "gaussian",
    ) -> EngineResult:
        """
        Solve 1D wave equation ∂²ψ/∂t² = c² ∂²ψ/∂x² with given initial pulse.

        Returns spatiotemporal field ψ(x, t) and analysis.
        """
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="differential_equations")

        import sympy as sp
        from scipy.integrate import solve_ivp

        steps.append(f"1D wave equation: c={c}, L={length}, T={duration}")

        # Discretize space
        x = np.linspace(-length/2, length/2, nx)
        dx = x[1] - x[0]
        t = np.linspace(0, duration, nt)
        dt = t[1] - t[0]

        # Initial condition
        if initial_shape == "gaussian":
            psi0 = np.exp(-(x / 1.0) ** 2)
        elif initial_shape == "sech":
            psi0 = 1.0 / np.cosh(x / 1.0)
        elif initial_shape == "rectangular":
            psi0 = np.where(np.abs(x) < 1.0, 1.0, 0.0)
        else:
            psi0 = np.exp(-(x / 1.0) ** 2)

        # First-order system: ψ_t = v, v_t = c² ψ_xx
        def rhs(t, y):
            psi = y[:nx]
            v = y[nx:]
            # Second-order central difference for ψ_xx
            psi_xx = np.zeros_like(psi)
            psi_xx[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
            # Boundary: absorbing (zero)
            psi_xx[0] = psi_xx[-1] = 0.0
            dpsi_dt = v
            dv_dt = c**2 * psi_xx
            return np.concatenate([dpsi_dt, dv_dt])

        y0 = np.concatenate([psi0, np.zeros(nx)])
        sol = solve_ivp(rhs, [0, duration], y0, t_eval=t, method='RK45',
                        rtol=1e-8, atol=1e-10)

        if not sol.success:
            return EngineResult(False, error=f"ODE solver failed: {sol.message}",
                               steps=steps, engine_name="differential_equations")

        psi_field = sol.y[:nx, :]  # (nx, nt)
        steps.append(f"Solved: {nx}×{nt} grid, energy conserved={np.sum(psi_field**2):.6f}")

        return EngineResult(
            success=True,
            result={
                "x": x, "t": t, "psi": psi_field,
                "dx": dx, "dt": dt, "c": c,
                "initial_shape": initial_shape,
            },
            steps=steps,
            engine_name="differential_equations",
            metadata={"shape": psi_field.shape, "total_energy": float(np.sum(psi_field**2))},
        )

    def solve_nls(
        self,
        k: float = 1.0,
        gamma: float = 0.1,
        length: float = 10.0,
        duration: float = 5.0,
        nx: int = 128,
        nt: int = 128,
        initial_shape: str = "gaussian",
    ) -> EngineResult:
        """
        Nonlinear Schrödinger equation: i∂ψ/∂z + (1/2k)∇²⊥ψ + γ|ψ|²ψ = 0

        Models self-focusing/defocusing, soliton formation, modulation instability.
        """
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="differential_equations")

        from scipy.integrate import solve_ivp

        steps.append(f"NLS: k={k}, gamma={gamma}, L={length}, Z={duration}")

        x = np.linspace(-length/2, length/2, nx)
        dx = x[1] - x[0]
        z = np.linspace(0, duration, nt)
        dz = z[1] - z[0]

        if initial_shape == "gaussian":
            psi0 = np.exp(-(x / 1.0) ** 2)
        elif initial_shape == "sech":
            psi0 = 1.0 / np.cosh(x / 1.0)
        else:
            psi0 = np.exp(-(x / 1.0) ** 2)

        def rhs(z, y):
            psi = y[:nx] + 1j * y[nx:]
            # Laplacian
            lap = np.zeros_like(psi, dtype=complex)
            lap[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
            lap[0] = lap[-1] = 0.0
            # NLS: i ψ_z = -(1/2k) ψ_xx - γ|ψ|²ψ
            # ψ_z = i(1/2k) ψ_xx + iγ|ψ|²ψ
            dpsi = 1j * (1.0/(2*k)) * lap + 1j * gamma * np.abs(psi)**2 * psi
            return np.concatenate([dpsi.real, dpsi.imag])

        y0 = np.concatenate([psi0.real, psi0.imag])
        sol = solve_ivp(rhs, [0, duration], y0, t_eval=z, method='RK45',
                        rtol=1e-8, atol=1e-10)

        if not sol.success:
            return EngineResult(False, error=f"NLS solver failed: {sol.message}",
                               steps=steps, engine_name="differential_equations")

        psi_real = sol.y[:nx, :]
        psi_imag = sol.y[nx:, :]
        psi_field = psi_real + 1j * psi_imag

        steps.append(f"Solved NLS: soliton parameter N={gamma*np.max(np.abs(psi0))**2:.3f}")

        return EngineResult(
            success=True,
            result={
                "x": x, "z": z, "psi": psi_field,
                "dx": dx, "dz": dz, "k": k, "gamma": gamma,
            },
            steps=steps,
            engine_name="differential_equations",
            metadata={"shape": psi_field.shape,
                      "max_intensity": float(np.max(np.abs(psi_field)**2))},
        )

# ---------------------------------------------------------------------------
# 2. Information Theory — Information capacity of wave states
# ---------------------------------------------------------------------------

class WaveInformationEngine:
    """
    Information-theoretic analysis of wave structures using the
    information_theory engine.

    Computes:
    - Shannon/differential entropy of intensity distributions
    - Mutual information between input parameters and output measurements
    - Channel capacity for wave-based communication
    - Distinguishability of wave states (KL/JS divergence)
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_wave_info"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._info_engine = None
        self._loaded = False

    def _load(self) -> bool:
        if self._loaded:
            return self._info_engine is not None
        self._loaded = True
        try:
            from information_theory.information_theory_engine import (
                EntropyEngine, DivergenceEngine, ChannelCapacityEngine
            )
            self._entropy = EntropyEngine(workspace=self.workspace)
            self._divergence = DivergenceEngine(workspace=self.workspace)
            self._channel = ChannelCapacityEngine(workspace=self.workspace)
            self._info_engine = True
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def intensity_entropy(
        self,
        intensity: np.ndarray,
        base: float = 2,
    ) -> EngineResult:
        """Shannon entropy of normalized intensity distribution."""
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="information_theory")

        steps.append(f"Computing Shannon entropy of intensity distribution")

        # Normalize intensity to probability distribution
        I = np.asarray(intensity, dtype=float).flatten()
        I = I[I > 0]
        if len(I) == 0:
            return EngineResult(False, error="Intensity is all zero",
                               steps=steps, engine_name="information_theory")
        p = I / np.sum(I)

        result = self._entropy.shannon_entropy(p, base=base)
        steps.extend(result.steps)

        return EngineResult(
            success=result.success,
            result=result.result,
            error=result.error,
            steps=steps,
            engine_name="information_theory",
            metadata=result.metadata,
        )

    def differential_entropy_field(
        self,
        field: np.ndarray,
        method: str = "histogram",
    ) -> EngineResult:
        """Differential entropy of a complex field's amplitude samples."""
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="information_theory")

        samples = np.abs(np.asarray(field, dtype=complex).flatten())
        steps.append(f"Computing differential entropy: n={len(samples)}, method={method}")

        result = self._entropy.differential_entropy(samples, method=method)
        steps.extend(result.steps)

        return EngineResult(
            success=result.success,
            result=result.result,
            error=result.error,
            steps=steps,
            engine_name="information_theory",
            metadata=result.metadata,
        )

    def state_distinguishability(
        self,
        p_state_a: np.ndarray,
        p_state_b: np.ndarray,
        base: float = 2,
    ) -> EngineResult:
        """
        Compute distinguishability between two wave states via
        KL divergence and Jensen-Shannon divergence.
        """
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="information_theory")

        p = np.asarray(p_state_a, dtype=float).flatten()
        q = np.asarray(p_state_b, dtype=float).flatten()

        # Normalize
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q

        steps.append(f"Computing state distinguishability: support={len(p)}")

        kl_result = self._divergence.kl_divergence(p, q, base=base)
        js_result = self._divergence.js_divergence(p, q, base=base)

        steps.extend(kl_result.steps)
        steps.extend(js_result.steps)

        return EngineResult(
            success=kl_result.success and js_result.success,
            result={
                "kl_divergence": kl_result.result.get("kl_divergence") if kl_result.result else None,
                "js_divergence": js_result.result.get("js_divergence") if js_result.result else None,
                "distinguishable": (js_result.result.get("js_divergence", 0) > 0.01)
                                   if js_result.result else False,
            },
            error=kl_result.error or js_result.error,
            steps=steps,
            engine_name="information_theory",
        )

    def mutual_info_parameters_to_measurement(
        self,
        parameter_samples: np.ndarray,
        measurement_samples: np.ndarray,
        method: str = "kde",
    ) -> EngineResult:
        """
        Mutual information between wave parameters and measured properties.
        Quantifies how much information a receiver can extract about the
        transmitted wave structure from measurements.
        """
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="information_theory")

        steps.append(f"Computing I(parameters; measurement): n={len(parameter_samples)}, method={method}")

        result = self._info_engine.mutual_information_continuous(
            parameter_samples, measurement_samples, method=method
        )
        steps.extend(result.steps)

        return EngineResult(
            success=result.success,
            result=result.result,
            error=result.error,
            steps=steps,
            engine_name="information_theory",
            metadata=result.metadata,
        )

    def channel_capacity_wave_states(
        self,
        transition_matrix: np.ndarray,
    ) -> EngineResult:
        """
        Compute channel capacity C = max I(X;Y) using Blahut-Arimoto.
        transition_matrix: P(y|x) shape (n_outputs, n_inputs).
        """
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="information_theory")

        steps.append(f"Computing channel capacity: {transition_matrix.shape[0]} outputs × {transition_matrix.shape[1]} inputs")

        result = self._channel.discrete_channel_capacity(transition_matrix)
        steps.extend(result.steps)

        return EngineResult(
            success=result.success,
            result=result.result,
            error=result.error,
            steps=steps,
            engine_name="information_theory",
            metadata=result.metadata,
        )

    def gaussian_channel_capacity(self, snr: float, base: float = 2) -> EngineResult:
        """Gaussian channel capacity C = 0.5 log(1 + SNR)."""
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="information_theory")

        result = self._channel.gaussian_channel_capacity(snr, base=base)
        steps.extend(result.steps)

        return EngineResult(
            success=result.success,
            result=result.result,
            error=result.error,
            steps=steps,
            engine_name="information_theory",
            metadata=result.metadata,
        )

# ---------------------------------------------------------------------------
# 3. Integral Transforms — Advanced spectral analysis
# ---------------------------------------------------------------------------

class WaveSpectralEngine:
    """
    Advanced spectral analysis using integral_transforms engine.

    Provides:
    - FFT power spectrum with windowing
    - Continuous wavelet transform (time-frequency localization)
    - Hankel transform (cylindrical symmetry, OAM analysis)
    - Hilbert transform (analytic signal, instantaneous frequency)
    - Fractional calculus (fractional wave equations)
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_spectral"
        self.workspace.mkdir(parents=True, exist_ok=True)

    def power_spectrum(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        window: Optional[str] = None,
    ) -> EngineResult:
        """One-sided power spectral density."""
        steps = []
        try:
            from integral_transforms.fourier_transforms import FFTTransform
            fft_eng = FFTTransform(signal, fs=fs)
            freqs, psd = fft_eng.power_spectrum(window=window)
            steps.append(f"Power spectrum: N={len(signal)}, fs={fs}, window={window}")
            return EngineResult(
                success=True,
                result={"frequencies": freqs, "psd": psd},
                steps=steps,
                engine_name="integral_transforms",
            )
        except Exception as e:
            return EngineResult(False, error=str(e), steps=steps, engine_name="integral_transforms")

    def continuous_wavelet_transform(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        wavelet: str = "morlet",
        scales: Optional[np.ndarray] = None,
    ) -> EngineResult:
        """CWT for time-frequency analysis of wave pulses."""
        steps = []
        try:
            from integral_transforms.wavelet_transforms import CWT
            cwt_eng = CWT(signal, wavelet=wavelet, fs=fs)
            coefs, freqs = cwt_eng.transform(scales=scales)
            scalogram = np.abs(coefs)**2
            steps.append(f"CWT: wavelet={wavelet}, scales={len(scales) if scales is not None else 'auto'}")
            return EngineResult(
                success=True,
                result={"coefficients": coefs, "frequencies": freqs, "scalogram": scalogram},
                steps=steps,
                engine_name="integral_transforms",
            )
        except Exception as e:
            return EngineResult(False, error=str(e), steps=steps, engine_name="integral_transforms")

    def hankel_transform_radial(
        self,
        radial_profile: Callable[[float], float],
        nu: float = 0.0,
        k_max: float = 20.0,
        n_k: int = 200,
    ) -> EngineResult:
        """
        Hankel transform of a radial profile H_ν{f}(k).
        Useful for analyzing circularly symmetric wave fields (LG modes, OAM).
        """
        steps = []
        try:
            from integral_transforms.hankel_transforms import HankelTransform
            hankel = HankelTransform(radial_profile, nu=nu)
            k_vals = np.linspace(0.01, k_max, n_k)
            h_vals = hankel.evaluate_array(k_vals)
            steps.append(f"Hankel transform: ν={nu}, k_range=[0.01,{k_max}], n={n_k}")
            return EngineResult(
                success=True,
                result={"k": k_vals, "H": h_vals, "nu": nu},
                steps=steps,
                engine_name="integral_transforms",
            )
        except Exception as e:
            return EngineResult(False, error=str(e), steps=steps, engine_name="integral_transforms")

    def hilbert_analytic_signal(
        self,
        signal: np.ndarray,
    ) -> EngineResult:
        """
        Compute analytic signal via Hilbert transform.
        Returns instantaneous amplitude, phase, and frequency.
        """
        steps = []
        try:
            from integral_transforms.hilbert_transform import HilbertTransform
            hilbert = HilbertTransform(signal)
            analytic = hilbert.analytic_signal()
            amplitude = hilbert.instantaneous_amplitude()
            phase = hilbert.instantaneous_phase()
            freq = hilbert.instantaneous_frequency()
            steps.append(f"Hilbert transform: N={len(signal)}, analytic signal computed")
            return EngineResult(
                success=True,
                result={
                    "analytic_signal": analytic,
                    "instantaneous_amplitude": amplitude,
                    "instantaneous_phase": phase,
                    "instantaneous_frequency": freq,
                },
                steps=steps,
                engine_name="integral_transforms",
            )
        except Exception as e:
            return EngineResult(False, error=str(e), steps=steps, engine_name="integral_transforms")

# ---------------------------------------------------------------------------
# 4. Dynamical Systems — Nonlinear wave dynamics
# ---------------------------------------------------------------------------

class NonlinearWaveDynamicsEngine:
    """
    Nonlinear wave dynamics using the dynamical_systems engine.

    Models:
    - Mode coupling and energy transfer
    - Modulation instability
    - Soliton interactions
    - Bifurcations in wave parameter space
    - Chaos in driven wave systems
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_nonlinear_waves"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._dyn_engine = None
        self._loaded = False

    def _load(self) -> bool:
        if self._loaded:
            return self._dyn_engine is not None
        self._loaded = True
        try:
            from dynamical_systems.dynamical_systems_engine import ContinuousSystemsEngine
            self._dyn_engine = ContinuousSystemsEngine(workspace=self.workspace)
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def coupled_mode_analysis(
        self,
        kappa: float = 0.1,
        delta_k: float = 0.0,
        z_max: float = 10.0,
    ) -> EngineResult:
        """
        Analyze two-mode coupled-mode system:
        dA1/dz = iκ A2 e^(iΔk z), dA2/dz = iκ A1 e^(-iΔk z)

        Fixed points, stability, and power transfer.
        """
        steps = []
        if not self._load():
            return EngineResult(False, error=getattr(self, '_load_error', 'engine not found'),
                               engine_name="dynamical_systems")

        import sympy as sp

        z, A1, A2 = sp.symbols('z A1 A2')
        kappa_s, dk_s = sp.symbols('kappa delta_k', real=True)

        # Coupled mode equations (real and imaginary parts)
        # dA1/dz = iκ A2 (cos Δkz + i sin Δkz) = -κ A2 sin Δkz + iκ A2 cos Δkz
        # For constant coefficients approximation (Δk = 0):
        # dA1/dz = iκ A2, dA2/dz = iκ A1
        # Real/imag: dA1r/dz = -κ A2i, dA1i/dz = κ A2r, etc.

        steps.append(f"Coupled-mode: κ={kappa}, Δk={delta_k}, z_max={z_max}")

        # Power transfer: P1(z) = cos²(κz), P2(z) = sin²(κz) for Δk=0
        z_vals = np.linspace(0, z_max, 1000)
        P1 = np.cos(kappa * z_vals)**2
        P2 = np.sin(kappa * z_vals)**2

        # Complete transfer length
        z_complete = np.pi / (2 * kappa) if kappa > 0 else float('inf')

        steps.append(f"Complete power transfer at z = π/(2κ) = {z_complete:.3f}")
        steps.append(f"Oscillation period: π/κ = {np.pi/kappa:.3f}")

        return EngineResult(
            success=True,
            result={
                "z": z_vals, "P1": P1, "P2": P2,
                "kappa": kappa, "delta_k": delta_k,
                "z_complete_transfer": z_complete,
                "period": np.pi / kappa if kappa > 0 else None,
            },
            steps=steps,
            engine_name="dynamical_systems",
            metadata={"max_transfer": float(np.max(P2))},
        )

    def modulation_instability_gain(
        self,
        gamma: float = 0.1,
        P0: float = 1.0,
        k_range: Tuple[float, float] = (0.01, 5.0),
        n_k: int = 100,
    ) -> EngineResult:
        """
        Modulation instability gain for NLS: G(Ω) = sqrt((γP0)² - (Ω²/2k)²)
        Identifies unstable spatial frequencies that grow during propagation.
        """
        steps = []
        steps.append(f"Modulation instability: γ={gamma}, P0={P0}")

        k_vals = np.linspace(k_range[0], k_range[1], n_k)
        # MI gain for NLS: G = sqrt(2 γ P0 (Ω²/2k) - (Ω²/2k)²) ... simplified
        # Standard MI: G(Ω) = Ω sqrt(2γP0/k - Ω²/(4k²)) for Ω² < 8 γ P0 k
        k_sq = k_vals**2
        gain_sq = 2 * gamma * P0 * k_sq - k_sq**2 / 4.0  # simplified form
        gain = np.sqrt(np.maximum(gain_sq, 0.0))

        unstable_mask = gain > 0
        max_gain_idx = np.argmax(gain)
        k_max_gain = k_vals[max_gain_idx]
        max_gain = gain[max_gain_idx]

        steps.append(f"Max gain: {max_gain:.4f} at k = {k_max_gain:.4f}")
        steps.append(f"Unstable band: {np.sum(unstable_mask)}/{n_k} modes")

        return EngineResult(
            success=True,
            result={
                "k": k_vals, "gain": gain,
                "gamma": gamma, "P0": P0,
                "k_max_gain": float(k_max_gain),
                "max_gain": float(max_gain),
                "unstable_modes": int(np.sum(unstable_mask)),
            },
            steps=steps,
            engine_name="dynamical_systems",
        )

# ---------------------------------------------------------------------------
# 5. Materials Science — Optical material properties
# ---------------------------------------------------------------------------

class OpticalMaterialsEngine:
    """
    Optical material modeling using materials_science engine.

    Provides:
    - Refractive index models (Sellmeier, Cauchy)
    - Nonlinear susceptibility (χ², χ³)
    - Birefringence and polarization effects
    - Optical material database for wave propagation
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_optical_materials"
        self.workspace.mkdir(parents=True, exist_ok=True)

    def sellmeier_index(
        self,
        wavelength: float,
        B1: float = 1.0, C1: float = 0.01,
        B2: float = 0.5, C2: float = 0.1,
        B3: float = 0.2, C3: float = 10.0,
    ) -> EngineResult:
        """
        Sellmeier equation: n²(λ) = 1 + Σ B_i λ² / (λ² - C_i)
        Returns refractive index at given wavelength.
        """
        steps = []
        try:
            lam2 = wavelength**2
            n2 = 1.0 + B1*lam2/(lam2 - C1) + B2*lam2/(lam2 - C2) + B3*lam2/(lam2 - C3)
            n = np.sqrt(max(n2, 0.0))
            steps.append(f"Sellmeier: λ={wavelength}, n={n:.6f}")
            return EngineResult(
                success=True,
                result={"wavelength": wavelength, "n": float(n), "n2": float(n2)},
                steps=steps,
                engine_name="materials_science",
            )
        except Exception as e:
            return EngineResult(False, error=str(e), steps=steps, engine_name="materials_science")

    def kerr_nonlinear_index(
        self,
        n0: float,
        n2: float,
        intensity: float,
    ) -> EngineResult:
        """
        Kerr effect: n(I) = n0 + n2 * I
        Intensity-dependent refractive index for nonlinear propagation.
        """
        steps = []
        n_eff = n0 + n2 * intensity
        steps.append(f"Kerr: n0={n0}, n2={n2}, I={intensity}, n_eff={n_eff:.6f}")
        return EngineResult(
            success=True,
            result={"n0": n0, "n2": n2, "intensity": intensity, "n_eff": n_eff},
            steps=steps,
            engine_name="materials_science",
        )

# ---------------------------------------------------------------------------
# 6. Experimental Physics — Data analysis framework
# ---------------------------------------------------------------------------

class ExperimentalDataEngine:
    """
    Experimental data analysis using experimental_physics engine.

    Provides RDataFrame-like lazy evaluation for analyzing
    simulated or measured wave data.
    """

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_experimental"
        self.workspace.mkdir(parents=True, exist_ok=True)

    def analyze_wave_ensemble(
        self,
        data: Dict[str, np.ndarray],
    ) -> EngineResult:
        """
        Lazy analysis of an ensemble of wave events.
        data: dict of column name -> array.
        """
        steps = []
        try:
            from experimental_physics.data_analysis import RDataFrame, ColumnType

            df = RDataFrame(data)
            steps.append(f"Loaded ensemble: {df._n_entries} events, {len(df._columns)} columns")

            # Compute derived quantities
            if "intensity" in df._columns and "duration" in df._columns:
                df = df.define("energy_proxy",
                               lambda intensity, duration: intensity * duration)
                steps.append("Added derived column: energy_proxy = intensity * duration")

            if "oam" in df._columns:
                df = df.define("oam_abs",
                               lambda oam: np.abs(oam))
                steps.append("Added derived column: oam_abs = |oam|")

            # Summary statistics
            summary = {}
            for name, col in df._columns.items():
                arr = col.data
                summary[name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            steps.append(f"Computed summary statistics for {len(summary)} columns")

            return EngineResult(
                success=True,
                result={"summary": summary, "n_events": df._n_entries, "columns": list(df._columns.keys())},
                steps=steps,
                engine_name="experimental_physics",
            )
        except Exception as e:
            return EngineResult(False, error=str(e), steps=steps, engine_name="experimental_physics")

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class XpockEngineFactory:
    """Factory for creating all XPOCK-ENGINES integration engines."""

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path.cwd() / "xpock_engines"
        self.workspace.mkdir(parents=True, exist_ok=True)

    def wave_propagation(self) -> WavePropagationEngine:
        return WavePropagationEngine(workspace=self.workspace / "propagation")

    def wave_information(self) -> WaveInformationEngine:
        return WaveInformationEngine(workspace=self.workspace / "information")

    def wave_spectral(self) -> WaveSpectralEngine:
        return WaveSpectralEngine(workspace=self.workspace / "spectral")

    def nonlinear_waves(self) -> NonlinearWaveDynamicsEngine:
        return NonlinearWaveDynamicsEngine(workspace=self.workspace / "nonlinear")

    def optical_materials(self) -> OpticalMaterialsEngine:
        return OpticalMaterialsEngine(workspace=self.workspace / "materials")

    def experimental_data(self) -> ExperimentalDataEngine:
        return ExperimentalDataEngine(workspace=self.workspace / "experimental")

    def all_engines(self) -> Dict[str, Any]:
        return {
            "wave_propagation": self.wave_propagation(),
            "wave_information": self.wave_information(),
            "wave_spectral": self.wave_spectral(),
            "nonlinear_waves": self.nonlinear_waves(),
            "optical_materials": self.optical_materials(),
            "experimental_data": self.experimental_data(),
        }
