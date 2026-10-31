"""
XPOCK — HX-241 Carrier Wave Modeling

Models physical wave propagation through different carrier media using
XPOCK's wave generation capabilities and ENGINES' differential equations
engine. This provides a physically-based carrier simulation layer for
testing HX-241's carrier abstraction without real hardware.

HX-241 architectural principle preserved: carriers provide physical
capabilities rather than defining the protocol. This module models
the physical layer only; HX-241 protocol logic remains carrier-agnostic.

Supported carrier media models:
- Free-space optical (line-of-sight, atmospheric attenuation, turbulence)
- Radio frequency (multipath, fading, Doppler, shadowing)
- Acoustic (absorption, multipath, speed-of-sound variation)
- Optical fiber (attenuation, dispersion, nonlinear effects)
- Magnetic induction (near-field, distance-dependent coupling)
- Pressure wave (mechanical, solid/liquid/gas media)

Each model produces:
- Complex channel transfer function H(f)
- Impulse response h(t)
- Impairment parameters for hx-sim integration
- Channel capacity estimates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from enum import Enum
from pathlib import Path
import numpy as np

# Speed of light in m/s
C_LIGHT = 299792458.0

from xpock.core import WaveEvent, WaveField
from xpock.properties import (
    TemporalProfile, PulseShape,
    SpatialMode, SpatialModeConfig,
    PolarizationTexture, PolarizationConfig, PolarizationState,
    TopologicalCharge, TopologicalChargeConfig,
)
from xpock.engines_integration import (
    WavePropagationEngine,
    WaveInformationEngine,
    WaveSpectralEngine,
    OpticalMaterialsEngine,
    EngineResult,
)


class CarrierMedium(Enum):
    """Physical carrier media supported for wave propagation modeling."""

    FREE_SPACE_OPTICAL = "free_space_optical"
    RADIO_FREQUENCY = "radio_frequency"
    ACOUSTIC = "acoustic"
    OPTICAL_FIBER = "optical_fiber"
    MAGNETIC_INDUCTION = "magnetic_induction"
    PRESSURE_WAVE = "pressure_wave"
    CUSTOM = "custom"


@dataclass
class CarrierWaveConfig:
    """Configuration for modeling a physical carrier wave.

    Attributes
    ----------
    medium : CarrierMedium
        Type of physical carrier medium.
    frequency_hz : float
        Carrier frequency in Hz.
    wavelength_m : float
        Wavelength in meters (computed from frequency if not set).
    distance_m : float
        Propagation distance in meters.
    transmit_power_w : float
        Transmit power in watts.
    tx_gain_dbi : float
        Transmit antenna/aperture gain in dBi.
    rx_gain_dbi : float
        Receive antenna/aperture gain in dBi.
    medium_specific : dict
        Medium-specific parameters (attenuation, refractive index, etc.).
    temporal_profile : dict
        Temporal pulse shape parameters.
    spatial_mode : dict
        Spatial mode parameters.
    polarization : dict
        Polarization state parameters.
    topological_charge : int
        OAM/topological charge imposed on the wave.
    """

    medium: CarrierMedium = CarrierMedium.FREE_SPACE_OPTICAL
    frequency_hz: float = 1e9
    wavelength_m: Optional[float] = None
    distance_m: float = 1.0
    transmit_power_w: float = 1.0
    tx_gain_dbi: float = 0.0
    rx_gain_dbi: float = 0.0
    medium_specific: Dict[str, Any] = field(default_factory=dict)
    temporal_profile: Dict[str, Any] = field(default_factory=dict)
    spatial_mode: Dict[str, Any] = field(default_factory=dict)
    polarization: Dict[str, Any] = field(default_factory=dict)
    topological_charge: int = 0

    def __post_init__(self) -> None:
        if self.wavelength_m is None:
            c = C_LIGHT  # speed of light
            self.wavelength_m = c / self.frequency_hz


@dataclass
class CarrierChannelModel:
    """Result of modeling a carrier channel.

    Attributes
    ----------
    config : CarrierWaveConfig
        Input configuration.
    frequency_response : np.ndarray
        Complex channel transfer function H(f) over frequency grid.
    frequency_grid_hz : np.ndarray
        Frequency grid in Hz.
    impulse_response : np.ndarray
        Channel impulse response h(t).
    time_grid_s : np.ndarray
        Time grid in seconds.
    path_loss_db : float
        Total path loss in dB.
    delay_spread_s : float
        RMS delay spread in seconds.
    coherence_bandwidth_hz : float
        Coherence bandwidth in Hz.
    doppler_spread_hz : float
        Doppler spread in Hz.
    capacity_bps : float
        Shannon channel capacity in bits/second.
    snr_db : float
        Signal-to-noise ratio in dB.
    impairment_params : dict
        Parameters suitable for hx-sim NetworkImpairment injection.
    medium_notes : List[str]
        Medium-specific observations.
    """

    config: CarrierWaveConfig
    frequency_response: np.ndarray
    frequency_grid_hz: np.ndarray
    impulse_response: np.ndarray
    time_grid_s: np.ndarray
    path_loss_db: float
    delay_spread_s: float
    coherence_bandwidth_hz: float
    doppler_spread_hz: float
    capacity_bps: float
    snr_db: float
    impairment_params: Dict[str, Any]
    medium_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "medium": self.config.medium.value,
            "frequency_hz": self.config.frequency_hz,
            "wavelength_m": self.config.wavelength_m,
            "distance_m": self.config.distance_m,
            "transmit_power_w": self.config.transmit_power_w,
            "path_loss_db": self.path_loss_db,
            "delay_spread_s": self.delay_spread_s,
            "coherence_bandwidth_hz": self.coherence_bandwidth_hz,
            "doppler_spread_hz": self.doppler_spread_hz,
            "capacity_bps": self.capacity_bps,
            "snr_db": self.snr_db,
            "impairment_params": self.impairment_params,
            "medium_notes": self.medium_notes,
        }


class CarrierWaveModeler:
    """Models physical wave propagation through carrier media.

    Uses XPOCK's wave generation (temporal profiles, spatial modes,
    polarization, topological charge) combined with ENGINES'
    differential equations engine to produce physically-based channel
    models for HX-241 carrier simulation.

    All models are mathematical simulations — they do not require
    real hardware and make no claims about experimental results.
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./xpock_hx241_carrier")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._prop_engine: Optional[WavePropagationEngine] = None
        self._info_engine: Optional[WaveInformationEngine] = None
        self._spectral_engine: Optional[WaveSpectralEngine] = None
        self._materials_engine: Optional[OpticalMaterialsEngine] = None

    # ------------------------------------------------------------------
    # Lazy engine loading
    # ------------------------------------------------------------------

    def _get_prop(self) -> WavePropagationEngine:
        if self._prop_engine is None:
            self._prop_engine = WavePropagationEngine(workspace=self.workspace / "propagation")
        return self._prop_engine

    def _get_info(self) -> WaveInformationEngine:
        if self._info_engine is None:
            self._info_engine = WaveInformationEngine(workspace=self.workspace / "information")
        return self._info_engine

    def _get_spectral(self) -> WaveSpectralEngine:
        if self._spectral_engine is None:
            self._spectral_engine = WaveSpectralEngine(workspace=self.workspace / "spectral")
        return self._spectral_engine

    def _get_materials(self) -> OpticalMaterialsEngine:
        if self._materials_engine is None:
            self._materials_engine = OpticalMaterialsEngine(workspace=self.workspace / "materials")
        return self._materials_engine

    # ------------------------------------------------------------------
    # Core modeling
    # ------------------------------------------------------------------

    def model_carrier(
        self,
        config: CarrierWaveConfig,
    ) -> CarrierChannelModel:
        """Model wave propagation through the specified carrier medium.

        Returns a CarrierChannelModel with frequency response, impulse
        response, path loss, capacity, and impairment parameters.
        """
        notes: List[str] = []

        # --- Compute free-space path loss ---
        c = C_LIGHT
        lambda_m = config.wavelength_m
        distance = config.distance_m

        # Free-space path loss: (4*pi*d/lambda)^2
        fspl_linear = (4.0 * np.pi * distance / lambda_m) ** 2
        fspl_db = 10.0 * np.log10(fspl_linear)

        # --- Medium-specific propagation ---
        if config.medium == CarrierMedium.FREE_SPACE_OPTICAL:
            result = self._model_free_space_optical(config, fspl_db, notes)
        elif config.medium == CarrierMedium.RADIO_FREQUENCY:
            result = self._model_radio_frequency(config, fspl_db, notes)
        elif config.medium == CarrierMedium.ACOUSTIC:
            result = self._model_acoustic(config, fspl_db, notes)
        elif config.medium == CarrierMedium.OPTICAL_FIBER:
            result = self._model_optical_fiber(config, fspl_db, notes)
        elif config.medium == CarrierMedium.MAGNETIC_INDUCTION:
            result = self._model_magnetic_induction(config, fspl_db, notes)
        elif config.medium == CarrierMedium.PRESSURE_WAVE:
            result = self._model_pressure_wave(config, fspl_db, notes)
        elif config.medium == CarrierMedium.CUSTOM:
            result = self._model_custom(config, fspl_db, notes)
        else:
            raise ValueError(f"Unknown carrier medium: {config.medium}")

        # --- Combine with free-space path loss ---
        result.path_loss_db += fspl_db

        # --- Compute SNR and capacity ---
        noise_figure_db = config.medium_specific.get("noise_figure_db", 3.0)
        boltzmann_k = 1.380649e-23
        temp_k = config.medium_specific.get("temperature_k", 290.0)
        bandwidth_hz = config.medium_specific.get("bandwidth_hz", config.frequency_hz * 0.1)

        noise_power_w = boltzmann_k * temp_k * bandwidth_hz * (10.0 ** (noise_figure_db / 10.0))
        received_power_w = config.transmit_power_w * (10.0 ** (-result.path_loss_db / 10.0))
        snr_linear = received_power_w / noise_power_w if noise_power_w > 0 else 0.0
        snr_db = 10.0 * np.log10(snr_linear) if snr_linear > 0 else -1000.0

        # Shannon capacity: C = B * log2(1 + SNR)
        capacity_bps = bandwidth_hz * np.log2(1.0 + snr_linear) if snr_linear > 0 else 0.0

        result.snr_db = float(snr_db)
        result.capacity_bps = float(capacity_bps)
        result.medium_notes = notes

        # --- Build impairment parameters for hx-sim ---
        result.impairment_params = {
            "packet_loss_rate": min(0.5, max(0.0, 1.0 - 10.0 ** (-result.path_loss_db / 20.0))),
            "base_latency_us": int(distance / c * 1e6),
            "jitter_us": int(result.delay_spread_s * 1e6 / 2.0),
            "corruption_rate": min(0.1, max(0.0, 1.0 / (1.0 + snr_linear))),
            "bandwidth_bps": int(min(bandwidth_hz, capacity_bps)),
        }

        return result

    # ------------------------------------------------------------------
    # Medium-specific models
    # ------------------------------------------------------------------

    def _model_free_space_optical(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Free-space optical: atmospheric attenuation, turbulence, scintillation."""

        # Atmospheric attenuation (simplified: clear air, visible/NIR)
        attenuation_db_per_km = config.medium_specific.get("attenuation_db_per_km", 0.2)
        distance_km = config.distance_m / 1000.0
        atm_loss_db = attenuation_db_per_km * distance_km

        # Turbulence-induced scintillation (Rytov approximation)
        # Scintillation index for plane wave: sigma_I^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6)
        cn2 = config.medium_specific.get("cn2", 1e-14)  # refractive index structure constant
        k = 2.0 * np.pi / config.wavelength_m
        L = config.distance_m
        scintillation_index = 1.23 * cn2 * k ** (7.0 / 6.0) * L ** (11.0 / 6.0)
        scintillation_index = min(scintillation_index, 1.0)  # cap at strong fluctuation

        notes.append(f"Atmospheric attenuation: {atm_loss_db:.2f} dB over {distance_km:.2f} km")
        notes.append(f"Scintillation index: {scintillation_index:.4f}")
        notes.append("Free-space optical: line-of-sight, no multipath")

        # Frequency response: flat with atmospheric loss
        n_freqs = 1024
        f_center = config.frequency_hz
        f_grid = np.linspace(f_center * 0.9, f_center * 1.1, n_freqs)
        wavelength_grid = C_LIGHT / f_grid

        # Path loss varies with frequency (1/lambda^2)
        path_loss_linear = (4.0 * np.pi * config.distance_m / wavelength_grid) ** 2
        atm_loss_linear = 10.0 ** (atm_loss_db / 10.0)
        h_f = 1.0 / np.sqrt(path_loss_linear * atm_loss_linear)

        # Add scintillation as slow fading
        h_f = h_f * (1.0 + 0.1 * scintillation_index * np.random.default_rng(42).standard_normal(n_freqs))

        # Impulse response: delta function (no multipath in free space)
        t_grid = np.linspace(-1e-6, 1e-6, 1024)
        h_t = np.zeros_like(t_grid, dtype=complex)
        h_t[len(t_grid) // 2] = np.abs(h_f[len(h_f) // 2])  # peak at t=0

        delay_spread = 0.0  # no multipath
        coherence_bw = f_grid[-1] - f_grid[0]  # wide coherence bandwidth
        doppler_spread = 0.0  # static

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db + atm_loss_db,
            delay_spread_s=delay_spread,
            coherence_bandwidth_hz=coherence_bw,
            doppler_spread_hz=doppler_spread,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    def _model_radio_frequency(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Radio frequency: multipath fading, shadowing, Doppler."""

        # Multipath model: exponential power delay profile
        rms_delay_spread = config.medium_specific.get("rms_delay_spread_us", 1.0) * 1e-6
        n_taps = 16
        tap_delays = np.linspace(0, 5 * rms_delay_spread, n_taps)
        tap_powers = np.exp(-tap_delays / rms_delay_spread)
        tap_powers = tap_powers / np.sum(tap_powers)

        # Random complex gains for each tap (Rayleigh fading)
        rng = np.random.default_rng(12345)
        tap_gains = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / np.sqrt(2.0)
        tap_gains = tap_gains * np.sqrt(tap_powers)

        # Shadowing (log-normal)
        shadow_std_db = config.medium_specific.get("shadow_std_db", 6.0)
        shadow_db = rng.normal(0.0, shadow_std_db)
        notes.append(f"Shadowing: {shadow_db:.2f} dB")
        notes.append(f"RMS delay spread: {rms_delay_spread * 1e6:.2f} us")
        notes.append(f"Number of multipath taps: {n_taps}")

        # Frequency response
        n_freqs = 1024
        f_center = config.frequency_hz
        f_grid = np.linspace(f_center * 0.5, f_center * 1.5, n_freqs)

        h_f = np.zeros(n_freqs, dtype=complex)
        for i, tap_delay in enumerate(tap_delays):
            h_f += tap_gains[i] * np.exp(-1j * 2.0 * np.pi * f_grid * tap_delay)

        # Apply path loss + shadowing
        path_loss_linear = (4.0 * np.pi * config.distance_m / config.wavelength_m) ** 2
        shadow_linear = 10.0 ** (shadow_db / 10.0)
        h_f = h_f / np.sqrt(path_loss_linear) / np.sqrt(shadow_linear)

        # Impulse response
        t_grid = np.linspace(0, 10 * rms_delay_spread, 1024)
        h_t = np.zeros_like(t_grid, dtype=complex)
        for i, tap_delay in enumerate(tap_delays):
            idx = int(tap_delay / (t_grid[1] - t_grid[0]))
            if 0 <= idx < len(h_t):
                h_t[idx] += tap_gains[i]

        coherence_bw = 1.0 / (5.0 * rms_delay_spread) if rms_delay_spread > 0 else f_grid[-1] - f_grid[0]
        doppler_spread = config.medium_specific.get("doppler_spread_hz", 10.0)

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db + abs(shadow_db),
            delay_spread_s=rms_delay_spread,
            coherence_bandwidth_hz=coherence_bw,
            doppler_spread_hz=doppler_spread,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    def _model_acoustic(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Acoustic: absorption, multipath, speed-of-sound variation."""

        # Speed of sound depends on medium
        sound_speed = config.medium_specific.get("sound_speed_m_s", 1500.0)  # water default
        wavelength = sound_speed / config.frequency_hz

        # Absorption (Thorpe's approximation for seawater, simplified)
        f_khz = config.frequency_hz / 1000.0
        absorption_db_per_km = 0.1 * f_khz**2 / (1.0 + f_khz**2) + 40.0 * f_khz**2 / (4100.0 + f_khz**2)
        distance_km = config.distance_m / 1000.0
        absorption_loss_db = absorption_db_per_km * distance_km

        notes.append(f"Sound speed: {sound_speed} m/s, wavelength: {wavelength:.4f} m")
        notes.append(f"Absorption: {absorption_loss_db:.2f} dB over {distance_km:.2f} km")
        notes.append("Acoustic: slow propagation, significant multipath")

        # Multipath: surface and bottom reflections
        n_paths = 5
        path_delays = np.linspace(
            config.distance_m / sound_speed,
            3.0 * config.distance_m / sound_speed,
            n_paths,
        )
        path_amplitudes = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
        path_phases = np.random.default_rng(999).uniform(0, 2 * np.pi, n_paths)

        # Frequency response
        n_freqs = 512
        f_grid = np.linspace(config.frequency_hz * 0.5, config.frequency_hz * 1.5, n_freqs)
        h_f = np.zeros(n_freqs, dtype=complex)
        for i in range(n_paths):
            h_f += path_amplitudes[i] * np.exp(1j * path_phases[i]) * np.exp(
                -1j * 2.0 * np.pi * f_grid * path_delays[i]
            )

        # Apply spreading loss (cylindrical for shallow water, spherical for deep)
        spreading = config.medium_specific.get("spreading", "cylindrical")
        if spreading == "spherical":
            spreading_loss_db = 20.0 * np.log10(config.distance_m / 1.0)
        else:
            spreading_loss_db = 10.0 * np.log10(config.distance_m / 1.0)

        h_f = h_f / np.sqrt(10.0 ** (spreading_loss_db / 10.0)) / np.sqrt(10.0 ** (absorption_loss_db / 10.0))

        # Impulse response
        t_grid = np.linspace(0, 5 * config.distance_m / sound_speed, 1024)
        h_t = np.zeros_like(t_grid, dtype=complex)
        dt = t_grid[1] - t_grid[0]
        for i in range(n_paths):
            idx = int(path_delays[i] / dt)
            if 0 <= idx < len(h_t):
                h_t[idx] += path_amplitudes[i] * np.exp(1j * path_phases[i])

        rms_delay_spread = np.std(path_delays)
        coherence_bw = 1.0 / (5.0 * rms_delay_spread) if rms_delay_spread > 0 else f_grid[-1] - f_grid[0]
        doppler_spread = config.medium_specific.get("doppler_spread_hz", 1.0)

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db + spreading_loss_db + absorption_loss_db,
            delay_spread_s=rms_delay_spread,
            coherence_bandwidth_hz=coherence_bw,
            doppler_spread_hz=doppler_spread,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    def _model_optical_fiber(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Optical fiber: attenuation, chromatic dispersion, Kerr nonlinearity."""

        # Fiber attenuation (standard SMF: ~0.2 dB/km at 1550nm)
        attenuation_db_per_km = config.medium_specific.get("attenuation_db_per_km", 0.2)
        distance_km = config.distance_m / 1000.0
        fiber_loss_db = attenuation_db_per_km * distance_km

        # Chromatic dispersion (ps/nm/km)
        dispersion_ps_nm_km = config.medium_specific.get("dispersion_ps_nm_km", 17.0)
        # Group delay spread: D * L * delta_lambda
        source_linewidth_nm = config.medium_specific.get("source_linewidth_nm", 0.1)
        delay_spread_s = abs(dispersion_ps_nm_km) * distance_km * source_linewidth_nm * 1e-12

        # Kerr nonlinearity: nonlinear phase shift
        gamma_w_km = config.medium_specific.get("gamma_w_km", 1.3)  # 1/(W*km)
        peak_power_w = config.transmit_power_w
        nonlinear_phase = gamma_w_km * peak_power_w * distance_km  # radians

        notes.append(f"Fiber attenuation: {fiber_loss_db:.2f} dB over {distance_km:.2f} km")
        notes.append(f"Chromatic dispersion: {dispersion_ps_nm_km} ps/nm/km")
        notes.append(f"Delay spread: {delay_spread_s * 1e12:.2f} ps")
        notes.append(f"Kerr nonlinear phase: {nonlinear_phase:.3f} rad")
        notes.append("Optical fiber: guided, low loss, dispersion-limited")

        # Frequency response: dispersion causes frequency-dependent phase
        n_freqs = 1024
        f_center = config.frequency_hz
        f_grid = np.linspace(f_center * 0.99, f_center * 1.01, n_freqs)
        df = f_grid - f_center

        # Dispersion phase: phi = -beta2 * L * omega^2 / 2
        # beta2 = -D * lambda^2 / (2*pi*c)
        beta2 = -dispersion_ps_nm_km * 1e-12 * 1e-9 * config.wavelength_m**2 / (2.0 * np.pi * C_LIGHT)
        phase_dispersion = beta2 * distance_km * 1000.0 * (2.0 * np.pi * df) ** 2 / 2.0

        # Attenuation
        att_linear = 10.0 ** (fiber_loss_db / 10.0)
        h_f = np.exp(-1j * phase_dispersion) / np.sqrt(att_linear)

        # Impulse response (approximate via IFFT)
        h_t = np.fft.ifft(np.fft.ifftshift(h_f))
        t_grid = np.fft.fftshift(np.fft.fftfreq(n_freqs, d=(f_grid[1] - f_grid[0])))

        coherence_bw = 1.0 / (5.0 * delay_spread_s) if delay_spread_s > 0 else f_grid[-1] - f_grid[0]
        doppler_spread = 0.0  # static fiber

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db + fiber_loss_db,
            delay_spread_s=delay_spread_s,
            coherence_bandwidth_hz=coherence_bw,
            doppler_spread_hz=doppler_spread,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    def _model_magnetic_induction(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Magnetic induction: near-field, distance-dependent coupling."""

        # MI coupling: falls off as 1/d^6 for near-field (magnetic dipole)
        # Very short range, no multipath
        distance = config.distance_m
        reference_distance = 0.1  # 10 cm reference
        coupling_db = 60.0 * np.log10(distance / reference_distance) if distance > reference_distance else 0.0

        notes.append(f"Near-field magnetic induction coupling: {coupling_db:.1f} dB")
        notes.append("Range: < 10 meters typically, no multipath")
        notes.append("Suitable for short-range, low-power HX-241 links")

        n_freqs = 512
        f_center = config.frequency_hz
        f_grid = np.linspace(f_center * 0.8, f_center * 1.2, n_freqs)

        # Flat frequency response within bandwidth
        coupling_linear = 10.0 ** (coupling_db / 10.0)
        h_f = np.ones(n_freqs, dtype=complex) / np.sqrt(1.0 / max(coupling_linear, 1e-12))

        t_grid = np.linspace(-1e-6, 1e-6, 1024)
        h_t = np.zeros_like(t_grid, dtype=complex)
        h_t[len(t_grid) // 2] = np.abs(h_f[len(h_f) // 2])

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db + abs(coupling_db),
            delay_spread_s=0.0,
            coherence_bandwidth_hz=f_grid[-1] - f_grid[0],
            doppler_spread_hz=0.0,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    def _model_pressure_wave(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Pressure wave: mechanical waves in solid/liquid/gas."""

        sound_speed = config.medium_specific.get("sound_speed_m_s", 343.0)  # air default
        medium_type = config.medium_specific.get("medium_type", "air")

        # Absorption depends on medium
        absorption_db_per_m = config.medium_specific.get("absorption_db_per_m", 0.01)
        absorption_loss_db = absorption_db_per_m * config.distance_m

        notes.append(f"Medium: {medium_type}, sound speed: {sound_speed} m/s")
        notes.append(f"Absorption: {absorption_loss_db:.2f} dB over {config.distance_m} m")
        notes.append("Pressure wave: mechanical, dispersive in solids")

        wavelength = sound_speed / config.frequency_hz
        n_freqs = 512
        f_grid = np.linspace(config.frequency_hz * 0.5, config.frequency_hz * 1.5, n_freqs)

        # Simple model: spreading + absorption
        spreading_loss_db = 20.0 * np.log10(config.distance_m / 1.0)
        h_f = np.ones(n_freqs, dtype=complex) / np.sqrt(
            10.0 ** (spreading_loss_db / 10.0) * 10.0 ** (absorption_loss_db / 10.0)
        )

        t_grid = np.linspace(0, 5 * config.distance_m / sound_speed, 1024)
        h_t = np.zeros_like(t_grid, dtype=complex)
        h_t[len(t_grid) // 2] = np.abs(h_f[len(h_f) // 2])

        rms_delay_spread = 0.1 * config.distance_m / sound_speed
        coherence_bw = 1.0 / (5.0 * rms_delay_spread)
        doppler_spread = config.medium_specific.get("doppler_spread_hz", 0.1)

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db + spreading_loss_db + absorption_loss_db,
            delay_spread_s=rms_delay_spread,
            coherence_bandwidth_hz=coherence_bw,
            doppler_spread_hz=doppler_spread,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    def _model_custom(
        self,
        config: CarrierWaveConfig,
        fspl_db: float,
        notes: List[str],
    ) -> CarrierChannelModel:
        """Custom model: user-defined transfer function."""

        custom_fn = config.medium_specific.get("transfer_function")
        if custom_fn is None:
            raise ValueError("Custom medium requires 'transfer_function' in medium_specific")

        n_freqs = 512
        f_center = config.frequency_hz
        f_grid = np.linspace(f_center * 0.8, f_center * 1.2, n_freqs)
        h_f = np.array([custom_fn(f) for f in f_grid], dtype=complex)

        t_grid = np.linspace(-1e-6, 1e-6, 1024)
        h_t = np.fft.ifft(np.fft.ifftshift(h_f))

        notes.append("Custom carrier model with user-defined transfer function")

        return CarrierChannelModel(
            config=config,
            frequency_response=h_f,
            frequency_grid_hz=f_grid,
            impulse_response=h_t,
            time_grid_s=t_grid,
            path_loss_db=fspl_db,
            delay_spread_s=0.0,
            coherence_bandwidth_hz=f_grid[-1] - f_grid[0],
            doppler_spread_hz=0.0,
            capacity_bps=0.0,
            snr_db=0.0,
            impairment_params={},
            medium_notes=notes,
        )

    # ------------------------------------------------------------------
    # Integration with HX-241 hx-sim impairment engine
    # ------------------------------------------------------------------

    def impairment_to_hx_sim(
        self,
        channel_model: CarrierChannelModel,
    ) -> Dict[str, Any]:
        """Convert channel model to hx-sim NetworkImpairment parameters.

        Returns a dict suitable for configuring hx-sim's NetworkImpairment.
        """
        return {
            "packet_loss_rate": channel_model.impairment_params.get("packet_loss_rate", 0.0),
            "duplication_rate": 0.0,
            "reordering_depth": 0,
            "corruption_rate": channel_model.impairment_params.get("corruption_rate", 0.0),
            "base_latency_us": channel_model.impairment_params.get("base_latency_us", 0),
            "jitter_us": channel_model.impairment_params.get("jitter_us", 0),
            "bandwidth_bps": channel_model.impairment_params.get("bandwidth_bps", 0),
            "partition": None,
            "enabled": True,
            "source": "xpock_carrier_wave_model",
            "medium": channel_model.config.medium.value,
        }

    # ------------------------------------------------------------------
    # Generate wave event for carrier transmission
    # ------------------------------------------------------------------

    def generate_transmit_wave(
        self,
        config: CarrierWaveConfig,
        grid_arrays: dict,
    ) -> WaveEvent:
        """Generate a wave event suitable for transmission over the carrier.

        Uses XPOCK's wave generation to create a modulated wave with
        the specified temporal, spatial, polarization, and topological
        properties.
        """
        from xpock.cli import generate_wave_event

        params = {
            "temporal_shape": config.temporal_profile.get("shape", "gaussian"),
            "temporal_amplitude": np.sqrt(config.transmit_power_w),
            "temporal_center": config.temporal_profile.get("center", 0.0),
            "temporal_duration": config.temporal_profile.get("duration", 1.0 / config.frequency_hz),
            "temporal_chirp_rate": config.temporal_profile.get("chirp_rate", 0.0),
            "temporal_carrier_freq": 2.0 * np.pi * config.frequency_hz,
            "spatial_mode": config.spatial_mode.get("mode", "gaussian"),
            "spatial_l": config.topological_charge,
            "spatial_waist": config.spatial_mode.get("waist", 0.1),
            "spatial_wavelength": config.wavelength_m,
            "polarization_state": config.polarization.get("state", "linear"),
            "polarization_angle": config.polarization.get("angle", 0.0),
            "topology_charge": config.topological_charge,
            "coherence_type": "full",
        }

        return generate_wave_event(params, grid_arrays)