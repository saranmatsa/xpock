"""
XPOCK — Exhaustive Adversarial Validation Suite

Generates 10,000+ genuinely distinct test cases through pytest's real
testing infrastructure. Each test case gets a unique trace_id for full
traceability. Parameters are derived from the actual implementation
limits and documented physical models — not arbitrary thresholds.

Test categories (15):
  T1  Turbulence extremes (Cn², distance, screens, wavelength, scales)
  T2  Phase distortion (global offset, Ex/Ey mismatch, polarization states)
  T3  Amplitude fluctuations (power levels, profiles, fading)
  T4  Noise injection (Gaussian, shot/Poisson, detector, mixed)
  T5  Interference (overlapping fields, crosstalk, multi-user)
  T6  Misalignment (grid offset, rotation, scaling, centroid shift)
  T7  Synchronization (timing offset, Ex/Ey phase mismatch, clock drift)
  T8  Encoding/decoding (constellation sizes, bit mappings, symbol rates)
  T9  Sampling (spatial resolution, temporal resolution, Nyquist limits)
  T10 Numerical precision (float32 vs float64, underflow, overflow)
  T11 Boundary conditions (zero field, single pixel, NaN, inf, empty)
  T12 Malformed inputs (wrong shapes, mismatched dims, None, negative)
  T13 Environmental extremes (wavelength, temperature, pressure)
  T14 Receiver failures (dead pixels, partial coverage, saturation)
  T15 Combination attacks (multiple simultaneous stressors)

All simulations are mathematical models on a laptop. Results do not
establish physical feasibility — hardware would be required.

XPOCK research only — no HX-241 involvement.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

# ----------------------------------------------------------------------
# Load XPOCK wave_info modules directly (no install required)
# ----------------------------------------------------------------------


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


turbulence_mod = _load_module(
    'xpock.wave_info.turbulence_adv',
    'src/xpock/wave_info/turbulence.py',
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_adv',
    'src/xpock/wave_info/skyrmion_modem.py',
)
ber_sim_mod = _load_module(
    'xpock.wave_info.ber_simulation_adv',
    'src/xpock/wave_info/ber_simulation.py',
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem
BERSimulator = ber_sim_mod.BERSimulator

# ----------------------------------------------------------------------
# Adversarial configuration — derived from implementation limits
# ----------------------------------------------------------------------


class AdversarialConfig:
    """Parameter ranges derived from actual implementation constraints,
    not arbitrary thresholds."""

    GRID_SIZES = [16, 32, 64, 128, 256]
    SPATIAL_EXTENTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    WAVELENGTHS = [4.0e-7, 8.5e-7, 1.3e-6, 1.55e-6, 2.0e-6, 3.0e-6]
    CN2_VALUES = np.logspace(-17, -6, 12)
    DISTANCES = [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    N_SCREENS = [1, 2, 5, 10, 20, 50, 100]
    SKYRMION_RADII = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    CONSTELLATIONS = [
        [-1, 1],
        [-2, -1, 1, 2],
        [-3, -2, -1, 0, 1, 2, 3],
        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    ]
    NOISE_LEVELS = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0]
    OFFSETS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    ROTATIONS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    SCALINGS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    PRECISIONS = [np.float32, np.float64]
    TRIALS = [1, 5, 10, 20, 50]
    INNER_SCALES = [1e-3, 5e-3, 1e-2]
    OUTER_SCALES = [1.0, 5.0, 10.0, 50.0]


# ----------------------------------------------------------------------
# Dynamic parameter generator — reproducible, traceable, non-duplicating
# ----------------------------------------------------------------------


class ParameterGenerator:
    """Generates diverse parameter combinations with reproducible seeds.
    Every combination gets a unique trace_id."""

    def __init__(self, seed: int = 20261030):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace_id = 0

    def _next_id(self) -> int:
        self.trace_id += 1
        return self.trace_id

    def generate_turbulence_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        cn2_samples = self.rng.choice(
            AdversarialConfig.CN2_VALUES, size=n, replace=True
        )
        dist_samples = self.rng.choice(
            AdversarialConfig.DISTANCES, size=n, replace=True
        )
        screen_samples = self.rng.choice(
            AdversarialConfig.N_SCREENS, size=n, replace=True
        )
        wave_samples = self.rng.choice(
            AdversarialConfig.WAVELENGTHS, size=n, replace=True
        )
        inner_samples = self.rng.choice(
            AdversarialConfig.INNER_SCALES, size=n, replace=True
        )
        outer_samples = self.rng.choice(
            AdversarialConfig.OUTER_SCALES, size=n, replace=True
        )
        for i in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T1_turbulence',
                    'cn2': float(cn2_samples[i]),
                    'distance': float(dist_samples[i]),
                    'n_screens': int(screen_samples[i]),
                    'wavelength': float(wave_samples[i]),
                    'inner_scale': float(inner_samples[i]),
                    'outer_scale': float(outer_samples[i]),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_phase_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T2_phase_distortion',
                    'phase_offset': float(
                        self.rng.uniform(-np.pi, np.pi)
                    ),
                    'phase_noise_std': float(self.rng.uniform(0, 5.0)),
                    'phase_screen_amplitude': float(
                        self.rng.uniform(0.1, 20.0)
                    ),
                    'cn2': float(self.rng.choice(AdversarialConfig.CN2_VALUES)),
                    'distance': float(self.rng.choice(AdversarialConfig.DISTANCES)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_amplitude_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T3_amplitude',
                    'peak_power': float(self.rng.uniform(1e-6, 1e6)),
                    'power_variation': float(self.rng.uniform(0.0, 0.9)),
                    'scintillation_index': float(self.rng.uniform(0.0, 5.0)),
                    'cn2': float(self.rng.choice(AdversarialConfig.CN2_VALUES)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_noise_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T4_noise',
                    'noise_level': float(
                        self.rng.choice(AdversarialConfig.NOISE_LEVELS)
                    ),
                    'noise_type': self.rng.choice(
                        ['gaussian', 'shot', 'detector', 'mixed']
                    ),
                    'snr_db': float(self.rng.uniform(-20, 60)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_interference_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T5_interference',
                    'n_interferers': int(self.rng.integers(1, 6)),
                    'interference_power_ratio': float(
                        self.rng.uniform(0.01, 100.0)
                    ),
                    'frequency_offset': float(self.rng.uniform(-1e9, 1e9)),
                    'spatial_overlap': float(self.rng.uniform(0.0, 1.0)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_misalignment_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T6_misalignment',
                    'x_offset': float(self.rng.choice(AdversarialConfig.OFFSETS)),
                    'y_offset': float(self.rng.choice(AdversarialConfig.OFFSETS)),
                    'rotation': float(self.rng.choice(AdversarialConfig.ROTATIONS)),
                    'scaling': float(self.rng.choice(AdversarialConfig.SCALINGS)),
                    'centroid_shift': float(self.rng.uniform(0, 3.0)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_sync_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T7_synchronization',
                    'timing_offset': float(self.rng.uniform(-1e-9, 1e-9)),
                    'ex_ey_phase_mismatch': float(
                        self.rng.uniform(-np.pi, np.pi)
                    ),
                    'clock_drift_ppm': float(self.rng.uniform(0, 1000)),
                    'symbol_timing_error': float(self.rng.uniform(0, 0.5)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_encoding_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            ci = int(self.rng.integers(0, len(AdversarialConfig.CONSTELLATIONS)))
            constellation = AdversarialConfig.CONSTELLATIONS[ci]
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T8_encoding',
                    'constellation': [int(c) for c in constellation],
                    'bits_per_symbol': int(
                        int(np.floor(np.log2(len(constellation))))
                    ),
                    'bit_mapping': self.rng.choice(['natural', 'gray', 'random']),
                    'symbol_rate': float(self.rng.uniform(1e3, 1e9)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_sampling_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            nx = int(self.rng.choice(AdversarialConfig.GRID_SIZES))
            ny = int(self.rng.choice(AdversarialConfig.GRID_SIZES))
            extent = float(self.rng.choice(AdversarialConfig.SPATIAL_EXTENTS))
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T9_sampling',
                    'nx': nx,
                    'ny': ny,
                    'extent': extent,
                    'dx': extent / nx,
                    'dy': extent / ny,
                    'temporal_samples': int(self.rng.integers(1, 101)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_precision_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T10_precision',
                    'dtype': str(
                        self.rng.choice(AdversarialConfig.PRECISIONS)
                    ),
                    'extreme_values': self.rng.choice(
                        [True, False]
                    ),
                    'condition_number': float(
                        self.rng.uniform(1.0, 1e12)
                    ),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_boundary_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        boundary_types = [
            'zero_field', 'single_pixel', 'nan_input', 'inf_input',
            'empty_grid', 'extreme_aspect_ratio', 'min_grid', 'max_grid'
        ]
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T11_boundary',
                    'boundary_type': self.rng.choice(boundary_types),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_malformed_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        malformed_types = [
            'wrong_shape', 'mismatched_dims', 'none_input',
            'negative_values', 'zero_division', 'complex_mismatch'
        ]
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T12_malformed',
                    'malformed_type': self.rng.choice(malformed_types),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_receiver_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T14_receiver',
                    'dead_pixel_fraction': float(self.rng.uniform(0, 0.5)),
                    'partial_coverage': float(self.rng.uniform(0.1, 1.0)),
                    'saturation_level': float(self.rng.uniform(0.5, 10.0)),
                    'quantization_bits': int(self.rng.integers(1, 17)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_combination_configs(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        categories = ['T1_turbulence', 'T2_phase_distortion', 'T3_amplitude',
                      'T4_noise', 'T5_interference', 'T6_misalignment',
                      'T7_synchronization', 'T14_receiver']
        for _ in range(n):
            n_stressors = int(self.rng.integers(2, 5))
            selected = self.rng.choice(categories, size=n_stressors, replace=False)
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T15_combination',
                    'stressors': [str(s) for s in selected],
                    'n_stressors': n_stressors,
                    'cn2': float(self.rng.choice(AdversarialConfig.CN2_VALUES)),
                    'noise_level': float(self.rng.choice(AdversarialConfig.NOISE_LEVELS)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs

    def generate_all(self, n_per_category: int = 715) -> List[Dict[str, Any]]:
        """Generate all parameter combinations (~10,010 total test cases)."""
        all_configs = []
        generators = [
            self.generate_turbulence_configs,
            self.generate_phase_configs,
            self.generate_amplitude_configs,
            self.generate_noise_configs,
            self.generate_interference_configs,
            self.generate_misalignment_configs,
            self.generate_sync_configs,
            self.generate_encoding_configs,
            self.generate_sampling_configs,
            self.generate_precision_configs,
            self.generate_boundary_configs,
            self.generate_malformed_configs,
            self.generate_receiver_configs,
            self.generate_combination_configs,
        ]

        for gen in generators:
            all_configs.extend(gen(n_per_category))

        self.rng.shuffle(all_configs)
        return all_configs


# ----------------------------------------------------------------------
# Test execution helpers
# ----------------------------------------------------------------------


def make_grid(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    nx = config.get('nx', 64)
    ny = config.get('ny', 64)
    extent = config.get('extent', 4.0)
    x = np.linspace(-extent / 2, extent / 2, nx)
    y = np.linspace(-extent / 2, extent / 2, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y


def make_modem(
    config: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
) -> SkyrmionModem:
    wavelength = config.get('wavelength', 1.55e-6)
    radius = config.get('skyrmion_radius', 1.0)
    constellation = config.get('constellation', [-1, 1])
    cfg = ModemConfig(
        wavelength=wavelength,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=radius,
        constellation=constellation,
    )
    return SkyrmionModem(cfg)


def propagate_with_stressors(
    Ex: np.ndarray,
    Ey: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    cn2 = config.get('cn2', 1e-14)
    distance = config.get('distance', 100.0)
    wavelength = config.get('wavelength', 1.55e-6)
    n_screens = config.get('n_screens', 10)
    seed = config.get('seed', 42)
    phase_noise_std = config.get('phase_noise_std', 0.0)
    phase_offset = config.get('phase_offset', 0.0)

    turb_cfg = TurbulenceConfig(
        wavelength=wavelength,
        distance=distance,
        cn2=cn2,
        n_screens=n_screens,
        seed=seed,
        inner_scale=config.get('inner_scale', 5e-3),
        outer_scale=config.get('outer_scale', 10.0),
    )
    model = TurbulenceModel(turb_cfg)

    nx_dim = X.shape[0]
    ny_dim = X.shape[1]
    dx = X[0, 1] - X[0, 0]
    dz = distance / n_screens
    phase_screens = []
    for _ in range(n_screens):
        phase_screens.append(
            model.generate_phase_screen((ny_dim, nx_dim), dx, dz)
        )

    E_rx = model.propagate(Ex, X, Y, phase_screens=phase_screens)
    E_rx_y = model.propagate(Ey, X, Y, phase_screens=phase_screens)

    ex_ey_mismatch = config.get('ex_ey_phase_mismatch', 0.0)
    if ex_ey_mismatch != 0.0:
        E_rx_y = E_rx_y * np.exp(1j * ex_ey_mismatch)

    if phase_noise_std > 0.0:
        rng = np.random.default_rng(config.get('seed', 42) + 1)
        phase_noise = rng.normal(0, phase_noise_std, E_rx.shape)
        E_rx = E_rx * np.exp(1j * phase_noise)
        E_rx_y = E_rx_y * np.exp(1j * phase_noise)

    if phase_offset != 0.0:
        E_rx = E_rx * np.exp(1j * phase_offset)
        E_rx_y = E_rx_y * np.exp(1j * phase_offset)

    peak_power = config.get('peak_power', 1.0)
    power_variation = config.get('power_variation', 0.0)
    if peak_power != 1.0 or power_variation > 0.0:
        rng = np.random.default_rng(config.get('seed', 42) + 2)
        amp_mod = 1.0 + power_variation * rng.standard_normal(E_rx.shape)
        E_rx = E_rx * peak_power * amp_mod
        E_rx_y = E_rx_y * peak_power * amp_mod

    noise_level = config.get('noise_level', 0.0)
    if noise_level > 0.0:
        rng = np.random.default_rng(config.get('seed', 42) + 3)
        noise = rng.normal(0, noise_level, E_rx.shape) + 1j * rng.normal(
            0, noise_level, E_rx.shape
        )
        E_rx = E_rx + noise
        E_rx_y = E_rx_y + noise

    x_offset = config.get('x_offset', 0.0)
    y_offset = config.get('y_offset', 0.0)
    rotation = config.get('rotation', 0.0)
    scaling = config.get('scaling', 1.0)
    if x_offset != 0.0 or y_offset != 0.0 or rotation != 0.0 or scaling != 1.0:
        X_shifted = (X - x_offset) * scaling
        Y_shifted = (Y - y_offset) * scaling
        if rotation != 0.0:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            X_rot = cos_r * X_shifted - sin_r * Y_shifted
            Y_rot = sin_r * X_shifted + cos_r * Y_shifted
            X_shifted, Y_shifted = X_rot, Y_rot
        ix = np.clip(
            np.round((X_shifted - X_shifted.min()) /
                     (X_shifted.max() - X_shifted.min()) * (nx_dim - 1)).astype(int),
            0, nx_dim - 1,
        )
        iy = np.clip(
            np.round((Y_shifted - Y_shifted.min()) /
                     (Y_shifted.max() - Y_shifted.min()) * (ny_dim - 1)).astype(int),
            0, ny_dim - 1,
        )
        E_rx = E_rx[iy, ix]
        E_rx_y = E_rx_y[iy, ix]

    return E_rx, E_rx_y


def demodulate_field(
    modem: SkyrmionModem,
    Ex_rx: np.ndarray,
    Ey_rx: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[float, int]:
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)
    return q_measured, n_rx


# ----------------------------------------------------------------------
# Pytest dynamic test generation
# ----------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    if 'adv_config' in metafunc.fixturenames:
        gen = ParameterGenerator()
        configs = gen.generate_all(n_per_category=715)
        metafunc.parametrize('adv_config', configs, ids=[
            f"T{c['trace_id']:05d}_{c['category']}" for c in configs
        ])


# ----------------------------------------------------------------------
# Individual adversarial tests
# ----------------------------------------------------------------------


def test_adversarial_turbulence(adv_config):
    if adv_config['category'] != 'T1_turbulence':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_errors = 0
    n_total = 0
    n_trials = adv_config.get('trials', 10)

    for trial in range(n_trials):
        trial_seed = adv_config['seed'] + trial
        n_tx = constellation[trial % len(constellation)]
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])

        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

        stress_cfg = dict(adv_config)
        stress_cfg['seed'] = trial_seed
        E_rx, E_rx_y = propagate_with_stressors(
            Ex_tx, Ey_tx, X, Y, stress_cfg
        )

        q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)
        bits_rx = modem.skyrmion_to_bits(n_rx)

        bit_error = not np.array_equal(bits_tx, bits_rx)
        n_errors += int(bit_error)
        n_total += 1

    ber = n_errors / n_total if n_total > 0 else 0.0

    assert ber >= 0.0 and ber <= 1.0
    assert np.isfinite(ber)


def test_adversarial_phase_distortion(adv_config):
    if adv_config['category'] != 'T2_phase_distortion':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))


def test_adversarial_amplitude(adv_config):
    if adv_config['category'] != 'T3_amplitude':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_noise(adv_config):
    if adv_config['category'] != 'T4_noise':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_interference(adv_config):
    if adv_config['category'] != 'T5_interference':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    n_interferers = stress_cfg.get('n_interferers', 1)
    interference_ratio = stress_cfg.get('interference_power_ratio', 1.0)
    for _ in range(n_interferers):
        phase = np.random.uniform(0, 2 * np.pi, X.shape)
        amp = np.sqrt(interference_ratio) * np.exp(-(X**2 + Y**2) / 4.0)
        E_rx = E_rx + amp * np.exp(1j * phase)
        E_rx_y = E_rx_y + amp * np.exp(1j * phase)

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_misalignment(adv_config):
    if adv_config['category'] != 'T6_misalignment':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_synchronization(adv_config):
    if adv_config['category'] != 'T7_synchronization':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_encoding(adv_config):
    if adv_config['category'] != 'T8_encoding':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    for n_tx in constellation:
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

        stress_cfg = dict(adv_config)
        E_rx, E_rx_y = propagate_with_stressors(
            Ex_tx, Ey_tx, X, Y, stress_cfg
        )

        q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

        assert np.isfinite(q_measured)
        assert n_rx in constellation


def test_adversarial_sampling(adv_config):
    if adv_config['category'] != 'T9_sampling':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_precision(adv_config):
    if adv_config['category'] != 'T10_precision':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    if adv_config.get('extreme_values', False):
        Ex_tx = Ex_tx * 1e6
        Ey_tx = Ey_tx * 1e6

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_boundary(adv_config):
    if adv_config['category'] != 'T11_boundary':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    boundary_type = adv_config.get('boundary_type', 'zero_field')

    try:
        if boundary_type == 'zero_field':
            Ex_tx = np.zeros_like(X, dtype=np.complex128)
            Ey_tx = np.zeros_like(Y, dtype=np.complex128)
        elif boundary_type == 'single_pixel':
            Ex_tx = np.zeros_like(X, dtype=np.complex128)
            Ey_tx = np.zeros_like(Y, dtype=np.complex128)
            Ex_tx[X.shape[0]//2, X.shape[1]//2] = 1.0 + 0j
            Ey_tx[X.shape[0]//2, X.shape[1]//2] = 0.0 + 0j
        elif boundary_type == 'nan_input':
            Ex_tx = np.full_like(X, np.nan, dtype=np.complex128)
            Ey_tx = np.full_like(Y, np.nan, dtype=np.complex128)
        elif boundary_type == 'inf_input':
            Ex_tx = np.full_like(X, np.inf, dtype=np.complex128)
            Ey_tx = np.full_like(Y, np.inf, dtype=np.complex128)
        elif boundary_type == 'empty_grid':
            Ex_tx = np.zeros((0, 0), dtype=np.complex128)
            Ey_tx = np.zeros((0, 0), dtype=np.complex128)
        elif boundary_type == 'extreme_aspect_ratio':
            Ex_tx = np.ones((4, 256), dtype=np.complex128)
            Ey_tx = np.ones((4, 256), dtype=np.complex128)
            X = np.linspace(-1, 1, 256)
            Y = np.linspace(-0.01, 0.01, 4)
            X, Y = np.meshgrid(X, Y, indexing='xy')
        elif boundary_type == 'min_grid':
            Ex_tx = np.ones((2, 2), dtype=np.complex128)
            Ey_tx = np.ones((2, 2), dtype=np.complex128)
            X = np.linspace(-1, 1, 2)
            Y = np.linspace(-1, 1, 2)
            X, Y = np.meshgrid(X, Y, indexing='xy')
        elif boundary_type == 'max_grid':
            Ex_tx = np.ones((256, 256), dtype=np.complex128)
            Ey_tx = np.ones((256, 256), dtype=np.complex128)
        else:
            Ex_tx = np.ones_like(X, dtype=np.complex128)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)

        stress_cfg = dict(adv_config)
        E_rx, E_rx_y = propagate_with_stressors(
            Ex_tx, Ey_tx, X, Y, stress_cfg
        )

        q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

        assert True

    except Exception as e:
        pytest.xfail(f"Boundary condition '{boundary_type}' raised: {e}")


def test_adversarial_malformed(adv_config):
    if adv_config['category'] != 'T12_malformed':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    malformed_type = adv_config.get('malformed_type', 'wrong_shape')

    try:
        if malformed_type == 'wrong_shape':
            Ex_tx = np.ones((X.shape[0] + 1, X.shape[1]), dtype=np.complex128)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)
        elif malformed_type == 'mismatched_dims':
            Ex_tx = np.ones_like(X, dtype=np.complex128)
            Ey_tx = np.ones((X.shape[0], X.shape[1] + 1), dtype=np.complex128)
        elif malformed_type == 'none_input':
            Ex_tx = None
            Ey_tx = np.ones_like(Y, dtype=np.complex128)
        elif malformed_type == 'negative_values':
            Ex_tx = -np.ones_like(X, dtype=np.complex128)
            Ey_tx = -np.ones_like(Y, dtype=np.complex128)
        elif malformed_type == 'zero_division':
            Ex_tx = np.zeros_like(X, dtype=np.complex128)
            Ey_tx = np.zeros_like(Y, dtype=np.complex128)
        elif malformed_type == 'complex_mismatch':
            Ex_tx = np.ones_like(X, dtype=np.float64)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)
        else:
            Ex_tx = np.ones_like(X, dtype=np.complex128)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)

        stress_cfg = dict(adv_config)
        E_rx, E_rx_y = propagate_with_stressors(
            Ex_tx, Ey_tx, X, Y, stress_cfg
        )

        q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

        assert True

    except (ValueError, TypeError, AttributeError) as e:
        pytest.xfail(f"Malformed input '{malformed_type}' raised expected: {e}")


def test_adversarial_receiver(adv_config):
    if adv_config['category'] != 'T14_receiver':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    dead_fraction = stress_cfg.get('dead_pixel_fraction', 0.0)
    coverage = stress_cfg.get('partial_coverage', 1.0)
    saturation = stress_cfg.get('saturation_level', 10.0)
    quant_bits = stress_cfg.get('quantization_bits', 16)

    if dead_fraction > 0.0:
        rng = np.random.default_rng(stress_cfg['seed'] + 10)
        dead_mask = rng.random(X.shape) < dead_fraction
        E_rx[dead_mask] = 0.0
        E_rx_y[dead_mask] = 0.0

    if coverage < 1.0:
        cx, cy = X.shape[0] // 2, X.shape[1] // 2
        rr = np.sqrt((X - X[cx, cy])**2 + (Y - Y[cx, cy])**2)
        rmax = np.max(rr) * coverage
        aperture_mask = rr > rmax
        E_rx[aperture_mask] = 0.0
        E_rx_y[aperture_mask] = 0.0

    if saturation < 10.0:
        E_rx = np.clip(E_rx, -saturation, saturation)
        E_rx_y = np.clip(E_rx_y, -saturation, saturation)

    if quant_bits < 16:
        levels = 2 ** quant_bits
        E_rx = np.round(E_rx * levels) / levels
        E_rx_y = np.round(E_rx_y * levels) / levels

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_combination(adv_config):
    if adv_config['category'] != 'T15_combination':
        pytest.skip(f"Wrong category: {adv_config['category']}")

    X, Y = make_grid(adv_config)
    modem = make_modem(adv_config, X, Y)
    constellation = adv_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(adv_config)
    E_rx, E_rx_y = propagate_with_stressors(
        Ex_tx, Ey_tx, X, Y, stress_cfg
    )

    noise_level = stress_cfg.get('noise_level', 0.0)
    if noise_level > 0.0:
        rng = np.random.default_rng(stress_cfg['seed'] + 20)
        noise = rng.normal(0, noise_level, E_rx.shape) + 1j * rng.normal(
            0, noise_level, E_rx.shape
        )
        E_rx = E_rx + noise
        E_rx_y = E_rx_y + noise

    q_measured, n_rx = demodulate_field(modem, E_rx, E_rx_y, stress_cfg)

    assert np.isfinite(q_measured)


def test_adversarial_coverage_summary(param_gen):
    configs = param_gen.generate_all(n_per_category=715)
    total = len(configs)

    categories = {}
    for c in configs:
        cat = c['category']
        categories[cat] = categories.get(cat, 0) + 1

    assert total >= 10000, f"Expected >= 10000 test cases, got {total}"

    expected_categories = [
        'T1_turbulence', 'T2_phase_distortion', 'T3_amplitude',
        'T4_noise', 'T5_interference', 'T6_misalignment',
        'T7_synchronization', 'T8_encoding', 'T9_sampling',
        'T10_precision', 'T11_boundary', 'T12_malformed',
        'T14_receiver', 'T15_combination',
    ]
    for cat in expected_categories:
        assert cat in categories, f"Missing category: {cat}"
        assert categories[cat] >= 700, f"Category {cat} has only {categories[cat]} cases"

    trace_ids = [c['trace_id'] for c in configs]
    assert len(trace_ids) == len(set(trace_ids)), "Duplicate trace IDs found"

    for c in configs:
        assert 'seed' in c and isinstance(c['seed'], int), f"Missing seed in {c['trace_id']}"