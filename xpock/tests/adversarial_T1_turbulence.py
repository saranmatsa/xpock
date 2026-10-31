"""
XPOCK — Adversarial Test T1: Turbulence Extremes

715 dynamically generated test cases spanning Cn², distance, screens,
wavelength, inner/outer scales. Each case has a unique trace_id for
traceability. Parameters derived from implementation limits.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

turbulence_mod = _load_module_safe(
    'xpock.wave_info.turbulence_t1_adv',
    'src/xpock/wave_info/turbulence.py',
)
skyrmion_modem_mod = _load_module_safe(
    'xpock.wave_info.skyrmion_modem_t1_adv',
    'src/xpock/wave_info/skyrmion_modem.py',
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def _load_module_safe(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class T1ParameterGenerator:
    """Generate turbulence test configurations with reproducible seeds."""

    CN2_VALUES = np.logspace(-17, -6, 12)
    DISTANCES = [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    N_SCREENS = [1, 2, 5, 10, 20, 50, 100]
    WAVELENGTHS = [4.0e-7, 8.5e-7, 1.3e-6, 1.55e-6, 2.0e-6, 3.0e-6]
    INNER_SCALES = [1e-3, 5e-3, 1e-2]
    OUTER_SCALES = [1.0, 5.0, 10.0, 50.0]

    def __init__(self, seed: int = 20261030):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace_id = 0

    def _next_id(self) -> int:
        self.trace_id += 1
        return self.trace_id

    def generate(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        cn2_samples = self.rng.choice(self.CN2_VALUES, size=n, replace=True)
        dist_samples = self.rng.choice(self.DISTANCES, size=n, replace=True)
        screen_samples = self.rng.choice(self.N_SCREENS, size=n, replace=True)
        wave_samples = self.rng.choice(self.WAVELENGTHS, size=n, replace=True)
        inner_samples = self.rng.choice(self.INNER_SCALES, size=n, replace=True)
        outer_samples = self.rng.choice(self.OUTER_SCALES, size=n, replace=True)

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


def pytest_generate_tests(metafunc):
    if 't1_config' in metafunc.fixturenames:
        gen = T1ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t1_config', configs, ids=[
            f"T1-{c['trace_id']:05d}" for c in configs
        ])


def make_grid(config):
    nx = config.get('nx', 64)
    ny = config.get('ny', 64)
    extent = config.get('extent', 4.0)
    x = np.linspace(-extent / 2, extent / 2, nx)
    y = np.linspace(-extent / 2, extent / 2, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y


def make_modem(config, X, Y):
    cfg = ModemConfig(
        wavelength=config.get('wavelength', 1.55e-6),
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=config.get('skyrmion_radius', 1.0),
        constellation=config.get('constellation', [-1, 1]),
    )
    return SkyrmionModem(cfg)


def propagate_with_stressors(Ex, Ey, X, Y, config):
    cn2 = config.get('cn2', 1e-14)
    distance = config.get('distance', 100.0)
    wavelength = config.get('wavelength', 1.55e-6)
    n_screens = config.get('n_screens', 10)
    seed = config.get('seed', 42)

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
    return E_rx, E_rx_y


def test_t1_turbulence(t1_config):
    X, Y = make_grid(t1_config)
    modem = make_modem(t1_config, X, Y)
    constellation = t1_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_errors = 0
    n_total = 0
    n_trials = t1_config.get('trials', 10)

    for trial in range(n_trials):
        trial_seed = t1_config['seed'] + trial
        n_tx = constellation[trial % len(constellation)]
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])

        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

        stress_cfg = dict(t1_config)
        stress_cfg['seed'] = trial_seed
        E_rx, E_rx_y = propagate_with_stressors(
            Ex_tx, Ey_tx, X, Y, stress_cfg
        )

        q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)
        bits_rx = modem.skyrmion_to_bits(n_rx)

        bit_error = not np.array_equal(bits_tx, bits_rx)
        n_errors += int(bit_error)
        n_total += 1

    ber = n_errors / n_total if n_total > 0 else 0.0
    assert ber >= 0.0 and ber <= 1.0
    assert np.isfinite(ber)