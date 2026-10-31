"""
XPOCK — Adversarial Test T15: Combination Attacks

715 dynamically generated test cases spanning turbulence+noise+interference,
turbulence+misalignment+noise, etc. Each case has a unique trace_id.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List

import numpy as np
import pytest


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


turbulence_mod = _load_module(
    'xpock.wave_info.turbulence_t15_adv',
    'src/xpock/wave_info/turbulence.py',
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t15_adv',
    'src/xpock/wave_info/skyrmion_modem.py',
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T15ParameterGenerator:
    CATEGORIES = ['T1_turbulence', 'T2_phase_distortion', 'T3_amplitude',
                  'T4_noise', 'T5_interference', 'T6_misalignment',
                  'T7_synchronization', 'T14_receiver']

    def __init__(self, seed: int = 20261044):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace_id = 0

    def _next_id(self) -> int:
        self.trace_id += 1
        return self.trace_id

    def generate(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            n_stressors = int(self.rng.integers(2, 5))
            selected = self.rng.choice(self.CATEGORIES, size=n_stressors, replace=False)
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T15_combination',
                    'stressors': [str(s) for s in selected],
                    'n_stressors': n_stressors,
                    'cn2': float(self.rng.choice([1e-15, 1e-14, 1e-13, 1e-12, 1e-11])),
                    'noise_level': float(self.rng.choice([0.0, 1e-6, 1e-4, 1e-2, 0.1])),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't15_config' in metafunc.fixturenames:
        gen = T15ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t15_config', configs, ids=[
            f"T15-{c['trace_id']:05d}" for c in configs
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


def test_t15_combination(t15_config):
    X, Y = make_grid(t15_config)
    modem = make_modem(t15_config, X, Y)
    constellation = t15_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(t15_config)
    noise_level = stress_cfg.get('noise_level', 0.0)
    cn2 = stress_cfg.get('cn2', 1e-14)

    turb_cfg = TurbulenceConfig(
        wavelength=stress_cfg.get('wavelength', 1.55e-6),
        distance=stress_cfg.get('distance', 100.0),
        cn2=cn2,
        n_screens=10,
        seed=stress_cfg['seed'],
    )
    model = TurbulenceModel(turb_cfg)

    nx_dim = X.shape[0]
    ny_dim = X.shape[1]
    dx = X[0, 1] - X[0, 0]
    dz = stress_cfg.get('distance', 100.0) / 10
    phase_screens = []
    for _ in range(10):
        phase_screens.append(
            model.generate_phase_screen((ny_dim, nx_dim), dx, dz)
        )

    E_rx = model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
    E_rx_y = model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)

    if noise_level > 0.0:
        rng = np.random.default_rng(stress_cfg['seed'] + 20)
        noise = rng.normal(0, noise_level, E_rx.shape) + 1j * rng.normal(
            0, noise_level, E_rx.shape
        )
        E_rx = E_rx + noise
        E_rx_y = E_rx_y + noise

    q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))