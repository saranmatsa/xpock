"""
XPOCK — Adversarial Test T5: Interference

715 dynamically generated test cases spanning overlapping fields, crosstalk,
multi-user interference. Each case has a unique trace_id for traceability.
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


skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t5_adv',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T5ParameterGenerator:
    def __init__(self, seed: int = 20261034):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace_id = 0

    def _next_id(self) -> int:
        self.trace_id += 1
        return self.trace_id

    def generate(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T5_interference',
                    'n_interferers': int(self.rng.integers(1, 6)),
                    'interference_power_ratio': float(self.rng.uniform(0.01, 100.0)),
                    'frequency_offset': float(self.rng.uniform(-1e9, 1e9)),
                    'spatial_overlap': float(self.rng.uniform(0.0, 1.0)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't5_config' in metafunc.fixturenames:
        gen = T5ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t5_config', configs, ids=[
            f"T5-{c['trace_id']:05d}" for c in configs
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


def test_t5_interference(t5_config):
    X, Y = make_grid(t5_config)
    modem = make_modem(t5_config, X, Y)
    constellation = t5_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(t5_config)
    n_interferers = stress_cfg.get('n_interferers', 1)
    interference_ratio = stress_cfg.get('interference_power_ratio', 1.0)

    rng = np.random.default_rng(stress_cfg['seed'] + 5)
    Ex_rx = Ex_tx.copy()
    Ey_rx = Ey_tx.copy()
    for _ in range(n_interferers):
        phase = rng.uniform(0, 2 * np.pi, X.shape)
        amp = np.sqrt(interference_ratio) * np.exp(-(X**2 + Y**2) / 4.0)
        Ex_rx = Ex_rx + amp * np.exp(1j * phase)
        Ey_rx = Ey_rx + amp * np.exp(1j * phase)

    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))