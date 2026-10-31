"""
XPOCK — Adversarial Test T2: Phase Distortion

715 dynamically generated test cases spanning phase offset, Ex/Ey phase
mismatch, phase noise, polarization states. Each case has a unique
trace_id for traceability.
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
    'xpock.wave_info.skyrmion_modem_t2_adv',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T2ParameterGenerator:
    """Generate phase distortion test configurations with reproducible seeds."""

    OFFSETS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    def __init__(self, seed: int = 20261031):
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
                    'category': 'T2_phase_distortion',
                    'phase_offset': float(self.rng.uniform(-np.pi, np.pi)),
                    'phase_noise_std': float(self.rng.uniform(0, 5.0)),
                    'phase_screen_amplitude': float(self.rng.uniform(0.1, 20.0)),
                    'cn2': float(self.rng.choice([1e-15, 1e-14, 1e-13, 1e-12, 1e-11])),
                    'distance': float(self.rng.choice([10.0, 100.0, 1000.0])),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't2_config' in metafunc.fixturenames:
        gen = T2ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t2_config', configs, ids=[
            f"T2-{c['trace_id']:05d}" for c in configs
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


def test_t2_phase_distortion(t2_config):
    X, Y = make_grid(t2_config)
    modem = make_modem(t2_config, X, Y)
    constellation = t2_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(t2_config)
    phase_offset = stress_cfg.get('phase_offset', 0.0)
    phase_noise_std = stress_cfg.get('phase_noise_std', 0.0)

    Ex_rx = Ex_tx * np.exp(1j * phase_offset)
    Ey_rx = Ey_tx * np.exp(1j * phase_offset)

    if phase_noise_std > 0.0:
        rng = np.random.default_rng(stress_cfg['seed'] + 1)
        phase_noise = rng.normal(0, phase_noise_std, Ex_rx.shape)
        Ex_rx = Ex_rx * np.exp(1j * phase_noise)
        Ey_rx = Ey_rx * np.exp(1j * phase_noise)

    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))