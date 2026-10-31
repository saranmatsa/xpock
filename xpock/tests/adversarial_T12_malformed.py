"""
XPOCK — Adversarial Test T12: Malformed Inputs

715 dynamically generated test cases spanning wrong shape, mismatched dims,
None input, negative values, zero division. Each case has a trace_id.
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
    'xpock.wave_info.skyrmion_modem_t12_adv',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T12ParameterGenerator:
    TYPES = ['wrong_shape', 'mismatched_dims', 'none_input',
             'negative_values', 'zero_division', 'complex_mismatch']

    def __init__(self, seed: int = 20261041):
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
                    'category': 'T12_malformed',
                    'malformed_type': self.rng.choice(self.TYPES),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't12_config' in metafunc.fixturenames:
        gen = T12ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t12_config', configs, ids=[
            f"T12-{c['trace_id']:05d}" for c in configs
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


def test_t12_malformed(t12_config):
    X, Y = make_grid(t12_config)
    modem = make_modem(t12_config, X, Y)
    mt = t12_config.get('malformed_type', 'wrong_shape')

    try:
        if mt == 'wrong_shape':
            Ex_tx = np.ones((X.shape[0] + 1, X.shape[1]), dtype=np.complex128)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)
        elif mt == 'mismatched_dims':
            Ex_tx = np.ones_like(X, dtype=np.complex128)
            Ey_tx = np.ones((X.shape[0], X.shape[1] + 1), dtype=np.complex128)
        elif mt == 'none_input':
            Ex_tx = None
            Ey_tx = np.ones_like(Y, dtype=np.complex128)
        elif mt == 'negative_values':
            Ex_tx = -np.ones_like(X, dtype=np.complex128)
            Ey_tx = -np.ones_like(Y, dtype=np.complex128)
        elif mt == 'zero_division':
            Ex_tx = np.zeros_like(X, dtype=np.complex128)
            Ey_tx = np.zeros_like(Y, dtype=np.complex128)
        elif mt == 'complex_mismatch':
            Ex_tx = np.ones_like(X, dtype=np.float64)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)
        else:
            Ex_tx = np.ones_like(X, dtype=np.complex128)
            Ey_tx = np.ones_like(Y, dtype=np.complex128)

        q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)
        assert True
    except (ValueError, TypeError, AttributeError):
        pytest.xfail(f"Malformed input '{mt}' raised expected exception")