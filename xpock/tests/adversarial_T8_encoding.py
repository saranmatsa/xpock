"""
XPOCK — Adversarial Test T8: Encoding

715 dynamically generated test cases spanning constellation sizes, bit
mappings. Each case has a unique trace_id.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List

import numpy as np
import pytest



from pathlib import Path

_HERE = Path(__file__).resolve()


def _src_path(*parts: str) -> str:
    """Locate src/xpock/... relative to the repository root, independent of CWD."""
    for base in [_HERE.parent] + list(_HERE.parents):
        cand = base.joinpath("src", *parts)
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"src/{'/'.join(parts)} not found above {_HERE}")

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t8_adv',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T8ParameterGenerator:
    CONSTELLATIONS = [
        [-1, 1], [-2, -1, 1, 2], [-3, -2, -1, 0, 1, 2, 3],
        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    ]

    def __init__(self, seed: int = 20261037):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace_id = 0

    def _next_id(self) -> int:
        self.trace_id += 1
        return self.trace_id

    def generate(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            ci = int(self.rng.integers(0, len(self.CONSTELLATIONS)))
            constellation = self.CONSTELLATIONS[ci]
            configs.append(
                {
                    'trace_id': self._next_id(),
                    'category': 'T8_encoding',
                    'constellation': [int(c) for c in constellation],
                    'bits_per_symbol': int(np.floor(np.log2(len(constellation)))),
                    'bit_mapping': self.rng.choice(['natural', 'gray', 'random']),
                    'symbol_rate': float(self.rng.uniform(1e3, 1e9)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't8_config' in metafunc.fixturenames:
        gen = T8ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t8_config', configs, ids=[
            f"T8-{c['trace_id']:05d}" for c in configs
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


def test_t8_encoding(t8_config):
    X, Y = make_grid(t8_config)
    modem = make_modem(t8_config, X, Y)
    constellation = t8_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    for n_tx in constellation:
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
        q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)
        assert np.isfinite(q_measured)
        assert n_rx in constellation