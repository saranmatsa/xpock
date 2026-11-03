"""
XPOCK — Adversarial Test T9: Sampling

715 dynamically generated test cases spanning grid sizes, spatial extents,
temporal samples. Each case has a unique trace_id.
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
    'xpock.wave_info.skyrmion_modem_t9_adv',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T9ParameterGenerator:
    GRID_SIZES = [16, 32, 64, 128, 256]
    SPATIAL_EXTENTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    def __init__(self, seed: int = 20261038):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace_id = 0

    def _next_id(self) -> int:
        self.trace_id += 1
        return self.trace_id

    def generate(self, n: int) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            nx = int(self.rng.choice(self.GRID_SIZES))
            ny = int(self.rng.choice(self.GRID_SIZES))
            extent = float(self.rng.choice(self.SPATIAL_EXTENTS))
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


def pytest_generate_tests(metafunc):
    if 't9_config' in metafunc.fixturenames:
        gen = T9ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t9_config', configs, ids=[
            f"T9-{c['trace_id']:05d}" for c in configs
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


def test_t9_sampling(t9_config):
    X, Y = make_grid(t9_config)
    modem = make_modem(t9_config, X, Y)
    constellation = t9_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))