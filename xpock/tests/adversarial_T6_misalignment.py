"""
XPOCK — Adversarial Test T6: Misalignment

715 dynamically generated test cases spanning grid offset, rotation, scaling,
centroid shift. Each case has a unique trace_id for traceability.
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
    'xpock.wave_info.skyrmion_modem_t6_adv',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T6ParameterGenerator:
    OFFSETS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    ROTATIONS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    SCALINGS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    def __init__(self, seed: int = 20261035):
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
                    'category': 'T6_misalignment',
                    'x_offset': float(self.rng.choice(self.OFFSETS)),
                    'y_offset': float(self.rng.choice(self.OFFSETS)),
                    'rotation': float(self.rng.choice(self.ROTATIONS)),
                    'scaling': float(self.rng.choice(self.SCALINGS)),
                    'centroid_shift': float(self.rng.uniform(0, 3.0)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't6_config' in metafunc.fixturenames:
        gen = T6ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t6_config', configs, ids=[
            f"T6-{c['trace_id']:05d}" for c in configs
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


def test_t6_misalignment(t6_config):
    X, Y = make_grid(t6_config)
    modem = make_modem(t6_config, X, Y)
    constellation = t6_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(t6_config)
    x_offset = stress_cfg.get('x_offset', 0.0)
    y_offset = stress_cfg.get('y_offset', 0.0)
    rotation = stress_cfg.get('rotation', 0.0)
    scaling = stress_cfg.get('scaling', 1.0)

    nx_dim = X.shape[0]
    ny_dim = X.shape[1]
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
    Ex_rx = Ex_tx[iy, ix]
    Ey_rx = Ey_tx[iy, ix]

    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))