"""
XPOCK — Adversarial Test T14: Receiver Failures

715 dynamically generated test cases spanning dead pixels, partial coverage,
saturation. Each case has a unique trace_id.
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
    'xpock.wave_info.skyrmion_modem_t14_adv',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T14ParameterGenerator:
    def __init__(self, seed: int = 20261043):
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
                    'category': 'T14_receiver',
                    'dead_pixel_fraction': float(self.rng.uniform(0, 0.5)),
                    'partial_coverage': float(self.rng.uniform(0.1, 1.0)),
                    'saturation_level': float(self.rng.uniform(0.5, 10.0)),
                    'quantization_bits': int(self.rng.integers(1, 17)),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't14_config' in metafunc.fixturenames:
        gen = T14ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t14_config', configs, ids=[
            f"T14-{c['trace_id']:05d}" for c in configs
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


def test_t14_receiver(t14_config):
    X, Y = make_grid(t14_config)
    modem = make_modem(t14_config, X, Y)
    constellation = t14_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(t14_config)
    dead_fraction = stress_cfg.get('dead_pixel_fraction', 0.0)
    coverage = stress_cfg.get('partial_coverage', 1.0)
    saturation = stress_cfg.get('saturation_level', 10.0)
    quant_bits = stress_cfg.get('quantization_bits', 16)

    if dead_fraction > 0.0:
        rng = np.random.default_rng(stress_cfg['seed'] + 10)
        dead_mask = rng.random(X.shape) < dead_fraction
        Ex_tx[dead_mask] = 0.0
        Ey_tx[dead_mask] = 0.0

    if coverage < 1.0:
        cx, cy = X.shape[0] // 2, X.shape[1] // 2
        rr = np.sqrt((X - X[cx, cy])**2 + (Y - Y[cx, cy])**2)
        rmax = np.max(rr) * coverage
        aperture_mask = rr > rmax
        Ex_tx[aperture_mask] = 0.0
        Ey_tx[aperture_mask] = 0.0

    if saturation < 10.0:
        Ex_tx = np.clip(Ex_tx, -saturation, saturation)
        Ey_tx = np.clip(Ey_tx, -saturation, saturation)

    if quant_bits < 16:
        levels = 2 ** quant_bits
        Ex_tx = np.round(Ex_tx * levels) / levels
        Ey_tx = np.round(Ey_tx * levels) / levels

    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))