"""
XPOCK — Adversarial Test T3: Amplitude Fluctuations

715 dynamically generated test cases spanning power levels, power variation,
scintillation index. Each case has a unique trace_id for traceability.
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
    'xpock.wave_info.skyrmion_modem_t3_adv',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


class T3ParameterGenerator:
    NOISE_LEVELS = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0]

    def __init__(self, seed: int = 20261032):
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
                    'category': 'T3_amplitude',
                    'peak_power': float(self.rng.uniform(1e-6, 1e6)),
                    'power_variation': float(self.rng.uniform(0.0, 0.9)),
                    'scintillation_index': float(self.rng.uniform(0.0, 5.0)),
                    'cn2': float(self.rng.choice([1e-15, 1e-14, 1e-13])),
                    'seed': int(self.rng.integers(0, 2**31 - 1)),
                }
            )
        return configs


def pytest_generate_tests(metafunc):
    if 't3_config' in metafunc.fixturenames:
        gen = T3ParameterGenerator()
        configs = gen.generate(715)
        metafunc.parametrize('t3_config', configs, ids=[
            f"T3-{c['trace_id']:05d}" for c in configs
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


def test_t3_amplitude(t3_config):
    X, Y = make_grid(t3_config)
    modem = make_modem(t3_config, X, Y)
    constellation = t3_config.get('constellation', [-1, 1])
    n_bits = int(np.floor(np.log2(len(constellation))))

    n_tx = constellation[0]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([bit_idx])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

    stress_cfg = dict(t3_config)
    peak_power = stress_cfg.get('peak_power', 1.0)
    power_variation = stress_cfg.get('power_variation', 0.0)

    if peak_power != 1.0 or power_variation > 0.0:
        rng = np.random.default_rng(stress_cfg['seed'] + 2)
        amp_mod = 1.0 + power_variation * rng.standard_normal(Ex_tx.shape)
        Ex_tx = Ex_tx * peak_power * amp_mod
        Ey_tx = Ey_tx * peak_power * amp_mod

    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    assert np.isfinite(q_measured)
    assert isinstance(n_rx, (int, np.integer))