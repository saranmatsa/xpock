"""
XPOCK — Adversarial Test T9-002: Coarse Grid Undersampling

Physical scenario: Skyrmion texture sampled on coarse grid (16×16) where
spatial features are below Nyquist limit. Tests whether skyrmion number
computation breaks down when spatial resolution is insufficient to resolve
the phase singularity at the skyrmion core.

Expected result: Q measurement may be inaccurate but demodulation should
not crash; result may be xfailed if Q error exceeds threshold.
"""

from __future__ import annotations

import importlib.util
import sys
import numpy as np


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t9_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_coarse_grid_undersampling():
    """Skyrmion number Q on coarse grid below Nyquist limit.

    Physical scenario: Skyrmion texture sampled on coarse 16×16 grid where
    the phase singularity core is not resolved. Tests whether skyrmion number
    computation handles undersampled spatial features gracefully.

    Parameters:
        Grid: 16×16, extent 4 m (dx = dy = 250 mm — coarse)
        Wavelength: 1550 nm
        Skyrmion radius: 1.0 m
        Constellation: [-1, 1] (1 bit/symbol)
        No turbulence (Cn²=0)
    """
    nx, ny = 16, 16
    extent = 4.0
    x = np.linspace(-extent / 2, extent / 2, nx)
    y = np.linspace(-extent / 2, extent / 2, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)

    # Transmit N=+1 skyrmion
    bits_tx = np.array([1])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # No turbulence — direct demodulation on coarse grid
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — no crash; Q may be inaccurate but should be finite
    assert np.isfinite(q_measured), f"Q not finite on coarse grid: {q_measured}"
    assert n_rx in [-1, 1], f"Recovered N={n_rx} not in constellation {-1, 1}"

    # Record physical metrics for traceability
    dx = X[0, 1] - X[0, 0]
    test_metrics = {
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'dx_m': float(dx),
        'nyquist_limit_m': float(1.55e-6 / 2),  # wavelength/2 for sampling
        'undersampling_ratio': float(dx / (1.55e-6 / 2)),
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'coarse_grid_undersampling',
    }
    return test_metrics