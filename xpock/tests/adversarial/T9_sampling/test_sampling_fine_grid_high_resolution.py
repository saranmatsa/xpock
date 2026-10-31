"""
XPOCK — Adversarial Test T9-001: Sampling — Fine Grid High-Resolution

Physical scenario: Skyrmion texture sampled on fine grid (256×256) over
small extent (2 m). High spatial resolution tests whether skyrmion number
computation converges with grid refinement. Fine grids capture more detail
of the phase singularity but increase computational cost.

Expected result: Q converges to correct value on fine grid, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t9_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_sampling_fine_grid_high_resolution():
    """Skyrmion number Q converges on fine grid (256×256, 2 m extent).

    Physical scenario: Skyrmion texture sampled on fine grid to test numerical
    convergence of skyrmion number computation. Fine grids (256×256) capture
    more detail of the phase singularity at the core but increase computational
    cost. Q should converge to the correct integer value as grid resolution
    increases.

    Parameters:
        Grid: 256×256, extent 2 m (dx = dy = 7.8 mm)
        Wavelength: 1550 nm
        Skyrmion radius: 1.0 m
        Constellation: [-1, 1] (1 bit/symbol)
        No turbulence (Cn²=0)
    """
    nx, ny = 256, 256
    extent = 2.0
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

    # No turbulence — direct demodulation on fine grid
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — Q should converge on fine grid
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.1, (
        f"Skyrmion number not converged on fine grid: Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    test_metrics = {
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'dx_m': float(dx),
        'dy_m': float(dy),
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'sampling_fine_grid_high_resolution',
    }
    return test_metrics