"""
XPOCK — Adversarial Test T6-001: Misalignment — Transmitter-Receiver Grid Offset

Physical scenario: Receiver grid is offset from transmitter grid by 0.5 m in x
and 0.3 m in y due to pointing error or platform vibration. This tests whether
skyrmion number can be recovered when the spatial grid is misaligned.

Expected result: Q preserved within decision boundary under moderate offset, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t6_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_misalignment_grid_offset():
    """Skyrmion number Q preserved under grid offset (x=0.5 m, y=0.3 m).

    Physical scenario: Receiver spatial grid is offset from transmitter grid
    by 0.5 m in x and 0.3 m in y due to pointing error or platform vibration.
    Skyrmion number computed from Stokes parameters should be robust to
    moderate spatial misalignment.

    Parameters:
        X offset: 0.5 m
        Y offset: 0.3 m
        Grid: 64×64, extent 4 m
        Wavelength: 1550 nm
        Skyrmion radius: 1.0 m
        Constellation: [-1, 1] (1 bit/symbol)
        No turbulence (Cn²=0)
    """
    nx, ny = 64, 64
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

    # Transmit N=-1 skyrmion
    bits_tx = np.array([0])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Apply grid offset — shift field coordinates
    x_offset = 0.5
    y_offset = 0.3
    ix = np.clip(
        np.round(((X - x_offset) - X.min()) / (X.max() - X.min()) * (nx - 1)).astype(int),
        0, nx - 1,
    )
    iy = np.clip(
        np.round(((Y - y_offset) - Y.min()) / (Y.max() - Y.min()) * (ny - 1)).astype(int),
        0, ny - 1,
    )
    Ex_rx = Ex_tx[iy, ix]
    Ey_rx = Ey_tx[iy, ix]

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under grid offset
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.6, (
        f"Skyrmion number outside decision boundary under grid offset: "
        f"Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'x_offset_m': float(x_offset),
        'y_offset_m': float(y_offset),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'misalignment_grid_offset',
    }
    return test_metrics