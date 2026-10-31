"""
XPOCK — Adversarial Test T14-001: Receiver Failure — Dead Pixels

Physical scenario: Receiver camera has dead pixels (10% of pixels return zero
signal). This simulates damaged detector elements or bad pixels in the
polarimetry camera. Skyrmion number computation from Stokes parameters should
be robust to partial pixel loss as long as enough spatial information remains.

Expected result: Q preserved within decision boundary under 10% dead pixels, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t14_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_receiver_dead_pixels():
    """Skyrmion number Q preserved under 10% dead pixel failure.

    Physical scenario: Receiver camera has 10% dead pixels that return zero
    signal. This simulates damaged detector elements in the polarimetry camera.
    Skyrmion number computed from normalized Stokes parameters should be
    robust to partial pixel loss as long as enough spatial information remains
    to compute the topological invariant.

    Parameters:
        Dead pixel fraction: 0.10 (10%)
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

    # Transmit N=+1 skyrmion
    bits_tx = np.array([1])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Apply dead pixels (10% of pixels zeroed)
    dead_fraction = 0.10
    rng = np.random.default_rng(42)
    dead_mask = rng.random(X.shape) < dead_fraction
    Ex_rx = Ex_tx.copy()
    Ey_rx = Ey_tx.copy()
    Ex_rx[dead_mask] = 0.0
    Ey_rx[dead_mask] = 0.0

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under 10% dead pixels
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.5, (
        f"Skyrmion number outside decision boundary under dead pixels: "
        f"Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'dead_pixel_fraction': float(dead_fraction),
        'dead_pixels': int(np.sum(dead_mask)),
        'total_pixels': int(nx * ny),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'receiver_dead_pixels',
    }
    return test_metrics