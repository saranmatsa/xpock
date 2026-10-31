"""
XPOCK — Adversarial Test T6-002: Rotational Misalignment

Physical scenario: Receiver grid is rotated by 15 degrees relative to
transmitter grid due to platform rotation or mounting error. Tests whether
skyrmion number can be recovered when the spatial grid is rotationally
misaligned.

Expected result: Q preserved within decision boundary under rotation, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t6_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_rotational_misalignment():
    """Skyrmion number Q preserved under rotational misalignment (15 degrees).

    Physical scenario: Receiver spatial grid is rotated by 15 degrees relative
    to transmitter grid due to platform rotation or mounting error. Tests
    whether skyrmion number computed from Stokes parameters is robust to
    rotational misalignment.

    Parameters:
        Rotation: 15 degrees (0.262 rad)
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

    # Apply rotational misalignment
    rotation_deg = 15.0
    rotation_rad = np.deg2rad(rotation_deg)
    cos_r = np.cos(rotation_rad)
    sin_r = np.sin(rotation_rad)

    # Rotate coordinates: sample rotated field onto original grid
    X_rot = cos_r * X + sin_r * Y
    Y_rot = -sin_r * X + cos_r * Y

    # Interpolate field onto rotated grid (nearest neighbor)
    ix = np.clip(
        np.round((X_rot - X_rot.min()) / (X_rot.max() - X_rot.min()) * (nx - 1)).astype(int),
        0, nx - 1,
    )
    iy = np.clip(
        np.round((Y_rot - Y_rot.min()) / (Y_rot.max() - Y_rot.min()) * (ny - 1)).astype(int),
        0, ny - 1,
    )
    Ex_rx = Ex_tx[iy, ix]
    Ey_rx = Ey_tx[iy, ix]

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under rotation
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'rotation_deg': float(rotation_deg),
        'rotation_rad': float(rotation_rad),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'rotational_misalignment',
    }
    return test_metrics