"""
XPOCK — Adversarial Test T14-003: Receiver Saturation/Clipping

Physical scenario: Receiver detector saturates at high intensity levels,
clipping the field amplitude. Tests whether skyrmion number computation
handles saturated/clipped input correctly.

Expected result: Q preserved within decision boundary under saturation, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t14_003',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_receiver_saturation_clipping():
    """Skyrmion number Q preserved under receiver saturation/clipping.

    Physical scenario: Receiver detector saturates at high intensity levels,
    clipping the field amplitude to a maximum value. This simulates detector
    saturation or ADC clipping. Tests whether skyrmion number computed from
    Stokes parameters is robust to amplitude clipping.

    Parameters:
        Saturation level: 0.5 (clip at 50% of max amplitude)
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

    # Apply saturation/clipping at 50% amplitude
    saturation_level = 0.5
    Ex_rx = np.clip(Ex_tx, -saturation_level, saturation_level)
    Ey_rx = np.clip(Ey_tx, -saturation_level, saturation_level)

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under saturation
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'saturation_level': float(saturation_level),
        'clipped_pixels_ex': int(np.sum(np.abs(Ex_tx) > saturation_level)),
        'clipped_pixels_ey': int(np.sum(np.abs(Ey_tx) > saturation_level)),
        'total_pixels': int(nx * ny),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'receiver_saturation_clipping',
    }
    return test_metrics