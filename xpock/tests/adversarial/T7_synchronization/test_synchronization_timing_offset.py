"""
XPOCK — Adversarial Test T7-001: Synchronization — Ex/Ey Timing Offset

Physical scenario: Receiver introduces a timing offset between Ex and Ey
components due to differential group delay in polarization optics. Timing
offset = 100 fs at 1550 nm wavelength. This tests whether skyrmion number
can be recovered when polarization components are temporally misaligned.

Expected result: Q preserved within decision boundary under small timing offset, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t7_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_synchronization_timing_offset():
    """Skyrmion number Q preserved under Ex/Ey timing offset (100 fs).

    Physical scenario: Receiver polarization optics introduce a differential
    group delay between Ex and Ey components. Timing offset = 100 fs at 1550 nm
    wavelength (corresponding to ~15 μm path difference). Skyrmion number
    computed from Stokes parameters should be robust to small timing offsets.

    Parameters:
        Timing offset: 100 fs
        Wavelength: 1550 nm
        Grid: 64×64, extent 4 m
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

    # Apply timing offset as phase shift between Ex and Ey
    # 100 fs at 1550 nm: phase = 2π * c * dt / λ
    timing_offset = 100e-15  # 100 fs
    c_light = 299792458.0
    wavelength = 1.55e-6
    phase_shift = 2 * np.pi * c_light * timing_offset / wavelength

    Ex_rx = Ex_tx
    Ey_rx = Ey_tx * np.exp(1j * phase_shift)

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under timing offset
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.6, (
        f"Skyrmion number outside decision boundary under timing offset: "
        f"Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'timing_offset_s': float(timing_offset),
        'phase_shift_rad': float(phase_shift),
        'wavelength_m': float(wavelength),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'synchronization_timing_offset',
    }
    return test_metrics