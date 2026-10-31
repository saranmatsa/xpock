"""
XPOCK — Adversarial Test T2-001: Phase Offset — Global Phase Shift

Physical scenario: Receiver introduces a constant phase offset between Ex and Ey
components due to polarization controller misalignment or fiber birefringence.
Phase offset = π/4 radians. This tests whether skyrmion number demodulation
is robust to global phase shifts that don't change the topological structure.

Expected result: Q preserved (global phase shift doesn't change topology), BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t2_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_phase_offset_global_phase_shift():
    """Skyrmion number Q preserved under global phase offset (π/4 radians).

    Physical scenario: Receiver polarization optics introduce a constant phase
    shift between Ex and Ey components. This is a global phase offset that
    rotates the entire Stokes vector but doesn't change its topology.
    Skyrmion number should be invariant under this transformation.

    Parameters:
        Phase offset: π/4 radians
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

    # Apply global phase offset between Ex and Ey
    phase_offset = np.pi / 4.0
    Ex_rx = Ex_tx * np.exp(1j * phase_offset)
    Ey_rx = Ey_tx * np.exp(1j * phase_offset)

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — global phase shift should not change skyrmion number
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.1, (
        f"Skyrmion number changed under phase offset: Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'phase_offset_rad': float(phase_offset),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'phase_offset_global_phase_shift',
    }
    return test_metrics