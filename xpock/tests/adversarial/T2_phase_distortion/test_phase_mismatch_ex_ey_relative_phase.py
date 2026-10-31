"""
XPOCK — Adversarial Test T2-002: Ex/Ey Phase Mismatch — Polarization Synchronization Error

Physical scenario: Receiver introduces a relative phase shift between Ex and Ey
components (not a global phase shift). This simulates polarization controller
misalignment or differential group delay between polarization modes.
Phase mismatch = π/2 radians. This tests whether skyrmion number demodulation
is robust to relative phase errors that DO change the Stokes vector structure.

Expected result: Q may change (relative phase shift alters topology), but
BER should still be 0 if Q stays within decision boundary.
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
    'xpock.wave_info.skyrmion_modem_t2_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_phase_mismatch_ex_ey_relative_phase():
    """Skyrmion number Q under Ex/Ey relative phase mismatch (π/2 radians).

    Physical scenario: Receiver polarization optics introduce a relative phase
    shift between Ex and Ey components (differential phase delay). Unlike a
    global phase shift, this changes the relative polarization state and can
    alter the skyrmion number. Tests whether the constellation {-1, 1} is
    robust to this type of synchronization error.

    Parameters:
        Ex/Ey phase mismatch: π/2 radians
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

    # Apply relative phase mismatch between Ex and Ey
    phase_mismatch = np.pi / 2.0
    Ex_rx = Ex_tx  # Ex unchanged
    Ey_rx = Ey_tx * np.exp(1j * phase_mismatch)  # Ey shifted

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — relative phase shift changes Q but should stay in decision region
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx in [-1, 1], f"Recovered N={n_rx} not in constellation {-1, 1}"
    # Q should be within half the constellation spacing (2.0 for {-1, 1})
    assert abs(q_measured - n_tx) < 0.5, (
        f"Skyrmion number outside decision boundary under phase mismatch: "
        f"Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'phase_mismatch_rad': float(phase_mismatch),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'phase_mismatch_ex_ey_relative_phase',
    }
    return test_metrics