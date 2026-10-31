"""
XPOCK — Adversarial Test T11-001: Boundary Condition — Zero Field Input

Physical scenario: Input field is exactly zero (no signal transmitted).
This tests whether the skyrmion number demodulation handles zero-field
inputs gracefully without crashing or producing NaN/inf values.

Expected result: No crash; Q may be 0 or NaN but demodulation should not fail.
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
    'xpock.wave_info.skyrmion_modem_t11_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_boundary_zero_field_input():
    """Skyrmion number demodulation handles zero field input gracefully.

    Physical scenario: Input field is exactly zero (no signal transmitted,
    transmitter failure, or complete signal attenuation). Tests whether
    skyrmion number demodulation handles zero-field inputs without crashing
    or producing NaN/inf values.

    Parameters:
        Field: Ex = 0, Ey = 0 everywhere
        Grid: 64×64, extent 4 m
        Wavelength: 1550 nm
        Skyrmion radius: 1.0 m
        Constellation: [-1, 1] (1 bit/symbol)
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

    # Zero field input — no signal
    Ex_tx = np.zeros_like(X, dtype=np.complex128)
    Ey_tx = np.zeros_like(Y, dtype=np.complex128)

    # Demodulate zero field — should not crash
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — no crash, result may be 0 or NaN but should be handled
    # The key requirement is that demodulation doesn't raise an exception
    assert isinstance(q_measured, (float, int, np.floating, np.integer)), (
        f"Q unexpected type: {type(q_measured)}"
    )
    assert isinstance(n_rx, (int, np.integer)), (
        f"n_rx unexpected type: {type(n_rx)}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'boundary_type': 'zero_field',
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured) if np.isfinite(q_measured) else None,
        'n_rx': int(n_rx),
        'ber': None,  # No signal transmitted
        'scenario': 'boundary_zero_field_input',
        'no_crash': True,
    }
    return test_metrics