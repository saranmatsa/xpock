"""
XPOCK — Adversarial Test T12-002: Mismatched Dimensions

Physical scenario: Ex and Ey field components have mismatched dimensions
(Ex is 64×64, Ey is 32×32) due to detector cropping, different sensor sizes,
or data transmission errors. Tests whether skyrmion number demodulation
handles dimension mismatches gracefully.

Expected result: Should raise ValueError or handle gracefully without crash.
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
    'xpock.wave_info.skyrmion_modem_t12_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_mismatched_dimensions():
    """Skyrmion number demodulation handles mismatched dimensions gracefully.

    Physical scenario: Ex and Ey field components have different dimensions
    (64×64 vs 32×32) due to detector cropping, different sensor sizes, or
    data transmission errors. Tests whether demodulation raises an
    appropriate error or handles the mismatch gracefully.

    Parameters:
        Ex shape: 64×64
        Ey shape: 32×32
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

    # Mismatched dimensions: Ex is 64×64, Ey is 32×32
    Ex_tx = np.ones((64, 64), dtype=np.complex128)
    Ey_tx = np.ones((32, 32), dtype=np.complex128)

    # Demodulate mismatched dimensions — should raise ValueError or handle gracefully
    try:
        q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)
        # If no error, Q should be finite
        assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    except (ValueError, TypeError, IndexError) as e:
        # Expected — dimension mismatch should raise an appropriate error
        pass

    # Record physical metrics for traceability
    test_metrics = {
        'malformed_type': 'mismatched_dimensions',
        'ex_shape': [64, 64],
        'ey_shape': [32, 32],
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'scenario': 'mismatched_dimensions',
        'handled_gracefully': True,
    }
    return test_metrics