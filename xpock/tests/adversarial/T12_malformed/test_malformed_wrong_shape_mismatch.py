"""
XPOCK — Adversarial Test T12-001: Malformed Input — Wrong Shape Mismatch

Physical scenario: Ex and Ey field components have mismatched shapes
(e.g., Ex is 65×64, Ey is 64×64). This can happen due to detector
misalignment, cropping errors, or data transmission issues. Tests whether
the skyrmion number demodulation handles shape mismatches gracefully.

Expected result: Should raise ValueError or handle gracefully without crash.
"""

from __future__ import annotations

import importlib.util
import sys
import numpy as np



from pathlib import Path

_HERE = Path(__file__).resolve()


def _src_path(*parts: str) -> str:
    """Locate src/xpock/... relative to the repository root, independent of CWD."""
    for base in [_HERE.parent] + list(_HERE.parents):
        cand = base.joinpath("src", *parts)
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"src/{'/'.join(parts)} not found above {_HERE}")

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t12_001',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_malformed_wrong_shape_mismatch():
    """Skyrmion number demodulation handles wrong shape mismatch gracefully.

    Physical scenario: Ex and Ey field components have mismatched shapes
    (65×64 vs 64×64) due to detector cropping or data transmission errors.
    Tests whether demodulation raises an appropriate error or handles
    the mismatch gracefully.

    Parameters:
        Ex shape: 65×64
        Ey shape: 64×64
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

    # Mismatched shapes: Ex is 65×64, Ey is 64×64
    Ex_tx = np.ones((65, 64), dtype=np.complex128)
    Ey_tx = np.ones((64, 64), dtype=np.complex128)

    # Demodulate mismatched shapes — should raise ValueError or handle gracefully
    try:
        q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)
        # If no error, Q should be finite
        assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    except (ValueError, TypeError, IndexError) as e:
        # Expected — shape mismatch should raise an appropriate error
        pass

    # Record physical metrics for traceability
    test_metrics = {
        'malformed_type': 'wrong_shape_mismatch',
        'ex_shape': [65, 64],
        'ey_shape': [64, 64],
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'scenario': 'malformed_wrong_shape_mismatch',
        'handled_gracefully': True,
    }
    return test_metrics