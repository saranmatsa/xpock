"""
XPOCK — Adversarial Test T11-002: NaN Input Field

Physical scenario: Input field contains NaN values (not-a-number) due to
detector overflow, corrupted data, or transmission errors. Tests whether
skyrmion number demodulation handles NaN inputs without crashing or
producing infinite results.

Expected result: No crash; Q may be NaN but demodulation should not fail.
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
    'xpock.wave_info.skyrmion_modem_t11_002',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_nan_input_field():
    """Skyrmion number demodulation handles NaN input field gracefully.

    Physical scenario: Input field contains NaN values due to detector
    overflow, corrupted pixel data, or transmission errors. Tests whether
    skyrmion number demodulation handles NaN inputs without crashing.

    Parameters:
        Field: Ex and Ey with NaN in central region
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

    # Field with NaN values in central region
    Ex_tx = np.ones((nx, ny), dtype=np.complex128)
    Ey_tx = np.ones((nx, ny), dtype=np.complex128)
    nan_mask = (np.abs(X) < 0.5) & (np.abs(Y) < 0.5)
    Ex_tx[nan_mask] = np.nan
    Ey_tx[nan_mask] = np.nan

    # Demodulate NaN field — should not crash
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — no crash, result may be NaN
    assert isinstance(q_measured, (float, int, np.floating, np.integer, type(np.nan))), (
        f"Q unexpected type: {type(q_measured)}"
    )
    assert isinstance(n_rx, (int, np.integer)), (
        f"n_rx unexpected type: {type(n_rx)}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'boundary_type': 'nan_input',
        'nan_fraction': float(np.mean(nan_mask)),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured) if np.isfinite(q_measured) else None,
        'n_rx': int(n_rx),
        'ber': None,
        'scenario': 'nan_input_field',
        'no_crash': True,
    }
    return test_metrics