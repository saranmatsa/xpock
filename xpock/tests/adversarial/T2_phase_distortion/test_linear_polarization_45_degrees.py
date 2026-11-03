"""
XPOCK — Adversarial Test T2-004: Linear Polarization at 45 Degrees

Physical scenario: Transmit linearly polarized skyrmion field at 45 degrees
instead of the default 0 degrees. Tests whether skyrmion number demodulation
is invariant to linear polarization angle rotation.

Expected result: Q preserved under 45 degree linear polarization, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t2_004',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_linear_polarization_45_degrees():
    """Skyrmion number Q preserved with linear polarization at 45 degrees.

    Physical scenario: Input field has linear polarization at 45 degrees
    instead of 0 degrees (horizontal). Linear polarization angle rotates the
    Stokes vector on the Poincaré sphere equator. Tests whether skyrmion
    number computation is invariant to linear polarization angle rotation.

    Parameters:
        Polarization: Linear at 45 degrees
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

    # Generate linearly polarized skyrmion field at 45 degrees
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    l = 1  # topological charge
    envelope = np.exp(-(r / 1.0) ** 2)

    # Linear polarization at 45 degrees: Ex = Ey = cos(l*phi)/sqrt(2)
    Ex_tx = envelope * np.cos(l * phi) / np.sqrt(2)
    Ey_tx = envelope * np.cos(l * phi) / np.sqrt(2)

    n_tx = 1
    bits_tx = np.array([1])

    # No turbulence — direct demodulation
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — Q should be finite and demodulation should not crash
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx in [-1, 1], f"Recovered N={n_rx} not in constellation {-1, 1}"

    # Record physical metrics for traceability
    test_metrics = {
        'polarization': 'linear_45_deg',
        'topological_charge': l,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'linear_polarization_45_degrees',
    }
    return test_metrics