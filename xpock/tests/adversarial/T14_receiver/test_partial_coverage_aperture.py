"""
XPOCK — Adversarial Test T14-002: Partial Coverage Aperture

Physical scenario: Receiver has partial coverage — only a circular aperture
in the center captures the field, with the outer region blocked. This simulates
a finite receiver aperture or obstructed optical path. Tests whether skyrmion
number can be recovered from partial spatial coverage.

Expected result: Q preserved within decision boundary under partial coverage, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t14_002',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_partial_coverage_aperture():
    """Skyrmion number Q preserved under partial coverage aperture.

    Physical scenario: Receiver has partial coverage — only a circular aperture
    in the center captures the field (70% radius coverage), with the outer
    region blocked. This simulates a finite receiver aperture or obstructed
    optical path. Tests whether skyrmion number computed from Stokes
    parameters is robust to partial spatial coverage.

    Parameters:
        Coverage: Circular aperture at 70% radius
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

    # Transmit N=-1 skyrmion
    bits_tx = np.array([0])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Apply partial coverage — circular aperture at center
    coverage_radius = 0.7  # fraction of max radius
    cx, cy = nx // 2, ny // 2
    r = np.sqrt((X - X[cx, cy]) ** 2 + (Y - Y[cx, cy]) ** 2)
    r_max = np.max(r)
    aperture_mask = r > coverage_radius * r_max
    Ex_rx = Ex_tx.copy()
    Ey_rx = Ey_tx.copy()
    Ex_rx[aperture_mask] = 0.0
    Ey_rx[aperture_mask] = 0.0

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under partial coverage
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'coverage_radius_fraction': float(coverage_radius),
        'aperture_pixels_blocked': int(np.sum(aperture_mask)),
        'aperture_pixels_total': int(nx * ny),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'partial_coverage_aperture',
    }
    return test_metrics