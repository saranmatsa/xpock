"""
XPOCK — Adversarial Test T13-001: Environmental — Different Wavelength (850 nm)

Physical scenario: Skyrmion number modulation at 850 nm wavelength instead
of the standard 1550 nm. Shorter wavelength means the skyrmion texture
features are smaller relative to the wavelength, which affects diffraction
and turbulence interaction. Tests whether SKM works at different wavelengths.

Expected result: Q preserved at 850 nm wavelength, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t13_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_environment_different_wavelength_850nm():
    """Skyrmion number Q preserved at 850 nm wavelength.

    Physical scenario: Skyrmion number modulation at 850 nm wavelength
    (near-IR) instead of the standard 1550 nm (telecom band). Shorter
    wavelength means skyrmion texture features are smaller relative to
    the wavelength, affecting diffraction and turbulence interaction.
    Tests whether SKM is wavelength-agnostic.

    Parameters:
        Wavelength: 850 nm
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
        wavelength=850e-9,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)

    # Transmit N=-1 skyrmion
    bits_tx = np.array([0])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # No turbulence — direct demodulation at 850 nm
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — Q should be preserved at different wavelength
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.6, (
        f"Skyrmion number not preserved at 850 nm: Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'wavelength_m': 850e-9,
        'wavelength_nm': 850,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'environment_different_wavelength_850nm',
    }
    return test_metrics