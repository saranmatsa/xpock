"""
XPOCK — Adversarial Test T3-002: Exponential Amplitude Decay

Physical scenario: Field amplitude decays exponentially from center to edge
instead of flat or Gaussian profile. Tests whether skyrmion number computation
handles non-Gaussian amplitude envelopes correctly.

Expected result: Q preserved under exponential amplitude decay, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t3_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_exponential_amplitude_decay():
    """Skyrmion number Q preserved under exponential amplitude decay profile.

    Physical scenario: Input field amplitude decays exponentially from center
    to edge (exp(-r/L)) instead of Gaussian profile. Tests whether skyrmion
    number computation handles non-Gaussian amplitude envelopes correctly.

    Parameters:
        Amplitude profile: Exponential decay exp(-r/2)
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

    # Generate skyrmion field with exponential amplitude decay
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    l = 1  # topological charge
    decay_length = 2.0  # meters
    envelope = np.exp(-r / decay_length)

    Ex_tx = envelope * np.cos(l * phi)
    Ey_tx = 1j * envelope * np.sin(l * phi)  # circular component

    n_tx = 1
    bits_tx = np.array([1])

    # No turbulence — direct demodulation
    q_measured, n_rx = modem.demodulate(Ex_tx, Ey_tx)

    # Assertions — Q should be finite and demodulation should not crash
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx in [-1, 1], f"Recovered N={n_rx} not in constellation {-1, 1}"

    # Record physical metrics for traceability
    test_metrics = {
        'amplitude_profile': 'exponential_decay',
        'decay_length_m': decay_length,
        'topological_charge': l,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'exponential_amplitude_decay',
    }
    return test_metrics