"""
XPOCK — Adversarial Test T4-001: Noise Injection — Additive Gaussian Noise

Physical scenario: Receiver detector introduces additive Gaussian noise
with noise level 0.01 (relative to signal). This simulates thermal noise,
shot noise, and detector readout noise. Skyrmion number computation from
Stokes parameters should be robust to moderate noise levels.

Expected result: Q preserved within decision boundary, BER = 0 at moderate noise.
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
    'xpock.wave_info.skyrmion_modem_t4_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_noise_injection_additive_gaussian():
    """Skyrmion number Q preserved under additive Gaussian noise (level=0.01).

    Physical scenario: Receiver detector adds Gaussian noise to measured
    field components. Noise level 0.01 represents moderate detector noise
    relative to signal amplitude. Skyrmion number computed from normalized
    Stokes parameters should be robust to this level of noise.

    Parameters:
        Noise level: 0.01 (additive Gaussian)
        Noise type: additive Gaussian
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

    # Apply additive Gaussian noise
    noise_level = 0.01
    rng = np.random.default_rng(42)
    noise_ex = rng.normal(0, noise_level, X.shape) + 1j * rng.normal(
        0, noise_level, X.shape
    )
    noise_ey = rng.normal(0, noise_level, X.shape) + 1j * rng.normal(
        0, noise_level, X.shape
    )
    Ex_rx = Ex_tx + noise_ex
    Ey_rx = Ey_tx + noise_ey

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary at moderate noise
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.5, (
        f"Skyrmion number outside decision boundary under noise: "
        f"Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'noise_level': float(noise_level),
        'noise_type': 'additive_gaussian',
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'noise_injection_additive_gaussian',
    }
    return test_metrics