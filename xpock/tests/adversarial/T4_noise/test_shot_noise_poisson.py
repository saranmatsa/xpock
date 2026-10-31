"""
XPOCK — Adversarial Test T4-002: Shot Noise (Poisson)

Physical scenario: Receiver detector introduces shot noise following Poisson
statistics, proportional to the square root of the signal intensity. This is
fundamental quantum noise present in all optical measurements. Tests whether
skyrmion number demodulation is robust to shot noise at typical light levels.

Expected result: Q preserved within decision boundary under shot noise, BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t4_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_shot_noise_poisson():
    """Skyrmion number Q preserved under Poisson shot noise.

    Physical scenario: Receiver detector introduces shot noise following
    Poisson statistics, where noise variance equals signal intensity.
    This is fundamental quantum noise present in all optical measurements.
    Tests whether skyrmion number computed from normalized Stokes parameters
    is robust to shot noise at typical photon flux levels.

    Parameters:
        Noise type: Shot noise (Poisson)
        Photon flux: 1e6 photons/pixel (typical)
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

    # Apply shot noise: variance proportional to signal intensity
    photon_flux = 1e6  # photons per pixel
    rng = np.random.default_rng(42)

    # Shot noise on amplitude: sqrt(signal) * Gaussian
    signal_intensity = np.abs(Ex_tx) ** 2 + np.abs(Ey_tx) ** 2
    noise_std = np.sqrt(np.maximum(signal_intensity, 0) / photon_flux)
    shot_noise_ex = rng.normal(0, 1, X.shape) * noise_std
    shot_noise_ey = rng.normal(0, 1, X.shape) * noise_std

    Ex_rx = Ex_tx + shot_noise_ex + 1j * shot_noise_ex
    Ey_rx = Ey_tx + shot_noise_ey + 1j * shot_noise_ey

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — Q should be within decision boundary under shot noise
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'noise_type': 'shot_noise_poisson',
        'photon_flux_per_pixel': photon_flux,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'ber': 0.0,
        'scenario': 'shot_noise_poisson',
    }
    return test_metrics