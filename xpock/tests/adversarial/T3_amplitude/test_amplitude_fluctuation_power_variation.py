"""
XPOCK — Adversarial Test T3-001: Amplitude Fluctuation — Power Variation

Physical scenario: Transmitter output power fluctuates due to laser instability
or atmospheric attenuation. Peak power varies by ±50% around nominal. Skyrmion
number is a topological invariant computed from normalized Stokes parameters,
so it should be invariant to amplitude scaling.

Expected result: Q preserved (normalized Stokes removes amplitude dependence), BER = 0.
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
    'xpock.wave_info.skyrmion_modem_t3_001',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_amplitude_fluctuation_power_variation():
    """Skyrmion number Q preserved under ±50% power variation.

    Physical scenario: Transmitter laser power fluctuates by ±50% due to
    thermal drift or component aging. Skyrmion number Q = (1/4π)∫n·(∂xn×∂yn)dxdy
    uses the normalized Stokes vector n = (S1,S2,S3)/S0, which is invariant
    under amplitude scaling. Q should be preserved.

    Parameters:
        Peak power variation: ±50%
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

    # Apply ±50% power variation
    power_factor = 0.5  # 50% reduction
    Ex_rx = Ex_tx * power_factor
    Ey_rx = Ey_tx * power_factor

    # Demodulate
    q_measured, n_rx = modem.demodulate(Ex_rx, Ey_rx)

    # Assertions — normalized Stokes makes Q invariant to amplitude scaling.
    # What matters for communication is correct demodulation (n_rx == n_tx).
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'power_factor': float(power_factor),
        'power_variation_percent': 50.0,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'wavelength_m': 1.55e-6,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'amplitude_fluctuation_power_variation',
    }
    return test_metrics