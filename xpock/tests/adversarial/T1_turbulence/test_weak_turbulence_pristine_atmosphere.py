"""
XPOCK — Adversarial Test T1-001: Weak Turbulence — Pristine Atmosphere

Physical scenario: Lab-scale free-space link in pristine atmospheric conditions.
Refractive index structure constant Cn² = 1e-17 m^(-2/3) (extremely weak turbulence,
approaching vacuum). Propagation distance 10 m at 1550 nm wavelength.

This is the best-case environment for skyrmion number modulation — minimal
wavefront distortion. Expected result: skyrmion number Q preserved to within
measurement error, BER = 0.

Tests the fundamental premise: does skyrmion number survive even the gentlest
propagation? This validates the baseline before testing harsher conditions.
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


turbulence_mod = _load_module(
    'xpock.wave_info.turbulence_t1_001',
    'src/xpock/wave_info/turbulence.py',
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t1_001',
    'src/xpock/wave_info/skyrmion_modem.py',
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_weak_turbulence_pristine_atmosphere():
    """Skyrmion number Q preserved through pristine atmosphere (Cn²=1e-17, 10 m).

    Physical scenario: Lab bench free-space optical link in temperature-controlled
    environment. Turbulence is negligible — phase screens add minimal distortion.
    Skyrmion number should be conserved to within numerical precision.

    Parameters:
        Cn²: 1e-17 m^(-2/3) (pristine)
        Distance: 10 m
        Wavelength: 1550 nm
        Grid: 64×64, extent 4 m
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

    # Transmit N=+1 skyrmion
    bits_tx = np.array([1])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Propagate through pristine atmosphere
    turb_cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=10.0,
        cn2=1e-17,
        n_screens=5,
        seed=42,
        inner_scale=5e-3,
        outer_scale=10.0,
    )
    model = TurbulenceModel(turb_cfg)

    dx = X[0, 1] - X[0, 0]
    dz = 10.0 / 5
    phase_screens = []
    for _ in range(5):
        phase_screens.append(
            model.generate_phase_screen((ny, nx), dx, dz)
        )

    E_rx = model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
    E_rx_y = model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)

    # Demodulate
    q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)

    # Assertions — derived from physical expectation
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.1, (
        f"Skyrmion number not preserved: Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'cn2': 1e-17,
        'distance_m': 10.0,
        'wavelength_m': 1.55e-6,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'weak_turbulence_pristine_atmosphere',
    }
    return test_metrics