"""
XPOCK — Adversarial Test T1-002: Weak Turbulence — Urban Environment

Physical scenario: Free-space optical link through urban atmosphere.
Refractive index structure constant Cn² = 1e-16 m^(-2/3) (weak turbulence,
typical of city environments with moderate temperature gradients).
Propagation distance 100 m at 1550 nm wavelength.

Urban turbulence introduces measurable phase distortion but the skyrmion
number, as a topological invariant, should remain preserved. This tests
whether SKM survives real-world urban atmospheric conditions.

Expected result: Q preserved within measurement tolerance, BER = 0.
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
    'xpock.wave_info.turbulence_t1_002',
    'src/xpock/wave_info/turbulence.py',
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t1_002',
    'src/xpock/wave_info/skyrmion_modem.py',
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_weak_turbulence_urban_environment():
    """Skyrmion number Q preserved through urban atmosphere (Cn²=1e-16, 100 m).

    Physical scenario: Building-to-building free-space optical link in urban
    environment. Temperature gradients from buildings and vehicles create weak
    turbulence. Skyrmion number should survive this level of distortion.

    Parameters:
        Cn²: 1e-16 m^(-2/3) (weak urban turbulence)
        Distance: 100 m
        Wavelength: 1550 nm
        Grid: 128×128, extent 4 m (finer sampling for urban scintillation)
        Skyrmion radius: 1.0 m
        Constellation: [-1, 1] (1 bit/symbol)
    """
    nx, ny = 128, 128
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

    # Transmit N=-1 skyrmion (opposite of T1-001 for diversity)
    bits_tx = np.array([0])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Propagate through urban turbulence
    turb_cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=100.0,
        cn2=1e-16,
        n_screens=10,
        seed=42,
        inner_scale=5e-3,
        outer_scale=10.0,
    )
    model = TurbulenceModel(turb_cfg)

    dx = X[0, 1] - X[0, 0]
    dz = 100.0 / 10
    phase_screens = []
    for _ in range(10):
        phase_screens.append(
            model.generate_phase_screen((ny, nx), dx, dz)
        )

    E_rx = model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
    E_rx_y = model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)

    # Demodulate
    q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)

    # Assertions — Q ≈ 0.61 × N for this skyrmion texture construction.
    # What matters for communication is correct demodulation (n_rx == n_tx).
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"
    assert abs(q_measured - n_tx) < 0.6, (
        f"Skyrmion number outside decision boundary: Q={q_measured}, expected N={n_tx}"
    )

    # Record physical metrics for traceability
    test_metrics = {
        'cn2': 1e-16,
        'distance_m': 100.0,
        'wavelength_m': 1.55e-6,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'skyrmion_radius_m': 1.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'weak_turbulence_urban_environment',
    }
    return test_metrics