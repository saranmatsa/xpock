"""
XPOCK — Adversarial Test T1-003: Moderate Turbulence — Coastal Environment

Physical scenario: Free-space optical link over coastal water.
Refractive index structure constant Cn² = 1e-15 m^(-2/3) (moderate turbulence,
typical of coastal environments with sea-air temperature differences).
Propagation distance 500 m at 1550 nm wavelength.

Coastal turbulence is stronger than urban due to water-land temperature
gradients and humidity variations. Skyrmion number should still survive
as a topological invariant, but Q measurement may show more variance.

Expected result: Q preserved within half the constellation spacing, BER = 0.
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


turbulence_mod = _load_module(
    'xpock.wave_info.turbulence_t1_003',
    _src_path("xpock", "wave_info", "turbulence.py"),
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t1_003',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_moderate_turbulence_coastal_environment():
    """Skyrmion number Q preserved through coastal turbulence (Cn²=1e-15, 500 m).

    Physical scenario: Shore-to-ship free-space optical link over coastal water.
    Temperature and humidity gradients between sea and air create moderate
    turbulence. Wavefront distortion is significant but skyrmion number,
    as a topological invariant, should remain intact.

    Parameters:
        Cn²: 1e-15 m^(-2/3) (moderate coastal turbulence)
        Distance: 500 m
        Wavelength: 1550 nm
        Grid: 128×128, extent 8 m (larger aperture for longer link)
        Skyrmion radius: 2.0 m (larger texture for longer propagation)
        Constellation: [-1, 1] (1 bit/symbol)
    """
    nx, ny = 128, 128
    extent = 8.0
    x = np.linspace(-extent / 2, extent / 2, nx)
    y = np.linspace(-extent / 2, extent / 2, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=2.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)

    # Transmit N=+1 skyrmion
    bits_tx = np.array([1])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Propagate through coastal turbulence
    turb_cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=500.0,
        cn2=1e-15,
        n_screens=20,
        seed=42,
        inner_scale=5e-3,
        outer_scale=10.0,
    )
    model = TurbulenceModel(turb_cfg)

    dx = X[0, 1] - X[0, 0]
    dz = 500.0 / 20
    phase_screens = []
    for _ in range(20):
        phase_screens.append(
            model.generate_phase_screen((ny, nx), dx, dz)
        )

    E_rx = model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
    E_rx_y = model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)

    # Demodulate
    q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)

    # Assertions — What matters for communication is correct demodulation (n_rx == n_tx).
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'cn2': 1e-15,
        'distance_m': 500.0,
        'wavelength_m': 1.55e-6,
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'skyrmion_radius_m': 2.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'moderate_turbulence_coastal_environment',
    }
    return test_metrics