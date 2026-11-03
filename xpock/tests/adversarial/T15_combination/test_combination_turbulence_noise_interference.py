"""
XPOCK — Adversarial Test T15-001: Combination Attack — Turbulence + Noise + Interference

Physical scenario: Simultaneous multiple stressors — moderate turbulence
(Cn²=1e-14), additive Gaussian noise (level=0.005), and overlapping field
interference (ratio=0.05). This tests whether skyrmion number modulation
can survive combined real-world impairments.

Expected result: Q preserved within decision boundary under combined stress, BER = 0.
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
    'xpock.wave_info.turbulence_t15_001',
    _src_path("xpock", "wave_info", "turbulence.py"),
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_t15_001',
    _src_path("xpock", "wave_info", "skyrmion_modem.py"),
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem


def test_combination_turbulence_noise_interference():
    """Skyrmion number Q preserved under combined turbulence + noise + interference.

    Physical scenario: Simultaneous multiple stressors mimic real-world
    conditions:
    - Moderate turbulence (Cn²=1e-14, 1000 m)
    - Additive Gaussian noise (level=0.005)
    - Overlapping field interference (ratio=0.05, -13 dB)

    Tests whether skyrmion number modulation can survive combined
    real-world impairments that would defeat conventional modulation.

    Parameters:
        Cn²: 1e-14 m^(-2/3) (moderate turbulence)
        Distance: 1000 m
        Wavelength: 1550 nm
        Noise level: 0.005 (additive Gaussian)
        Interference ratio: 0.05 (-13 dB)
        Grid: 128×128, extent 8 m
        Skyrmion radius: 2.0 m
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

    # Transmit N=-1 skyrmion
    bits_tx = np.array([0])
    Ex_tx, Ey_tx, n_tx = modem.modulate(bits_tx, X, Y)

    # Propagate through moderate turbulence
    turb_cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=1000.0,
        cn2=1e-14,
        n_screens=20,
        seed=42,
        inner_scale=5e-3,
        outer_scale=10.0,
    )
    model = TurbulenceModel(turb_cfg)

    dx = X[0, 1] - X[0, 0]
    dz = 1000.0 / 20
    phase_screens = []
    for _ in range(20):
        phase_screens.append(
            model.generate_phase_screen((ny, nx), dx, dz)
        )

    E_rx = model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
    E_rx_y = model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)

    # Apply additive Gaussian noise
    noise_level = 0.005
    rng = np.random.default_rng(42)
    noise_ex = rng.normal(0, noise_level, X.shape) + 1j * rng.normal(
        0, noise_level, X.shape
    )
    noise_ey = rng.normal(0, noise_level, X.shape) + 1j * rng.normal(
        0, noise_level, X.shape
    )
    E_rx = E_rx + noise_ex
    E_rx_y = E_rx_y + noise_ey

    # Apply overlapping field interference
    interference_ratio = 0.05
    phase = rng.uniform(0, 2 * np.pi, X.shape)
    amp = np.sqrt(interference_ratio) * np.exp(-(X**2 + Y**2) / 8.0)
    E_rx = E_rx + amp * np.exp(1j * phase)
    E_rx_y = E_rx_y + amp * np.exp(1j * phase)

    # Demodulate
    q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)

    # Assertions — What matters for communication is correct demodulation (n_rx == n_tx).
    assert np.isfinite(q_measured), f"Q not finite: {q_measured}"
    assert n_rx == n_tx, f"N={n_tx} demodulated to {n_rx}, Q={q_measured}"

    # Record physical metrics for traceability
    test_metrics = {
        'cn2': 1e-14,
        'distance_m': 1000.0,
        'wavelength_m': 1.55e-6,
        'noise_level': float(noise_level),
        'interference_ratio': float(interference_ratio),
        'grid': f'{nx}x{ny}',
        'extent_m': extent,
        'skyrmion_radius_m': 2.0,
        'q_measured': float(q_measured),
        'n_tx': int(n_tx),
        'n_rx': int(n_rx),
        'q_error': float(abs(q_measured - n_tx)),
        'ber': 0.0,
        'scenario': 'combination_turbulence_noise_interference',
    }
    return test_metrics