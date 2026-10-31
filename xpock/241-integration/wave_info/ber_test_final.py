#!/usr/bin/env python3
"""
XPOCK — Skyrmion BER Test: 2-state constellation {-1, 1} across turbulence levels.
Tests whether skyrmion number modulation provides reliable information transfer
through turbulent atmospheric channels.
"""

import importlib.util, sys
import numpy as np

spec = importlib.util.spec_from_file_location(
    'skyrmion_modem', '241-integration/wave_info/skyrmion_modem.py'
)
modem_mod = importlib.util.module_from_spec(spec)
sys.modules['skyrmion_modem'] = modem_mod
spec.loader.exec_module(modem_mod)
SkyrmionModem = modem_mod.SkyrmionModem
ModemConfig = modem_mod.ModemConfig

spec2 = importlib.util.spec_from_file_location(
    'turbulence', '241-integration/wave_info/turbulence.py'
)
turbulence_mod = importlib.util.module_from_spec(spec2)
sys.modules['turbulence'] = turbulence_mod
spec2.loader.exec_module(turbulence_mod)
TurbulenceModel = turbulence_mod.TurbulenceModel
TurbulenceConfig = turbulence_mod.TurbulenceConfig

nx, ny = 64, 64
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y, indexing='xy')

cfg = ModemConfig(
    wavelength=1.55e-6,
    grid_x=X,
    grid_y=Y,
    skyrmion_radius=1.0,
    constellation=[-1, 1],
)
modem = SkyrmionModem(cfg)

print(f'Constellation: {modem.constellation}, bits/symbol: {modem.n_bits}')
print(f'{"Cn²":>12} {"N_tx":>6} {"Q_meas":>10} {"Q_std":>10} {"BER":>8} ({r"n_err":>3}/{"n_trials":>5})')
print('-' * 70)

for cn2 in [1e-15, 1e-14, 1e-13, 1e-12]:
    turb_cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=100.0,
        cn2=cn2,
        n_screens=5,
        seed=42,
    )
    turb_model = TurbulenceModel(turb_cfg)

    n_errors_total = 0
    n_trials_total = 0
    q_values = []

    for n_tx in constellation:
        bits_tx = np.array([constellation.index(n_tx)])
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)

        n_errors = 0
        n_trials = 30

        for _ in range(n_trials):
            E_rx = turb_model.propagate(Ex_tx, X, Y)
            E_rx_y = turb_model.propagate(Ey_tx, X, Y)
            q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)
            q_values.append(q_measured)

            bits_rx = modem.skyrmion_to_bits(n_rx)
            bit_error = not np.array_equal(bits_tx, bits_rx)
            n_errors += int(bit_error)

        ber = n_errors / n_trials
        q_mean = np.mean(q_values)
        q_std = np.std(q_values)
        print(f"{cn2:.0e}  {n_tx:>6}  {q_meas:.4f}   {q_std:.4f}   {ber:.4f} ({n_errors}/{n_trials})")
        n_errors_total += n_errors
        n_trials_total += n_trials

print()
print(f"Overall BER: {n_errors_total}/{n_trials_total} = {n_errors_total/n_trials_total:.6f}")