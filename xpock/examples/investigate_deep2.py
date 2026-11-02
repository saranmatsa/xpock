#!/usr/bin/env python3
"""
XPOCK — Deep Investigation: Fresh phase screens per trial

1. INTENSITY ROBUSTNESS: Why is intensity BER=0 at Cn²=1e-11?
   Test with FRESH phase screens per trial

2. HIGH Q_STD AT WEAK TURBULENCE: Q_std=0.98 at Cn²=1e-13 but BER=0
   Test with FRESH phase screens, many trials, examine Q distribution
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xpock.wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from xpock.wave_info.skyrmion_modem import ModemConfig, SkyrmionModem


def total_power(field, dx, dy):
    return float(np.sum(np.abs(field) ** 2) * dx * dy)


def add_aperture(field, X, Y, radius):
    """Clip field outside circular aperture."""
    mask = np.sqrt(X**2 + Y**2) > radius
    field_ap = field.copy()
    field_ap[mask] = 0.0
    return field_ap


print("=" * 70)
print("INVESTIGATION 1: Intensity robustness (FRESH screens per trial)")
print("=" * 70)

nx = ny = 64
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y, indexing='xy')
dx = x[1] - x[0]
dy = y[1] - y[0]

# Gaussian beam normalized to unit power
R2 = X**2 + Y**2
waist = 1.0
gaussian = np.exp(-R2 / waist**2).astype(np.complex128)
norm = np.sqrt(np.sum(np.abs(gaussian) ** 2) * dx * dy)
gaussian = gaussian / norm

print(f"Grid: {nx}x{ny}, extent=6m, dx={dx:.4f}m, waist={waist}m")
print(f"Input power: {total_power(gaussian, dx, dy):.6f}")
print()

cn2 = 1e-11
distance = 100.0
n_screens = 20

for aperture_radius in [None, 1.5]:
    cfg = TurbulenceConfig(
        wavelength=1.55e-6, distance=distance, cn2=cn2,
        n_screens=n_screens, seed=42,
    )
    model = TurbulenceModel(cfg)
    dz = distance / n_screens

    errors = 0
    total = 40
    powers_05 = []
    powers_15 = []

    for trial in range(20):
        # FRESH screens for each trial
        phase_screens = []
        for _ in range(n_screens):
            phase_screens.append(
                model.generate_phase_screen((ny, nx), dx, dz)
            )

        for level in [0.5, 1.5]:
            E_tx = level * gaussian
            E_rx = model.propagate(
                E_tx, X, Y, phase_screens=phase_screens,
            )

            if aperture_radius is not None:
                E_rx = add_aperture(E_rx, X, Y, aperture_radius)

            measured_I = float(np.sum(np.abs(E_rx) ** 2) * dx * dy)

            if level == 0.5:
                powers_05.append(measured_I)
                nearest = 0.5 if abs(measured_I - 0.5) < abs(measured_I - 1.5) else 1.5
                if nearest != 0.5:
                    errors += 1
            else:
                powers_15.append(measured_I)
                nearest = 0.5 if abs(measured_I - 0.5) < abs(measured_I - 1.5) else 1.5
                if nearest != 1.5:
                    errors += 1

    ber = errors / total
    ap_str = f"aperture={aperture_radius}m" if aperture_radius else "no aperture (full grid)"
    print(f"  Cn²={cn2:.0e}, {ap_str}: BER={ber:.6f} ({errors}/{total})")
    print(f"    Ī(0.5)={np.mean(powers_05):.4f}±{np.std(powers_05):.4f}, "
          f"Ī(1.5)={np.mean(powers_15):.4f}±{np.std(powers_15):.4f}")
    print(f"    Separation: {np.mean(powers_15) - np.mean(powers_05):.4f}, "
          f"combined std: {np.sqrt(np.std(powers_05)**2 + np.std(powers_15)**2):.4f}")

print()
print("If intensity BER stays 0 with fresh screens, power conservation makes")
print("total-intensity detection robust even under extreme turbulence.")

print()
print("=" * 70)
print("INVESTIGATION 2: Q distribution at weak turbulence (FRESH screens)")
print("=" * 70)

constellation = [-1, 1]
cfg = ModemConfig(
    wavelength=1.55e-6, grid_x=X, grid_y=Y,
    skyrmion_radius=1.0, constellation=constellation,
)
modem = SkyrmionModem(cfg)

cn2 = 1e-13

print(f"Cn²={cn2}, 500 trials each for N=-1 and N=+1")
print()

for n_tx in [-1, 1]:
    cfg_t = TurbulenceConfig(
        wavelength=1.55e-6, distance=100.0, cn2=cn2,
        n_screens=5, seed=42,
    )
    model = TurbulenceModel(cfg_t)
    dz = 100.0 / 5

    qs = []
    for trial in range(500):
        # FRESH screens for each trial
        phase_screens = []
        for _ in range(5):
            phase_screens.append(
                model.generate_phase_screen((ny, nx), dx, dz)
            )

        bits_tx = np.array([0 if n_tx == -1 else 1])
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
        E_rx = model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
        E_rx_y = model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)
        q, n_rx = modem.demodulate(E_rx, E_rx_y)
        qs.append(q)

    qs = np.array(qs)
    n_crossings = int(np.sum(np.abs(qs) < 0.5))
    ber = n_crossings / len(qs)
    print(f"  N={n_tx:+d}: Q mean={np.mean(qs):+.4f}, std={np.std(qs):.4f}")
    print(f"    Q range: [{qs.min():+.4f}, {qs.max():+.4f}]")
    print(f"    |Q|<0.5 count: {n_crossings} / 500 (BER={ber:.6f})")

print()
print("Conclusion: If |Q|<0.5 never occurs, BER=0 is correct even though")
print("Q_std may be high. Q fluctuates around ±1 but rarely crosses 0.")
