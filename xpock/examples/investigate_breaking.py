#!/usr/bin/env python3
"""
XPOCK — Intensity Breaking Point Test

Tests intensity detection with closer levels to find where turbulence
causes errors. Also tests 4-state SKM vs 4-mode OAM vs 4-level intensity.
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
    mask = np.sqrt(X**2 + Y**2) > radius
    field_ap = field.copy()
    field_ap[mask] = 0.0
    return field_ap


print("=" * 70)
print("INTENSITY BREAKING POINT: Closer levels, stronger turbulence")
print("=" * 70)

nx = ny = 64
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y, indexing='xy')
dx = x[1] - x[0]
dy = y[1] - y[0]

waist = 1.0
gaussian = np.exp(-(X**2 + Y**2) / waist**2).astype(np.complex128)
norm = np.sqrt(np.sum(np.abs(gaussian) ** 2) * dx * dy)
gaussian = gaussian / norm

# Test 1: Closer intensity levels at strong turbulence
print("\n--- Test 1: Intensity with closer levels ---")
level_pairs = [(0.9, 1.1), (0.8, 1.2), (0.7, 1.3), (0.6, 1.4), (0.5, 1.5)]

for lo, hi in level_pairs:
    cn2 = 1e-12
    n_trials = 100
    n_screens = 10
    dz = 100.0 / n_screens

    errors = 0
    total = n_trials * 2
    powers_lo = []
    powers_hi = []

    for trial in range(n_trials):
        seed = np.random.randint(0, 2**31)
        cfg = TurbulenceConfig(
            wavelength=1.55e-6, distance=100.0, cn2=cn2,
            n_screens=n_screens, seed=int(seed),
            amplitude_turbulence=True,
        )
        model = TurbulenceModel(cfg)

        phase_screens = [model.generate_phase_screen((ny, nx), dx, dz) for _ in range(n_screens)]
        amp_screens = [model.generate_amplitude_screen((ny, nx), dx, dz) for _ in range(n_screens)]

        for level in [lo, hi]:
            E_tx = level * gaussian
            E_rx = model.propagate(
                E_tx, X, Y,
                phase_screens=phase_screens,
                amplitude_screens=amp_screens,
            )
            measured_I = float(np.sum(np.abs(E_rx) ** 2) * dx * dy)

            if level == lo:
                powers_lo.append(measured_I)
                nearest = lo if abs(measured_I - lo) < abs(measured_I - hi) else hi
                if nearest != lo:
                    errors += 1
            else:
                powers_hi.append(measured_I)
                nearest = lo if abs(measured_I - lo) < abs(measured_I - hi) else hi
                if nearest != hi:
                    errors += 1

    ber = errors / total
    sep = np.mean(powers_hi) - np.mean(powers_lo)
    comb_std = np.sqrt(np.std(powers_lo)**2 + np.std(powers_hi)**2)
    print(f"  Levels {lo:.1f}/{hi:.1f}: BER={ber:.4f} ({errors}/{total}), "
          f"sep={sep:.3f}, comb_std={comb_std:.4f}, sep/std={sep/max(comb_std,1e-12):.1f}")

print()
print("If BER=0 even for 0.9/1.1 levels, intensity is extremely robust.")
print("If BER rises, we've found intensity's breaking point.")

# Test 2: 4-state SKM vs 4-mode OAM vs 4-level intensity
print()
print("=" * 70)
print("Test 2: 4-state comparison at Cn²=1e-13")
print("=" * 70)

constellation = [-2, -1, 1, 2]
oam_modes = [-2, -1, 1, 2]
intensity_levels = [0.5, 1.0, 1.5, 2.0]

cfg = ModemConfig(
    wavelength=1.55e-6, grid_x=X, grid_y=Y,
    skyrmion_radius=1.0, constellation=constellation,
)
modem = SkyrmionModem(cfg)
n_bits = modem.n_bits

cn2 = 1e-13
n_trials = 100
distance = 100.0
n_screens = 5

cfg_t = TurbulenceConfig(
    wavelength=1.55e-6, distance=distance, cn2=cn2,
    n_screens=n_screens, seed=42,
    amplitude_turbulence=True,
)
model = TurbulenceModel(cfg_t)
dz = distance / n_screens

# SKM
skm_errors = 0
skm_total = 0
qs = []

for trial in range(n_trials):
    n_tx = constellation[np.random.randint(0, len(constellation))]
    bit_idx = constellation.index(n_tx)
    bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
    Ex_tx_norm = Ex_tx * np.sqrt(1.0 / total_power(Ex_tx, dx, dy))
    Ey_tx_norm = Ey_tx * np.sqrt(1.0 / total_power(Ey_tx, dx, dy))

    seed = np.random.randint(0, 2**31)
    m = TurbulenceModel(cfg_t)
    phase_screens = [m.generate_phase_screen((ny, nx), dx, dz) for _ in range(n_screens)]
    amp_screens = [m.generate_amplitude_screen((ny, nx), dx, dz) for _ in range(n_screens)]

    E_rx = m.propagate(Ex_tx_norm, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)
    E_rx_y = m.propagate(Ey_tx_norm, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)

    q, n_rx = modem.demodulate(E_rx, E_rx_y)
    qs.append(q)
    bits_rx = modem.skyrmion_to_bits(n_rx)
    skm_errors += int(not np.array_equal(bits_tx, bits_rx))
    skm_total += 1

skm_ber = skm_errors / skm_total
print(f"  SKM (4-state): BER={skm_ber:.4f} ({skm_errors}/{skm_total}), "
      f"Q mean={np.mean(qs):+.3f}, Q std={np.std(qs):.3f}")

# OAM (4 modes)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)
w_z = waist
oam_fields = {}
for l in oam_modes:
    mf = (np.sqrt(2) * R / w_z) ** abs(l) * np.exp(-R**2 / w_z**2) * np.exp(1j * l * Phi)
    nrm = np.sqrt(np.sum(np.abs(mf)**2) * dx * dy)
    if nrm > 0:
        mf = mf / nrm
    oam_fields[l] = mf

oam_errors = 0
oam_total = 0
overlaps = []

for trial in range(n_trials):
    l_tx = oam_modes[np.random.randint(0, len(oam_modes))]
    bit_idx = oam_modes.index(l_tx)
    bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
    E_tx = oam_fields[l_tx].copy()

    seed = np.random.randint(0, 2**31)
    m = TurbulenceModel(cfg_t)
    phase_screens = [m.generate_phase_screen((ny, nx), dx, dz) for _ in range(n_screens)]
    amp_screens = [m.generate_amplitude_screen((ny, nx), dx, dz) for _ in range(n_screens)]

    E_rx = m.propagate(E_tx, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)

    ov = {}
    for l_rx in oam_modes:
        ov[l_rx] = abs(np.sum(E_rx * np.conj(oam_fields[l_rx])) * dx * dy)
    l_rx = max(ov, key=ov.get)
    overlaps.append(ov[l_tx])

    bits_rx = np.array([(oam_modes.index(l_rx) >> i) & 1 for i in range(n_bits - 1, -1, -1)])
    oam_errors += int(not np.array_equal(bits_tx, bits_rx))
    oam_total += 1

oam_ber = oam_errors / oam_total
print(f"  OAM (4-mode):   BER={oam_ber:.4f} ({oam_errors}/{oam_total}), "
      f"overlap={np.mean(overlaps):.3f}")

# Intensity (4 levels)
int_errors = 0
int_total = 0
ints = []

for trial in range(n_trials):
    level_idx = np.random.randint(0, len(intensity_levels))
    level = intensity_levels[level_idx]
    bits_tx = np.array([(level_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
    E_tx = level * gaussian

    seed = np.random.randint(0, 2**31)
    m = TurbulenceModel(cfg_t)
    phase_screens = [m.generate_phase_screen((ny, nx), dx, dz) for _ in range(n_screens)]
    amp_screens = [m.generate_amplitude_screen((ny, nx), dx, dz) for _ in range(n_screens)]

    E_rx = m.propagate(E_tx, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)
    measured_I = float(np.sum(np.abs(E_rx) ** 2) * dx * dy)
    ints.append(measured_I)

    nearest_idx = int(np.argmin([abs(measured_I - l) for l in intensity_levels]))
    bits_rx = np.array([(nearest_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
    int_errors += int(not np.array_equal(bits_tx, bits_rx))
    int_total += 1

int_ber = int_errors / int_total
print(f"  Intensity (4-level): BER={int_ber:.4f} ({int_errors}/{int_total}), "
      f"mean I={np.mean(ints):.3f}")

print()
print("4-state summary: SKM vs OAM vs Intensity")
print(f"  SKM:  {skm_ber:.4f}  OAM:  {oam_ber:.4f}  Int:  {int_ber:.4f}")
