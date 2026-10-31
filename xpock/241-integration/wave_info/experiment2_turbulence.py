#!/usr/bin/env python3
"""
XPOCK — Experiment 2: Skyrmion Number Under High Turbulence

Compares SKM vs OAM vs Intensity modulation under increasing turbulence.
Uses proper multi-screen propagation (different phase screen per step).

Hypothesis: SKM maintains low BER at higher Cn² than OAM and Intensity
modulation due to topological protection of skyrmion number.
"""

import importlib.util
import sys
import numpy as np

spec = importlib.util.spec_from_file_location(
    'skyrmion_modem', 'src/xpock/wave_info/skyrmion_modem.py'
)
modem_mod = importlib.util.module_from_spec(spec)
sys.modules['skyrmion_modem'] = modem_mod
spec.loader.exec_module(modem_mod)
SkyrmionModem = modem_mod.SkyrmionModem
ModemConfig = modem_mod.ModemConfig

spec2 = importlib.util.spec_from_file_location(
    'turbulence', 'src/xpock/wave_info/turbulence.py'
)
turbulence_mod = importlib.util.module_from_spec(spec2)
sys.modules['turbulence'] = turbulence_mod
spec2.loader.exec_module(turbulence_mod)
TurbulenceModel = turbulence_mod.TurbulenceModel
TurbulenceConfig = turbulence_mod.TurbulenceConfig

from scipy.fft import fft2, ifft2, fftfreq

nx, ny = 64, 64
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y, indexing='xy')

constellation = [-1, 1]
oam_modes = [-1, 1]
intensity_levels = [0.5, 1.5]

cfg = ModemConfig(
    wavelength=1.55e-6,
    grid_x=X,
    grid_y=Y,
    skyrmion_radius=1.0,
    constellation=constellation,
)
modem = SkyrmionModem(cfg)

print("=" * 70)
print("EXPERIMENT 2: High-Turbulence Comparison (SKM vs OAM vs Intensity)")
print("=" * 70)
print(f"Grid: {nx}x{ny}, Wavelength: 1.55 um, Distance: 1000 m")
print(f"Constellation: {constellation} (1 bit/symbol)")
print()

cn2_values = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]

print(f"{'Cn²':>12} {'SKM BER':>10} {'OAM BER':>10} {'Int BER':>10} {'SKM Adv (dB)':>14}")
print("-" * 60)

k = 2.0 * np.pi / 1.55e-6
waist = 1.0
dx = X[0, 1] - X[0, 0]

# Pre-compute OAM modes
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)
mode_fields = {}
for l in oam_modes:
    radial = (np.sqrt(2.0) * R / waist) ** abs(l) * np.exp(-R**2 / waist**2)
    azimuthal = np.exp(1j * l * Phi)
    mf = radial * azimuthal
    norm = np.sqrt(np.sum(np.abs(mf)**2) * dx * dx)
    if norm > 0:
        mf = mf / norm
    mode_fields[l] = mf

# Pre-compute Gaussian beam
gaussian = np.exp(-(X**2 + Y**2) / waist**2).astype(np.complex128)
norm = np.sqrt(np.sum(np.abs(gaussian)**2) * dx * dx)
if norm > 0:
    gaussian = gaussian / norm

for cn2 in cn2_values:
    turb_cfg = TurbulenceConfig(
        wavelength=1.55e-6,
        distance=1000.0,
        cn2=cn2,
        n_screens=20,
        seed=42,
    )
    turb_model = TurbulenceModel(turb_cfg)

    nx_dim = X.shape[0]
    ny_dim = X.shape[1]
    dz = 1000.0 / 20
    # Generate different phase screens for each step
    phase_screens = []
    for _ in range(20):
        phase_screens.append(
            turb_model.generate_phase_screen((ny_dim, nx_dim), dx, dz)
        )

    # --- SKM test ---
    skm_errors = 0
    skm_total = 0
    for n_tx in constellation:
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([bit_idx])
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
        for _ in range(30):
            E_rx = turb_model.propagate(Ex_tx, X, Y, phase_screens=phase_screens)
            E_rx_y = turb_model.propagate(Ey_tx, X, Y, phase_screens=phase_screens)
            q_meas, n_rx = modem.demodulate(E_rx, E_rx_y)
            bits_rx = modem.skyrmion_to_bits(n_rx)
            if not np.array_equal(bits_tx, bits_rx):
                skm_errors += 1
            skm_total += 1
    skm_ber = skm_errors / skm_total

    # --- OAM test ---
    oam_errors = 0
    oam_total = 0
    for l_tx in oam_modes:
        bit_idx = oam_modes.index(l_tx)
        bits_tx = np.array([bit_idx])
        E_tx = mode_fields[l_tx].copy()
        for _ in range(30):
            E_rx = turb_model.propagate(E_tx, X, Y, phase_screens=phase_screens)
            overlaps = {}
            for l_rx in oam_modes:
                mf = mode_fields[l_rx]
                ov = np.abs(np.sum(E_rx * np.conj(mf)) * dx * dx)
                overlaps[l_rx] = ov
            l_rx = max(overlaps, key=overlaps.get)
            bits_rx = np.array([oam_modes.index(l_rx)])
            if not np.array_equal(bits_tx, bits_rx):
                oam_errors += 1
            oam_total += 1
    oam_ber = oam_errors / oam_total

    # --- Intensity test ---
    int_errors = 0
    int_total = 0
    for level_idx, level in enumerate(intensity_levels):
        bit_idx = level_idx
        bits_tx = np.array([bit_idx])
        E_tx = level * gaussian
        for _ in range(30):
            E_rx = turb_model.propagate(E_tx, X, Y, phase_screens=phase_screens)
            measured_I = float(np.sum(np.abs(E_rx)**2) * dx * dx)
            nearest_idx = int(np.argmin([abs(measured_I - l) for l in intensity_levels]))
            bits_rx = np.array([nearest_idx])
            if not np.array_equal(bits_tx, bits_rx):
                int_errors += 1
            int_total += 1
    int_ber = int_errors / int_total

    skm_adv = float('inf') if skm_ber == 0 and oam_ber > 0 else \
              (10.0 * np.log10(oam_ber / skm_ber) if skm_ber > 0 and oam_ber > 0 else 0.0)
    adv_str = f"{skm_adv:.2f}" if skm_adv != float('inf') else "inf"

    print(f"{cn2:.0e}  {skm_ber:.6f}  {oam_ber:.6f}  {int_ber:.6f}  {adv_str:>14}")

print()
print("Rytov variance at Cn²=1e-10: σ_R² = 1.23 × Cn² × k^(7/6) × L^(11/6)")
L = 1000.0
sigma_R2 = 1.23 * 1e-10 * k ** (7.0 / 6.0) * L ** (11.0 / 6.0)
print(f"  σ_R² = {sigma_R2:.2f} (strong turbulence regime)")
print("=" * 70)