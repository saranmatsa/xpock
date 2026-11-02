#!/usr/bin/env python3
"""
XPOCK — High-Trial BER with Clopper-Pearson Confidence Intervals

Runs 5000 trials at Cn²=1e-13 for SKM vs OAM vs Intensity.
Computes exact 95% CI bounds on BER.
"""

import sys
import os
import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xpock.wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from xpock.wave_info.skyrmion_modem import ModemConfig, SkyrmionModem
from xpock.wave_info.fair_comparison import normalize_power, total_power


def clopper_pearson_upper(n_errors: int, n_trials: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson exact upper bound on binomial proportion."""
    if n_errors == 0:
        return 1.0 - alpha ** (1.0 / n_trials)
    return float(beta.ppf(1.0 - alpha, n_errors + 1, n_trials - n_errors))


def main():
    nx = ny = 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    dx = x[1] - x[0]

    # Gaussian beam
    waist = 1.0
    gaussian = np.exp(-(X**2 + Y**2) / waist**2).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(gaussian)**2) * dx * dx)
    gaussian = gaussian / norm

    # SKM modem
    constellation = [-1, 1]
    n_bits = 1
    cfg = ModemConfig(wavelength=1.55e-6, grid_x=X, grid_y=Y,
                      skyrmion_radius=1.0, constellation=constellation)
    modem = SkyrmionModem(cfg)

    # OAM fields
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    oam_fields = {}
    for l in constellation:
        mf = (np.sqrt(2) * R / waist) ** abs(l) * np.exp(-R**2 / waist**2) * np.exp(1j * l * Phi)
        nrm = np.sqrt(np.sum(np.abs(mf)**2) * dx * dx)
        if nrm > 0:
            mf = mf / nrm
        oam_fields[l] = mf

    intensity_levels = [0.5, 1.5]

    cn2 = 1e-13
    distance = 100.0
    n_trials = 5000

    cfg_t = TurbulenceConfig(wavelength=1.55e-6, distance=distance, cn2=cn2,
                             n_screens=10, seed=42, amplitude_turbulence=True)
    dz = distance / 10

    rng = np.random.default_rng(12345)

    errors = {'skm': 0, 'oam': 0, 'int': 0}

    print(f"Running {n_trials} trials at Cn²={cn2:.0e}...")
    for trial in range(n_trials):
        seed_t = int(rng.integers(0, 2**31))
        model = TurbulenceModel(cfg_t)
        model._rng = np.random.default_rng(seed_t)
        phase_screens = [model.generate_phase_screen((ny, nx), dx, dz) for _ in range(10)]
        amp_screens = [model.generate_amplitude_screen((ny, nx), dx, dz) for _ in range(10)]

        # --- SKM ---
        n_tx = constellation[rng.integers(0, len(constellation))]
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        Ex_t, Ey_t, _ = modem.modulate(bits_tx, X, Y)
        Ex_t = normalize_power(Ex_t, dx, dx, 1.0)
        Ey_t = normalize_power(Ey_t, dx, dx, 1.0)
        E_rx = model.propagate(Ex_t, X, Y, phase_screens=phase_screens,
                               amplitude_screens=amp_screens)
        E_rx_y = model.propagate(Ey_t, X, Y, phase_screens=phase_screens,
                                 amplitude_screens=amp_screens)
        q, n_rx = modem.demodulate(E_rx, E_rx_y)
        bits_rx = modem.skyrmion_to_bits(n_rx)
        errors['skm'] += int(not np.array_equal(bits_tx, bits_rx))

        # --- OAM ---
        l_tx = constellation[rng.integers(0, len(constellation))]
        bit_idx_o = constellation.index(l_tx)
        bits_tx_o = np.array([(bit_idx_o >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        E_tx_o = oam_fields[l_tx].copy()
        E_rx_o = model.propagate(E_tx_o, X, Y, phase_screens=phase_screens,
                                 amplitude_screens=amp_screens)
        ov = {l_rx: abs(np.sum(E_rx_o * np.conj(oam_fields[l_rx])) * dx * dx)
              for l_rx in constellation}
        l_rx_o = max(ov, key=ov.get)
        bits_rx_o = np.array([constellation.index(l_rx_o) >> i & 1 for i in range(n_bits - 1, -1, -1)])
        errors['oam'] += int(not np.array_equal(bits_tx_o, bits_rx_o))

        # --- Intensity ---
        level_idx = rng.integers(0, len(intensity_levels))
        level = intensity_levels[level_idx]
        bits_tx_i = np.array([(level_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        E_tx_i = level * gaussian
        E_rx_i = model.propagate(E_tx_i, X, Y, phase_screens=phase_screens,
                                 amplitude_screens=amp_screens)
        measured_I = float(np.sum(np.abs(E_rx_i) ** 2) * dx * dx)
        nearest = intensity_levels[int(np.argmin([abs(measured_I - l) for l in intensity_levels]))]
        bits_rx_i = np.array([intensity_levels.index(nearest) >> i & 1 for i in range(n_bits - 1, -1, -1)])
        errors['int'] += int(not np.array_equal(bits_tx_i, bits_rx_i))

        if (trial + 1) % 1000 == 0:
            print(f"  {trial+1}/{n_trials} trials...", flush=True)

    print()
    for scheme in ['skm', 'oam', 'int']:
        ber = errors[scheme] / n_trials
        ci_upper = clopper_pearson_upper(errors[scheme], n_trials)
        print(f"{scheme.upper():6s}: BER={ber:.6e} ({errors[scheme]}/{n_trials}), "
              f"95% CI upper={ci_upper:.6e}")

    skm_ber = errors['skm'] / n_trials
    oam_ber = errors['oam'] / n_trials
    if skm_ber > 0:
        print(f"SKM advantage vs OAM: {oam_ber/skm_ber:.1f}x")
    else:
        print("SKM advantage vs OAM: infinite (BER=0)")

    if skm_ber == 0:
        rule_of_three = 3.0 / n_trials
        print(f"Rule of three upper bound (0 errors): {rule_of_three:.6e}")
        print(f"To claim BER ≤ 1e-6 with 95% confidence, need ~3×10⁶ trials with 0 errors")


if __name__ == "__main__":
    main()
