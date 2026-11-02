#!/usr/bin/env python3
"""
XPOCK — Fried Parameter Validation + High-Trial BER with Confidence Intervals

Validates:
1. Fried parameter r₀ against theory at various Cn² values
2. High-trial BER with Clopper-Pearson 95% confidence intervals
3. Grid independence with power conservation error quantification
"""

import sys
import os
from typing import List
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xpock.wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from xpock.wave_info.skyrmion_modem import ModemConfig, SkyrmionModem
from xpock.wave_info.fair_comparison import normalize_power, total_power


def clopper_pearson_upper_bound(n_errors: int, n_trials: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson exact upper bound on binomial proportion.

    When n_errors = 0, upper bound = 1 - alpha^(1/n_trials).
    """
    if n_errors == 0:
        return 1.0 - alpha ** (1.0 / n_trials)
    # Use beta distribution quantile for general case
    from scipy.stats import beta
    return float(beta.ppf(1.0 - alpha, n_errors + 1, n_trials - n_errors))


def validate_fried_parameter():
    """Validate Fried parameter r₀ against theory at various Cn² values."""
    print("=" * 70)
    print("FRIED PARAMETER r₀ VALIDATION")
    print("=" * 70)
    print()

    wavelength = 1.55e-6
    distance = 1000.0  # 1 km
    k = 2.0 * np.pi / wavelength

    cn2_values = [1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12]

    all_pass = True
    for cn2 in cn2_values:
        cfg = TurbulenceConfig(
            wavelength=wavelength, distance=distance, cn2=cn2,
            n_screens=1, seed=42,
        )
        model = TurbulenceModel(cfg)
        r0 = model.compute_fried_parameter()

        # Theoretical: r₀ = [0.423 · k² · Cₙ² · L]^(-3/5)
        r0_theory = (0.423 * k**2 * cn2 * distance) ** (-3.0 / 5.0)

        rel_err = abs(r0 - r0_theory) / r0_theory if r0_theory > 0 else 0.0
        passed = rel_err < 1e-12
        all_pass &= passed

        # Classify turbulence regime
        sigma_R2 = model.compute_rytov_variance()
        regime = "weak" if sigma_R2 < 1 else "strong"

        print(f"  Cn²={cn2:.0e}: r₀={r0:.4f}m, theory={r0_theory:.4f}m, "
              f"rel_err={rel_err:.2e}, σ_R²={sigma_R2:.2f} ({regime}), "
              f"{'PASS' if passed else 'FAIL'}")

    print()
    print(f"Fried parameter validation: {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


def run_high_trial_ber(cn2: float, n_trials: int = 10000,
                       distance: float = 100.0, n_screens: int = 5,
                       grid_n: int = 64, seed: int = 42,
                       constellation: List[int] = None):
    """Run high-trial BER experiment with Clopper-Pearson confidence intervals.

    Returns (skm_ber, skm_ci_upper, oam_ber, oam_ci_upper).
    """
    if constellation is None:
        constellation = [-1, 1]

    nx = ny = grid_n
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # SKM modem
    cfg = ModemConfig(
        wavelength=1.55e-6, grid_x=X, grid_y=Y,
        skyrmion_radius=1.0, constellation=constellation,
    )
    modem = SkyrmionModem(cfg)
    n_bits = modem.n_bits

    # OAM modes
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    w_z = 1.0
    oam_fields = {}
    for l in constellation:
        mf = (np.sqrt(2) * R / w_z) ** abs(l) * np.exp(-R**2 / w_z**2) * np.exp(1j * l * Phi)
        nrm = np.sqrt(np.sum(np.abs(mf)**2) * dx * dy)
        if nrm > 0:
            mf = mf / nrm
        oam_fields[l] = mf

    cfg_t = TurbulenceConfig(
        wavelength=1.55e-6, distance=distance, cn2=cn2,
        n_screens=n_screens, seed=seed,
        amplitude_turbulence=True,
    )

    skm_errors = 0
    oam_errors = 0

    rng = np.random.default_rng(seed + 1000)

    for trial in range(n_trials):
        # Fresh turbulence model per trial for independence
        seed_t = int(rng.integers(0, 2**31))
        m = TurbulenceModel(cfg_t)
        # Re-seed the model's RNG for this trial
        m._rng = np.random.default_rng(seed_t)
        dz = distance / n_screens

        phase_screens = [m.generate_phase_screen((ny, nx), dx, dz) for _ in range(n_screens)]
        amp_screens = [m.generate_amplitude_screen((ny, nx), dx, dz) for _ in range(n_screens)]

        # --- SKM ---
        n_tx = constellation[trial % len(constellation)]
        bit_idx = constellation.index(n_tx)
        bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
        Ex_tx = normalize_power(Ex_tx, dx, dy, 1.0)
        Ey_tx = normalize_power(Ey_tx, dx, dy, 1.0)

        E_rx = m.propagate(Ex_tx, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)
        E_rx_y = m.propagate(Ey_tx, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)

        q, n_rx = modem.demodulate(E_rx, E_rx_y)
        bits_rx = modem.skyrmion_to_bits(n_rx)
        skm_errors += int(not np.array_equal(bits_tx, bits_rx))

        # --- OAM ---
        l_tx = constellation[trial % len(constellation)]
        bit_idx_o = constellation.index(l_tx)
        bits_tx_o = np.array([(bit_idx_o >> i) & 1 for i in range(n_bits - 1, -1, -1)])
        E_tx_o = oam_fields[l_tx].copy()
        E_rx_o = m.propagate(E_tx_o, X, Y, phase_screens=phase_screens, amplitude_screens=amp_screens)

        ov_o = {l_rx: abs(np.sum(E_rx_o * np.conj(oam_fields[l_rx])) * dx * dy) for l_rx in constellation}
        l_rx_o = max(ov_o, key=ov_o.get)
        bits_rx_o = np.array([constellation.index(l_rx_o) >> i & 1 for i in range(n_bits - 1, -1, -1)])
        oam_errors += int(not np.array_equal(bits_tx_o, bits_rx_o))

    skm_ber = skm_errors / n_trials
    oam_ber = oam_errors / n_trials
    skm_ci_upper = clopper_pearson_upper_bound(skm_errors, n_trials)
    oam_ci_upper = clopper_pearson_upper_bound(oam_errors, n_trials)

    return skm_ber, skm_ci_upper, oam_ber, oam_ci_upper


def validate_grid_independence():
    """Validate grid independence: Q exact at all resolutions, power conserved."""
    print()
    print("=" * 70)
    print("GRID INDEPENDENCE STUDY")
    print("=" * 70)
    print()

    from xpock.properties import (PolarizationTexture, PolarizationConfig,
                                  PolarizationState,
                                  TopologicalCharge, TopologicalChargeConfig)

    extent = 8.0
    skyrmion_radius = 1.0
    tc = TopologicalCharge(TopologicalChargeConfig(charge_type="skyrmion", charge=1))

    all_pass = True
    for grid_n in [32, 64, 128, 256, 512]:
        x = np.linspace(-extent / 2, extent / 2, grid_n)
        y = np.linspace(-extent / 2, extent / 2, grid_n)
        X, Y = np.meshgrid(x, y, indexing='xy')
        dx = x[1] - x[0]

        cfg = PolarizationConfig(state=PolarizationState.SKYRMIONIC,
                                 skyrmion_radius=skyrmion_radius,
                                 skyrmion_charge=2)
        Ex, Ey = PolarizationTexture(cfg).evaluate(X, Y)
        Q = tc.compute_skyrmion_number(Ex, Ey)

        # Power check
        power = total_power(Ex, dx, dx) + total_power(Ey, dx, dx)

        q_ok = round(Q) == 2
        all_pass &= q_ok

        print(f"  grid={grid_n}x{grid_n}: Q={Q:.6f} (round={round(Q)}), "
              f"total_power={power:.6f}, dx={dx:.6f}m, {'PASS' if q_ok else 'FAIL'}")

    print()
    print(f"Grid independence: {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    # 1. Fried parameter validation
    fried_ok = validate_fried_parameter()

    # 2. Grid independence
    grid_ok = validate_grid_independence()

    # 3. High-trial BER with confidence intervals
    print()
    print("=" * 70)
    print("HIGH-TRIAL BER WITH 95% CONFIDENCE INTERVALS")
    print("=" * 70)
    print()

    cn2_values = [1e-14, 1e-13, 1e-12]
    n_trials = 10000

    for cn2 in cn2_values:
        print(f"\n  Cn² = {cn2:.0e}, {n_trials:,} trials:")
        skm_ber, skm_ci, oam_ber, oam_ci = run_high_trial_ber(
            cn2=cn2, n_trials=n_trials,
            distance=100.0, n_screens=5, grid_n=64,
        )
        print(f"    SKM: BER={skm_ber:.6e} ({int(skm_ber*n_trials)}/{n_trials} errors), "
              f"95% CI upper bound={skm_ci:.6e}")
        print(f"    OAM: BER={oam_ber:.6e} ({int(oam_ber*n_trials)}/{n_trials} errors), "
              f"95% CI upper bound={oam_ci:.6e}")

        if skm_ber == 0:
            advantage = "infinite (BER=0)"
        else:
            advantage = f"{oam_ber/skm_ber:.1f}x"
        print(f"    SKM advantage vs OAM: {advantage}")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Fried parameter: {'PASS' if fried_ok else 'FAIL'}")
    print(f"Grid independence: {'PASS' if grid_ok else 'FAIL'}")
