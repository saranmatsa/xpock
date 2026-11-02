#!/usr/bin/env python3
"""
XPOCK — End-to-End Validation Runner

Runs all validation experiments:
1. Turbulence model validation against von Karman spectrum theory
2. Skyrmion number quantization (Q = q exactness, grid convergence)
3. Fair SKM vs OAM vs Intensity comparison under identical constraints
4. Receiver noise sensitivity
5. Grid convergence study

Results are logged to EXPERIMENT_LOG.md and printed to stdout.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xpock.wave_info.fair_comparison import (
    validate_turbulence_model,
    validate_skyrmion_quantization,
    FairComparison,
    ValidationResult,
)

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'EXPERIMENT_LOG.md')


def log_result(category: str, results: list, detail: str = ""):
    """Append validation results to the experiment log."""
    with open(LOG_PATH, 'a') as f:
        f.write(f"\n## {category} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if detail:
            f.write(f"\n{detail}\n\n")
        f.write("| Check | Passed | Measured | Expected | Rel Error | Notes |\n")
        f.write("|-------|--------|----------|----------|-----------|-------|\n")
        for r in results:
            f.write(f"| {r.name} | {'✓' if r.passed else '✗'} | "
                    f"{r.measured:.6g} | {r.expected:.6g} | {r.rel_error:.6g} | {r.detail} |\n")
        n_passed = sum(1 for r in results if r.passed)
        f.write(f"\n**Summary: {n_passed}/{len(results)} checks passed**\n")


def log_comparison(results: list, detail: str = ""):
    """Append fair comparison results to the experiment log."""
    with open(LOG_PATH, 'a') as f:
        f.write(f"\n## Fair Comparison — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if detail:
            f.write(f"\n{detail}\n\n")
        f.write("| Scheme | Cn² | Trials | Errors | BER | SER | Extra |\n")
        f.write("|--------|-----|--------|--------|-----|-----|-------|\n")
        for r in results:
            extra = ""
            if r.q_mean is not None:
                extra += f"Q̄={r.q_mean:+.4f},σ_Q={r.q_std:.4f}"
            if r.overlap_mean is not None:
                extra += f"overlap={r.overlap_mean:.4f}"
            if r.intensity_mean is not None:
                extra += f"Î={r.intensity_mean:.4f}"
            f.write(f"| {r.scheme} | {r.cn2:.0e} | {r.n_trials} | {r.n_errors} | "
                    f"{r.ber:.6f} | {r.ser:.6f} | {extra} |\n")


def run_all():
    print("=" * 70)
    print("XPOCK — END-TO-END SCIENTIFIC VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_passed = True

    # ---------------------------------------------------------------
    # Experiment V1/V2: Turbulence model + skyrmion quantization
    # ---------------------------------------------------------------
    print("--- V1/V2: Physics Validation ---")
    print()

    print("[V2a] Turbulence model vs von Karman spectrum theory")
    turb_results = validate_turbulence_model(
        wavelength=1.55e-6, distance=1000.0, cn2=1e-13,
        inner_scale=5e-3, outer_scale=10.0,
        n_trials=20, grid_n=128, grid_extent=4.0,
    )
    for r in turb_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.name}: {status} (measured={r.measured:.6g}, "
              f"expected={r.expected:.6g}, rel_err={r.rel_error:.4g})")
        if not r.passed:
            all_passed = False
    log_result("V2 Turbulence Model Validation", turb_results)

    print()
    print("[V1] Skyrmion number quantization (Q = q)")
    skyrmion_results = validate_skyrmion_quantization(
        charges=[-4, -3, -2, -1, 1, 2, 3, 4],
        grid_n=128, grid_extent=8.0, skyrmion_radius=1.0,
    )
    for r in skyrmion_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.name}: {status} (measured={r.measured:.6g}, "
              f"expected={r.expected:.6g}, rel_err={r.rel_error:.4g})")
        if not r.passed:
            all_passed = False
    log_result("V1 Skyrmion Quantization", skyrmion_results)

    # ---------------------------------------------------------------
    # Experiment V5/V6: Fair comparison without receiver noise
    # ---------------------------------------------------------------
    print()
    print("--- V5/V6: Fair Comparison (No Receiver Noise) ---")
    print()

    nx = ny = 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    comp = FairComparison(
        grid_x=X, grid_y=Y, wavelength=1.55e-6,
        skyrmion_radius=1.0, seed=42,
        receiver_noise_std=0.0,  # Ideal receiver
    )

    cn2_values = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11]
    results = comp.compare_all(
        skm_constellation=[-1, 1],
        oam_modes=[-1, 1],
        intensity_levels=[0.5, 1.5],
        cn2_values=cn2_values,
        n_trials=50,
        distance=100.0,
        n_screens=5,
        waist=1.0,
        verbose=True,
    )
    log_comparison(results, "Ideal receiver (noise_std=0), 100 m link, 64x64 grid")

    # Summary: does SKM show advantage?
    skm_bers = [r.ber for r in results if r.scheme == "SKM"]
    oam_bers = [r.ber for r in results if r.scheme == "OAM"]
    int_bers = [r.ber for r in results if r.scheme == "Intensity"]

    print()
    print("[Summary — Ideal Receiver]")
    for cn2, sk, om, it in zip(cn2_values, skm_bers, oam_bers, int_bers):
        print(f"  Cn²={cn2:.0e}: SKM={sk:.6f}  OAM={om:.6f}  Int={it:.6f}")

    log_result("V5 Fair Comparison — Ideal Receiver",
               [ValidationResult(
                   name=f"SKM_advantage_Cn2={cn2:.0e}",
                   passed=sk <= om,
                   measured=sk, expected=om,
                   rel_error=abs(sk - om) / max(om, 1e-12),
                   detail=f"SKM BER={sk:.6f}, OAM BER={om:.6f}"
               ) for cn2, sk, om in zip(cn2_values, skm_bers, oam_bers)])

    # ---------------------------------------------------------------
    # Experiment V3: Receiver noise sensitivity
    # ---------------------------------------------------------------
    print()
    print("--- V3: Receiver Noise Sensitivity ---")
    print()

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    for noise_std in noise_levels:
        comp_n = FairComparison(
            grid_x=X, grid_y=Y, wavelength=1.55e-6,
            skyrmion_radius=1.0, seed=42,
            receiver_noise_std=noise_std,
        )
        results_n = comp_n.compare_all(
            skm_constellation=[-1, 1],
            oam_modes=[-1, 1],
            intensity_levels=[0.5, 1.5],
            cn2_values=[1e-13],
            n_trials=50,
            distance=100.0,
            n_screens=5,
            waist=1.0,
            verbose=False,
        )
        skm = [r for r in results_n if r.scheme == "SKM"][0]
        oam = [r for r in results_n if r.scheme == "OAM"][0]
        intensity = [r for r in results_n if r.scheme == "Intensity"][0]
        print(f"  noise_std={noise_std}: SKM={skm.ber:.6f}  OAM={oam.ber:.6f}  "
              f"Int={intensity.ber:.6f}  (Q_std={skm.q_std:.4f})")
        log_comparison(results_n, f"Receiver noise_std={noise_std}")

    # ---------------------------------------------------------------
    # Experiment V7: Grid convergence
    # ---------------------------------------------------------------
    print()
    print("--- V7: Grid Convergence ---")
    print()

    for grid_n in [32, 64, 128]:
        x_g = np.linspace(-3, 3, grid_n)
        y_g = np.linspace(-3, 3, grid_n)
        X_g, Y_g = np.meshgrid(x_g, y_g, indexing='xy')
        comp_g = FairComparison(
            grid_x=X_g, grid_y=Y_g, wavelength=1.55e-6,
            skyrmion_radius=1.0, seed=42,
            receiver_noise_std=0.0,
        )
        results_g = comp_g.compare_all(
            skm_constellation=[-1, 1],
            oam_modes=[-1, 1],
            intensity_levels=[0.5, 1.5],
            cn2_values=[1e-13],
            n_trials=30,
            distance=100.0,
            n_screens=5,
            waist=1.0,
            verbose=False,
        )
        skm_g = [r for r in results_g if r.scheme == "SKM"][0]
        oam_g = [r for r in results_g if r.scheme == "OAM"][0]
        print(f"  grid={grid_n}x{grid_n}: SKM={skm_g.ber:.6f} (Q_std={skm_g.q_std:.4f})  "
              f"OAM={oam_g.ber:.6f}")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Overall physics validation: {'PASS' if all_passed else 'FAIL'}")
    print(f"Logged to: {LOG_PATH}")


if __name__ == "__main__":
    run_all()
