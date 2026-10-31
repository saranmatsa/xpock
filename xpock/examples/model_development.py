#!/usr/bin/env python3
"""
XPOCK — Advanced Model Development with ENGINES Integration

Demonstrates using C:\XPOCK\ENGINES to build actual models for
information-bearing wave structures:

1. Wave propagation models (differential_equations engine)
2. Information capacity analysis (information_theory engine)
3. Advanced spectral analysis (integral_transforms engine)
4. Nonlinear wave dynamics (dynamical_systems engine)
5. Optical material properties (materials_science engine)
6. Experimental data analysis (experimental_physics engine)

Run with:  python examples/model_development.py
"""

from pathlib import Path
import numpy as np
import json

from xpock import (
    XpockEngineFactory,
    WavePropagationEngine,
    WaveInformationEngine,
    WaveSpectralEngine,
    NonlinearWaveDynamicsEngine,
    OpticalMaterialsEngine,
    ExperimentalDataEngine,
    EngineResult,
)
from xpock.cli import generate_wave_event
from xpock.analysis import WaveAnalyzer


def main():
    out = Path("./xpock_output")
    out.mkdir(parents=True, exist_ok=True)

    factory = XpockEngineFactory(workspace=out / "engine_workspace")

    print("=" * 70)
    print("XPOCK — Advanced Model Development with ENGINES")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. WAVE PROPAGATION (differential_equations engine)
    # ------------------------------------------------------------------
    print("\n[1] Wave Propagation — 1D Wave Equation")
    prop = factory.wave_propagation()
    result = prop.solve_wave_equation_1d(
        c=1.0, length=10.0, duration=5.0,
        nx=128, nt=128, initial_shape="gaussian"
    )
    if result.success:
        print(f"  ✓ {result.engine_name}: {result.result['x'].shape[0]}×{result.result['t'].shape[0]} grid")
        print(f"    Total energy: {result.metadata['total_energy']:.6f}")
        for s in result.steps:
            print(f"    {s}")
    else:
        print(f"  ✗ {result.error}")

    print("\n[1b] Nonlinear Schrödinger — Soliton Formation")
    result_nls = prop.solve_nls(
        k=1.0, gamma=0.5, length=10.0, duration=5.0,
        nx=128, nt=128, initial_shape="sech"
    )
    if result_nls.success:
        print(f"  ✓ {result_nls.engine_name}: max intensity = {result_nls.metadata['max_intensity']:.6f}")
        for s in result_nls.steps:
            print(f"    {s}")
    else:
        print(f"  ✗ {result_nls.error}")

    # ------------------------------------------------------------------
    # 2. INFORMATION THEORY (information_theory engine)
    # ------------------------------------------------------------------
    print("\n[2] Information Theory — Wave State Distinguishability")
    info = factory.wave_information()

    # Create two different wave states (Gaussian vs LG with OAM)
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # State A: Gaussian beam
    psi_a = np.exp(-(X**2 + Y**2))
    I_a = np.abs(psi_a)**2
    p_a = I_a / np.sum(I_a)

    # State B: Laguerre-Gaussian with OAM l=1
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    psi_b = r * np.exp(-(X**2 + Y**2)) * np.exp(1j * phi)
    I_b = np.abs(psi_b)**2
    p_b = I_b / np.sum(I_b)

    result_dist = info.state_distinguishability(p_a, p_b, base=2)
    if result_dist.success:
        print(f"  ✓ {result_dist.engine_name}: distinguishable = {result_dist.result['distinguishable']}")
        print(f"    JS divergence: {result_dist.result['js_divergence']:.6f} bits")
        print(f"    KL divergence: {result_dist.result['kl_divergence']:.6f} bits")
        for s in result_dist.steps:
            print(f"    {s}")
    else:
        print(f"  ✗ {result_dist.error}")

    print("\n[2b] Gaussian Channel Capacity")
    for snr in [0.1, 1.0, 10.0, 100.0]:
        result_cap = info.gaussian_channel_capacity(snr, base=2)
        if result_cap.success:
            print(f"    SNR={snr:6.1f}: C = {result_cap.result['capacity']:.4f} bits")

    # ------------------------------------------------------------------
    # 3. SPECTRAL ANALYSIS (integral_transforms engine)
    # ------------------------------------------------------------------
    print("\n[3] Spectral Analysis — Multi-Domain Analysis")
    spectral = factory.wave_spectral()

    # Create a chirped pulse
    t = np.linspace(-10, 10, 1024)
    pulse = np.exp(-(t/2.0)**2) * np.exp(1j * 5.0 * t**2)  # chirped Gaussian

    result_fft = spectral.power_spectrum(pulse, fs=1.0, window="hann")
    if result_fft.success:
        psd = result_fft.result["psd"]
        freqs = result_fft.result["frequencies"]
        peak_idx = np.argmax(psd)
        print(f"  ✓ FFT: peak frequency = {freqs[peak_idx]:.3f}, peak PSD = {psd[peak_idx]:.6f}")
        for s in result_fft.steps:
            print(f"    {s}")

    result_hilbert = spectral.hilbert_analytic_signal(pulse)
    if result_hilbert.success:
        amp = result_hilbert.result["instantaneous_amplitude"]
        freq = result_hilbert.result["instantaneous_frequency"]
        print(f"  ✓ Hilbert: mean amplitude = {np.mean(amp):.4f}, "
              f"mean instantaneous freq = {np.mean(freq):.4f}")
        for s in result_hilbert.steps:
            print(f"    {s}")

    result_cwt = spectral.continuous_wavelet_transform(pulse, fs=1.0, wavelet="morlet")
    if result_cwt.success:
        scalogram = result_cwt.result["scalogram"]
        print(f"  ✓ CWT: scalogram shape = {scalogram.shape}")
        for s in result_cwt.steps:
            print(f"    {s}")

    # ------------------------------------------------------------------
    # 4. NONLINEAR WAVE DYNAMICS (dynamical_systems engine)
    # ------------------------------------------------------------------
    print("\n[4] Nonlinear Wave Dynamics — Coupled Modes & MI")
    nonlinear = factory.nonlinear_waves()

    result_cm = nonlinear.coupled_mode_analysis(kappa=0.1, delta_k=0.0, z_max=20.0)
    if result_cm.success:
        print(f"  ✓ Coupled modes: complete transfer at z = {result_cm.result['z_complete_transfer']:.3f}")
        print(f"    Max power transfer: {result_cm.metadata['max_transfer']:.6f}")
        for s in result_cm.steps:
            print(f"    {s}")

    result_mi = nonlinear.modulation_instability_gain(gamma=0.1, P0=1.0)
    if result_mi.success:
        print(f"  ✓ Modulation instability: max gain = {result_mi.result['max_gain']:.4f} at k = {result_mi.result['k_max_gain']:.4f}")
        print(f"    Unstable modes: {result_mi.result['unstable_modes']}")
        for s in result_mi.steps:
            print(f"    {s}")

    # ------------------------------------------------------------------
    # 5. OPTICAL MATERIALS (materials_science engine)
    # ------------------------------------------------------------------
    print("\n[5] Optical Materials — Kerr Nonlinearity & Sellmeier")
    materials = factory.optical_materials()

    result_n = materials.sellmeier_index(wavelength=1.55)
    if result_n.success:
        print(f"  ✓ Sellmeier at 1.55 μm: n = {result_n.result['n']:.6f}")
        for s in result_n.steps:
            print(f"    {s}")

    result_kerr = materials.kerr_nonlinear_index(n0=1.45, n2=2.6e-20, intensity=1e12)
    if result_kerr.success:
        print(f"  ✓ Kerr: n_eff = {result_kerr.result['n_eff']:.10f}")
        for s in result_kerr.steps:
            print(f"    {s}")

    # ------------------------------------------------------------------
    # 6. EXPERIMENTAL DATA ANALYSIS (experimental_physics engine)
    # ------------------------------------------------------------------
    print("\n[6] Experimental Data Analysis — Wave Ensemble")
    experimental = factory.experimental_data()

    # Build ensemble from parameter sweep
    grid = {"x": np.linspace(-3, 3, 64), "y": np.linspace(-3, 3, 64),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 64)}

    ensemble_data = {
        "oam_charge": [],
        "pulse_duration": [],
        "total_power": [],
        "peak_intensity": [],
        "beam_waist": [],
    }

    from xpock.parameter_space import ParameterSpace, SweepConfig, SweepType
    sweeps = [
        SweepConfig(name="spatial_l", sweep_type=SweepType.CUSTOM, values=[0, 1, 2, 3]),
        SweepConfig(name="temporal_duration", sweep_type=SweepType.LINEAR, start=0.5, stop=2.0, num=4),
    ]
    fixed = {
        "temporal_shape": "gaussian", "spatial_mode": "laguerre_gaussian",
        "spatial_p": 0, "spatial_waist": 1.0,
        "polarization_state": "linear", "topology_charge": 0,
        "coherence_type": "full",
    }
    space = ParameterSpace(sweeps=sweeps, fixed_params=fixed)

    analyzer = WaveAnalyzer()
    for p in space:
        ev = generate_wave_event(p, grid)
        props = analyzer.analyze(ev)
        ensemble_data["oam_charge"].append(p["spatial_l"])
        ensemble_data["pulse_duration"].append(p["temporal_duration"])
        ensemble_data["total_power"].append(props.total_power)
        ensemble_data["peak_intensity"].append(props.peak_intensity)
        ensemble_data["beam_waist"].append(props.beam_waist_r or 0.0)

    ensemble_data = {k: np.array(v) for k, v in ensemble_data.items()}

    result_ens = experimental.analyze_wave_ensemble(ensemble_data)
    if result_ens.success:
        print(f"  ✓ Ensemble: {result_ens.result['n_events']} events, {len(result_ens.result['columns'])} columns")
        for s in result_ens.steps:
            print(f"    {s}")
        for col, stats in result_ens.result["summary"].items():
            print(f"    {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
    else:
        print(f"  ✗ {result_ens.error}")

    # ------------------------------------------------------------------
    # 7. SAVE MODEL RESULTS
    # ------------------------------------------------------------------
    model_results = {
        "wave_propagation": {
            "linear_wave": result.success if result else False,
            "nls": result_nls.success if result_nls else False,
        },
        "information": {
            "distinguishable": result_dist.result.get("distinguishable") if result_dist and result_dist.result else False,
            "js_divergence": result_dist.result.get("js_divergence") if result_dist and result_dist.result else None,
        },
        "spectral": {
            "fft_peak_freq": float(result_fft.result["frequencies"][np.argmax(result_fft.result["psd"])]) if result_fft and result_fft.result else None,
        },
        "nonlinear": {
            "coupled_transfer_z": result_cm.result.get("z_complete_transfer") if result_cm and result_cm.result else None,
            "mi_max_gain": result_mi.result.get("max_gain") if result_mi and result_mi.result else None,
        },
        "materials": {
            "sellmeier_n": result_n.result.get("n") if result_n and result_n.result else None,
        },
        "ensemble": {
            "n_events": result_ens.result.get("n_events") if result_ens and result_ens.result else 0,
        },
    }

    results_path = out / "model_development_results.json"
    with open(results_path, "w") as f:
        json.dump(model_results, f, indent=2, default=str)
    print(f"\n[7] Model results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("All ENGINES integrated and models developed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
