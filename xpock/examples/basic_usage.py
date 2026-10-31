#!/usr/bin/env python3
"""
XPOCK — Example usage script.

Demonstrates:
1. Generating a single wave event with combined properties
2. Running a parameter sweep
3. Analyzing measurable properties
4. Validating numerical correctness
5. Saving events and visualizations

Run with:  python examples/basic_usage.py
"""

from pathlib import Path
import numpy as np

from xpock.config import Config
from xpock.cli import generate_wave_event
from xpock.analysis import WaveAnalyzer
from xpock.validation import validate_wave_event
from xpock.visualization import visualize_event
from xpock.parameter_space import ParameterSpace, SweepConfig, SweepType


def main():
    out = Path("./xpock_output")
    out.mkdir(parents=True, exist_ok=True)

    # --- Build a grid ---
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    z = np.array([0.0])
    t = np.linspace(-5, 5, 128)
    grid = {"x": x, "y": y, "z": z, "t": t}

    # --- Example 1: Single event — Gaussian pulse, LG spatial mode (OAM=1),
    #              linear polarization, topological charge imprinted ---
    params = {
        "temporal_shape": "gaussian",
        "temporal_amplitude": 1.0,
        "temporal_duration": 1.0,
        "temporal_chirp_rate": 0.0,
        "temporal_carrier_freq": 0.0,
        "spatial_mode": "laguerre_gaussian",
        "spatial_p": 0,
        "spatial_l": 1,          # OAM charge = 1
        "spatial_waist": 1.0,
        "spatial_wavelength": 1.0,
        "spatial_position": (0.0, 0.0, 0.0),
        "polarization_state": "linear",
        "polarization_angle": 0.0,
        "topology_charge_type": "winding",
        "topology_charge": 1,    # additional vortex charge
        "topology_core_radius": 0.1,
        "coherence_type": "full",
    }

    print("Generating single wave event...")
    event = generate_wave_event(params, grid)

    # --- Validate ---
    val = validate_wave_event(event)
    print(val.summary())

    # --- Analyze ---
    analyzer = WaveAnalyzer()
    props = analyzer.analyze(event)
    print(f"\nTotal power: {props.total_power:.6f}")
    print(f"Peak intensity: {props.peak_intensity:.6f}")
    print(f"Temporal FWHM: {props.temporal_duration_fwhm}")
    print(f"Beam waist (r): {props.beam_waist_r}")
    print(f"OAM per photon: {props.oam_per_photon}")
    print(f"Winding number: {props.winding_number}")
    print(f"Degree of polarization: {props.degree_of_polarization}")
    print(f"Schmidt number (nonseparability): {props.schmidt_number}")

    # --- Save + visualize ---
    evt_path = out / "example_event.json"
    event.to_json(evt_path)
    print(f"\nSaved event to {evt_path}")

    vis = visualize_event(event, output_dir=out)
    n_plots = sum(len(v) for v in vis.values() if isinstance(v, list))
    print(f"Generated {n_plots} visualizations in {out}")

    # --- Example 2: Parameter sweep over OAM charge and pulse duration ---
    print("\nRunning parameter sweep...")
    # Use a smaller grid for the sweep to stay laptop-friendly
    x_s = np.linspace(-3, 3, 64)
    y_s = np.linspace(-3, 3, 64)
    z_s = np.array([0.0])
    t_s = np.linspace(-3, 3, 64)
    grid_s = {"x": x_s, "y": y_s, "z": z_s, "t": t_s}

    sweeps = [
        SweepConfig(name="spatial_l", sweep_type=SweepType.CUSTOM, values=[0, 1, 2, 3]),
        SweepConfig(name="temporal_duration", sweep_type=SweepType.LINEAR, start=0.5, stop=2.0, num=4),
    ]
    fixed = {
        "temporal_shape": "gaussian",
        "spatial_mode": "laguerre_gaussian",
        "spatial_p": 0,
        "spatial_waist": 1.0,
        "polarization_state": "linear",
        "topology_charge": 0,
        "coherence_type": "full",
    }
    space = ParameterSpace(sweeps=sweeps, fixed_params=fixed)
    print(f"Sweep size: {len(space)} configurations")

    n_valid = 0
    for i, p in enumerate(space):
        ev = generate_wave_event(p, grid_s)
        v = validate_wave_event(ev)
        pr = WaveAnalyzer().analyze(ev)
        if v.is_valid:
            n_valid += 1
        print(f"  [{i:02d}] l={p['spatial_l']}, dur={p['temporal_duration']:.2f}, "
              f"valid={v.is_valid}, power={pr.total_power:.4f}")

    print(f"\nSweep complete: {n_valid}/{len(space)} valid events")


if __name__ == "__main__":
    main()
