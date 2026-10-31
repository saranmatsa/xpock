"""Command-line interface for XPOCK.

Usage:
    xpock generate --config configs/default.yaml
    xpock sweep --config configs/sweep.yaml --output ./results
    xpock analyze --event events/evt_*.json
    xpock visualize --event events/evt_*.json --output ./figures
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List

from xpock.config import Config
from xpock.core import WaveEvent, WaveField
from xpock.properties import (
    TemporalProfile, PulseShape,
    SpatialMode, SpatialModeConfig,
    PolarizationTexture, PolarizationConfig, PolarizationState,
    TopologicalCharge, TopologicalChargeConfig,
    CoherenceModel, CoherenceConfig,
    CoupledMode, CoupledModeConfig,
)
from xpock.parameter_space import ParameterSpace, SweepConfig, SweepType
from xpock.analysis import WaveAnalyzer, analyze_event
from xpock.visualization import WaveVisualizer, visualize_event
from xpock.validation import validate_wave_event, validate_batch

import numpy as np


# -----------------------------------------------------------------------
# Wave generator: assembles properties into a WaveEvent
# -----------------------------------------------------------------------

def generate_wave_event(
    parameters: dict,
    grid_arrays: dict,
) -> WaveEvent:
    """Generate a single WaveEvent from a parameter dict and grid arrays.

    Parameters dict may contain:
      temporal_shape, temporal_amplitude, temporal_center, temporal_duration,
      temporal_chirp_rate, temporal_carrier_freq
      spatial_mode, spatial_n, spatial_m, spatial_p, spatial_l,
      spatial_waist, spatial_wavelength, spatial_position (x,y,z),
      spatial_aperture_radius
      polarization_state, polarization_angle, polarization_handedness,
      polarization_skyrmion_radius, polarization_skyrmion_charge
      topology_charge_type, topology_charge, topology_core_radius
      coherence_type, coherence_time, coherence_length, spectral_width
      coupled_kappa, coupled_length, coupled_phase_mismatch
    """
    x = grid_arrays["x"]
    y = grid_arrays["y"]
    z = grid_arrays.get("z", np.array([0.0]))
    t = grid_arrays["t"]

    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = float(z[len(z) // 2]) if len(z) > 0 else 0.0

    # --- Temporal envelope ---
    tc = TemporalProfile(
        shape=PulseShape(parameters.get("temporal_shape", "gaussian")),
        amplitude=parameters.get("temporal_amplitude", 1.0),
        center=parameters.get("temporal_center", 0.0),
        duration=parameters.get("temporal_duration", 1.0),
        chirp_rate=parameters.get("temporal_chirp_rate", 0.0),
        carrier_freq=parameters.get("temporal_carrier_freq", 0.0),
    )
    env_t = tc.evaluate(t)  # shape (nt,)

    # --- Spatial mode ---
    smc = SpatialModeConfig(
        mode=parameters.get("spatial_mode", "gaussian"),
        n=parameters.get("spatial_n", 0),
        m=parameters.get("spatial_m", 0),
        p=parameters.get("spatial_p", 0),
        l=parameters.get("spatial_l", 0),
        waist=parameters.get("spatial_waist", 1.0),
        wavelength=parameters.get("spatial_wavelength", 1.0),
        position=parameters.get("spatial_position", (0.0, 0.0, 0.0)),
        aperture_radius=parameters.get("spatial_aperture_radius"),
    )
    sm = SpatialMode(smc)
    psi_xy = sm.evaluate(X, Y, Z)  # shape (ny, nx)

    # --- Combine spatial * temporal ---
    # psi_xy: (ny, nx), env_t: (nt,) -> (ny, nx, nt)
    psi = psi_xy[:, :, np.newaxis] * env_t[np.newaxis, np.newaxis, :]

    # --- Polarization ---
    pol_cfg = PolarizationConfig(
        state=PolarizationState(parameters.get("polarization_state", "linear")),
        angle=parameters.get("polarization_angle", 0.0),
        handedness=parameters.get("polarization_handedness", 1),
        skyrmion_radius=parameters.get("polarization_skyrmion_radius", 1.0),
        skyrmion_charge=parameters.get("polarization_skyrmion_charge", 1),
    )
    pol = PolarizationTexture(pol_cfg)
    Ex_pol, Ey_pol = pol.evaluate(X, Y)  # (ny, nx)

    # Apply polarization to field components
    Ex = psi * Ex_pol[:, :, np.newaxis]
    Ey = psi * Ey_pol[:, :, np.newaxis]
    Ez = np.zeros_like(Ex)

    # --- Topological charge ---
    if parameters.get("topology_charge", 0) != 0:
        tc_obj = TopologicalCharge(TopologicalChargeConfig(
            charge_type=parameters.get("topology_charge_type", "winding"),
            charge=parameters.get("topology_charge", 1),
            core_radius=parameters.get("topology_core_radius", 0.1),
        ))
        Ex = tc_obj.apply_phase_singularity(Ex, X, Y)
        Ey = tc_obj.apply_phase_singularity(Ey, X, Y)

    # --- Coherence ---
    coh_cfg = CoherenceConfig(
        coherence_type=parameters.get("coherence_type", "full"),
        coherence_time=parameters.get("coherence_time", 1.0),
        coherence_length=parameters.get("coherence_length", 1.0),
        spectral_width=parameters.get("spectral_width", 0.1),
    )
    coh = CoherenceModel(coh_cfg)
    # For full coherence, just use intensity; for partial, coherence averaging
    # is applied at analysis time. Here we note it in parameters.

    # --- Coupled modes ---
    if parameters.get("coupled_kappa", 0.0) != 0.0:
        cm_cfg = CoupledModeConfig(
            coupling_coefficient=parameters.get("coupled_kappa", 0.1),
            interaction_length=parameters.get("coupled_length", 1.0),
            phase_mismatch=parameters.get("coupled_phase_mismatch", 0.0),
        )
        cm = CoupledMode(cm_cfg)
        z_arr = np.array([Z])
        Ex_evolved, Ey_evolved = cm.evolve(Ex[:, :, 0], Ey[:, :, 0], z_arr)
        Ex = Ex_evolved[:, :, np.newaxis] * env_t[np.newaxis, np.newaxis, :]
        Ey = Ey_evolved[:, :, np.newaxis] * env_t[np.newaxis, np.newaxis, :]

    # Build 4D field: (1, ny, nx, nt) for consistency
    Ex_4d = Ex[np.newaxis, :, :, :]
    Ey_4d = Ey[np.newaxis, :, :, :]
    Ez_4d = Ez[np.newaxis, :, :, :]

    field = WaveField(
        Ex=Ex_4d,
        Ey=Ey_4d,
        Ez=Ez_4d,
        x=x,
        y=y,
        z=z,
        t=t,
    )

    event = WaveEvent.create(parameters=parameters, field=field)
    return event


# -----------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    grid = config.build_grid_arrays()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "temporal_shape": config.temporal.shape,
        "temporal_amplitude": config.temporal.amplitude,
        "temporal_center": config.temporal.center,
        "temporal_duration": config.temporal.duration,
        "temporal_chirp_rate": config.temporal.chirp_rate,
        "temporal_carrier_freq": config.temporal.carrier_freq,
        "spatial_mode": config.spatial.mode,
        "spatial_n": config.spatial.n,
        "spatial_m": config.spatial.m,
        "spatial_p": config.spatial.p,
        "spatial_l": config.spatial.l,
        "spatial_waist": config.spatial.waist,
        "spatial_wavelength": config.spatial.wavelength,
        "spatial_position": (config.spatial.position_x, config.spatial.position_y, config.spatial.position_z),
        "spatial_aperture_radius": config.spatial.aperture_radius,
        "polarization_state": config.polarization.state,
        "polarization_angle": config.polarization.angle,
        "polarization_handedness": config.polarization.handedness,
        "polarization_skyrmion_radius": config.polarization.skyrmion_radius,
        "polarization_skyrmion_charge": config.polarization.skyrmion_charge,
        "topology_charge_type": config.topology.charge_type,
        "topology_charge": config.topology.charge,
        "topology_core_radius": config.topology.core_radius,
        "coherence_type": config.coherence.coherence_type,
        "coherence_time": config.coherence.coherence_time,
        "coherence_length": config.coherence.coherence_length,
        "spectral_width": config.coherence.spectral_width,
    }
    params.update(config.fixed_params)

    event = generate_wave_event(params, grid)

    # Validate
    val = validate_wave_event(event)
    event.validation = val.checks
    event.properties = analyze_event(event).to_dict()

    print(val.summary())

    # Save event
    if config.save_events:
        evt_path = output_dir / f"event_{event.event_id[:8]}.json"
        event.to_json(evt_path)
        print(f"Saved event: {evt_path}")

    # Visualize
    if config.generate_visualizations:
        vis_paths = visualize_event(event, output_dir=output_dir)
        print(f"Generated {sum(len(v) for v in vis_paths.values() if isinstance(v, list))} visualizations")

    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    grid = config.build_grid_arrays()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter space from config sweeps
    from xpock.parameter_space import SweepConfig as PSC, SweepType as PST
    sweeps = []
    for s in config.sweeps:
        st = PST(s.sweep_type)
        sweeps.append(PSC(name=s.name, sweep_type=st, start=s.start, stop=s.stop,
                         num=s.num, values=s.values))

    space = ParameterSpace(sweeps=sweeps, fixed_params=config.fixed_params)
    print(f"Parameter space size: {len(space)} configurations")

    analyzer = WaveAnalyzer()
    visualizer = WaveVisualizer(output_dir=output_dir)

    results = []
    for i, params in enumerate(space):
        event = generate_wave_event(params, grid)
        val = validate_wave_event(event)
        event.validation = val.checks
        props = analyzer.analyze(event)
        event.properties = props.to_dict()

        # Save
        if config.save_events:
            evt_path = output_dir / f"event_{i:06d}_{event.event_id[:8]}.json"
            event.to_json(evt_path)

        # Visualize subset
        if config.generate_visualizations and i < 5:
            visualizer.generate_all(event)

        results.append({
            "index": i,
            "event_id": event.event_id,
            "parameters": params,
            "valid": val.is_valid,
            "total_power": props.total_power,
        })

    # Write summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Sweep complete. {len(results)} events generated. Summary: {summary_path}")

    n_valid = sum(1 for r in results if r["valid"])
    print(f"Valid events: {n_valid}/{len(results)}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    event_paths = list(Path(".").glob(args.event)) if not Path(args.event).exists() else [Path(args.event)]
    for ep in event_paths:
        event = WaveEvent.from_json(ep)
        props = analyze_event(event)
        val = validate_wave_event(event)
        print(f"\n=== Event: {event.event_id[:8]} ({ep.name}) ===")
        print(val.summary())
        print(f"Total power: {props.total_power:.6f}")
        print(f"Peak intensity: {props.peak_intensity:.6f}")
        if props.temporal_duration_fwhm:
            print(f"Temporal FWHM: {props.temporal_duration_fwhm:.4f}")
        if props.beam_waist_r:
            print(f"Beam waist (rms): {props.beam_waist_r:.4f}")
        if props.oam_per_photon is not None:
            print(f"OAM per photon: {props.oam_per_photon:.4f} hbar")
        if props.winding_number is not None:
            print(f"Winding number: {props.winding_number}")
        if props.degree_of_polarization is not None:
            print(f"Degree of polarization: {props.degree_of_polarization:.4f}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    event_paths = list(Path(".").glob(args.event)) if not Path(args.event).exists() else [Path(args.event)]
    output_dir = Path(args.output) if args.output else Path("./xpock_output")
    for ep in event_paths:
        event = WaveEvent.from_json(ep)
        paths = visualize_event(event, output_dir=output_dir)
        print(f"\nEvent {event.event_id[:8]}:")
        for k, v in paths.items():
            n = len(v) if isinstance(v, list) else 0
            print(f"  {k}: {n} plots")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Generate a default configuration file."""
    config = Config()
    out = Path(args.output) if args.output else Path("configs/default.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    config.save(out)
    print(f"Default config written to: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xpock",
        description="XPOCK — Wave-Structure Generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate a single wave event")
    p_gen.add_argument("--config", required=True, help="YAML config file")
    p_gen.set_defaults(func=cmd_generate)

    # sweep
    p_sw = sub.add_parser("sweep", help="Run a parameter sweep")
    p_sw.add_argument("--config", required=True, help="YAML config file with sweeps")
    p_sw.add_argument("--output", default="./xpock_output", help="Output directory")
    p_sw.set_defaults(func=cmd_sweep)

    # analyze
    p_an = sub.add_parser("analyze", help="Analyze saved event(s)")
    p_an.add_argument("event", help="Event JSON file or glob pattern")
    p_an.set_defaults(func=cmd_analyze)

    # visualize
    p_vis = sub.add_parser("visualize", help="Visualize saved event(s)")
    p_vis.add_argument("event", help="Event JSON file or glob pattern")
    p_vis.add_argument("--output", default="./xpock_output", help="Output directory")
    p_vis.set_defaults(func=cmd_visualize)

    # config
    p_cfg = sub.add_parser("config", help="Generate a default config file")
    p_cfg.add_argument("--output", default="configs/default.yaml", help="Output path")
    p_cfg.set_defaults(func=cmd_config)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
