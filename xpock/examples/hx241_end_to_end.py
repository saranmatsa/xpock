#!/usr/bin/env python3
"""
XPOCK — HX-241 End-to-End Integration Example

Demonstrates the complete workflow of using XPOCK capabilities to support
HX-241 communication architecture development:

1. Generate a wave structure with modulated HX-241 frame data
2. Model the physical carrier channel (path loss, delay spread, SNR, capacity)
3. Analyze channel capacity and wave state distinguishability
4. Generate physically-based hx-sim impairment parameters
5. Analyze received signal to recover modulation properties
6. Demodulate back to dimension assignments

Run with:  python examples/hx241_end_to_end.py
"""

from pathlib import Path
import numpy as np
import json

from xpock import (
    HX241Integration,
    CarrierWaveConfig,
    CarrierMedium,
    create_default_scheme,
    modulate_frame_payload,
    demodulate_wave_event,
    impairment_to_hx_sim_dict,
    create_link_scenario_library,
    WaveSignalAnalyzer,
    WaveAnalyzer,
)
from xpock.cli import generate_wave_event


def main():
    out = Path("./xpock_output")
    out.mkdir(parents=True, exist_ok=True)

    integration = HX241Integration(workspace=out / "hx241_workspace")
    scheme = create_default_scheme()

    print("=" * 70)
    print("XPOCK — HX-241 End-to-End Integration Example")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Build grid and generate a wave structure with modulated data
    # ------------------------------------------------------------------
    print("\n[1] Wave Structure Generation with Modulated HX-241 Data")

    x = np.linspace(-5, 5, 64)
    y = np.linspace(-5, 5, 64)
    z = np.array([0.0])
    t = np.linspace(-5, 5, 64)
    grid = {"x": x, "y": y, "z": z, "t": t}

    # Simulate an HX-241 frame payload (e.g., a Control frame with session setup)
    frame_payload = b"\x02\x01\x00\x01\x00\x00\x00\x01"  # 8 bytes = 64 bits

    mod_result = modulate_frame_payload(frame_payload, scheme, grid)
    print(f"  ✓ Modulated {mod_result.encoded_bits} bits onto wave structure")
    print(f"  ✓ Constellation size: {mod_result.scheme.num_states()}")
    print(f"  ✓ Dimension assignments:")
    for dim, val in mod_result.dimension_assignments.items():
        print(f"    - {dim.value}: {val}")
    print(f"  ✓ Event ID: {mod_result.event.event_id[:8]}")
    print(f"  ✓ Field shape: {mod_result.event.field.shape}")

    # ------------------------------------------------------------------
    # Step 2: Model the physical carrier channel
    # ------------------------------------------------------------------
    print("\n[2] Physical Carrier Channel Modeling")

    # Test multiple carrier media
    carrier_scenarios = [
        (CarrierMedium.FREE_SPACE_OPTICAL, 1.0, 200e12, 0.001),     # 1m, 1550nm, 1mW
        (CarrierMedium.RADIO_FREQUENCY, 100.0, 2.4e9, 0.1),         # 100m, 2.4GHz, 100mW
        (CarrierMedium.OPTICAL_FIBER, 1000.0, 193e12, 0.001),       # 1km, 1550nm, 1mW
        (CarrierMedium.ACOUSTIC, 100.0, 20e3, 1.0),                 # 100m, 20kHz, 1W
    ]

    channel_results = {}
    for medium, distance, freq, power in carrier_scenarios:
        config = CarrierWaveConfig(
            medium=medium,
            frequency_hz=freq,
            distance_m=distance,
            transmit_power_w=power,
        )
        channel = integration.carrier_modeler().model_carrier(config)
        channel_results[medium.value] = channel

        print(f"\n  Carrier: {medium.value}")
        print(f"    Distance: {distance} m, Frequency: {freq/1e9:.1f} GHz, Power: {power} W")
        print(f"    Path loss: {channel.path_loss_db:.1f} dB")
        print(f"    SNR: {channel.snr_db:.1f} dB")
        print(f"    Delay spread: {channel.delay_spread_s*1e6:.2f} μs")
        print(f"    Coherence bandwidth: {channel.coherence_bandwidth_hz:.1f} Hz")
        print(f"    Doppler spread: {channel.doppler_spread_hz:.1f} Hz")
        print(f"    Channel capacity: {channel.capacity_bps:.1f} bps")
        print(f"    Notes: {'; '.join(channel.medium_notes[:2])}")

    # ------------------------------------------------------------------
    # Step 3: Channel capacity and distinguishability analysis
    # ------------------------------------------------------------------
    print("\n[3] Channel Capacity & Wave State Distinguishability")

    analyzer = integration.capacity_analyzer()

    # Use the best carrier scenario (free-space optical at 1m)
    best_channel = channel_results["free_space_optical"]
    snr_db = max(best_channel.snr_db, -100.0)

    capacity = analyzer.analyze_capacity(
        snr_db=snr_db,
        bandwidth_hz=1e6,
        wave_events=[mod_result.event],
        error_correction_overhead=0.25,
    )

    print(f"  SNR: {snr_db:.1f} dB")
    print(f"  Bandwidth: {capacity.bandwidth_hz:.0f} Hz")
    print(f"  Shannon capacity: {capacity.shannon_capacity_bps:.1f} bps")
    print(f"  Spectral efficiency: {capacity.spectral_efficiency_bps_hz:.4f} bps/Hz")
    print(f"  Usable capacity (75% FEC): {capacity.usable_capacity_bps:.1f} bps")
    print(f"  Distinguishable states: {capacity.distinguishable_states}")
    print(f"  Bits per symbol: {capacity.bits_per_symbol:.3f}")
    print(f"  Net information rate: {capacity.net_information_rate_bps:.1f} bps")
    print(f"  Min JS divergence: {capacity.min_js_divergence:.4f}")

    # Distinguishability across multiple wave structures
    print("\n  Wave state distinguishability (multiple OAM charges):")
    grid_small = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
                  "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}

    events = []
    for l in [0, 1, 2, 3]:
        params = {
            "temporal_shape": "gaussian",
            "temporal_duration": 1.0,
            "spatial_mode": "laguerre_gaussian",
            "spatial_l": l,
            "spatial_waist": 1.0,
        }
        ev = generate_wave_event(params, grid_small)
        events.append(ev)

    dist_result = analyzer.analyze_distinguishability(events)
    print(f"    Structures: {dist_result.n_structures}")
    print(f"    Min distance: {dist_result.min_distance:.4f} (pair {dist_result.most_confusable_pair})")
    print(f"    Max distance: {dist_result.max_distance:.4f} (pair {dist_result.most_distinct_pair})")
    print(f"    Clusters: {len(dist_result.clusters)} (sizes: {[len(c) for c in dist_result.clusters]})")

    # Feature independence
    feat_result = analyzer.analyze_feature_independence(events)
    print(f"\n  Feature independence analysis:")
    print(f"    Independent feature groups: {feat_result['n_independent_groups']}")
    for group in feat_result["approximate_independent_groups"]:
        print(f"      - {group}")
    if feat_result["highly_correlated_pairs"]:
        print(f"    Highly correlated pairs: {len(feat_result['highly_correlated_pairs'])}")

    # ------------------------------------------------------------------
    # Step 4: Physically-based impairment parameters for hx-sim
    # ------------------------------------------------------------------
    print("\n[4] Physically-Based Impairment Parameters (hx-sim compatible)")

    impairment = integration.impairment_modeler().model_impairment(
        CarrierWaveConfig(
            medium=CarrierMedium.RADIO_FREQUENCY,
            frequency_hz=2.4e9,
            distance_m=100.0,
            transmit_power_w=0.1,
            medium_specific={
                "rms_delay_spread_us": 0.5,
                "shadow_std_db": 6.0,
                "doppler_spread_hz": 10.0,
            },
        )
    )

    hx_sim_params = impairment_to_hx_sim_dict(impairment)
    print(f"  Model: {impairment.model.value}")
    print(f"  Path loss: {impairment.path_loss_db:.1f} dB")
    print(f"  SNR: {impairment.snr_db:.1f} dB")
    print(f"  hx-sim parameters:")
    for key, val in hx_sim_params.items():
        if key not in ("source", "medium", "path_loss_db", "snr_db", "delay_spread_s",
                       "coherence_bandwidth_hz", "doppler_spread_hz"):
            print(f"    {key}: {val}")

    # Link scenario library
    print("\n  Link scenario library (pre-configured impairment profiles):")
    library = create_link_scenario_library()
    for profile in library:
        params = impairment_to_hx_sim_dict(profile)
        print(f"    {profile.medium:25s} | PL={profile.path_loss_db:6.1f} dB | "
              f"SNR={profile.snr_db:6.1f} dB | Loss={params['packet_loss_rate']:.3f} | "
              f"Corr={params['corruption_rate']:.3f} | Jitter={params['jitter_us']} μs")

    # ------------------------------------------------------------------
    # Step 5: Receiver-side signal analysis
    # ------------------------------------------------------------------
    print("\n[5] Receiver-Side Signal Analysis")

    signal_analyzer = integration.signal_analyzer()
    analysis = signal_analyzer.analyze_received_signal(mod_result.event)

    print(f"  Event ID: {analysis.event_id[:8]}")
    print(f"  Recovery confidence: {analysis.recovery_confidence:.3f}")
    print(f"  SNR estimate: {analysis.snr_estimate_db:.1f} dB")
    print(f"  Bandwidth estimate: {analysis.bandwidth_estimate_hz:.1f} Hz")
    print(f"  Duration estimate: {analysis.duration_estimate_s:.4f} s")
    print(f"  Recovered modulation parameters:")

    for key, val in analysis.recovered_modulation.items():
        print(f"    - {key}: {val}")

    print(f"  Feature counts by domain:")
    print(f"    Temporal: {len(analysis.temporal_features)}")
    print(f"    Spectral: {len(analysis.spectral_features)}")
    print(f"    Spatial: {len(analysis.spatial_features)}")
    print(f"    Polarization: {len(analysis.polarization_features)}")
    print(f"    Topological: {len(analysis.topological_features)}")

    # ------------------------------------------------------------------
    # Step 6: Demodulation
    # ------------------------------------------------------------------
    print("\n[6] Wave Structure Demodulation")

    demod_result = demodulate_wave_event(mod_result.event, scheme)
    print(f"  Demodulation error: {demod_result.demodulation_error}")
    print(f"  Recovered bits: {demod_result.recovered_bits}")
    print(f"  Recovered dimensions:")
    for dim, val in demod_result.recovered_dimensions.items():
        conf = demod_result.confidence.get(dim, 0.0)
        print(f"    - {dim.value}: {val} (confidence: {conf:.2f})")

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print("\n[7] Saving Results")

    results = {
        "modulation": mod_result.to_dict(),
        "demodulation": demod_result.to_dict(),
        "capacity": capacity.to_dict(),
        "distinguishability": dist_result.to_dict(),
        "impairment": {
            "model": impairment.model.value,
            "path_loss_db": impairment.path_loss_db,
            "snr_db": impairment.snr_db,
            "delay_spread_s": impairment.delay_spread_s,
            "coherence_bandwidth_hz": impairment.coherence_bandwidth_hz,
            "doppler_spread_hz": impairment.doppler_spread_hz,
            "hx_sim_params": hx_sim_params,
            "notes": impairment.notes,
        },
        "signal_analysis": analysis.to_dict(),
        "link_library": [
            {
                "medium": p.medium,
                "path_loss_db": p.path_loss_db,
                "snr_db": p.snr_db,
                "hx_sim_params": impairment_to_hx_sim_dict(p),
            }
            for p in library
        ],
    }

    results_path = out / "hx241_end_to_end_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ Results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("HX-241 end-to-end integration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
