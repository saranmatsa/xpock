"""
Tests for XPOCK → HX-241 integration modules.

Tests cover:
1. Carrier wave modeling (all media types)
2. Channel capacity analysis
3. Wave-structure modulation/demodulation
4. Physically-based impairment models
5. Receiver-side spectral analysis
6. End-to-end HX-241 integration workflow

Run with:  python -m pytest tests/test_hx241_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from xpock import (
    HX241Integration,
    CarrierWaveModeler,
    CarrierWaveConfig,
    CarrierChannelModel,
    CarrierMedium,
    ChannelCapacityAnalyzer,
    WaveStructureSignature,
    WaveModulator,
    ModulationScheme,
    ModulationDimension,
    ModulationResult,
    DemodulationResult,
    create_default_scheme,
    ImpairmentModeler,
    ImpairmentProfile,
    ImpairmentModel,
    impairment_to_hx_sim_dict,
    create_link_scenario_library,
    WaveSignalAnalyzer,
    ReceivedSignalAnalysis,
    AnalysisDomain,
    WaveEvent,
    WaveField,
    CapacityAnalysisResult,
    DistinguishabilityAnalysisResult,
    modulate_frame_payload,
    demodulate_wave_event,
)
from xpock.cli import generate_wave_event


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def grid_arrays():
    """Standard grid arrays for wave generation."""
    x = np.linspace(-5, 5, 64)
    y = np.linspace(-5, 5, 64)
    z = np.array([0.0])
    t = np.linspace(-5, 5, 64)
    return {"x": x, "y": y, "z": z, "t": t}


@pytest.fixture
def sample_event(grid_arrays):
    """Create a sample wave event for testing."""
    params = {
        "temporal_shape": "gaussian",
        "temporal_duration": 1.0,
        "spatial_mode": "gaussian",
        "spatial_waist": 1.0,
        "polarization_state": "linear",
        "topology_charge": 0,
        "coherence_type": "full",
    }
    return generate_wave_event(params, grid_arrays)


# -----------------------------------------------------------------------
# Carrier wave modeling tests
# -----------------------------------------------------------------------

@pytest.mark.parametrize("medium", [m for m in CarrierMedium if m != CarrierMedium.CUSTOM])
def test_carrier_model_all_media(medium):
    """Test carrier wave modeling for all media types."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=medium,
        frequency_hz=1e9,
        distance_m=100.0,
        transmit_power_w=1.0,
    )
    result = modeler.model_carrier(config)

    assert isinstance(result, CarrierChannelModel)
    assert result.config.medium == medium
    assert np.all(np.isfinite(result.frequency_response))
    assert np.all(np.isfinite(result.impulse_response))
    assert np.isfinite(result.path_loss_db)
    assert np.isfinite(result.snr_db)
    assert np.isfinite(result.capacity_bps)
    assert result.capacity_bps >= 0
    assert "packet_loss_rate" in result.impairment_params
    assert "corruption_rate" in result.impairment_params
    assert "base_latency_us" in result.impairment_params


def test_carrier_model_free_space_optical():
    """Test free-space optical carrier model in detail."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.FREE_SPACE_OPTICAL,
        frequency_hz=200e12,  # 1550 nm
        distance_m=1000.0,
        transmit_power_w=0.001,
        medium_specific={"attenuation_db_per_km": 0.2, "cn2": 1e-14},
    )
    result = modeler.model_carrier(config)

    assert result.path_loss_db > 0  # Should have path loss
    assert result.snr_db > -1000  # Should have finite SNR
    assert len(result.medium_notes) > 0  # Should have notes


def test_carrier_model_radio_frequency():
    """Test RF carrier model with multipath."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.RADIO_FREQUENCY,
        frequency_hz=2.4e9,
        distance_m=1000.0,
        transmit_power_w=0.1,
        medium_specific={
            "rms_delay_spread_us": 0.5,
            "shadow_std_db": 6.0,
            "doppler_spread_hz": 10.0,
        },
    )
    result = modeler.model_carrier(config)

    assert result.delay_spread_s > 0  # Should have multipath delay spread
    assert result.doppler_spread_hz > 0  # Should have Doppler spread


def test_carrier_model_optical_fiber():
    """Test optical fiber carrier model with dispersion."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.OPTICAL_FIBER,
        frequency_hz=193e12,
        distance_m=10000.0,
        transmit_power_w=0.001,
        medium_specific={
            "attenuation_db_per_km": 0.2,
            "dispersion_ps_nm_km": 17.0,
            "source_linewidth_nm": 0.1,
        },
    )
    result = modeler.model_carrier(config)

    assert result.delay_spread_s > 0  # Should have dispersion-induced delay spread


def test_carrier_model_magnetic_induction():
    """Test magnetic induction carrier model (near-field)."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.MAGNETIC_INDUCTION,
        frequency_hz=10e6,
        distance_m=1.0,
        transmit_power_w=0.01,
    )
    result = modeler.model_carrier(config)

    assert result.delay_spread_s == 0.0  # No multipath in near-field MI


def test_carrier_model_custom():
    """Test custom carrier model with user-defined transfer function."""
    modeler = CarrierWaveModeler()

    def custom_tf(f):
        return np.exp(-1j * 2.0 * np.pi * f * 1e-6)  # Pure delay

    config = CarrierWaveConfig(
        medium=CarrierMedium.CUSTOM,
        frequency_hz=1e9,
        distance_m=100.0,
        transmit_power_w=1.0,
        medium_specific={"transfer_function": custom_tf},
    )
    result = modeler.model_carrier(config)

    assert np.all(np.isfinite(result.frequency_response))


def test_impairment_to_hx_sim():
    """Test conversion of channel model to hx-sim impairment parameters."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.RADIO_FREQUENCY,
        frequency_hz=2.4e9,
        distance_m=1000.0,
        transmit_power_w=0.1,
    )
    channel = modeler.model_carrier(config)

    impairment = modeler.impairment_to_hx_sim(channel)

    assert "packet_loss_rate" in impairment
    assert "corruption_rate" in impairment
    assert "base_latency_us" in impairment
    assert "jitter_us" in impairment
    assert "bandwidth_bps" in impairment
    assert 0.0 <= impairment["packet_loss_rate"] <= 1.0
    assert 0.0 <= impairment["corruption_rate"] <= 1.0


# -----------------------------------------------------------------------
# Channel capacity analysis tests
# -----------------------------------------------------------------------

def test_capacity_analyzer_basic(sample_event):
    """Test basic channel capacity analysis."""
    analyzer = ChannelCapacityAnalyzer()
    result = analyzer.analyze_capacity(
        snr_db=10.0,
        bandwidth_hz=1e6,
        wave_events=[sample_event],
        error_correction_overhead=0.25,
    )

    assert isinstance(result, CapacityAnalysisResult)
    assert result.snr_db == 10.0
    assert result.bandwidth_hz == 1e6
    assert result.shannon_capacity_bps > 0
    assert result.spectral_efficiency_bps_hz > 0
    assert result.distinguishable_states >= 1
    assert result.bits_per_symbol >= 0
    assert result.net_information_rate_bps >= 0


def test_capacity_analyzer_snr_scaling():
    """Test that capacity increases with SNR."""
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}

    results = []
    for snr in [0.0, 10.0, 20.0]:
        params = {
            "temporal_shape": "gaussian",
            "temporal_duration": 1.0,
            "spatial_mode": "gaussian",
            "spatial_waist": 1.0,
        }
        ev = generate_wave_event(params, grid)
        r = analyzer.analyze_capacity(snr_db=snr, bandwidth_hz=1e6,
                                       wave_events=[ev])
        results.append(r)

    assert results[0].shannon_capacity_bps < results[1].shannon_capacity_bps
    assert results[1].shannon_capacity_bps < results[2].shannon_capacity_bps


def test_capacity_analyzer_bandwidth_scaling():
    """Test that capacity increases with bandwidth."""
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}

    results = []
    for bw in [1e5, 1e6, 1e7]:
        params = {
            "temporal_shape": "gaussian",
            "temporal_duration": 1.0,
            "spatial_mode": "gaussian",
            "spatial_waist": 1.0,
        }
        ev = generate_wave_event(params, grid)
        r = analyzer.analyze_capacity(snr_db=10.0, bandwidth_hz=bw,
                                       wave_events=[ev])
        results.append(r)

    assert results[0].shannon_capacity_bps < results[1].shannon_capacity_bps
    assert results[1].shannon_capacity_bps < results[2].shannon_capacity_bps


def test_distinguishability_analysis(sample_event):
    """Test pairwise distinguishability analysis."""
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}

    # Create events with different parameters
    events = []
    for l in [0, 1, 2]:
        params = {
            "temporal_shape": "gaussian",
            "temporal_duration": 1.0,
            "spatial_mode": "laguerre_gaussian",
            "spatial_l": l,
            "spatial_waist": 1.0,
        }
        ev = generate_wave_event(params, grid)
        events.append(ev)

    result = analyzer.analyze_distinguishability(events)

    assert isinstance(result, DistinguishabilityAnalysisResult)
    assert result.n_structures == 3
    assert result.pairwise_matrix.shape == (3, 3)
    assert result.min_distance >= 0
    assert result.max_distance >= result.min_distance


def test_feature_independence(sample_event):
    """Test feature independence analysis."""
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
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
        ev = generate_wave_event(params, grid)
        events.append(ev)

    result = analyzer.analyze_feature_independence(events)

    assert "n_features" in result
    assert "feature_names" in result
    assert "correlation_matrix" in result
    assert "highly_correlated_pairs" in result
    assert "approximate_independent_groups" in result
    assert result["n_features"] > 0


# -----------------------------------------------------------------------
# Wave-structure modulation tests
# -----------------------------------------------------------------------

def test_modulation_scheme_creation():
    """Test modulation scheme creation and bit capacity."""
    scheme = ModulationScheme()
    scheme.add_dimension(
        ModulationDimension.TEMPORAL_SHAPE,
        ["gaussian", "sech", "super_gaussian"],
    )
    scheme.add_dimension(
        ModulationDimension.OAM_CHARGE,
        [0, 1, 2],
    )

    assert scheme.capacity_bits() > 0
    assert scheme.num_states() == 9  # 3 * 3
    assert scheme.bits_per_dimension[ModulationDimension.TEMPORAL_SHAPE] == pytest.approx(np.log2(3))


def test_modulation_scheme_default():
    """Test default modulation scheme."""
    scheme = create_default_scheme()
    assert scheme.capacity_bits() > 0
    assert scheme.num_states() > 1


def test_modulate_bits(sample_event, grid_arrays):
    """Test bit modulation onto wave structure."""
    scheme = create_default_scheme()
    modulator = WaveModulator(scheme=scheme, grid_arrays=grid_arrays)

    result = modulator.modulate_bits("101101")

    assert isinstance(result, ModulationResult)
    assert result.encoded_bits > 0
    assert result.event is not None
    assert result.event.field.Ex is not None
    assert result.constellation_index >= 0
    assert result.constellation_index < modulator.constellation_size()
    assert len(result.dimension_assignments) > 0


def test_modulate_symbol_index(sample_event, grid_arrays):
    """Test modulation by constellation index."""
    scheme = create_default_scheme()
    modulator = WaveModulator(scheme=scheme, grid_arrays=grid_arrays)

    result = modulator.modulate_bits("000", symbol_index=5)

    assert result.constellation_index == 5
    assert result.event is not None


def test_modulate_demodulate_roundtrip(grid_arrays):
    """Test modulate → demodulate roundtrip."""
    scheme = create_default_scheme()
    modulator = WaveModulator(scheme=scheme, grid_arrays=grid_arrays)

    # Modulate
    mod_result = modulator.modulate_bits("110011")

    # Demodulate
    demod_result = modulator.demodulate_event(mod_result.event)

    assert demod_result.recovered_bits > 0
    assert demod_result.demodulation_error is None
    assert len(demod_result.recovered_dimensions) > 0


def test_modulate_frame_payload(grid_arrays):
    """Test modulating HX-241 frame payload bytes."""
    scheme = create_default_scheme()
    payload = b"\x01\x02\x03\x04"  # 4 bytes = 32 bits

    result = modulate_frame_payload(payload, scheme, grid_arrays)

    assert isinstance(result, ModulationResult)
    assert result.encoded_bits <= 32  # May be truncated to capacity
    assert result.event is not None


def test_demodulate_wave_event(sample_event):
    """Test demodulating a wave event."""
    scheme = create_default_scheme()
    result = demodulate_wave_event(sample_event, scheme)

    assert isinstance(result, DemodulationResult)
    assert result.demodulation_error is None
    assert len(result.recovered_dimensions) > 0


def test_constellation_size(grid_arrays):
    """Test constellation size computation."""
    scheme = ModulationScheme()
    scheme.add_dimension(ModulationDimension.OAM_CHARGE, [0, 1, 2, 3])
    scheme.add_dimension(ModulationDimension.TEMPORAL_SHAPE,
                         ["gaussian", "sech"])

    modulator = WaveModulator(scheme=scheme, grid_arrays=grid_arrays)
    assert modulator.constellation_size() == 8  # 4 * 2


# -----------------------------------------------------------------------
# Impairment model tests
# -----------------------------------------------------------------------

def test_impairment_modeler_radio():
    """Test impairment modeling for RF link."""
    modeler = ImpairmentModeler()
    profile = modeler.generate_impairment_scenario(
        medium=CarrierMedium.RADIO_FREQUENCY,
        distance_m=1000.0,
        frequency_hz=2.4e9,
        transmit_power_w=0.1,
        rms_delay_spread_us=0.5,
        shadow_std_db=6.0,
        doppler_spread_hz=10.0,
    )

    assert isinstance(profile, ImpairmentProfile)
    assert profile.medium == "radio_frequency"
    assert profile.model == ImpairmentModel.MULTIPATH_FADING
    assert np.isfinite(profile.path_loss_db)
    assert np.isfinite(profile.snr_db)
    assert np.isfinite(profile.delay_spread_s)


def test_impairment_modeler_free_space_optical():
    """Test impairment modeling for free-space optical link."""
    modeler = ImpairmentModeler()
    profile = modeler.generate_impairment_scenario(
        medium=CarrierMedium.FREE_SPACE_OPTICAL,
        distance_m=100.0,
        frequency_hz=200e12,
        transmit_power_w=0.001,
        attenuation_db_per_km=0.2,
        cn2=1e-14,
    )

    assert profile.model == ImpairmentModel.ATMOSPHERIC_TURBULENCE
    assert np.isfinite(profile.path_loss_db)


def test_impairment_modeler_optical_fiber():
    """Test impairment modeling for optical fiber link."""
    modeler = ImpairmentModeler()
    profile = modeler.generate_impairment_scenario(
        medium=CarrierMedium.OPTICAL_FIBER,
        distance_m=10000.0,
        frequency_hz=193e12,
        transmit_power_w=0.001,
        attenuation_db_per_km=0.2,
        dispersion_ps_nm_km=17.0,
    )

    assert profile.model == ImpairmentModel.FIBER_DISPERSION
    assert profile.delay_spread_s > 0


def test_impairment_to_hx_sim_dict():
    """Test conversion to hx-sim impairment dict."""
    modeler = ImpairmentModeler()
    profile = modeler.generate_impairment_scenario(
        medium=CarrierMedium.ACOUSTIC,
        distance_m=500.0,
        frequency_hz=20e3,
        transmit_power_w=1.0,
    )

    hx_sim_params = impairment_to_hx_sim_dict(profile)

    assert "packet_loss_rate" in hx_sim_params
    assert "corruption_rate" in hx_sim_params
    assert "base_latency_us" in hx_sim_params
    assert "jitter_us" in hx_sim_params
    assert "bandwidth_bps" in hx_sim_params
    assert hx_sim_params["enabled"] is True


def test_link_scenario_library():
    """Test creation of link scenario library."""
    profiles = create_link_scenario_library()

    assert len(profiles) == 6
    for profile in profiles:
        assert isinstance(profile, ImpairmentProfile)
        assert np.isfinite(profile.path_loss_db)
        assert np.isfinite(profile.snr_db)
        assert 0.0 <= profile.derived_params.get("packet_loss_rate", 0) <= 1.0


def test_batch_scenarios():
    """Test batch scenario generation."""
    modeler = ImpairmentModeler()
    scenarios = [
        {
            "medium": CarrierMedium.RADIO_FREQUENCY,
            "distance_m": 100.0,
            "frequency_hz": 2.4e9,
            "transmit_power_w": 0.1,
        },
        {
            "medium": CarrierMedium.OPTICAL_FIBER,
            "distance_m": 1000.0,
            "frequency_hz": 193e12,
            "transmit_power_w": 0.001,
        },
    ]
    profiles = modeler.batch_scenarios(scenarios)

    assert len(profiles) == 2


# -----------------------------------------------------------------------
# Receiver-side spectral analysis tests
# -----------------------------------------------------------------------

def test_signal_analyzer_basic(sample_event):
    """Test basic signal analysis."""
    analyzer = WaveSignalAnalyzer()
    result = analyzer.analyze_received_signal(sample_event)

    assert isinstance(result, ReceivedSignalAnalysis)
    assert result.event_id == sample_event.event_id
    assert result.recovery_confidence >= 0.0
    assert result.recovery_confidence <= 1.0
    assert len(result.notes) > 0


def test_signal_analyzer_domains(sample_event):
    """Test analysis with specific domains."""
    analyzer = WaveSignalAnalyzer()
    result = analyzer.analyze_received_signal(
        sample_event,
        domains=[AnalysisDomain.TIME, AnalysisDomain.FREQUENCY],
    )

    assert len(result.temporal_features) > 0
    assert len(result.spectral_features) > 0


def test_signal_analyzer_compare():
    """Test comparing two signal analyses."""
    analyzer = WaveSignalAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}

    params1 = {"temporal_shape": "gaussian", "temporal_duration": 1.0,
               "spatial_mode": "gaussian", "spatial_waist": 1.0}
    params2 = {"temporal_shape": "gaussian", "temporal_duration": 1.0,
               "spatial_mode": "gaussian", "spatial_waist": 1.0}

    ev1 = generate_wave_event(params1, grid)
    ev2 = generate_wave_event(params2, grid)

    a1 = analyzer.analyze_received_signal(ev1)
    a2 = analyzer.analyze_received_signal(ev2)

    comparison = analyzer.compare_signals(a1, a2)

    assert "similarity" in comparison
    assert "match" in comparison
    assert 0.0 <= comparison["similarity"] <= 1.0


# -----------------------------------------------------------------------
# End-to-end integration test
# -----------------------------------------------------------------------

def test_end_to_end_hx241_workflow(grid_arrays):
    """
    End-to-end test: modulate → model carrier → analyze capacity →
    generate impairment → analyze received signal → demodulate.
    """
    integration = HX241Integration()
    scheme = create_default_scheme()

    # Step 1: Modulate frame payload onto wave structure
    payload = b"\x0A\x0B\x0C"  # 3 bytes = 24 bits
    mod_result = modulate_frame_payload(payload, scheme, grid_arrays)
    assert mod_result.event is not None

    # Step 2: Model the carrier channel (short-range optical link for reasonable SNR)
    carrier_config = CarrierWaveConfig(
        medium=CarrierMedium.FREE_SPACE_OPTICAL,
        frequency_hz=200e12,
        distance_m=1.0,  # 1 meter - short range for reasonable SNR
        transmit_power_w=0.001,
    )
    channel = integration.carrier_modeler().model_carrier(carrier_config)
    assert channel.snr_db > -1000

    # Step 3: Analyze channel capacity (may be 0 at very low SNR, which is valid)
    capacity = integration.capacity_analyzer().analyze_capacity(
        snr_db=max(channel.snr_db, -100.0),  # Clamp for numerical stability
        bandwidth_hz=1e6,
        wave_events=[mod_result.event],
    )
    assert capacity.shannon_capacity_bps >= 0

    # Step 4: Generate impairment parameters
    impairment = integration.impairment_modeler().model_impairment(carrier_config)
    hx_sim_params = impairment_to_hx_sim_dict(impairment)
    assert hx_sim_params["enabled"] is True

    # Step 5: Analyze received signal (simulated as same event)
    analysis = integration.signal_analyzer().analyze_received_signal(mod_result.event)
    assert analysis.recovery_confidence >= 0.0

    # Step 6: Demodulate
    demod_result = demodulate_wave_event(mod_result.event, scheme)
    assert demod_result.demodulation_error is None

    # All steps completed successfully
    assert True


# -----------------------------------------------------------------------
# HX-241 integration factory tests
# -----------------------------------------------------------------------

def test_hx241_integration_factory():
    """Test HX241Integration factory."""
    integration = HX241Integration()
    components = integration.all_components()

    assert "carrier_modeler" in components
    assert "capacity_analyzer" in components
    assert "modulator" in components
    assert "impairment_modeler" in components
    assert "signal_analyzer" in components

    assert isinstance(components["carrier_modeler"], CarrierWaveModeler)
    assert isinstance(components["capacity_analyzer"], ChannelCapacityAnalyzer)
    assert isinstance(components["modulator"], WaveModulator)
    assert isinstance(components["impairment_modeler"], ImpairmentModeler)
    assert isinstance(components["signal_analyzer"], WaveSignalAnalyzer)
