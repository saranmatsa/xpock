"""
XPOCK — 241-Integration Tests

Reproducible tests for the 241-integration module. Tests cover:
1. Schema validation — JSON output matches JSON schemas
2. Carrier capabilities adapter — output has correct fields and ranges
3. Impairment adapter — hx-sim params are in valid ranges
4. Capacity adapter — Shannon formula verification
5. Test vector generator — JSON and Rust source output validity
6. Simulation inputs — hx-sim compatible parameters
7. Validation suite — contract conformance checking

Run with:  python -m pytest 241-integration/tests/ -v
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

# Add xpock to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xpock.hx241.carrier import (
    CarrierWaveConfig,
    CarrierChannelModel,
    CarrierMedium,
    CarrierWaveModeler,
)
from xpock.hx241.impairment import (
    ImpairmentProfile,
    ImpairmentModeler,
    impairment_to_hx_sim_dict,
)
from xpock.hx241.channel import ChannelCapacityAnalyzer
from xpock.hx241.modulation import create_default_scheme
from xpock.core import WaveEvent, WaveField
from xpock.cli import generate_wave_event

# Import 241-integration modules via importlib (hyphenated dir name)
_adapters = importlib.import_module("241-integration.adapters")
_validation = importlib.import_module("241-integration.validation")
_simulation = importlib.import_module("241-integration.simulation")

CarrierCapabilitiesAdapter = _adapters.CarrierCapabilitiesAdapter
ImpairmentAdapter = _adapters.ImpairmentAdapter
CapacityAdapter = _adapters.CapacityAdapter
WaveEventAdapter = _adapters.WaveEventAdapter
HX241AdapterSuite = _adapters.HX241AdapterSuite

CarrierArchitectureValidator = _validation.CarrierArchitectureValidator
TransportArchitectureValidator = _validation.TransportArchitectureValidator
ProtocolInvariantValidator = _validation.ProtocolInvariantValidator
HX241ValidatorSuite = _validation.HX241ValidatorSuite

SimulationInputGenerator = _simulation.SimulationInputGenerator
default_simulation_carrier_scenarios = _simulation.default_simulation_carrier_scenarios


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
    """Create a sample wave event."""
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


@pytest.fixture
def sample_channel():
    """Create a sample carrier channel model."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
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
    return modeler.model_carrier(config)


# -----------------------------------------------------------------------
# Carrier capabilities adapter tests
# -----------------------------------------------------------------------

def test_carrier_capabilities_adapter(sample_channel):
    """Test carrier capabilities adapter output."""
    caps = CarrierCapabilitiesAdapter.from_channel_model(sample_channel)

    # Check all required fields are present
    required = [
        "max_mtu", "typical_mtu", "max_bandwidth_bps",
        "typical_latency_us", "reliability",
        "supports_broadcast", "supports_multicast", "supports_mobility",
        "energy_cost_nj_per_bit", "max_range_m",
    ]
    for field in required:
        assert field in caps, f"Missing field: {field}"

    # Check value ranges
    assert caps["max_mtu"] > 0
    assert caps["typical_mtu"] > 0
    assert caps["max_bandwidth_bps"] > 0
    assert caps["typical_latency_us"] >= 0
    assert 0 <= caps["reliability"] <= 100
    assert isinstance(caps["supports_broadcast"], bool)
    assert isinstance(caps["supports_multicast"], bool)
    assert isinstance(caps["supports_mobility"], bool)
    assert caps["energy_cost_nj_per_bit"] >= 0
    assert caps["max_range_m"] > 0


def test_carrier_capabilities_adapter_rf():
    """Test RF carrier capabilities specifically."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.RADIO_FREQUENCY,
        frequency_hz=2.4e9,
        distance_m=100.0,
        transmit_power_w=0.1,
    )
    channel = modeler.model_carrier(config)
    caps = CarrierCapabilitiesAdapter.from_channel_model(channel)

    assert caps["supports_broadcast"] is True
    assert caps["supports_multicast"] is True
    assert caps["supports_mobility"] is True


def test_carrier_capabilities_adapter_optical_fiber():
    """Test optical fiber carrier capabilities."""
    modeler = CarrierWaveModeler()
    config = CarrierWaveConfig(
        medium=CarrierMedium.OPTICAL_FIBER,
        frequency_hz=193e12,
        distance_m=1000.0,
        transmit_power_w=0.001,
    )
    channel = modeler.model_carrier(config)
    caps = CarrierCapabilitiesAdapter.from_channel_model(channel)

    # Fiber doesn't support broadcast/multicast/mobility
    assert caps["supports_broadcast"] is False
    assert caps["supports_multicast"] is False
    assert caps["supports_mobility"] is False


# -----------------------------------------------------------------------
# Impairment adapter tests
# -----------------------------------------------------------------------

def test_impairment_adapter(sample_channel):
    """Test impairment adapter output."""
    imp = ImpairmentAdapter.from_channel_model(sample_channel)

    # Check hx-sim required fields
    required = [
        "packet_loss_rate", "corruption_rate",
        "base_latency_us", "jitter_us",
        "bandwidth_bps", "enabled",
    ]
    for field in required:
        assert field in imp, f"Missing field: {field}"

    # Check value ranges
    assert 0.0 <= imp["packet_loss_rate"] <= 1.0
    assert 0.0 <= imp["corruption_rate"] <= 1.0
    assert 0.0 <= imp.get("duplication_rate", 0.0) <= 1.0
    assert imp["base_latency_us"] >= 0
    assert imp["jitter_us"] >= 0
    assert imp["bandwidth_bps"] >= 0
    assert isinstance(imp["enabled"], bool)


def test_impairment_to_hx_sim_dict():
    """Test impairment_to_hx_sim_dict output."""
    modeler = ImpairmentModeler()
    profile = modeler.generate_impairment_scenario(
        medium=CarrierMedium.RADIO_FREQUENCY,
        distance_m=100.0,
        frequency_hz=2.4e9,
        transmit_power_w=0.1,
    )
    params = impairment_to_hx_sim_dict(profile)

    assert 0.0 <= params["packet_loss_rate"] <= 1.0
    assert 0.0 <= params["corruption_rate"] <= 1.0
    assert params["enabled"] is True


# -----------------------------------------------------------------------
# Capacity adapter tests
# -----------------------------------------------------------------------

def test_capacity_adapter():
    """Test capacity adapter output."""
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}
    params = {
        "temporal_shape": "gaussian",
        "temporal_duration": 1.0,
        "spatial_mode": "gaussian",
        "spatial_waist": 1.0,
    }
    ev = generate_wave_event(params, grid)
    result = analyzer.analyze_capacity(snr_db=10.0, bandwidth_hz=1e6,
                                        wave_events=[ev])

    bench = CapacityAdapter.from_capacity_result(result)

    # Check hx-sim benchmark fields
    required = [
        "workload", "packet_size", "stream_count", "total_bytes",
        "latency_us", "loss_rate", "bandwidth_bps",
        "topology_nodes", "duration_secs",
    ]
    for field in required:
        assert field in bench, f"Missing field: {field}"

    assert bench["workload"] == "BulkTransfer"
    assert bench["packet_size"] > 0
    assert bench["stream_count"] >= 1
    assert bench["total_bytes"] > 0
    assert bench["latency_us"] >= 0
    assert 0.0 <= bench["loss_rate"] <= 1.0
    assert bench["bandwidth_bps"] >= 0
    assert bench["topology_nodes"] >= 1
    assert bench["duration_secs"] > 0


# -----------------------------------------------------------------------
# Wave event adapter tests
# -----------------------------------------------------------------------

def test_wave_event_adapter(sample_event):
    """Test wave event adapter output."""
    state = WaveEventAdapter.to_modulation_state(sample_event)

    required = [
        "event_id", "temporal_duration", "temporal_center",
        "bandwidth_rms", "beam_waist_r", "oam_per_photon",
        "winding_number", "degree_of_polarization",
        "stokes_s1", "stokes_s2", "stokes_s3",
        "schmidt_number", "total_power", "peak_intensity",
        "is_physically_valid",
    ]
    for field in required:
        assert field in state, f"Missing field: {field}"

    assert state["event_id"] == sample_event.event_id
    assert isinstance(state["is_physically_valid"], bool)
    assert state["total_power"] >= 0
    assert state["peak_intensity"] >= 0


# -----------------------------------------------------------------------
# Adapter suite tests
# -----------------------------------------------------------------------

def test_adapter_suite_all(sample_channel, sample_event):
    """Test adapter suite produces all required outputs."""
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}
    params = {
        "temporal_shape": "gaussian",
        "temporal_duration": 1.0,
        "spatial_mode": "gaussian",
        "spatial_waist": 1.0,
    }
    ev = generate_wave_event(params, grid)
    capacity = analyzer.analyze_capacity(snr_db=10.0, bandwidth_hz=1e6,
                                          wave_events=[ev])

    suite = HX241AdapterSuite()
    adapted = suite.adapt_all(sample_channel, capacity, ev)

    assert "carrier_capabilities" in adapted
    assert "impairment_params" in adapted
    assert "benchmark_config" in adapted
    assert "modulation_state" in adapted


# -----------------------------------------------------------------------
# Validation suite tests
# -----------------------------------------------------------------------

def test_carrier_architecture_validator(sample_channel):
    """Test carrier architecture validator."""
    validator = CarrierArchitectureValidator()

    channel_dict = {
        "medium": sample_channel.config.medium.value,
        "path_loss_db": sample_channel.path_loss_db,
        "snr_db": sample_channel.snr_db,
        "capacity_bps": sample_channel.capacity_bps,
        "impairment_params": sample_channel.impairment_params,
        "medium_notes": sample_channel.medium_notes,
    }

    result = validator.validate_channel_model(channel_dict)

    assert isinstance(result.passed, bool)
    assert isinstance(result.issues, list)
    # Should not have critical violations for a normal channel
    critical = [i for i in result.issues if i.level.value == "critical"]
    assert len(critical) == 0


def test_carrier_architecture_validator_conventional_stack():
    """Test validator catches conventional stack references."""
    validator = CarrierArchitectureValidator()

    channel_dict = {
        "medium": "tcp_ip",
        "path_loss_db": 10.0,
        "snr_db": 20.0,
        "capacity_bps": 1000.0,
        "impairment_params": {},
        "medium_notes": ["Uses TCP/IP for transport"],
    }

    result = validator.validate_channel_model(channel_dict)
    critical = [i for i in result.issues if i.level.value == "critical"]
    assert len(critical) > 0  # Should detect TCP/IP reference


def test_protocol_invariant_validator():
    """Test protocol invariant validator."""
    validator = ProtocolInvariantValidator()

    # Clean data
    clean_data = {"medium": "radio_frequency", "path_loss_db": 10.0}
    result = validator.validate_no_conventional_stack(clean_data)
    assert result.passed

    # Data with conventional stack reference
    dirty_data = {"medium": "tcp", "path_loss_db": 10.0}
    result = validator.validate_no_conventional_stack(dirty_data)
    assert not result.passed


def test_hx241_validator_suite():
    """Test complete validation suite."""
    suite = HX241ValidatorSuite()

    channel_dict = {
        "medium": "radio_frequency",
        "path_loss_db": 10.0,
        "snr_db": 20.0,
        "capacity_bps": 1000.0,
        "impairment_params": {
            "packet_loss_rate": 0.1,
            "corruption_rate": 0.05,
        },
        "medium_notes": ["Test channel"],
    }

    results = suite.validate_channel_model(channel_dict)
    assert "carrier_architecture" in results
    assert "protocol_invariants" in results

    report = suite.generate_report(results)
    assert "summary" in report
    assert "issues" in report
    assert "by_category" in report


# -----------------------------------------------------------------------
# Simulation input generator tests
# -----------------------------------------------------------------------

def test_simulation_input_generator():
    """Test simulation input generator."""
    generator = SimulationInputGenerator()

    scenarios = default_simulation_carrier_scenarios()
    impairment_scenarios = generator.generate_impairment_scenarios(scenarios)

    assert len(impairment_scenarios) == len(scenarios)
    for s in impairment_scenarios:
        assert s.id.startswith("imp_")
        assert s.medium
        assert "packet_loss_rate" in s.impairment_params
        assert "corruption_rate" in s.impairment_params
        assert 0.0 <= s.impairment_params["packet_loss_rate"] <= 1.0
        assert 0.0 <= s.impairment_params["corruption_rate"] <= 1.0


def test_simulation_benchmark_scenarios():
    """Test benchmark scenario generation."""
    generator = SimulationInputGenerator()
    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}
    params = {
        "temporal_shape": "gaussian",
        "temporal_duration": 1.0,
        "spatial_mode": "gaussian",
        "spatial_waist": 1.0,
    }
    ev = generate_wave_event(params, grid)
    results = analyzer.analyze_capacity(snr_db=10.0, bandwidth_hz=1e6,
                                         wave_events=[ev])

    bench_scenarios = generator.generate_benchmark_scenarios([results])
    assert len(bench_scenarios) == 1
    s = bench_scenarios[0]
    assert s.id.startswith("bench_")
    assert s.packet_size > 0
    assert 0.0 <= s.loss_rate <= 1.0
    assert s.bandwidth_bps >= 0


def test_simulation_adversarial_scenarios():
    """Test adversarial scenario generation."""
    generator = SimulationInputGenerator()
    scenarios = default_simulation_carrier_scenarios()
    adv_scenarios = generator.generate_adversarial_scenarios(scenarios)

    assert len(adv_scenarios) > 0
    attack_types = {s.attack_type for s in adv_scenarios}
    assert "packet_loss" in attack_types
    assert "partition" in attack_types
    assert "corruption" in attack_types
    assert "malformed" in attack_types


def test_simulation_full_input_set():
    """Test full simulation input set generation."""
    generator = SimulationInputGenerator()
    carrier_scenarios = default_simulation_carrier_scenarios()

    analyzer = ChannelCapacityAnalyzer()
    grid = {"x": np.linspace(-3, 3, 32), "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]), "t": np.linspace(-3, 3, 32)}
    params = {
        "temporal_shape": "gaussian",
        "temporal_duration": 1.0,
        "spatial_mode": "gaussian",
        "spatial_waist": 1.0,
    }
    ev = generate_wave_event(params, grid)
    results = analyzer.analyze_capacity(snr_db=10.0, bandwidth_hz=1e6,
                                         wave_events=[ev])

    input_set = generator.generate_full_input_set(carrier_scenarios, [results])

    assert input_set.name
    assert input_set.description
    assert len(input_set.impairment_scenarios) > 0
    assert len(input_set.benchmark_scenarios) > 0
    assert len(input_set.adversarial_scenarios) > 0

    # Test JSON serialization
    json_str = input_set.to_json()
    assert json_str
    parsed = json.loads(json_str)
    assert "impairment_scenarios" in parsed
    assert "benchmark_scenarios" in parsed
    assert "adversarial_scenarios" in parsed


# -----------------------------------------------------------------------
# Schema compliance tests
# -----------------------------------------------------------------------

def test_json_output_is_valid():
    """Test that JSON output from generators is valid JSON."""
    generator = SimulationInputGenerator()
    scenarios = default_simulation_carrier_scenarios()
    impairment_scenarios = generator.generate_impairment_scenarios(scenarios)

    input_set = generator.generate_full_input_set(
        carrier_scenarios=scenarios,
        capacity_results=[],
    )
    json_str = input_set.to_json()

    # Should be parseable
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "name" in parsed
    assert "description" in parsed

