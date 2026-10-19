"""
XPOCK — HX-241 Simulation Inputs for hx-sim

Generates simulation inputs for HX-241's hx-sim crate:
- NetworkImpairment configurations for ImpairmentEngine
- BenchmarkConfig for BenchmarkHarness
- Adversarial test scenarios for AdversarialTestHarness
- Network topology definitions for simulation

All outputs are language-agnostic JSON that can be deserialized by Rust's serde
into hx-sim's native types. XPOCK/Python is NOT a runtime dependency.

hx-sim integration points:
- hx-sim/src/impairment.rs: ImpairmentEngine, NetworkImpairment
- hx-sim/src/benchmark.rs: BenchmarkHarness, BenchmarkConfig, BenchmarkResults
- hx-sim/src/adversarial.rs: AdversarialTestHarness
- hx-sim/src/link_degradation.rs: Link degradation detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np

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
    create_link_scenario_library,
)
from xpock.hx241.channel import ChannelCapacityAnalyzer, CapacityAnalysisResult


@dataclass
class ImpairmentScenario:
    """A single impairment scenario for hx-sim testing."""

    id: str
    name: str
    description: str
    medium: str
    impairment_params: Dict[str, Any]
    duration_us: int
    enabled: bool = True


@dataclass
class BenchmarkScenario:
    """A single benchmark scenario for hx-sim's BenchmarkHarness."""

    id: str
    name: str
    workload: str
    packet_size: int
    stream_count: int
    total_bytes: int
    latency_us: int
    loss_rate: float
    bandwidth_bps: int
    topology_nodes: int
    duration_secs: int


@dataclass
class AdversarialScenario:
    """A single adversarial test scenario."""

    id: str
    name: str
    attack_type: str  # packet_loss, partition, corruption, malicious_routing, malformed
    target: str
    parameters: Dict[str, Any]
    expected_behavior: str


@dataclass
class SimulationInputSet:
    """Complete simulation input set for hx-sim."""

    name: str
    description: str
    impairment_scenarios: List[ImpairmentScenario] = field(default_factory=list)
    benchmark_scenarios: List[BenchmarkScenario] = field(default_factory=list)
    adversarial_scenarios: List[AdversarialScenario] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for Rust serde consumption."""
        return json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "impairment_scenarios": [self._impairment_to_dict(s) for s in self.impairment_scenarios],
                "benchmark_scenarios": [self._benchmark_to_dict(s) for s in self.benchmark_scenarios],
                "adversarial_scenarios": [self._adversarial_to_dict(s) for s in self.adversarial_scenarios],
                "metadata": self.metadata,
            },
            indent=2,
            default=str,
        )

    @staticmethod
    def _impairment_to_dict(s: ImpairmentScenario) -> Dict[str, Any]:
        return {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "medium": s.medium,
            "impairment_params": s.impairment_params,
            "duration_us": s.duration_us,
            "enabled": s.enabled,
        }

    @staticmethod
    def _benchmark_to_dict(s: BenchmarkScenario) -> Dict[str, Any]:
        return {
            "id": s.id,
            "name": s.name,
            "workload": s.workload,
            "packet_size": s.packet_size,
            "stream_count": s.stream_count,
            "total_bytes": s.total_bytes,
            "latency_us": s.latency_us,
            "loss_rate": s.loss_rate,
            "bandwidth_bps": s.bandwidth_bps,
            "topology_nodes": s.topology_nodes,
            "duration_secs": s.duration_secs,
        }

    @staticmethod
    def _adversarial_to_dict(s: AdversarialScenario) -> Dict[str, Any]:
        return {
            "id": s.id,
            "name": s.name,
            "attack_type": s.attack_type,
            "target": s.target,
            "parameters": s.parameters,
            "expected_behavior": s.expected_behavior,
        }


class SimulationInputGenerator:
    """
    Generates hx-sim compatible simulation inputs from XPOCK models.

    Produces:
    1. Impairment scenarios from carrier channel models
    2. Benchmark scenarios from channel capacity analysis
    3. Adversarial scenarios from threat model analysis
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./241-integration/simulation")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._impairment_modeler: Optional[ImpairmentModeler] = None
        self._capacity_analyzer: Optional[ChannelCapacityAnalyzer] = None

    def _get_impairment_modeler(self) -> ImpairmentModeler:
        if self._impairment_modeler is None:
            self._impairment_modeler = ImpairmentModeler(
                workspace=str(self.workspace / "impairment")
            )
        return self._impairment_modeler

    def _get_capacity_analyzer(self) -> ChannelCapacityAnalyzer:
        if self._capacity_analyzer is None:
            self._capacity_analyzer = ChannelCapacityAnalyzer(
                workspace=str(self.workspace / "capacity")
            )
        return self._capacity_analyzer

    # ------------------------------------------------------------------
    # Impairment scenarios
    # ----------------------------------------------------------------------

    def generate_impairment_scenarios(
        self,
        carrier_scenarios: List[Dict[str, Any]],
    ) -> List[ImpairmentScenario]:
        """Generate impairment scenarios from carrier model scenarios."""
        modeler = self._get_impairment_modeler()
        scenarios: List[ImpairmentScenario] = []

        for i, scenario in enumerate(carrier_scenarios):
            profile = modeler.generate_impairment_scenario(
                medium=CarrierMedium(scenario["medium"]),
                distance_m=scenario["distance_m"],
                frequency_hz=scenario["frequency_hz"],
                transmit_power_w=scenario.get("transmit_power_w", 1.0),
                **scenario.get("medium_specific", {}),
            )

            hx_sim_params = impairment_to_hx_sim_dict(profile)

            # Duration based on distance and propagation speed
            duration_us = int(scenario["distance_m"] / 299792458.0 * 1e6 * 1000)  # 1000x round trip

            scenarios.append(ImpairmentScenario(
                id=f"imp_{i:04d}",
                name=f"{scenario['medium']}_{scenario['distance_m']}m_{scenario['frequency_hz']}Hz",
                description=f"{profile.medium} link: {profile.path_loss_db:.1f} dB path loss, {profile.snr_db:.1f} dB SNR",
                medium=profile.medium,
                impairment_params=hx_sim_params,
                duration_us=duration_us,
            ))

        return scenarios

    # ------------------------------------------------------------------
    # Benchmark scenarios
    # ----------------------------------------------------------------------

    def generate_benchmark_scenarios(
        self,
        capacity_results: List[CapacityAnalysisResult],
    ) -> List[BenchmarkScenario]:
        """Generate benchmark scenarios from capacity analysis results."""
        scenarios: List[BenchmarkScenario] = []

        for i, result in enumerate(capacity_results):
            scenarios.append(BenchmarkScenario(
                id=f"bench_{i:04d}",
                name=f"SNR={result.snr_db:.0f}dB_BW={result.bandwidth_hz/1e3:.0f}kHz",
                workload="BulkTransfer",
                packet_size=max(64, int(result.spectral_efficiency_bps_hz * 100)),
                stream_count=1,
                total_bytes=int(result.usable_capacity_bps * 10),
                latency_us=int(1e6 / max(1, result.symbols_per_second)),
                loss_rate=float(1.0 - min(1.0, result.distinguishable_states / 256.0)),
                bandwidth_bps=int(result.shannon_capacity_bps),
                topology_nodes=2,
                duration_secs=30,
            ))

        return scenarios

    # ------------------------------------------------------------------
    # Adversarial scenarios
    # ----------------------------------------------------------------------

    def generate_adversarial_scenarios(
        self,
        carrier_scenarios: List[Dict[str, Any]],
    ) -> List[AdversarialScenario]:
        """Generate adversarial test scenarios based on threat model."""
        scenarios: List[AdversarialScenario] = []

        for i, scenario in enumerate(carrier_scenarios):
            medium = scenario["medium"]
            distance = scenario["distance_m"]

            # Packet loss attack (DoS)
            scenarios.append(AdversarialScenario(
                id=f"adv_loss_{i:04d}",
                name=f"Packet loss DoS on {medium} ({distance}m)",
                attack_type="packet_loss",
                target="all_nodes",
                parameters={
                    "loss_rate": 0.5,
                    "duration_us": 5000000,  # 5 seconds
                },
                expected_behavior="Session survives; retransmission recovers data",
            ))

            # Partition attack (eclipse)
            scenarios.append(AdversarialScenario(
                id=f"adv_part_{i:04d}",
                name=f"Network partition on {medium} ({distance}m)",
                attack_type="partition",
                target="victim_node",
                parameters={
                    "partition_type": "Clean",
                    "duration_us": 10000000,  # 10 seconds
                },
                expected_behavior="Session migrates to alternate carrier; state preserved",
            ))

            # Corruption attack (Byzantine)
            scenarios.append(AdversarialScenario(
                id=f"adv_corr_{i:04d}",
                name=f"Data corruption on {medium} ({distance}m)",
                attack_type="corruption",
                target="all_nodes",
                parameters={
                    "corruption_rate": 0.1,
                },
                expected_behavior="CRC32 detects corruption; corrupted frames discarded",
            ))

            # Malformed packet injection
            scenarios.append(AdversarialScenario(
                id=f"adv_malf_{i:04d}",
                name=f"Malformed packet injection on {medium} ({distance}m)",
                attack_type="malformed",
                target="all_nodes",
                parameters={
                    "malformed_rate": 0.05,
                },
                expected_behavior="Malformed frames ignored; session continues",
            ))

        return scenarios

    # ------------------------------------------------------------------
    # Complete simulation input set
    # ----------------------------------------------------------------------

    def generate_full_input_set(
        self,
        carrier_scenarios: List[Dict[str, Any]],
        capacity_results: List[CapacityAnalysisResult],
    ) -> SimulationInputSet:
        """Generate a complete simulation input set."""
        impairment_scenarios = self.generate_impairment_scenarios(carrier_scenarios)
        benchmark_scenarios = self.generate_benchmark_scenarios(capacity_results)
        adversarial_scenarios = self.generate_adversarial_scenarios(carrier_scenarios)

        return SimulationInputSet(
            name="xpock_hx241_simulation_inputs",
            description=(
                "Complete simulation input set for HX-241's hx-sim crate. "
                "Generated by XPOCK's physically-based carrier models and "
                "channel capacity analysis."
            ),
            impairment_scenarios=impairment_scenarios,
            benchmark_scenarios=benchmark_scenarios,
            adversarial_scenarios=adversarial_scenarios,
            metadata={
                "n_impairment_scenarios": len(impairment_scenarios),
                "n_benchmark_scenarios": len(benchmark_scenarios),
                "n_adversarial_scenarios": len(adversarial_scenarios),
                "generated_by": "xpock.hx241.simulation",
            },
        )

    # ------------------------------------------------------------------
    # File output
    # ----------------------------------------------------------------------

    def save_json(
        self,
        input_set: SimulationInputSet,
        filename: str = "hx241_simulation_inputs.json",
    ) -> Path:
        """Save simulation inputs as JSON file."""
        path = self.workspace / filename
        path.write_text(input_set.to_json(), encoding="utf-8")
        return path

    def save_all(
        self,
        input_set: SimulationInputSet,
        json_filename: str = "hx241_simulation_inputs.json",
    ) -> Path:
        """Save simulation inputs as JSON."""
        return self.save_json(input_set, json_filename)


# ------------------------------------------------------------------
# Default scenario definitions for simulation
# ------------------------------------------------------------------

def default_simulation_carrier_scenarios() -> List[Dict[str, Any]]:
    """Default carrier scenarios for simulation input generation."""
    return [
        {
            "medium": "radio_frequency",
            "frequency_hz": 2.4e9,
            "distance_m": 100.0,
            "transmit_power_w": 0.1,
            "medium_specific": {
                "rms_delay_spread_us": 0.5,
                "shadow_std_db": 6.0,
                "doppler_spread_hz": 10.0,
            },
        },
        {
            "medium": "free_space_optical",
            "frequency_hz": 200e12,
            "distance_m": 10.0,
            "transmit_power_w": 0.001,
            "medium_specific": {
                "attenuation_db_per_km": 0.2,
                "cn2": 1e-14,
            },
        },
        {
            "medium": "acoustic",
            "frequency_hz": 20e3,
            "distance_m": 500.0,
            "transmit_power_w": 1.0,
            "medium_specific": {
                "sound_speed_m_s": 1500.0,
                "absorption_db_per_km": 0.5,
                "spreading": "cylindrical",
                "doppler_spread_hz": 1.0,
            },
        },
        {
            "medium": "optical_fiber",
            "frequency_hz": 193e12,
            "distance_m": 5000.0,
            "transmit_power_w": 0.001,
            "medium_specific": {
                "attenuation_db_per_km": 0.2,
                "dispersion_ps_nm_km": 17.0,
                "source_linewidth_nm": 0.1,
            },
        },
    ]


def generate_and_save_simulation_inputs(
    workspace: Optional[str] = None,
    capacity_results: Optional[List[CapacityAnalysisResult]] = None,
) -> Path:
    """
    Convenience: generate and save simulation inputs for hx-sim.

    Parameters
    ----------
    workspace : str, optional
        Output directory for simulation input files.
    capacity_results : list, optional
        Pre-computed capacity analysis results. If not provided,
        default results are generated.

    Returns
    -------
    Path to the saved JSON file.
    """
    from xpock.hx241.channel import ChannelCapacityAnalyzer

    if capacity_results is None:
        analyzer = ChannelCapacityAnalyzer()
        # Generate sample wave events for capacity analysis
        from xpock.cli import generate_wave_event
        import numpy as np

        grid = {
            "x": np.linspace(-3, 3, 32),
            "y": np.linspace(-3, 3, 32),
            "z": np.array([0.0]),
            "t": np.linspace(-3, 3, 32),
        }

        wave_events = []
        for params in [
            {
                "temporal_shape": "gaussian",
                "temporal_duration": 1.0,
                "spatial_mode": "gaussian",
                "spatial_waist": 1.0,
                "polarization_state": "linear",
            },
            {
                "temporal_shape": "sech",
                "temporal_duration": 1.0,
                "spatial_mode": "laguerre_gaussian",
                "spatial_l": 1,
                "spatial_waist": 1.0,
                "polarization_state": "circular",
            },
        ]:
            ev = generate_wave_event(params, grid)
            wave_events.append(ev)

        capacity_results = []
        for snr in [-20.0, 0.0, 20.0]:
            for bw in [1e4, 1e6]:
                result = analyzer.analyze_capacity(
                    snr_db=snr,
                    bandwidth_hz=bw,
                    wave_events=wave_events,
                    error_correction_overhead=0.25,
                )
                capacity_results.append(result)

    generator = SimulationInputGenerator(workspace=workspace)
    input_set = generator.generate_full_input_set(
        carrier_scenarios=default_simulation_carrier_scenarios(),
        capacity_results=capacity_results,
    )

    return generator.save_all(input_set)
