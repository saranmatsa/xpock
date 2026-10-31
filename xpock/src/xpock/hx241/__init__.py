"""
XPOCK — HX-241 Integration Package

Integrates XPOCK wave-structure generation, propagation, spectral analysis,
and information theory with HX-241's carrier-agnostic communication
architecture.

Integration boundary: XPOCK models the physical carrier layer (wave
propagation, channel capacity, modulation, impairment, receiver analysis).
HX-241 protocol logic remains untouched and carrier-agnostic.

Architectural compliance:
- No IP/TCP/UDP/QUIC/DNS/HTTP dependencies
- HX-241 protocol logic is not modified
- Carriers provide physical capabilities, not protocol definitions
- ENGINES remains an independent, reusable asset
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pathlib import Path

# Carrier wave modeling
from xpock.hx241.carrier import (
    CarrierWaveConfig,
    CarrierChannelModel,
    CarrierMedium,
    CarrierWaveModeler,
)

# Channel capacity & information analysis
from xpock.hx241.channel import (
    WaveStructureSignature,
    CapacityAnalysisResult,
    DistinguishabilityAnalysisResult,
    DistinguishabilityMetric,
    ChannelCapacityAnalyzer,
)

# Wave-structure modulation
from xpock.hx241.modulation import (
    ModulationDimension,
    ModulationScheme,
    ModulationResult,
    DemodulationResult,
    WaveModulator,
    create_default_scheme,
    modulate_frame_payload,
    demodulate_wave_event,
)

# Physically-based impairment models
from xpock.hx241.impairment import (
    ImpairmentModel,
    ImpairmentProfile,
    ImpairmentModeler,
    impairment_to_hx_sim_dict,
    create_link_scenario_library,
)

# Receiver-side spectral analysis
from xpock.hx241.spectral import (
    AnalysisDomain,
    SpectralFeature,
    ReceivedSignalAnalysis,
    WaveSignalAnalyzer,
)

# Test vector generation for HX-241 Rust consumption
from xpock.hx241.test_vectors import (
    TestVectorGenerator,
    TestVectorSet,
    default_carrier_scenarios,
    default_impairment_scenarios,
    default_snr_values,
    default_bandwidth_values,
    default_scheme_configs,
    generate_and_save_test_vectors,
)

__version__ = "0.1.0"

__all__ = [
    # Carrier
    "CarrierWaveConfig",
    "CarrierChannelModel",
    "CarrierMedium",
    "CarrierWaveModeler",
    # Channel capacity
    "WaveStructureSignature",
    "CapacityAnalysisResult",
    "DistinguishabilityAnalysisResult",
    "DistinguishabilityMetric",
    "ChannelCapacityAnalyzer",
    # Modulation
    "ModulationDimension",
    "ModulationScheme",
    "ModulationResult",
    "DemodulationResult",
    "WaveModulator",
    "create_default_scheme",
    "modulate_frame_payload",
    "demodulate_wave_event",
    # Impairment
    "ImpairmentModel",
    "ImpairmentProfile",
    "ImpairmentModeler",
    "impairment_to_hx_sim_dict",
    "create_link_scenario_library",
    # Spectral
    "AnalysisDomain",
    "SpectralFeature",
    "ReceivedSignalAnalysis",
    "WaveSignalAnalyzer",
    # Test vectors
    "TestVectorGenerator",
    "TestVectorSet",
    "default_carrier_scenarios",
    "default_impairment_scenarios",
    "default_snr_values",
    "default_bandwidth_values",
    "default_scheme_configs",
    "generate_and_save_test_vectors",
    "__version__",
]


# Convenience factory
class HX241Integration:
    """
    Factory for creating all XPOCK-HX-241 integration components.

    Provides a unified interface for:
    1. Carrier wave modeling (physical layer simulation)
    2. Channel capacity analysis (information-theoretic limits)
    3. Wave-structure modulation (frame-to-wave mapping)
    4. Impairment modeling (hx-sim compatible parameters)
    5. Receiver-side spectral analysis (wave property extraction)
    6. Test vector generation (Rust-compatible output for hx-sim)
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./xpock_hx241")
        self.workspace.mkdir(parents=True, exist_ok=True)

    def carrier_modeler(self) -> CarrierWaveModeler:
        return CarrierWaveModeler(workspace=str(self.workspace / "carrier"))

    def capacity_analyzer(self) -> ChannelCapacityAnalyzer:
        return ChannelCapacityAnalyzer(workspace=str(self.workspace / "capacity"))

    def modulator(self, scheme=None) -> WaveModulator:
        return WaveModulator(scheme=scheme, grid_arrays={})

    def impairment_modeler(self) -> ImpairmentModeler:
        return ImpairmentModeler(workspace=str(self.workspace / "impairment"))

    def signal_analyzer(self) -> WaveSignalAnalyzer:
        return WaveSignalAnalyzer(workspace=str(self.workspace / "spectral"))

    def test_vector_generator(self) -> TestVectorGenerator:
        return TestVectorGenerator(workspace=str(self.workspace / "vectors"))

    def all_components(self) -> Dict[str, Any]:
        return {
            "carrier_modeler": self.carrier_modeler(),
            "capacity_analyzer": self.capacity_analyzer(),
            "modulator": self.modulator(),
            "impairment_modeler": self.impairment_modeler(),
            "signal_analyzer": self.signal_analyzer(),
            "test_vector_generator": self.test_vector_generator(),
        }