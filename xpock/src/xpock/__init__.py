"""XPOCK — Wave-Structure Generator.

XPOCK generates and explores finite physical wave events with independently
controllable physical properties. The goal is to systematically explore wave
structures that could serve as distinguishable representations of information.

The generator is intentionally domain-agnostic: it produces mathematically
valid finite wave fields from first-principles models, not arbitrary visual
patterns presented as physical waves. All candidates are clearly distinguished
from experimentally demonstrated physical states.

Core design principles:
- Physically meaningful mathematical models only
- Laptop-compatible (no GPU, no specialized hardware)
- Extensible: new physical properties can be added as research identifies them
- Combinations of properties, not just isolated variables
- Clear separation between generated candidates and physical reality

Main entry points:
    WaveEvent          — a single generated wave event with full metadata
    WaveField          — the underlying field data (scalar or vector)
    ParameterSpace     — systematic exploration of parameter combinations
    WaveAnalyzer       — measurable property calculation
    WaveVisualizer     — visualizations of generated fields
    Config             — YAML-based configuration
"""

from xpock.core import WaveEvent, WaveField, WaveMode, PolarizationState
from xpock.properties import (
    TemporalProfile,
    SpatialMode,
    PolarizationTexture,
    TopologicalCharge,
    CoherenceModel,
    CoupledMode,
)
from xpock.parameter_space import ParameterSpace, SweepConfig
from xpock.analysis import WaveAnalyzer, MeasurableProperties
from xpock.visualization import WaveVisualizer
from xpock.config import Config
from xpock.validation import ValidationResult, validate_wave_event
from xpock.engines_integration import (
    XpockEngineFactory,
    WavePropagationEngine,
    WaveInformationEngine,
    WaveSpectralEngine,
    NonlinearWaveDynamicsEngine,
    OpticalMaterialsEngine,
    ExperimentalDataEngine,
    EngineResult,
)
from xpock.hx241 import (
    HX241Integration,
    CarrierWaveConfig,
    CarrierChannelModel,
    CarrierMedium,
    CarrierWaveModeler,
    WaveStructureSignature,
    CapacityAnalysisResult,
    DistinguishabilityAnalysisResult,
    DistinguishabilityMetric,
    ChannelCapacityAnalyzer,
    ModulationDimension,
    ModulationScheme,
    ModulationResult,
    DemodulationResult,
    WaveModulator,
    create_default_scheme,
    modulate_frame_payload,
    demodulate_wave_event,
    ImpairmentModel,
    ImpairmentProfile,
    ImpairmentModeler,
    impairment_to_hx_sim_dict,
    create_link_scenario_library,
    AnalysisDomain,
    SpectralFeature,
    ReceivedSignalAnalysis,
    WaveSignalAnalyzer,
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
    "WaveEvent",
    "WaveField",
    "WaveMode",
    "PolarizationState",
    "TemporalProfile",
    "SpatialMode",
    "PolarizationTexture",
    "TopologicalCharge",
    "CoherenceModel",
    "CoupledMode",
    "ParameterSpace",
    "SweepConfig",
    "WaveAnalyzer",
    "MeasurableProperties",
    "WaveVisualizer",
    "Config",
    "ValidationResult",
    "validate_wave_event",
    "XpockEngineFactory",
    "WavePropagationEngine",
    "WaveInformationEngine",
    "WaveSpectralEngine",
    "NonlinearWaveDynamicsEngine",
    "OpticalMaterialsEngine",
    "ExperimentalDataEngine",
    "EngineResult",
    # HX-241 integration
    "HX241Integration",
    "CarrierWaveConfig",
    "CarrierChannelModel",
    "CarrierMedium",
    "CarrierWaveModeler",
    "WaveStructureSignature",
    "CapacityAnalysisResult",
    "DistinguishabilityAnalysisResult",
    "DistinguishabilityMetric",
    "ChannelCapacityAnalyzer",
    "ModulationDimension",
    "ModulationScheme",
    "ModulationResult",
    "DemodulationResult",
    "WaveModulator",
    "create_default_scheme",
    "modulate_frame_payload",
    "demodulate_wave_event",
    "ImpairmentModel",
    "ImpairmentProfile",
    "ImpairmentModeler",
    "impairment_to_hx_sim_dict",
    "create_link_scenario_library",
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