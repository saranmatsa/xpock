"""XPOCK — Wave-Structure Generator.

Finite physical wave events with independently controllable physical properties:
amplitude, phase, frequency, spectral structure, temporal pulse shapes,
polarization (including spatially varying), spatial modes, OAM, phase
singularities, spatiotemporal vortices, nonseparable wave packets, topological
structures (skyrmionic polarization textures, generalized topological
invariants), coherence, coupled modes, and combinations thereof.

All models are mathematically generated candidates. Nothing here is an
experimentally demonstrated physical state.
"""

from .core import WaveConfig, WaveEvent, WaveGenerator, generate_wave, list_properties
from .registry import PropertyRegistry, register_property

__version__ = "0.1.0"
__all__ = [
    "WaveConfig",
    "WaveEvent",
    "WaveGenerator",
    "generate_wave",
    "list_properties",
    "PropertyRegistry",
    "register_property",
]
