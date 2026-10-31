"""
XPOCK — wave_info package

Modules for simulating wave-structure communication through turbulent
atmosphere:

- turbulence.py   — Kolmogorov/von Karman phase screen propagation model
- skyrmion_modem.py — Skyrmion number modulation/demodulation modem
- ber_simulation.py — End-to-end BER simulation pipeline
- comparison.py   — Comparison against OAM and intensity modulation baselines

All simulations are mathematical models on a laptop. Results do not
establish physical feasibility — hardware would be required.

XPOCK research only — no HX-241 involvement.
"""

from xpock.wave_info.turbulence import (
    TurbulenceConfig,
    TurbulenceModel,
    TurbulenceResult,
)
from xpock.wave_info.skyrmion_modem import (
    ModemConfig,
    ModemResult,
    SkyrmionModem,
)

__all__ = [
    "TurbulenceConfig",
    "TurbulenceModel",
    "TurbulenceResult",
    "ModemConfig",
    "ModemResult",
    "SkyrmionModem",
]