#!/usr/bin/env python3
"""
XPOCK — Experiment 1: Skyrmion Number Conservation Under Free-Space Propagation

Uses power-of-2 constellation [-3,-2,-1,0,1,2,3,4] (3 bits) to avoid
bit-mapping wrapping issues.

Hypothesis: Skyrmion number Q is preserved during propagation through
free space (no turbulence). Q should be identical at input and output.

Falsification: If |Q_out - Q_in| > 0.1 for any N, topological protection
is invalid for this implementation.
"""

import importlib.util
import sys
import numpy as np

spec = importlib.util.spec_from_file_location(
    'skyrmion_modem', '241-integration/wave_info/skyrmion_modem.py'
)
modem_mod = importlib.util.module_from_spec(spec)
sys.modules['skyrmion_modem'] = modem_mod
spec.loader.exec_module(modem_mod)
SkyrmionModem = modem_mod.SkyrmionModem
ModemConfig = modem_mod.ModemConfig

spec2 = importlib.util.spec_from_file_location(
    'turbulence', '241-integration/wave_info/turbulence.py'
)
turbulence_mod = importlib.util.module_from_spec(spec2)
sys.modules['turbulence'] = turbulence_mod
spec2.loader.exec_module(turbulence_mod)
TurbulenceModel = turbulence_mod.TurbulenceModel
TurbulenceConfig = turbulence_mod.TurbulenceConfig

nx, ny = 64, 64
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y, indexing='xy')

# Power-of-2 constellation: 8 states, 3 bits
constellation = [-3, -2, -1, 0, 1, 2, 3, 4]

cfg = ModemConfig(
    wavelength=1.55e-6,
    grid_x=X,
    grid_y=Y,
    skyrmion_radius=1.0,
    constellation=constellation,
)
modem = SkyrmionModem(cfg)

print("=" * 70)
print("EXPERIMENT 1: Skyrmion Number Conservation (Free Space)")
print("=" * 70)
print(f"Constellation: {constellation} ({modem.n_bits} bits/symbol)")
print(f"Propagation: 100 m free space, Cn² = 0 (no turbulence)")
print()

turb_cfg = TurbulenceConfig(
    wavelength=1.55e-6,
    distance=100.0,
    cn2=0.0,
    n_screens=10,
    seed=42,
)
turb_model = TurbulenceModel(turb_cfg)

all_passed = True
for n_tx in constellation:
    bit_idx = modem.constellation.index(n_tx)
    bits_tx = np.array([(bit_idx >> i) & 1 for i in range(modem.n_bits - 1, -1, -1)])

    Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
    q_in, _ = modem.demodulate(Ex_tx, Ey_tx)

    Ex_out = turb_model.propagate(Ex_tx, X, Y)
    Ey_out = turb_model.propagate(Ey_tx, X, Y)
    q_out, _ = modem.demodulate(Ex_out, Ey_out)

    q_diff = abs(q_out - q_in)
    passed = q_diff < 0.1
    if not passed:
        all_passed = False

    status = "PASS" if passed else "FAIL"
    print(f"  N={n_tx:>2}: Q_in={q_in:+.6f}, Q_out={q_out:+.6f}, "
          f"|ΔQ|={q_diff:.6f}  [{status}]")

print()
if all_passed:
    print("RESULT: PASS — Skyrmion number conserved under free-space propagation")
else:
    print("RESULT: FAIL — Topological protection claim is INVALID")
print("=" * 70)