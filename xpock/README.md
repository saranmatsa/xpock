# XPOCK — Wave-Structure Generator

## Overview

XPOCK generates and explores **finite physical wave events** with as many
independently controllable physical properties as practical. The goal is to
systematically explore wave structures that could provide useful,
distinguishable representations of information — without assuming any
particular property is the answer.

> **Status:** Initial generator. Mathematical models only. Not a
> communication system, not a protocol, not a network stack.

## What it generates

| Property | Models |
|---|---|
| Amplitude / phase / frequency | Temporal envelopes with carrier + chirp |
| Spectral structure | Temporal & spatial Fourier analysis |
| Temporal pulse shapes | Gaussian, sech, super-Gaussian, rectangular, exponential, bisectorial, custom |
| Spatial modes | Gaussian, Hermite-Gaussian, Laguerre-Gaussian (OAM), Bessel, super-Gaussian, spherical, plane wave, nonseparable |
| Polarization | Linear, circular, elliptical, radial, azimuthal, spatially varying, skyrmionic, custom Jones |
| Topological | Winding numbers, skyrmion numbers, phase singularities |
| Coherence | Full / partial via mutual coherence function |
| Coupled modes | Two-mode coupled-mode theory |
| Combinations | All properties compose multiplicatively |

## Quick start

```bash
# Install
cd C:/XPOCK/xpock
pip install -e .

# Generate a default config
xpock config --output configs/my_config.yaml

# Generate a single wave event
xpock generate --config configs/default.yaml

# Run a parameter sweep (edit configs/default.yaml to add sweeps first)
xpock sweep --config configs/default.yaml --output ./sweep_results

# Analyze a saved event
xpock analyze xpock_output/event_*.json

# Visualize a saved event
xpock visualize xpock_output/event_*.json --output ./figures
```

Or from Python:

```python
from xpock.cli import generate_wave_event
from xpock.analysis import WaveAnalyzer
import numpy as np

x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)
t = np.linspace(-5, 5, 128)
grid = {"x": x, "y": y, "z": np.array([0.0]), "t": t}

params = {
    "temporal_shape": "gaussian",
    "temporal_duration": 1.0,
    "spatial_mode": "laguerre_gaussian",
    "spatial_l": 1,      # OAM
    "polarization_state": "skyrmionic",
    "topology_charge": 1,
}
event = generate_wave_event(params, grid)
props = WaveAnalyzer().analyze(event)
print(props.total_power, props.oam_per_photon)
```

## C++ and Fortran 90 performance kernels

`src/cpp/wave_kernels.{hpp,cpp}` — header-declared C++ kernels for temporal
envelopes, spatial modes, Jones vectors, Stokes parameters, and winding
number estimation. Build with CMake:

```bash
cd src/cpp
cmake -B build -S .
cmake --build build
```

`src/fortran/wave_ops.f90` — Fortran 90 module with Gaussian beam,
Fresnel diffraction, complex array ops, and a self-contained radix-2 FFT:

```bash
cd src/fortran
gfortran -O3 -fPIC -shared -o libxpock_fortran.so wave_ops.f90
```

These are optional accelerators; the Python core works standalone.

## Constraints honored

- Runs locally on an ordinary laptop (NumPy/SciPy/Matplotlib only)
- Physically meaningful mathematical models (no arbitrary visual patterns)
- Clearly distinguishes generated candidates from experimental states
- No assumption that any one wave property is the solution
- No complete network stack / protocol / communications platform
- Extensible: add new properties as research identifies them

## Directory layout

```
xpock/
  src/xpock/        # Python package (core, properties, analysis, ...)
  src/cpp/          # C++ performance kernels
  src/fortran/      # Fortran 90 numerical module
  tests/            # pytest suite
  examples/         # example scripts
  configs/          # YAML configurations
  xpock_output/     # generated events + figures (runtime)
```

## License

MIT — see LICENSE.
