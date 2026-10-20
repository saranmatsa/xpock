# IMPLEMENTATION_PLAN.md — XPOCK Wave-Structure Information Communication

**Date:** 2026-10-19
**Project:** XPOCK (independent of HX-241)
**Status:** Ready for implementation
**Strongest candidate:** Optical Skyrmion Number Modulation (SkM)

---

## 1. Summary

This plan describes how to test whether optical skyrmion number modulation can serve as a physically realizable information-bearing wave structure. The mechanism: encode information as the skyrmion number N (a topological integer invariant) of an optical polarization texture; transmit through free space; recover at receiver via Stokes polarimetry + skyrmion number computation.

**Key claim:** Skyrmion number is a topological invariant preserved during propagation, making it inherently robust against atmospheric turbulence-induced wavefront distortion — a primary failure mode of conventional free-space optical communication.

**Scope boundary:** This plan is for XPOCK research only. No HX-241 involvement. All simulation is laptop-based (NumPy/SciPy/Matplotlib). No claims of physical feasibility from simulation alone.

---

## 2. Physical Mechanism

### 2.1 What is an Optical Skyrmion?

An optical skyrmion is a spatially varying polarization field where the normalized Stokes vector n(x,y) = (S1,S2,S3)/S0 wraps the Poincaré sphere. The skyrmion number Q is:

Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy

Q is integer-valued and topologically protected: any continuous deformation of the field that doesn't create/destroy singularities preserves Q.

**Information encoding:** Select N from constellation C = {n₁, n₂, ..., n_M} where M = 2^K. Each activation conveys K bits (index modulation). Example constellation: C = {-2, -1, 0, 1, 2} → M=5, K≈2.32 bits (use floor(2)=2 bits with 4 used states, or 3 bits with 8 states).

**Transmitter:** Generate skyrmionic beam |Ψ_Nsk⟩ by interfering two orthogonal Laguerre-Gaussian modes with different OAM charges l₁, l₂ and orthogonal polarizations:

|Ψ⟩ = |l₁⟩|H⟩ + |l₂⟩|V⟩

The skyrmion number is Q = l₁ - l₂ (for this construction).

**Receiver:** Measure full Stokes field S(x,y) via polarimetry (6 intensity projections: H,V,D,A,R,L). Compute:

n(x,y) = (S1,S2,S3)/S0
Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy

Hard-decide: ˆN = argminₙ |Q − n| over constellation C.

### 2.2 Why This Could Work

- **Topological protection:** Q cannot change under continuous deformation. Atmospheric turbulence distorts the field but (in principle) doesn't change Q unless singularities are created/destroyed.
- **Experimental precedent:** Wang et al. (2026) demonstrated skyrmion number preservation through 270 m free-space turbulence link. Hara et al. (2026) proposed SkM and analyzed performance under turbulence.
- **Direct measurement:** Skyrmion number is computable from Stokes polarimetry — a mature, well-understood measurement technique.

### 2.3 Why It Might Not Work

- **Receiver complexity:** Full Stokes polarimetry requires 6 intensity measurements at every spatial point. Real-time decoding at high bit rates is unproven.
- **Information density:** Single skyrmion number per field → log₂(M) bits per field. Skyrmion bags can encode multiple numbers per field but increase measurement complexity.
- **Turbulence threshold:** Maximum Cₙ² where Q remains distinguishable is uncharacterized.
- **Finite aperture effects:** The skyrmion number integral is over infinite transverse plane; finite receiver aperture truncates the integral, introducing error.

---

## 3. Laptop-Only Simulation Strategy

XPOCK already has the core wave generator. The missing pieces are: (1) turbulent propagation model, (2) receiver-side skyrmion number measurement, (3) end-to-end BER simulation. All can be simulated on a laptop.

### 3.1 Simulation Pipeline

```
generate skyrmionic field (XPOCK properties)
    ↓
propagate through turbulent atmosphere (custom phase screen model)
    ↓
add measurement noise (shot noise, detector noise)
    ↓
measure Stokes field (simulated polarimetry: 6 projections)
    ↓
compute skyrmion number (XPOCK TopologicalCharge.compute_skyrmion_number)
    ↓
hard-decide skyrmion number → recovered bits
    ↓
compute BER vs transmitted bits
```

### 3.2 Implementation Steps

#### Step 3.1: Skyrmion Field Generator (XPOCK)

Use existing `PolarizationTexture` with `state=SKYRMIONIC` and configurable `skyrmion_charge`. Generate events for each constellation value N ∈ {-2,-1,0,1,2} (5 states → 2 bits per field with one unused, or use {-3,-2,-1,0,1,2,3,4} → 3 bits).

```python
from xpock.properties import PolarizationConfig, PolarizationState, PolarizationTexture
import numpy as np

x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)
X, Y = np.meshgrid(x, y, indexing='xy')

def generate_skyrmion_field(charge N, radius R=1.0):
    cfg = PolarizationConfig(
        state=PolarizationState.SKYRMIONIC,
        skyrmion_radius=R,
        skyrmion_charge=N
    )
    pol = PolarizationTexture(cfg)
    Ex, Ey = pol.evaluate(X, Y)
    return Ex, Ey
```

#### Step 3.2: Turbulent Propagation Model

Implement atmospheric turbulence using the Kolmogorov model. The refractive index structure constant Cₙ² determines turbulence strength. Generate phase screens using the von Kármán spectrum:

Φₙ(κ) = 0.033 Cₙ² κ^(-11/3) exp(-κ²/κₘ²)

where κₘ = 5.92/l₀ (inner scale l₀ ≈ 5-10 mm).

Phase screen generation:
1. Create random phase screen in Fourier space with spectrum Φₙ(κ)
2. Inverse FFT to get spatial phase screen φ(x,y)
2. Apply phase screen to field: E_turb(x,y) = E(x,y) · exp(i·φ(x,y))
3. For strong turbulence, use multiple phase screens along propagation path

Implementation (pure NumPy/SciPy):
```python
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftfreq

def generate_phase_screen(shape, dx, Cn2, wavelength, L, n_screens=1):
    """Generate Kolmogorov turbulence phase screen(s).
    
    Returns phase screen(s) in radians.
    """
    ny, nx = shape
    kx = 2*np.pi*fftfreq(nx, dx)
    ky = 2*np.pi*fftfreq(ny, dx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Inner and outer scale cutoffs
    l0 = 5e-3  # 5 mm inner scale
    L0 = 10.0  # 10 m outer scale
    km = 5.92 / l0
    k0 = 2*np.pi / L0
    
    # Von Karman spectrum
    Phi = 0.033 * Cn2 * K**(-11/3) * np.exp(-K**2/km**2)
    Phi[K < k0] = 0  # Remove energy below outer scale
    
    # Generate random phase screen
    rng = np.random.default_rng()
    phase_screen = rng.standard_normal((ny, nx))
    phase_screen = fft2(phase_screen)
    phase_screen = phase_screen * np.sqrt(Phi * dx * dy)
    phase_screen = np.real(ifft2(phase_screen))
    
    # Convert OPD (meters) to phase (radians): phi = 2*pi*OPD/lambda
    phase_screen = 2*np.pi * phase_screen / wavelength
    
    return phase_screen

def propagate_through_turbulence(E, x, y, Cn2, wavelength, L, z_steps=10):
    """Propagate field through turbulent atmosphere.
    
    Uses split-step: free-space propagation + phase screen at each step.
    """
    dz = L / z_steps
    k = 2*np.pi / wavelength
    
    # Free-space propagation kernel (Fresnel)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ny, nx = x.shape
    kx = 2*np.pi*fftfreq(nx, dx)
    ky = 2*np.pi*fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    H = np.exp(-1j * K2 * dz / (2*k))
    
    E_out = E.copy()
    for _ in range(z_steps):
        # Propagate dz/2
        E_hat = fft2(E_out)
        E_hat = E_hat * H
        E_out = ifft2(E_hat)
        
        # Apply phase screen
        phi = generate_phase_screen(E_out.shape, dx, Cn2, wavelength, dz)
        E_out = E_out * np.exp(1j * phi)
        
        # Propagate dz/2
        E_hat = fft2(E_out)
        E_hat = E_hat * H
        E_out = ifft2(E_hat)
    
    return E_out
```

#### Step 3.3: Receiver-Side Skyrmion Number Measurement

Simulate Stokes polarimetry: measure 6 intensity projections (H, V, D, A, R, L) at each spatial point, compute Stokes parameters, then skyrmion number.

```python
from xpock.properties import TopologicalCharge, TopologicalChargeConfig

def measure_skyrmion_number(Ex_turb, Ey_turb, x, y):
    """Measure skyrmion number from turbulent field via Stokes polarimetry.
    
    Simulates: measure 6 Stokes projections, compute S0,S1,S2,S3,
    compute skyrmion number Q = (1/4pi) ∫ n · (∂x n × ∂y n) dx dy
    """
    # Simulate Stokes measurement from field
    # S0 = |Ex|^2 + |Ey|^2
    # S1 = |Ex|^2 - |Ey|^2
    # S2 = 2*Re(Ex*conj(Ey))
    # S3 = 2*Im(Ex*conj(Ey))
    
    S0 = np.abs(Ex_turb)**2 + np.abs(Ey_turb)**2
    S1 = np.abs(Ex_turb)**2 - np.abs(Ey_turb)**2
    S2 = 2.0*np.real(Ex_turb*np.conj(Ey_turb))
    S3 = 2.0*np.imag(Ex_turb*np.conj(Ey_turb))
    
    # Add measurement noise (shot noise ~ sqrt(I))
    rng = np.random.default_rng(42)
    S0_noisy = S0 + rng.normal(0, np.sqrt(np.maximum(S0, 0))*0.01, S0.shape)
    S1_noisy = S1 + rng.normal(0, np.sqrt(np.maximum(S0, 0))*0.01, S1.shape)
    S2_noisy = S2 + rng.normal(0, np.sqrt(np.maximum(S0, 0))*0.01, S2.shape)
    S3_noisy = S3 + rng.normal(0, np.sqrt(np.maximum(S0, 0))*0.01, S3.shape)
    
    # Compute skyrmion number using XPOCK's method
    tc = TopologicalCharge(TopologicalChargeConfig(charge_type='skyrmion', charge=1))
    Q_measured = tc.compute_skyrmion_number(Ex_turb, Ey_turb)
    
    return Q_measured
```

#### Step 3.4: End-to-End BER Simulation

For each constellation value N ∈ {-3,-2,-1,0,1,2,3,4} (8 states → 3 bits):
1. Generate skyrmion field with charge N
2. Propagate through turbulence (Cₙ² ∈ {1e-15, 1e-14, 1e-13, 5e-13, 1e-12} m^(-2/3))
3. Measure skyrmion number at receiver
4. Hard-decide: ˆN = nearest constellation value
5. Compute bit error rate over N_trials = 100 trials per (N, Cₙ²) pair

Compare against:
- Ideal (no turbulence): BER should be 0
- Conventional OAM mode multiplexing (same number of modes, no topological protection)
- Standard Gaussian beam with intensity modulation (baseline)

### 3.3 Simulation Outputs

For each (skyrmion_number, Cₙ²) pair:
- Mean measured skyrmion number ˆQ
- Standard deviation of ˆQ
- BER (bit error rate)
- Confusion matrix (transmitted N vs received ˆN)
- Minimum distinguishable skyrmion number spacing ΔN_min at each Cₙ²

---

## 4. Quantitative Comparisons

### 4.1 Skyrmion Number Modulation vs Conventional Modulation

| Metric | Skyrmion Number Modulation | OAM Multiplexing | Gaussian Intensity Modulation |
|--------|---------------------------|------------------|------------------------------|
| Information carrier | Topological integer invariant | Spatial mode orthogonality | Intensity amplitude |
| Robustness to turbulence | High (topological protection) | Low (mode coupling/divergence) | Low (scintillation fading) |
| Receiver measurement | Stokes polarimetry + integral | OAM mode demultiplexing | Photodiode |
| Bits per field | log₂(M) (M = distinguishable N values) | log₂(M) (M = OAM modes) | log₂(M) (M = intensity levels) |
| Demonstrated range | 270 m (skyrmion number preserved) | 100-1000 m (OAM multiplexing) | km (standard FSO) |
| Channel capacity vs Shannon | Not yet measured | Does not exceed (Topić 2015) | Standard (Shannon limit applies) |

### 4.2 Expected Simulation Results (Hypotheses to Test)

**Hypothesis 1:** Skyrmion number BER < OAM mode BER at equivalent turbulence (Cₙ² > 1e-14 m^(-2/3))

**Hypothesis 2:** Skyrmion number remains integer-valued (within measurement error) for Cₙ² ≤ 1e-13 m^(-2/3), but degrades rapidly above

**Hypothesis 3:** Skyrmion bag (multiple skyrmions per field) achieves higher bits/field than single skyrmion, but with higher measurement complexity

### 4.3 Success Criteria

The mechanism is **viable** if:
- BER < 10⁻³ at Cₙ² ≤ 1e-13 m^(-2/3) for 8-state constellation (3 bits/field)
- Skyrmion number measurement error < 0.5 (i.e., nearest integer correctly identified >99% of time)
- Information density ≥ 1 bit/field (2 distinguishable skyrmion numbers)

The mechanism is **not viable** if:
- BER > 0.1 at Cₙ² ≤ 1e-14 m^(-2/3)
- Skyrmion number cannot be distinguished from neighbors at any practical turbulence level
- Information density < 0.5 bits/field

---

## 5. Falsification Experiments

### Experiment 1: Skyrmion Number Conservation Under Propagation

**Hypothesis:** Skyrmion number Q is preserved during propagation through free space (no turbulence).

**Test:** Generate skyrmionic field with charge N ∈ {-3,-2,-1,0,1,2,3}. Propagate through 100 m free space (no turbulence). Compute Q at input and output. Q should be identical.

**Falsification:** If Q changes by more than 0.1 (measurement error) over 100 m free space propagation, the topological protection claim is invalid for this implementation.

### Experiment 2: Skyrmion Number Under Turbulence

**Hypothesis:** Skyrmion number remains within ±0.5 of transmitted value for Cₙ² ≤ 1e-13 m^(-2/3).

**Test:** Generate skyrmion field with N=1. Propagate through turbulence with Cₙ² ∈ {1e-15, 1e-14, 1e-13, 5e-13, 1e-12}. Compute Q_measured at receiver. 100 trials per Cₙ².

**Falsification:** If |Q_measured - 1| > 0.5 for more than 1% of trials at Cₙ² = 1e-13, topological protection fails at practically relevant turbulence.

### Experiment 3: Constellation Distinguishability

**Hypothesis:** 8-state constellation (N ∈ {-3,-2,-1,0,1,2,3,4}) is distinguishable at Cₙ² = 1e-13 with BER < 10⁻³.

**Test:** Generate all 8 skyrmion states. Propagate through Cₙ² = 1e-13 turbulence. Measure Q for each. Compute confusion matrix and BER.

**Falsification:** If BER > 10⁻³, the constellation is too dense for this turbulence level. Reduce M until BER < 10⁻³; report achievable bits/field.

### Experiment 4: Skyrmion Bag vs Single Skyrmion

**Hypothesis:** Skyrmion bag (4 enclosed skyrmions, N ∈ {-2,-1,0,1,2} each → 2 pairs → 2×log₂(5) ≈ 4.64 bits) achieves higher information density than single skyrmion (log₂(5) ≈ 2.32 bits) at same turbulence.

**Test:** Generate skyrmion bag with 4 enclosed skyrmions. Propagate through Cₙ² = 1e-13. Measure all 4 skyrmion numbers. Compute BER per skyrmion number and total bits/field.

**Falsification:** If bag BER per skyrmion > single skyrmion BER at same turbulence, bagging doesn't improve information density under turbulence.

### Experiment 5: Source-Position Encoding (Nonlinear Computation with Linear Optics)

**Hypothesis:** Source position in a structured medium produces nonlinear measured field that can encode information.

**Test:** Implement source-position encoding simulation: create structured medium (random dielectric distribution), compute field at receiver for different source positions, measure nonlinearity of source-position-to-field mapping.

**Falsification:** If measured field varies linearly with source position (within noise), nonlinear computation via source-position encoding is not viable for this medium configuration.

---

## 6. Implementation Dependencies

### XPOCK Modules (already exist, need integration):
- `xpock.properties.PolarizationTexture` — generates skyrmion polarization fields ✓
- `xpock.properties.TopologicalCharge` — computes skyrmion number ✓
- `xpock.analysis.WaveAnalyzer` — computes measurable properties ✓
- `xpock.parameter_space.ParameterSpace` — systematic parameter sweeps ✓
- `xpock.visualization.WaveVisualizer` — field visualizations ✓
- `xpock.validation.validate_wave_event` — field validation ✓

### XPOCK Modules (need to write):
- Turbulence propagation model (Kolmogorov phase screens, split-step Fourier)
- Receiver-side Stokes polarimetry simulation
- End-to-end BER simulation loop
- Constellation generation and hard-decision demodulation
- Comparison with OAM multiplexing baseline

### ENGINES Modules (could be used, not yet integrated):
- `differential_equations_engine` — wave equation solvers for propagation
- `information_theory_engine` — Shannon capacity, mutual information, Blahut-Arimoto
- `integral_transforms/fourier_transforms` — FFT-based propagation
- `dynamical_systems_engine` — nonlinear wave dynamics (for nonlinear computation exploration)
- `materials_science_engine` — optical material properties (for structured medium design)

---

## 7. Timeline (Weeks)

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Implement turbulence propagation model + skyrmion field generator | Working simulate() function |
| 2 | Implement receiver-side skyrmion measurement + hard-decision demodulation | Working demodulate() function |
| 3 | Run Experiment 1 (skyrmion number conservation) + Experiment 2 (turbulence tolerance) | BER vs Cₙ² data |
| 4 | Run Experiment 3 (constellation distinguishability) + Experiment 4 (skyrmion bags) | Bits/field vs turbulence |
| 5 | Run Experiment 5 (source-position encoding) | Go/no-go on nonlinear optics approach |
| 6 | Compile results, write final report, compare with conventional modulation | Final recommendation |

---

## 8. Go/No-Go Criteria

**Proceed to hardware prototype if:**
- BER < 10⁻³ at Cₙ² ≤ 1e-13 m^(-2/3) for ≥ 3-bit constellation
- Skyrmion number measurement error < 0.25 (99% correct nearest-integer)
- Information density ≥ 2 bits/field
- End-to-end simulation reproduces published 270 m results (qualitatively)

**Pivot to alternative mechanism if:**
- Skyrmion number BER > 0.1 at Cₙ² ≤ 1e-14
- Receiver measurement too slow/complex for practical use
- Information density < 1 bit/field after optimization

**Abandon wave-structure approach if:**
- All candidate mechanisms show BER > 0.3 at Cₙ² ≤ 1e-13
- No wave property shows topological protection under turbulence
- Information density cannot exceed conventional intensity modulation

---

## 9. Files to Create (in XPOCK repo)

| File | Purpose |
|------|---------|
| `src/xpock/wave_info/turbulence.py` | Kolmogorov turbulence propagation model |
| `src/xpock/wave_info/skyrmion_modem.py` | Skyrmion modulate/demodulate |
| `src/xpock/wave_info/ber_simulation.py` | End-to-end BER simulation |
| `src/xpock/wave_info/comparison.py` | Comparison with OAM/intensity modulation |
| `tests/test_wave_info.py` | Tests for all new modules |
| `examples/wave_info_demo.py` | Example end-to-end simulation |
| `FINDINGS.md` | Research findings (this document's sibling) |
| `IMPLEMENTATION_PLAN.md` | This document |

---

**Next step:** Implement `src/xpock/wave_info/turbulence.py` and `src/xpock/wave_info/skyrmion_modem.py`, then run Experiment 1.
