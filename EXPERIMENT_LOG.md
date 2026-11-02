# XPOCK End-to-End Scientific Validation Log

This log documents the systematic validation of XPOCK's wave-structure
communication simulation. Following the numerical-physics-simulation-audit
skill procedure:

## Validation Protocol

1. **Enumerate all simulation modules and duplicates** — diff canonical vs parallel copies
2. **Validate turbulence models against theory** — Rytov variance, power conservation, n_screens independence
3. **Validate topological invariants** — skyrmion/winding number quantization
4. **Validate polarization metrics** — pointwise DOP
5. **Ensure statistical independence of trials** — fresh phase screens per trial
6. **Ensure fair multi-scheme comparison** — shared turbulence model, identical conditions
7. **Verify numerical stability and convergence** — grid resolution, CFL-like limits
8. **End-to-end communication performance** — SKM vs OAM vs intensity under identical constraints

## Experiment Categories

| ID | Category | Focus |
|----|----------|-------|
| V1 | Topological charge quantization | Skyrmion Q = q exactness, extent-independence |
| V2 | Turbulence physics | Rytov variance, power conservation, n_screens independence |
| V3 | Receiver physics | DOP pointwise vs average-first, noise statistics |
| V4 | Statistical independence | Fresh screens per trial, RNG advancement |
| V5 | Fair comparison | SKM vs OAM vs intensity under identical turbulence |
| V6 | Symbol distinguishability | Confusion matrices, BER vs turbulence |
| V7 | Edge cases | Boundary conditions, grid convergence, numerical stability |

Each experiment entry records: hypothesis, method, raw numbers, pass/fail, and interpretation.

## V2 Turbulence Model Validation — 2026-11-02 21:00:08
| Check | Passed | Measured | Expected | Rel Error | Notes |
|-------|--------|----------|----------|-----------|-------|
| phase_screen_variance_matches_spectrum | ✓ | 168.867 | 168.867 | 0 | empirical=168.866775, theory=168.866775, rel_err=0.0000 |
| turbulence_independent_of_n_screens | ✓ | 0.0428987 | 0 | 0.0237767 | phase_stds=['1.8021', '1.7828', '1.8257', '1.8164', '1.7942'], spread=0.0429 |
| power_conservation_split_step | ✓ | 6.31878e-15 | 0 | 6.31878e-15 | max relative power error=0.000000 |

**Summary: 3/3 checks passed**

## V1 Skyrmion Quantization — 2026-11-02 21:00:08
| Check | Passed | Measured | Expected | Rel Error | Notes |
|-------|--------|----------|----------|-----------|-------|
| skyrmion_Q_q=-4 | ✓ | -4 | -4 | 2.22045e-16 | Q=-4.000000, round(Q)=-4 |
| skyrmion_Q_q=-3 | ✓ | -3 | -3 | 0 | Q=-3.000000, round(Q)=-3 |
| skyrmion_Q_q=-2 | ✓ | -2 | -2 | 2.22045e-16 | Q=-2.000000, round(Q)=-2 |
| skyrmion_Q_q=-1 | ✓ | -1 | -1 | 0 | Q=-1.000000, round(Q)=-1 |
| skyrmion_Q_q=1 | ✓ | 1 | 1 | 0 | Q=1.000000, round(Q)=1 |
| skyrmion_Q_q=2 | ✓ | 2 | 2 | 2.22045e-16 | Q=2.000000, round(Q)=2 |
| skyrmion_Q_q=3 | ✓ | 3 | 3 | 0 | Q=3.000000, round(Q)=3 |
| skyrmion_Q_q=4 | ✓ | 4 | 4 | 2.22045e-16 | Q=4.000000, round(Q)=4 |
| skyrmion_quantization_all_charges | ✓ | 1 | 1 | 0 | All charges [-4, -3, -2, -1, 1, 2, 3, 4] quantized exactly |
| skyrmion_grid_conv_n=32 | ✓ | 2 | 2 | 4.44089e-16 | Q=2.000000 at n=32 |
| skyrmion_grid_conv_n=64 | ✓ | 2 | 2 | 4.44089e-16 | Q=2.000000 at n=64 |
| skyrmion_grid_conv_n=128 | ✓ | 2 | 2 | 4.44089e-16 | Q=2.000000 at n=128 |
| skyrmion_grid_conv_n=256 | ✓ | 2 | 2 | 4.44089e-16 | Q=2.000000 at n=256 |
| skyrmion_grid_convergence | ✓ | 1 | 1 | 0 | Q=2 exact at n=32,64,128,256 |

**Summary: 14/14 checks passed**

## Fair Comparison — 2026-11-02 21:00:12

Ideal receiver (noise_std=0), 100 m link, 64x64 grid

| Scheme | Cn² | Trials | Errors | BER | SER | Extra |
|--------|-----|--------|--------|-----|-----|-------|
| SKM | 1e-15 | 50 | 0 | 0.000000 | 0.000000 | Q̄=+0.1200,σ_Q=0.9928 |
| OAM | 1e-15 | 50 | 0 | 0.000000 | 0.000000 | overlap=0.8848 |
| Intensity | 1e-15 | 50 | 0 | 0.000000 | 0.000000 | Î=1.2900 |
| SKM | 1e-14 | 50 | 0 | 0.000000 | 0.000000 | Q̄=-0.1200,σ_Q=0.9928 |
| OAM | 1e-14 | 50 | 8 | 0.160000 | 0.160000 | overlap=0.4798 |
| Intensity | 1e-14 | 50 | 0 | 0.000000 | 0.000000 | Î=1.1300 |
| SKM | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Q̄=+0.0000,σ_Q=1.0000 |
| OAM | 1e-13 | 50 | 23 | 0.460000 | 0.460000 | overlap=0.0912 |
| Intensity | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Î=1.2100 |
| SKM | 1e-12 | 50 | 0 | 0.000000 | 0.000000 | Q̄=+0.0800,σ_Q=0.9968 |
| OAM | 1e-12 | 50 | 26 | 0.520000 | 0.520000 | overlap=0.0340 |
| Intensity | 1e-12 | 50 | 0 | 0.000000 | 0.000000 | Î=1.4500 |
| SKM | 1e-11 | 50 | 0 | 0.000000 | 0.000000 | Q̄=-0.0400,σ_Q=0.9992 |
| OAM | 1e-11 | 50 | 26 | 0.520000 | 0.520000 | overlap=0.0277 |
| Intensity | 1e-11 | 50 | 0 | 0.000000 | 0.000000 | Î=1.4500 |

## V5 Fair Comparison — Ideal Receiver — 2026-11-02 21:00:12
| Check | Passed | Measured | Expected | Rel Error | Notes |
|-------|--------|----------|----------|-----------|-------|
| SKM_advantage_Cn2=1e-15 | ✓ | 0 | 0 | 0 | SKM BER=0.000000, OAM BER=0.000000 |
| SKM_advantage_Cn2=1e-14 | ✓ | 0 | 0.16 | 1 | SKM BER=0.000000, OAM BER=0.160000 |
| SKM_advantage_Cn2=1e-13 | ✓ | 0 | 0.46 | 1 | SKM BER=0.000000, OAM BER=0.460000 |
| SKM_advantage_Cn2=1e-12 | ✓ | 0 | 0.52 | 1 | SKM BER=0.000000, OAM BER=0.520000 |
| SKM_advantage_Cn2=1e-11 | ✓ | 0 | 0.52 | 1 | SKM BER=0.000000, OAM BER=0.520000 |

**Summary: 5/5 checks passed**

## Fair Comparison — 2026-11-02 21:00:13

Receiver noise_std=0.0

| Scheme | Cn² | Trials | Errors | BER | SER | Extra |
|--------|-----|--------|--------|-----|-----|-------|
| SKM | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Q̄=+0.2000,σ_Q=0.9798 |
| OAM | 1e-13 | 50 | 22 | 0.440000 | 0.440000 | overlap=0.0856 |
| Intensity | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Î=1.3300 |

## Fair Comparison — 2026-11-02 21:00:14

Receiver noise_std=0.05

| Scheme | Cn² | Trials | Errors | BER | SER | Extra |
|--------|-----|--------|--------|-----|-----|-------|
| SKM | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Q̄=-0.0803,σ_Q=0.9967 |
| OAM | 1e-13 | 50 | 22 | 0.440000 | 0.440000 | overlap=0.0856 |
| Intensity | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Î=1.4133 |

## Fair Comparison — 2026-11-02 21:00:15

Receiver noise_std=0.1

| Scheme | Cn² | Trials | Errors | BER | SER | Extra |
|--------|-----|--------|--------|-----|-----|-------|
| SKM | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Q̄=-0.0813,σ_Q=0.9968 |
| OAM | 1e-13 | 50 | 23 | 0.460000 | 0.460000 | overlap=0.0857 |
| Intensity | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Î=1.4236 |

## Fair Comparison — 2026-11-02 21:00:16

Receiver noise_std=0.2

| Scheme | Cn² | Trials | Errors | BER | SER | Extra |
|--------|-----|--------|--------|-----|-----|-------|
| SKM | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Q̄=-0.0860,σ_Q=1.0070 |
| OAM | 1e-13 | 50 | 23 | 0.460000 | 0.460000 | overlap=0.0859 |
| Intensity | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Î=1.4655 |

## Fair Comparison — 2026-11-02 21:00:16

Receiver noise_std=0.3

| Scheme | Cn² | Trials | Errors | BER | SER | Extra |
|--------|-----|--------|--------|-----|-----|-------|
| SKM | 1e-13 | 50 | 14 | 0.280000 | 0.280000 | Q̄=-0.1150,σ_Q=1.6365 |
| OAM | 1e-13 | 50 | 24 | 0.480000 | 0.480000 | overlap=0.0860 |
| Intensity | 1e-13 | 50 | 0 | 0.000000 | 0.000000 | Î=1.5355 |
