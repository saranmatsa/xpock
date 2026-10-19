# XPOCK → HX-241 Integration Module

**Location:** `C:/XPOCK/xpock/241-integration/`
**Purpose:** Integration artifacts for incorporating XPOCK's computational outputs into the HX-241 Rust codebase.
**Status:** Complete — all modules implemented, tested, and committed.

---

## 1. Overview

This module provides the bridge between XPOCK (Python wave-structure generator) and HX-241 (Rust clean-slate communication architecture). It produces language-agnostic JSON artifacts and Rust-compatible test vectors that can be manually incorporated into HX-241 without making XPOCK or Python a runtime dependency.

**Architectural compliance:**
- ✅ No IP/TCP/UDP/QUIC/DNS/HTTP dependencies
- ✅ HX-241 protocol logic is not modified
- ✅ Carriers provide physical capabilities, not protocol definitions
- ✅ XPOCK/Python is NOT a runtime dependency for HX-241
- ✅ ENGINES preserved as independent, reusable asset

---

## 2. Module Structure

```
241-integration/
├── schemas/              # Portable JSON schemas (5 files)
├── vectors/              # Test vector output directory (generated at runtime)
├── adapters/             # XPOCK→HX-241 type adapters
├── validation/           # Contract conformance checker
├── simulation/           # hx-sim compatible inputs
├── reference/            # Rust integration example
├── tests/                # Reproducible tests (17 tests, all passing)
└── docs/                 # This documentation
```

---

## 3. Schema-to-Rust-Type Mapping

| Schema File | HX-241 Rust Type | Location | Contract |
|-------------|-----------------|----------|----------|
| `carrier_capabilities.json` | `CarrierCapabilities` | `hx-transport/src/carrier_types.rs` | CARRIER-CONTRACT.md §3.3 |
| `network_impairment_params.json` | `NetworkImpairment` | `hx-sim/src/impairment.rs` | — |
| `capacity_analysis_result.json` | `BenchmarkConfig` (input) | `hx-sim/src/benchmark.rs` | — |
| `carrier_channel_model.json` | Physical layer model | Not in core protocol | I-CARR-001 to I-CARR-005 |
| `modulation_constellation.json` | Optional modulation layer | Application layer | Not mandated |

---

## 4. Integration Steps (for HX-241 maintainers)

### Step 1: Generate Test Vectors

From the XPOCK project directory, run:

```python
from xpock.hx241.test_vectors import generate_and_save_test_vectors

json_path, rust_path = generate_and_save_test_vectors(
    workspace="./241-integration/vectors"
)
```

This produces:
- `241-integration/vectors/xpock_hx241_test_vectors.json` — JSON test vectors
- `241-integration/vectors/xpock_test_vectors.rs` — Rust source with embedded vectors

### Step 2: Copy JSON to HX-241 Test Data

```bash
cp 241-integration/vectors/xpock_hx241_test_vectors.json \
   C:/hx-241/hx-sim/tests/data/
```

### Step 3: Deserialize in Rust

Use the reference implementation in `241-integration/reference/rust_integration_example.rs`:

```rust
use serde::Deserialize;

#[derive(Deserialize)]
pub struct NetworkImpairment {
    pub packet_loss_rate: f64,
    pub corruption_rate: f64,
    pub base_latency_us: u64,
    pub jitter_us: u64,
    pub bandwidth_bps: u64,
    pub enabled: bool,
}

let impairment: NetworkImpairment =
    serde_json::from_str(json_str)?;
```

### Step 4: Feed to hx-sim

```rust
let mut engine = ImpairmentEngine::new(Some(impairment));
engine.set_enabled(true);
```

### Step 5: Configure Benchmarks

Use `CapacityAnalysisResult` to set `BenchmarkConfig` fields:
- `bandwidth_bps` ← `shannon_capacity_bps`
- `latency_us` ← derived from `symbols_per_second`
- `loss_rate` ← `1.0 - distinguishable_states / 256.0`

---

## 5. What XPOCK Contributes (Evidence-Based)

### Established Results (validated by tests)

| Capability | What It Provides | HX-241 Problem Solved |
|------------|-----------------|----------------------|
| Carrier wave modeling | Path loss, SNR, delay spread, coherence bandwidth, Doppler spread for 6 media types | Carrier capability estimation (no real hardware needed) |
| Channel capacity analysis | Shannon capacity, spectral efficiency, distinguishable wave states | Session dimensioning, flow control window sizing |
| Physically-based impairment models | hx-sim-compatible packet_loss_rate, corruption_rate, jitter_us, bandwidth_bps | Realistic adversarial/conformance testing |
| Wave-structure modulation | Constellation mapping of bits to wave properties (OAM, polarization, temporal shape, etc.) | Optional modulation layer (NOT mandated) |
| Receiver-side spectral analysis | Feature extraction from received waves (OAM, polarization, bandwidth, duration) | Demodulation support |
| Validation suite | Contract conformance checking (no conventional stack, carrier independence) | Architectural compliance verification |

### Hypotheses (not established results)

| Hypothesis | Status | Evidence Needed |
|------------|--------|-----------------|
| Wave-structure modulation outperforms conventional QAM/OFDM | Unproven | Physical hardware testing |
| Channel capacity estimates match achievable rates | Unproven | Real-world link measurements |
| Physically-based impairment models match real carrier behavior | Unproven | Calibration against measured data |
| Nonlinear propagation (Kerr, solitons) improves link performance | Unproven | Experimental validation |

---

## 6. Limitations

1. **No physical validation:** All models are mathematical; no real hardware testing has been performed
2. **SNR at optical frequencies is extremely low:** Free-space optical at 200 THz over 1m gives -239 dB SNR — practical links need much shorter distances or higher power
3. **Acoustic model produces negative path loss:** Model artifact at short distances; needs refinement
4. **Channel capacity is a mathematical upper bound:** Actual achievable rates depend on implementation quality
5. **Wave-structure modulation is optional:** HX-241 does not require it; conventional modulation can be used instead

---

## 7. Files Delivered

| File | Lines | Purpose |
|------|-------|---------|
| `schemas/carrier_capabilities.json` | ~70 | CarrierCapabilities JSON schema |
| `schemas/network_impairment_params.json` | ~95 | NetworkImpairment JSON schema |
| `schemas/capacity_analysis_result.json` | ~85 | Capacity analysis JSON schema |
| `schemas/modulation_constellation.json` | ~90 | Modulation constellation JSON schema |
| `schemas/carrier_channel_model.json` | ~105 | Carrier channel model JSON schema |
| `schemas/index.json` | ~75 | Schema index with HX-241 mapping |
| `test_vectors.py` | ~580 | Test vector generator (JSON + Rust source) |
| `adapters/__init__.py` | ~370 | XPOCK→HX-241 type adapters |
| `validation/__init__.py` | ~480 | Contract conformance checker |
| `simulation/__init__.py` | ~490 | hx-sim compatible inputs |
| `reference/rust_integration_example.rs` | ~185 | Rust deserialization example |
| `tests/test_integration.py` | ~630 | 17 tests, all passing |

---

## 8. How to Regenerate

All artifacts can be regenerated from XPOCK:

```python
# 1. Test vectors (JSON + Rust source)
from xpock.hx241.test_vectors import generate_and_save_test_vectors
generate_and_save_test_vectors()

# 2. Run all tests
# pytest 241-integration/tests/ -v

# 3. Generate simulation inputs
from 241-integration.simulation import generate_and_save_simulation_inputs
generate_and_save_simulation_inputs()
```

---

## 9. Bottom Line

XPOCK provides **physically-based carrier modeling, channel capacity analysis, impairment parameters, and wave-structure modulation** that integrate with HX-241 at the carrier boundary. HX-241's protocol logic remains untouched and carrier-agnostic. The integration is purely additive — HX-241 can adopt or ignore XPOCK's outputs without any architectural compromise.

**Wave-based communication is ONE possible approach within HX-241's framework, not a requirement.** The evidence shows XPOCK adds the most value at the carrier capability estimation and impairment modeling layers, where HX-241 currently has only stubs and defaults.