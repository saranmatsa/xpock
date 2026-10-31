# XPOCK → HX-241 Integration Research

**Date:** 2026-10-19
**Status:** Research → Implementation
**Repositories:** C:/XPOCK/xpock (XPOCK), C:/hx-241 (HX-241), C:/XPOCK/ENGINES (ENGINES)

---

## 1. Executive Summary

HX-241 is a clean-slate communication architecture designed to replace the fundamental architecture of the Internet. It is explicitly **not** an overlay, compatibility layer, or extension of existing networking infrastructure. It must not depend on IP, TCP, UDP, QUIC, DNS, HTTP, or any conventional protocol stack.

XPOCK is a wave-structure generator that models finite physical wave events with independently controllable properties (amplitude, phase, frequency, temporal shape, spatial mode, polarization, topological charge, coherence, coupled modes). ENGINES is a proprietary computational library with engines for differential equations, information theory, integral transforms, dynamical systems, materials science, experimental physics, CFD, and more.

**Core finding:** XPOCK's capabilities map directly to concrete HX-241 problems at the physical carrier and information-theoretic layers, without violating HX-241's carrier-agnostic architecture. The integration is at the **boundary** between HX-241's abstract carrier interface and the physical reality of wave propagation — exactly where HX-241 needs pluggable capability models.

**No wave-based communication is assumed to be necessary or superior.** This research objectively evaluates where XPOCK adds value and where it does not.

---

## 2. HX-241 Architecture Analysis

### 2.1 Layer Model (from ARCHITECTURE.md, ARCHITECTURE-CONTRACT.md)

```
APPLICATION LAYER (external)
    ↑
hx-net: Integration, CLI, Daemon, Public API
    ↑
hx-fabric: Fabric Orchestrator
    ↑
hx-resilience: Resilience Protocols (conflict, offline queue, anti-entropy)
    ↑
┌─────────────┬─────────────┬─────────────┐
│ hx-storage  │ hx-sync     │ hx-session  │
│ ChunkStore  │ Transfer    │ Streams     │
│ FastCDC     │ DHT         │ Datagrams   │
│ Manifest    │ Dedup       │ Flow/Cong   │
└─────────────┴─────────────┴─────────────┘
    ↑
hx-routing: Path Computation (RoutingEngine, GeographicTopology)
    ↑
hx-discovery: Peer Discovery & Bootstrap (DiscoveryEngine)
    ↑
hx-transport: Frame Encoding & Carrier Abstraction
    ↑
hx-types: Core Identifiers & Invariants (FOUNDATION)
```

**Dependency rule:** No crate depends on any crate above it. Traits define boundaries; implementations live in higher crates. `hx-net` is the sole integration point.

### 2.2 Key Types and Interfaces

| Type | Definition | Layer |
|------|-----------|-------|
| `NodeID` | BLAKE3(Ed25519_pubkey), 32 bytes | hx-types |
| `ObjectID` / `ChunkID` | BLAKE3(content), 32 bytes | hx-types/storage |
| `PathID` | SHA256(src, dst, nonce), 16 bytes | hx-types/routing |
| `SessionID` | SHA256(initiator, responder, nonce), 16 bytes | hx-types/session |
| `StreamID` | 64-bit int per session | hx-types/session |
| `CarrierAddr` | CarrierType + opaque Vec<u8> | hx-transport |
| `TransportAddr` | NodeID + opaque Vec<u8> endpoint | hx-transport |
| `FrameHeader` | 109 bytes (magic, version, type, flags, src/dst NodeID, PathID, seq, payload_len, CRC32, ext_len) | hx-transport |

**Carrier independence invariants (I-CARR-001 to I-CARR-005):**
- HX-241 MUST NOT depend on any specific carrier technology
- CarrierType values 1-32 are opaque; semantics defined by plugins
- CarrierAddr address bytes are opaque to protocol
- Protocol does NOT assume carrier properties (MTU, latency, reliability)
- All carrier capabilities are HINTS only

### 2.3 Security Contract (from CRYPTO-CONTRACT.md)

| Primitive | Algorithm | Purpose |
|-----------|-----------|---------|
| Signing | Ed25519 | Identity signatures |
| Key Agreement | X25519 | ECDH (Noise IK, session keys) |
| Symmetric AEAD | XChaCha20-Poly1305 | Frame/payload encryption |
| KDF | HKDF-SHA256 | Key derivation |
| Hash | BLAKE3 | NodeID, ObjectID, ChunkID |

Noise IK handshake for session establishment. All post-handshake frames encrypted.

### 2.4 Current Implementation Gaps

From ARCHITECTURE-BASELINE.md, TRANSPORT-BASELINE.md, CARRIER-BASELINE.md:

1. **No real carrier implementations** — only `LoopbackCarrier` exists. SerialCarrier send() is stubbed. No Bluetooth, Wi-Fi Direct, LoRa, optical, acoustic, etc.
2. **No physical carrier modeling** — no way to simulate how waves propagate through different media (attenuation, multipath, dispersion, scattering)
3. **No channel capacity analysis** — no information-theoretic limits on distinguishable wave states
4. **No wave-structure modulation** — no mapping from HX-241 frame bits to physical wave properties
5. **Generic impairment models** — hx-sim has loss/delay/jitter/corruption but not physically-based models tied to carrier properties
6. **No spectral analysis** — no receiver-side signal analysis for recovering wave structure properties
7. **Fabric crate is empty** — hx-fabric is a 5-line stub
8. **No end-to-end physical validation** — cannot test HX-241 over simulated physical channels

---

## 3. Concrete Problems HX-241 Must Solve

### Problem 1: Carrier Capability Estimation
**What:** Before selecting a carrier, HX-241 needs to know its capabilities (MTU, bandwidth, latency, reliability, energy cost, range). These are currently hardcoded defaults or stubs.

**Why it matters:** `CarrierCapabilities` are hints used by `TransportManager.select_best()` (I-CARR-013: lowest energy_cost → lowest latency → highest bandwidth). Without accurate capability estimates, path selection is suboptimal.

**XPOCK value:** Wave propagation models can estimate path loss, delay spread, coherence bandwidth, Doppler spread, and SNR from physical parameters (frequency, distance, medium type). These map directly to `CarrierCapabilities` fields.

### Problem 2: Channel Capacity and Distinguishable States
**What:** HX-241 needs to know how much information can be reliably transmitted over a carrier, and how many distinguishable wave states exist for modulation.

**Why it matters:** Determines bits per symbol, symbols per second, and net information rate. Without this, HX-241 cannot dimension sessions, flow control windows, or congestion control parameters.

**XPOCK value:** Information theory engine (Shannon entropy, mutual information, channel capacity via Blahut-Arimoto, Gaussian channel capacity) + wave structure distinguishability analysis (JS divergence between feature signatures).

### Problem 3: Physically-Based Impairment Models
**What:** hx-sim's `NetworkImpairment` has generic parameters (packet_loss_rate, corruption_rate, base_latency_us, jitter_us) but no physical basis for these values.

**Why it matters:** Adversarial testing and conformance testing need realistic impairment profiles tied to carrier physics (atmospheric turbulence, multipath fading, fiber dispersion, acoustic absorption).

**XPOCK value:** Carrier wave models produce physically-based impairment parameters (path loss → packet loss rate, delay spread → jitter, SNR → corruption rate, coherence bandwidth → bandwidth limit).

### Problem 4: Wave-Structure Modulation
**What:** HX-241 frames need to be represented as physical wave structures for transmission. The mapping from frame bits to wave properties is undefined.

**Why it matters:** Without a modulation abstraction, HX-241 cannot operate over physical carriers. The modulation must be carrier-agnostic — the protocol should not depend on QAM, OFDM, or any specific modulation.

**XPOCK value:** Wave generation (temporal profiles, spatial modes, polarization, topological charge) provides a rich parameter space for encoding information. XPOCK can define an abstract modulation layer that maps frame fields to wave properties without assuming any specific physical modulation.

### Problem 5: Receiver-Side Signal Analysis
**What:** A receiver needs to analyze an incoming wave to recover the encoded information (which wave properties were modulated).

**Why it matters:** Without receiver-side analysis, there is no demodulation, no information recovery, no end-to-end communication.

**XPOCK value:** Spectral analysis engine (FFT, CWT, Hankel, Hilbert) + ENGINES integral transforms + wave analyzer (OAM, winding number, polarization, Schmidt number) provides receiver-side feature extraction.

### Problem 6: Nonlinear Propagation Effects
**What:** Wave propagation through physical media involves nonlinear effects (Kerr nonlinearity, self-focusing, modulation instability, soliton formation) that affect signal integrity.

**Why it matters:** For high-power or long-distance links, nonlinear effects can distort or destroy information. HX-241 needs to model these to dimension links correctly.

**XPOCK value:** Nonlinear Schrödinger equation solver (differential equations engine), coupled-mode analysis, modulation instability gain calculation (dynamical systems engine).

---

## 4. Alternative Approaches (Objective Evaluation)

### Approach A: Conventional Modulation (QAM, OFDM, PSK)
**Description:** Use established digital modulation schemes to map bits to wave properties.

**Pros:** Well-understood, mature technology, proven performance.

**Cons:**
- Assumes specific carrier properties (RF, optical carrier)
- Violates carrier-agnostic principle — QAM doesn't make sense for acoustic or pressure-wave carriers
- Ties HX-241 to conventional communications theory
- Not extensible to unconventional wave properties (OAM, polarization textures, topological charge)

**Verdict:** Rejected as the primary modulation approach for HX-241. Could be one option among many within an abstract modulation layer.

### Approach B: Wave-Structure Modulation (XPOCK's approach)
**Description:** Map information to independently controllable wave properties (temporal shape, spatial mode, polarization, OAM, topological charge, coherence).

**Pros:**
- Fully carrier-agnostic — same abstraction works for optical, RF, acoustic, pressure wave
- Extensible — new wave properties can be added as research identifies them
- Rich parameter space — many dimensions for information encoding
- Aligns with HX-241's intent-driven communication principle

**Cons:**
- Less mature than conventional modulation
- Receiver-side analysis is more complex
- Requires physical validation (not yet done)

**Verdict:** Recommended as the abstract modulation layer. Conventional modulation can be one implementation option within this layer.

### Approach C: Hybrid (Abstract + Conventional)
**Description:** Define an abstract modulation interface. XPOCK provides wave-structure modulation as one implementation. Conventional QAM/OFDM can be added later if needed.

**Pros:** Best of both worlds — carrier-agnostic abstraction with option to use proven modulation.

**Cons:** More complex to implement.

**Verdict:** Recommended. XPOCK implements the wave-structure side; the interface is abstract enough to accept other implementations.

---

## 5. Integration Strategy: Boundary Model

XPOCK integrates with HX-241 at the **carrier boundary** — the interface between HX-241's abstract `Carrier` trait and physical wave reality. XPOCK does NOT modify HX-241 protocol logic. It provides:

1. **Carrier wave models** → physically-based `CarrierCapabilities` estimates
2. **Channel capacity analysis** → information-theoretic limits for session dimensioning
3. **Wave-structure modulation** → abstract mapping from HX-241 frames to wave properties
4. **Physically-based impairment models** → realistic `NetworkImpairment` parameters for hx-sim
5. **Receiver-side spectral analysis** → wave property extraction from received signals
6. **Nonlinear propagation modeling** → link dimensioning for nonlinear media

**Dependencies:** XPOCK → ENGINES (information theory, differential equations, dynamical systems, integral transforms, materials). XPOCK → HX-241 types (NodeID, PathID, CarrierAddr, etc.) only for interface compatibility. No reverse dependency.

**Architectural compliance:** XPOCK does not depend on IP/TCP/UDP/QUIC/DNS/HTTP. It models physical waves abstractly. HX-241 protocol logic remains untouched.

---

## 6. Interface Requirements

### 6.1 Carrier Capabilities Estimation Interface

```python
class CarrierWaveModeler:
    def model_carrier(self, config: CarrierWaveConfig) -> CarrierChannelModel
    def impairment_to_hx_sim(self, channel_model: CarrierChannelModel) -> dict
    def generate_transmit_wave(self, config, grid_arrays) -> WaveEvent
```

**Inputs:** medium type, frequency, wavelength, distance, power, gains, medium-specific parameters
**Outputs:** frequency response, impulse response, path loss, delay spread, coherence bandwidth, Doppler spread, capacity, SNR, impairment parameters

### 6.2 Channel Capacity Analysis Interface

```python
class ChannelCapacityAnalyzer:
    def analyze_capacity(self, snr_db, bandwidth_hz, wave_events, fec_overhead) -> CapacityAnalysisResult
    def analyze_distinguishability(self, wave_events, metric) -> DistinguishabilityAnalysisResult
    def analyze_feature_independence(self, wave_events) -> dict
```

**Inputs:** SNR, bandwidth, list of wave events, distinguishability metric
**Outputs:** Shannon capacity, spectral efficiency, distinguishable states, bits per symbol, feature correlation matrix

### 6.3 Wave-Structure Modulation Interface

```python
class WaveModulator:
    def modulate_frame(self, frame_bytes, wave_properties) -> WaveEvent
    def demodulate_wave(self, wave_event) -> dict  # recovered properties
```

**Inputs:** HX-241 frame bytes, wave property assignments
**Outputs:** WaveEvent for transmission; recovered property dict for reception

### 6.4 Impairment Model Interface

```python
def channel_to_impairment(channel_model: CarrierChannelModel) -> NetworkImpairmentParams
```

**Outputs:** packet_loss_rate, corruption_rate, base_latency_us, jitter_us, bandwidth_bps for hx-sim

---

## 7. Dependencies and Limitations

### Dependencies
- **ENGINES** (C:/XPOCK/ENGINES): information theory, differential equations, dynamical systems, integral transforms, materials science, experimental physics
- **XPOCK core**: wave generation, analysis, visualization
- **HX-241 types** (read-only interface): NodeID, PathID, CarrierAddr, CarrierType, CarrierCapabilities — used only for type compatibility, no Rust dependency

### Limitations (hypotheses, not established results)
1. Wave-structure modulation has not been physically validated — no real hardware testing
2. Channel capacity estimates are mathematical models — actual achievable rates depend on implementation
3. Physically-based impairment models need calibration against real carrier measurements
4. Receiver-side analysis assumes idealized signal models — real receivers have noise, offset, drift

### What XPOCK does NOT do
- Does not implement HX-241 protocol logic (no frame encoding, no session management, no routing)
- Does not depend on any conventional networking stack
- Does not replace hx-sim's impairment engine — it feeds parameters to it
- Does not make claims about experimental wave transmission

---

## 8. Measurable Evaluation Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Carrier capability estimation accuracy | Within 30% of physical model | Compare model output to known physics |
| Channel capacity computation | Matches Shannon formula | Verify C = B·log2(1+SNR) |
| Wave state distinguishability | JS divergence > 0.05 for distinct states | Pairwise analysis of generated wave events |
| Modulation/demodulation roundtrip | Property recovery > 90% accuracy | Modulate → propagate → demodulate |
| Impairment parameter generation | Valid hx-sim NetworkImpairment | Parameters within valid ranges [0,1] for rates |
| No conventional stack dependency | Zero imports of socket/TCP/UDP | Code audit |
| Test coverage | >80% of integration modules | pytest coverage |

---

## 9. Implementation Priority

1. **P0 (Immediate):** Carrier wave modeling + channel capacity analysis (directly solves Problems 1, 2)
2. **P1 (Next):** Wave-structure modulation interface + physically-based impairment models (Problems 3, 4)
3. **P2 (Later):** Receiver-side spectral analysis + nonlinear propagation modeling (Problems 5, 6)

P0 provides immediate value: HX-241 can estimate carrier capabilities and channel capacity without real hardware. P1 enables actual wave-based communication. P2 enables receiver-side processing and nonlinear link dimensioning.

---

## 10. Conclusion

XPOCK offers genuine, architecturally-compliant value to HX-241 at the carrier boundary. The integration does not violate any HX-241 invariant, does not depend on conventional networking, and preserves ENGINES as an independent asset. The most valuable immediate integration is **carrier wave modeling + channel capacity analysis**, which directly addresses HX-241's lack of physical carrier modeling and information-theoretic limits.

Wave-based communication is one possible modulation approach within HX-241's carrier-agnostic framework, not a requirement. XPOCK provides the tools to investigate it objectively, with measurable criteria for success or failure.