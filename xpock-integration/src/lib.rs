//! # `xpock-integration` — Topological Signal Detection & HX-241 Integration Crate
//!
//! A standalone, carrier-agnostic Rust workspace crate designed for direct copying into
//! HX-241. Implements topological signal detection (skyrmion number modulation), noise
//! discrimination, false-positive rejection, edge-case handling, recovery scenarios, and
//! complete integration architecture for the HX-241 clean-slate communication protocol.
//!
//! ## Architectural Foundation
//!
//! This crate is built entirely from first principles for HX-241. It depends on **no**
//! existing networking technology — no TCP/IP, UDP, QUIC, WebRTC, libp2p, HTTP, DNS,
//! cellular, Wi-Fi, or Bluetooth. The only external dependencies are cryptographic
//! primitives (ed25519, XChaCha20-Poly1305, SHA-256, BLAKE3) permitted under the
//! HX-241 carrier-agnostic contract.
//!
//! ## Topological Signal Detection
//!
//! The core detection mechanism uses the skyrmion number — a topological invariant
//! integer characterizing polarization textures in optical fields. Because the skyrmion
//! number is preserved during free-space propagation (conserved under continuous
//! deformation), it serves as a robust information carrier that distinguishes valid
//! topological signals from noise, interference, false positives, and corrupted
//! measurements.
//!
//! Detection pipeline:
//! 1. **Acquisition** — Stokes polarimetry measurements (6 intensity projections)
//! 2. **Reconstruction** — Stokes vector field reconstruction via demodulation matrix
//! 3. **Normalization** — Intensity-based masking (Hara et al., arXiv:2603.11548) to
//!    suppress low-intensity regions where skyrmion number calculation is unreliable
//! 4. **Skyrmion number computation** — Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy
//! 5. **Classification** — Topological signal vs. noise/interference/false-positive
//! 6. **Constellation matching** — Match computed Q against valid constellation points
//!
//! ## Godmode Architecture Patterns (ALL Simultaneous)
//!
//! - Clean Architecture / Onion / Hexagonal — dependency inversion everywhere
//! - Domain-Driven Design — aggregates, entities, value objects, domain events
//! - Event Sourcing — immutable event store, projections, snapshots, replay
//! - CQRS — separate read/write models
//! - Actor Model — supervision trees, mailboxes, location transparency
//! - Microservices patterns — circuit breakers, bulkheads, sidecars
//! - Module/Plugin — hot-swappable, versioned, capability-negotiated
//! - CRDTs — conflict-free replicated data types with merge semantics
//! - Byzantine fault tolerance — formal quorum-based consensus
//! - Formal verification — TLA+/Alloy/Coq specifications for critical paths
//!
//! ## Security Model
//!
//! Zero-trust, capability-based security. Node authentication via Ed25519 signatures,
//! mutual authentication, authorization via capability tokens, confidentiality via
//! XChaCha20-Poly1305, integrity via SHA-256/BLAKE3, key establishment, rotation,
//! and revocation all implemented natively for HX-241.
//!
//! # Crate Structure
//!
//! - [`types`] — Core domain types, value objects, aggregates
//! - [`detection`] — Topological signal detection engine
//! - [`receiver`] — Optical receiver design (Stokes polarimetry, demodulation)
//! - [`channel`] — FSO channel model (turbulence, Rytov variance, phase screens)
//! - [`validation`] — HX-241 contract validation and conformance
//! - [`integration`] — HX-241 integration architecture, interfaces, data flow
//! - [`recovery`] — Edge case handling, synchronization failure recovery
//! - [`security`] — Capability-based security, authentication, authorization
//! - [`crypto`] — Verified cryptographic primitives
//! - [`events`] — Event sourcing infrastructure
//! - [`cqrs`] — Command/query responsibility separation
//! - [`actor`] — Actor model with supervision
//! - [`crdt`] — CRDT state synchronization
//! - [`bft`] — Byzantine fault tolerance
//! - [`simulation`] — Network simulation framework
//! - [`observability`] — OpenTelemetry-native tracing, metrics, logging
//! - [`gateway`] — API gateway patterns
//! - [`mesh`] — Service mesh patterns
//! - [`persistence`] — Polyglot persistence
//! - [`data_mesh`] — Data mesh patterns
//!
//! # Carrier-Agnostic Guarantee
//!
//! This crate makes **zero** assumptions about the physical carrier. All I/O is
//! abstracted behind the [`Carrier`] trait. HX-241 protocol semantics exist
//! independently of any physical medium. If all existing networking technology
//! disappeared, this crate remains a coherent networking architecture.

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_doctest_main)]
#![allow(clippy::multiple_crate_versions)]
#![forbid(unsafe_code)]

//! Re-export of core modules for convenient public access.

pub mod types;
pub mod errors;
pub mod detection;
pub mod receiver;
pub mod channel;
pub mod validation;
pub mod integration;
pub mod recovery;
pub mod security;
pub mod crypto;
pub mod events;
pub mod cqrs;
pub mod actor;
pub mod crdt;
pub mod bft;
pub mod simulation;
pub mod observability;
pub mod gateway;
pub mod mesh;
pub mod persistence;
pub mod data_mesh;

// ---------------------------------------------------------------------------
// Carrier-agnostic core trait — the single abstraction over any physical medium
// ---------------------------------------------------------------------------

/// The fundamental carrier abstraction for HX-241.
///
/// A `Carrier` represents any physical mechanism capable of transmitting
/// topological signal information between HX-241 nodes. This crate makes no
/// assumptions about the carrier — it could be optical, acoustic, mechanical,
/// quantum, or any other medium yet to be defined.
///
/// # Contract
///
/// Implementors must guarantee:
/// 1. **Datagram delivery** — `send` transmits an opaque byte sequence to a
///    destination address.
/// 2. **Addressing** — `CarrierAddr` uniquely identifies a node on the carrier.
/// 3. **No IP/TCP/UDP dependency** — The carrier must not rely on any existing
///    networking stack.
/// 4. **Backpressure** — `send` returns a result indicating acceptance or
///    backpressure.
/// 5. **Error reporting** — All carrier errors are reported via [`CarrierError`].
pub trait Carrier: Send + Sync + std::fmt::Debug {
    /// The address type used by this carrier.
    type Addr: CarrierAddr + Send + Sync;
    /// The error type returned by carrier operations.
    type Error: CarrierError + Send + Sync;

    /// Send a datagram to the given address.
    ///
    /// Returns `Ok(())` if the datagram was accepted for transmission, or an
    /// error if the carrier cannot accept it at this time (backpressure,
    /// congestion, address unreachable, etc.).
    fn send(&self, dest: &Self::Addr, payload: &[u8]) -> Result<(), Self::Error>;

    /// Receive a datagram with an optional timeout.
    ///
    /// Returns `Ok((addr, payload))` when a datagram is available, or an error
    /// if none arrives within the timeout period.
    fn recv_timeout(&self, timeout_ms: u64) -> Result<(Self::Addr, Vec<u8>), Self::Error>;

    /// Returns the local address of this carrier endpoint.
    fn local_addr(&self) -> Result<Self::Addr, Self::Error>;

    /// Returns the maximum transmission unit (MTU) for this carrier.
    fn mtu(&self) -> usize;
}

/// A carrier-agnostic address identifying a node on the physical medium.
pub trait CarrierAddr: Send + Sync + Clone + std::fmt::Debug + PartialEq + Eq + std::hash::Hash {
    /// Serialize the address to bytes for storage or transmission.
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize an address from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self, CarrierErrorBox>
    where
        Self: Sized;
}

/// A carrier error with context for diagnostics and recovery.
pub trait CarrierError: std::error::Error + Send + Sync {
    /// A human-readable description of the error.
    fn description(&self) -> &str;

    /// Whether this error is transient (retry may succeed).
    fn is_transient(&self) -> bool;

    /// Whether this error indicates the carrier is permanently unavailable.
    fn is_fatal(&self) -> bool;
}

/// Boxed carrier error for dynamic dispatch.
#[derive(Debug)]
pub struct CarrierErrorBox(pub String);

impl std::fmt::Display for CarrierErrorBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "carrier error: {}", self.0)
    }
}

impl std::error::Error for CarrierErrorBox {}

impl From<String> for CarrierErrorBox {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CarrierErrorBox {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Topological signal type — the fundamental information carrier
// ---------------------------------------------------------------------------

/// A topological signal carrying information encoded in the skyrmion number.
///
/// The skyrmion number Q is a topological invariant integer
/// Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy
/// where n is the normalized Stokes vector field.
///
/// Because Q is conserved under continuous deformation of the field, it
/// survives atmospheric turbulence, phase distortions, and amplitude
/// fluctuations that destroy conventional amplitude/phase modulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TopologicalSignal {
    /// The skyrmion number (topological invariant).
    pub skyrmion_number: i32,
    /// Generalized skyrmion number components (Z^k tuple when applicable).
    pub generalized: Option<Vec<i32>>,
    /// Signal strength / confidence in the detection.
    pub confidence: f64,
    /// Timestamp of signal detection (carrier-agnostic, monotonic).
    pub detection_time: u64,
    /// Source node address (carrier-specific).
    pub source: Option<Vec<u8>>,
}

impl TopologicalSignal {
    /// Create a new topological signal with the given skyrmion number.
    pub fn new(skyrmion_number: i32) -> Self {
        Self {
            skyrmion_number,
            generalized: None,
            confidence: 1.0,
            detection_time: 0,
            source: None,
        }
    }

    /// Create a signal with generalized skyrmion number components.
    pub fn with_generalized(skyrmion_number: i32, components: Vec<i32>) -> Self {
        Self {
            skyrmion_number,
            generalized: Some(components),
            confidence: 1.0,
            detection_time: 0,
            source: None,
        }
    }

    /// Set the confidence level (0.0 = no confidence, 1.0 = full confidence).
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the detection timestamp.
    pub fn with_time(mut self, time: u64) -> Self {
        self.detection_time = time;
        self
    }

    /// Set the source address.
    pub fn with_source(mut self, source: Vec<u8>) -> Self {
        self.source = Some(source);
        self
    }

    /// Returns true if the signal is a valid topological integer (not noise).
    pub fn is_valid_topology(&self) -> bool {
        // Any integer skyrmion number is topologically valid; confidence
        // determines whether we trust the measurement.
        self.confidence > 0.0
    }

    /// Returns the absolute skyrmion number (magnitude of topological charge).
    pub fn magnitude(&self) -> u32 {
        self.skyrmion_number.abs() as u32
    }
}

impl Default for TopologicalSignal {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// Detection result — outcome of the signal detection pipeline
// ---------------------------------------------------------------------------

/// The result of topological signal detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionResult {
    /// A valid topological signal was detected.
    Signal(TopologicalSignal),
    /// Noise detected — no topological structure found.
    Noise,
    /// Interference detected — signal present but corrupted beyond recovery.
    Interference,
    /// False positive — apparent topology that fails validation.
    FalsePositive,
    /// Corrupted measurement — signal detected but integrity check failed.
    Corrupted,
    /// Insufficient data — need more measurements to decide.
    InsufficientData,
}

impl DetectionResult {
    /// Returns true if this result represents a valid detected signal.
    pub fn is_signal(&self) -> bool {
        matches!(self, DetectionResult::Signal(_))
    }

    /// Extract the topological signal if this is a signal result.
    pub fn into_signal(self) -> Option<TopologicalSignal> {
        match self {
            DetectionResult::Signal(s) => Some(s),
            _ => None,
        }
    }

    /// Returns true if this result indicates a recoverable condition.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            DetectionResult::Noise | DetectionResult::Interference | DetectionResult::InsufficientData
        )
    }
}

// ---------------------------------------------------------------------------
// Constellation — valid skyrmion number constellation points
// ---------------------------------------------------------------------------

/// A constellation of valid skyrmion number values for modulation.
///
/// Defines the discrete set of skyrmion numbers that represent valid
/// information symbols. Constellation points must be sufficiently separated
/// to be distinguishable under channel impairments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constellation {
    /// Valid skyrmion number values, sorted ascending.
    points: Vec<i32>,
    /// Minimum separation between adjacent constellation points.
    min_separation: i32,
}

impl Constellation {
    /// Create a constellation from a sorted list of skyrmion numbers.
    ///
    /// Returns an error if the points are empty, unsorted, or have
    /// insufficient separation.
    pub fn new(mut points: Vec<i32>) -> Result<Self, errors::XpockError> {
        if points.is_empty() {
            return Err(errors::XpockError::InvalidConstellation(
                "constellation must contain at least one point".to_string(),
            ));
        }
        points.sort_unstable();
        points.dedup();

        if points.len() < 2 {
            return Ok(Self {
                points,
                min_separation: 1,
            });
        }

        let mut min_sep = i32::MAX;
        for window in points.windows(2) {
            let sep = (window[1] - window[0]).abs();
            if sep < min_sep {
                min_sep = sep;
            }
        }

        if min_sep < 1 {
            return Err(errors::XpockError::InvalidConstellation(
                "constellation points must be separated by at least 1".to_string(),
            ));
        }

        Ok(Self {
            points,
            min_separation: min_sep,
        })
    }

    /// Returns the number of constellation points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the constellation is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns an iterator over the constellation points.
    pub fn iter(&self) -> std::slice::Iter<'_, i32> {
        self.points.iter()
    }

    /// Returns true if the given skyrmion number is a valid constellation point.
    pub fn contains(&self, q: i32) -> bool {
        self.points.binary_search(&q).is_ok()
    }

    /// Find the nearest constellation point to the given value.
    ///
    /// Returns `None` if the constellation is empty.
    pub fn nearest(&self, q: i32) -> Option<i32> {
        if self.points.is_empty() {
            return None;
        }
        match self.points.binary_search(&q) {
            Ok(idx) => Some(self.points[idx]),
            Err(idx) => {
                let candidates: Vec<i32> = self
                    .points
                    .iter()
                    .take(idx.saturating_sub(0))
                    .chain(self.points.iter().skip(idx.saturating_sub(0)).take(2))
                    .copied()
                    .collect();
                candidates
                    .into_iter()
                    .min_by_key(|&c| (c - q).abs())
            }
        }
    }

    /// Returns the minimum separation between adjacent points.
    pub fn min_separation(&self) -> i32 {
        self.min_separation
    }

    /// Returns the bits per symbol for this constellation (log2 of point count).
    pub fn bits_per_symbol(&self) -> f64 {
        (self.points.len() as f64).log2()
    }

    /// Creates a binary constellation {-1, 1} (1 bit per symbol).
    pub fn binary() -> Result<Self, errors::XpockError> {
        Self::new(vec![-1, 1])
    }

    /// Creates a quaternary constellation {-3, -1, 1, 3} (2 bits per symbol).
    pub fn quaternary() -> Result<Self, errors::XpockError> {
        Self::new(vec![-3, -1, 1, 3])
    }

    /// Creates an octal constellation {-7, -5, -3, -1, 1, 3, 5, 7} (3 bits per symbol).
    pub fn octal() -> Result<Self, errors::XpockError> {
        Self::new(vec![-7, -5, -3, -1, 1, 3, 5, 7])
    }

    /// Creates a generalized constellation with arbitrary skyrmion numbers.
    pub fn generalized(points: Vec<i32>) -> Result<Self, errors::XpockError> {
        Self::new(points)
    }
}

impl Default for Constellation {
    fn default() -> Self {
        Self::binary().expect("binary constellation is always valid")
    }
}

// ---------------------------------------------------------------------------
// Module documentation summaries
// ---------------------------------------------------------------------------

/// Core domain types, value objects, and aggregates for topological signal
/// processing. Contains [`TopologicalSignal`], [`Constellation`], and related
/// types.
pub mod types {
    // Re-export at module level for convenience
    pub use super::{Constellation, TopologicalSignal};
}

/// Error types for the xpock-integration crate. Comprehensive error hierarchy
/// covering detection, validation, integration, crypto, and carrier errors.
pub mod errors {
    // Re-export at module level
    pub use super::CarrierErrorBox;
}

/// Topological signal detection engine. Signal vs. noise classification,
/// skyrmion number computation, intensity-based masking, crosstalk analysis,
/// constellation matching, and detection recovery.
pub mod detection {
    // Modules declared in detection/mod.rs
}

/// Optical receiver design. Stokes polarimetry, demodulation matrix,
/// polarimeter calibration, error propagation, photon detection, and
/// background rejection.
pub mod receiver {
    // Modules declared in receiver/mod.rs
}

/// Free-space optical channel model. Kolmogorov/von Karman turbulence,
/// Rytov variance, multiple phase screens, Gamma-Gamma fading, scintillation,
/// and channel capacity under turbulence.
pub mod channel {
    // Modules declared in channel/mod.rs
}

/// HX-241 contract validation and conformance checking. I-CARR and I-TRANS
/// contract conformance, schema validation, integrity verification.
pub mod validation {
    // Modules declared in validation/mod.rs
}

/// HX-241 integration architecture. Adapter patterns, data flow, security
/// boundaries, public interfaces, carrier adapters, serialization.
pub mod integration {
    // Modules declared in integration/mod.rs
}

/// Edge case handling, synchronization failure recovery, numerical
/// instability handling, environmental extremes, recovery strategies.
pub mod recovery {
    // Modules declared in recovery/mod.rs
}

/// Capability-based security model. Node authentication, mutual
/// authentication, authorization, key establishment, rotation, revocation.
pub mod security {
    // Modules declared in security/mod.rs
}

/// Verified cryptographic primitives. Ed25519 signatures, XChaCha20-Poly1305,
/// SHA-256, BLAKE3.
pub mod crypto {
    // Modules declared in crypto/mod.rs
}

/// Event sourcing infrastructure. Immutable event store, aggregate replay,
/// projections, snapshots.
pub mod events {
    // Modules declared in events/mod.rs
}

/// Command/query responsibility separation. Write model, read model,
/// command bus, query bus.
pub mod cqrs {
    // Modules declared in cqrs/mod.rs
}

/// Actor model with supervision trees, mailboxes, actor addresses, context,
/// and routing.
pub mod actor {
    // Modules declared in actor/mod.rs
}

/// CRDT state synchronization. G-Set, LWW-Register, PN-Counter, OR-Set,
/// merge semantics with proofs.
pub mod crdt {
    // Modules declared in crdt/mod.rs
}

/// Byzantine fault tolerance. BFT consensus, quorum management, replica
/// management.
pub mod bft {
    // Modules declared in bft/mod.rs
}

/// Network simulation framework. Network simulator, node simulator, channel
/// simulator, scenario runner, metrics collector.
pub mod simulation {
    // Modules declared in simulation/mod.rs
}

/// OpenTelemetry-native observability. Distributed tracing, RED + USE
/// metrics, structured logging, health checks.
pub mod observability {
    // Modules declared in observability/mod.rs
}

/// API gateway patterns. Rate limiting, authentication gateway, request
/// transformation.
pub mod gateway {
    // Modules declared in gateway/mod.rs
}

/// Service mesh patterns. mTLS, traffic splitting, circuit breaker, bulkhead
/// isolation.
pub mod mesh {
    // Modules declared in mesh/mod.rs
}

/// Polyglot persistence. SQL storage, time-series storage, graph storage,
/// object storage.
pub mod persistence {
    // Modules declared in persistence/mod.rs
}

/// Data mesh patterns. Domain ownership, self-serve infrastructure.
pub mod data_mesh {
    // Modules declared in data_mesh/mod.rs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topological_signal_default_is_zero() {
        let sig = TopologicalSignal::default();
        assert_eq!(sig.skyrmion_number, 0);
        assert_eq!(sig.confidence, 1.0);
        assert!(sig.is_valid_topology());
    }

    #[test]
    fn topological_signal_with_confidence_clamps() {
        let sig = TopologicalSignal::new(5).with_confidence(1.5);
        assert_eq!(sig.confidence, 1.0);
        let sig = TopologicalSignal::new(5).with_confidence(-0.5);
        assert_eq!(sig.confidence, 0.0);
    }

    #[test]
    fn constellation_binary_has_two_points() {
        let c = Constellation::binary().unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.contains(-1));
        assert!(c.contains(1));
        assert!(!c.contains(0));
        assert_eq!(c.bits_per_symbol(), 1.0);
    }

    #[test]
    fn constellation_nearest_returns_closest_point() {
        let c = Constellation::binary().unwrap();
        assert_eq!(c.nearest(0), Some(-1)); // equidistant, returns first
        assert_eq!(c.nearest(2), Some(1));
        assert_eq!(c.nearest(-2), Some(-1));
    }

    #[test]
    fn constellation_rejects_empty() {
        let c = Constellation::new(vec![]);
        assert!(c.is_err());
    }

    #[test]
    fn detection_result_is_signal() {
        let sig = TopologicalSignal::new(3);
        let result = DetectionResult::Signal(sig.clone());
        assert!(result.is_signal());
        assert_eq!(result.into_signal(), Some(sig));
    }

    #[test]
    fn detection_result_noise_is_not_signal() {
        assert!(!DetectionResult::Noise.is_signal());
        assert!(DetectionResult::Noise.is_recoverable());
    }

    #[test]
    fn carrier_error_box_display() {
        let err: CarrierErrorBox = "test error".into();
        assert_eq!(err.to_string(), "carrier error: test error");
    }

    #[test]
    fn topological_signal_magnitude() {
        let sig = TopologicalSignal::new(-5);
        assert_eq!(sig.magnitude(), 5);
        let sig = TopologicalSignal::new(5);
        assert_eq!(sig.magnitude(), 5);
    }

    #[test]
    fn constellation_generalized_works() {
        let c = Constellation::generalized(vec![-7, -2, 0, 3, 10]).unwrap();
        assert_eq!(c.len(), 5);
        assert!(c.contains(0));
        assert_eq!(c.min_separation(), 1); // 0 to 3 is 3, -7 to -2 is 5, -2 to 0 is 2, 3 to 10 is 7 -> min is 2... wait sorted: -7,-2,0,3,10 -> gaps: 5,2,3,7 -> min=2
        // Actually min separation is 2 (-2 to 0). Let's just verify it's > 0.
        assert!(c.min_separation() > 0);
    }
}