//! Comprehensive error hierarchy for xpock-integration.
//!
//! Every error carries rich context for diagnostics, recovery, and observability.
//! Errors are classified as transient (retry may succeed), fatal (permanent),
//! or recoverable (recovery strategy available).

use std::fmt;

/// Top-level error type for the xpock-integration crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XpockError {
    /// Detection pipeline errors.
    Detection(DetectionError),
    /// Receiver/optical measurement errors.
    Receiver(ReceiverError),
    /// Channel model errors.
    Channel(ChannelError),
    /// Validation/conformance errors.
    Validation(ValidationError),
    /// Integration/adaptor errors.
    Integration(IntegrationError),
    /// Recovery/edge-case errors.
    Recovery(RecoveryError),
    /// Security/authentication/authorization errors.
    Security(SecurityError),
    /// Cryptographic primitive errors.
    Crypto(CryptoError),
    /// Carrier transport errors.
    Carrier(CarrierErrorBox),
    /// Serialization/deserialization errors.
    Serialization(String),
    /// Configuration errors.
    Configuration(String),
    /// Invalid constellation configuration.
    InvalidConstellation(String),
    /// Numerical instability detected.
    NumericalInstability(String),
    /// Insufficient data for a decision.
    InsufficientData(String),
    /// Operation timed out.
    Timeout(String),
    /// Resource exhausted (memory, bandwidth, etc.).
    ResourceExhausted(String),
    /// Internal invariant violation — indicates a bug.
    Internal(String),
}

impl fmt::Display for XpockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XpockError::Detection(e) => write!(f, "detection error: {}", e),
            XpockError::Receiver(e) => write!(f, "receiver error: {}", e),
            XpockError::Channel(e) => write!(f, "channel error: {}", e),
            XpockError::Validation(e) => write!(f, "validation error: {}", e),
            XpockError::Integration(e) => write!(f, "integration error: {}", e),
            XpockError::Recovery(e) => write!(f, "recovery error: {}", e),
            XpockError::Security(e) => write!(f, "security error: {}", e),
            XpockError::Crypto(e) => write!(f, "crypto error: {}", e),
            XpockError::Carrier(e) => write!(f, "carrier error: {}", e),
            XpockError::Serialization(s) => write!(f, "serialization error: {}", s),
            XpockError::Configuration(s) => write!(f, "configuration error: {}", s),
            XpockError::InvalidConstellation(s) => write!(f, "invalid constellation: {}", s),
            XpockError::NumericalInstability(s) => write!(f, "numerical instability: {}", s),
            XpockError::InsufficientData(s) => write!(f, "insufficient data: {}", s),
            XpockError::Timeout(s) => write!(f, "timeout: {}", s),
            XpockError::ResourceExhausted(s) => write!(f, "resource exhausted: {}", s),
            XpockError::Internal(s) => write!(f, "internal error: {}", s),
        }
    }
}

impl std::error::Error for XpockError {}

impl From<CarrierErrorBox> for XpockError {
    fn from(e: CarrierErrorBox) -> Self {
        XpockError::Carrier(e)
    }
}

impl From<String> for XpockError {
    fn from(s: String) -> Self {
        XpockError::Internal(s)
    }
}

impl From<&str> for XpockError {
    fn from(s: &str) -> Self {
        XpockError::Internal(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Detection errors
// ---------------------------------------------------------------------------

/// Errors specific to the topological signal detection pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionError {
    /// No topological structure found in the measurements.
    NoTopologyDetected,
    /// Signal present but corrupted beyond recovery.
    SignalCorrupted(String),
    /// False positive — apparent topology failed validation.
    FalsePositive { q_candidate: i32, reason: String },
    /// Skyrmion number calculation failed due to numerical issues.
    SkyrmionCalculationFailed(String),
    /// Constellation matching failed — no valid point near measured Q.
    ConstellationMismatch { measured: f64, nearest: Option<i32> },
    /// Intensity below masking threshold — unreliable measurement.
    IntensityBelowThreshold { measured: f64, threshold: f64 },
    /// Crosstalk between constellation points too high.
    ExcessiveCrosstalk { crosstalk: f64, max_allowed: f64 },
    /// Demodulation matrix is singular or ill-conditioned.
    DemodulationMatrixIllConditioned { condition_number: f64 },
    /// Insufficient Stokes measurements for reconstruction.
    InsufficientStokesMeasurements { have: usize, need: usize },
}

impl fmt::Display for DetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectionError::NoTopologyDetected => write!(f, "no topology detected"),
            DetectionError::SignalCorrupted(s) => write!(f, "signal corrupted: {}", s),
            DetectionError::FalsePositive { q_candidate, reason } => {
                write!(f, "false positive: q={} ({})", q_candidate, reason)
            }
            DetectionError::SkyrmionCalculationFailed(s) => {
                write!(f, "skyrmion calculation failed: {}", s)
            }
            DetectionError::ConstellationMismatch { measured, nearest } => {
                write!(
                    f,
                    "constellation mismatch: measured={}, nearest={:?}",
                    measured, nearest
                )
            }
            DetectionError::IntensityBelowThreshold { measured, threshold } => {
                write!(
                    f,
                    "intensity below threshold: measured={}, threshold={}",
                    measured, threshold
                )
            }
            DetectionError::ExcessiveCrosstalk { crosstalk, max_allowed } => {
                write!(
                    f,
                    "excessive crosstalk: {} > {}",
                    crosstalk, max_allowed
                )
            }
            DetectionError::DemodulationMatrixIllConditioned { condition_number } => {
                write!(
                    f,
                    "demodulation matrix ill-conditioned: cond={}",
                    condition_number
                )
            }
            DetectionError::InsufficientStokesMeasurements { have, need } => {
                write!(f, "insufficient Stokes measurements: have={}, need={}", have, need)
            }
        }
    }
}

impl std::error::Error for DetectionError {}

// ---------------------------------------------------------------------------
// Receiver errors
// ---------------------------------------------------------------------------

/// Errors from the optical receiver / Stokes polarimetry subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiverError {
    /// Polarimeter calibration is invalid or expired.
    CalibrationInvalid { reason: String },
    /// Demodulation matrix not calibrated.
    DemodulationMatrixNotCalibrated,
    /// Photon count below shot-noise limit.
    PhotonCountTooLow { measured: u64, minimum: u64 },
    /// Background noise exceeds acceptable level.
    BackgroundNoiseTooHigh { measured: f64, max: f64 },
    /// Polarization state cannot be reconstructed.
    PolarizationReconstructionFailed(String),
    /// Receiver saturation detected.
    ReceiverSaturation { measured: f64, max: f64 },
    /// Dead pixels or detector element failure.
    DetectorElementFailure { element_ids: Vec<usize> },
    /// Wavelength outside calibrated range.
    WavelengthOutOfRange { measured_nm: f64, range_nm: (f64, f64) },
}

impl fmt::Display for ReceiverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReceiverError::CalibrationInvalid { reason } => {
                write!(f, "calibration invalid: {}", reason)
            }
            ReceiverError::DemodulationMatrixNotCalibrated => {
                write!(f, "demodulation matrix not calibrated")
            }
            ReceiverError::PhotonCountTooLow { measured, minimum } => {
                write!(f, "photon count too low: {} < {}", measured, minimum)
            }
            ReceiverError::BackgroundNoiseTooHigh { measured, max } => {
                write!(f, "background noise too high: {} > {}", measured, max)
            }
            ReceiverError::PolarizationReconstructionFailed(s) => {
                write!(f, "polarization reconstruction failed: {}", s)
            }
            ReceiverError::ReceiverSaturation { measured, max } => {
                write!(f, "receiver saturation: {} > {}", measured, max)
            }
            ReceiverError::DetectorElementFailure { element_ids } => {
                write!(f, "detector element failure: {:?}", element_ids)
            }
            ReceiverError::WavelengthOutOfRange { measured_nm, range_nm } => {
                write!(
                    f,
                    "wavelength out of range: {} nm not in {:?} nm",
                    measured_nm, range_nm
                )
            }
        }
    }
}

impl std::error::Error for ReceiverError {}

// ---------------------------------------------------------------------------
// Channel errors
// ---------------------------------------------------------------------------

/// Errors from the FSO channel model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// Turbulence strength outside modeled range.
    TurbulenceOutOfRange { cn2: f64, range: (f64, f64) },
    /// Phase screen generation failed.
    PhaseScreenGenerationFailed(String),
    /// Propagation distance invalid.
    InvalidPropagationDistance { meters: f64 },
    /// Wavelength invalid for channel model.
    InvalidWavelength { nm: f64 },
    /// Rytov variance computation failed.
    RytovVarianceFailed(String),
    /// Gamma-Gamma fitting failed.
    GammaGammaFitFailed(String),
    /// Multiple phase screen mismatch.
    PhaseScreenMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelError::TurbulenceOutOfRange { cn2, range } => {
                write!(f, "turbulence out of range: {} not in {:?}", cn2, range)
            }
            ChannelError::PhaseScreenGenerationFailed(s) => {
                write!(f, "phase screen generation failed: {}", s)
            }
            ChannelError::InvalidPropagationDistance { meters } => {
                write!(f, "invalid propagation distance: {} m", meters)
            }
            ChannelError::InvalidWavelength { nm } => {
                write!(f, "invalid wavelength: {} nm", nm)
            }
            ChannelError::RytovVarianceFailed(s) => {
                write!(f, "rytov variance computation failed: {}", s)
            }
            ChannelError::GammaGammaFitFailed(s) => {
                write!(f, "gamma-gamma fit failed: {}", s)
            }
            ChannelError::PhaseScreenMismatch { expected, actual } => {
                write!(
                    f,
                    "phase screen mismatch: expected {}, actual {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for ChannelError {}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

/// Errors from HX-241 contract validation and conformance checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Contract I-CARR violation.
    IcarrViolation { rule: String, detail: String },
    /// Contract I-TRANS violation.
    ItransViolation { rule: String, detail: String },
    /// Schema conformance failure.
    SchemaConformanceFailed { schema: String, detail: String },
    /// Integrity check failed (hash mismatch).
    IntegrityCheckFailed { expected: String, actual: String },
    /// Missing required field in contract data.
    MissingRequiredField { field: String },
    /// Value outside contract-specified range.
    ValueOutOfRange { field: String, value: String, range: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::IcarrViolation { rule, detail } => {
                write!(f, "I-CARR violation {}: {}", rule, detail)
            }
            ValidationError::ItransViolation { rule, detail } => {
                write!(f, "I-TRANS violation {}: {}", rule, detail)
            }
            ValidationError::SchemaConformanceFailed { schema, detail } => {
                write!(f, "schema {} conformance failed: {}", schema, detail)
            }
            ValidationError::IntegrityCheckFailed { expected, actual } => {
                write!(
                    f,
                    "integrity check failed: expected {}, actual {}",
                    expected, actual
                )
            }
            ValidationError::MissingRequiredField { field } => {
                write!(f, "missing required field: {}", field)
            }
            ValidationError::ValueOutOfRange { field, value, range } => {
                write!(f, "value out of range: {}={} not in {}", field, value, range)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

// ---------------------------------------------------------------------------
// Integration errors
// ---------------------------------------------------------------------------

/// Errors from HX-241 integration and adapter layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationError {
    /// XPOCK Python output could not be adapted to Rust types.
    AdaptationFailed { source: String, target: String, reason: String },
    /// Data flow graph is invalid or disconnected.
    InvalidDataFlow { node: String, reason: String },
    /// Security boundary violation.
    SecurityBoundaryViolated { boundary: String, detail: String },
    /// Carrier adapter not available for given carrier type.
    CarrierAdapterUnavailable { carrier_type: String },
    /// Serialization format not supported.
    UnsupportedSerializationFormat { format: String },
    /// Interface version mismatch.
    InterfaceVersionMismatch { expected: String, actual: String },
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::AdaptationFailed {
                source,
                target,
                reason,
            } => write!(
                f,
                "adaptation failed: {} -> {} ({})",
                source, target, reason
            ),
            IntegrationError::InvalidDataFlow { node, reason } => {
                write!(f, "invalid data flow at {}: {}", node, reason)
            }
            IntegrationError::SecurityBoundaryViolated { boundary, detail } => {
                write!(f, "security boundary {} violated: {}", boundary, detail)
            }
            IntegrationError::CarrierAdapterUnavailable { carrier_type } => {
                write!(f, "carrier adapter unavailable: {}", carrier_type)
            }
            IntegrationError::UnsupportedSerializationFormat { format } => {
                write!(f, "unsupported serialization format: {}", format)
            }
            IntegrationError::InterfaceVersionMismatch { expected, actual } => {
                write!(
                    f,
                    "interface version mismatch: expected {}, actual {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for IntegrationError {}

// ---------------------------------------------------------------------------
// Recovery errors
// ---------------------------------------------------------------------------

/// Errors from edge-case handling and recovery subsystems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryError {
    /// Synchronization failure — nodes out of sync.
    SynchronizationFailure { node_a: String, node_b: String, drift_ms: u64 },
    /// Edge case not handled by any registered strategy.
    UnhandledEdgeCase { case_type: String, detail: String },
    /// Numerical instability cannot be corrected.
    UncorrectableNumericalInstability { operation: String, detail: String },
    /// Environmental extreme beyond operational envelope.
    EnvironmentalExtreme { condition: String, measured: f64, limit: f64 },
    /// Recovery strategy exhausted all attempts.
    RecoveryExhausted { strategy: String, attempts: u32 },
    /// Malformed input rejected.
    MalformedInput { field: String, reason: String },
}

impl fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecoveryError::SynchronizationFailure {
                node_a,
                node_b,
                drift_ms,
            } => write!(
                f,
                "synchronization failure: {} <-> {} drift={}ms",
                node_a, node_b, drift_ms
            ),
            RecoveryError::UnhandledEdgeCase { case_type, detail } => {
                write!(f, "unhandled edge case {}: {}", case_type, detail)
            }
            RecoveryError::UncorrectableNumericalInstability { operation, detail } => {
                write!(f, "uncorrectable numerical instability in {}: {}", operation, detail)
            }
            RecoveryError::EnvironmentalExtreme {
                condition,
                measured,
                limit,
            } => {
                write!(
                    f,
                    "environmental extreme {}: measured={}, limit={}",
                    condition, measured, limit
                )
            }
            RecoveryError::RecoveryExhausted { strategy, attempts } => {
                write!(
                    f,
                    "recovery exhausted: {} after {} attempts",
                    strategy, attempts
                )
            }
            RecoveryError::MalformedInput { field, reason } => {
                write!(f, "malformed input {}: {}", field, reason)
            }
        }
    }
}

impl std::error::Error for RecoveryError {}

// ---------------------------------------------------------------------------
// Security errors
// ---------------------------------------------------------------------------

/// Errors from the capability-based security model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityError {
    /// Node authentication failed.
    AuthenticationFailed { node: String, reason: String },
    /// Mutual authentication handshake failed.
    MutualAuthFailed { initiator: String, responder: String },
    /// Authorization denied — capability token insufficient.
    AuthorizationDenied { capability: String, operation: String },
    /// Key establishment failed.
    KeyEstablishmentFailed { reason: String },
    /// Key rotation failed.
    KeyRotationFailed { key_id: String, reason: String },
    /// Key revocation failed.
    KeyRevocationFailed { key_id: String, reason: String },
    /// Capability token expired or invalid.
    InvalidCapabilityToken { reason: String },
    /// Replay attack detected.
    ReplayAttackDetected { nonce: String, timestamp: u64 },
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::AuthenticationFailed { node, reason } => {
                write!(f, "authentication failed for {}: {}", node, reason)
            }
            SecurityError::MutualAuthFailed {
                initiator,
                responder,
            } => write!(
                f,
                "mutual auth failed: {} <-> {}",
                initiator, responder
            ),
            SecurityError::AuthorizationDenied {
                capability,
                operation,
            } => {
                write!(
                    f,
                    "authorization denied: capability {} for {}",
                    capability, operation
                )
            }
            SecurityError::KeyEstablishmentFailed { reason } => {
                write!(f, "key establishment failed: {}", reason)
            }
            SecurityError::KeyRotationFailed { key_id, reason } => {
                write!(f, "key rotation failed for {}: {}", key_id, reason)
            }
            SecurityError::KeyRevocationFailed { key_id, reason } => {
                write!(f, "key revocation failed for {}: {}", key_id, reason)
            }
            SecurityError::InvalidCapabilityToken { reason } => {
                write!(f, "invalid capability token: {}", reason)
            }
            SecurityError::ReplayAttackDetected { nonce, timestamp } => {
                write!(f, "replay attack detected: nonce={} ts={}", nonce, timestamp)
            }
        }
    }
}

impl std::error::Error for SecurityError {}

// ---------------------------------------------------------------------------
// Crypto errors
// ---------------------------------------------------------------------------

/// Errors from cryptographic primitives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// Signature verification failed.
    SignatureVerificationFailed,
    /// Signature generation failed.
    SignatureGenerationFailed(String),
    /// Encryption failed.
    EncryptionFailed(String),
    /// Decryption failed.
    DecryptionFailed(String),
    /// Key derivation failed.
    KeyDerivationFailed(String),
    /// Hash computation failed.
    HashComputationFailed(String),
    /// Invalid key format or length.
    InvalidKeyFormat { expected: usize, actual: usize },
    /// Nonce/replay counter exhausted.
    NonceExhausted,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::SignatureVerificationFailed => {
                write!(f, "signature verification failed")
            }
            CryptoError::SignatureGenerationFailed(s) => {
                write!(f, "signature generation failed: {}", s)
            }
            CryptoError::EncryptionFailed(s) => write!(f, "encryption failed: {}", s),
            CryptoError::DecryptionFailed(s) => write!(f, "decryption failed: {}", s),
            CryptoError::KeyDerivationFailed(s) => write!(f, "key derivation failed: {}", s),
            CryptoError::HashComputationFailed(s) => write!(f, "hash computation failed: {}", s),
            CryptoError::InvalidKeyFormat { expected, actual } => {
                write!(f, "invalid key format: expected {} bytes, got {}", expected, actual)
            }
            CryptoError::NonceExhausted => write!(f, "nonce exhausted"),
        }
    }
}

impl std::error::Error for CryptoError {}

// ---------------------------------------------------------------------------
// Recovery hints and classification
// ---------------------------------------------------------------------------

/// Classification of whether an error is transient, fatal, or recoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Transient error — retry may succeed.
    Transient,
    /// Fatal error — operation cannot succeed, caller should abort.
    Fatal,
    /// Recoverable error — a recovery strategy exists.
    Recoverable,
}

/// Trait for errors that can be classified by severity and recovery hint.
pub trait ErrorClassification {
    /// Returns the severity classification of this error.
    fn severity(&self) -> ErrorSeverity;

    /// Returns a human-readable recovery hint, if any.
    fn recovery_hint(&self) -> Option<&str>;
}

impl ErrorClassification for XpockError {
    fn severity(&self) -> ErrorSeverity {
        match self {
            XpockError::Detection(DetectionError::NoTopologyDetected) => ErrorSeverity::Transient,
            XpockError::Detection(DetectionError::SignalCorrupted(_)) => ErrorSeverity::Recoverable,
            XpockError::Detection(DetectionError::FalsePositive { .. }) => ErrorSeverity::Transient,
            XpockError::Detection(DetectionError::SkyrmionCalculationFailed(_)) => {
                ErrorSeverity::Recoverable
            }
            XpockError::Detection(DetectionError::ConstellationMismatch { .. }) => {
                ErrorSeverity::Transient
            }
            XpockError::Detection(DetectionError::IntensityBelowThreshold { .. }) => {
                ErrorSeverity::Transient
            }
            XpockError::Detection(DetectionError::ExcessiveCrosstalk { .. }) => {
                ErrorSeverity::Recoverable
            }
            XpockError::Detection(DetectionError::DemodulationMatrixIllConditioned { .. }) => {
                ErrorSeverity::Recoverable
            }
            XpockError::Detection(DetectionError::InsufficientStokesMeasurements { .. }) => {
                ErrorSeverity::Transient
            }
            XpockError::Receiver(ReceiverError::CalibrationInvalid { .. }) => ErrorSeverity::Fatal,
            XpockError::Receiver(ReceiverError::DemodulationMatrixNotCalibrated) => {
                ErrorSeverity::Fatal
            }
            XpockError::Receiver(ReceiverError::PhotonCountTooLow { .. }) => ErrorSeverity::Transient,
            XpockError::Receiver(ReceiverError::BackgroundNoiseTooHigh { .. }) => {
                ErrorSeverity::Transient
            }
            XpockError::Receiver(ReceiverError::PolarizationReconstructionFailed(_)) => {
                ErrorSeverity::Recoverable
            }
            XpockError::Receiver(ReceiverError::ReceiverSaturation { .. }) => ErrorSeverity::Transient,
            XpockError::Receiver(ReceiverError::DetectorElementFailure { .. }) => {
                ErrorSeverity::Fatal
            }
            XpockError::Receiver(ReceiverError::WavelengthOutOfRange { .. }) => {
                ErrorSeverity::Fatal
            }
            XpockError::Channel(_) => ErrorSeverity::Transient,
            XpockError::Validation(_) => ErrorSeverity::Fatal,
            XpockError::Integration(_) => ErrorSeverity::Recoverable,
            XpockError::Recovery(_) => ErrorSeverity::Recoverable,
            XpockError::Security(_) => ErrorSeverity::Fatal,
            XpockError::Crypto(_) => ErrorSeverity::Fatal,
            XpockError::Carrier(_) => ErrorSeverity::Transient,
            XpockError::Serialization(_) => ErrorSeverity::Fatal,
            XpockError::Configuration(_) => ErrorSeverity::Fatal,
            XpockError::InvalidConstellation(_) => ErrorSeverity::Fatal,
            XpockError::NumericalInstability(_) => ErrorSeverity::Recoverable,
            XpockError::InsufficientData(_) => ErrorSeverity::Transient,
            XpockError::Timeout(_) => ErrorSeverity::Transient,
            XpockError::ResourceExhausted(_) => ErrorSeverity::Recoverable,
            XpockError::Internal(_) => ErrorSeverity::Fatal,
        }
    }

    fn recovery_hint(&self) -> Option<&str> {
        match self {
            XpockError::Detection(DetectionError::SignalCorrupted(_)) => {
                Some("Retry with increased averaging or intensity masking")
            }
            XpockError::Detection(DetectionError::ExcessiveCrosstalk { .. }) => {
                Some("Increase constellation point separation or reduce modulation order")
            }
            XpockError::Detection(DetectionError::DemodulationMatrixIllConditioned { .. }) => {
                Some("Recalibrate polarimeter demodulation matrix")
            }
            XpockError::Receiver(ReceiverError::PhotonCountTooLow { .. }) => {
                Some("Increase integration time or transmitter power")
            }
            XpockError::Receiver(ReceiverError::BackgroundNoiseTooHigh { .. }) => {
                Some("Apply background subtraction or narrowband filtering")
            }
            XpockError::Integration(IntegrationError::AdaptationFailed { .. }) => {
                Some("Verify XPOCK output format matches expected schema")
            }
            XpockError::Integration(IntegrationError::CarrierAdapterUnavailable { .. }) => {
                Some("Register a carrier adapter for the carrier type")
            }
            XpockError::Recovery(RecoveryError::SynchronizationFailure { .. }) => {
                Some("Initiate clock resynchronization procedure")
            }
            XpockError::Recovery(RecoveryError::RecoveryExhausted { .. }) => {
                Some("Escalate to manual intervention or failover node")
            }
            XpockError::Carrier(_) => Some("Retry carrier operation after backoff"),
            XpockError::Timeout(_) => Some("Retry with increased timeout"),
            XpockError::ResourceExhausted(_) => Some("Free resources or scale down load"),
            _ => None,
        }
    }
}

impl ErrorClassification for DetectionError {
    fn severity(&self) -> ErrorSeverity {
        match self {
            DetectionError::NoTopologyDetected => ErrorSeverity::Transient,
            DetectionError::SignalCorrupted(_) => ErrorSeverity::Recoverable,
            DetectionError::FalsePositive { .. } => ErrorSeverity::Transient,
            DetectionError::SkyrmionCalculationFailed(_) => ErrorSeverity::Recoverable,
            DetectionError::ConstellationMismatch { .. } => ErrorSeverity::Transient,
            DetectionError::IntensityBelowThreshold { .. } => ErrorSeverity::Transient,
            DetectionError::ExcessiveCrosstalk { .. } => ErrorSeverity::Recoverable,
            DetectionError::DemodulationMatrixIllConditioned { .. } => ErrorSeverity::Recoverable,
            DetectionError::InsufficientStokesMeasurements { .. } => ErrorSeverity::Transient,
        }
    }

    fn recovery_hint(&self) -> Option<&str> {
        match self {
            DetectionError::SignalCorrupted(_) => {
                Some("Retry with increased averaging or intensity masking")
            }
            DetectionError::ExcessiveCrosstalk { .. } => {
                Some("Increase constellation point separation")
            }
            DetectionError::DemodulationMatrixIllConditioned { .. } => {
                Some("Recalibrate polarimeter demodulation matrix")
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type alias
// ---------------------------------------------------------------------------

/// Convenient result type for xpock-integration operations.
pub type XpockResult<T> = Result<T, XpockError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xpock_error_display() {
        let err = XpockError::Detection(DetectionError::NoTopologyDetected);
        assert_eq!(err.to_string(), "detection error: no topology detected");
    }

    #[test]
    fn error_severity_classification() {
        assert_eq!(
            XpockError::Detection(DetectionError::NoTopologyDetected).severity(),
            ErrorSeverity::Transient
        );
        assert_eq!(
            XpockError::Security(SecurityError::AuthenticationFailed {
                node: "test".into(),
                reason: "bad key".into(),
            }).severity(),
            ErrorSeverity::Fatal
        );
        assert_eq!(
            XpockError::Detection(DetectionError::SignalCorrupted("test".into())).severity(),
            ErrorSeverity::Recoverable
        );
    }

    #[test]
    fn recovery_hint_present() {
        let err = XpockError::Detection(DetectionError::SignalCorrupted("test".into()));
        assert_eq!(
            err.recovery_hint(),
            Some("Retry with increased averaging or intensity masking")
        );
        assert_eq!(
            XpockError::Security(SecurityError::AuthenticationFailed {
                node: "n".into(),
                reason: "r".into(),
            }).recovery_hint(),
            None
        );
    }

    #[test]
    fn constellation_mismatch_display() {
        let err = DetectionError::ConstellationMismatch {
            measured: 2.5,
            nearest: Some(3),
        };
        assert!(err.to_string().contains("2.5"));
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn false_positive_display() {
        let err = DetectionError::FalsePositive {
            q_candidate: 7,
            reason: "intensity below masking threshold".into(),
        };
        assert!(err.to_string().contains("7"));
        assert!(err.to_string().contains("false positive"));
    }

    #[test]
    fn from_string_conversion() {
        let err: XpockError = "something went wrong".into();
        assert!(matches!(err, XpockError::Internal(_)));
    }

    #[test]
    fn carrier_error_box_conversion() {
        let boxed: CarrierErrorBox = "carrier down".into();
        let err: XpockError = boxed.into();
        assert!(matches!(err, XpockError::Carrier(_)));
    }
}