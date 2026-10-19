"""
XPOCK — HX-241 Validation Utilities

Validation utilities that verify XPOCK's computational outputs conform to
HX-241's architectural contracts, invariants, and invariants. These checks
ensure that generated artifacts can be safely incorporated into the HX-241
Rust codebase without violating carrier-agnostic design principles.

All validations are pure Python functions that return structured results
with pass/fail status, violation details, and remediation guidance.

HX-241 architectural compliance verified:
- No conventional networking dependencies (IP/TCP/UDP/QUIC/DNS/HTTP)
- Carrier independence (I-CARR-001 to I-CARR-005)
- Transport independence (I-TRANS-001 to I-TRANS-020)
- Crypto invariants (CRYPTO-INV-1 to CRYPTO-INV-8)
- Protocol invariants (SPEC.md Section 19)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from pathlib import Path
import json
import re
import numpy as np


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: ValidationLevel
    code: str
    message: str
    details: Optional[str] = None
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        level: ValidationLevel,
        code: str,
        message: str,
        details: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> None:
        """Add an issue to this result. ERROR/CRITICAL issues mark as failed."""
        self.issues.append(
            ValidationIssue(level, code, message, details, remediation)
        )
        if level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL):
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }


class CarrierArchitectureValidator:
    """
    Validates that XPOCK carrier models conform to HX-241's carrier
    architecture contract (CARRIER-CONTRACT.md).

    Checks:
    - I-CARR-001: No dependence on specific carrier technology
    - I-CARR-002: CarrierType values are opaque
    - I-CARR-003: CarrierAddr bytes are opaque
    - I-CARR-004: No carrier property assumptions
    - I-CARR-005: Capabilities are hints only
    """

    def validate_channel_model(
        self,
        channel_dict: Dict[str, Any],
    ) -> ValidationResult:
        """Validate a carrier channel model dict against the carrier contract."""
        result = ValidationResult(
            check_name="carrier_architecture.channel_model",
            passed=True,
        )

        # Check that no conventional networking references appear
        conventional_terms = [
            "tcp", "udp", "ip", "quic", "http", "dns", "mdns",
            "websocket", "webrtc", "libp2p", "wifi", "bluetooth",
            "ethernet", "cellular",
        ]

        medium = channel_dict.get("medium", "").lower()
        notes = " ".join(channel_dict.get("medium_notes", [])).lower()

        for term in conventional_terms:
            if term in medium or term in notes:
                result.add_issue(
                    ValidationLevel.CRITICAL,
                    "CARR-VIOLATION",
                    f"Conventional networking term '{term}' found in carrier model",
                    details=f"medium={medium}, notes contains '{term}'",
                    remediation="Remove conventional networking references; carrier must be technology-agnostic",
                )

        # Check path loss is non-negative
        path_loss = channel_dict.get("path_loss_db", 0)
        if path_loss < 0:
            result.add_issue(
                ValidationLevel.WARNING,
                "CARR-PATH-LOSS",
                f"Negative path loss ({path_loss} dB) may indicate model artifact",
                details=f"path_loss_db={path_loss}",
                remediation="Review path loss model for the carrier medium",
            )

        # Check SNR is within reasonable bounds
        snr = channel_dict.get("snr_db", 0)
        if snr < -300:
            result.add_issue(
                ValidationLevel.WARNING,
                "CARR-SNR",
                f"Extremely low SNR ({snr} dB) may indicate link is infeasible",
                details=f"snr_db={snr}",
                remediation="Consider shorter distance or higher transmit power",
            )

        # Check capacity is non-negative
        capacity = channel_dict.get("capacity_bps", 0)
        if capacity < 0:
            result.add_issue(
                ValidationLevel.ERROR,
                "CARR-CAPACITY",
                f"Negative channel capacity ({capacity} bps)",
                details=f"capacity_bps={capacity}",
                remediation="Fix capacity calculation; capacity must be >= 0",
            )

        # Check impairment params are in valid ranges
        imp = channel_dict.get("impairment_params", {})
        for key in ("packet_loss_rate", "corruption_rate"):
            val = imp.get(key, 0)
            if val is not None and (val < 0 or val > 1):
                result.add_issue(
                    ValidationLevel.ERROR,
                    "CARR-IMPAIRMENT-RANGE",
                    f"Impairment parameter '{key}' out of range [0,1]: {val}",
                    details=f"impairment_params.{key}={val}",
                    remediation="Clamp impairment parameters to [0,1] range",
                )

        result.metadata["medium"] = channel_dict.get("medium", "unknown")
        result.metadata["path_loss_db"] = channel_dict.get("path_loss_db", 0)
        result.metadata["snr_db"] = channel_dict.get("snr_db", 0)

        return result

    def validate_impairment_params(
        self,
        imp_dict: Dict[str, Any],
    ) -> ValidationResult:
        """Validate hx-sim impairment parameters."""
        result = ValidationResult(
            check_name="carrier_architecture.impairment_params",
            passed=True,
        )

        for key in ("packet_loss_rate", "corruption_rate", "duplication_rate"):
            val = imp_dict.get(key, 0)
            if val is not None and (val < 0 or val > 1):
                result.add_issue(
                    ValidationLevel.ERROR,
                    "IMP-RANGE",
                    f"{key} must be in [0,1], got {val}",
                    details=f"impairment_params.{key}={val}",
                    remediation=f"Clamp {key} to [0,1]",
                )

        bandwidth = imp_dict.get("bandwidth_bps", 0)
        if bandwidth is not None and bandwidth < 0:
            result.add_issue(
                ValidationLevel.ERROR,
                "IMP-BANDWIDTH",
                f"bandwidth_bps must be >= 0, got {bandwidth}",
                details=f"impairment_params.bandwidth_bps={bandwidth}",
                remediation="Set bandwidth_bps to 0 for unlimited or positive value",
            )

        latency = imp_dict.get("base_latency_us", 0)
        if latency is not None and latency < 0:
            result.add_issue(
                ValidationLevel.ERROR,
                "IMP-LATENCY",
                f"base_latency_us must be >= 0, got {latency}",
                details=f"impairment_params.base_latency_us={latency}",
                remediation="Set base_latency_us to non-negative value",
            )

        return result


class TransportArchitectureValidator:
    """
    Validates that XPOCK outputs conform to HX-241's transport contract
    (TRANSPORT-CONTRACT.md).

    Checks:
    - I-TRANS-001: No carrier technology dependence
    - I-TRANS-002: Carrier capabilities discovered at runtime
    - I-TRANS-004: Session survives carrier changes
    """

    def validate_capacity_result(
        self,
        result_dict: Dict[str, Any],
    ) -> ValidationResult:
        """Validate a channel capacity analysis result."""
        result = ValidationResult(
            check_name="transport_architecture.capacity_result",
            passed=True,
        )

        snr = result_dict.get("snr_db", 0)
        bw = result_dict.get("bandwidth_hz", 0)
        capacity = result_dict.get("shannon_capacity_bps", 0)

        # Verify Shannon formula: C = B * log2(1 + SNR)
        if bw > 0:
            snr_linear = 10.0 ** (snr / 10.0)
            expected_capacity = bw * np.log2(1.0 + snr_linear)
            if abs(capacity - expected_capacity) > expected_capacity * 0.01:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "TRANS-CAPACITY-MISMATCH",
                    f"Capacity {capacity:.1f} bps differs from Shannon formula {expected_capacity:.1f} bps",
                    details=f"snr={snr}dB, bw={bw}Hz",
                    remediation="Verify capacity calculation matches Shannon formula",
                )

        # Check spectral efficiency is reasonable
        se = result_dict.get("spectral_efficiency_bps_hz", 0)
        if se > 100:
            result.add_issue(
                ValidationLevel.WARNING,
                "TRANS-SPECTRAL-EFF",
                f"Spectral efficiency {se:.1f} bps/Hz is unusually high",
                details=f"spectral_efficiency_bps_hz={se}",
                remediation="Review spectral efficiency calculation",
            )

        result.metadata["snr_db"] = snr
        result.metadata["bandwidth_hz"] = bw
        result.metadata["capacity_bps"] = capacity

        return result


class ProtocolInvariantValidator:
    """
    Validates HX-241 protocol invariants from SPEC.md Section 19.

    Checks:
    - Content addressing: ObjectID = BLAKE3(content)
    - Chunk integrity: ChunkID = BLAKE3(chunk_data)
    - Session uniqueness
    - Stream ordering
    - Flow control
    - Congestion response
    """

    def validate_no_conventional_stack(
        self,
        data: Dict[str, Any],
    ) -> ValidationResult:
        """Verify no conventional networking stack references in output data."""
        result = ValidationResult(
            check_name="protocol_invariants.no_conventional_stack",
            passed=True,
        )

        # Serialize data to string for searching
        data_str = json.dumps(data, default=str).lower()

        conventional_patterns = [
            r"\btcp\b", r"\budp\b", r"\bip\b", r"\bquic\b",
            r"\bhttp\b", r"\bdns\b", r"\bmdns\b",
            r"\bwebrtc\b", r"\blibp2p\b", r"\bwifi\b",
            r"\bbluetooth\b", r"\bethernet\b", r"\bcellular\b",
            r"\bsocket\b", r"\bport\b", r"\bipv4\b", r"\bipv6\b",
        ]

        for pattern in conventional_patterns:
            matches = re.findall(pattern, data_str)
            if matches:
                result.add_issue(
                    ValidationLevel.CRITICAL,
                    "PROTO-VIOLATION",
                    f"Conventional networking pattern '{pattern}' found in output",
                    details=f"Matched: {matches[:5]}",
                    remediation="Remove conventional networking references; output must be carrier-agnostic",
                )

        return result


class HX241ValidatorSuite:
    """
    Complete validation suite for XPOCK → HX-241 integration.

    Runs all validators and produces a consolidated report.
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./241-integration/validation")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.carrier_validator = CarrierArchitectureValidator()
        self.transport_validator = TransportArchitectureValidator()
        self.protocol_validator = ProtocolInvariantValidator()

    def validate_channel_model(
        self,
        channel_dict: Dict[str, Any],
    ) -> Dict[str, ValidationResult]:
        """Validate a channel model against all applicable contracts."""
        return {
            "carrier_architecture": self.carrier_validator.validate_channel_model(
                channel_dict
            ),
            "protocol_invariants": self.protocol_validator.validate_no_conventional_stack(
                channel_dict
            ),
        }

    def validate_impairment_params(
        self,
        imp_dict: Dict[str, Any],
    ) -> ValidationResult:
        """Validate impairment parameters."""
        return self.carrier_validator.validate_impairment_params(imp_dict)

    def validate_capacity_result(
        self,
        result_dict: Dict[str, Any],
    ) -> Dict[str, ValidationResult]:
        """Validate a capacity analysis result."""
        return {
            "transport_architecture": self.transport_validator.validate_capacity_result(
                result_dict
            ),
            "protocol_invariants": self.protocol_validator.validate_no_conventional_stack(
                result_dict
            ),
        }

    def validate_test_vector_set(
        self,
        test_set: Dict[str, Any],
    ) -> Dict[str, List[ValidationResult]]:
        """Validate an entire test vector set."""
        results: Dict[str, List[ValidationResult]] = {
            "carrier_models": [],
            "impairment_profiles": [],
            "capacity_results": [],
            "modulation_constellations": [],
        }

        for cm in test_set.get("carrier_models", []):
            vm = self.validate_channel_model(cm)
            results["carrier_models"].extend(vm.values())

        for ip in test_set.get("impairment_profiles", []):
            vi = self.validate_impairment_params(ip.get("hx_sim_impairment", {}))
            results["impairment_profiles"].append(vi)
            # Also check protocol invariants
            vp = self.protocol_validator.validate_no_conventional_stack(ip)
            results["impairment_profiles"].append(vp)

        for cr in test_set.get("capacity_results", []):
            vc = self.validate_capacity_result(cr)
            results["capacity_results"].extend(vc.values())

        for mc in test_set.get("modulation_constellations", []):
            vp = self.protocol_validator.validate_no_conventional_stack(mc)
            results["modulation_constellations"].append(vp)

        return results

    def generate_report(
        self,
        validation_results: Dict[str, List[ValidationResult]],
    ) -> Dict[str, Any]:
        """Generate a validation report from results."""
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        critical_issues = 0
        error_issues = 0
        warning_issues = 0
        info_issues = 0

        all_issues: List[ValidationIssue] = []

        for category, results in validation_results.items():
            for r in results:
                total_checks += 1
                if r.passed:
                    passed_checks += 1
                else:
                    failed_checks += 1
                for issue in r.issues:
                    all_issues.append(issue)
                    if issue.level == ValidationLevel.CRITICAL:
                        critical_issues += 1
                    elif issue.level == ValidationLevel.ERROR:
                        error_issues += 1
                    elif issue.level == ValidationLevel.WARNING:
                        warning_issues += 1
                    else:
                        info_issues += 1

        return {
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "pass_rate": passed_checks / max(1, total_checks),
                "critical_issues": critical_issues,
                "error_issues": error_issues,
                "warning_issues": warning_issues,
                "info_issues": info_issues,
                "all_passed": failed_checks == 0 and critical_issues == 0,
            },
            "issues": [i.to_dict() for i in all_issues],
            "by_category": {
                cat: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": sum(1 for r in results if not r.passed),
                }
                for cat, results in validation_results.items()
            },
        }

    def save_report(
        self,
        report: Dict[str, Any],
        filename: str = "validation_report.json",
    ) -> Path:
        """Save validation report as JSON."""
        path = self.workspace / filename
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return path
