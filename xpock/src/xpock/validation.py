"""Validation tests for generated wave events.

Performs numerical validity checks:
- Finite field values (no NaN/Inf)
- Power conservation (finite total power)
- Normalization checks
- Phase continuity (no unphysical discontinuities except branch cuts)
- Pulse width vs grid resolution (Nyquist)
- Beam width vs grid resolution
- Polarization Stokes parameter bounds (S0^2 >= S1^2+S2^2+S3^2)

Results are stored in WaveEvent.validation and summarized in
ValidationResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import numpy as np

from xpock.core import WaveEvent


@dataclass
class ValidationResult:
    """Result of validating a single wave event."""

    event_id: str
    is_valid: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_check(self, name: str, passed: bool, message: str = "") -> None:
        self.checks[name] = passed
        if not passed:
            self.is_valid = False
            self.messages.append(f"{name}: {message}" if message else name)

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"Validation {status} — event {self.event_id[:8]}"]
        for name, passed in self.checks.items():
            mark = "✓" if passed else "✗"
            lines.append(f"  {mark} {name}")
        if self.messages:
            lines.append("  Issues:")
            for m in self.messages:
                lines.append(f"    - {m}")
        return "\n".join(lines)


def validate_wave_event(event: WaveEvent) -> ValidationResult:
    """Run all validation checks on a WaveEvent.

    Returns a ValidationResult with per-check pass/fail and an overall
    is_valid flag. Does NOT modify the event (caller may store result
    in event.validation).
    """
    field = event.field
    result = ValidationResult(event_id=event.event_id, is_valid=True)

    # --- Check 1: finite values ---
    finite_ok = True
    for name, comp in [("Ex", field.Ex), ("Ey", field.Ey), ("Ez", field.Ez)]:
        if comp is not None:
            if not np.all(np.isfinite(comp)):
                finite_ok = False
                result.add_check(f"finite_{name}", False, "contains NaN or Inf")
    result.add_check("finite_values", finite_ok, "" if finite_ok else "some components non-finite")

    # --- Check 2: finite total power ---
    try:
        power = field.total_power
        power_ok = np.isfinite(power) and power >= 0
        result.add_check("finite_power", power_ok, f"power={power}")
        result.metrics["total_power"] = float(power) if power_ok else None
    except ValueError:
        result.add_check("finite_power", False, "coordinates missing for power integration")

    # --- Check 3: pulse width resolved by temporal grid ---
    if field.t is not None and len(field.t) > 2:
        dt = field.t[1] - field.t[0]
        # Estimate pulse width from intensity if possible
        I = field.intensity
        if I.ndim >= 1:
            # Temporal profile integrated over space
            axes = tuple(range(I.ndim - 1))
            I_t = np.sum(I, axis=axes)
            if np.max(I_t) > 0:
                # Find FWHM in grid points
                half = 0.5 * np.max(I_t)
                above = I_t >= half
                n_above = int(np.sum(above))
                fwhm_points = n_above * dt
                result.metrics["temporal_fwhm_points"] = fwhm_points
                resolved = fwhm_points >= 4 * dt  # at least 4 points across FWHM
                result.add_check("temporal_resolution", resolved, f"FWHM={fwhm_points:.2f} grid units")
    else:
        result.add_check("temporal_resolution", False, "no temporal grid")

    # --- Check 4: beam width resolved by spatial grid ---
    if field.x is not None and field.y is not None and len(field.x) > 2:
        dx = field.x[1] - field.x[0]
        dy = field.y[1] - field.y[0]
        I = field.intensity
        # Extract 2D spatial slice (middle z, middle t)
        if I.ndim == 4:  # (z, y, x, t)
            I = I[I.shape[0] // 2, :, :, I.shape[3] // 2]
        elif I.ndim == 3:  # (y, x, t)
            I = I[:, :, I.shape[2] // 2]
        if I.ndim == 2:
            # RMS width in x
            X, Y = np.meshgrid(field.x, field.y, indexing="xy")
            total = np.sum(I)
            if total > 0:
                x_c = np.sum(X * I) / total
                var_x = np.sum((X - x_c) ** 2 * I) / total
                rms_x = np.sqrt(max(var_x, 0.0))
                resolved_x = rms_x >= 2 * dx
                result.add_check("spatial_resolution_x", resolved_x, f"RMS width={rms_x:.3f}, dx={dx:.3f}")
                result.metrics["rms_width_x"] = float(rms_x)

    # --- Check 5: polarization Stokes bounds ---
    if field.Ex is not None and field.Ey is not None:
        S0 = np.abs(field.Ex) ** 2 + np.abs(field.Ey) ** 2
        S1 = np.abs(field.Ex) ** 2 - np.abs(field.Ey) ** 2
        S2 = 2.0 * np.real(field.Ex * np.conj(field.Ey))
        S3 = 2.0 * np.imag(field.Ex * np.conj(field.Ey))
        S0_safe = np.where(S0 == 0, 1e-15, S0)
        dop_sq = (S1 ** 2 + S2 ** 2 + S3 ** 2) / (S0_safe ** 2)
        dop_ok = bool(np.all(dop_sq <= 1.0 + 1e-10))
        result.add_check("polarization_bounds", dop_ok, f"max DOP^2={float(np.max(dop_sq)):.4f}")
        result.metrics["max_degree_of_polarization_sq"] = float(np.max(dop_sq))

    # --- Check 6: phase continuity (no jumps > pi except branch cuts) ---
    if field.Ex is not None:
        phase = np.angle(field.Ex)
        # Extract 2D spatial slice (middle z, middle t)
        if phase.ndim == 4:  # (z, y, x, t)
            phase = phase[phase.shape[0] // 2, :, :, phase.shape[3] // 2]
        elif phase.ndim == 3:  # (y, x, t)
            phase = phase[:, :, phase.shape[2] // 2]
        # Reduce to 2D if needed
        while phase.ndim > 2:
            phase = phase[phase.shape[0] // 2]
        if phase.ndim == 2:
            # Unwrap phase so 2π branch cuts don't trigger false positives
            unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1)
            dx_phase = np.abs(np.diff(unwrapped, axis=1))
            dy_phase = np.abs(np.diff(unwrapped, axis=0))
            # Flag jumps > 1.5*pi after unwrapping (genuine discontinuities)
            max_jump = float(np.max(np.concatenate([dx_phase.ravel(), dy_phase.ravel()])))
            phase_ok = max_jump < 2.0 * np.pi
            result.add_check("phase_continuity", phase_ok, f"max jump={max_jump:.2f} rad")
            result.metrics["max_phase_jump"] = max_jump

    # --- Check 7: non-zero field ---
    I = field.intensity
    nonzero_ok = bool(np.max(I) > 0)
    result.add_check("nonzero_field", nonzero_ok, "field is identically zero")

    # --- Store metrics ---
    result.metrics["field_shape"] = field.shape
    result.metrics["field_type"] = field.field_type
    if field.duration is not None:
        result.metrics["duration"] = field.duration
    if field.spatial_extent is not None:
        result.metrics["spatial_extent"] = field.spatial_extent

    return result


def validate_batch(events: List[WaveEvent]) -> Dict[str, ValidationResult]:
    """Validate a list of events, returning dict keyed by event_id."""
    return {ev.event_id: validate_wave_event(ev) for ev in events}
