"""
XPOCK — HX-241 Wave-Structure Modulation

Abstract modulation layer for HX-241: maps HX-241 frame data to physical
wave structure properties and back. This is carrier-agnostic — the same
abstraction works for optical, RF, acoustic, magnetic induction, or any
other physical carrier.

HX-241 architectural principle preserved: this module defines an abstract
interface. HX-241 protocol logic does not depend on any specific modulation
(QAM, OFDM, PSK, wave-structure, etc.). Wave-structure modulation is one
possible implementation among many.

Modulation dimensions (independently controllable wave properties):
- Temporal: pulse shape, duration, chirp rate, carrier frequency
- Spatial: mode (Gaussian, HG, LG/OAM, Bessel), waist, position
- Polarization: linear, circular, elliptical, radial, azimuthal, skyrmionic
- Topological: winding number, OAM charge
- Coherence: full, partial, incoherent
- Coupled modes: coupling coefficient, phase mismatch

Each dimension can encode information. The total information capacity is
the product of distinguishable states across all dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np

from xpock.core import WaveEvent, WaveField, WaveMode, PolarizationState
from xpock.properties import (
    TemporalProfile, PulseShape,
    SpatialMode, SpatialModeConfig,
    PolarizationTexture, PolarizationConfig,
    TopologicalCharge, TopologicalChargeConfig,
    CoherenceModel, CoherenceConfig,
    CoupledMode, CoupledModeConfig,
)
from xpock.cli import generate_wave_event
from xpock.analysis import WaveAnalyzer


class ModulationDimension(Enum):
    """Wave properties that can be modulated for information encoding."""

    TEMPORAL_SHAPE = "temporal_shape"        # Gaussian, sech, super-Gaussian, etc.
    TEMPORAL_DURATION = "temporal_duration"  # Pulse width
    TEMPORAL_CHIRP = "temporal_chirp_rate"   # Quadratic phase chirp
    CARRIER_FREQUENCY = "carrier_frequency"  # Optical/RF carrier
    SPATIAL_MODE = "spatial_mode"            # Gaussian, HG, LG, Bessel
    OAM_CHARGE = "oam_charge"                # Topological charge l
    POLARIZATION_STATE = "polarization_state" # Linear, circular, elliptical, etc.
    POLARIZATION_ANGLE = "polarization_angle" # Linear polarization angle
    BEAM_WAIST = "beam_waist"                # Spatial beam width
    COHERENCE_TYPE = "coherence_type"        # Full, partial, incoherent
    TOPOLOGICAL_CHARGE = "topological_charge" # Additional vortex charge


@dataclass
class ModulationScheme:
    """Defines how HX-241 frame bits map to wave structure properties.

    Each dimension has a list of distinguishable states. The number of
    bits encodable per dimension is log2(num_states). The total bits per
    wave structure is the sum across all modulated dimensions.

    Attributes
    ----------
    dimensions : dict
        Maps ModulationDimension -> list of state values.
    enabled_dimensions : list
        Which dimensions are actively used for modulation.
    bits_per_dimension : dict
        Computed bits per dimension (log2 of num states).
    total_bits_per_symbol : int
        Sum of bits across all enabled dimensions.
    """

    dimensions: Dict[ModulationDimension, List[Any]] = field(default_factory=dict)
    enabled_dimensions: List[ModulationDimension] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bits_per_dimension: Dict[ModulationDimension, float] = {}
        self.total_bits_per_symbol = 0
        self._compute_bits()

    def _compute_bits(self) -> None:
        """Compute bits per dimension and total."""
        self.bits_per_dimension = {}
        self.total_bits_per_symbol = 0
        for dim in self.enabled_dimensions:
            if dim in self.dimensions and len(self.dimensions[dim]) > 0:
                n_states = len(self.dimensions[dim])
                bits = np.log2(n_states)
                self.bits_per_dimension[dim] = float(bits)
                self.total_bits_per_symbol += int(bits)  # floor for integer bits

    def add_dimension(
        self,
        dimension: ModulationDimension,
        states: List[Any],
    ) -> None:
        """Add a modulation dimension with its distinguishable states."""
        self.dimensions[dimension] = list(states)
        if dimension not in self.enabled_dimensions:
            self.enabled_dimensions.append(dimension)
        self._compute_bits()

    def remove_dimension(self, dimension: ModulationDimension) -> None:
        """Remove a modulation dimension."""
        self.dimensions.pop(dimension, None)
        if dimension in self.enabled_dimensions:
            self.enabled_dimensions.remove(dimension)
        self._compute_bits()

    def capacity_bits(self) -> int:
        """Return total bits encodable per wave structure symbol."""
        return self.total_bits_per_symbol

    def num_states(self) -> int:
        """Total number of distinguishable wave structures (product of states)."""
        total = 1
        for dim in self.enabled_dimensions:
            if dim in self.dimensions:
                total *= len(self.dimensions[dim])
        return total


@dataclass
class ModulationResult:
    """Result of modulating HX-241 data onto a wave structure."""

    scheme: ModulationScheme
    event: WaveEvent
    encoded_bits: int
    dimension_assignments: Dict[ModulationDimension, Any]
    constellation_index: int  # Index in the constellation of all possible states
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoded_bits": self.encoded_bits,
            "dimension_assignments": {
                k.value: v for k, v in self.dimension_assignments.items()
            },
            "constellation_index": self.constellation_index,
            "total_bits_per_symbol": self.scheme.capacity_bits(),
            "notes": self.notes,
        }


@dataclass
class DemodulationResult:
    """Result of demodulating a received wave structure back to properties."""

    recovered_dimensions: Dict[ModulationDimension, Any]
    recovered_bits: int
    confidence: Dict[ModulationDimension, float]  # 0-1 confidence per dimension
    demodulation_error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recovered_dimensions": {
                k.value: v for k, v in self.recovered_dimensions.items()
            },
            "recovered_bits": self.recovered_bits,
            "confidence": {k.value: v for k, v in self.confidence.items()},
            "error": self.demodulation_error,
            "notes": self.notes,
        }


class WaveModulator:
    """
    Modulates HX-241 frame data onto wave structure properties.

    The modulator takes a bitstream (from HX-241 frame payload) and maps
    it to a constellation of wave structures. Each symbol (group of bits)
    selects one point in the constellation — a specific combination of
    wave property values.

    This is carrier-agnostic: the same constellation definition works
    regardless of the physical carrier medium. The actual wave generation
    is delegated to XPOCK's generate_wave_event function.
    """

    def __init__(
        self,
        scheme: Optional[ModulationScheme] = None,
        grid_arrays: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.scheme = scheme or ModulationScheme()
        self.grid_arrays = grid_arrays or {}
        self._constellation: List[Dict[ModulationDimension, Any]] = []
        self._build_constellation()
        self.analyzer = WaveAnalyzer()

    def _build_constellation(self) -> None:
        """Build the full constellation of wave structures from dimension states."""
        if not self.scheme.enabled_dimensions:
            self._constellation = [{}]
            return

        dims = self.scheme.enabled_dimensions
        state_lists = [self.scheme.dimensions[d] for d in dims]

        import itertools
        self._constellation = []
        for combo in itertools.product(*state_lists):
            assignment = dict(zip(dims, combo))
            self._constellation.append(assignment)

    def modulate_bits(
        self,
        bits: Union[str, List[int], np.ndarray],
        symbol_index: Optional[int] = None,
    ) -> ModulationResult:
        """
        Modulate a bitstring onto a wave structure.

        Parameters
        ----------
        bits : str, list, or ndarray
            The bitstring to encode. Length must match scheme capacity
            (or be padded/truncated).
        symbol_index : int, optional
            If provided, directly select constellation point by index
            (for testing/deterministic modulation).

        Returns
        -------
        ModulationResult with the generated WaveEvent and assignments.
        """
        # Convert bits to integer
        if isinstance(bits, str):
            bit_val = int(bits, 2) if bits else 0
            n_bits = len(bits)
        elif isinstance(bits, np.ndarray):
            bit_str = "".join(str(int(b)) for b in bits.flatten())
            bit_val = int(bit_str, 2) if bit_str else 0
            n_bits = len(bit_str)
        else:
            bit_str = "".join(str(int(b)) for b in bits)
            bit_val = int(bit_str, 2) if bit_str else 0
            n_bits = len(bit_str)

        capacity = self.scheme.capacity_bits()
        if n_bits > capacity:
            # Truncate to capacity
            bit_val = bit_val & ((1 << capacity) - 1)
            n_bits = capacity

        # Select constellation point
        if symbol_index is not None:
            constellation_idx = symbol_index % len(self._constellation)
        else:
            constellation_idx = bit_val % len(self._constellation)

        assignment = self._constellation[constellation_idx]

        # Generate wave event from assignment
        params = self._assignment_to_params(assignment)
        event = generate_wave_event(params, self.grid_arrays)

        notes = [
            f"Modulated {n_bits} bits onto wave structure",
            f"Constellation size: {len(self._constellation)}",
            f"Selected index: {constellation_idx}",
            f"Dimensions: {[d.value for d in self.scheme.enabled_dimensions]}",
        ]

        return ModulationResult(
            scheme=self.scheme,
            event=event,
            encoded_bits=n_bits,
            dimension_assignments=assignment,
            constellation_index=constellation_idx,
            notes=notes,
        )

    def _assignment_to_params(
        self,
        assignment: Dict[ModulationDimension, Any],
    ) -> Dict[str, Any]:
        """Convert dimension assignments to generate_wave_event parameters."""
        params: Dict[str, Any] = {}

        # Temporal parameters
        if ModulationDimension.TEMPORAL_SHAPE in assignment:
            shape = assignment[ModulationDimension.TEMPORAL_SHAPE]
            params["temporal_shape"] = shape.value if isinstance(shape, Enum) else str(shape)
        if ModulationDimension.TEMPORAL_DURATION in assignment:
            params["temporal_duration"] = float(assignment[ModulationDimension.TEMPORAL_DURATION])
        if ModulationDimension.TEMPORAL_CHIRP in assignment:
            params["temporal_chirp_rate"] = float(assignment[ModulationDimension.TEMPORAL_CHIRP])
        if ModulationDimension.CARRIER_FREQUENCY in assignment:
            params["temporal_carrier_freq"] = 2.0 * np.pi * float(
                assignment[ModulationDimension.CARRIER_FREQUENCY]
            )

        # Spatial parameters
        if ModulationDimension.SPATIAL_MODE in assignment:
            mode = assignment[ModulationDimension.SPATIAL_MODE]
            params["spatial_mode"] = mode.value if isinstance(mode, Enum) else str(mode)
        if ModulationDimension.OAM_CHARGE in assignment:
            params["spatial_l"] = int(assignment[ModulationDimension.OAM_CHARGE])
            params["topology_charge"] = int(assignment[ModulationDimension.OAM_CHARGE])
        if ModulationDimension.BEAM_WAIST in assignment:
            params["spatial_waist"] = float(assignment[ModulationDimension.BEAM_WAIST])

        # Polarization parameters
        if ModulationDimension.POLARIZATION_STATE in assignment:
            state = assignment[ModulationDimension.POLARIZATION_STATE]
            params["polarization_state"] = state.value if isinstance(state, Enum) else str(state)
        if ModulationDimension.POLARIZATION_ANGLE in assignment:
            params["polarization_angle"] = float(assignment[ModulationDimension.POLARIZATION_ANGLE])

        # Coherence
        if ModulationDimension.COHERENCE_TYPE in assignment:
            params["coherence_type"] = str(assignment[ModulationDimension.COHERENCE_TYPE])

        # Topological charge (separate from OAM if both enabled)
        if ModulationDimension.TOPOLOGICAL_CHARGE in assignment:
            params["topology_charge"] = int(assignment[ModulationDimension.TOPOLOGICAL_CHARGE])

        # Set defaults for unset dimensions
        params.setdefault("temporal_shape", "gaussian")
        params.setdefault("temporal_duration", 1.0)
        params.setdefault("temporal_chirp_rate", 0.0)
        params.setdefault("temporal_carrier_freq", 0.0)
        params.setdefault("spatial_mode", "gaussian")
        params.setdefault("spatial_l", 0)
        params.setdefault("spatial_waist", 1.0)
        params.setdefault("spatial_wavelength", 1.0)
        params.setdefault("polarization_state", "linear")
        params.setdefault("polarization_angle", 0.0)
        params.setdefault("topology_charge", 0)
        params.setdefault("coherence_type", "full")

        return params

    def demodulate_event(
        self,
        event: WaveEvent,
    ) -> DemodulationResult:
        """
        Demodulate a received wave structure back to dimension assignments.

        Analyzes the wave event's measurable properties and finds the
        closest constellation point. Returns recovered dimensions with
        confidence scores.
        """
        props = self.analyzer.analyze(event)
        notes: List[str] = []

        if not self._constellation:
            return DemodulationResult(
                recovered_dimensions={},
                recovered_bits=0,
                confidence={},
                demodulation_error="Empty constellation",
                notes=["No modulation dimensions defined"],
            )

        # Build feature vector from event properties
        event_features = self._event_to_feature_vector(event, props)

        # Find closest constellation point
        best_idx = 0
        best_distance = float("inf")
        best_assignment = self._constellation[0]

        for i, assignment in enumerate(self._constellation):
            # Generate reference features for this assignment
            ref_params = self._assignment_to_params(assignment)
            # We can't easily compute features without regenerating,
            # so we compare using the assignment values directly
            distance = self._assignment_distance(assignment, event_features, props)
            if distance < best_distance:
                best_distance = distance
                best_idx = i
                best_assignment = assignment

        # Compute confidence per dimension
        confidence = self._compute_dimension_confidence(
            best_assignment, event_features, props
        )

        # Count recovered bits
        n_dims = len(self.scheme.enabled_dimensions)
        bits_per_dim = self.scheme.capacity_bits() / max(n_dims, 1)
        recovered_bits = int(bits_per_dim * n_dims)

        notes = [
            f"Demodulated wave event {event.event_id[:8]}",
            f"Selected constellation index: {best_idx}",
            f"Distance: {best_distance:.4f}",
            f"Recovered dimensions: {len(best_assignment)}",
        ]

        return DemodulationResult(
            recovered_dimensions=best_assignment,
            recovered_bits=recovered_bits,
            confidence=confidence,
            demodulation_error=None,
            notes=notes,
        )

    def _event_to_feature_vector(
        self,
        event: WaveEvent,
        props,
    ) -> Dict[str, float]:
        """Extract modulation-relevant features from a wave event."""
        features: Dict[str, float] = {}

        features["temporal_duration"] = props.temporal_duration_fwhm or 0.0
        features["bandwidth"] = props.bandwidth_rms or 0.0
        features["beam_waist"] = props.beam_waist_r or 0.0
        features["oam"] = props.oam_per_photon or 0.0
        features["winding_number"] = props.winding_number or 0.0
        features["polarization"] = props.degree_of_polarization or 0.0
        features["total_power"] = props.total_power
        features["schmidt_number"] = props.schmidt_number or 1.0

        return features

    def _assignment_distance(
        self,
        assignment: Dict[ModulationDimension, Any],
        event_features: Dict[str, float],
        props,
    ) -> float:
        """Compute distance between an assignment and observed event features."""
        distance = 0.0
        n_compared = 0

        # Temporal duration
        if ModulationDimension.TEMPORAL_DURATION in assignment:
            expected = float(assignment[ModulationDimension.TEMPORAL_DURATION])
            observed = event_features.get("temporal_duration", 0.0)
            if expected > 0:
                distance += abs(observed - expected) / expected
                n_compared += 1

        # OAM charge
        if ModulationDimension.OAM_CHARGE in assignment:
            expected = float(assignment[ModulationDimension.OAM_CHARGE])
            observed = props.oam_per_photon or 0.0
            distance += abs(observed - expected) / max(abs(expected), 1.0)
            n_compared += 1

        # Polarization angle
        if ModulationDimension.POLARIZATION_ANGLE in assignment:
            expected = float(assignment[ModulationDimension.POLARIZATION_ANGLE])
            observed = props.stokes_s1 or 0.0  # Rough proxy
            distance += abs(observed - expected) / np.pi
            n_compared += 1

        # Beam waist
        if ModulationDimension.BEAM_WAIST in assignment:
            expected = float(assignment[ModulationDimension.BEAM_WAIST])
            observed = event_features.get("beam_waist", 0.0)
            if expected > 0:
                distance += abs(observed - expected) / expected
                n_compared += 1

        return distance / max(n_compared, 1)

    def _compute_dimension_confidence(
        self,
        assignment: Dict[ModulationDimension, Any],
        event_features: Dict[str, float],
        props,
    ) -> Dict[ModulationDimension, float]:
        """Compute confidence (0-1) for each recovered dimension."""
        confidence: Dict[ModulationDimension, float] = {}

        for dim, expected in assignment.items():
            conf = 1.0  # Default high confidence

            if dim == ModulationDimension.TEMPORAL_DURATION:
                observed = event_features.get("temporal_duration", 0.0)
                exp_val = float(expected)
                if exp_val > 0:
                    rel_err = abs(observed - exp_val) / exp_val
                    conf = max(0.0, 1.0 - rel_err)
            elif dim == ModulationDimension.OAM_CHARGE:
                observed = props.oam_per_photon or 0.0
                exp_val = float(expected)
                if exp_val != 0:
                    rel_err = abs(observed - exp_val) / abs(exp_val)
                    conf = max(0.0, 1.0 - rel_err)
                else:
                    conf = 1.0 if abs(observed) < 0.1 else 0.5
            elif dim == ModulationDimension.BEAM_WAIST:
                observed = event_features.get("beam_waist", 0.0)
                exp_val = float(expected)
                if exp_val > 0:
                    rel_err = abs(observed - exp_val) / exp_val
                    conf = max(0.0, 1.0 - rel_err)
            else:
                conf = 0.8  # Default for unmeasured dimensions

            confidence[dim] = float(np.clip(conf, 0.0, 1.0))

        return confidence

    def constellation_size(self) -> int:
        """Return the number of points in the constellation."""
        return len(self._constellation)


# -----------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------

def create_default_scheme() -> ModulationScheme:
    """Create a default modulation scheme with common dimensions."""
    scheme = ModulationScheme()
    scheme.add_dimension(
        ModulationDimension.TEMPORAL_SHAPE,
        [PulseShape.GAUSSIAN, PulseShape.SECH, PulseShape.SUPER_GAUSSIAN],
    )
    scheme.add_dimension(
        ModulationDimension.OAM_CHARGE,
        [0, 1, 2, 3],
    )
    scheme.add_dimension(
        ModulationDimension.POLARIZATION_STATE,
        [PolarizationState.LINEAR, PolarizationState.CIRCULAR],
    )
    scheme.add_dimension(
        ModulationDimension.TEMPORAL_DURATION,
        [0.5, 1.0, 2.0],
    )
    return scheme


def modulate_frame_payload(
    payload_bytes: bytes,
    scheme: ModulationScheme,
    grid_arrays: dict,
    symbol_index: Optional[int] = None,
) -> ModulationResult:
    """
    Convenience: modulate HX-241 frame payload bytes onto a wave structure.

    Converts bytes to bitstring, then modulates using the scheme.
    """
    bit_string = "".join(f"{b:08b}" for b in payload_bytes)
    modulator = WaveModulator(scheme=scheme, grid_arrays=grid_arrays)
    return modulator.modulate_bits(bit_string, symbol_index=symbol_index)


def demodulate_wave_event(
    event: WaveEvent,
    scheme: ModulationScheme,
) -> DemodulationResult:
    """Convenience: demodulate a wave event back to dimension assignments."""
    modulator = WaveModulator(scheme=scheme)
    return modulator.demodulate_event(event)
