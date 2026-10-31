"""Core wave field and event data structures.

This module defines the fundamental data types used throughout XPOCK:
- WaveField: the underlying complex field (scalar or vector) on a grid
- WaveMode: enumeration of supported wave mode families
- PolarizationState: polarization description (Jones vector, Mueller, etc.)
- WaveEvent: a complete generated wave event with parameters, field, metadata

All fields are finite-support (windowed) physical wave events, not infinite
plane waves. This is essential for modelling real, detectable signals.

Design notes:
- Fields are represented on uniform Cartesian grids (x, y, z, t)
- Complex-valued fields encode amplitude and phase natively
- Vector fields (Ex, Ey, Ez) support full polarization modeling
- Every event carries a complete parameter record for reproducibility
- Numerical validity checks are built into field construction
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
import numpy as np
import json
from datetime import datetime, timezone
import uuid


class WaveMode(Enum):
    """Families of spatial/temporal wave modes supported by the generator."""

    GAUSSIAN = "gaussian"              # Fundamental Gaussian mode
    HERMITE_GAUSSIAN = "hermite_gaussian"   # HG_nm modes
    LAGUERRE_GAUSSIAN = "laguerre_gaussian" # LG_pl modes (carries OAM)
    BESSEL = "bessel"                  # Bessel beams (non-diffracting ideal)
    SUPER_GAUSSIAN = "super_gaussian"  # Flattened temporal/spatial profile
    SECH = "sech"                      # Hyperbolic secant pulse (soliton-like)
    PLANE_WAVE = "plane_wave"          # Windowed plane wave
    SPHERICAL = "spherical"            # Spherical wave (finite aperture)
    NONSEPARABLE = "nonseparable"      # Space-time non-factorizable packet
    CUSTOM = "custom"                  # User-defined field


class PolarizationState(Enum):
    """Polarization descriptions supported."""

    LINEAR = "linear"                  # Linear polarization at angle theta
    CIRCULAR = "circular"              # Left/right circular
    ELLIPTICAL = "elliptical"          # General elliptical
    RADIAL = "radial"                  # Radially varying (vector beam)
    AZIMUTHAL = "azimuthal"            # Azimuthally varying (vector beam)
    SPATIALLY_VARYING = "spatially_varying"  # Jones vector varies over (x,y)
    SKYRMIONIC = "skyrmionic"          # Skyrmion polarization texture
    UNPOLARIZED = "unpolarized"        # Partially polarized / mixed state
    CUSTOM_JONES = "custom_jones"      # Arbitrary Jones vector field


@dataclass
class WaveField:
    """Container for a complex-valued wave field on a spatiotemporal grid.

    Attributes
    ----------
    Ex, Ey, Ez : np.ndarray, optional
        Complex electric field components. At least one must be provided.
        Shape: (nz, ny, nx, nt) or (ny, nx, nt) for 2D+time.
    x, y, z, t : np.ndarray
        Coordinate arrays defining the grid.
    field_type : str
        "scalar" if only one component, "vector" if multiple.
    metadata : dict
        Additional free-form metadata.
    """

    Ex: Optional[np.ndarray] = None
    Ey: Optional[np.ndarray] = None
    Ez: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    field_type: str = "scalar"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field consistency after construction."""
        components = [c for c in (self.Ex, self.Ey, self.Ez) if c is not None]
        if not components:
            raise ValueError("At least one field component (Ex/Ey/Ez) must be provided.")

        # Determine field type
        n_components = sum(1 for c in (self.Ex, self.Ey, self.Ez) if c is not None)
        self.field_type = "scalar" if n_components == 1 else "vector"

        # Validate shapes match
        ref_shape = components[0].shape
        for i, comp in enumerate(components):
            if comp.shape != ref_shape:
                raise ValueError(
                    f"Field component {i} shape {comp.shape} does not match reference {ref_shape}"
                )

        # Validate coordinates if provided
        if self.t is not None and self.t.shape[-1] != ref_shape[-1]:
            raise ValueError(
                f"Time axis length {self.t.shape[-1]} does not match field time dimension {ref_shape[-1]}"
            )

        # Check for NaN/Inf
        for name, comp in [("Ex", self.Ex), ("Ey", self.Ey), ("Ez", self.Ez)]:
            if comp is not None:
                if not np.all(np.isfinite(comp)):
                    raise ValueError(f"Field component {name} contains NaN or Inf values.")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the field data."""
        comp = self.Ex if self.Ex is not None else (self.Ey if self.Ey is not None else self.Ez)
        return comp.shape

    @property
    def intensity(self) -> np.ndarray:
        """Total intensity |Ex|^2 + |Ey|^2 + |Ez|^2."""
        I = np.zeros_like(self.Ex if self.Ex is not None else self.Ey, dtype=np.float64)
        for comp in (self.Ex, self.Ey, self.Ez):
            if comp is not None:
                I = I + np.abs(comp) ** 2
        return I

    @property
    def total_power(self) -> float:
        """Integrated power over the full spatiotemporal volume."""
        if self.x is None or self.y is None or self.t is None:
            raise ValueError("Spatial and temporal coordinates required for power integration.")
        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        dy = self.y[1] - self.y[0] if len(self.y) > 1 else 1.0
        dz = self.z[1] - self.z[0] if self.z is not None and len(self.z) > 1 else 1.0
        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 1.0
        return float(np.sum(self.intensity) * dx * dy * dz * dt)

    @property
    def duration(self) -> Optional[float]:
        """Temporal duration of the grid."""
        if self.t is None or len(self.t) < 2:
            return None
        return float(self.t[-1] - self.t[0])

    @property
    def spatial_extent(self) -> Optional[Tuple[float, float, Optional[float]]]:
        """(x_extent, y_extent, z_extent_or_None)."""
        if self.x is None or self.y is None:
            return None
        x_ext = float(self.x[-1] - self.x[0])
        y_ext = float(self.y[-1] - self.y[0])
        z_ext = float(self.z[-1] - self.z[0]) if self.z is not None and len(self.z) > 1 else None
        return (x_ext, y_ext, z_ext)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary (arrays as lists)."""
        return {
            "field_type": self.field_type,
            "shape": self.shape,
            "metadata": self.metadata,
            "total_power": self.total_power,
            "duration": self.duration,
            "spatial_extent": self.spatial_extent,
        }


@dataclass
class WaveEvent:
    """A complete generated wave event.

    A WaveEvent bundles:
    - The parameter configuration used to generate it
    - The resulting WaveField
    - Computed measurable properties (filled in by WaveAnalyzer)
    - Numerical validity checks
    - Output data references

    Attributes
    ----------
    event_id : str
        Unique identifier (UUID4).
    timestamp : str
        ISO-8601 UTC timestamp of generation.
    parameters : dict
        Full parameter configuration used for generation.
    field : WaveField
        The generated field data.
    properties : dict
        Measurable properties (computed post-generation).
    validation : dict
        Numerical validity check results.
    notes : str
        Free-form researcher notes.
    """

    event_id: str
    timestamp: str
    parameters: Dict[str, Any]
    field: WaveField
    properties: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(
        cls,
        parameters: Dict[str, Any],
        field: WaveField,
        notes: str = "",
    ) -> "WaveEvent":
        """Factory: create a new WaveEvent with auto-generated ID and timestamp."""
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters=parameters,
            field=field,
            notes=notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "field": self.field.to_dict(),
            "properties": self.properties,
            "validation": self.validation,
            "notes": self.notes,
        }

    def to_json(self, path: Union[str, "PathLike"], indent: int = 2) -> None:
        """Write the event to a JSON file."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)

    @classmethod
    def from_json(cls, path: Union[str, "PathLike"]) -> "WaveEvent":
        """Load a WaveEvent from a JSON file (field arrays reconstructed).

        Uses object.__new__ to bypass WaveField.__post_init__ validation,
        since JSON does not store the full field arrays.
        """
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct field without __post_init__ validation (JSON does not
        # store full arrays). Field metadata is preserved.
        field = object.__new__(WaveField)
        field.field_type = data["field"]["field_type"]
        field.metadata = data["field"].get("metadata", {})
        field.Ex = None
        field.Ey = None
        field.Ez = None
        field.x = None
        field.y = None
        field.z = None
        field.t = None

        return cls(
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            parameters=data["parameters"],
            field=field,
            properties=data.get("properties", {}),
            validation=data.get("validation", {}),
            notes=data.get("notes", ""),
        )


# Type alias for path-like
from os import PathLike
