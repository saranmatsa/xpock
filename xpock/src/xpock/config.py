"""Configuration management for XPOCK.

Loads/saves YAML configuration files that define:
- Grid parameters (spatial/temporal extents, resolution)
- Wave generation parameters (temporal profile, spatial mode, polarization)
- Parameter sweep definitions
- Analysis and visualization options

Configuration is strongly typed via dataclasses but serialized to YAML
for human readability and researcher-friendly editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class GridConfig:
    """Spatiotemporal grid definition."""

    x_min: float = -5.0
    x_max: float = 5.0
    nx: int = 128
    y_min: float = -5.0
    y_max: float = 5.0
    ny: int = 128
    z_min: float = 0.0
    z_max: float = 1.0
    nz: int = 1
    t_min: float = -5.0
    t_max: float = 5.0
    nt: int = 128

    def build_arrays(self) -> Dict[str, np.ndarray]:
        import numpy as np

        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        z = np.linspace(self.z_min, self.z_max, self.nz)
        t = np.linspace(self.t_min, self.t_max, self.nt)
        return {"x": x, "y": y, "z": z, "t": t}


@dataclass
class TemporalConfig:
    """Temporal pulse configuration."""

    shape: str = "gaussian"
    amplitude: float = 1.0
    center: float = 0.0
    duration: float = 1.0
    chirp_rate: float = 0.0
    carrier_freq: float = 0.0


@dataclass
class SpatialConfig:
    """Spatial mode configuration."""

    mode: str = "gaussian"
    n: int = 0
    m: int = 0
    p: int = 0
    l: int = 0
    waist: float = 1.0
    wavelength: float = 1.0
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    aperture_radius: Optional[float] = None


@dataclass
class PolarizationConfigYAML:
    """Polarization configuration."""

    state: str = "linear"
    angle: float = 0.0
    handedness: int = 1
    skyrmion_radius: float = 1.0
    skyrmion_charge: int = 1


@dataclass
class TopologyConfig:
    """Topological charge configuration."""

    charge_type: str = "winding"
    charge: int = 1
    core_radius: float = 0.1


@dataclass
class CoherenceConfigYAML:
    """Coherence configuration."""

    coherence_type: str = "full"
    coherence_time: float = 1.0
    coherence_length: float = 1.0
    spectral_width: float = 0.1


@dataclass
class SweepConfigYAML:
    """Parameter sweep definition for YAML."""

    name: str
    sweep_type: str = "linear"
    start: Optional[float] = None
    stop: Optional[float] = None
    num: int = 10
    values: Optional[List[Any]] = None


@dataclass
class XpockConfig:
    """Top-level XPOCK configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    polarization: PolarizationConfigYAML = field(default_factory=PolarizationConfigYAML)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    coherence: CoherenceConfigYAML = field(default_factory=CoherenceConfigYAML)
    sweeps: List[SweepConfigYAML] = field(default_factory=list)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "./xpock_output"
    generate_visualizations: bool = True
    save_events: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "XpockConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XpockConfig":
        grid = GridConfig(**data.get("grid", {}))
        temporal = TemporalConfig(**data.get("temporal", {}))
        spatial = SpatialConfig(**data.get("spatial", {}))
        polarization = PolarizationConfigYAML(**data.get("polarization", {}))
        topology = TopologyConfig(**data.get("topology", {}))
        coherence = CoherenceConfigYAML(**data.get("coherence", {}))
        sweeps = [SweepConfigYAML(**s) for s in data.get("sweeps", [])]

        return cls(
            grid=grid,
            temporal=temporal,
            spatial=spatial,
            polarization=polarization,
            topology=topology,
            coherence=coherence,
            sweeps=sweeps,
            fixed_params=data.get("fixed_params", {}),
            output_dir=data.get("output_dir", "./xpock_output"),
            generate_visualizations=data.get("generate_visualizations", True),
            save_events=data.get("save_events", True),
        )

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "grid": asdict(self.grid),
            "temporal": asdict(self.temporal),
            "spatial": asdict(self.spatial),
            "polarization": asdict(self.polarization),
            "topology": asdict(self.topology),
            "coherence": asdict(self.coherence),
            "sweeps": [asdict(s) for s in self.sweeps],
            "fixed_params": self.fixed_params,
            "output_dir": self.output_dir,
            "generate_visualizations": self.generate_visualizations,
            "save_events": self.save_events,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid": asdict(self.grid),
            "temporal": asdict(self.temporal),
            "spatial": asdict(self.spatial),
            "polarization": asdict(self.polarization),
            "topology": asdict(self.topology),
            "coherence": asdict(self.coherence),
            "sweeps": [asdict(s) for s in self.sweeps],
            "fixed_params": self.fixed_params,
        }


class Config:
    """Convenience wrapper around XpockConfig with helper methods."""

    def __init__(self, config: Optional[XpockConfig] = None):
        self._config = config or XpockConfig()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return Config(XpockConfig.from_yaml(path))

    def save(self, path: str | Path) -> None:
        self._config.to_yaml(path)

    @property
    def grid(self) -> GridConfig:
        return self._config.grid

    @property
    def temporal(self) -> TemporalConfig:
        return self._config.temporal

    @property
    def spatial(self) -> SpatialConfig:
        return self._config.spatial

    @property
    def polarization(self) -> PolarizationConfigYAML:
        return self._config.polarization

    @property
    def topology(self) -> TopologyConfig:
        return self._config.topology

    @property
    def coherence(self) -> CoherenceConfigYAML:
        return self._config.coherence

    @property
    def sweeps(self) -> List[SweepConfigYAML]:
        return self._config.sweeps

    @property
    def fixed_params(self) -> Dict[str, Any]:
        return self._config.fixed_params

    @property
    def output_dir(self) -> str:
        return self._config.output_dir

    @property
    def generate_visualizations(self) -> bool:
        return self._config.generate_visualizations

    @property
    def save_events(self) -> bool:
        return self._config.save_events

    def build_grid_arrays(self) -> Dict[str, np.ndarray]:
        return self._config.grid.build_arrays()
