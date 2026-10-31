"""Parameter space exploration.

Systematically varies individual parameters and combines multiple values
across supported properties. Supports:
- Configurable parameter ranges and resolution
- Systematic parameter sweeps (1D, 2D, N-D grid)
- Random sampling (Latin hypercube-style)
- Progressive refinement of promising regions
- No arbitrary small limits on configuration count (bounded by memory/time)

Design notes:
- ParameterSpace generates a list of parameter dicts, one per configuration
- Each config is passed to the wave generator to produce a WaveEvent
- SweepConfig defines a single parameter's sweep: linear, log, or custom values
- Combinations are Cartesian products across parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Iterator, Union, Callable
from enum import Enum
import numpy as np
import itertools


class SweepType(Enum):
    """How a parameter's values are generated."""

    LINEAR = "linear"       # np.linspace(start, stop, num)
    LOG = "log"             # np.logspace(start, stop, num) (base 10)
    CUSTOM = "custom"       # Explicit list of values
    FUNCTION = "function"   # Callable(index) -> value


@dataclass
class SweepConfig:
    """Definition of a sweep over a single parameter.

    Attributes
    ----------
    name : str
        Parameter key in the configuration dictionary.
    sweep_type : SweepType
    start, stop : float
        Range endpoints (for LINEAR/LOG).
    num : int
        Number of samples.
    values : list, optional
        Explicit values (for CUSTOM).
    fn : callable, optional
        Callable(i) -> value (for FUNCTION).
    """

    name: str
    sweep_type: SweepType = SweepType.LINEAR
    start: Optional[float] = None
    stop: Optional[float] = None
    num: int = 10
    values: Optional[List[Any]] = None
    fn: Optional[Callable[[int], Any]] = None

    def generate(self) -> List[Any]:
        """Generate the list of values for this parameter."""
        if self.sweep_type == SweepType.LINEAR:
            if self.start is None or self.stop is None:
                raise ValueError(f"LINEAR sweep requires start and stop for {self.name}")
            return list(np.linspace(self.start, self.stop, self.num))
        if self.sweep_type == SweepType.LOG:
            if self.start is None or self.stop is None:
                raise ValueError(f"LOG sweep requires start and stop for {self.name}")
            return list(np.logspace(self.start, self.stop, self.num, base=10.0))
        if self.sweep_type == SweepType.CUSTOM:
            if self.values is None:
                raise ValueError(f"CUSTOM sweep requires values for {self.name}")
            return list(self.values)
        if self.sweep_type == SweepType.FUNCTION:
            if self.fn is None:
                raise ValueError(f"FUNCTION sweep requires fn for {self.name}")
            return [self.fn(i) for i in range(self.num)]
        raise ValueError(f"Unknown sweep type: {self.sweep_type}")


@dataclass
class ParameterSpace:
    """N-dimensional parameter space for systematic wave exploration.

    Combines multiple SweepConfig objects into a Cartesian product of
    configurations. Each configuration is a dict suitable for passing to
    the wave generator.

    Attributes
    ----------
    sweeps : list of SweepConfig
    fixed_params : dict
        Parameters held constant across all configurations.
    max_combinations : int
        Safety cap to prevent runaway memory use (default 1_000_000).
    """

    sweeps: List[SweepConfig]
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    max_combinations: int = 1_000_000

    def __post_init__(self) -> None:
        total = 1
        for s in self.sweeps:
            if s.sweep_type == SweepType.CUSTOM and s.values:
                total *= len(s.values)
            else:
                total *= s.num
        if total > self.max_combinations:
            raise ValueError(
                f"Parameter space size {total} exceeds max_combinations {self.max_combinations}. "
                "Reduce resolution or increase max_combinations intentionally."
            )

    @property
    def size(self) -> int:
        """Total number of configurations in the Cartesian product."""
        total = 1
        for s in self.sweeps:
            if s.sweep_type == SweepType.CUSTOM and s.values:
                total *= len(s.values)
            else:
                total *= s.num
        return total

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield parameter dicts, one per configuration."""
        value_lists = [s.generate() for s in self.sweeps]
        names = [s.name for s in self.sweeps]
        for combo in itertools.product(*value_lists):
            params = dict(self.fixed_params)
            params.update(dict(zip(names, combo)))
            yield params

    def __len__(self) -> int:
        return self.size

    def sample_random(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Draw n_samples random configurations from the space.

        Uses simple random sampling (not Latin hypercube) to keep
        dependencies minimal. Good for exploratory sweeps.
        """
        rng = np.random.default_rng(seed)
        value_lists = [s.generate() for s in self.sweeps]
        names = [s.name for s in self.sweeps]
        samples: List[Dict[str, Any]] = []
        for _ in range(n_samples):
            combo = [rng.choice(vlist) for vlist in value_lists]
            params = dict(self.fixed_params)
            params.update(dict(zip(names, combo)))
            samples.append(params)
        return samples

    def refine_region(
        self,
        center_params: Dict[str, Any],
        relative_width: float = 0.2,
        num_per_dim: int = 5,
    ) -> "ParameterSpace":
        """Create a refined sub-space around a promising configuration.

        For each numeric parameter in center_params, creates a linear sweep
        spanning [center*(1-width), center*(1+width)]. Non-numeric params
        are held fixed.

        Useful for progressively finer exploration of promising regions.
        """
        new_sweeps: List[SweepConfig] = []
        new_fixed: Dict[str, Any] = dict(self.fixed_params)

        for name, center in center_params.items():
            if isinstance(center, (int, float, np.integer, np.floating)) and not isinstance(center, bool):
                c = float(center)
                half = c * relative_width
                new_sweeps.append(
                    SweepConfig(
                        name=name,
                        sweep_type=SweepType.LINEAR,
                        start=c - half,
                        stop=c + half,
                        num=num_per_dim,
                    )
                )
            else:
                new_fixed[name] = center

        # Keep any sweeps not being refined as fixed at their first value
        for s in self.sweeps:
            if s.name not in center_params:
                vals = s.generate()
                new_fixed[s.name] = vals[0] if vals else None

        return ParameterSpace(
            sweeps=new_sweeps,
            fixed_params=new_fixed,
            max_combinations=self.max_combinations,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sweeps": [
                {
                    "name": s.name,
                    "sweep_type": s.sweep_type.value,
                    "start": s.start,
                    "stop": s.stop,
                    "num": s.num,
                    "values": s.values,
                }
                for s in self.sweeps
            ],
            "fixed_params": self.fixed_params,
            "max_combinations": self.max_combinations,
            "size": self.size,
        }