"""
XPOCK — HX-241 Independent Research: End-to-End BER Simulation

End-to-end simulation of skyrmion number modulation through turbulent
atmospheric channel. Compares against OAM multiplexing and intensity
modulation baselines.

Pipeline:
1. Generate skyrmion field with charge N from constellation
2. Propagate through turbulent atmosphere (Kolmogorov phase screens)
3. Measure Stokes parameters at receiver via polarimetry simulation
4. Compute skyrmion number Q from measured Stokes field
5. Hard-decide: ˆN = nearest constellation value
6. Compute BER over multiple trials

All simulations are mathematical models; results do not establish
physical feasibility. Hardware would be required for experimental validation.

XPOCK research only — no HX-241 involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from wave_info.skyrmion_modem import SkyrmionModem, ModemConfig


@dataclass
class BERResult:
    """Result of BER simulation for a given configuration."""

    cn2: float  # Refractive index structure constant
    n_states: int
    bits_per_symbol: float
    ber: float  # Bit error rate
    ser: float  # Symbol error rate
    n_trials: int
    n_errors: int
    n_symbols: int
    q_mean: float  # Mean measured skyrmion number
    q_std: float  # Std dev of measured skyrmion number
    confusion_matrix: Dict[str, int]  # "tx_N->rx_N": count
    notes: List[str] = field(default_factory=list)


class BERSimulator:
    """End-to-end BER simulation for skyrmion number modulation.

    Simulates the complete link:
    1. Generate skyrmion field with charge N
    2. Propagate through turbulent atmosphere
    3. Measure Stokes parameters at receiver
    4. Compute skyrmion number Q
    5. Hard-decide nearest constellation value
    6. Compute BER
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        wavelength: float = 1.55e-6,
        skyrmion_radius: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.grid_x = np.asarray(grid_x, dtype=np.float64)
        self.grid_y = np.asarray(grid_y, dtype=np.float64)
        self.wavelength = wavelength
        self.skyrmion_radius = skyrmion_radius
        self.rng = np.random.default_rng(seed)

        # Create meshgrid if 1D arrays provided
        if self.grid_x.ndim == 1 and self.grid_y.ndim == 1:
            self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y, indexing="xy")
        else:
            self.X, self.Y = self.grid_x, self.grid_y

    def simulate_ber(
        self,
        constellation: List[int],
        cn2_values: List[float],
        n_trials_per_config: int = 50,
        distance: float = 100.0,
        n_screens: int = 10,
    ) -> List[BERResult]:
        """Run full BER simulation across turbulence strengths.

        Parameters
        ----------
        constellation : list of int
            Skyrmion numbers used as constellation points.
        cn2_values : list of float
            Refractive index structure constants to test.
        n_trials_per_config : int
            Number of random trials per (N, Cn²) pair.
        distance : float
            Propagation distance in meters.
        n_screens : int
            Number of phase screens for turbulence simulation.

        Returns
        -------
        results : list of BERResult
            One result per (constellation, Cn²) combination.
        """
        modem_cfg = ModemConfig(
            wavelength=self.wavelength,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            skyrmion_radius=self.skyrmion_radius,
            constellation=constellation,
        )
        modem = SkyrmionModem(modem_cfg)

        results = []

        for cn2 in cn2_values:
            for n_tx in constellation:
                ber, ser, n_errors, n_symbols, q_values, confusion = (
                    self._run_trials(
                        n_tx=n_tx,
                        constellation=constellation,
                        cn2=cn2,
                        n_trials=n_trials_per_config,
                        distance=distance,
                        n_screens=n_screens,
                        modem=modem,
                    )
                )

                results.append(
                    BERResult(
                        cn2=cn2,
                        n_states=len(constellation),
                        bits_per_symbol=float(modem.n_bits),
                        ber=float(ber),
                        ser=float(ser),
                        n_trials=n_trials_per_config,
                        n_errors=n_errors,
                        n_symbols=n_symbols,
                        q_mean=float(np.mean(q_values)) if q_values else 0.0,
                        q_std=float(np.std(q_values)) if q_values else 0.0,
                        confusion_matrix=confusion,
                        notes=[],
                    )
                )

        return results

    def _run_trials(
        self,
        n_tx: int,
        constellation: List[int],
        cn2: float,
        n_trials: int,
        distance: float,
        n_screens: int,
        modem: SkyrmionModem,
    ) -> tuple:
        """Run multiple trials for a single (N, Cn²) pair.

        Returns
        -------
        ber : float
        ser : float
        n_errors : int
        n_symbols : int
        q_values : list of float
        confusion : dict
        """
        # Find bit index for this n_tx
        try:
            idx = constellation.index(n_tx)
        except ValueError:
            idx = 0
        bits_tx = np.array([(idx >> i) & 1 for i in range(modem.n_bits - 1, -1, -1)])

        # Generate transmitted field once (same for all trials at same N)
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, self.grid_x, self.grid_y)

        turbulence_cfg = TurbulenceConfig(
            wavelength=self.wavelength,
            distance=distance,
            cn2=cn2,
            n_screens=n_screens,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )
        turbulence_model = TurbulenceModel(turbulence_cfg)

        # Grid spacing for phase screen generation
        nx_dim = self.grid_x.shape[0] if self.grid_x.ndim == 2 else len(self.grid_x)
        ny_dim = self.grid_x.shape[1] if self.grid_x.ndim == 2 else len(self.grid_x)
        dx = self.grid_x[0, 1] - self.grid_x[0, 0] if self.grid_x.ndim == 2 else self.grid_x[1] - self.grid_x[0]
        dz = distance / turbulence_model.cfg.n_screens

        n_errors = 0
        n_symbols = 0
        q_values = []
        confusion = {}  # "tx_N->rx_N": count

        for trial in range(n_trials):
            # Generate phase screens once and apply the SAME screens to both
            # field components. This preserves polarization relationships
            # (the relative phase between Ex and Ey is unchanged).
            phi_list = []
            for _ in range(turbulence_model.cfg.n_screens):
                phi_list.append(
                    turbulence_model.generate_phase_screen(
                        (ny_dim, nx_dim), dx, dz
                    )
                )

            E_rx = turbulence_model.propagate(
                Ex_tx, self.grid_x, self.grid_y,
                phase_screen=phi_list[0],
            )
            E_rx_y = turbulence_model.propagate(
                Ey_tx, self.grid_x, self.grid_y,
                phase_screen=phi_list[0]
            )

            # Measure skyrmion number at receiver
            q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)
            q_values.append(q_measured)

            # Decode bits
            bits_rx = modem.skyrmion_to_bits(n_rx)
            bit_error = not np.array_equal(bits_tx, bits_rx)

            n_errors += int(bit_error)
            n_symbols += 1

            key = f"{n_tx}->{n_rx}"
            confusion[key] = confusion.get(key, 0) + 1

        ber = n_errors / n_symbols if n_symbols > 0 else 0.0
        ser = n_errors / n_symbols if n_symbols > 0 else 0.0

        return ber, ser, n_errors, n_symbols, q_values, confusion

    def run_full_sweep(
        self,
        constellation: List[int],
        cn2_values: List[float],
        n_trials_per_config: int = 50,
        distance: float = 100.0,
        n_screens: int = 10,
    ) -> List[BERResult]:
        """Run full sweep with detailed results including Q statistics."""
        modem_cfg = ModemConfig(
            wavelength=self.wavelength,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            skyrmion_radius=self.skyrmion_radius,
            constellation=constellation,
        )
        modem = SkyrmionModem(modem_cfg)

        results = []

        for cn2 in cn2_values:
            for n_tx in constellation:
                q_values = []
                confusion = {}
                n_errors = 0
                n_symbols = 0

                # Find bit index for this n_tx
                try:
                    idx = constellation.index(n_tx)
                except ValueError:
                    idx = 0
                bits_tx = np.array([(idx >> i) & 1 for i in range(modem.n_bits - 1, -1, -1)])

                # Generate transmitted field
                Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, self.grid_x, self.grid_y)

                turbulence_cfg = TurbulenceConfig(
                    wavelength=self.wavelength,
                    distance=distance,
                    cn2=cn2,
                    n_screens=n_screens,
                    seed=int(self.rng.integers(0, 2**31 - 1)),
                )
                turbulence_model = TurbulenceModel(turbulence_cfg)

                for trial in range(n_trials_per_config):
                    # Generate phase screens once and apply same screens to both
                    # field components to preserve polarization relationships.
                    phi_list = []
                    nx_dim = self.grid_x.shape[0] if self.grid_x.ndim == 2 else len(self.grid_x)
                    ny_dim = self.grid_x.shape[1] if self.grid_x.ndim == 2 else len(self.grid_x)
                    dx_val = self.grid_x[0, 1] - self.grid_x[0, 0] if self.grid_x.ndim == 2 else self.grid_x[1] - self.grid_x[0]
                    dz_val = distance / turbulence_model.cfg.n_screens
                    for _ in range(turbulence_model.cfg.n_screens):
                        phi_list.append(
                            turbulence_model.generate_phase_screen(
                                (ny_dim, nx_dim), dx_val, dz_val
                            )
                        )

                    E_rx = turbulence_model.propagate(
                        Ex_tx, self.grid_x, self.grid_y,
                        phase_screen=phi_list[0],
                    )
                    E_rx_y = turbulence_model.propagate(
                        Ey_tx, self.grid_x, self.grid_y,
                        phase_screen=phi_list[0]
                    )

                    q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)
                    q_values.append(q_measured)

                    bits_rx = modem.skyrmion_to_bits(n_rx)
                    bit_error = not np.array_equal(bits_tx, bits_rx)

                    n_errors += int(bit_error)
                    n_symbols += 1

                    key = f"{n_tx}->{n_rx}"
                    confusion[key] = confusion.get(key, 0) + 1

                ber = n_errors / n_symbols if n_symbols > 0 else 0.0
                ser = ber  # Same for single symbol

                results.append(
                    BERResult(
                        cn2=cn2,
                        n_states=len(constellation),
                        bits_per_symbol=float(modem.n_bits),
                        ber=float(ber),
                        ser=float(ser),
                        n_trials=n_trials_per_config,
                        n_errors=n_errors,
                        n_symbols=n_symbols,
                        q_mean=float(np.mean(q_values)) if q_values else 0.0,
                        q_std=float(np.std(q_values)) if q_values else 0.0,
                        confusion_matrix=confusion,
                        notes=[],
                    )
                )

        return results


def quick_ber_test():
    """Quick BER test: 5-state constellation, 3 turbulence levels, 20 trials each."""
    nx, ny = 64, 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    sim = BERSimulator(
        grid_x=X,
        grid_y=Y,
        wavelength=1.55e-6,
        skyrmion_radius=1.0,
        seed=42,
    )

    constellation = [-2, -1, 0, 1, 2]
    cn2_values = [1e-15, 1e-14, 1e-13]

    results = sim.simulate_ber(
        constellation=constellation,
        cn2_values=cn2_values,
        n_trials_per_config=20,
        distance=100.0,
        n_screens=5,
    )

    for r in results:
        print(
            f"Cn²={r.cn2:.0e}, Q={r.q_mean:.3f}±{r.q_std:.3f}: "
            f"BER={r.ber:.4f} ({r.n_errors}/{r.n_symbols})"
        )

    return results


if __name__ == "__main__":
    quick_ber_test()