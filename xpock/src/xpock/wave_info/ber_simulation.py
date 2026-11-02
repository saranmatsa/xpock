"""
XPOCK — End-to-End BER Simulation for Skyrmion Number Modulation

Complete communication pipeline:
  1. Generate skyrmionic polarization field with charge N
  2. Propagate through turbulent atmosphere (Kolmogorov phase screens)
  3. Measure Stokes field at receiver via simulated polarimetry
  4. Compute skyrmion number Q from Stokes parameters
  5. Hard-decide: ˆN = nearest constellation value
  6. Compute BER over multiple trials

Also includes OAM multiplexing baseline for fair comparison.

All simulations are mathematical models on a laptop. Results do not
establish physical feasibility — hardware would be required.

XPOCK research only — no HX-241 involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from xpock.wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from xpock.wave_info.skyrmion_modem import (
    ModemConfig,
    SkyrmionModem,
    ModemResult,
)


@dataclass
class BERResult:
    """BER simulation result for a single (N, Cn²) configuration."""

    n_tx: int                      # Transmitted skyrmion number
    cn2: float                     # Turbulence strength
    n_bits: int                    # Bits per symbol
    ber: float                     # Bit error rate
    ser: float                     # Symbol error rate
    n_trials: int                  # Number of trials
    n_errors: int                  # Number of bit errors
    n_symbols: int                 # Number of symbols transmitted
    q_mean: float                  # Mean measured skyrmion number
    q_std: float                   # Std dev of measured skyrmion number
    confusion_matrix: Dict[str, int]  # "tx_N->rx_N": count
    notes: List[str] = field(default_factory=list)


@dataclass
class OAMResult:
    """OAM multiplexing baseline result for comparison."""

    l_tx: int                      # Transmitted OAM mode
    cn2: float                     # Turbulence strength
    ber: float                     # Bit error rate
    ser: float                     # Symbol error rate
    n_trials: int
    n_errors: int
    n_symbols: int
    mode_overlap_mean: float       # Mean overlap with transmitted mode
    mode_overlap_std: float
    notes: List[str] = field(default_factory=list)


class BERSimulator:
    """
    End-to-end BER simulator for skyrmion number modulation.

    Simulates the complete link from bit generation through propagation
    to bit recovery, computing BER vs turbulence strength.
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

        # Grid spacing
        nx_dim = self.grid_x.shape[0] if self.grid_x.ndim == 2 else len(self.grid_x)
        ny_dim = self.grid_x.shape[1] if self.grid_x.ndim == 2 else len(self.grid_x)
        self.dx = self.grid_x[0, 1] - self.grid_x[0, 0] if self.grid_x.ndim == 2 else self.grid_x[1] - self.grid_x[0]
        self.dy = self.grid_y[1, 0] - self.grid_y[0, 0] if self.grid_y.ndim == 2 else self.grid_y[1] - self.grid_y[0]

    def _make_turb_model(self, cn2: float, distance: float, n_screens: int) -> TurbulenceModel:
        """Build a turbulence model for a (Cn², distance) configuration."""
        turb_cfg = TurbulenceConfig(
            wavelength=self.wavelength,
            distance=distance,
            cn2=cn2,
            n_screens=n_screens,
            seed=int(self.rng.integers(0, 2 ** 31 - 1)),
        )
        return TurbulenceModel(turb_cfg)

    def _fresh_screens(self, turb_model: TurbulenceModel, n_screens: int,
                       dz: float, ny_dim: int, nx_dim: int) -> List[np.ndarray]:
        """Generate one fresh, independent set of phase screens for a trial."""
        return [
            turb_model.generate_phase_screen((ny_dim, nx_dim), self.dx, dz)
            for _ in range(n_screens)
        ]

    def run_skm_ber(
        self,
        constellation: List[int],
        cn2_values: List[float],
        n_trials_per_config: int = 100,
        distance: float = 100.0,
        n_screens: int = 10,
        verbose: bool = True,
        turb_models: Optional[Dict[float, TurbulenceModel]] = None,
    ) -> List[BERResult]:
        """
        Run skyrmion number modulation BER simulation.

        Parameters
        ----------
        constellation : list of int
            Skyrmion numbers used as constellation points.
        cn2_values : list of float
            Refractive index structure constants to test.
        n_trials_per_config : int
            Trials per (N, Cn²) pair. Each trial uses an independent phase
            screen realization.
        distance : float
            Propagation distance in meters.
        n_screens : int
            Number of phase screens along path.
        verbose : bool
            Print progress.
        turb_models : dict of float -> TurbulenceModel, optional
            Pre-built turbulence models keyed by Cn². When supplied (e.g. by
            compare_skm_vs_oam) the SAME model is reused so SKM and OAM see
            identical turbulence statistics. Otherwise one model per Cn² is
            created internally.

        Returns
        -------
        results : list of BERResult
        """
        modem_cfg = ModemConfig(
            wavelength=self.wavelength,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            skyrmion_radius=self.skyrmion_radius,
            constellation=constellation,
        )
        modem = SkyrmionModem(modem_cfg)
        n_bits = modem.n_bits

        if verbose:
            print("=" * 70)
            print("SKYRMION NUMBER MODULATION — BER SIMULATION")
            print("=" * 70)
            print(f"Constellation: {constellation}")
            print(f"States: {len(constellation)}, Bits/symbol: {n_bits}")
            print(f"Distance: {distance} m, Wavelength: {self.wavelength*1e9:.0f} nm")
            print(f"Grid: {self.grid_x.shape[0]}x{self.grid_x.shape[1]}, dx={self.dx:.3f} m")
            print(f"Trials per config: {n_trials_per_config}")
            print(f"Cn² values: {cn2_values}")
            print()

        results = []
        total_configs = len(constellation) * len(cn2_values)
        config_idx = 0
        nx_dim = self.grid_x.shape[0]
        ny_dim = self.grid_x.shape[1]

        for cn2 in cn2_values:
            turb_model = turb_models[cn2] if (turb_models is not None and cn2 in turb_models) \
                else self._make_turb_model(cn2, distance, n_screens)
            dz = distance / n_screens

            for n_tx in constellation:
                config_idx += 1
                if verbose:
                    print(
                        f"[{config_idx}/{total_configs}] "
                        f"N={n_tx}, Cn²={cn2:.0e} ... "
                    )

                # Map N to bit index
                try:
                    bit_idx = constellation.index(n_tx)
                except ValueError:
                    bit_idx = 0
                bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])

                # Generate skyrmion field (same for all trials at this N)
                Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, self.grid_x, self.grid_y)

                n_errors = 0
                n_symbols = 0
                q_values = []
                confusion = {}

                for _ in range(n_trials_per_config):
                    # Fresh phase screens for this trial -> statistical independence.
                    # Same screens applied to Ex and Ey to preserve polarization.
                    phase_screens = self._fresh_screens(
                        turb_model, n_screens, dz, ny_dim, nx_dim
                    )
                    E_rx = turb_model.propagate(
                        Ex_tx, self.grid_x, self.grid_y,
                        phase_screens=phase_screens,
                    )
                    E_rx_y = turb_model.propagate(
                        Ey_tx, self.grid_x, self.grid_y,
                        phase_screens=phase_screens,
                    )

                    # Receiver: measure skyrmion number
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
                ser = ber  # Same for single-symbol transmission

                results.append(
                    BERResult(
                        n_tx=n_tx,
                        cn2=cn2,
                        n_bits=n_bits,
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

                if verbose:
                    print(f"BER={ber:.6f} ({n_errors}/{n_symbols}), Q={np.mean(q_values):+.4f}±{np.std(q_values):.4f}")

        if verbose:
            print()
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for r in results:
                print(
                    f"  N={r.n_tx:>3}, Cn²={r.cn2:.0e}: "
                    f"BER={r.ber:.6f} ({r.n_errors}/{r.n_symbols}), "
                    f"Q={r.q_mean:+.4f}±{r.q_std:.4f}"
                )

        return results

    def run_oam_ber(
        self,
        oam_modes: List[int],
        cn2_values: List[float],
        n_trials_per_config: int = 100,
        distance: float = 100.0,
        n_screens: int = 10,
        waist: float = 1.0,
        verbose: bool = True,
        turb_models: Optional[Dict[float, TurbulenceModel]] = None,
    ) -> List[OAMResult]:
        """
        Run OAM multiplexing baseline BER simulation.

        OAM modes are Laguerre-Gaussian modes with azimuthal index l.
        Information encoded as mode index l. Receiver computes overlap
        with each mode and picks the closest match.

        Under turbulence, OAM modes couple to neighboring modes,
        causing errors. This provides a fair comparison baseline.
        """
        from xpock.properties import SpatialMode, SpatialModeConfig

        if verbose:
            print("=" * 70)
            print("OAM MULTIPLEXING — BER SIMULATION (BASELINE)")
            print("=" * 70)
            print(f"OAM modes: {oam_modes}")
            print(f"States: {len(oam_modes)}, Bits/symbol: {int(np.floor(np.log2(len(oam_modes))))}")
            print(f"Distance: {distance} m, Wavelength: {self.wavelength*1e9:.0f} nm")
            print(f"Beam waist: {waist} m")
            print(f"Trials per config: {n_trials_per_config}")
            print(f"Cn² values: {cn2_values}")
            print()

        n_bits = int(np.floor(np.log2(len(oam_modes))))
        results = []
        total_configs = len(oam_modes) * len(cn2_values)
        config_idx = 0

        # Pre-compute all OAM mode fields on the grid
        mode_fields = {}
        X = self.grid_x
        Y = self.grid_y
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        for l in oam_modes:
            # Generate LG_0^l mode field directly:
            # (sqrt(2)*R/w)^|l| * exp(-R^2/w^2) * exp(i*l*Phi)
            k = 2.0 * np.pi / self.wavelength
            z_R = np.pi * waist**2 / self.wavelength
            w_z = waist * np.sqrt(1.0 + (0.0 / z_R)**2)  # at z=0
            R_over_w = np.sqrt(2.0) * R / w_z
            radial = R_over_w ** abs(l) * np.exp(-R**2 / w_z**2)
            azimuthal = np.exp(1j * l * Phi)
            mode_field = radial * azimuthal
            # Normalize
            norm = np.sqrt(np.sum(np.abs(mode_field)**2) * self.dx * self.dy)
            if norm > 0:
                mode_field = mode_field / norm
            mode_fields[l] = mode_field

        nx_dim = self.grid_x.shape[0]
        ny_dim = self.grid_x.shape[1]

        for cn2 in cn2_values:
            turb_model = turb_models[cn2] if (turb_models is not None and cn2 in turb_models) \
                else self._make_turb_model(cn2, distance, n_screens)
            dz = distance / n_screens

            for l_tx in oam_modes:
                config_idx += 1
                if verbose:
                    print(
                        f"[{config_idx}/{total_configs}] "
                        f"OAM l={l_tx}, Cn²={cn2:.0e} ... "
                    )

                # Map l to bit index
                try:
                    bit_idx = oam_modes.index(l_tx)
                except ValueError:
                    bit_idx = 0
                bits_tx = np.array([(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])

                # Transmitted OAM field
                E_tx = mode_fields[l_tx].copy()

                n_errors = 0
                n_symbols = 0
                overlaps = []
                confusion = {}

                for _ in range(n_trials_per_config):
                    # Fresh phase screens for this trial -> statistical independence
                    phase_screens = self._fresh_screens(
                        turb_model, n_screens, dz, ny_dim, nx_dim
                    )
                    # Propagate through turbulence
                    E_rx = turb_model.propagate(
                        E_tx, self.grid_x, self.grid_y,
                        phase_screens=phase_screens,
                    )

                    # Receiver: compute overlap with each OAM mode
                    overlap_vals = {}
                    for l_rx in oam_modes:
                        mode_rx = mode_fields[l_rx]
                        overlap = np.abs(np.sum(E_rx * np.conj(mode_rx)) * self.dx * self.dy)
                        overlap_vals[l_rx] = overlap

                    # Hard-decide: pick mode with max overlap
                    l_rx = max(overlap_vals, key=overlap_vals.get)
                    overlaps.append(overlap_vals[l_tx])

                    bits_rx = np.array([(oam_modes.index(l_rx) >> i) & 1 for i in range(n_bits - 1, -1, -1)])
                    bit_error = not np.array_equal(bits_tx, bits_rx)
                    n_errors += int(bit_error)
                    n_symbols += 1

                    key = f"{l_tx}->{l_rx}"
                    confusion[key] = confusion.get(key, 0) + 1

                ber = n_errors / n_symbols if n_symbols > 0 else 0.0
                ser = ber

                results.append(
                    OAMResult(
                        l_tx=l_tx,
                        cn2=cn2,
                        ber=float(ber),
                        ser=float(ser),
                        n_trials=n_trials_per_config,
                        n_errors=n_errors,
                        n_symbols=n_symbols,
                        mode_overlap_mean=float(np.mean(overlaps)) if overlaps else 0.0,
                        mode_overlap_std=float(np.std(overlaps)) if overlaps else 0.0,
                        notes=[],
                    )
                )

                if verbose:
                    print(f"BER={ber:.6f} ({n_errors}/{n_symbols}), overlap={np.mean(overlaps):.4f}±{np.std(overlaps):.4f}")

        if verbose:
            print()
            print("=" * 70)
            print("OAM SUMMARY")
            print("=" * 70)
            for r in results:
                print(
                    f"  l={r.l_tx:>3}, Cn²={r.cn2:.0e}: "
                    f"BER={r.ber:.6f} ({r.n_errors}/{r.n_symbols}), "
                    f"overlap={r.mode_overlap_mean:.4f}±{r.mode_overlap_std:.4f}"
                )

        return results

    def compare_skm_vs_oam(
        self,
        skm_constellation: List[int],
        oam_modes: List[int],
        cn2_values: List[float],
        n_trials_per_config: int = 50,
        distance: float = 100.0,
        n_screens: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run both SKM and OAM simulations and return comparison.

        Returns dict with 'skm_results', 'oam_results', 'comparison'
        where comparison summarizes BER at each Cn² level.
        """
        if verbose:
            print("#" * 70)
            print("# SKM vs OAM COMPARISON")
            print("#" * 70)
            print()

        # One turbulence model per Cn², shared by both schemes for a fair
        # comparison (identical turbulence statistics).
        turb_models: Dict[float, TurbulenceModel] = {}
        for cn2 in cn2_values:
            turb_models[cn2] = self._make_turb_model(cn2, distance, n_screens)

        skm_results = self.run_skm_ber(
            constellation=skm_constellation,
            cn2_values=cn2_values,
            n_trials_per_config=n_trials_per_config,
            distance=distance,
            n_screens=n_screens,
            verbose=verbose,
            turb_models=turb_models,
        )

        print()
        oam_results = self.run_oam_ber(
            oam_modes=oam_modes,
            cn2_values=cn2_values,
            n_trials_per_config=n_trials_per_config,
            distance=distance,
            n_screens=n_screens,
            verbose=verbose,
            turb_models=turb_models,
        )

        # Build comparison: average BER per Cn² across all states
        comparison = []
        for cn2 in cn2_values:
            skm_bers = [r.ber for r in skm_results if r.cn2 == cn2]
            oam_bers = [r.ber for r in oam_results if r.cn2 == cn2]
            skm_ber_avg = np.mean(skm_bers) if skm_bers else 0.0
            oam_ber_avg = np.mean(oam_bers) if oam_bers else 0.0
            comparison.append({
                'cn2': cn2,
                'skm_ber': float(skm_ber_avg),
                'oam_ber': float(oam_ber_avg),
                'skm_ber_max': float(max(skm_bers)) if skm_bers else 0.0,
                'oam_ber_max': float(max(oam_bers)) if oam_bers else 0.0,
                'advantage_db': float(10 * np.log10(oam_ber_avg / skm_ber_avg)) if skm_ber_avg > 0 and oam_ber_avg > 0 else float('inf'),
            })

        if verbose:
            print()
            print("#" * 70)
            print("# COMPARISON SUMMARY")
            print("#" * 70)
            print(f"{'Cn²':>12} {'SKM BER':>12} {'OAM BER':>12} {'Advantage (dB)':>16}")
            print("-" * 52)
            for c in comparison:
                adv = c['advantage_db']
                adv_str = f"{adv:.2f}" if adv != float('inf') else "inf"
                print(f"{c['cn2']:.0e}  {c['skm_ber']:.6e}  {c['oam_ber']:.6e}  {adv_str:>16}")

        return {
            'skm_results': skm_results,
            'oam_results': oam_results,
            'comparison': comparison,
        }


def quick_ber_test():
    """Quick BER test: 2-state SKM constellation, 3 turbulence levels."""
    nx, ny = 64, 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    sim = BERSimulator(
        grid_x=X,
        grid_y=Y,
        wavelength=1.55e-6,
        skyrmion_radius=1.0,
        seed=42,
    )

    results = sim.run_skm_ber(
        constellation=[-1, 1],
        cn2_values=[1e-15, 1e-14, 1e-13],
        n_trials_per_config=30,
        distance=100.0,
        n_screens=5,
        verbose=True,
    )

    print()
    print(f"Overall: {sum(r.n_errors for r in results)}/{sum(r.n_symbols for r in results)} errors")
    return results


if __name__ == "__main__":
    quick_ber_test()