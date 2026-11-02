"""
XPOCK — Comparison: Skyrmion Number Modulation vs OAM vs Intensity Modulation

Fair comparison of three information-carrying wave properties:

1. Skyrmion Number Modulation (SKM) — topological integer invariant
2. OAM Multiplexing — orbital angular momentum mode index
3. Intensity Modulation — Gaussian beam amplitude levels

All three use the same spatial grid, wavelength, propagation distance,
and turbulence model. Only the information carrier differs.

Metrics compared:
- BER vs turbulence strength Cn²
- Information density (bits per field)
- Robustness to turbulence
- Receiver measurement complexity

XPOCK research only — no HX-241 involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from xpock.wave_info.turbulence import TurbulenceConfig, TurbulenceModel
from xpock.wave_info.ber_simulation import BERSimulator, BERResult, OAMResult


@dataclass
class ModulationComparison:
    """Side-by-side comparison of modulation schemes."""

    scheme_name: str
    bits_per_symbol: int
    cn2_values: List[float]
    ber_results: List[float]  # Average BER per Cn²
    max_ber: float            # Worst BER across all Cn²
    min_ber: float            # Best BER across all Cn²
    notes: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full comparison report across all schemes."""

    skm: ModulationComparison
    oam: ModulationComparison
    intensity: ModulationComparison
    distance: float
    wavelength: float
    grid_size: Tuple[int, int]
    n_trials: int
    winner_at_cn2: Dict[float, str]  # Cn² -> best scheme name
    skm_advantage_db: Dict[float, float]  # Cn² -> SKM advantage over OAM in dB


class WaveModulationComparator:
    """
    Compare SKM, OAM, and intensity modulation under identical conditions.

    Runs all three schemes through the same turbulence model and
    produces a fair comparison of BER vs turbulence strength.
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
        self.dx = self.grid_x[0, 1] - self.grid_x[0, 0] if self.grid_x.ndim == 2 else self.grid_x[1] - self.grid_x[0]
        self.dy = self.grid_y[1, 0] - self.grid_y[0, 0] if self.grid_y.ndim == 2 else self.grid_y[1] - self.grid_y[0]

        self.sim = BERSimulator(
            grid_x=grid_x,
            grid_y=grid_y,
            wavelength=wavelength,
            skyrmion_radius=skyrmion_radius,
            seed=seed,
        )

    def compare_all(
        self,
        skm_constellation: List[int],
        oam_modes: List[int],
        intensity_levels: List[float],
        cn2_values: List[float],
        n_trials_per_config: int = 50,
        distance: float = 100.0,
        n_screens: int = 10,
        waist: float = 1.0,
        verbose: bool = True,
    ) -> ComparisonReport:
        """
        Run full comparison of SKM, OAM, and intensity modulation.

        Parameters
        ----------
        skm_constellation : list of int
            Skyrmion numbers (e.g., [-1, 1] for 1 bit, [-3..4] for 3 bits).
        oam_modes : list of int
            OAM mode indices (e.g., [-1, 1] for 1 bit, [-3..4] for 3 bits).
        intensity_levels : list of float
            Intensity levels for Gaussian beam (e.g., [0.3, 1.0] for 1 bit).
        cn2_values : list of float
            Turbulence strengths to test.
        n_trials_per_config : int
            Trials per configuration.
        distance : float
            Propagation distance in meters.
        n_screens : int
            Number of phase screens.
        waist : float
            Gaussian beam waist for intensity modulation.
        verbose : bool
            Print progress.

        Returns
        -------
        report : ComparisonReport
        """
        if verbose:
            print("#" * 70)
            print("# THREE-WAY MODULATION COMPARISON")
            print("#" * 70)
            print(f"Distance: {distance} m, Wavelength: {self.wavelength*1e9:.0f} nm")
            print(f"Grid: {self.grid_x.shape[0]}x{self.grid_x.shape[1]}")
            print(f"Cn² values: {cn2_values}")
            print(f"Trials per config: {n_trials_per_config}")
            print()

        # Build one turbulence model per Cn², shared across all three schemes
        # for a fair comparison (identical turbulence statistics).
        turb_models: Dict[float, TurbulenceModel] = {}
        for cn2 in cn2_values:
            turb_cfg = TurbulenceConfig(
                wavelength=self.wavelength,
                distance=distance,
                cn2=cn2,
                n_screens=n_screens,
                seed=int(self.rng.integers(0, 2 ** 31 - 1)),
            )
            turb_models[cn2] = TurbulenceModel(turb_cfg)

        # --- SKM ---
        if verbose:
            print("-" * 70)
            print("SKYRMION NUMBER MODULATION")
            print("-" * 70)
        skm_result = self.sim.compare_skm_vs_oam(
            skm_constellation=skm_constellation,
            oam_modes=oam_modes,
            cn2_values=cn2_values,
            n_trials_per_config=n_trials_per_config,
            distance=distance,
            n_screens=n_screens,
            verbose=verbose,
            turb_models=turb_models,
        )
        skm_results = skm_result['skm_results']

        # --- OAM ---
        if verbose:
            print()
            print("-" * 70)
            print("OAM MULTIPLEXING (BASELINE)")
            print("-" * 70)
        oam_results = skm_result['oam_results']

        # --- Intensity Modulation ---
        if verbose:
            print()
            print("-" * 70)
            print("INTENSITY MODULATION (BASELINE)")
            print("-" * 70)
        intensity_results = self._run_intensity_ber(
            intensity_levels=intensity_levels,
            cn2_values=cn2_values,
            n_trials_per_config=n_trials_per_config,
            distance=distance,
            n_screens=n_screens,
            waist=waist,
            verbose=verbose,
            turb_models=turb_models,
        )

        # Build comparison summaries
        skm_comp = self._build_modulation_comparison(
            "Skyrmion Number Modulation",
            int(np.floor(np.log2(len(skm_constellation)))),
            cn2_values,
            skm_results,
        )
        oam_comp = self._build_modulation_comparison(
            "OAM Multiplexing",
            int(np.floor(np.log2(len(oam_modes)))),
            cn2_values,
            oam_results,
        )
        intensity_comp = self._build_modulation_comparison(
            "Intensity Modulation",
            int(np.floor(np.log2(len(intensity_levels)))),
            cn2_values,
            intensity_results,
        )

        # Determine winner at each Cn²
        winner_at_cn2 = {}
        skm_advantage_db = {}
        for cn2 in cn2_values:
            skm_ber = next((r.ber for r in skm_results if r.cn2 == cn2), 1.0)
            oam_ber = next((r.ber for r in oam_results if r.cn2 == cn2), 1.0)
            int_ber = next((r.ber for r in intensity_results if r.cn2 == cn2), 1.0)

            bers = {'SKM': skm_ber, 'OAM': oam_ber, 'Intensity': int_ber}
            winner = min(bers, key=bers.get)
            winner_at_cn2[cn2] = winner

            if skm_ber > 0 and oam_ber > 0:
                skm_advantage_db[cn2] = 10.0 * np.log10(oam_ber / skm_ber)
            elif skm_ber == 0 and oam_ber > 0:
                skm_advantage_db[cn2] = float('inf')
            else:
                skm_advantage_db[cn2] = 0.0

        report = ComparisonReport(
            skm=skm_comp,
            oam=oam_comp,
            intensity=intensity_comp,
            distance=distance,
            wavelength=self.wavelength,
            grid_size=(self.grid_x.shape[0], self.grid_x.shape[1]),
            n_trials=n_trials_per_config,
            winner_at_cn2=winner_at_cn2,
            skm_advantage_db=skm_advantage_db,
        )

        if verbose:
            self._print_report(report)

        return report

    def _run_intensity_ber(
        self,
        intensity_levels: List[float],
        cn2_values: List[float],
        n_trials_per_config: int,
        distance: float,
        n_screens: int,
        waist: float,
        verbose: bool,
        turb_models: Optional[Dict[float, TurbulenceModel]] = None,
    ) -> List:
        """
        Run intensity modulation BER simulation.

        Gaussian beam with different peak intensity levels encoding bits.
        Receiver measures total intensity and picks nearest level.
        Under turbulence, scintillation causes intensity fading -> errors.

        Uses all n_screens (previously only screen 0 was used, and propagate
        was called with the wrong keyword 'phase_screen' which crashed the
        three-way comparison). Every trial draws an independent phase-screen
        realization for statistical independence.
        """
        from xpock.properties import SpatialMode, SpatialModeConfig

        if verbose:
            print("=" * 70)
            print("INTENSITY MODULATION (BASELINE)")
            print("=" * 70)
            print(f"Intensity levels: {intensity_levels}")
            print(f"States: {len(intensity_levels)}, Bits/symbol: {int(np.floor(np.log2(len(intensity_levels))))}")
            print(f"Distance: {distance} m, Wavelength: {self.wavelength*1e9:.0f} nm")
            print(f"Beam waist: {waist} m")
            print(f"Trials per config: {n_trials_per_config}")
            print(f"Cn² values: {cn2_values}")
            print()

        n_bits = int(np.floor(np.log2(len(intensity_levels))))
        results = []

        # Pre-compute Gaussian beam profile
        X = self.grid_x
        Y = self.grid_y
        R2 = X**2 + Y**2
        gaussian = np.exp(-R2 / waist**2).astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(gaussian)**2) * self.dx * self.dy)
        if norm > 0:
            gaussian = gaussian / norm

        nx_dim = self.grid_x.shape[0]
        ny_dim = self.grid_x.shape[1]

        total_configs = len(intensity_levels) * len(cn2_values)
        config_idx = 0

        for cn2 in cn2_values:
            turb_model = turb_models[cn2] if (turb_models is not None and cn2 in turb_models) \
                else TurbulenceModel(TurbulenceConfig(
                    wavelength=self.wavelength,
                    distance=distance,
                    cn2=cn2,
                    n_screens=n_screens,
                    seed=int(self.rng.integers(0, 2 ** 31 - 1)),
                ))
            dz = distance / n_screens

            for level_idx, level in enumerate(intensity_levels):
                config_idx += 1
                if verbose:
                    print(
                        f"[{config_idx}/{total_configs}] "
                        f"Intensity level={level:.2f}, Cn²={cn2:.0e} ... "
                    )

                bits_tx = np.array([(level_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])
                E_tx = level * gaussian

                n_errors = 0
                n_symbols = 0
                intensities = []
                confusion = {}

                for _ in range(n_trials_per_config):
                    # Fresh phase screens for this trial -> statistical independence
                    phase_screens = [
                        turb_model.generate_phase_screen((ny_dim, nx_dim), self.dx, dz)
                        for _ in range(n_screens)
                    ]
                    # Propagate through ALL n_screens with the correct keyword
                    E_rx = turb_model.propagate(
                        E_tx, self.grid_x, self.grid_y,
                        phase_screens=phase_screens,
                    )

                    # Receiver: measure total intensity
                    measured_intensity = float(np.sum(np.abs(E_rx)**2) * self.dx * self.dy)
                    intensities.append(measured_intensity)

                    # Hard-decide: nearest intensity level
                    nearest_idx = int(np.argmin([abs(measured_intensity - l) for l in intensity_levels]))
                    bits_rx = np.array([(nearest_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)])

                    bit_error = not np.array_equal(bits_tx, bits_rx)
                    n_errors += int(bit_error)
                    n_symbols += 1

                    key = f"{level:.2f}->{intensity_levels[nearest_idx]:.2f}"
                    confusion[key] = confusion.get(key, 0) + 1

                ber = n_errors / n_symbols if n_symbols > 0 else 0.0

                # Create a simple result object (reuse OAMResult structure)
                class IntensityResult:
                    def __init__(self, **kw):
                        self.__dict__.update(kw)

                results.append(IntensityResult(
                    l_tx=level,
                    cn2=cn2,
                    ber=float(ber),
                    ser=float(ber),
                    n_trials=n_trials_per_config,
                    n_errors=n_errors,
                    n_symbols=n_symbols,
                    mode_overlap_mean=float(np.mean(intensities)) if intensities else 0.0,
                    mode_overlap_std=float(np.std(intensities)) if intensities else 0.0,
                    notes=[],
                ))

                if verbose:
                    print(f"BER={ber:.6f} ({n_errors}/{n_symbols}), I={np.mean(intensities):.4f}±{np.std(intensities):.4f}")

        return results

    def _build_modulation_comparison(
        self,
        scheme_name: str,
        bits_per_symbol: int,
        cn2_values: List[float],
        results: List,
    ) -> ModulationComparison:
        """Build a ModulationComparison from raw results."""
        bers_per_cn2 = []
        for cn2 in cn2_values:
            cn2_results = [r for r in results if getattr(r, 'cn2', None) == cn2]
            if cn2_results:
                bers_per_cn2.append(np.mean([r.ber for r in cn2_results]))
            else:
                bers_per_cn2.append(0.0)

        return ModulationComparison(
            scheme_name=scheme_name,
            bits_per_symbol=bits_per_symbol,
            cn2_values=list(cn2_values),
            ber_results=[float(b) for b in bers_per_cn2],
            max_ber=float(max(bers_per_cn2)) if bers_per_cn2 else 0.0,
            min_ber=float(min(bers_per_cn2)) if bers_per_cn2 else 0.0,
        )

    def _print_report(self, report: ComparisonReport) -> None:
        """Print formatted comparison report."""
        print()
        print("#" * 70)
        print("# COMPARISON REPORT")
        print("#" * 70)
        print(f"{'Cn²':>12} {'SKM BER':>12} {'OAM BER':>12} {'Int BER':>12} {'Winner':>10} {'SKM Adv (dB)':>14}")
        print("-" * 74)
        for cn2 in report.skm.cn2_values:
            skm_idx = report.skm.cn2_values.index(cn2)
            oam_idx = report.oam.cn2_values.index(cn2)
            int_idx = report.intensity.cn2_values.index(cn2)

            skm_b = report.skm.ber_results[skm_idx]
            oam_b = report.oam.ber_results[oam_idx]
            int_b = report.intensity.ber_results[int_idx]

            winner = report.winner_at_cn2.get(cn2, '?')
            adv = report.skm_advantage_db.get(cn2, 0.0)
            adv_str = f"{adv:.2f}" if adv != float('inf') else "inf"

            print(
                f"{cn2:.0e}  {skm_b:.6e}  {oam_b:.6e}  {int_b:.6e}  "
                f"{winner:>10}  {adv_str:>14}"
            )

        print()
        print("Summary:")
        print(f"  SKM:       bits/symbol={report.skm.bits_per_symbol}, "
              f"max BER={report.skm.max_ber:.6e}, min BER={report.skm.min_ber:.6e}")
        print(f"  OAM:       bits/symbol={report.oam.bits_per_symbol}, "
              f"max BER={report.oam.max_ber:.6e}, min BER={report.oam.min_ber:.6e}")
        print(f"  Intensity: bits/symbol={report.intensity.bits_per_symbol}, "
              f"max BER={report.intensity.max_ber:.6e}, min BER={report.intensity.min_ber:.6e}")

        skm_wins = sum(1 for w in report.winner_at_cn2.values() if w == 'SKM')
        total = len(report.winner_at_cn2)
        print(f"  SKM wins at {skm_wins}/{total} turbulence levels")


def quick_comparison_test():
    """Quick comparison: SKM vs OAM vs Intensity at 3 turbulence levels."""
    nx, ny = 64, 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    comp = WaveModulationComparator(
        grid_x=X,
        grid_y=Y,
        wavelength=1.55e-6,
        skyrmion_radius=1.0,
        seed=42,
    )

    report = comp.compare_all(
        skm_constellation=[-1, 1],      # 1 bit
        oam_modes=[-1, 1],              # 1 bit
        intensity_levels=[0.5, 1.5],    # 1 bit
        cn2_values=[1e-15, 1e-14, 1e-13],
        n_trials_per_config=30,
        distance=100.0,
        n_screens=5,
        verbose=True,
    )

    return report


if __name__ == "__main__":
    quick_comparison_test()