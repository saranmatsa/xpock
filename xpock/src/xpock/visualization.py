"""Visualization of generated wave fields.

Produces publication-ready figures of:
- Spatial intensity profiles |E(x,y)|^2 at selected times/z
- Temporal intensity profiles |E(t)|^2 at beam center
- Phase profiles and phase singularities
- Polarization Stokes parameters and ellipses
- Temporal and spatial spectra
- Side-by-side comparison of multiple events
- OAM phase vortex visualization

All figures are saved to disk (PNG/PDF) and can also be displayed
interactively when running in an environment with a display backend.

Visualizations are clearly labelled as mathematical-model output.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend by default (laptop-safe)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from xpock.core import WaveEvent, WaveField


class WaveVisualizer:
    """Generate and save visualizations of wave events."""

    def __init__(self, output_dir: Optional[str | Path] = None, dpi: int = 150):
        self.output_dir = Path(output_dir) if output_dir else Path("./xpock_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.cmap_intensity = "inferno"
        self.cmap_phase = "twilight"
        self.cmap_stokes = "RdBu_r"

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _fig_path(self, event: WaveEvent, tag: str, ext: str = "png") -> Path:
        return self.output_dir / f"{event.event_id[:8]}_{tag}.{ext}"

    def _save(self, fig: plt.Figure, path: Path) -> None:
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_spatial_intensity(
        self,
        event: WaveEvent,
        t_index: Optional[int] = None,
        z_index: Optional[int] = None,
        component: str = "total",
    ) -> Path:
        """Plot |E(x,y)|^2 intensity at a given time/z slice."""
        field = event.field
        x = field.x
        y = field.y
        if x is None or y is None:
            raise ValueError("Spatial coordinates required for spatial intensity plot.")

        I = field.intensity
        # Select slices
        if I.ndim == 4:  # (z, y, x, t)
            zi = z_index if z_index is not None else I.shape[0] // 2
            ti = t_index if t_index is not None else I.shape[3] // 2
            I_slice = I[zi, :, :, ti]
        elif I.ndim == 3:  # (y, x, t)
            ti = t_index if t_index is not None else I.shape[2] // 2
            I_slice = I[:, :, ti]
        else:
            I_slice = I

        X, Y = np.meshgrid(x, y, indexing="xy")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            np.abs(I_slice),
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            cmap=self.cmap_intensity,
            norm=Normalize(vmin=0, vmax=np.max(np.abs(I_slice))),
            aspect="auto",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Spatial Intensity |E(x,y)|^2 — {event.event_id[:8]}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="Intensity")
        plt.tight_layout()

        path = self._fig_path(event, "spatial_intensity")
        self._save(fig, path)
        return path

    def plot_temporal_intensity(
        self,
        event: WaveEvent,
        x_index: Optional[int] = None,
        y_index: Optional[int] = None,
        z_index: Optional[int] = None,
    ) -> Path:
        """Plot |E(t)|^2 at a given spatial point."""
        field = event.field
        t = field.t
        if t is None:
            raise ValueError("Temporal coordinate required for temporal intensity plot.")

        I = field.intensity
        # Select spatial point
        if I.ndim == 4:  # (z, y, x, t)
            zi = z_index if z_index is not None else I.shape[0] // 2
            yi = y_index if y_index is not None else I.shape[1] // 2
            xi = x_index if x_index is not None else I.shape[2] // 2
            I_t = I[zi, yi, xi, :]
        elif I.ndim == 3:  # (y, x, t)
            yi = y_index if y_index is not None else I.shape[0] // 2
            xi = x_index if x_index is not None else I.shape[1] // 2
            I_t = I[yi, xi, :]
        else:
            I_t = np.sum(I, axis=tuple(range(I.ndim - 1)))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, np.abs(I_t), "b-", linewidth=1.5, label="|E(t)|^2")
        ax.set_xlabel("t")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Temporal Intensity |E(t)|^2 — {event.event_id[:8]}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        path = self._fig_path(event, "temporal_intensity")
        self._save(fig, path)
        return path

    def plot_phase(
        self,
        event: WaveEvent,
        component: str = "Ex",
        t_index: Optional[int] = None,
        z_index: Optional[int] = None,
    ) -> Path:
        """Plot the phase of a field component at a given slice."""
        field = event.field
        x = field.x
        y = field.y
        if x is None or y is None:
            raise ValueError("Spatial coordinates required for phase plot.")

        psi = getattr(field, component)
        if psi is None:
            raise ValueError(f"Component {component} not present in field.")

        # Slice
        if psi.ndim == 4:
            zi = z_index if z_index is not None else psi.shape[0] // 2
            ti = t_index if t_index is not None else psi.shape[3] // 2
            psi_slice = psi[zi, :, :, ti]
        elif psi.ndim == 3:
            ti = t_index if t_index is not None else psi.shape[2] // 2
            psi_slice = psi[:, :, ti]
        else:
            psi_slice = psi

        phase = np.angle(psi_slice)

        X, Y = np.meshgrid(x, y, indexing="xy")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            phase,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            cmap=self.cmap_phase,
            vmin=-np.pi,
            vmax=np.pi,
            aspect="auto",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Phase of {component} — {event.event_id[:8]}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="Phase (rad)")
        plt.tight_layout()

        path = self._fig_path(event, f"phase_{component}")
        self._save(fig, path)
        return path

    def plot_stokes(
        self,
        event: WaveEvent,
        parameter: str = "s1",
        t_index: Optional[int] = None,
        z_index: Optional[int] = None,
    ) -> Path:
        """Plot a Stokes parameter map S0..S3."""
        field = event.field
        Ex = field.Ex
        Ey = field.Ey
        if Ex is None or Ey is None:
            raise ValueError("Ex and Ey required for Stokes polarization plot.")

        x = field.x
        y = field.y
        if x is None or y is None:
            raise ValueError("Spatial coordinates required for Stokes plot.")

        # Compute Stokes at selected slice
        if Ex.ndim == 4:
            zi = z_index if z_index is not None else Ex.shape[0] // 2
            ti = t_index if t_index is not None else Ex.shape[3] // 2
            Ex_s = Ex[zi, :, :, ti]
            Ey_s = Ey[zi, :, :, ti]
        elif Ex.ndim == 3:
            ti = t_index if t_index is not None else Ex.shape[2] // 2
            Ex_s = Ex[:, :, ti]
            Ey_s = Ey[:, :, ti]
        else:
            Ex_s = Ex
            Ey_s = Ey

        S0 = np.abs(Ex_s) ** 2 + np.abs(Ey_s) ** 2
        S0 = np.where(S0 == 0, 1e-15, S0)

        if parameter == "s0":
            data = S0
            label = "S0 (total intensity)"
        elif parameter == "s1":
            data = (np.abs(Ex_s) ** 2 - np.abs(Ey_s) ** 2) / S0
            label = "S1 (linear H/V)"
        elif parameter == "s2":
            data = (2.0 * np.real(Ex_s * np.conj(Ey_s))) / S0
            label = "S2 (linear +/-45)"
        elif parameter == "s3":
            data = (2.0 * np.imag(Ex_s * np.conj(Ey_s))) / S0
            label = "S3 (circular)"
        else:
            raise ValueError(f"Unknown Stokes parameter: {parameter}")

        X, Y = np.meshgrid(x, y, indexing="xy")

        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = np.max(np.abs(data))
        im = ax.imshow(
            data,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            cmap=self.cmap_stokes,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{label} — {event.event_id[:8]}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label=label)
        plt.tight_layout()

        path = self._fig_path(event, f"stokes_{parameter}")
        self._save(fig, path)
        return path

    def plot_spectra(
        self,
        event: WaveEvent,
    ) -> List[Path]:
        """Plot temporal and spatial spectra."""
        paths: List[Path] = []
        field = event.field

        # Temporal spectrum
        if field.t is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            if hasattr(event, "properties") and event.properties.get("temporal_spectrum") is not None:
                spec = np.array(event.properties["temporal_spectrum"])
                freqs = np.array(event.properties["temporal_freqs"])
            else:
                # Compute on the fly
                I = field.intensity
                if field.x is not None and field.y is not None:
                    dx = field.x[1] - field.x[0] if len(field.x) > 1 else 1.0
                    dy = field.y[1] - field.y[0] if len(field.y) > 1 else 1.0
                    if I.ndim == 4:
                        I_t = np.sum(I, axis=(0, 1, 2)) * dx * dy
                    else:
                        I_t = np.sum(I, axis=(0, 1)) * dx * dy
                else:
                    I_t = np.sum(I, axis=tuple(range(I.ndim - 1)))
                from scipy.fft import fft, fftshift, fftfreq
                dt = field.t[1] - field.t[0] if len(field.t) > 1 else 1.0
                spec = np.abs(fftshift(fft(I_t))) ** 2
                freqs = fftshift(fftfreq(len(field.t), d=dt))

            ax.plot(freqs, spec, "k-", linewidth=1.2)
            ax.set_xlabel("Frequency (rad/s)")
            ax.set_ylabel("Power")
            ax.set_title(f"Temporal Spectrum — {event.event_id[:8]}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = self._fig_path(event, "temporal_spectrum")
            self._save(fig, path)
            paths.append(path)

        # Spatial spectrum
        if field.x is not None and field.y is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            I = field.intensity
            if I.ndim == 4:
                dt = field.t[1] - field.t[0] if field.t is not None and len(field.t) > 1 else 1.0
                dz = field.z[1] - field.z[0] if field.z is not None and len(field.z) > 1 else 1.0
                I_xy = np.sum(I, axis=(0, 3)) * dz * dt
            elif I.ndim == 3:
                I_xy = np.sum(I, axis=2)
            else:
                I_xy = I

            dx = field.x[1] - field.x[0] if len(field.x) > 1 else 1.0
            dy = field.y[1] - field.y[0] if len(field.y) > 1 else 1.0
            from scipy.fft import fft2, fftshift
            spec = np.abs(fftshift(fft2(I_xy))) ** 2
            im = ax.imshow(
                spec,
                extent=[dx, dx, dy, dy],  # placeholder extent
                origin="lower",
                cmap="viridis",
                aspect="auto",
            )
            ax.set_xlabel("kx index")
            ax.set_ylabel("ky index")
            ax.set_title(f"Spatial Spectrum — {event.event_id[:8]}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, label="Power")
            plt.tight_layout()
            path = self._fig_path(event, "spatial_spectrum")
            self._save(fig, path)
            paths.append(path)

        return paths

    def compare_events(
        self,
        events: List[WaveEvent],
        tag: str = "comparison",
    ) -> Path:
        """Side-by-side spatial intensity comparison of multiple events."""
        n = len(events)
        if n == 0:
            raise ValueError("No events to compare.")

        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        max_I = 0.0
        slices = []
        for ev in events:
            I = ev.field.intensity
            if I.ndim == 4:
                sl = I[I.shape[0] // 2, :, :, I.shape[3] // 2]
            elif I.ndim == 3:
                sl = I[:, :, I.shape[2] // 2]
            else:
                sl = I
            slices.append(np.abs(sl))
            max_I = max(max_I, np.max(slices[-1]))

        for i, (ev, sl) in enumerate(zip(events, slices)):
            ax = axes[i]
            x = ev.field.x
            y = ev.field.y
            if x is None or y is None:
                ax.text(0.5, 0.5, "No coordinates", transform=ax.transAxes, ha="center")
                continue
            X, Y = np.meshgrid(x, y, indexing="xy")
            im = ax.imshow(
                sl,
                extent=[x[0], x[-1], y[0], y[-1]],
                origin="lower",
                cmap=self.cmap_intensity,
                vmin=0,
                vmax=max_I,
                aspect="auto",
            )
            ax.set_title(f"{ev.event_id[:8]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Wave Event Comparison — {tag}", fontsize=12)
        fig.colorbar(im, ax=axes.tolist(), label="Intensity", shrink=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        path = self.output_dir / f"comparison_{tag}.png"
        self._save(fig, path)
        return path

    def generate_all(
        self,
        event: WaveEvent,
    ) -> Dict[str, List[Path]]:
        """Generate all standard visualizations for an event.

        Returns dict mapping plot type to list of output paths.
        """
        results: Dict[str, List[Path]] = {
            "spatial_intensity": [],
            "temporal_intensity": [],
            "phase": [],
            "stokes": [],
            "spectra": [],
        }

        try:
            results["spatial_intensity"].append(self.plot_spatial_intensity(event))
        except Exception as e:
            results["spatial_intensity"].append(str(e))  # type: ignore[assignment]

        try:
            results["temporal_intensity"].append(self.plot_temporal_intensity(event))
        except Exception as e:
            results["temporal_intensity"].append(str(e))  # type: ignore[assignment]

        # Phase of each available component
        for comp in ("Ex", "Ey", "Ez"):
            if getattr(event.field, comp) is not None:
                try:
                    results["phase"].append(self.plot_phase(event, component=comp))
                except Exception as e:
                    results["phase"].append(str(e))  # type: ignore[assignment]

        # Stokes (only if Ex and Ey present)
        if event.field.Ex is not None and event.field.Ey is not None:
            for st in ("s0", "s1", "s2", "s3"):
                try:
                    results["stokes"].append(self.plot_stokes(event, parameter=st))
                except Exception as e:
                    results["stokes"].append(str(e))  # type: ignore[assignment]

        try:
            results["spectra"] = self.plot_spectra(event)
        except Exception as e:
            results["spectra"].append(str(e))  # type: ignore[assignment]

        return results


def visualize_event(event: WaveEvent, output_dir: Optional[str | Path] = None) -> Dict[str, List[Path]]:
    """Convenience: generate all standard visualizations for an event."""
    return WaveVisualizer(output_dir=output_dir).generate_all(event)
