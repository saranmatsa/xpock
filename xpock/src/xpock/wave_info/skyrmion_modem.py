"""
XPOCK — HX-241 Independent Research: Skyrmion Number Modulation Modem

Implements skyrmion number modulation (SkM) for free-space optical communication.

Information is encoded as the topological skyrmion number N of an optical
polarization texture. The skyrmion number is a topological invariant:
Q = (1/4π) ∫ n · (∂x n × ∂y n) dx dy

where n = (S1,S2,S3)/S0 is the normalized Stokes vector.

Transmitter:
- Generate skyrmionic polarization field with charge N from constellation
- The field is |Ψ⟩ = |l₁⟩|H⟩ + |l₂⟩|V⟩ where skyrmion number Q = l₁ - l₂

Receiver:
- Measure full Stokes field via polarimetry (6 intensity projections)
- Compute skyrmion number Q from Stokes parameters
- Hard-decide: ˆN = nearest constellation value

All models are mathematical simulations; results do not establish physical
feasibility. Hardware would be required for experimental validation.

XPOCK research only — no HX-241 involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from xpock.core import WaveEvent, WaveField
from xpock.properties import (
    PolarizationTexture,
    PolarizationConfig,
    PolarizationState,
    TopologicalCharge,
    TopologicalChargeConfig,
)

# Speed of light
C_LIGHT = 299792458.0


@dataclass
class ModemConfig:
    """Configuration for skyrmion number modulation modem.

    Attributes
    ----------
    wavelength : float
        Optical wavelength in meters (default 1550 nm).
    grid_x : np.ndarray
        x coordinates in meters.
    grid_y : np.ndarray
        y coordinates in meters.
    skyrmion_radius : float
        Skyrmion texture radius in meters.
    constellation : list of int
        Skyrmion numbers used as constellation points.
    """

    wavelength: float = 1.55e-6
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None
    skyrmion_radius: float = 1.0
    constellation: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    receiver_noise_std: float = 0.0
        # Standard deviation of additive complex Gaussian detector noise,
        # relative to the RMS field amplitude. 0.0 = ideal noiseless receiver.
        # >0 models photon/read noise in the Stokes measurement, which makes
        # the recovered skyrmion number (and hence BER) stochastic and
        # receiver-realistic. Opt-in; existing behaviour is unchanged at 0.


@dataclass
class ModemResult:
    """Result of skyrmion modulation/demodulation."""

    transmitted_n: int
    received_q: float
    recovered_n: int
    bit_error: bool
    bits_transmitted: int
    bits_received: int
    snr_db: float
    path_loss_db: float
    notes: List[str] = field(default_factory=list)


class SkyrmionModem:
    """Skyrmion number modulation modem for free-space optical communication.

    Encodes information as the topological skyrmion number of an optical
    polarization texture. The skyrmion number is a topological invariant
    that (in principle) is preserved during propagation through free space,
    offering potential robustness against atmospheric turbulence.

    This is a mathematical model; physical implementation would require
    polarization modulators, Stokes polarimetry cameras, and precise
    alignment.
    """

    def __init__(self, config: ModemConfig):
        self.cfg = config
        self._constellation = sorted(set(config.constellation))
        n_states = len(self._constellation)
        # ceil(log2(n_states)) so that every constellation state is reachable:
        # with n_bits = ceil(log2(n_states)), indices 0 .. 2**n_bits - 1 cover
        # 0 .. n_states - 1. floor(log2(n_states)) would silently leave the
        # highest states unreachable (e.g. a 5-state constellation with 2 bits
        # can only ever reach indices 0..3).
        self._n_bits = int(np.ceil(np.log2(n_states))) if n_states > 1 else 1
        if self._n_bits < 1:
            raise ValueError(
                f"Constellation must have at least 2 values, got {n_states}"
            )

    @property
    def constellation(self) -> List[int]:
        return self._constellation

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def n_states(self) -> int:
        return len(self._constellation)

    def bits_to_skyrmion(self, bits: np.ndarray) -> int:
        """Map bit string to skyrmion number.

        Parameters
        ----------
        bits : array-like of int (0/1)
            Bit string of length n_bits. Every bit must be 0 or 1.

        Returns
        -------
        n : int
            Skyrmion number from constellation.

        Notes
        -----
        Mapping: bit string -> unsigned integer index -> constellation[index].
        Valid indices are 0 .. n_states - 1. With n_bits = ceil(log2(n_states))
        every constellation state has at least one bit string (some indices in
        0 .. 2**n_bits - 1 are unused for non-power-of-two constellations and
        are rejected as invalid input rather than silently folded back by a
        modulo, which would make some states unreachable and others double).
        Incomplete final bit groups (length != n_bits) are invalid input.
        """
        bits = np.asarray(bits, dtype=int)
        if len(bits) != self._n_bits:
            raise ValueError(
                f"Expected {self._n_bits} bits, got {len(bits)} "
                f"(incomplete final bit groups are invalid input)"
            )
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError(f"Bits must be 0 or 1, got {bits}")

        # Convert bit string to integer index (MSB first)
        idx = 0
        for b in bits:
            idx = (idx << 1) | int(b)

        if idx >= self.n_states:
            raise ValueError(
                f"Bit string encodes index {idx}, which is outside the "
                f"constellation range 0..{self.n_states - 1} "
                f"(non-power-of-two alphabet: indices >= n_states are unused)"
            )
        return self._constellation[idx]

    def skyrmion_to_bits(self, n: int) -> np.ndarray:
        """Map skyrmion number back to bit string.

        Parameters
        ----------
        n : int
            Received skyrmion number. Must be a member of the constellation;
            values outside the constellation are invalid input (the demodulator
            uses :meth:`_nearest_constellation` for its hard decision).

        Returns
        -------
        bits : np.ndarray of int
            Recovered bit string of length n_bits.

        Raises
        ------
        ValueError
            If ``n`` is not a constellation member.
        """
        try:
            idx = self._constellation.index(int(n))
        except ValueError:
            raise ValueError(
                f"Skyrmion number {n} is not in constellation {self._constellation}"
            ) from None

        bits = []
        for i in range(self._n_bits - 1, -1, -1):
            bits.append((idx >> i) & 1)
        return np.array(bits, dtype=int)

    def modulate(
        self, bits: np.ndarray, grid_x: Optional[np.ndarray] = None,
        grid_y: Optional[np.ndarray] = None
    ) -> tuple:
        """Modulate bit string onto a skyrmionic polarization field.

        Parameters
        ----------
        bits : array-like of int
            Bit string of length n_bits.
        grid_x, grid_y : np.ndarray, optional
            Spatial coordinates. If None, uses config grids.

        Returns
        -------
        Ex : (ny, nx) complex array
            Ex component of the optical field.
        Ey : (ny, nx) complex array
            Ey component of the optical field.
        n_tx : int
            Transmitted skyrmion number.
        """
        n_tx = self.bits_to_skyrmion(bits)

        x = grid_x if grid_x is not None else self.cfg.grid_x
        y = grid_y if grid_y is not None else self.cfg.grid_y
        if x is None or y is None:
            raise ValueError("Grid arrays must be provided or set in config.")

        # Generate skyrmionic polarization texture
        # Use Laguerre-Gaussian spatial mode with OAM charge = N
        # Combined with linear polarization to create skyrmion texture
        cfg = PolarizationConfig(
            state=PolarizationState.SKYRMIONIC,
            angle=0.0,
            handedness=+1,
            skyrmion_radius=self.cfg.skyrmion_radius,
            skyrmion_charge=n_tx,
        )
        pol = PolarizationTexture(cfg)
        Ex, Ey = pol.evaluate(x, y)

        return Ex, Ey, n_tx

    def demodulate(
        self, Ex: np.ndarray, Ey: np.ndarray
    ) -> tuple:
        """Demodulate skyrmion number from received polarization field.

        Parameters
        ----------
        Ex : (ny, nx) complex array
            Received Ex component.
        Ey : (ny, nx) complex array
            Received Ey component.

        Returns
        -------
        q_measured : float
            Computed skyrmion number (continuous value).
        n_rx : int
            Hard-decided skyrmion number (nearest constellation value).
        """
        # Receiver realism: additive complex Gaussian detector noise (photon/
        # read noise) on the measured Stokes field. Opt-in via
        # ModemConfig.receiver_noise_std (relative to RMS field amplitude);
        # 0.0 reproduces the ideal noiseless polarimeter.
        if self.cfg.receiver_noise_std > 0.0:
            rms = float(np.sqrt(np.mean(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)))
            if rms > 0.0:
                rng = np.random.default_rng()
                sigma = self.cfg.receiver_noise_std * rms / np.sqrt(2.0)
                Ex = Ex + sigma * (rng.standard_normal(Ex.shape) + 1j * rng.standard_normal(Ex.shape))
                Ey = Ey + sigma * (rng.standard_normal(Ey.shape) + 1j * rng.standard_normal(Ey.shape))

        tc = TopologicalCharge(TopologicalChargeConfig(
            charge_type="skyrmion",
            charge=1,
            core_radius=self.cfg.skyrmion_radius / 10.0,
        ))
        q_measured = tc.compute_skyrmion_number(Ex, Ey)
        n_rx = self._nearest_constellation(q_measured)
        return q_measured, n_rx

    def _nearest_constellation(self, q: float) -> int:
        """Find nearest constellation value to measured q."""
        return self._constellation[int(np.argmin([abs(q - c) for c in self._constellation]))]

    def encode_decode(
        self, bits: np.ndarray, Ex_rx: np.ndarray, Ey_rx: np.ndarray
    ) -> ModemResult:
        """Encode bits, then demodulate from received field.

        Convenience method that combines modulation and demodulation
        for end-to-end testing.
        """
        n_tx = self.bits_to_skyrmion(bits)
        q_measured, n_rx = self.demodulate(Ex_rx, Ey_rx)

        bits_rx = self.skyrmion_to_bits(n_rx)
        bit_error = not np.array_equal(bits, bits_rx)

        return ModemResult(
            transmitted_n=n_tx,
            received_q=q_measured,
            recovered_n=n_rx,
            bit_error=bool(bit_error),
            bits_transmitted=len(bits),
            bits_received=len(bits_rx),
            snr_db=0.0,  # Filled by simulation runner
            path_loss_db=0.0,
            notes=[],
        )


def quick_modem_test():
    """Quick smoke test of skyrmion modem."""
    nx, ny = 64, 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-2, -1, 0, 1, 2],
    )
    modem = SkyrmionModem(cfg)

    # Test modulation/demodulation round-trip over ALL constellation states.
    # With 5 states, n_bits = ceil(log2(5)) = 3; indices 0..4 are valid,
    # indices 5..7 are rejected as invalid input.
    for n_tx in modem.constellation:
        bit_idx = modem.constellation.index(n_tx)
        bits_tx = np.array(
            [(bit_idx >> i) & 1 for i in range(modem.n_bits - 1, -1, -1)]
        )
        Ex, Ey, n_returned = modem.modulate(bits_tx, X, Y)
        assert n_returned == n_tx, f"{n_tx}: modulate returned {n_returned}"
        q_measured, n_rx = modem.demodulate(Ex, Ey)
        bits_rx = modem.skyrmion_to_bits(n_rx)
        print(
            f"  N={n_tx:+d} bits={bits_tx.tolist()} Q={q_measured:+.4f} "
            f"-> N_rx={n_rx} bits_rx={bits_rx.tolist()} "
            f"roundtrip={np.array_equal(bits_tx, bits_rx)}"
        )
        assert n_rx == n_tx

    # Invalid input: index outside constellation range must be rejected
    try:
        modem.bits_to_skyrmion(np.array([1, 1, 1]))  # index 7 >= 5 states
        raise AssertionError("bits_to_skyrmion accepted out-of-range index 7")
    except ValueError:
        pass
    try:
        modem.skyrmion_to_bits(99)  # not in constellation
        raise AssertionError("skyrmion_to_bits accepted non-constellation value")
    except ValueError:
        pass

    print(f"Constellation: {modem.constellation}")
    print(f"Bits per symbol: {modem.n_bits}")

    return modem


if __name__ == "__main__":
    quick_modem_test()