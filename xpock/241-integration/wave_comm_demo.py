#!/usr/bin/env python3
"""
XPOCK — Wave-Structure Communication Demo

END-TO-END COMMUNICATION: Encode bits → generate skyrmion wave → propagate
through turbulence → measure Stokes → decode skyrmion number → recover bits.

This is NOT theory. This is a working communication pipeline:
1. Text message → bit stream
2. Bit stream → skyrmion constellation symbols
3. Symbols → optical polarization fields (Ex, Ey)
4. Propagate through turbulent atmosphere
5. Receiver measures full Stokes field via polarimetry
6. Compute skyrmion number Q from Stokes parameters
7. Map Q → nearest constellation symbol → bits
8. Compare transmitted vs received text

No IP, TCP, UDP, Wi-Fi, Bluetooth, or any existing network stack.
Communication happens purely through physical wave structure.
"""

from __future__ import annotations

import importlib.util
import sys
import numpy as np

# ----------------------------------------------------------------------
# Load XPOCK modules (no install required — direct file import)
# ----------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    'skyrmion_modem', 'src/xpock/wave_info/skyrmion_modem.py'
)
modem_mod = importlib.util.module_from_spec(spec)
sys.modules['skyrmion_modem'] = modem_mod
spec.loader.exec_module(modem_mod)
SkyrmionModem = modem_mod.SkyrmionModem
ModemConfig = modem_mod.ModemConfig

spec2 = importlib.util.spec_from_file_location(
    'turbulence', 'src/xpock/wave_info/turbulence.py'
)
turbulence_mod = importlib.util.module_from_spec(spec2)
sys.modules['turbulence'] = turbulence_mod
spec2.loader.exec_module(turbulence_mod)
TurbulenceModel = turbulence_mod.TurbulenceModel
TurbulenceConfig = turbulence_mod.TurbulenceConfig


# ----------------------------------------------------------------------
# Text ↔ bits conversion
# ----------------------------------------------------------------------
def text_to_bits(message: str) -> np.ndarray:
    """Convert ASCII text to bit array."""
    return np.array([(ord(c) >> i) & 1 for c in message for i in range(7, -1, -1)], dtype=int)


def bits_to_text(bits: np.ndarray) -> str:
    """Convert bit array back to ASCII text."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | int(bits[i + j])
        chars.append(chr(byte))
    return ''.join(chars)


# ----------------------------------------------------------------------
# Communication channel
# ----------------------------------------------------------------------
class WaveCommChannel:
    """
    End-to-end wave-structure communication channel.

    Transmits information through topological wave structures (skyrmion
    polarization textures) without any conventional networking stack.
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        wavelength: float = 1.55e-6,
        skyrmion_radius: float = 1.0,
        seed: int = 42,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.wavelength = wavelength
        self.skyrmion_radius = skyrmion_radius
        self.seed = seed

        # 2-state constellation {-1, +1} = 1 bit per skyrmion symbol
        self.cfg = ModemConfig(
            wavelength=wavelength,
            grid_x=grid_x,
            grid_y=grid_y,
            skyrmion_radius=skyrmion_radius,
            constellation=[-1, 1],
        )
        self.modem = SkyrmionModem(self.cfg)

    def transmit(
        self,
        message: str,
        cn2: float = 1e-14,
        distance: float = 100.0,
        n_screens: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Transmit a text message through the wave channel.

        Parameters
        ----------
        message : str
            Text to transmit.
        cn2 : float
            Refractive index structure constant (turbulence strength).
        distance : float
            Propagation distance in meters.
        n_screens : int
            Number of phase screens for turbulence.
        verbose : bool
            Print progress.

        Returns
        -------
        result : dict
            Transmission statistics and recovered message.
        """
        if verbose:
            print("=" * 70)
            print("WAVE-STRUCTURE COMMUNICATION TRANSMISSION")
            print("=" * 70)
            print(f"Message:      '{message}' ({len(message)} chars)")
            print(f"Bits:         {len(message) * 8}")
            print(f"Constellation: {self.modem.constellation}")
            print(f"Bits/symbol:  {self.modem.n_bits}")
            print(f"Turbulence:   Cn² = {cn2:.0e}, distance = {distance} m")
            print()

        # Step 1: Convert text to bits
        bits_tx = text_to_bits(message)
        if verbose:
            print(f"[1/6] Text → bits: {len(bits_tx)} bits")

        # Step 2: Group bits into skyrmion symbols (1 bit per symbol)
        symbols_tx = []  # list of skyrmion numbers N
        for bit in bits_tx:
            n = self.modem.constellation[int(bit)]  # 0→-1, 1→+1
            symbols_tx.append(n)

        if verbose:
            print(f"[2/6] Bits → skyrmion symbols: {len(symbols_tx)} symbols")

        # Step 3-6: Transmit each symbol through the channel
        bits_rx = []
        q_values = []
        n_errors = 0

        turb_cfg = TurbulenceConfig(
            wavelength=self.wavelength,
            distance=distance,
            cn2=cn2,
            n_screens=n_screens,
            seed=self.seed,
        )
        turb_model = TurbulenceModel(turb_cfg)

        # Pre-generate phase screens once (same screens for all symbols)
        nx_dim = self.grid_x.shape[0]
        ny_dim = self.grid_x.shape[1]
        dx = self.grid_x[0, 1] - self.grid_x[0, 0]
        dz = distance / n_screens
        phase_screens = []
        for _ in range(n_screens):
            phase_screens.append(
                turb_model.generate_phase_screen((ny_dim, nx_dim), dx, dz)
            )

        for sym_idx, n_tx in enumerate(symbols_tx):
            bits_tx_sym = np.array([self.modem.constellation.index(n_tx)])

            # Generate skyrmion field for this symbol
            Ex_tx, Ey_tx, _ = self.modem.modulate(bits_tx_sym, self.grid_x, self.grid_y)

            # Propagate through turbulence (same phase screens for Ex and Ey)
            E_rx = turb_model.propagate(Ex_tx, self.grid_x, self.grid_y, phase_screens=phase_screens)
            E_rx_y = turb_model.propagate(Ey_tx, self.grid_x, self.grid_y, phase_screens=phase_screens)

            # Receiver: measure Stokes field, compute skyrmion number
            q_measured, n_rx = self.modem.demodulate(E_rx, E_rx_y)
            q_values.append(q_measured)

            # Decode bits
            bits_rx_sym = self.modem.skyrmion_to_bits(n_rx)
            bits_rx.append(int(bits_rx_sym[0]))

            # Check error
            if not np.array_equal(bits_tx_sym, bits_rx_sym):
                n_errors += 1

            if verbose and sym_idx < 5:
                print(f"  Symbol {sym_idx}: N_tx={n_tx}, Q={q_measured:+.4f}, "
                      f"N_rx={n_rx}, bit={int(bits_tx_sym[0])}→{int(bits_rx_sym[0])}")

        if verbose:
            print(f"[3/6] Propagated {len(symbols_tx)} symbols through turbulence")
            print(f"[4/6] Measured skyrmion numbers at receiver")
            print(f"[5/6] Decoded {len(bits_rx)} bits")

        # Step 6: Recover text
        bits_rx_arr = np.array(bits_rx, dtype=int)
        message_rx = bits_to_text(bits_rx_arr)

        ber = n_errors / len(symbols_tx) if symbols_tx else 0.0
        success = message_rx == message

        if verbose:
            print(f"[6/6] Recovered message: '{message_rx}'")
            print(f"Errors: {n_errors}/{len(symbols_tx)} symbols, BER = {ber:.6f}")
            print(f"Transmission {'SUCCEEDED' if success else 'FAILED'}")
            print("=" * 70)

        return {
            'message_tx': message,
            'message_rx': message_rx,
            'bits_tx': bits_tx.tolist(),
            'bits_rx': bits_rx,
            'symbols_tx': symbols_tx,
            'q_values': q_values,
            'n_errors': n_errors,
            'n_symbols': len(symbols_tx),
            'ber': ber,
            'success': success,
            'cn2': cn2,
            'distance': distance,
        }


def run_demo():
    """Run the complete end-to-end communication demo."""
    # Grid setup: 64×64 spatial grid, 3 m × 3 m
    nx, ny = 64, 64
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    channel = WaveCommChannel(
        grid_x=X,
        grid_y=Y,
        wavelength=1.55e-6,
        skyrmion_radius=1.0,
        seed=42,
    )

    # Test messages of increasing length
    test_messages = [
        "HI",           # 2 chars, 16 bits
        "HELLO",        # 5 chars, 40 bits
        "XPOCK WORKS",  # 11 chars, 88 bits
    ]

    all_passed = True
    for msg in test_messages:
        result = channel.transmit(msg, cn2=1e-14, distance=100.0, n_screens=5, verbose=True)
        if not result['success']:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED — Communication established through wave structure!")
    else:
        print("SOME TESTS FAILED — Communication needs improvement.")

    return all_passed


if __name__ == '__main__':
    run_demo()