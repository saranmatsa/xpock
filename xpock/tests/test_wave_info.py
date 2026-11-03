"""
XPOCK — Tests for wave_info modules

Tests for:
- turbulence.py — Kolmogorov/von Karman phase screen propagation
- skyrmion_modem.py — Skyrmion number modulation/demodulation
- ber_simulation.py — End-to-end BER simulation
- comparison.py — Comparison with OAM and intensity modulation

All tests are mathematical model validations; they do not establish
physical feasibility.
"""

import numpy as np
import importlib.util
import sys
import pytest


# ----------------------------------------------------------------------
# Load wave_info modules directly (no install required)
# ----------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


turbulence_mod = _load_module(
    'xpock.wave_info.turbulence_test',
    'src/xpock/wave_info/turbulence.py',
)
skyrmion_modem_mod = _load_module(
    'xpock.wave_info.skyrmion_modem_test',
    'src/xpock/wave_info/skyrmion_modem.py',
)
ber_sim_mod = _load_module(
    'xpock.wave_info.ber_simulation_test',
    'src/xpock/wave_info/ber_simulation.py',
)

TurbulenceConfig = turbulence_mod.TurbulenceConfig
TurbulenceModel = turbulence_mod.TurbulenceModel
ModemConfig = skyrmion_modem_mod.ModemConfig
SkyrmionModem = skyrmion_modem_mod.SkyrmionModem
BERSimulator = ber_sim_mod.BERSimulator


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _make_grid(nx=64, ny=64, extent=3.0):
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y


# ----------------------------------------------------------------------
# Turbulence tests
# ----------------------------------------------------------------------
def test_turbulence_config_defaults():
    """TurbulenceConfig has sensible defaults."""
    cfg = TurbulenceConfig()
    assert cfg.cn2 == 1e-14
    assert cfg.wavelength == 1.55e-6
    assert cfg.distance == 100.0
    assert cfg.n_screens == 10
    assert cfg.seed is None


def test_turbulence_power_conservation():
    """Power is conserved through turbulence propagation (unitary split-step)."""
    X, Y = _make_grid()
    E_in = np.exp(-(X**2 + Y**2)).astype(np.complex128)
    power_in = np.sum(np.abs(E_in)**2)

    cfg = TurbulenceConfig(cn2=1e-13, distance=100.0, n_screens=10, seed=42)
    model = TurbulenceModel(cfg)
    E_out = model.propagate(E_in, X, Y)
    power_out = np.sum(np.abs(E_out)**2)

    assert np.isclose(power_in, power_out, rtol=1e-3), (
        f"Power not conserved: {power_in} -> {power_out}"
    )


def test_turbulence_phase_screen_amplitude():
    """Phase screen amplitude scales correctly with Cn²."""
    cfg_low = TurbulenceConfig(cn2=1e-15, distance=100.0, seed=42)
    cfg_high = TurbulenceConfig(cn2=1e-10, distance=100.0, seed=42)
    model_low = TurbulenceModel(cfg_low)
    model_high = TurbulenceModel(cfg_high)

    dx = 6.0 / 64
    phi_low = model_low.generate_phase_screen((64, 64), dx, 100.0)
    phi_high = model_high.generate_phase_screen((64, 64), dx, 100.0)

    # Higher Cn² should give larger phase variations
    assert np.std(phi_high) > np.std(phi_low), (
        f"Phase std not scaling with Cn²: {np.std(phi_low)} vs {np.std(phi_high)}"
    )
    # At Cn²=1e-10, phase std should be order 10+ rad (strong turbulence)
    assert np.std(phi_high) > 1.0, (
        f"Phase std too low at Cn²=1e-10: {np.std(phi_high)}"
    )


def test_turbulence_rytov_variance():
    """Rytov variance computation is finite and positive."""
    cfg = TurbulenceConfig(cn2=1e-13, distance=1000.0, wavelength=1.55e-6)
    model = TurbulenceModel(cfg)
    sigma_R2 = model.compute_rytov_variance()
    assert sigma_R2 > 0
    assert np.isfinite(sigma_R2)


# ----------------------------------------------------------------------
# Skyrmion modem tests
# ----------------------------------------------------------------------
def test_modem_constellation_bits():
    """Modem computes correct bits per symbol from constellation size."""
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)
    assert modem.n_bits == 1
    assert modem.n_states == 2
    assert modem.constellation == [-1, 1]


def test_modem_modulate_demodulate_roundtrip():
    """Modulate then demodulate recovers the original skyrmion number.

    Uses 2-state constellation [-1, 1] (1 bit/symbol) where Q values
    (-0.61, +0.61) are well-separated and distinguishable.
    """
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)

    for n_tx in [-1, 1]:
        bit_idx = modem.constellation.index(n_tx)
        bits_tx = np.array([bit_idx])
        Ex, Ey, n_returned = modem.modulate(bits_tx, X, Y)
        assert n_returned == n_tx
        q_measured, n_rx = modem.demodulate(Ex, Ey)
        assert n_rx == n_tx, f"N={n_tx}: demodulated to {n_rx}, Q={q_measured}"


def test_modem_bits_to_skyrmion_mapping():
    """bits_to_skyrmion maps bit strings to constellation values."""
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)

    # Bit 0 -> constellation[0] = -1
    assert modem.bits_to_skyrmion(np.array([0])) == -1
    # Bit 1 -> constellation[1] = 1
    assert modem.bits_to_skyrmion(np.array([1])) == 1


def test_modem_skyrmion_to_bits_mapping():
    """skyrmion_to_bits maps skyrmion numbers back to bit strings.

    Requires exact constellation membership: values outside the
    constellation are invalid input (nearest-neighbor hard decision is the
    demodulator's job, via _nearest_constellation).
    """
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-1, 1],
    )
    modem = SkyrmionModem(cfg)

    assert np.array_equal(modem.skyrmion_to_bits(-1), np.array([0]))
    assert np.array_equal(modem.skyrmion_to_bits(1), np.array([1]))

    # Out-of-constellation values are rejected as invalid input
    with pytest.raises(ValueError):
        modem.skyrmion_to_bits(-2)  # nearest to -1, but not a member
    with pytest.raises(ValueError):
        modem.skyrmion_to_bits(2)   # nearest to 1, but not a member
    with pytest.raises(ValueError):
        modem.skyrmion_to_bits(99)


def test_modem_non_power_of_two_constellation():
    """All constellation states must be reachable (no silent discard).

    5 states -> n_bits = ceil(log2(5)) = 3; indices 0..4 valid,
    indices 5..7 rejected as invalid input.
    """
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6,
        grid_x=X,
        grid_y=Y,
        skyrmion_radius=1.0,
        constellation=[-2, -1, 0, 1, 2],
    )
    modem = SkyrmionModem(cfg)
    assert modem.n_bits == 3
    assert modem.n_states == 5

    # Every state reachable via its index bit string
    for n_tx in modem.constellation:
        bit_idx = modem.constellation.index(n_tx)
        bits_tx = np.array(
            [(bit_idx >> i) & 1 for i in range(modem.n_bits - 1, -1, -1)]
        )
        assert modem.bits_to_skyrmion(bits_tx) == n_tx
        assert np.array_equal(modem.skyrmion_to_bits(n_tx), bits_tx)

    # Unused indices (>= n_states) are invalid input
    with pytest.raises(ValueError):
        modem.bits_to_skyrmion(np.array([1, 1, 1]))  # index 7
    with pytest.raises(ValueError):
        modem.bits_to_skyrmion(np.array([1, 1, 0]))  # index 6

    # Wrong group length and non-binary values are invalid input
    with pytest.raises(ValueError):
        modem.bits_to_skyrmion(np.array([0, 1]))  # incomplete final group
    with pytest.raises(ValueError):
        modem.bits_to_skyrmion(np.array([0, 1, 2]))  # not a bit


# ----------------------------------------------------------------------
# BER simulation tests
# ----------------------------------------------------------------------
def test_ber_simulator_initialization():
    """BERSimulator initializes with correct grid and parameters."""
    X, Y = _make_grid()
    sim = BERSimulator(grid_x=X, grid_y=Y, wavelength=1.55e-6, seed=42)
    assert sim.wavelength == 1.55e-6
    assert sim.skyrmion_radius == 1.0
    assert sim.grid_x.shape == X.shape


def test_ber_simulator_skm_zero_turbulence():
    """BER=0 with no turbulence (Cn²=0)."""
    X, Y = _make_grid(extent=3.0)
    sim = BERSimulator(grid_x=X, grid_y=Y, wavelength=1.55e-6, seed=42)

    results = sim.run_skm_ber(
        constellation=[-1, 1],
        cn2_values=[0.0],
        n_trials_per_config=20,
        distance=100.0,
        n_screens=5,
        verbose=False,
    )

    assert len(results) == 2  # N=-1 and N=1
    for r in results:
        assert r.ber == 0.0, f"BER should be 0 with no turbulence, got {r.ber}"
        assert r.n_errors == 0


def test_ber_simulator_skm_weak_turbulence():
    """BER=0 at weak turbulence (Cn²=1e-14)."""
    X, Y = _make_grid(extent=3.0)
    sim = BERSimulator(grid_x=X, grid_y=Y, wavelength=1.55e-6, seed=42)

    results = sim.run_skm_ber(
        constellation=[-1, 1],
        cn2_values=[1e-14],
        n_trials_per_config=30,
        distance=100.0,
        n_screens=5,
        verbose=False,
    )

    for r in results:
        assert r.ber == 0.0, f"BER should be 0 at Cn²=1e-14, got {r.ber}"


def test_ber_result_fields():
    """BERResult has all required fields."""
    X, Y = _make_grid()
    sim = BERSimulator(grid_x=X, grid_y=Y, seed=42)
    results = sim.run_skm_ber(
        constellation=[-1, 1],
        cn2_values=[1e-14],
        n_trials_per_config=10,
        distance=100.0,
        n_screens=3,
        verbose=False,
    )
    r = results[0]
    assert hasattr(r, 'n_tx')
    assert hasattr(r, 'cn2')
    assert hasattr(r, 'ber')
    assert hasattr(r, 'ser')
    assert hasattr(r, 'n_trials')
    assert hasattr(r, 'n_errors')
    assert hasattr(r, 'n_symbols')
    assert hasattr(r, 'q_mean')
    assert hasattr(r, 'q_std')
    assert hasattr(r, 'confusion_matrix')
    assert isinstance(r.confusion_matrix, dict)
    assert 0.0 <= r.ber <= 1.0


# ----------------------------------------------------------------------
# Regression tests for Phase III/IV/VI repairs
# ----------------------------------------------------------------------


def test_split_step_matches_direct_fresnel():
    """Split-step propagation with cn2=0 must equal direct Fresnel
    propagation over the *configured* distance (not 2x it), independent
    of n_screens. Zero distance is the identity.

    Regression for the split-step kernel bug where the Fresnel kernel
    was applied at full dz in both half-steps, doubling the path.
    """
    from scipy.fft import fft2, ifft2, fftfreq

    n = 64
    x = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    dx = x[1] - x[0]
    k = 2.0 * np.pi / 1.55e-6
    kx = 2.0 * np.pi * fftfreq(n, dx)
    KX, KY = np.meshgrid(kx, kx)
    K2 = KX**2 + KY**2

    def fresnel_direct(E, L):
        return ifft2(fft2(E) * np.exp(-1j * K2 * L / (2.0 * k)))

    E_in = np.exp(-(X**2 + Y**2) / 0.5**2).astype(np.complex128)

    for L in [1.0, 10.0, 100.0, 1000.0]:
        for n_screens in [1, 3, 10]:
            cfg = TurbulenceConfig(
                cn2=0.0, wavelength=1.55e-6, distance=L,
                n_screens=n_screens, seed=1,
            )
            E_split = TurbulenceModel(cfg).propagate(E_in, X, Y)
            err = float(np.max(np.abs(E_split - fresnel_direct(E_in, L))))
            assert err < 1e-9, (
                f"split-step != direct Fresnel at L={L}, n_screens={n_screens}: {err}"
            )

    # Zero distance -> identity
    cfg = TurbulenceConfig(
        cn2=0.0, wavelength=1.55e-6, distance=0.0, n_screens=5, seed=1
    )
    E0 = TurbulenceModel(cfg).propagate(E_in, X, Y)
    assert float(np.max(np.abs(E0 - E_in))) < 1e-12


class _WrongDemodModem(SkyrmionModem):
    """Modem whose demodulate returns the next constellation value
    cyclically (deterministic wrong decision) — exercises bit-vs-symbol
    error counting independently of turbulence physics."""

    def demodulate(self, Ex, Ey):
        q, _ = super().demodulate(Ex, Ey)
        n_rx = self._nearest_constellation(q)
        idx = self.constellation.index(n_rx)
        wrong = self.constellation[(idx + 1) % len(self.constellation)]
        return float(wrong), wrong


def test_ber_counting_bit_errors_independent_of_symbol_errors():
    """BER and SER are distinct quantities.

    4-state constellation [-2,-1,1,2] (2 bits/symbol). Deterministically
    demodulate to the next constellation state cyclically:
      idx0 [0,0]->idx1 [0,1]: 1 bit error
      idx1 [0,1]->idx2 [1,0]: 2 bit errors
      idx2 [1,0]->idx3 [1,1]: 1 bit error
      idx3 [1,1]->idx0 [0,0]: 2 bit errors
    Uniform symbols -> mean 1.5 bit errors / 2 bits => BER = 0.75,
    SER = 1.0. If bit and symbol counting were conflated, BER would
    incorrectly equal SER.
    """
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6, grid_x=X, grid_y=Y,
        skyrmion_radius=1.0, constellation=[-2, -1, 1, 2],
    )
    sim = BERSimulator(grid_x=X, grid_y=Y, wavelength=1.55e-6, seed=42)
    modem = _WrongDemodModem(cfg)
    n_bits = modem.n_bits  # 2
    n_errors = 0
    n_symbol_errors = 0
    n_symbols = 0
    cfg_t = TurbulenceConfig(
        cn2=0.0, wavelength=1.55e-6, distance=0.0, n_screens=3, seed=1
    )  # distance 0: propagate is identity; this test checks counting, not physics
    turb = TurbulenceModel(cfg_t)
    # Each state modulates its own field; demodulate returns the next
    # constellation state cyclically (deterministic wrong decision).
    for n_tx in modem.constellation:
        bit_idx = modem.constellation.index(n_tx)
        bits_tx = np.array(
            [(bit_idx >> i) & 1 for i in range(n_bits - 1, -1, -1)]
        )
        Ex_tx, Ey_tx, _ = modem.modulate(bits_tx, X, Y)
        p_tx = float(np.sum(np.abs(Ex_tx) ** 2 + np.abs(Ey_tx) ** 2) * sim.dx * sim.dy)
        scale = np.sqrt(1.0 / p_tx)
        Ex_tx = Ex_tx * scale
        Ey_tx = Ey_tx * scale
        for _ in range(10):
            E_rx = turb.propagate(Ex_tx, X, Y)
            E_rx_y = turb.propagate(Ey_tx, X, Y)
            q_measured, n_rx = modem.demodulate(E_rx, E_rx_y)
            bits_rx = modem.skyrmion_to_bits(n_rx)
            n_errors += int(np.sum(bits_tx != bits_rx))
            n_symbol_errors += int(np.sum(bits_tx != bits_rx) > 0)
            n_symbols += 1
    assert n_symbols == 40
    assert n_errors == 10 * (1 + 2 + 1 + 2)  # 60
    assert n_symbol_errors == 40
    ber = n_errors / (n_symbols * n_bits)
    ser = n_symbol_errors / n_symbols
    assert abs(ber - 0.75) < 1e-12
    assert ser == 1.0
    assert ber != ser  # BER and SER must NOT coincide


def test_skm_joint_unit_total_power():
    """The SKM two-component field normalizes to unit TOTAL power
    (|Ex|^2 + |Ey|^2 integrated), matching scalar baselines — not
    unit power per component (which would be 2x launch power), and not
    the raw grid-pixel count (~4000x)."""
    X, Y = _make_grid()
    cfg = ModemConfig(
        wavelength=1.55e-6, grid_x=X, grid_y=Y,
        skyrmion_radius=1.0, constellation=[-2, -1, 0, 1, 2],
    )
    modem = SkyrmionModem(cfg)
    Ex, Ey, _ = modem.modulate(np.array([0, 0, 0]), X, Y)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    raw = float(np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2) * dx * dy)
    assert raw > 10.0  # raw power really is the pixel-count scale
    scale = np.sqrt(1.0 / raw)
    total = float(np.sum(np.abs(Ex * scale) ** 2 + np.abs(Ey * scale) ** 2) * dx * dy)
    assert abs(total - 1.0) < 1e-12


def test_ber_ci_zero_errors_upper_bound_only():
    """Zero observed errors yields BER=0 but a non-zero CI upper bound
    (rule-of-three style): never claim true BER is exactly zero."""
    X, Y = _make_grid()
    sim = BERSimulator(grid_x=X, grid_y=Y, wavelength=1.55e-6, seed=42)
    results = sim.run_skm_ber(
        constellation=[-1, 1],
        cn2_values=[0.0],
        n_trials_per_config=20,
        distance=100.0,
        n_screens=3,
        verbose=False,
    )
    for r in results:
        assert r.ber == 0.0
        assert r.n_errors == 0
        assert r.ber_ci_lo == 0.0
        assert 0.0 < r.ber_ci_hi < 1.0  # upper bound, not exact zero


# ----------------------------------------------------------------------
# Run all tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    tests = [
        test_turbulence_config_defaults,
        test_turbulence_power_conservation,
        test_turbulence_phase_screen_amplitude,
        test_turbulence_rytov_variance,
        test_split_step_matches_direct_fresnel,
        test_modem_constellation_bits,
        test_modem_modulate_demodulate_roundtrip,
        test_modem_bits_to_skyrmion_mapping,
        test_modem_skyrmion_to_bits_mapping,
        test_modem_non_power_of_two_constellation,
        test_ber_simulator_initialization,
        test_ber_simulator_skm_zero_turbulence,
        test_ber_simulator_skm_weak_turbulence,
        test_ber_counting_bit_errors_independent_of_symbol_errors,
        test_skm_joint_unit_total_power,
        test_ber_ci_zero_errors_upper_bound_only,
        test_ber_result_fields,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)