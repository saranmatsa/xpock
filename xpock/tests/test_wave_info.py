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
    """skyrmion_to_bits maps skyrmion numbers back to bit strings."""
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

    # Nearest constellation for out-of-range value
    assert np.array_equal(modem.skyrmion_to_bits(-2), np.array([0]))  # nearest to -1
    assert np.array_equal(modem.skyrmion_to_bits(2), np.array([1]))   # nearest to 1


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
# Run all tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    tests = [
        test_turbulence_config_defaults,
        test_turbulence_power_conservation,
        test_turbulence_phase_screen_amplitude,
        test_turbulence_rytov_variance,
        test_modem_constellation_bits,
        test_modem_modulate_demodulate_roundtrip,
        test_modem_bits_to_skyrmion_mapping,
        test_modem_skyrmion_to_bits_mapping,
        test_ber_simulator_initialization,
        test_ber_simulator_skm_zero_turbulence,
        test_ber_simulator_skm_weak_turbulence,
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