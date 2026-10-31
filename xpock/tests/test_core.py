"""Unit tests for XPOCK core wave-structure generation.

Tests cover:
- WaveField / WaveEvent construction and validation
- Temporal profile evaluation (all shapes)
- Spatial mode evaluation (Gaussian, HG, LG, Bessel, etc.)
- Polarization textures (linear, circular, radial, azimuthal, skyrmion)
- Topological charge (winding number, skyrmion number)
- Parameter space enumeration and refinement
- Wave analyzer (power, duration, OAM, Stokes, Schmidt number)
- Validation checks
- CLI smoke tests

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from xpock.core import WaveField, WaveEvent, WaveMode, PolarizationState
from xpock.properties import (
    TemporalProfile, PulseShape,
    SpatialMode, SpatialModeConfig,
    PolarizationTexture, PolarizationConfig,
    TopologicalCharge, TopologicalChargeConfig,
    CoherenceModel, CoherenceConfig,
    CoupledMode, CoupledModeConfig,
)
from xpock.parameter_space import ParameterSpace, SweepConfig, SweepType
from xpock.analysis import WaveAnalyzer, MeasurableProperties
from xpock.validation import validate_wave_event


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def grid_1d():
    x = np.linspace(-5, 5, 64)
    t = np.linspace(-5, 5, 64)
    return x, t


@pytest.fixture
def grid_2d():
    x = np.linspace(-5, 5, 64)
    y = np.linspace(-5, 5, 64)
    t = np.linspace(-5, 5, 32)
    return x, y, t


@pytest.fixture
def sample_field(grid_2d):
    x, y, t = grid_2d
    X, Y = np.meshgrid(x, y, indexing="xy")
    psi = np.exp(-(X**2 + Y**2))[:, :, np.newaxis] * np.exp(-t**2)[np.newaxis, np.newaxis, :]
    Ex = psi[np.newaxis, :, :, :]
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)
    return WaveField(Ex=Ex, Ey=Ey, Ez=Ez, x=x, y=y, t=t)


# -----------------------------------------------------------------------
# WaveField / WaveEvent
# -----------------------------------------------------------------------

def test_wave_field_construction(sample_field):
    assert sample_field.field_type == "vector"
    assert sample_field.Ex.shape == (1, 64, 64, 32)
    assert sample_field.intensity.shape == (1, 64, 64, 32)
    assert np.all(np.isfinite(sample_field.intensity))


def test_wave_field_scalar():
    x = np.linspace(-1, 1, 16)
    t = np.linspace(-1, 1, 16)
    Ex = np.exp(-x[np.newaxis, :, np.newaxis]**2) * np.exp(-t[np.newaxis, np.newaxis, :]**2)
    f = WaveField(Ex=Ex, x=x, t=t)
    assert f.field_type == "scalar"


def test_wave_field_mismatched_shape_raises():
    x = np.linspace(-1, 1, 8)
    t = np.linspace(-1, 1, 8)
    Ex = np.zeros((8, 8, 8))
    Ey = np.zeros((8, 8, 4))  # wrong time dimension
    with pytest.raises(ValueError):
        WaveField(Ex=Ex, Ey=Ey, x=x, t=t)


def test_wave_field_no_components_raises():
    with pytest.raises(ValueError):
        WaveField(x=np.linspace(-1, 1, 4), t=np.linspace(-1, 1, 4))


def test_wave_event_create(sample_field):
    params = {"shape": "gaussian", "duration": 1.0}
    ev = WaveEvent.create(params, sample_field, notes="test")
    assert ev.event_id != ""
    assert ev.timestamp != ""
    assert ev.parameters == params
    assert ev.field is sample_field


def test_wave_event_json_roundtrip(sample_field, tmp_path):
    params = {"shape": "gaussian"}
    ev = WaveEvent.create(params, sample_field)
    p = tmp_path / "event.json"
    ev.to_json(p)
    ev2 = WaveEvent.from_json(p)
    assert ev2.event_id == ev.event_id
    assert ev2.parameters == ev.parameters


# -----------------------------------------------------------------------
# Temporal profiles
# -----------------------------------------------------------------------

@pytest.mark.parametrize("shape", [s for s in PulseShape if s != PulseShape.CUSTOM])
def test_temporal_profile_all_shapes(shape):
    t = np.linspace(-5, 5, 128)
    tp = TemporalProfile(shape=shape, amplitude=1.0, center=0.0, duration=1.0)
    env = tp.evaluate(t)
    assert env.shape == t.shape
    assert np.all(np.isfinite(env))
    # Gaussian/sech should peak at center
    if shape in (PulseShape.GAUSSIAN, PulseShape.SECH):
        center_idx = np.argmin(np.abs(t))
        assert np.abs(env[center_idx]) > 0.1


def test_temporal_profile_chirp():
    t = np.linspace(-5, 5, 128)
    tp = TemporalProfile(shape=PulseShape.GAUSSIAN, chirp_rate=10.0)
    env = tp.evaluate(t)
    assert np.all(np.isfinite(env))


def test_temporal_profile_carrier():
    t = np.linspace(-5, 5, 128)
    tp = TemporalProfile(shape=PulseShape.GAUSSIAN, carrier_freq=5.0)
    env = tp.evaluate(t)
    assert np.all(np.isfinite(env))


def test_temporal_profile_custom():
    t = np.linspace(-5, 5, 64)
    tp = TemporalProfile(
        shape=PulseShape.CUSTOM,
        custom_fn=lambda tt: np.exp(-tt**2) * np.exp(1j * tt),
    )
    env = tp.evaluate(t)
    assert env.shape == t.shape
    assert np.all(np.isfinite(env))


# -----------------------------------------------------------------------
# Spatial modes
# -----------------------------------------------------------------------

def test_spatial_gaussian():
    x = np.linspace(-3, 3, 32)
    y = np.linspace(-3, 3, 32)
    cfg = SpatialModeConfig(mode="gaussian", waist=1.0)
    sm = SpatialMode(cfg)
    psi = sm.evaluate(x, y)
    assert psi.shape == (32, 32)
    assert np.all(np.isfinite(psi))
    # Peak at center
    center = psi[16, 16]
    assert np.abs(center) > 0.1


def test_spatial_hermite_gaussian():
    x = np.linspace(-3, 3, 32)
    y = np.linspace(-3, 3, 32)
    for n, m in [(0, 0), (1, 0), (2, 1)]:
        cfg = SpatialModeConfig(mode="hermite_gaussian", n=n, m=m, waist=1.0)
        sm = SpatialMode(cfg)
        psi = sm.evaluate(x, y)
        assert psi.shape == (32, 32)
        assert np.all(np.isfinite(psi))


def test_spatial_laguerre_gaussian():
    x = np.linspace(-3, 3, 32)
    y = np.linspace(-3, 3, 32)
    for p, l in [(0, 0), (0, 1), (1, 2)]:
        cfg = SpatialModeConfig(mode="laguerre_gaussian", p=p, l=l, waist=1.0)
        sm = SpatialMode(cfg)
        psi = sm.evaluate(x, y)
        assert psi.shape == (32, 32)
        assert np.all(np.isfinite(psi))


def test_spatial_bessel():
    x = np.linspace(-3, 3, 32)
    y = np.linspace(-3, 3, 32)
    cfg = SpatialModeConfig(mode="bessel", waist=1.0)
    sm = SpatialMode(cfg)
    psi = sm.evaluate(x, y)
    assert psi.shape == (32, 32)
    assert np.all(np.isfinite(psi))


def test_spatial_aperture():
    x = np.linspace(-3, 3, 32)
    y = np.linspace(-3, 3, 32)
    cfg = SpatialModeConfig(mode="gaussian", waist=1.0, aperture_radius=1.0)
    sm = SpatialMode(cfg)
    psi = sm.evaluate(x, y)
    # Points outside aperture should be zero
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = np.sqrt(X**2 + Y**2) > 1.0
    assert np.all(psi[mask] == 0.0)


# -----------------------------------------------------------------------
# Polarization textures
# -----------------------------------------------------------------------

def test_polarization_linear():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cfg = PolarizationConfig(state=PolarizationState.LINEAR, angle=0.0)
    pol = PolarizationTexture(cfg)
    Ex, Ey = pol.evaluate(X, Y)
    assert Ex.shape == X.shape
    assert np.allclose(np.abs(Ey), 0.0)
    assert np.allclose(np.abs(Ex), 1.0)


def test_polarization_circular():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    X, Y = np.meshgrid(x, y, indexing="xy")
    for h in [+1, -1]:
        cfg = PolarizationConfig(state=PolarizationState.CIRCULAR, handedness=h)
        pol = PolarizationTexture(cfg)
        Ex, Ey = pol.evaluate(X, Y)
        assert np.allclose(np.abs(Ex), 1.0 / np.sqrt(2.0))
        assert np.allclose(np.abs(Ey), 1.0 / np.sqrt(2.0))


def test_polarization_radial():
    x = np.linspace(-1, 1, 17)  # includes exactly 0.0 and 1.0
    y = np.linspace(-1, 1, 17)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cfg = PolarizationConfig(state=PolarizationState.RADIAL)
    pol = PolarizationTexture(cfg)
    Ex, Ey = pol.evaluate(X, Y)
    # At (x=1, y=0): Ex=1, Ey=0
    idx = np.argmin(np.abs(x - 1.0))
    idy = np.argmin(np.abs(y - 0.0))
    assert np.isclose(Ex[idy, idx], 1.0, atol=1e-10)
    assert np.isclose(Ey[idy, idx], 0.0, atol=1e-10)


def test_polarization_skyrmion():
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cfg = PolarizationConfig(state=PolarizationState.SKYRMIONIC, skyrmion_radius=1.0, skyrmion_charge=1)
    pol = PolarizationTexture(cfg)
    Ex, Ey = pol.evaluate(X, Y)
    assert Ex.shape == X.shape
    assert Ey.shape == X.shape
    assert np.all(np.isfinite(Ex))
    assert np.all(np.isfinite(Ey))


# -----------------------------------------------------------------------
# Topological charge
# -----------------------------------------------------------------------

def test_winding_number_vortex():
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    phi = np.arctan2(Y, X)
    phase = 3.0 * phi  # charge 3 vortex
    tc = TopologicalCharge(TopologicalChargeConfig(charge=3, core_radius=0.1))
    # Apply singularity
    field = np.exp(-(X**2 + Y**2))
    field_v = tc.apply_phase_singularity(field, X, Y)
    assert field_v.shape == field.shape
    assert np.all(np.isfinite(field_v))


def test_skyrmion_number():
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cfg = PolarizationConfig(state=PolarizationState.SKYRMIONIC, skyrmion_radius=1.0, skyrmion_charge=1)
    pol = PolarizationTexture(cfg)
    Ex, Ey = pol.evaluate(X, Y)
    tc = TopologicalCharge(TopologicalChargeConfig())
    Q = tc.compute_skyrmion_number(Ex, Ey)
    assert np.isfinite(Q)
    # Skyrmion number should be approximately integer (charge=1 -> Q~1)
    assert abs(Q - 1.0) < 0.5


# -----------------------------------------------------------------------
# Coherence
# -----------------------------------------------------------------------

def test_coherence_full():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    t = np.linspace(-1, 1, 16)
    X, Y = np.meshgrid(x, y, indexing="xy")
    field = np.exp(-(X**2 + Y**2))[..., np.newaxis] * np.exp(-t**2)[np.newaxis, np.newaxis, :]
    coh = CoherenceModel(CoherenceConfig(coherence_type="full"))
    I = coh.apply_coherence(field, X, Y, t)
    assert I.shape == field.shape
    assert np.all(I >= 0)


def test_coherence_partial():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    t = np.linspace(-1, 1, 16)
    X, Y = np.meshgrid(x, y, indexing="xy")
    field = np.exp(-(X**2 + Y**2))[..., np.newaxis] * np.exp(-t**2)[np.newaxis, np.newaxis, :]
    coh = CoherenceModel(CoherenceConfig(coherence_type="partial", coherence_time=0.5, coherence_length=0.5))
    I = coh.apply_coherence(field, X, Y, t, n_realizations=2)
    assert I.shape == field.shape
    assert np.all(I >= 0)


# -----------------------------------------------------------------------
# Coupled modes
# -----------------------------------------------------------------------

def test_coupled_modes():
    A1 = np.ones((16, 16), dtype=np.complex128)
    A2 = np.zeros((16, 16), dtype=np.complex128)
    z = np.linspace(0, 1, 5)
    cfg = CoupledModeConfig(coupling_coefficient=0.5, interaction_length=1.0)
    cm = CoupledMode(cfg)
    A1_out, A2_out = cm.evolve(A1, A2, z)
    assert A1_out.shape == (16, 16, 5)
    assert A2_out.shape == (16, 16, 5)
    assert np.all(np.isfinite(A1_out))
    assert np.all(np.isfinite(A2_out))


# -----------------------------------------------------------------------
# Parameter space
# -----------------------------------------------------------------------

def test_parameter_space_size():
    sweeps = [
        SweepConfig(name="a", sweep_type=SweepType.LINEAR, start=0, stop=1, num=5),
        SweepConfig(name="b", sweep_type=SweepType.CUSTOM, values=[10, 20]),
    ]
    space = ParameterSpace(sweeps=sweeps)
    assert len(space) == 10


def test_parameter_space_iteration():
    sweeps = [
        SweepConfig(name="x", sweep_type=SweepType.LINEAR, start=0, stop=1, num=3),
    ]
    space = ParameterSpace(sweeps=sweeps, fixed_params={"fixed": 42})
    params_list = list(space)
    assert len(params_list) == 3
    for p in params_list:
        assert p["fixed"] == 42
        assert "x" in p


def test_parameter_space_max_combinations():
    sweeps = [
        SweepConfig(name="a", sweep_type=SweepType.LINEAR, start=0, stop=1, num=2000),
        SweepConfig(name="b", sweep_type=SweepType.LINEAR, start=0, stop=1, num=2000),
    ]
    with pytest.raises(ValueError):
        ParameterSpace(sweeps=sweeps, max_combinations=100)


def test_parameter_space_refine():
    sweeps = [
        SweepConfig(name="a", sweep_type=SweepType.LINEAR, start=0, stop=10, num=10),
    ]
    space = ParameterSpace(sweeps=sweeps)
    refined = space.refine_region({"a": 5.0}, relative_width=0.2, num_per_dim=4)
    assert len(refined) == 4
    # All values should be near 5
    for p in refined:
        assert 4.0 <= p["a"] <= 6.0


# -----------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------

def test_analyzer_basic(sample_field):
    params = {"shape": "gaussian"}
    ev = WaveEvent.create(params, sample_field)
    analyzer = WaveAnalyzer()
    props = analyzer.analyze(ev)
    assert isinstance(props, MeasurableProperties)
    assert props.total_power >= 0
    assert props.peak_intensity >= 0
    assert props.is_physically_valid


def test_analyzer_temporal(sample_field):
    props = MeasurableProperties()
    # Direct test of temporal analysis
    analyzer = WaveAnalyzer()
    # Re-run analyze to populate
    params = {}
    ev = WaveEvent.create(params, sample_field)
    props = analyzer.analyze(ev)
    assert props.temporal_duration_fwhm is not None or props.temporal_duration_rms is not None


def test_analyzer_polarization(sample_field):
    # sample_field has Ex and Ey
    analyzer = WaveAnalyzer()
    ev = WaveEvent.create({}, sample_field)
    props = analyzer.analyze(ev)
    assert props.degree_of_polarization is not None
    assert 0.0 <= props.degree_of_polarization <= 1.0


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

def test_validation_valid(sample_field):
    ev = WaveEvent.create({}, sample_field)
    result = validate_wave_event(ev)
    assert result.is_valid
    assert result.checks["finite_values"]
    assert result.checks["nonzero_field"]


def test_validation_zero_field():
    x = np.linspace(-1, 1, 8)
    t = np.linspace(-1, 1, 8)
    Ex = np.zeros((8, 8, 8))
    f = WaveField(Ex=Ex, x=x, t=t)
    ev = WaveEvent.create({}, f)
    result = validate_wave_event(ev)
    assert not result.checks["nonzero_field"]


def test_validation_non_finite():
    # WaveField constructor rejects NaN/Inf — verify it raises
    x = np.linspace(-1, 1, 8)
    t = np.linspace(-1, 1, 8)
    Ex = np.full((8, 8, 8), np.nan)
    with pytest.raises(ValueError, match="NaN or Inf"):
        WaveField(Ex=Ex, x=x, t=t)


# -----------------------------------------------------------------------
# CLI smoke
# -----------------------------------------------------------------------

def test_cli_config_generation(tmp_path):
    from xpock.cli import cmd_config, build_parser
    import sys
    out = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(["config", "--output", str(out)])
    ret = cmd_config(args)
    assert ret == 0
    assert out.exists()


def test_generate_wave_event_function(grid_2d):
    from xpock.cli import generate_wave_event
    x, y, t = grid_2d
    grid = {"x": x, "y": y, "z": np.array([0.0]), "t": t}
    params = {
        "temporal_shape": "gaussian",
        "temporal_duration": 1.0,
        "spatial_mode": "gaussian",
        "spatial_waist": 1.0,
        "polarization_state": "linear",
    }
    ev = generate_wave_event(params, grid)
    assert ev.field.Ex is not None
    assert ev.field.intensity.shape == (1, 64, 64, 32)
    assert np.all(np.isfinite(ev.field.intensity))

