"""
XPOCK — HX-241 Physically-Based Impairment Models

Converts XPOCK carrier wave channel models into hx-sim NetworkImpairment
parameters, providing physically-grounded impairment profiles for HX-241
adversarial and conformance testing.

hx-sim's NetworkImpairment has generic parameters (packet_loss_rate,
corruption_rate, base_latency_us, jitter_us, bandwidth_bps). This module
derives those from physical carrier models (path loss, delay spread,
Doppler spread, coherence bandwidth, SNR) rather than using arbitrary
values.

HX-241 architectural principle: impairment models are physical carrier
properties, not protocol assumptions. The protocol treats carriers as
opaque; the impairment parameters describe what the carrier does to
the signal, not what the protocol expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path
import numpy as np

from xpock.hx241.carrier import (
    CarrierWaveConfig,
    CarrierChannelModel,
    CarrierMedium,
    CarrierWaveModeler,
)


class ImpairmentModel(Enum):
    """Physically-based impairment model types."""

    FREE_SPACE_PATH_LOSS = "free_space_path_loss"
    ATMOSPHERIC_TURBULENCE = "atmospheric_turbulence"
    MULTIPATH_FADING = "multipath_fading"
    FIBER_DISPERSION = "fiber_dispersion"
    ACOUSTIC_ABSORPTION = "acoustic_absorption"
    MAGNETIC_COUPLING_LOSS = "magnetic_coupling_loss"
    PRESSURE_WAVE_ATTENUATION = "pressure_wave_attenuation"
    COMPOSITE = "composite"  # Combined effects


@dataclass
class ImpairmentProfile:
    """Physically-based impairment profile for a carrier link.

    Attributes
    ----------
    model : ImpairmentModel
        Primary impairment mechanism.
    path_loss_db : float
        Total path loss in dB.
    snr_db : float
        Signal-to-noise ratio in dB.
    delay_spread_s : float
        RMS delay spread in seconds.
    coherence_bandwidth_hz : float
        Coherence bandwidth in Hz.
    doppler_spread_hz : float
        Doppler spread in Hz.
    derived_params : dict
        Parameters converted to hx-sim NetworkImpairment format.
    medium : str
        Carrier medium name.
    notes : list
        Human-readable observations.
    """

    model: ImpairmentModel
    path_loss_db: float
    snr_db: float
    delay_spread_s: float
    coherence_bandwidth_hz: float
    doppler_spread_hz: float
    derived_params: Dict[str, Any]
    medium: str
    notes: List[str] = field(default_factory=list)

    def to_hx_sim_impairment(self) -> Dict[str, Any]:
        """Return parameters suitable for hx-sim NetworkImpairment."""
        return dict(self.derived_params)


class ImpairmentModeler:
    """
    Derives physically-based hx-sim impairment parameters from carrier
    channel models.

    The mapping from physical channel properties to impairment parameters:
    - Path loss + SNR → packet_loss_rate (higher loss = higher loss rate)
    - SNR → corruption_rate (lower SNR = higher bit corruption)
    - Delay spread → jitter_us (multipath causes delay variation)
    - Coherence bandwidth → bandwidth_bps (limits usable bandwidth)
    - Doppler spread → additional jitter (mobility-induced frequency shift)
    - Base latency → propagation delay (distance / speed)
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./xpock_hx241_impairment")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._modeler: Optional[CarrierWaveModeler] = None

    def _get_modeler(self) -> CarrierWaveModeler:
        if self._modeler is None:
            self._modeler = CarrierWaveModeler(workspace=str(self.workspace / "carrier"))
        return self._modeler

    def model_impairment(
        self,
        config: CarrierWaveConfig,
    ) -> ImpairmentProfile:
        """
        Model a carrier link and produce physically-based impairment parameters.

        Steps:
        1. Build channel model using CarrierWaveModeler
        2. Extract physical parameters (path loss, SNR, delay spread, etc.)
        3. Map to hx-sim NetworkImpairment parameters
        4. Return ImpairmentProfile
        """
        modeler = self._get_modeler()
        channel = modeler.model_carrier(config)

        # Determine primary impairment model
        primary_model = self._classify_impairment(channel.config.medium, channel)

        # Derive hx-sim parameters from physical parameters
        derived = self._derive_impairment_params(channel)

        notes = list(channel.medium_notes)
        notes.append(f"Primary impairment model: {primary_model.value}")
        notes.append(f"SNR: {channel.snr_db:.1f} dB")
        notes.append(f"Path loss: {channel.path_loss_db:.1f} dB")

        return ImpairmentProfile(
            model=primary_model,
            path_loss_db=channel.path_loss_db,
            snr_db=channel.snr_db,
            delay_spread_s=channel.delay_spread_s,
            coherence_bandwidth_hz=channel.coherence_bandwidth_hz,
            doppler_spread_hz=channel.doppler_spread_hz,
            derived_params=derived,
            medium=channel.config.medium.value,
            notes=notes,
        )

    def _classify_impairment(
        self,
        medium: CarrierMedium,
        channel: CarrierChannelModel,
    ) -> ImpairmentModel:
        """Classify the primary impairment mechanism for a medium."""
        mapping = {
            CarrierMedium.FREE_SPACE_OPTICAL: ImpairmentModel.ATMOSPHERIC_TURBULENCE,
            CarrierMedium.RADIO_FREQUENCY: ImpairmentModel.MULTIPATH_FADING,
            CarrierMedium.ACOUSTIC: ImpairmentModel.ACOUSTIC_ABSORPTION,
            CarrierMedium.OPTICAL_FIBER: ImpairmentModel.FIBER_DISPERSION,
            CarrierMedium.MAGNETIC_INDUCTION: ImpairmentModel.MAGNETIC_COUPLING_LOSS,
            CarrierMedium.PRESSURE_WAVE: ImpairmentModel.PRESSURE_WAVE_ATTENUATION,
            CarrierMedium.CUSTOM: ImpairmentModel.COMPOSITE,
        }
        return mapping.get(medium, ImpairmentModel.COMPOSITE)

    def _derive_impairment_params(
        self,
        channel: CarrierChannelModel,
    ) -> Dict[str, Any]:
        """
        Map physical channel parameters to hx-sim NetworkImpairment fields.

        All mappings are heuristic and should be calibrated against real
        carrier measurements. They are starting points, not ground truth.
        """
        snr_db = channel.snr_db
        path_loss_db = channel.path_loss_db
        delay_spread = channel.delay_spread_s
        coherence_bw = channel.coherence_bandwidth_hz
        doppler = channel.doppler_spread_hz

        # --- Packet loss rate from path loss ---
        # Higher path loss → higher probability of packet loss
        # Model: packet_loss = 1 - 10^(-path_loss/20) clamped to [0, 0.9]
        path_loss_linear = 10.0 ** (-path_loss_db / 20.0)
        packet_loss_rate = float(np.clip(1.0 - path_loss_linear, 0.0, 0.9))

        # --- Corruption rate from SNR ---
        # Lower SNR → higher bit error rate → higher corruption
        # Model: corruption = 1 / (1 + SNR_linear) clamped to [0, 0.5]
        snr_linear = 10.0 ** (snr_db / 10.0)
        corruption_rate = float(np.clip(1.0 / (1.0 + snr_linear), 0.0, 0.5))

        # --- Base latency from propagation delay ---
        # Already computed in channel model as base_latency_us
        base_latency_us = channel.impairment_params.get("base_latency_us", 0)

        # --- Jitter from delay spread and Doppler ---
        # Delay spread causes multipath delay variation
        # Doppler causes frequency shift → timing jitter
        jitter_from_delay = int(delay_spread * 1e6 / 2.0)  # half of RMS delay spread
        jitter_from_doppler = int(doppler * 0.1)  # heuristic
        jitter_us = max(jitter_from_delay, jitter_from_doppler)

        # --- Bandwidth from coherence bandwidth ---
        # Coherence bandwidth limits usable signal bandwidth
        bandwidth_bps = int(min(coherence_bw, channel.impairment_params.get("bandwidth_bps", float("inf"))))

        return {
            "packet_loss_rate": packet_loss_rate,
            "duplication_rate": 0.0,
            "reordering_depth": 0,
            "corruption_rate": corruption_rate,
            "base_latency_us": base_latency_us,
            "jitter_us": jitter_us,
            "bandwidth_bps": bandwidth_bps,
            "partition": None,
            "enabled": True,
            "source": "xpock_impairment_modeler",
            "medium": channel.config.medium.value,
            "path_loss_db": path_loss_db,
            "snr_db": snr_db,
            "delay_spread_s": delay_spread,
            "coherence_bandwidth_hz": coherence_bw,
            "doppler_spread_hz": doppler,
        }

    def generate_impairment_scenario(
        self,
        medium: CarrierMedium,
        distance_m: float,
        frequency_hz: float,
        transmit_power_w: float = 1.0,
        **medium_kwargs,
    ) -> ImpairmentProfile:
        """
        Convenience: create an impairment profile for a carrier link scenario.

        Parameters
        ----------
        medium : CarrierMedium
            Type of carrier medium.
        distance_m : float
            Link distance in meters.
        frequency_hz : float
            Carrier frequency in Hz.
        transmit_power_w : float
            Transmit power in watts.
        **medium_kwargs
            Medium-specific parameters (attenuation, dispersion, etc.).

        Returns
        -------
        ImpairmentProfile with derived hx-sim parameters.
        """
        config = CarrierWaveConfig(
            medium=medium,
            frequency_hz=frequency_hz,
            distance_m=distance_m,
            transmit_power_w=transmit_power_w,
            medium_specific=medium_kwargs,
        )
        return self.model_impairment(config)

    def batch_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> List[ImpairmentProfile]:
        """
        Generate impairment profiles for multiple link scenarios.

        Each scenario dict should contain keys for model_impairment config:
        medium, distance_m, frequency_hz, transmit_power_w, plus medium-specific kwargs.
        """
        profiles = []
        for scenario in scenarios:
            profile = self.generate_impairment_scenario(**scenario)
            profiles.append(profile)
        return profiles


# -----------------------------------------------------------------------
# HX-241 hx-sim integration helper
# -----------------------------------------------------------------------

def impairment_to_hx_sim_dict(
    profile: ImpairmentProfile,
) -> Dict[str, Any]:
    """
    Convert an ImpairmentProfile to a dict matching hx-sim NetworkImpairment
    fields. Can be passed directly to hx-sim's ImpairmentEngine.set_config().

    Returns dict with keys:
    packet_loss_rate, duplication_rate, reordering_depth, corruption_rate,
    base_latency_us, jitter_us, bandwidth_bps, partition, enabled
    """
    return profile.to_hx_sim_impairment()


def create_link_scenario_library() -> List[ImpairmentProfile]:
    """
    Create a library of common HX-241 link scenarios for testing.

    Returns impairment profiles for:
    - Short-range free-space optical (10m, 200THz)
    - Long-range RF (1km, 2.4GHz)
    - Underwater acoustic (500m, 20kHz)
    - Optical fiber (10km, 193THz)
    - Near-field magnetic induction (1m, 10MHz)
    - Indoor pressure wave (5m, 40kHz)
    """
    modeler = ImpairmentModeler()

    scenarios = [
        {
            "medium": CarrierMedium.FREE_SPACE_OPTICAL,
            "distance_m": 10.0,
            "frequency_hz": 200e12,  # 1550 nm
            "transmit_power_w": 0.001,  # 1 mW
            "attenuation_db_per_km": 0.2,
            "cn2": 1e-14,
        },
        {
            "medium": CarrierMedium.RADIO_FREQUENCY,
            "distance_m": 1000.0,
            "frequency_hz": 2.4e9,
            "transmit_power_w": 0.1,  # 100 mW
            "rms_delay_spread_us": 0.5,
            "shadow_std_db": 6.0,
            "doppler_spread_hz": 10.0,
        },
        {
            "medium": CarrierMedium.ACOUSTIC,
            "distance_m": 500.0,
            "frequency_hz": 20e3,
            "transmit_power_w": 1.0,
            "sound_speed_m_s": 1500.0,
            "absorption_db_per_km": 0.5,
            "spreading": "cylindrical",
            "doppler_spread_hz": 1.0,
        },
        {
            "medium": CarrierMedium.OPTICAL_FIBER,
            "distance_m": 10000.0,
            "frequency_hz": 193e12,  # 1550 nm
            "transmit_power_w": 0.001,
            "attenuation_db_per_km": 0.2,
            "dispersion_ps_nm_km": 17.0,
            "source_linewidth_nm": 0.1,
            "gamma_w_km": 1.3,
        },
        {
            "medium": CarrierMedium.MAGNETIC_INDUCTION,
            "distance_m": 1.0,
            "frequency_hz": 10e6,
            "transmit_power_w": 0.01,
        },
        {
            "medium": CarrierMedium.PRESSURE_WAVE,
            "distance_m": 5.0,
            "frequency_hz": 40e3,
            "transmit_power_w": 0.1,
            "sound_speed_m_s": 343.0,
            "medium_type": "air",
            "absorption_db_per_m": 0.01,
            "doppler_spread_hz": 0.1,
        },
    ]

    return modeler.batch_scenarios(scenarios)
