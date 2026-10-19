"""
XPOCK — HX-241 Adapters Module

Adapters that map XPOCK's computational outputs to HX-241's Rust types and
interfaces. These are pure Python functions that produce data structures
matching HX-241's Rust schemas. The output is serialized to JSON and can be
deserialized by Rust's serde.

HX-241 architectural compliance:
- No IP/TCP/UDP/QUIC/DNS/HTTP dependencies
- HX-241 protocol logic is not modified or dependent on these adapters
- Adapters translate XPOCK physical models → HX-241 type boundaries
- XPOCK/Python is NOT a runtime dependency for HX-241; output is JSON only

Adapter coverage:
1. CarrierCapabilities → hx_transport::carrier_types::CarrierCapabilities
2. ChannelModel → hx_sim::impairment::NetworkImpairment
3. CapacityAnalysis → hx-sim benchmark configuration
4. WaveStructure → modulation constellation data
5. ImpairmentProfile → hx-sim ImpairmentEngine config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np

from xpock.hx241.carrier import (
    CarrierWaveConfig,
    CarrierChannelModel,
    CarrierMedium,
    CarrierWaveModeler,
)
from xpock.hx241.impairment import (
    ImpairmentProfile,
    ImpairmentModeler,
    impairment_to_hx_sim_dict,
)
from xpock.hx241.channel import ChannelCapacityAnalyzer, CapacityAnalysisResult
from xpock.hx241.modulation import (
    ModulationScheme,
    WaveModulator,
    create_default_scheme,
    ModulationResult,
)
from xpock.core import WaveEvent


@dataclass
class CarrierCapabilitiesAdapter:
    """
    Adapts XPOCK carrier channel model output to HX-241's
    CarrierCapabilities struct (hx_transport::carrier_types).

    Maps physical parameters (path loss, delay spread, coherence bandwidth,
    Doppler spread, SNR) to CarrierCapabilities fields:
    - max_mtu / typical_mtu: derived from coherence bandwidth and path loss
    - max_bandwidth_bps: derived from coherence bandwidth
    - typical_latency_us: derived from propagation delay
    - reliability: derived from SNR
    - energy_cost_nj_per_bit: derived from transmit power and distance
    - max_range_m: from carrier config
    """

    @staticmethod
    def from_channel_model(
        channel: CarrierChannelModel,
    ) -> Dict[str, Any]:
        """
        Convert a CarrierChannelModel to CarrierCapabilities-compatible dict.

        Returns a dict matching the carrier_capabilities.json schema,
        ready for JSON serialization and Rust serde deserialization.
        """
        config = channel.config

        # --- max_mtu from coherence bandwidth ---
        # Higher coherence bandwidth → larger MTU possible
        # Cap at reasonable values based on path loss
        coherence_bw = channel.coherence_bandwidth_hz
        path_loss = channel.path_loss_db

        if coherence_bw > 0:
            # MTU scales with coherence bandwidth but is limited by path loss
            mtu_factor = min(1.0, 10.0 ** (-path_loss / 20.0))
            max_mtu = int(min(65535, max(64, coherence_bw * mtu_factor / 1000.0)))
        else:
            max_mtu = 1500  # Default Ethernet-like MTU

        typical_mtu = max_mtu

        # --- max_bandwidth_bps from coherence bandwidth ---
        max_bandwidth_bps = int(coherence_bw) if coherence_bw > 0 else 1000000

        # --- typical_latency_us from propagation delay ---
        # base_latency_us is already in impairment_params
        base_latency_us = channel.impairment_params.get(
            "base_latency_us",
            int(config.distance_m / 299792458.0 * 1e6),
        )
        typical_latency_us = base_latency_us

        # --- reliability from SNR ---
        snr_db = channel.snr_db
        # SNR → reliability score 0-100
        # SNR > 20dB → 100, SNR < -20dB → 0
        if snr_db >= 20.0:
            reliability = 100
        elif snr_db <= -20.0:
            reliability = 0
        else:
            reliability = int(50 + snr_db * 2.5)  # Linear mapping
            reliability = max(0, min(100, reliability))

        # --- supports_broadcast/multicast/mobility from medium ---
        medium = config.medium
        supports_broadcast = medium in (
            CarrierMedium.RADIO_FREQUENCY,
            CarrierMedium.FREE_SPACE_OPTICAL,
            CarrierMedium.PRESSURE_WAVE,
        )
        supports_multicast = medium in (
            CarrierMedium.RADIO_FREQUENCY,
            CarrierMedium.FREE_SPACE_OPTICAL,
        )
        supports_mobility = medium in (
            CarrierMedium.RADIO_FREQUENCY,
            CarrierMedium.PRESSURE_WAVE,
            CarrierMedium.MAGNETIC_INDUCTION,
        )

        # --- energy_cost_nj_per_bit from transmit power ---
        # Rough estimate: transmit power * distance → energy per bit
        tx_power_w = config.transmit_power_w
        distance_m = config.distance_m
        energy_cost_nj_per_bit = int(
            tx_power_w * distance_m * 1e9 / max(1, channel.capacity_bps)
        ) if channel.capacity_bps > 0 else 1000

        # --- max_range_m from config ---
        max_range_m = int(config.distance_m)

        return {
            "max_mtu": max_mtu,
            "typical_mtu": typical_mtu,
            "max_bandwidth_bps": max_bandwidth_bps,
            "typical_latency_us": typical_latency_us,
            "reliability": reliability,
            "supports_broadcast": supports_broadcast,
            "supports_multicast": supports_multicast,
            "supports_mobility": supports_mobility,
            "energy_cost_nj_per_bit": energy_cost_nj_per_bit,
            "max_range_m": max_range_m,
        }

    @staticmethod
    def from_impairment_profile(
        profile: ImpairmentProfile,
    ) -> Dict[str, Any]:
        """
        Convert an ImpairmentProfile to CarrierCapabilities-compatible dict.
        """
        # Build a minimal channel model from the profile
        config = CarrierWaveConfig(
            medium=CarrierMedium(profile.medium),
            frequency_hz=1e9,  # Default, not stored in profile
            distance_m=1.0,    # Default, not stored in profile
            transmit_power_w=1.0,
        )

        # Create a synthetic channel model from profile data
        channel = CarrierChannelModel(
            config=config,
            frequency_response=np.array([1.0 + 0j]),
            frequency_grid_hz=np.array([1e9]),
            impulse_response=np.array([1.0 + 0j]),
            time_grid_s=np.array([0.0]),
            path_loss_db=profile.path_loss_db,
            delay_spread_s=profile.delay_spread_s,
            coherence_bandwidth_hz=profile.coherence_bandwidth_hz,
            doppler_spread_hz=profile.doppler_spread_hz,
            capacity_bps=0.0,
            snr_db=profile.snr_db,
            impairment_params=impairment_to_hx_sim_dict(profile),
            medium_notes=profile.notes,
        )

        return CarrierCapabilitiesAdapter.from_channel_model(channel)


@dataclass
class ImpairmentAdapter:
    """
    Adapts XPOCK impairment models to hx-sim's NetworkImpairment format.

    Maps physical channel parameters to hx-sim NetworkImpairment fields
    for use in ImpairmentEngine and adversarial tests.
    """

    @staticmethod
    def from_channel_model(
        channel: CarrierChannelModel,
    ) -> Dict[str, Any]:
        """
        Convert a CarrierChannelModel to hx-sim NetworkImpairment-compatible dict.

        Returns impairment_params directly from the channel model, which
        are already in hx-sim format.
        """
        return dict(channel.impairment_params)

    @staticmethod
    def from_impairment_profile(
        profile: ImpairmentProfile,
    ) -> Dict[str, Any]:
        """
        Convert an ImpairmentProfile to hx-sim NetworkImpairment-compatible dict.
        """
        return impairment_to_hx_sim_dict(profile)


@dataclass
class CapacityAdapter:
    """
    Adapts XPOCK channel capacity analysis to hx-sim benchmark configuration.

    Maps capacity analysis results to hx-sim BenchmarkConfig fields for
    transport layer benchmarking.
    """

    @staticmethod
    def from_capacity_result(
        result: CapacityAnalysisResult,
    ) -> Dict[str, Any]:
        """
        Convert a CapacityAnalysisResult to hx-sim benchmark configuration.

        Returns a dict suitable for configuring hx-sim's BenchmarkHarness
        with realistic throughput, latency, and loss parameters.
        """
        return {
            "workload": "BulkTransfer",
            "packet_size": max(64, int(result.spectral_efficiency_bps_hz * 100)),
            "stream_count": 1,
            "total_bytes": int(result.usable_capacity_bps * 10),
            "latency_us": int(1e6 / max(1, result.symbols_per_second)),
            "loss_rate": float(
                1.0 - min(1.0, result.distinguishable_states / 256.0)
            ),
            "bandwidth_bps": int(result.shannon_capacity_bps),
            "topology_nodes": 2,
            "duration_secs": 30,
        }


@dataclass
class WaveEventAdapter:
    """
    Adapts XPOCK WaveEvent to HX-241 modulation data.

    Extracts wave structure properties that can be used as modulation
    state identifiers in HX-241's optional wave-structure modulation layer.
    """

    @staticmethod
    def to_modulation_state(
        event: WaveEvent,
    ) -> Dict[str, Any]:
        """
        Convert a WaveEvent to a modulation state dict.

        Extracts measurable properties from the wave event that can be
        used to identify which constellation point was transmitted.
        """
        analyzer = WaveAnalyzer()  # Imported lazily to avoid circular deps
        from xpock.analysis import WaveAnalyzer as _WA
        analyzer = _WA()
        props = analyzer.analyze(event)

        return {
            "event_id": event.event_id,
            "temporal_duration": props.temporal_duration_fwhm,
            "temporal_center": props.temporal_center,
            "bandwidth_rms": props.bandwidth_rms,
            "beam_waist_r": props.beam_waist_r,
            "oam_per_photon": props.oam_per_photon,
            "winding_number": props.winding_number,
            "degree_of_polarization": props.degree_of_polarization,
            "stokes_s1": props.stokes_s1,
            "stokes_s2": props.stokes_s2,
            "stokes_s3": props.stokes_s3,
            "schmidt_number": props.schmidt_number,
            "total_power": props.total_power,
            "peak_intensity": props.peak_intensity,
            "is_physically_valid": props.is_physically_valid,
        }


# -----------------------------------------------------------------------
# Batch adapter for complete test vector generation
# -----------------------------------------------------------------------


class HX241AdapterSuite:
    """
    Complete adapter suite for mapping XPOCK outputs to HX-241 types.

    Provides a unified interface for adapting all XPOCK integration
    outputs to HX-241-compatible data structures.
    """

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = Path(workspace) if workspace else Path("./241-integration/adapters")
        self.workspace.mkdir(parents=True, exist_ok=True)

    def adapt_carrier_capabilities(
        self,
        channel: CarrierChannelModel,
    ) -> Dict[str, Any]:
        """Adapt a carrier channel model to CarrierCapabilities."""
        return CarrierCapabilitiesAdapter.from_channel_model(channel)

    def adapt_impairment(
        self,
        channel: CarrierChannelModel,
    ) -> Dict[str, Any]:
        """Adapt a carrier channel model to hx-sim NetworkImpairment params."""
        return ImpairmentAdapter.from_channel_model(channel)

    def adapt_capacity(
        self,
        result: CapacityAnalysisResult,
    ) -> Dict[str, Any]:
        """Adapt a capacity analysis result to hx-sim benchmark config."""
        return CapacityAdapter.from_capacity_result(result)

    def adapt_modulation_state(
        self,
        event: WaveEvent,
    ) -> Dict[str, Any]:
        """Adapt a wave event to a modulation state dict."""
        return WaveEventAdapter.to_modulation_state(event)

    def adapt_all(
        self,
        channel: CarrierChannelModel,
        capacity: CapacityAnalysisResult,
        event: WaveEvent,
    ) -> Dict[str, Any]:
        """Adapt all outputs to HX-241-compatible structures."""
        return {
            "carrier_capabilities": self.adapt_carrier_capabilities(channel),
            "impairment_params": self.adapt_impairment(channel),
            "benchmark_config": self.adapt_capacity(capacity),
            "modulation_state": self.adapt_modulation_state(event),
        }

    def save_adapted(
        self,
        adapted: Dict[str, Any],
        filename: str = "adapted_hx241_output.json",
    ) -> Path:
        """Save adapted output as JSON for Rust consumption."""
        path = self.workspace / filename
        path.write_text(json.dumps(adapted, indent=2, default=str), encoding="utf-8")
        return path
