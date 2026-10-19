// Auto-generated reference: HX-241 Rust integration with XPOCK test vectors
// This file shows how to deserialize XPOCK's JSON test vectors into HX-241
// native types using serde. Copy into hx-sim/src/ or hx-transport/src/ as needed.
//
// NO PYTHON/XPOCK DEPENDENCY AT RUNTIME. The JSON files are static data.

use serde::{Deserialize, Serialize};

// ============================================================================
// CarrierCapabilities — maps from xpock/241-integration/schemas/carrier_capabilities.json
// Rust location: hx-transport/src/carrier_types.rs
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CarrierCapabilities {
    pub max_mtu: u32,
    pub typical_mtu: u32,
    pub max_bandwidth_bps: u64,
    pub typical_latency_us: u64,
    pub reliability: u8,
    pub supports_broadcast: bool,
    pub supports_multicast: bool,
    pub supports_mobility: bool,
    pub energy_cost_nj_per_bit: u32,
    pub max_range_m: u32,
}

// ============================================================================
// NetworkImpairment — maps from xpock/241-integration/schemas/network_impairment_params.json
// Rust location: hx-sim/src/impairment.rs
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkImpairment {
    pub packet_loss_rate: f64,
    pub duplication_rate: f64,
    pub reordering_depth: usize,
    pub corruption_rate: f64,
    pub base_latency_us: u64,
    pub jitter_us: u64,
    pub bandwidth_bps: u64,
    pub partition: Option<PartitionConfig>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub side_a: Vec<crate::hx_types::addressing::NodeID>,
    pub side_b: Vec<crate::hx_types::addressing::NodeID>,
    pub partition_type: PartitionType,
    pub duration_us: u64,
    pub start_time: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PartitionType {
    Clean = 0x01,
    Dirty = 0x02,
    AsymmetricAB = 0x03,
    AsymmetricBA = 0x04,
}

// ============================================================================
// CapacityAnalysisResult — maps from xpock/241-integration/schemas/capacity_analysis_result.json
// Rust location: hx-sim/src/benchmark.rs (for BenchmarkConfig)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityAnalysisResult {
    pub snr_db: f64,
    pub bandwidth_hz: f64,
    pub shannon_capacity_bps: f64,
    pub spectral_efficiency_bps_hz: f64,
    pub usable_capacity_bps: f64,
    pub distinguishable_states: usize,
    pub bits_per_symbol: f64,
    pub symbols_per_second: f64,
    pub net_information_rate_bps: f64,
    pub min_js_divergence: f64,
    pub avg_js_divergence: f64,
}

// ============================================================================
// CarrierChannelModel — maps from xpock/241-integration/schemas/carrier_channel_model.json
// Physical layer model output; NOT part of HX-241 core protocol
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarrierChannelModel {
    pub id: String,
    pub medium: String,
    pub frequency_hz: f64,
    pub wavelength_m: f64,
    pub distance_m: f64,
    pub transmit_power_w: f64,
    pub path_loss_db: f64,
    pub snr_db: f64,
    pub delay_spread_s: f64,
    pub coherence_bandwidth_hz: f64,
    pub doppler_spread_hz: f64,
    pub capacity_bps: f64,
    pub impairment_params: NetworkImpairment,
    pub medium_notes: Vec<String>,
}

// ============================================================================
// Example: Deserializing XPOCK test vectors in Rust
// ============================================================================

/// Load XPOCK test vectors from JSON and extract impairment parameters
/// for hx-sim's ImpairmentEngine.
pub fn load_impairment_scenarios(json_str: &str) -> Result<Vec<NetworkImpairment>, serde_json::Error> {
    #[derive(Deserialize)]
    struct TestVectorSet {
        impairment_profiles: Vec<ImpairmentProfileWrapper>,
    }

    #[derive(Deserialize)]
    struct ImpairmentProfileWrapper {
        hx_sim_impairment: NetworkImpairment,
    }

    let set: TestVectorSet = serde_json::from_str(json_str)?;
    Ok(set
        .impairment_profiles
        .into_iter()
        .map(|p| p.hx_sim_impairment)
        .collect())
}

/// Load carrier capabilities from JSON test vectors.
pub fn load_carrier_capabilities(json_str: &str) -> Result<Vec<CarrierCapabilities>, serde_json::Error> {
    #[derive(Deserialize)]
    struct TestVectorSet {
        carrier_models: Vec<CarrierModelWrapper>,
    }

    #[derive(Deserialize)]
    struct CarrierModelWrapper {
        carrier_capabilities: CarrierCapabilities,
    }

    let set: TestVectorSet = serde_json::from_str(json_str)?;
    Ok(set
        .carrier_models
        .into_iter()
        .map(|m| m.carrier_capabilities)
        .collect())
}

// ============================================================================
// Integration notes for HX-241 maintainers
// ============================================================================

/*
INTEGRATION INSTRUCTIONS:

1. Copy the JSON test vector files from xpock/241-integration/vectors/ into
   your hx-sim test data directory (e.g., hx-sim/tests/data/).

2. Add serde and serde_json dependencies to hx-sim's Cargo.toml if not
   already present (they are already in the workspace dependencies).

3. Deserialize the JSON using the struct definitions above, or generate
   your own serde structs from the JSON schemas in xpock/241-integration/schemas/.

4. Pass the deserialized NetworkImpairment to hx-sim's ImpairmentEngine:
       let impairment: NetworkImpairment = serde_json::from_str(json_str)?;
       let mut engine = ImpairmentEngine::new(Some(impairment));
       engine.set_enabled(true);

5. Use CapacityAnalysisResult to configure hx-sim's BenchmarkHarness:
       let config = BenchmarkConfig {
           bandwidth_bps: capacity.shannon_capacity_bps as u64,
           latency_us: capacity.typical_latency_us,
           loss_rate: 1.0 - (capacity.distinguishable_states as f64 / 256.0),
           ..Default::default()
       };

ARCHITECTURAL COMPLIANCE:
- No IP/TCP/UDP/QUIC/DNS/HTTP dependencies in this file
- HX-241 protocol logic is not modified
- Carrier capabilities are HINTS only (I-CARR-005)
- XPOCK/Python is NOT a runtime dependency; JSON is static data
*/
