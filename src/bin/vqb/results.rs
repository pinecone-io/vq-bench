//! The results JSON schema: one run, its datasets, and per-method metrics.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

/// Score-latency summary, microseconds per query.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Timing {
    pub avg: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

#[derive(Debug, Serialize)]
pub struct RunMeta {
    pub name: String,
    pub seed: u64,
    #[serde(rename = "k")]
    pub ks: Vec<usize>,
    #[serde(rename = "temp")]
    pub temperatures: Vec<f64>,
    pub n_reconstruct: usize,
    pub timestamp: u64,
    /// Worker threads used for encoding.
    pub threads: usize,
    /// Logical CPU cores on the machine that produced this run.
    pub cores: usize,
    /// Target architecture (`std::env::consts::ARCH`).
    pub arch: String,
    /// Operating system (`std::env::consts::OS`).
    pub os: String,
}

#[derive(Debug, Serialize)]
pub struct Run {
    pub meta: RunMeta,
    pub datasets: Vec<DatasetResult>,
}

#[derive(Debug, Serialize)]
pub struct DatasetResult {
    pub dataset: String,
    pub dim: usize,
    pub n_base: usize,
    pub n_eval: usize,
    pub n_candidates: usize,
    pub methods: Vec<MethodResult>,
}

/// One quantizer's metrics on one dataset. Optional fields are emitted only when
/// the config requested that metric.
#[derive(Debug, Serialize)]
pub struct MethodResult {
    pub label: String,
    pub bits_per_dim: f64,
    pub model_bits_per_dim: f64,
    pub code_bits_per_dim: f64,
    pub encode_s: f64,
    /// Peak additional heap during encoding (bytes above the pre-encode baseline).
    pub encode_peak_bytes: u64,
    /// `encode_peak_bytes` normalized per base vector encoded.
    pub encode_peak_bytes_per_vec: f64,
    pub score_us: Timing,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recon_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mse_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bias_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mse_recon: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bias_recon: Option<f64>,
    /// recall@k, keyed by k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recalls: Option<BTreeMap<usize, f64>>,
    /// SOS@k, keyed by k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sos: Option<BTreeMap<usize, f64>>,
    /// exp-SOS@k over exp(score/tau)-transformed scores, keyed by temperature then k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp_sos: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    /// Softmax KL divergence, keyed by temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_kl: Option<BTreeMap<String, f64>>,
    /// Softmax total-variation distance, keyed by temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_tv: Option<BTreeMap<String, f64>>,
}

/// Pretty-print the run to a JSON file.
pub fn write_json(path: &Path, run: &Run) -> Result<()> {
    let s = serde_json::to_string_pretty(run).context("serialize results")?;
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
