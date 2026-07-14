//! Run config: the JSON spec, sweep expansion, and validation.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::registry;

/// One run config: a sweep over datasets × methods × k-values under one seed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Registry names of the datasets to run.
    pub datasets: Vec<String>,
    /// The quantizers to run, with their (possibly swept) parameters.
    pub methods: Vec<MethodConfig>,
    /// Metrics to report.
    pub metrics: Vec<String>,
    /// `k` values for recall@k / SOS@k.
    #[serde(rename = "k", default = "default_ks")]
    pub ks: Vec<usize>,
    /// Softmax temperatures for the kl/tv metrics.
    #[serde(rename = "temp", default = "default_temperatures")]
    pub temperatures: Vec<f64>,
    /// Master seed for all sampling and seeded primitives.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Database vectors sampled for reconstruction metrics (default: all of `db`).
    #[serde(default)]
    pub n_reconstruct: Option<usize>,
    /// Eval queries sampled for scoring metrics (default: all of `eval`).
    #[serde(default)]
    pub n_eval: Option<usize>,
    /// Calibration queries passed to `fit` (default: all of `calib`).
    #[serde(default)]
    pub n_calib: Option<usize>,
    /// Database vectors sampled to form the searched DB (default: all of `base`).
    /// Shrinking it recomputes the candidates and ground truth over the subset.
    #[serde(default)]
    pub n_base: Option<usize>,
    /// Base vectors sampled for `fit` (default: all of the searched DB).
    #[serde(default)]
    pub n_fit: Option<usize>,
    /// Worker threads for encoding (default: all logical cores; capped at the
    /// machine's available parallelism). Overridden by the `RAYON_NUM_THREADS`
    /// environment variable.
    #[serde(default)]
    pub threads: Option<usize>,
}

/// A quantizer selection: its catalog `name` plus parameters. Each parameter is
/// either a scalar or an array (an array sweeps that parameter).
#[derive(Debug, Deserialize)]
pub struct MethodConfig {
    pub name: String,
    #[serde(flatten)]
    pub params: BTreeMap<String, Value>,
}

fn default_ks() -> Vec<usize> {
    vec![1, 10, 50]
}
fn default_seed() -> u64 {
    1
}
fn default_temperatures() -> Vec<f64> {
    vec![0.5, 1.0, 2.0]
}

/// Metrics this harness knows how to compute, each with a one-line description.
pub const KNOWN_METRICS: &[(&str, &str)] = &[
    (
        "recall",
        "recall@k: overlap of approx vs true top-k within each query's candidates",
    ),
    (
        "sos",
        "SOS@k: sum (over all queries and top-k candidates) of approx scores divided by sum of true scores",
    ),
    ("mse_score", "mean squared error of the estimated scores"),
    (
        "mse_recon",
        "mean squared error of the reconstructed vectors",
    ),
    (
        "bias_score",
        "mean signed error (bias) of the estimated scores",
    ),
    (
        "bias_recon",
        "mean residual (bias) of the reconstructed vectors",
    ),
    (
        "kl",
        "softmax KL divergence of approx vs true scores, per temperature",
    ),
    (
        "tv",
        "softmax total-variation distance of approx vs true scores, per temperature",
    ),
];

/// Size and cost metrics reported for every method regardless of config — the
/// always-present columns of a results file.
pub const RESOURCE_METRICS: &[(&str, &str)] = &[
    ("bits_per_dim", "total encoded size, bits per dimension"),
    ("model_bits_per_dim", "shared-model size, bits per dimension"),
    ("code_bits_per_dim", "per-vector code size, bits per dimension"),
    ("encode_time", "wall-clock time to encode the base (seconds)"),
    ("encode_memory", "peak additional heap during encode (bytes)"),
    (
        "encode_memory_per_vec",
        "encode peak heap normalized per base vector (bytes)",
    ),
    ("score_time", "per-query score latency avg/p50/p90/p99 (µs)"),
    ("recon_time", "per-vector reconstruction time (µs)"),
];

/// One concrete run: a method name with its parameters fully resolved to scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMethod {
    pub name: String,
    pub params: BTreeMap<String, Value>,
}

impl ResolvedMethod {
    /// The method name: the display `family` name plus parameters in parentheses,
    /// e.g. `MinMax (b=2)` (just `family` when there are no parameters).
    pub fn label(&self, family: &str) -> String {
        if self.params.is_empty() {
            return family.to_string();
        }
        let parts: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{family} ({})", parts.join(", "))
    }
}

impl RunConfig {
    /// Parse a config from a JSON file.
    pub fn parse(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: RunConfig = serde_json::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Ok(cfg)
    }

    /// Expand every method's array-valued parameters into the cartesian product
    /// of concrete runs (scalars pass through unchanged).
    pub fn expand(&self) -> Vec<ResolvedMethod> {
        let mut out = Vec::new();
        for method in &self.methods {
            for params in cartesian(&method.params) {
                out.push(ResolvedMethod {
                    name: method.name.clone(),
                    params,
                });
            }
        }
        out
    }

    /// Check datasets, method names, and metrics against what the harness knows.
    /// Returns one message per problem; empty means the config is runnable.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.datasets.is_empty() {
            problems.push("no datasets listed".to_string());
        }
        if self.methods.is_empty() {
            problems.push("no methods listed".to_string());
        }
        for ds in &self.datasets {
            if let Err(e) = registry::resolve(ds) {
                problems.push(e.to_string());
            }
        }
        for method in &self.methods {
            if !vqb::catalog::is_known(&method.name) {
                problems.push(format!(
                    "unknown quantizer `{}` (see `vqb show quantizers`)",
                    method.name
                ));
            }
        }
        for metric in &self.metrics {
            if !KNOWN_METRICS.iter().any(|(n, _)| *n == metric) {
                let known = KNOWN_METRICS
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ");
                problems.push(format!("unknown metric `{metric}` (known: {known})"));
            }
        }
        problems
    }
}

/// Cartesian product of a parameter map: every array field is a sweep axis,
/// every scalar passes through. `{bits: [2,4]}` → two maps `{bits:2}`,`{bits:4}`.
fn cartesian(params: &BTreeMap<String, Value>) -> Vec<BTreeMap<String, Value>> {
    let mut combos = vec![BTreeMap::new()];
    for (key, value) in params {
        let choices: Vec<Value> = match value {
            Value::Array(items) => items.clone(),
            scalar => vec![scalar.clone()],
        };
        let mut next = Vec::with_capacity(combos.len() * choices.len());
        for base in &combos {
            for choice in &choices {
                let mut m = base.clone();
                m.insert(key.clone(), choice.clone());
                next.push(m);
            }
        }
        combos = next;
    }
    combos
}

/// Ensure a config has no validation problems, returning a combined error.
pub fn require_valid(cfg: &RunConfig) -> Result<()> {
    let problems = cfg.validate();
    if !problems.is_empty() {
        bail!("invalid config:\n  - {}", problems.join("\n  - "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn method(name: &str, params: Value) -> MethodConfig {
        MethodConfig {
            name: name.to_string(),
            params: serde_json::from_value(params).unwrap(),
        }
    }

    #[test]
    fn expand_sweeps_array_params() {
        let cfg = RunConfig {
            datasets: vec!["arxiv-nomic-768".into()],
            methods: vec![method("minmax", json!({ "b": [2, 4, 8] }))],
            metrics: vec![],
            ks: default_ks(),
            temperatures: default_temperatures(),
            seed: 1,
            n_reconstruct: None,
            n_eval: None,
            n_calib: None,
            n_base: None,
            n_fit: None,
            threads: None,
        };
        let runs = cfg.expand();
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].label("MinMax"), "MinMax (b=2)");
        assert_eq!(runs[2].label("MinMax"), "MinMax (b=8)");
    }

    #[test]
    fn expand_is_cartesian_over_multiple_axes() {
        let cfg = RunConfig {
            datasets: vec![],
            methods: vec![method("pq", json!({ "b": [2, 4], "segments": [8, 16] }))],
            metrics: vec![],
            ks: default_ks(),
            temperatures: default_temperatures(),
            seed: 1,
            n_reconstruct: None,
            n_eval: None,
            n_calib: None,
            n_base: None,
            n_fit: None,
            threads: None,
        };
        assert_eq!(cfg.expand().len(), 4);
    }

    #[test]
    fn scalar_param_is_a_single_run() {
        let cfg = RunConfig {
            datasets: vec![],
            methods: vec![method("rabitq", json!({}))],
            metrics: vec![],
            ks: default_ks(),
            temperatures: default_temperatures(),
            seed: 1,
            n_reconstruct: None,
            n_eval: None,
            n_calib: None,
            n_base: None,
            n_fit: None,
            threads: None,
        };
        let runs = cfg.expand();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].label("RaBitQ"), "RaBitQ");
    }

    #[test]
    fn validate_flags_unknowns() {
        let cfg = RunConfig {
            datasets: vec!["nope".into()],
            methods: vec![method("bogus", json!({}))],
            metrics: vec!["recall".into(), "weird".into()],
            ks: default_ks(),
            temperatures: default_temperatures(),
            seed: 1,
            n_reconstruct: None,
            n_eval: None,
            n_calib: None,
            n_base: None,
            n_fit: None,
            threads: None,
        };
        let problems = cfg.validate();
        assert_eq!(problems.len(), 3); // unknown dataset, quantizer, metric
    }
}
