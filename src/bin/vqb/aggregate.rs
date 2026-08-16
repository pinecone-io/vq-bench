//! Reduce a raw capture to results: compute each requested metric per method.
//! Shared by `vqb run` (in-memory) and `vqb eval` (from a `.raw` file).

use std::collections::BTreeMap;

use crate::bench;
use crate::raw::{RawData, RawDataset, RawMethod};
use crate::results::{DatasetResult, MethodResult, Run, RunMeta, Timing};

fn want(metrics: &[String], name: &str) -> bool {
    metrics.iter().any(|m| m == name)
}

/// Round to 6 significant figures — keeps the results JSON readable without
/// losing meaningful precision (e.g. `1.5746560688e-7` → `1.57466e-7`).
fn sig6(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let scale = 10f64.powi(5 - x.abs().log10().floor() as i32);
    (x * scale).round() / scale
}

fn sig6_k(m: BTreeMap<usize, f64>) -> BTreeMap<usize, f64> {
    m.into_iter().map(|(k, v)| (k, sig6(v))).collect()
}
fn sig6_s(m: BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    m.into_iter().map(|(k, v)| (k, sig6(v))).collect()
}
fn sig6_kt(m: BTreeMap<String, BTreeMap<usize, f64>>) -> BTreeMap<String, BTreeMap<usize, f64>> {
    m.into_iter().map(|(t, ks)| (t, sig6_k(ks))).collect()
}
fn sig6_timing(t: &Timing) -> Timing {
    Timing {
        avg: sig6(t.avg),
        p50: sig6(t.p50),
        p90: sig6(t.p90),
        p99: sig6(t.p99),
    }
}

/// Reduce one method's raw capture to its results row. `d` supplies the shared
/// dataset facts (true scores, references, `n_base`); `d.methods` is unused, so
/// the streaming runner can pass a head-only `RawDataset`.
pub fn method_result(
    d: &RawDataset,
    m: &RawMethod,
    metrics: &[String],
    ks: &[usize],
    temps: &[f64],
    seed: u64,
) -> MethodResult {
    let (score_mse, score_bias) = bench::score_mse_bias(&d.true_scores, &m.approx_scores);
    let (kl, tv) = bench::softmax_kl_tv(&d.true_scores, &m.approx_scores, temps);
    let recons = m.recons.as_ref();
    MethodResult {
        label: m.label.clone(),
        bits_per_dim: sig6(m.bits_per_dim),
        model_bits_per_dim: sig6(m.model_bits_per_dim),
        code_bits_per_dim: sig6(m.code_bits_per_dim),
        fit_s: sig6(m.fit_s),
        fit_peak_bytes: m.fit_peak_bytes,
        encode_s: sig6(m.encode_s),
        encode_peak_bytes: m.encode_peak_bytes,
        encode_peak_bytes_per_vec: sig6(m.encode_peak_bytes as f64 / d.n_base as f64),
        score_us: sig6_timing(&m.score_us),
        recon_us: m.recon_us.map(sig6),
        mse_score: want(metrics, "mse_score").then_some(sig6(score_mse)),
        bias_score: want(metrics, "bias_score").then_some(sig6(score_bias)),
        mse_recon: recons
            .filter(|_| want(metrics, "mse_recon"))
            .map(|r| sig6(bench::recon_mse(&d.references, r))),
        bias_recon: recons
            .filter(|_| want(metrics, "bias_recon"))
            .map(|r| sig6(bench::recon_bias(&d.references, r))),
        recalls: want(metrics, "recall")
            .then(|| sig6_k(bench::recalls(&d.true_scores, &m.approx_scores, ks, seed))),
        sos: want(metrics, "sos")
            .then(|| sig6_k(bench::sos(&d.true_scores, &m.approx_scores, ks, seed))),
        exp_sos: want(metrics, "exp_sos")
            .then(|| sig6_kt(bench::exp_sos(&d.true_scores, &m.approx_scores, ks, temps, seed))),
        score_kl: want(metrics, "kl").then_some(sig6_s(kl)),
        score_tv: want(metrics, "tv").then_some(sig6_s(tv)),
    }
}

/// One dataset's results from its raw capture.
pub fn dataset_result(
    d: &RawDataset,
    metrics: &[String],
    ks: &[usize],
    temps: &[f64],
    seed: u64,
) -> DatasetResult {
    DatasetResult {
        dataset: d.dataset.clone(),
        dim: d.dim,
        n_base: d.n_base,
        n_eval: d.n_eval,
        n_candidates: d.n_candidates,
        methods: d
            .methods
            .iter()
            .map(|m| method_result(d, m, metrics, ks, temps, seed))
            .collect(),
    }
}

/// Reduce a whole capture into a `Run` (used by `vqb eval`). Metrics, `ks`, and
/// `temps` come from the eval config, not the capture's original run meta.
pub fn run(data: &RawData, metrics: &[String], ks: &[usize], temps: &[f64]) -> Run {
    let m = &data.meta;
    Run {
        meta: RunMeta {
            name: m.name.clone(),
            seed: m.seed,
            ks: ks.to_vec(),
            temperatures: temps.to_vec(),
            n_reconstruct: m.n_reconstruct,
            timestamp: m.timestamp,
            threads: m.threads,
            cores: m.cores,
            arch: m.arch.clone(),
            os: m.os.clone(),
        },
        datasets: data
            .datasets
            .iter()
            .map(|d| dataset_result(d, metrics, ks, temps, m.seed))
            .collect(),
    }
}
