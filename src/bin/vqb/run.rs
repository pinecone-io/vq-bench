//! The run driver: load each dataset, run each quantizer
//! (fit→encode→reconstruct→score), capture raw outputs, and write raw + results.

use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use ndarray::{s, Array2, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::io::Write;
use vqb::Quantizer;

use crate::config::{ResolvedMethod, RunConfig};
use crate::dataset::{self, Base, Loaded};
use crate::results::{Run, RunMeta, Timing};
use crate::{aggregate, bench, codes, config, factory, mem, raw, registry, results};
use raw::{RawDataset, RawMeta, RawMethod};

/// Where a method's per-vector codes live during score/reconstruct: either in
/// memory (a fresh `run`) or an on-disk store (reused from a prior `encode`).
/// Both are addressed by base-row index; neither path streams the whole set.
enum Codes {
    Mem(Vec<Vec<u8>>),
    Disk(codes::CodeStore),
}

impl Codes {
    /// The codes for `idx`, owned so callers can build a `&[&[u8]]` from them.
    /// `idx` is always a small subset (a query's candidates, or the recon sample).
    fn gather(&self, idx: &[usize]) -> Vec<Vec<u8>> {
        match self {
            Codes::Mem(codes) => idx.iter().map(|&i| codes[i].clone()).collect(),
            Codes::Disk(store) => idx.iter().map(|&i| store.get(i).unwrap()).collect(),
        }
    }
}

/// Seeded subset of `0..len` of size `n` (identity order when `n >= len`).
fn subset_indices(len: usize, n: usize, seed: u64) -> Vec<usize> {
    if n >= len {
        return (0..len).collect();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..len).collect();
    idx.shuffle(&mut rng);
    idx.truncate(n);
    idx
}

// Seeded subsamples shared by `run` and `encode` so both see the same DB, calib,
// and fit rows (encoded codes must align row-for-row with what `run` addresses).

/// The searched DB as a subsample of `base`, or `None` to use the full base. A streamed
/// base is gathered into memory here: `subset_indices` shuffles, so DB row order does not
/// follow file order, and streaming it would cost a positioned read per row.
fn db_subsample(loaded: &Loaded, cfg: &RunConfig) -> Result<Option<Base>> {
    let idx = subset_indices(
        loaded.base.nrows(),
        cfg.n_base.unwrap_or(usize::MAX),
        cfg.seed ^ 0xba5e,
    );
    if idx.len() == loaded.base.nrows() {
        return Ok(None);
    }
    Ok(Some(Base::Mem(loaded.base.gather(&idx)?)))
}

/// The calibration queries passed to `fit`, subsampled from `calib`.
fn calib_subsample(loaded: &Loaded, cfg: &RunConfig) -> Option<Array2<f32>> {
    loaded.calib.as_ref().map(|c| {
        let idx = subset_indices(c.nrows(), cfg.n_calib.unwrap_or(usize::MAX), cfg.seed ^ 0xca11b);
        c.select(Axis(0), &idx)
    })
}

/// The base rows used only for `fit`, or `None` to fit on the whole DB.
fn fit_subsample(db: &Base, cfg: &RunConfig) -> Result<Option<Array2<f32>>> {
    let idx = subset_indices(db.nrows(), cfg.n_fit.unwrap_or(usize::MAX), cfg.seed ^ 0xf17f);
    if idx.len() == db.nrows() {
        return Ok(None);
    }
    Ok(Some(db.gather(&idx)?))
}

/// The rows `fit` sees: the `n_fit` subsample, or the whole DB when unset. Unset only
/// works on a resident DB, which `config::require_streamable` has already established.
fn fit_rows<'a>(db: &'a Base, fit: Option<&'a Array2<f32>>) -> Result<ArrayView2<'a, f32>> {
    match fit {
        Some(f) => Ok(f.view()),
        None => db
            .resident()
            .context("`--stream` needs `n_fit` set: fit would otherwise read the whole base"),
    }
}

/// The `(n_base, n_fit, n_calib)` the subsamples above resolve to, from row counts
/// alone — `subset_indices` keeps `min(n, len)` rows, so a code file's identity can
/// be derived from a dataset's shapes before it is loaded. `check_identity` asserts the
/// two agree.
fn resolved_counts(
    base_rows: usize,
    calib_rows: Option<usize>,
    cfg: &RunConfig,
) -> (usize, usize, usize) {
    let n_base = cfg.n_base.unwrap_or(usize::MAX).min(base_rows);
    let n_fit = cfg.n_fit.unwrap_or(usize::MAX).min(n_base);
    let n_calib = calib_rows.map_or(0, |c| cfg.n_calib.unwrap_or(usize::MAX).min(c));
    (n_base, n_fit, n_calib)
}

/// The code-file identity a config resolves to, from row counts alone. The one place it
/// is built: `encode` needs it before the load to decide whether to skip a dataset, and
/// `run` needs it before deciding whether to gather any rows at all.
fn identity_for(
    base_rows: usize,
    calib_rows: Option<usize>,
    dim: usize,
    cfg: &RunConfig,
) -> codes::Identity {
    let (n_base, n_fit, n_calib) = resolved_counts(base_rows, calib_rows, cfg);
    codes::Identity {
        seed: cfg.seed,
        n_base,
        dim,
        n_fit,
        n_calib,
    }
}

/// Assert the subsamples actually gathered match the identity their codes will be filed
/// under. A drift would mean a model fitted on one row count recorded as another, so
/// `run` would reuse codes for a config `encode` never saw — checked on every encode
/// rather than left to a test, since it is silent and the codes outlive the run.
fn check_identity(
    id: &codes::Identity,
    db: &Base,
    fit: Option<&Array2<f32>>,
    calib: Option<&Array2<f32>>,
) {
    let (n_base, dim) = db.dim();
    assert_eq!(
        (
            n_base,
            dim,
            fit.map_or(n_base, Array2::nrows),
            calib.map_or(0, Array2::nrows)
        ),
        (id.n_base, id.dim, id.n_fit, id.n_calib),
        "gathered subsamples disagree with the resolved row counts"
    );
}

/// avg/p50/p90/p99 over per-query latencies (µs).
fn timing(mut us: Vec<f64>) -> Timing {
    if us.is_empty() {
        return Timing::default();
    }
    let avg = us.iter().sum::<f64>() / us.len() as f64;
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |p: f64| us[((p * (us.len() - 1) as f64).round() as usize).min(us.len() - 1)];
    Timing {
        avg,
        p50: pick(0.50),
        p90: pick(0.90),
        p99: pick(0.99),
    }
}

fn config_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .to_string()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Logical CPU cores on this machine (1 if the count is unavailable).
fn logical_cores() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Resolve the encode worker-thread count (`RAYON_NUM_THREADS` → config → all logical
/// cores) and install the global rayon pool. The request is capped at the machine's
/// available parallelism: encoding is CPU-bound and embarrassingly parallel, so more
/// workers than cores buys no speedup and only adds context-switching overhead and
/// noise to `encode_s`. Returns the count actually in effect.
fn init_thread_pool(cfg_threads: Option<usize>) -> usize {
    let cores = logical_cores();
    let requested = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .or_else(|| cfg_threads.filter(|&n| n > 0))
        .unwrap_or(cores);
    if requested > cores {
        eprintln!("requested {requested} threads > {cores} available; capping at {cores}");
    }
    let n = requested.min(cores);
    // Fails only if a global pool already exists; ignore so the count still reflects it.
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    // Keep all parallelism at the chunk level: faer defaults to Par::Rayon(0), so its
    // matmuls would re-split over the same pool inside each already-parallel encode chunk
    // — nested work that blows up peak memory and regresses past the P-core count.
    faer::set_global_parallelism(faer::Par::Seq);
    rayon::current_num_threads()
}

/// Rows per encode chunk. The driver feeds the base to `Quantizer::encode` in chunks,
/// which the rayon pool runs concurrently, so peak encode memory is up to ~one chunk
/// per worker thread plus the codes — never a second full copy of the base. Encode is
/// neighbor-blind, so chunking (and running the chunks in parallel) does not change the
/// output; this memory/parallelism policy lives in the harness, not in any quantizer.
const ENCODE_CHUNK: usize = 8192;

/// Encode the whole base by feeding `q.encode` one row chunk at a time and
/// concatenating the per-vector codes in row order. Chunks are independent (encode is
/// neighbor-blind), so they run on the rayon pool; the indexed collect keeps row order,
/// making the output byte-identical to a serial run for any thread count.
fn encode_in_chunks<Q: Quantizer + ?Sized>(
    q: &Q,
    model: &[u8],
    base: ArrayView2<f32>,
    chunk: usize,
) -> Vec<Vec<u8>> {
    let n = base.nrows();
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(chunk)
        .map(|s| (s, (s + chunk).min(n)))
        .collect();
    ranges
        .into_par_iter()
        .map(|(start, end)| q.encode(model, base.slice(s![start..end, ..])))
        .collect::<Vec<Vec<Vec<u8>>>>()
        .into_iter()
        .flatten()
        .collect()
}

/// Say what a streamed run is doing, since it changes both where the codes live and what
/// `encode_memory` counts.
fn announce_stream(mode: dataset::Mode, threads: usize) {
    if let dataset::Mode::Stream { block_mb } = mode {
        eprintln!(
            "--stream: base read from disk in ≤{block_mb} MiB blocks, codes written to \
             results/codes/ ({threads}-chunk window; encode_memory includes the read block)"
        );
    }
}

/// Rows per read block, and rows per encode chunk within it. The block is what a streamed
/// base reads at once, so `--block-mb` caps it; the chunk then splits the block across the
/// workers, which is what keeps a tight budget costing memory rather than parallelism.
/// Encode is neighbor-blind, so the split is free to move — any chunking produces the same
/// codes (`encode_in_chunks_matches_one_shot`).
fn encode_shape(dim: usize, threads: usize, mode: dataset::Mode) -> (usize, usize) {
    let workers = threads.max(1);
    let block = match mode {
        dataset::Mode::Resident => workers * ENCODE_CHUNK,
        dataset::Mode::Stream { block_mb } => {
            ((block_mb << 20) / (dim.max(1) * 4)).clamp(1, workers * ENCODE_CHUNK)
        }
    };
    (block, (block / workers).clamp(1, ENCODE_CHUNK))
}

/// How a finished store's size reads: a stride when every code matched the last, the
/// mean when they came out ragged and rows are addressed by the lengths table.
fn store_size(store: &codes::CodeStore, n_base: usize) -> String {
    store.width().map_or_else(
        || {
            format!(
                "{n_base} rows, {:.1} B avg",
                store.code_bytes() as f64 / n_base.max(1) as f64
            )
        },
        |w| format!("{n_base} rows × {w} B"),
    )
}

/// Fit `q`, encode the DB into a fresh code store at `path`, and reopen it. Shared by
/// `encode` and a streamed `run`, so both produce byte-identical stores.
///
/// Chunks within a block run concurrently but flush in row order, so the store matches a
/// serial encode at any thread count and any `(block, chunk)` split. Bit-for-bit, except
/// where a quantizer's `encode` runs a batched matmul: the accumulation order depends on
/// the batch shape, so a handful of codes can land the other side of a rounding boundary.
#[allow(clippy::too_many_arguments)]
fn fit_and_store(
    q: &dyn Quantizer,
    path: &Path,
    id: &codes::Identity,
    label: &str,
    db: &Base,
    fit_base: ArrayView2<f32>,
    calib: Option<ArrayView2<f32>>,
    threads: usize,
    shape: (usize, usize),
    scratch: &mut Vec<f32>,
) -> Result<codes::CodeStore> {
    let (block, chunk) = shape;
    let baseline = mem::current();
    mem::reset_peak();
    let t = Instant::now();
    let model = q.fit(fit_base, calib);
    let fit_s = t.elapsed().as_secs_f64();
    let fit_peak_bytes = mem::peak().saturating_sub(baseline) as u64;

    let mut writer =
        codes::CodeWriter::create(path, id, threads, fit_s, fit_peak_bytes, label, &model)?;
    // Baseline taken after fit, so the retained model counts as encode's ground
    // rather than as encode's growth.
    let baseline = mem::current();
    mem::reset_peak();
    let t = Instant::now();
    db.for_blocks(block, scratch, |_, rows| {
        let n = rows.nrows();
        let ranges: Vec<(usize, usize)> = (0..n)
            .step_by(chunk)
            .map(|s| (s, (s + chunk).min(n)))
            .collect();
        let encoded: Vec<Vec<Vec<u8>>> = ranges
            .par_iter()
            .map(|&(a, b)| q.encode(&model, rows.slice(s![a..b, ..])))
            .collect();
        for chunk in &encoded {
            writer.push_chunk(chunk)?;
        }
        Ok(())
    })?;
    let encode_s = t.elapsed().as_secs_f64();
    let encode_peak_bytes = mem::peak().saturating_sub(baseline) as u64;
    writer.finish(encode_s, encode_peak_bytes)?;
    codes::CodeStore::open(path)
}

/// Exact dot products of each eval query against its candidates, gathering one query's
/// candidate rows at a time so a streamed base never holds more than a single pool.
fn true_scores(eval: &Array2<f32>, db: &Base, candidates: &[Vec<usize>]) -> Result<Vec<Vec<f32>>> {
    candidates
        .iter()
        .enumerate()
        .map(|(qi, cand)| {
            let rows = db.gather(cand)?;
            let q = eval.row(qi);
            Ok(rows.rows().into_iter().map(|r| q.dot(&r)).collect())
        })
        .collect()
}

fn rows_to_vec(a: &Array2<f32>) -> Vec<Vec<f32>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}

// A subsampled DB's candidates come from `dataset::top_candidates`, which folds the DB one
// block at a time instead of materializing an `n_eval × n_db` score matrix.

// --- progress table --------------------------------------------------------

/// Print the per-dataset descriptor + column header + rule.
fn table_header(
    _name: &str,
    dim: usize,
    n_base: usize,
    n_fit: usize,
    n_eval: usize,
    n_candidates: usize,
    dk: usize,
) {
    eprintln!("  dim {dim} · base {n_base} · fit {n_fit} · eval {n_eval} · n_cand {n_candidates}");
    eprintln!(
        "  {:<22}{:>10}{:>12}{:>13}{:>9}{:>10}{:>9}{:>10}",
        "method",
        "bits/dim",
        format!("recall@{dk}"),
        "mse_recon",
        "fit(s)",
        "fit(MB)",
        "enc(s)",
        "enc(MB)"
    );
    eprintln!("  {}", "─".repeat(95));
}

/// The metric cells for one finished method (the label cell is printed first).
fn row_tail(
    rm: &RawMethod,
    true_scores: &[Vec<f32>],
    references: &[Vec<f32>],
    metrics: &[String],
    dk: usize,
    seed: u64,
) -> String {
    let want = |n: &str| metrics.iter().any(|m| m == n);
    let dash = || "—".to_string();
    let fixed = |x: f64| format!("{x:.4}");
    let sci = |x: f64| format!("{x:.2e}");

    let recall = if want("recall") {
        fixed(bench::recalls(true_scores, &rm.approx_scores, &[dk], seed)[&dk])
    } else {
        dash()
    };
    let mse_r = match (&rm.recons, want("mse_recon")) {
        (Some(r), true) => sci(bench::recon_mse(references, r)),
        _ => dash(),
    };
    let fit_mb = rm.fit_peak_bytes as f64 / 1e6;
    let enc_mb = rm.encode_peak_bytes as f64 / 1e6;
    format!(
        "{:>10}{:>12}{:>13}{:>9.1}{:>10.1}{:>9.1}{:>10.1}",
        fixed(rm.bits_per_dim),
        recall,
        mse_r,
        rm.fit_s,
        fit_mb,
        rm.encode_s,
        enc_mb
    )
}

/// Run one quantizer by encoding the base in memory, capturing its raw outputs.
#[allow(clippy::too_many_arguments)]
fn run_method(
    label: String,
    q: &dyn Quantizer,
    base: ArrayView2<f32>,
    fit_base: ArrayView2<f32>,
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
    candidates: &[Vec<usize>],
    recon_idx: &[usize],
) -> RawMethod {
    let (n_base, dim) = base.dim();
    let baseline = mem::current();
    mem::reset_peak();
    let t = Instant::now();
    let model = q.fit(fit_base, calib.map(|c| c.view()));
    let fit_s = t.elapsed().as_secs_f64();
    let fit_peak_bytes = mem::peak().saturating_sub(baseline) as u64;

    // Baseline taken after fit, so the retained model counts as encode's ground
    // rather than as encode's growth.
    let baseline = mem::current();
    mem::reset_peak();
    let t = Instant::now();
    let codes = encode_in_chunks(q, &model, base, ENCODE_CHUNK);
    let encode_s = t.elapsed().as_secs_f64();
    let encode_peak_bytes = mem::peak().saturating_sub(baseline) as u64;

    let code_bytes: usize = codes.iter().map(Vec::len).sum();
    score_and_reconstruct(
        label,
        q,
        &model,
        n_base,
        dim,
        code_bytes,
        fit_s,
        fit_peak_bytes,
        encode_s,
        encode_peak_bytes,
        &Codes::Mem(codes),
        eval,
        candidates,
        recon_idx,
    )
}

/// Run one quantizer from codes already persisted to disk (skips fit + encode).
#[allow(clippy::too_many_arguments)]
fn run_method_cached(
    label: String,
    q: &dyn Quantizer,
    store: codes::CodeStore,
    dim: usize,
    eval: &Array2<f32>,
    candidates: &[Vec<usize>],
    recon_idx: &[usize],
) -> RawMethod {
    let n_base = store.len();
    let model = store.model().to_vec();
    let code_bytes = store.code_bytes();
    let fit_s = store.fit_s();
    let fit_peak_bytes = store.fit_peak_bytes();
    let encode_s = store.encode_s();
    let encode_peak_bytes = store.encode_peak_bytes();
    score_and_reconstruct(
        label,
        q,
        &model,
        n_base,
        dim,
        code_bytes,
        fit_s,
        fit_peak_bytes,
        encode_s,
        encode_peak_bytes,
        &Codes::Disk(store),
        eval,
        candidates,
        recon_idx,
    )
}

/// Score every query against its candidates and reconstruct the sampled vectors,
/// assembling the `RawMethod`. Shared by the in-memory and cached paths; the byte
/// split is `(model.len(), code_bytes)`, matching `vqb::byte_split`.
#[allow(clippy::too_many_arguments)]
fn score_and_reconstruct(
    label: String,
    q: &dyn Quantizer,
    model: &[u8],
    n_base: usize,
    dim: usize,
    code_bytes: usize,
    fit_s: f64,
    fit_peak_bytes: u64,
    encode_s: f64,
    encode_peak_bytes: u64,
    codes: &Codes,
    eval: &Array2<f32>,
    candidates: &[Vec<usize>],
    recon_idx: &[usize],
) -> RawMethod {
    let (mb, cb) = (model.len(), code_bytes);

    // Score each query against its own candidates (internally batched over them).
    let mut approx_scores = Vec::with_capacity(eval.nrows());
    let mut latencies = Vec::with_capacity(eval.nrows());
    for (qi, cand) in candidates.iter().enumerate() {
        let owned = codes.gather(cand);
        let cand_codes: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let query = eval.slice(s![qi..qi + 1, ..]);
        let t = Instant::now();
        let scores = q.score(model, query, &cand_codes);
        latencies.push(t.elapsed().as_nanos() as f64 / 1000.0);
        approx_scores.push(scores.row(0).to_vec());
    }

    // Reconstruct the sampled vectors.
    let owned = codes.gather(recon_idx);
    let recon_codes: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
    let t = Instant::now();
    let recons = q.reconstruct(model, &recon_codes);
    let recon_us = (!recon_idx.is_empty())
        .then(|| t.elapsed().as_nanos() as f64 / 1000.0 / recon_idx.len() as f64);

    RawMethod {
        label,
        bits_per_dim: bench::bits_per_dim(mb + cb, n_base, dim),
        model_bits_per_dim: bench::bits_per_dim(mb, n_base, dim),
        code_bits_per_dim: bench::bits_per_dim(cb, n_base, dim),
        fit_s,
        fit_peak_bytes,
        encode_s,
        encode_peak_bytes,
        score_us: timing(latencies),
        recon_us,
        approx_scores,
        recons: Some(rows_to_vec(&recons)),
    }
}

/// Run every method on one loaded dataset, streaming each method's raw capture
/// to `writer` and reducing it to results as it finishes (so the whole run's
/// scores/reconstructions never sit in memory at once). Returns the reduced
/// per-dataset results.
#[allow(clippy::too_many_arguments)]
fn run_dataset<W: std::io::Write>(
    name: &str,
    loaded: &Loaded,
    methods: &[ResolvedMethod],
    cfg: &RunConfig,
    fresh: bool,
    mode: dataset::Mode,
    threads: usize,
    writer: &mut raw::RawWriter<W>,
) -> Result<results::DatasetResult> {
    // The searched DB: the full base, or a seeded subsample when `n_base` is set.
    let db_storage = db_subsample(loaded, cfg)?;
    let subsampled = db_storage.is_some();
    let db = db_storage.as_ref().unwrap_or(&loaded.base);
    let (n_base, dim) = db.dim();
    // What a stored code file must match to be reusable here, from row counts alone —
    // which is what lets the encode-side costs below wait on a cache miss.
    let id = identity_for(
        loaded.base.nrows(),
        loaded.calib.as_ref().map(Array2::nrows),
        dim,
        cfg,
    );
    // Whether anything will actually be encoded. Both the fit rows and the read buffer are
    // encode-side costs, and gathering `n_fit` rows off a streamed base is a positioned
    // read each, so neither is paid for a dataset whose every method is already stored.
    let encoding = fresh
        || methods
            .iter()
            .any(|m| id.stored(name, &m.label(vqb::catalog::display(&m.name))).is_none());
    // One read buffer for every method's encode pass, sized and allocated before the first
    // `encode_peak_bytes` window so the metric is never charged for it.
    let shape = encode_shape(dim, threads, mode);
    let mut scratch =
        vec![0f32; if encoding && db.resident().is_none() { shape.0 * dim } else { 0 }];

    let eval_idx = subset_indices(
        loaded.eval.nrows(),
        cfg.n_eval.unwrap_or(usize::MAX),
        cfg.seed ^ 0x9e37,
    );
    let eval = loaded.eval.select(Axis(0), &eval_idx);

    // Candidates + ground truth. The dataset's shipped neighbors index the full base,
    // so a subsampled DB must recompute the exact top-L candidates over the subsample.
    let cand_width = loaded.eval_candidates.first().map_or(0, Vec::len);
    let (candidates, true_scores) = if subsampled {
        // The subsample is resident (see `db_subsample`), so this pass owns its buffer.
        dataset::top_candidates(&eval, db, cand_width, &mut Vec::new())?
    } else {
        let cands: Vec<Vec<usize>> = eval_idx
            .iter()
            .map(|&i| loaded.eval_candidates[i].clone())
            .collect();
        let truths = true_scores(&eval, db, &cands)?;
        (cands, truths)
    };
    let n_candidates = candidates.first().map_or(0, Vec::len);

    // The candidate pool is baked into the dataset (`vqb data get --candidates L`), so
    // `config::validate` can't see its width — warn here when a requested k outruns it,
    // since recall/SOS@k are then clamped to the pool rather than the true top-k.
    if let Some(&k) = cfg.ks.iter().filter(|&&k| k > n_candidates).max() {
        eprintln!(
            "  warning: k={k} exceeds the {n_candidates}-candidate pool; \
             recall/SOS@k are clamped (rebuild with `vqb data get {name} --candidates {k}`)"
        );
    }

    let recon_idx = subset_indices(
        n_base,
        cfg.n_reconstruct.unwrap_or(usize::MAX),
        cfg.seed ^ 0xF17,
    );
    let references = rows_to_vec(&db.gather(&recon_idx)?);

    let calib = calib_subsample(loaded, cfg);

    // Base rows used only for `fit` (encode/score still run over the whole DB).
    let fit_storage = if encoding {
        let fit = fit_subsample(db, cfg)?;
        check_identity(&id, db, fit.as_ref(), calib.as_ref());
        fit
    } else {
        None
    };

    // The dataset's shared facts (no methods yet); `true_scores`/`references` move
    // in and are read back from `head` for scoring metrics and the progress table.
    let head = RawDataset {
        dataset: name.to_string(),
        dim,
        n_base,
        n_eval: eval.nrows(),
        n_candidates,
        candidates: candidates
            .iter()
            .map(|c| c.iter().map(|&i| i as u32).collect())
            .collect(),
        true_scores,
        recon_indices: recon_idx.iter().map(|&i| i as u32).collect(),
        references,
        methods: Vec::new(),
    };
    writer.begin_dataset(&head, methods.len())?;

    // Progress table: one row per quantizer, filled in as each finishes.
    let dk = if cfg.ks.contains(&10) {
        10
    } else {
        *cfg.ks.last().unwrap_or(&1)
    };
    table_header(name, dim, n_base, id.n_fit, head.n_eval, n_candidates, dk);

    let mut method_results = Vec::with_capacity(methods.len());
    for m in methods {
        let q = factory::build(m, cfg.seed, dim)?;
        let label = m.label(vqb::catalog::display(&m.name)); // method name "MinMax (b=2)"
        eprint!("  {label:<22}"); // label cell first; metrics fill on completion
        let _ = std::io::stderr().flush();
        // Reuse codes persisted by a prior `vqb encode` when their full identity
        // matches this config; otherwise encode in memory (keeps plain `run`
        // self-contained).
        let cache_path = codes::path_for(name, &label);
        let cached = if fresh { None } else { id.stored(name, &label) };
        // Thread count the cached codes were encoded with (for the reuse note).
        let reused_threads = cached.as_ref().map(codes::CodeStore::threads);
        let rm = match (cached, db.resident()) {
            (Some(store), _) => {
                run_method_cached(label, &*q, store, dim, &eval, &candidates, &recon_idx)
            }
            // Streamed: the code set has no more business in memory than the base does, so
            // encode through the store and score from it, exactly as `encode` + `run` would.
            (None, None) => {
                let store = fit_and_store(
                    &*q,
                    &cache_path,
                    &id,
                    &label,
                    db,
                    fit_rows(db, fit_storage.as_ref())?,
                    calib.as_ref().map(Array2::view),
                    threads,
                    shape,
                    &mut scratch,
                )?;
                run_method_cached(label, &*q, store, dim, &eval, &candidates, &recon_idx)
            }
            (None, Some(base)) => run_method(
                label,
                &*q,
                base,
                fit_rows(db, fit_storage.as_ref())?,
                &eval,
                calib.as_ref(),
                &candidates,
                &recon_idx,
            ),
        };
        eprintln!(
            "{}",
            row_tail(
                &rm,
                &head.true_scores,
                &head.references,
                &cfg.metrics,
                dk,
                cfg.seed,
            )
        );
        // Tell the user this method skipped fit+encode by reusing a code file.
        if let Some(t) = reused_threads {
            eprintln!(
                "    ↳ codes found at {} (encoded with {t} thread(s)) — skipping encode",
                cache_path.display()
            );
        }
        // Stream the raw capture, reduce it to results, then drop it — the heavy
        // approx_scores/recons for this method are freed before the next one.
        writer.write_method(&rm)?;
        method_results.push(aggregate::method_result(
            &head,
            &rm,
            &cfg.metrics,
            &cfg.ks,
            &cfg.temperatures,
            cfg.seed,
        ));
    }

    Ok(results::DatasetResult {
        dataset: name.to_string(),
        dim,
        n_base,
        n_eval: head.n_eval,
        n_candidates,
        methods: method_results,
    })
}

/// `vqb run <config>`: run the config, writing `results/raw/<exp>.raw` and
/// `results/<exp>.json`.
pub fn run(config_path: &Path, fresh: bool, mode: dataset::Mode) -> Result<()> {
    let cfg = RunConfig::parse(config_path)?;
    config::require_valid(&cfg)?;
    let exp = config_stem(config_path);
    let methods = cfg.expand();
    let threads = init_thread_pool(cfg.threads);
    eprintln!("encoding with {threads} thread(s)");
    announce_stream(mode, threads);
    if fresh {
        eprintln!("--fresh: ignoring stored codes, encoding from scratch");
    }

    let meta = RawMeta {
        name: exp.clone(),
        seed: cfg.seed,
        ks: cfg.ks.clone(),
        temperatures: cfg.temperatures.clone(),
        n_reconstruct: cfg.n_reconstruct.unwrap_or(0),
        timestamp: now_secs(),
        threads,
        cores: logical_cores(),
        arch: std::env::consts::ARCH.to_string(),
        os: std::env::consts::OS.to_string(),
    };

    // Open the raw capture and write its header now; each dataset/method is
    // appended as it finishes rather than buffered into one giant tree.
    std::fs::create_dir_all("results/raw").context("create results/raw")?;
    let raw_path = format!("results/raw/{exp}.raw");
    let file = std::io::BufWriter::new(
        std::fs::File::create(&raw_path).with_context(|| format!("creating {raw_path}"))?,
    );
    let mut writer = raw::RawWriter::new(file, &meta, cfg.datasets.len())?;

    let mut datasets = Vec::with_capacity(cfg.datasets.len());
    for ds in &cfg.datasets {
        let entry = registry::resolve(ds)?;
        eprint!("\nloading {} … ", entry.name);
        let _ = std::io::stderr().flush();
        let t = Instant::now();
        let loaded = dataset::load(&entry.local_path(), mode)?;
        eprintln!("{:.1}s", t.elapsed().as_secs_f64());
        datasets.push(run_dataset(
            entry.name,
            &loaded,
            &methods,
            &cfg,
            fresh,
            mode,
            threads,
            &mut writer,
        )?);
    }
    writer.finish()?;

    let run = Run {
        meta: RunMeta {
            name: meta.name,
            seed: meta.seed,
            ks: meta.ks,
            temperatures: meta.temperatures,
            n_reconstruct: meta.n_reconstruct,
            timestamp: meta.timestamp,
            threads: meta.threads,
            cores: meta.cores,
            arch: meta.arch.clone(),
            os: meta.os.clone(),
        },
        datasets,
    };
    let json_path = format!("results/{exp}.json");
    results::write_json(Path::new(&json_path), &run)?;

    println!("wrote {raw_path} and {json_path}");
    Ok(())
}

/// `vqb eval <config> <raw>`: recompute metrics from a `.raw` capture.
pub fn eval(config_path: &Path, raw_arg: &Path) -> Result<()> {
    let cfg = RunConfig::parse(config_path)?;
    config::require_valid(&cfg)?;
    let exp = config_stem(config_path);
    let raw_path = if raw_arg.is_file() {
        raw_arg.to_path_buf()
    } else {
        raw_arg.join(format!("{exp}.raw"))
    };
    let data = raw::read(&raw_path)?;
    let run = aggregate::run(&data, &cfg.metrics, &cfg.ks, &cfg.temperatures);
    std::fs::create_dir_all("results").context("create results")?;
    let json_path = format!("results/{exp}.json");
    results::write_json(Path::new(&json_path), &run)?;
    println!("wrote {json_path}");
    Ok(())
}

/// `vqb encode <config>`: fit + encode every dataset × method and stream the
/// codes to `results/codes/<dataset>/<method>.codes` for later `run` to reuse.
/// No scoring or reconstruction — this is the memory-bounded encode pass. Methods
/// whose codes are already stored under a matching identity are skipped unless
/// `fresh`.
pub fn encode_to_disk(config_path: &Path, fresh: bool, mode: dataset::Mode) -> Result<()> {
    let cfg = RunConfig::parse(config_path)?;
    config::require_valid(&cfg)?;
    let methods = cfg.expand();
    let threads = init_thread_pool(cfg.threads);
    eprintln!("encoding with {threads} thread(s)");
    announce_stream(mode, threads);
    if fresh {
        eprintln!("--fresh: ignoring stored codes, encoding from scratch");
    }

    for ds in &cfg.datasets {
        let entry = registry::resolve(ds)?;
        // With every method already stored, skip the multi-GB load as well — the
        // identity comes from the dataset's shapes, which cost only a header read.
        if !fresh && fully_encoded(entry, &methods, &cfg) {
            eprintln!(
                "\n{} — all {} method(s) cached, skipping",
                entry.name,
                methods.len()
            );
            continue;
        }
        eprint!("\nloading {} … ", entry.name);
        let _ = std::io::stderr().flush();
        let t = Instant::now();
        let loaded = dataset::load(&entry.local_path(), mode)?;
        eprintln!("{:.1}s", t.elapsed().as_secs_f64());
        encode_dataset(entry.name, &loaded, &methods, &cfg, mode, threads, fresh)?;
    }
    Ok(())
}

/// Whether every method's codes are already stored under this config's identity,
/// judged from the dataset's shapes alone. A dataset whose shapes can't be read is
/// never skipped — `dataset::load` reports that failure with a better message.
fn fully_encoded(entry: &registry::Dataset, methods: &[ResolvedMethod], cfg: &RunConfig) -> bool {
    let Ok(shapes) = dataset::identity_shapes(&entry.local_path()) else {
        return false;
    };
    let id = identity_for(shapes.base_rows, shapes.calib_rows, shapes.dim, cfg);
    methods.iter().all(|m| {
        let label = m.label(vqb::catalog::display(&m.name));
        id.stored(entry.name, &label).is_some()
    })
}

/// Fit + encode every method on one dataset, streaming codes to disk. The DB /
/// calib / fit subsamples match `run_dataset` exactly, so the codes align
/// row-for-row with the DB `run` will address.
fn encode_dataset(
    name: &str,
    loaded: &Loaded,
    methods: &[ResolvedMethod],
    cfg: &RunConfig,
    mode: dataset::Mode,
    threads: usize,
    fresh: bool,
) -> Result<()> {
    let db_storage = db_subsample(loaded, cfg)?;
    let db = db_storage.as_ref().unwrap_or(&loaded.base);
    let (n_base, dim) = db.dim();
    let calib = calib_subsample(loaded, cfg);
    let fit_storage = fit_subsample(db, cfg)?;
    let id = identity_for(
        loaded.base.nrows(),
        loaded.calib.as_ref().map(Array2::nrows),
        dim,
        cfg,
    );
    check_identity(&id, db, fit_storage.as_ref(), calib.as_ref());
    // As in `run_dataset`: one read buffer for every method, allocated before the first
    // measurement window.
    let shape = encode_shape(dim, threads, mode);
    let mut scratch = vec![0f32; if db.resident().is_some() { 0 } else { shape.0 * dim }];

    for m in methods {
        let label = m.label(vqb::catalog::display(&m.name));
        let path = codes::path_for(name, &label);
        // Before `fit` — that's where the cost starts. Reached only on a partial
        // hit; an all-cached dataset never gets here (see `fully_encoded`).
        if !fresh && id.stored(name, &label).is_some() {
            eprintln!("  {label:<22} → {} (cached — skipping)", path.display());
            continue;
        }
        let q = factory::build(m, cfg.seed, dim)?;
        let store = fit_and_store(
            &*q,
            &path,
            &id,
            &label,
            db,
            fit_rows(db, fit_storage.as_ref())?,
            calib.as_ref().map(Array2::view),
            threads,
            shape,
            &mut scratch,
        )?;
        eprintln!(
            "  {label:<22} → {} ({}, fit {:.1}s, enc {:.1}s, {threads} thread(s))",
            path.display(),
            store_size(&store, n_base),
            store.fit_s(),
            store.encode_s(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub stage with a content-only, row-independent code (one byte per row),
    /// so chunked encoding must reproduce the one-shot codes exactly.
    struct RowByte;
    impl vqb::Primitive for RowByte {
        fn describe() -> &'static str {
            "one content byte per row (test stage)"
        }
        fn encode(&self, _m: &[u8], v: ArrayView2<f32>) -> Vec<Vec<u8>> {
            v.rows()
                .into_iter()
                .map(|r| vec![(r.sum() as i64 as i8) as u8])
                .collect()
        }
        fn apply(&self, _m: &[u8], _v: &mut Array2<f32>, _c: &[&[u8]]) {}
        fn reconstruct(&self, _m: &[u8], _c: &[&[u8]], _child: Option<ArrayView2<f32>>) -> Array2<f32> {
            Array2::zeros((0, 0))
        }
        fn score(
            &self,
            _m: &[u8],
            q: ArrayView2<f32>,
            c: &[&[u8]],
            _child: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            Array2::zeros((q.nrows(), c.len()))
        }
    }

    #[test]
    fn encode_in_chunks_matches_one_shot() {
        let base = Array2::from_shape_fn((25, 3), |(i, j)| (i * 2 + j) as f32 - 5.0);
        let q = vqb::AsQuantizer(vqb::Pipeline::new(3, vec![Box::new(RowByte)]).unwrap());
        let model = q.fit(base.view(), None);
        let one_shot = q.encode(&model, base.view());
        // Every chunk size (incl. 1, an exact divisor, and larger-than-n) agrees, and
        // the row order is preserved.
        for chunk in [1usize, 4, 5, 25, 100] {
            assert_eq!(
                encode_in_chunks(&q, &model, base.view(), chunk),
                one_shot,
                "chunk {chunk}"
            );
        }
    }

    /// Stub stage whose code length depends on the row's content, so a quantizer can
    /// emit a different number of bytes per vector. Content-only, like `RowByte`, so
    /// chunking still can't change the output.
    struct RaggedRow;
    impl vqb::Primitive for RaggedRow {
        fn describe() -> &'static str {
            "a content-length code (test stage)"
        }
        fn encode(&self, _m: &[u8], v: ArrayView2<f32>) -> Vec<Vec<u8>> {
            v.rows()
                .into_iter()
                .map(|r| vec![r.len() as u8; r.sum().abs() as usize % 4 + 1])
                .collect()
        }
        fn apply(&self, _m: &[u8], _v: &mut Array2<f32>, _c: &[&[u8]]) {}
        fn reconstruct(&self, _m: &[u8], _c: &[&[u8]], _child: Option<ArrayView2<f32>>) -> Array2<f32> {
            Array2::zeros((0, 0))
        }
        fn score(
            &self,
            _m: &[u8],
            q: ArrayView2<f32>,
            c: &[&[u8]],
            _child: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            Array2::zeros((q.nrows(), c.len()))
        }
        fn code_bytes(&self, _m: &[u8], _in_dim: usize) -> Option<usize> {
            None
        }
    }

    /// A quantizer whose per-vector code length varies survives the whole `encode`
    /// path: chunked encode → on-disk store → read back by row index, with the size
    /// metric reporting the actual sum of code lengths.
    #[test]
    fn variable_length_codes_round_trip_through_the_store() {
        let base = Array2::from_shape_fn((25, 3), |(i, j)| (i * 2 + j) as f32 - 5.0);
        let q = vqb::AsQuantizer(vqb::Pipeline::new(3, vec![Box::new(RaggedRow)]).unwrap());
        let model = q.fit(base.view(), None);
        let one_shot = q.encode(&model, base.view());
        assert!(
            one_shot.iter().any(|c| c.len() != one_shot[0].len()),
            "the stub must actually produce ragged codes"
        );

        let path = std::env::temp_dir().join("vqb_run_test_variable.codes");
        let id = codes::Identity {
            seed: 1,
            n_base: base.nrows(),
            dim: 3,
            n_fit: 25,
            n_calib: 0,
        };
        let mut w = codes::CodeWriter::create(&path, &id, 1, 0.0, 0, "Stub", &model).unwrap();
        for chunk in encode_in_chunks(&q, &model, base.view(), 4).chunks(4) {
            w.push_chunk(chunk).unwrap();
        }
        let (width, code_bytes) = w.finish(0.0, 0).unwrap();
        assert_eq!(width, None);
        assert_eq!(code_bytes, one_shot.iter().map(Vec::len).sum::<usize>());

        let store = codes::CodeStore::open(&path).unwrap();
        assert_eq!(store.code_bytes(), code_bytes);
        let gathered = Codes::Disk(store).gather(&(0..base.nrows()).collect::<Vec<_>>());
        assert_eq!(gathered, one_shot);
        std::fs::remove_file(&path).ok();
    }

    /// A block read serves at most one chunk per worker, and is a whole number of chunks
    /// so a streamed encode splits at the same rows a resident one does.
    #[test]
    fn encode_shape_keeps_parallelism_within_the_budget() {
        let mb = |n| dataset::Mode::Stream { block_mb: n };
        // Resident: a chunk per worker, full chunks — unchanged from before streaming.
        assert_eq!(encode_shape(768, 8, dataset::Mode::Resident), (8 * ENCODE_CHUNK, ENCODE_CHUNK));
        // A generous budget lands in the same place.
        assert_eq!(encode_shape(768, 8, mb(4096)), (8 * ENCODE_CHUNK, ENCODE_CHUNK));
        // A tight one shrinks the block *and* the chunk, so all 8 workers still get work.
        for budget in [1usize, 16, 64] {
            let (block, chunk) = encode_shape(768, 8, mb(budget));
            assert!(block * 768 * 4 <= (budget << 20).max(3072), "budget {budget}");
            assert_eq!(chunk, (block / 8).clamp(1, ENCODE_CHUNK), "budget {budget}");
            assert!(block >= chunk * 7, "budget {budget}: block splits over the workers");
        }
        // Never zero, however small the budget.
        assert_eq!(encode_shape(768, 8, mb(0)), (1, 1));
    }

    fn cfg_with(n_base: Option<usize>, n_fit: Option<usize>, n_calib: Option<usize>) -> RunConfig {
        RunConfig {
            datasets: vec![],
            methods: vec![],
            metrics: vec![],
            ks: vec![10],
            temperatures: vec![],
            seed: 7,
            n_reconstruct: None,
            n_eval: None,
            n_calib,
            n_base,
            n_fit,
            threads: None,
        }
    }

    #[test]
    fn resolved_counts_clamps_to_the_available_rows() {
        // Unset fields keep every row; a calib-less dataset resolves to 0.
        let all = cfg_with(None, None, None);
        assert_eq!(resolved_counts(100, None, &all), (100, 100, 0));
        assert_eq!(resolved_counts(100, Some(30), &all), (100, 100, 30));
        // Requests larger than the data clamp to it.
        let big = cfg_with(Some(500), Some(500), Some(500));
        assert_eq!(resolved_counts(100, Some(30), &big), (100, 100, 30));
        // `n_fit` clamps to the subsampled DB, not the full base.
        assert_eq!(
            resolved_counts(100, None, &cfg_with(Some(40), Some(60), None)),
            (40, 40, 0)
        );
        assert_eq!(
            resolved_counts(100, Some(30), &cfg_with(Some(40), Some(10), Some(5))),
            (40, 10, 5)
        );
    }

    /// `encode` skips a dataset on the shape-derived identity while `run` builds it
    /// from the loaded arrays; `check_identity` asserts the two agree, so drive it over
    /// the config shapes that make the subsamples differ.
    #[test]
    fn identity_agrees_with_the_loaded_subsamples() {
        let loaded = Loaded {
            base: Base::Mem(Array2::from_shape_fn((50, 4), |(i, j)| (i + j) as f32)),
            eval: Array2::zeros((2, 4)),
            calib: Some(Array2::zeros((20, 4))),
            eval_candidates: vec![vec![0], vec![1]],
        };
        for (nb, nf, nc) in [
            (None, None, None),
            (Some(30), Some(10), Some(5)),
            (Some(500), Some(500), Some(500)),
            (Some(50), None, None), // n_base == base rows: no subsample allocated
        ] {
            let cfg = cfg_with(nb, nf, nc);
            let db_storage = db_subsample(&loaded, &cfg).unwrap();
            let db = db_storage.as_ref().unwrap_or(&loaded.base);
            let calib = calib_subsample(&loaded, &cfg);
            let fit = fit_subsample(db, &cfg).unwrap();
            let id = identity_for(loaded.base.nrows(), Some(20), 4, &cfg);
            check_identity(&id, db, fit.as_ref(), calib.as_ref());
            assert_eq!((id.seed, id.dim), (7, 4));
        }
    }
}
