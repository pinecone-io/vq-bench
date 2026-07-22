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
use vqb::{NamedQuantizer, Quantizer};

use crate::config::{ResolvedMethod, RunConfig};
use crate::dataset::{self, Loaded};
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

/// The searched DB as a subsample of `base`, or `None` to use the full base.
fn db_subsample(loaded: &Loaded, cfg: &RunConfig) -> Option<Array2<f32>> {
    let idx = subset_indices(
        loaded.base.nrows(),
        cfg.n_base.unwrap_or(usize::MAX),
        cfg.seed ^ 0xba5e,
    );
    (idx.len() != loaded.base.nrows()).then(|| loaded.base.select(Axis(0), &idx))
}

/// The calibration queries passed to `fit`, subsampled from `calib`.
fn calib_subsample(loaded: &Loaded, cfg: &RunConfig) -> Option<Array2<f32>> {
    loaded.calib.as_ref().map(|c| {
        let idx = subset_indices(c.nrows(), cfg.n_calib.unwrap_or(usize::MAX), cfg.seed ^ 0xca11b);
        c.select(Axis(0), &idx)
    })
}

/// The base rows used only for `fit`, or `None` to fit on the whole DB.
fn fit_subsample(db: &Array2<f32>, cfg: &RunConfig) -> Option<Array2<f32>> {
    let idx = subset_indices(db.nrows(), cfg.n_fit.unwrap_or(usize::MAX), cfg.seed ^ 0xf17f);
    (idx.len() != db.nrows()).then(|| db.select(Axis(0), &idx))
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
fn encode_in_chunks<Q: Quantizer + Sync>(
    q: &Q,
    model: &[u8],
    base: &Array2<f32>,
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

/// Exact dot products of each eval query against its candidates.
fn true_scores(eval: &Array2<f32>, base: &Array2<f32>, candidates: &[Vec<usize>]) -> Vec<Vec<f32>> {
    candidates
        .iter()
        .enumerate()
        .map(|(qi, cand)| {
            let q = eval.row(qi);
            cand.iter().map(|&i| q.dot(&base.row(i))).collect()
        })
        .collect()
}

fn rows_to_vec(a: &Array2<f32>) -> Vec<Vec<f32>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// Exact top-`l` candidates (indices into `db`) and their true scores, per eval query.
/// Used when the searched DB is subsampled, since the dataset's shipped neighbors index
/// the full base and no longer apply.
fn recompute_candidates(
    eval: &Array2<f32>,
    db: &Array2<f32>,
    l: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let scores = vqb::matmul(eval.view(), db.t()); // n_eval × n_db
    let l = l.min(db.nrows());
    let mut candidates = Vec::with_capacity(eval.nrows());
    let mut truths = Vec::with_capacity(eval.nrows());
    for row in scores.rows() {
        let s = row.to_vec();
        let mut idx: Vec<usize> = (0..s.len()).collect();
        idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(l);
        truths.push(idx.iter().map(|&i| s[i]).collect());
        candidates.push(idx);
    }
    (candidates, truths)
}

// --- progress table --------------------------------------------------------

/// Print the per-dataset descriptor + column header + rule.
fn table_header(
    _name: &str,
    dim: usize,
    n_base: usize,
    n_eval: usize,
    n_candidates: usize,
    dk: usize,
) {
    eprintln!("  dim {dim} · base {n_base} · eval {n_eval} · n_cand {n_candidates}");
    eprintln!(
        "  {:<22}{:>10}{:>12}{:>13}{:>13}{:>9}{:>10}",
        "method",
        "bits/dim",
        format!("recall@{dk}"),
        "mse_score",
        "mse_recon",
        "enc(s)",
        "enc(MB)"
    );
    eprintln!("  {}", "─".repeat(89));
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
    let mse_s = if want("mse_score") {
        sci(bench::score_mse_bias(true_scores, &rm.approx_scores).0)
    } else {
        dash()
    };
    let mse_r = match (&rm.recons, want("mse_recon")) {
        (Some(r), true) => sci(bench::recon_mse(references, r)),
        _ => dash(),
    };
    let enc_mb = rm.encode_peak_bytes as f64 / 1e6;
    format!(
        "{:>10}{:>12}{:>13}{:>13}{:>9.1}{:>10.1}",
        fixed(rm.bits_per_dim),
        recall,
        mse_s,
        mse_r,
        rm.encode_s,
        enc_mb
    )
}

/// Run one quantizer by encoding the base in memory, capturing its raw outputs.
#[allow(clippy::too_many_arguments)]
fn run_method(
    label: String,
    q: &NamedQuantizer,
    base: &Array2<f32>,
    fit_base: ArrayView2<f32>,
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
    candidates: &[Vec<usize>],
    recon_idx: &[usize],
) -> RawMethod {
    let (n_base, dim) = base.dim();
    let model = q.fit(fit_base, calib.map(|c| c.view()));

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
    q: &NamedQuantizer,
    store: codes::CodeStore,
    dim: usize,
    eval: &Array2<f32>,
    candidates: &[Vec<usize>],
    recon_idx: &[usize],
) -> RawMethod {
    let n_base = store.len();
    let model = store.model().to_vec();
    let code_bytes = store.code_bytes();
    let encode_s = store.encode_s();
    let encode_peak_bytes = store.encode_peak_bytes();
    score_and_reconstruct(
        label,
        q,
        &model,
        n_base,
        dim,
        code_bytes,
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
    q: &NamedQuantizer,
    model: &[u8],
    n_base: usize,
    dim: usize,
    code_bytes: usize,
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
fn run_dataset<W: std::io::Write>(
    name: &str,
    loaded: &Loaded,
    methods: &[ResolvedMethod],
    cfg: &RunConfig,
    fresh: bool,
    writer: &mut raw::RawWriter<W>,
) -> Result<results::DatasetResult> {
    // The searched DB: the full base, or a seeded subsample when `n_base` is set.
    let db_storage = db_subsample(loaded, cfg);
    let subsampled = db_storage.is_some();
    let db = db_storage.as_ref().unwrap_or(&loaded.base);
    let (n_base, dim) = db.dim();

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
        recompute_candidates(&eval, db, cand_width)
    } else {
        let cands: Vec<Vec<usize>> = eval_idx
            .iter()
            .map(|&i| loaded.eval_candidates[i].clone())
            .collect();
        let truths = true_scores(&eval, db, &cands);
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
    let references: Vec<Vec<f32>> = recon_idx.iter().map(|&i| db.row(i).to_vec()).collect();

    let calib = calib_subsample(loaded, cfg);

    // Base rows used only for `fit` (encode/score still run over the whole DB).
    let fit_storage = fit_subsample(db, cfg);

    // Resolved fit/calib row counts — part of a code file's identity, since they
    // determine the model. Computed identically in `encode_dataset`.
    let n_fit = fit_storage.as_ref().map_or(n_base, |f| f.nrows());
    let n_calib = calib.as_ref().map_or(0, |c| c.nrows());

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
    table_header(name, dim, n_base, head.n_eval, n_candidates, dk);

    let mut method_results = Vec::with_capacity(methods.len());
    for m in methods {
        let q = factory::build(m, cfg.seed, dim)?;
        let label = m.label(&q.name); // method name "MinMax (b=2)" from the family name
        eprint!("  {label:<22}"); // label cell first; metrics fill on completion
        let _ = std::io::stderr().flush();
        // Reuse codes persisted by a prior `vqb encode` when their full identity
        // matches this config; otherwise encode in memory (keeps plain `run`
        // self-contained).
        let cache_path = codes::path_for(name, &label);
        let cached = if fresh {
            None
        } else {
            codes::CodeStore::open(&cache_path)
                .ok()
                .filter(|s| s.matches(cfg.seed, n_base, dim, n_fit, n_calib, &label))
        };
        // Thread count the cached codes were encoded with (for the reuse note).
        let reused_threads = cached.as_ref().map(codes::CodeStore::threads);
        let rm = match cached {
            Some(store) => run_method_cached(label, &q, store, dim, &eval, &candidates, &recon_idx),
            None => {
                let fit_base = fit_storage.as_ref().map_or_else(|| db.view(), |f| f.view());
                run_method(
                    label,
                    &q,
                    db,
                    fit_base,
                    &eval,
                    calib.as_ref(),
                    &candidates,
                    &recon_idx,
                )
            }
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
pub fn run(config_path: &Path, fresh: bool) -> Result<()> {
    let cfg = RunConfig::parse(config_path)?;
    config::require_valid(&cfg)?;
    let exp = config_stem(config_path);
    let methods = cfg.expand();
    let threads = init_thread_pool(cfg.threads);
    eprintln!("encoding with {threads} thread(s)");
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
        let loaded = dataset::load(&entry.local_path())?;
        eprintln!("{:.1}s", t.elapsed().as_secs_f64());
        datasets.push(run_dataset(entry.name, &loaded, &methods, &cfg, fresh, &mut writer)?);
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
/// codes to `results/codes/<exp>/<dataset>/<method>.codes` for later `run` to
/// reuse. No scoring or reconstruction — this is the memory-bounded encode pass.
pub fn encode_to_disk(config_path: &Path) -> Result<()> {
    let cfg = RunConfig::parse(config_path)?;
    config::require_valid(&cfg)?;
    let methods = cfg.expand();
    let threads = init_thread_pool(cfg.threads);
    eprintln!("encoding with {threads} thread(s)");

    for ds in &cfg.datasets {
        let entry = registry::resolve(ds)?;
        eprint!("\nloading {} … ", entry.name);
        let _ = std::io::stderr().flush();
        let t = Instant::now();
        let loaded = dataset::load(&entry.local_path())?;
        eprintln!("{:.1}s", t.elapsed().as_secs_f64());
        encode_dataset(entry.name, &loaded, &methods, &cfg, threads)?;
    }
    Ok(())
}

/// Fit + encode every method on one dataset, streaming codes to disk. The DB /
/// calib / fit subsamples match `run_dataset` exactly, so the codes align
/// row-for-row with the DB `run` will address.
fn encode_dataset(
    name: &str,
    loaded: &Loaded,
    methods: &[ResolvedMethod],
    cfg: &RunConfig,
    threads: usize,
) -> Result<()> {
    let db_storage = db_subsample(loaded, cfg);
    let db = db_storage.as_ref().unwrap_or(&loaded.base);
    let (n_base, dim) = db.dim();
    let calib = calib_subsample(loaded, cfg);
    let fit_storage = fit_subsample(db, cfg);
    let n_fit = fit_storage.as_ref().map_or(n_base, |f| f.nrows());
    let n_calib = calib.as_ref().map_or(0, |c| c.nrows());

    // Encode chunks concurrently but flush them in row order, so the on-disk codes are
    // byte-identical to a serial encode regardless of thread count. Processing one
    // window of `threads` chunks at a time bounds peak memory to ~that many chunks —
    // the disk store, not RAM, holds the full code set.
    let window = threads.max(1);

    for m in methods {
        let q = factory::build(m, cfg.seed, dim)?;
        let label = m.label(&q.name);
        let fit_base = fit_storage.as_ref().map_or_else(|| db.view(), |f| f.view());
        let model = q.fit(fit_base, calib.as_ref().map(|c| c.view()));

        let path = codes::path_for(name, &label);
        let mut writer = codes::CodeWriter::create(
            &path, cfg.seed, dim, n_base, n_fit, n_calib, threads, &label, &model,
        )?;
        let baseline = mem::current();
        mem::reset_peak();
        let t = Instant::now();
        let ranges: Vec<(usize, usize)> = (0..n_base)
            .step_by(ENCODE_CHUNK)
            .map(|s| (s, (s + ENCODE_CHUNK).min(n_base)))
            .collect();
        for group in ranges.chunks(window) {
            let encoded: Vec<Vec<Vec<u8>>> = group
                .par_iter()
                .map(|&(start, end)| q.encode(&model, db.slice(s![start..end, ..])))
                .collect();
            for chunk in &encoded {
                writer.push_chunk(chunk)?;
            }
        }
        let encode_s = t.elapsed().as_secs_f64();
        let encode_peak_bytes = mem::peak().saturating_sub(baseline) as u64;
        let (width, _) = writer.finish(encode_s, encode_peak_bytes)?;
        eprintln!(
            "  {label:<22} → {} ({n_base} rows × {width} B, {encode_s:.1}s, {threads} thread(s))",
            path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub quantizer with a content-only, row-independent code (one byte per row),
    /// so chunked encoding must reproduce the one-shot codes exactly.
    struct RowByte;
    impl Quantizer for RowByte {
        fn fit(&self, _v: ArrayView2<f32>, _q: Option<ArrayView2<f32>>) -> Vec<u8> {
            Vec::new()
        }
        fn encode(&self, _m: &[u8], v: ArrayView2<f32>) -> Vec<Vec<u8>> {
            v.rows()
                .into_iter()
                .map(|r| vec![(r.sum() as i64 as i8) as u8])
                .collect()
        }
        fn reconstruct(&self, _m: &[u8], _c: &[&[u8]]) -> Array2<f32> {
            Array2::zeros((0, 0))
        }
        fn score(&self, _m: &[u8], q: ArrayView2<f32>, c: &[&[u8]]) -> Array2<f32> {
            Array2::zeros((q.nrows(), c.len()))
        }
    }

    #[test]
    fn encode_in_chunks_matches_one_shot() {
        let base = Array2::from_shape_fn((25, 3), |(i, j)| (i * 2 + j) as f32 - 5.0);
        let q = RowByte;
        let model = q.fit(base.view(), None);
        let one_shot = q.encode(&model, base.view());
        // Every chunk size (incl. 1, an exact divisor, and larger-than-n) agrees, and
        // the row order is preserved.
        for chunk in [1usize, 4, 5, 25, 100] {
            assert_eq!(
                encode_in_chunks(&q, &model, &base, chunk),
                one_shot,
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn recompute_candidates_picks_exact_top_l() {
        // One query [1,0]; dots with db rows: 0.0, 1.0, 0.5, -1.0 → top-2 is rows 1 then 2.
        let eval = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let db = Array2::from_shape_vec((4, 2), vec![0., 1., 1., 0., 0.5, 0., -1., 0.]).unwrap();
        let (cands, truths) = recompute_candidates(&eval, &db, 2);
        assert_eq!(cands, vec![vec![1, 2]]);
        assert_eq!(truths, vec![vec![1.0, 0.5]]);
        // Scores are descending within each query's candidates.
        assert!(truths[0].windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn recompute_candidates_clamps_l_to_db_size() {
        let eval = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let db = Array2::from_shape_vec((2, 2), vec![1., 0., 0., 1.]).unwrap();
        let (cands, _) = recompute_candidates(&eval, &db, 100);
        assert_eq!(cands[0].len(), 2);
    }
}
