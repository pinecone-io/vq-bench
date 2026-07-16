//! `vqb` — the vq-bench run harness CLI.
//!
//! Drives JSON run configs over datasets and named quantizers. This is
//! the only place that touches datasets, metrics, and reporting (the library
//! stays a pure quantization toolkit).

mod config;
mod mem;
mod merge;
mod registry;
mod view;

// Counting allocator so the runner can measure peak heap during encoding. Must be
// installed in every build, independent of the `hdf5` feature.
#[global_allocator]
static GLOBAL: mem::Counting = mem::Counting;

// The run compute path needs HDF5 (dataset I/O); gated so the core CLI
// (`show`, `data list/info`, `run --dry-run`) still builds without the system lib.
#[cfg(feature = "hdf5")]
mod aggregate;
#[cfg(feature = "hdf5")]
mod bench;
#[cfg(feature = "hdf5")]
mod codes;
#[cfg(feature = "hdf5")]
mod dataset;
#[cfg(feature = "hdf5")]
mod factory;
#[cfg(feature = "hdf5")]
mod raw;
#[cfg(feature = "hdf5")]
mod results;
#[cfg(feature = "hdf5")]
mod run;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use config::RunConfig;

#[derive(Parser)]
#[command(
    name = "vqb",
    about = "vq-bench: run vector-quantization benchmarks",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Manage datasets.
    Data {
        #[command(subcommand)]
        action: DataCmd,
    },
    /// Run a config.
    Run {
        /// Path to the run config JSON.
        config: PathBuf,
        /// Validate the config and check availability without running.
        #[arg(long)]
        dry_run: bool,
        /// Encode from scratch this run, ignoring any stored codes.
        #[arg(long)]
        fresh: bool,
    },
    /// Run fit+encode on a config, storing the model and codes to disk for later evaluation.
    Encode {
        /// Path to the experiment JSON.
        config: PathBuf,
    },
    /// Recompute metrics from a prior run's `.raw` outputs.
    Eval { config: PathBuf, raw_dir: PathBuf },
    /// Merge results JSON files that share run metadata into one.
    Merge {
        /// Two or more results JSONs to combine (later files win on duplicates).
        #[arg(num_args = 2..)]
        inputs: Vec<PathBuf>,
        /// Output path (default: `results/merged.json`).
        #[arg(long, short)]
        out: Option<PathBuf>,
    },
    /// Render a results JSON to a standalone HTML dashboard and open it.
    View {
        /// Path to a results JSON (e.g. `results/<run>.json`).
        results: PathBuf,
        /// Output HTML path (default: `results/html/<stem>.html`).
        #[arg(long, short)]
        out: Option<PathBuf>,
        /// Write the HTML without opening it in a browser.
        #[arg(long)]
        no_open: bool,
    },
    /// List what's implemented in vq-bench.
    Show {
        #[command(subcommand)]
        what: ShowCmd,
    },
    /// Publish a results JSON to `docs/results/` (and rebuild the manifest).
    Publish {
        results: PathBuf,
        /// Skip rebuilding `docs/results/manifest.json` afterward.
        #[arg(long)]
        no_index: bool,
    },
    /// Rebuild the manifest from the published result JSONs.
    Index,
}

#[derive(Subcommand)]
enum DataCmd {
    /// List all datasets in the registry.
    #[command(visible_alias = "ls")]
    List,
    /// Download and format a dataset into the required HDF5 structure.
    Get { dataset: String },
    /// Print metadata for a dataset.
    #[command(visible_alias = "i")]
    Info { dataset: String },
}

#[derive(Subcommand)]
enum ShowCmd {
    /// List the implemented named quantizers.
    #[command(visible_alias = "q")]
    Quantizers,
    /// List the implemented primitives, by subdirectory.
    #[command(visible_alias = "p")]
    Primitives,
    /// List the metric names a config may request.
    #[command(visible_alias = "m")]
    Metrics,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Data { action } => data(action),
        Command::Run {
            config,
            dry_run,
            fresh,
        } => run(&config, dry_run, fresh),
        Command::Encode { config } => encode(&config),
        Command::Eval { config, raw_dir } => eval(&config, &raw_dir),
        Command::Merge { inputs, out } => merge::merge(&inputs, out.as_deref()),
        Command::View {
            results,
            out,
            no_open,
        } => view::write(&results, out.as_deref(), no_open),
        Command::Show { what } => show(what),
        Command::Publish { results, no_index } => publish(&results, no_index),
        Command::Index => index(),
    }
}

fn data(action: DataCmd) -> Result<()> {
    match action {
        DataCmd::List => {
            println!("{:<32} {:>5}  {:<6} LOCAL", "NAME", "DIM", "SOURCE");
            for d in registry::DATASETS {
                let local = if d.is_local() { "yes" } else { "no" };
                println!("{:<32} {:>5}  {:<6} {}", d.name, d.dim, d.source, local);
            }
            Ok(())
        }
        DataCmd::Info { dataset } => {
            let d = registry::resolve(&dataset)?;
            println!("name:   {}", d.name);
            println!("dim:    {}", d.dim);
            println!("source: {}", d.source);
            println!("url:    {}", d.url());
            println!("file:   {}", d.local_path().display());
            println!(
                "local:  {}",
                if d.is_local() { "present" } else { "missing" }
            );
            println!("arrays: base, calib, eval, eval_candidates");
            Ok(())
        }
        DataCmd::Get { dataset } => data_get(&dataset),
    }
}

#[cfg(feature = "hdf5")]
fn data_get(name: &str) -> Result<()> {
    dataset::get(registry::resolve(name)?)
}

#[cfg(not(feature = "hdf5"))]
fn data_get(_name: &str) -> Result<()> {
    bail!("`vqb data get` needs the hdf5 feature; rebuild with default features")
}

fn run(path: &std::path::Path, dry_run: bool, fresh: bool) -> Result<()> {
    let cfg = RunConfig::parse(path)?;
    let problems = cfg.validate();
    let runs = cfg.expand();

    if dry_run {
        let count = |n: Option<usize>| n.map_or_else(|| "all".to_string(), |n| n.to_string());
        println!("config: {}", path.display());
        println!("seed: {}", cfg.seed);
        println!(
            "n_base: {}  n_fit: {}  n_reconstruct: {}  n_eval: {}  n_calib: {}",
            count(cfg.n_base),
            count(cfg.n_fit),
            count(cfg.n_reconstruct),
            count(cfg.n_eval),
            count(cfg.n_calib)
        );
        println!("k: {:?}", cfg.ks);
        println!("temp: {:?}", cfg.temperatures);
        println!("metrics: {}", cfg.metrics.join(", "));
        // Show registry names (a config may hold a prefix); unresolvable names
        // fall back to the raw string and are reported in `problems`.
        let shown: Vec<&str> = cfg
            .datasets
            .iter()
            .map(|ds| registry::resolve(ds).map_or(ds.as_str(), |d| d.name))
            .collect();
        println!("\ndatasets ({}):", shown.len());
        for ds in &shown {
            println!("  - {ds}");
        }
        println!("\nruns ({} = methods × swept params):", runs.len());
        for ds in &shown {
            for m in &runs {
                println!("  - {ds} / {}", m.label(vqb::catalog::display(&m.name)));
            }
        }
        if problems.is_empty() {
            println!(
                "\nOK: {} dataset(s) × {} run(s) = {} runs",
                cfg.datasets.len(),
                runs.len(),
                cfg.datasets.len() * runs.len()
            );
            Ok(())
        } else {
            bail!("validation failed:\n  - {}", problems.join("\n  - "));
        }
    } else {
        config::require_valid(&cfg)?;
        real_run(path, fresh)
    }
}

#[cfg(feature = "hdf5")]
fn real_run(path: &Path, fresh: bool) -> Result<()> {
    run::run(path, fresh)
}

#[cfg(not(feature = "hdf5"))]
fn real_run(_path: &Path, _fresh: bool) -> Result<()> {
    bail!("`vqb run` needs the hdf5 feature; rebuild with default features")
}

fn encode(path: &Path) -> Result<()> {
    let cfg = RunConfig::parse(path)?;
    config::require_valid(&cfg)?;
    real_encode(path)
}

#[cfg(feature = "hdf5")]
fn real_encode(path: &Path) -> Result<()> {
    run::encode_to_disk(path)
}

#[cfg(not(feature = "hdf5"))]
fn real_encode(_path: &Path) -> Result<()> {
    bail!("`vqb encode` needs the hdf5 feature; rebuild with default features")
}

#[cfg(feature = "hdf5")]
fn eval(config: &Path, raw_dir: &Path) -> Result<()> {
    run::eval(config, raw_dir)
}

#[cfg(not(feature = "hdf5"))]
fn eval(_config: &Path, _raw_dir: &Path) -> Result<()> {
    bail!("`vqb eval` needs the hdf5 feature; rebuild with default features")
}

/// Directory of published results, served as the vq-bench.com site.
const PUBLISH_DIR: &str = "docs/results";

/// Copy a results JSON into `docs/results/`, then rebuild the manifest (unless skipped).
fn publish(results: &Path, no_index: bool) -> Result<()> {
    if !results.is_file() {
        bail!("no such results file: {}", results.display());
    }
    let name = results
        .file_name()
        .context("results path has no file name")?;
    let dest = Path::new(PUBLISH_DIR).join(name);
    std::fs::create_dir_all(PUBLISH_DIR).context("create docs/results")?;
    std::fs::copy(results, &dest).with_context(|| format!("copying to {}", dest.display()))?;
    println!("published {} -> {}", results.display(), dest.display());
    if !no_index {
        index()?;
    }
    Ok(())
}

/// Manifest written to `docs/results/manifest.json` — the index the site reads.
#[derive(Serialize)]
struct Manifest {
    generated: u64,
    runs: Vec<RunEntry>,
}
#[derive(Serialize)]
struct RunEntry {
    name: String,
    path: String,
    datasets: Vec<DatasetSummary>,
    timestamp: u64,
}
#[derive(Serialize)]
struct DatasetSummary {
    dataset: String,
    n_base: usize,
}

/// The fields the manifest needs from a published results file (others ignored).
#[derive(Deserialize)]
struct ResultHead {
    meta: MetaHead,
    datasets: Vec<DatasetHead>,
}
#[derive(Deserialize)]
struct MetaHead {
    name: String,
    #[serde(default)]
    timestamp: u64,
}
#[derive(Deserialize)]
struct DatasetHead {
    dataset: String,
    #[serde(default)]
    n_base: usize,
}

/// Rebuild `docs/results/manifest.json` by scanning the published result JSONs.
fn index() -> Result<()> {
    let dir = Path::new(PUBLISH_DIR);
    std::fs::create_dir_all(dir).context("create docs/results")?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {PUBLISH_DIR}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    paths.sort();

    let mut runs = Vec::new();
    for path in paths {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if name == "manifest.json" || path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        match serde_json::from_str::<ResultHead>(&text) {
            Ok(head) => runs.push(RunEntry {
                name: head.meta.name,
                path: name,
                datasets: head
                    .datasets
                    .into_iter()
                    .map(|d| DatasetSummary {
                        dataset: d.dataset,
                        n_base: d.n_base,
                    })
                    .collect(),
                timestamp: head.meta.timestamp,
            }),
            Err(e) => eprintln!("skipping {name}: not a results file ({e})"),
        }
    }
    runs.sort_by(|a, b| a.name.cmp(&b.name));

    let generated = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let manifest = Manifest { generated, runs };
    let out = dir.join("manifest.json");
    std::fs::write(&out, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("writing {}", out.display()))?;
    println!(
        "indexed {} run(s) -> {}",
        manifest.runs.len(),
        out.display()
    );
    Ok(())
}

fn show(what: ShowCmd) -> Result<()> {
    match what {
        ShowCmd::Quantizers => {
            println!("Quantizers:");
            for q in vqb::catalog::QUANTIZERS {
                println!("  {:<16} {}", q.key, q.describe);
            }
        }
        ShowCmd::Primitives => {
            println!("Primitives:");
            for (dir, prims) in vqb::primitive_catalog::groups() {
                println!("  {dir}");
                for (p, desc) in *prims {
                    println!("    {p:<14} {desc}");
                }
            }
        }
        ShowCmd::Metrics => {
            println!("Metrics (request in a config's `metrics`):");
            for (name, desc) in config::KNOWN_METRICS {
                println!("  {name:<12} {desc}");
            }
            println!("\nResource metrics (always reported):");
            for (name, desc) in config::RESOURCE_METRICS {
                println!("  {name:<26} {desc}");
            }
        }
    }
    Ok(())
}
