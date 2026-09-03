# VQ-bench: a composable vector quantization framework

## About 

VQ-bench is an open-source benchmark for vector quantization. It is maintained by [Amir Ingber](https://scholar.google.com/citations?user=0IkkBzQAAAAJ&hl=en), [Edo Liberty](https://edoliberty.com/) ([Pinecone](https://pinecone.io/)), and [Ashwin Padaki](https://apadaki.github.io/) (University of Pennsylvania).

See [vq-bench.com](https://www.vq-bench.com) for current benchmarks.

## Usage

This page explains how to use VQ-bench.

### Installation

Begin by cloning the `vq-bench` repository and installing the `vqb` binary.

```bash
cd vq-bench
cargo install --path .    # installs the vqb binary
```

Run `vqb` to see the full list of commands.

### Downloading a dataset

The first thing you'll want to do is download a dataset using the following commands:
```bash
vqb data list          # list all available datasets
vqb data get <name>    # download a dataset
```

Dataset names accept any unique prefix; for example, `vqb data get arxiv` will download the dataset `arxiv-nomic-768-normalized`.

Every dataset consists of four components:
- `base` is the set of vectors to encode and score.
- `eval` is the set of evaluation queries and `eval_candidates` are the candidates to be scored.
- `calib` is an optional set of calibration queries.

By default, `eval_candidates` consists of the top 100 candidates for each query. To use a different number of candidates per query, run `vqb data get <name> -l <num_candidates>`.

### Writing a run configuration

A run configuration is a single JSON file describing the datasets, evaluation parameters, quantizers, and quantizer parameters to run. Refer to the following table:

| Field | Meaning |
|---|---|
| `datasets` | dataset names (or unique prefixes) to run |
| `methods` | quantizers to run, formatted as `{ "name": <key>, params }` |
| `metrics` | quality metrics to report (default: all metrics) |
| `k` | the `k` values for recall@k / SOS@k (default `[1, 10, 50]`) |
| `temp` | softmax temperatures for the `kl` / `tv` metrics (default `[0.5, 1.0, 2.0]`) |
| `seed` | master seed for all sampling and seeded primitives (default `1`) |
| `n_reconstruct`, `n_eval` | the number of reconstructions and evaluation queries to measure |
| `n_base`, `n_fit`, `n_calib` | optionally subsample the base vectors, fit vectors, or calibration queries |
| `threads` | number of threads used in encoding (default: all logical cores; overridden by the environment variable `RAYON_NUM_THREADS`) |

Run `vqb show quantizers` and `vqb show metrics` to see the names of the quantizers and metrics supported in VQ-bench. The following is an example of a run configuration which evaluates two quantizers on the `arxiv` dataset using three quality metrics.

```json
// configs/minmax-compare.json
{
  "datasets": ["arxiv"],
  "seed": 1,
  "n_reconstruct": 1000,
  "n_eval": 1000,
  "k": [1, 10, 20],
  "methods": [
    { "name": "minmax", "b": [2, 4, 6] },
    { "name": "e_rabitq", "b": [2, 4, 6] }
  ],
  "metrics": ["recall", "mse_score", "mse_recon"]
}
```

### Running a benchmark

The next step is to run the configuration file. Always begin with a `--dry-run` first: it validates the datasets, quantizers, metrics, and parameter names and prints the full run details without computing anything.

```bash
vqb run configs/minmax-compare.json --dry-run    # validate the config
vqb run configs/minmax-compare.json              # run the config
```

Each run measures size (bits per dimension) and the given quality metrics, as well as additional resource metrics (fit and encode time and peak memory, per-query latency, etc.). Results are written to `results/`, including a `.raw` capture of the reconstructions and scoring
values, and an aggregated `.json`. See [Output locations](#output-locations) to put them elsewhere.

The remaining commands split `run` into reusable stages: `encode` does the
expensive fit + encode once, and `eval` recomputes the metrics from an existing `.raw`.
```bash
vqb run    <config> --fresh    # re-encode from scratch, ignoring any cached codes
vqb encode <config>            # cache codes to results/codes/ for a later run to reuse
vqb encode <config> --fresh    # re-encode from scratch, overwriting cached codes
vqb eval   <config> <raw>      # recompute metrics directly from a prior run's .raw
vqb merge  <a> <b>             # combine JSON results files
```

### Output locations

The following environment variables specify where datasets, codes, and results are stored.

| Flag | Environment | Default | Holds |
|---|---|---|---|
| `--data-dir` | `VQB_DATA_DIR` | `./data` | downloaded datasets |
| `--results-dir` | `VQB_RESULTS_DIR` | `./results` | results JSON, plus `raw/` and `html/` |
| `--codes-dir` | `VQB_CODES_DIR` | `<results-dir>/codes` | per-method code stores |
| `--publish-dir` | `VQB_PUBLISH_DIR` | `./docs/results` | what `publish` copies in and the site serves |

### Streaming a dataset larger than memory

By default every command loads the base vectors whole, so a dataset has to fit in RAM. `--stream` reads them from disk a block at a time of size `--block-mb` (default 256MB).

```bash
vqb data get <name> -l 100 --stream   # recompute candidates without loading the full base
vqb encode <config> --stream          # fit + encode a base larger than RAM
vqb run    <config> --stream          # ... and score it too
```

`--stream` requires `n_fit` and `n_reconstruct` in the config, since both otherwise default to every base row.

### Viewing the results

After the run completes, use `vqb view <results.json>` to open a standalone HTML visualization.

## Contributing

We welcome two main types of contributions to vq-bench: adding a **primitive** (one stage of a quantization pipeline), and adding a **quantizer** (a named family that run configurations can select).

### Adding a primitive

A primitive is one stage of a quantization pipeline: a conditioner, rounder, or splitter (see the [overview](https://www.vq-bench.com/docs.html#overview)). In order to add a primitive to vq-bench, you must do two things:

1. **Write a primitive file `src/primitives/<group>/<name>.rs`**, where `<group>` is `conditioners`, `rounders`, or `splitters`.
2. **Add one line in that group's `mod.rs`**, inside the `primitives! { ... }` list:

```rust
primitives! { Primitive:
    minmax => MinMax,
    // ...
    center => Center,
}
```

In the primitive file, you must specify a public type (e.g., `MinMax` or `Center`), and this type must implement the `Primitive` trait. (Splitters instead implement the `Splitter` trait in `src/splitter.rs`, whose methods return one output per branch.)

#### The Primitive trait

The `Primitive` trait (`src/primitive.rs`) specifies everything a stage must do for the pipeline to fit, encode, reconstruct, and score through it.

Required:

| Method | What it does |
|---|---|
| `describe()` | one-line description printed by `vqb show p` |
| `apply(model, vectors, codes)` | transform the batch of vectors into what downstream stages see (a conditioner applies its transformation; a rounder passes down its residual) |
| `reconstruct(model, codes, child_recons)` | rebuild the vectors from the codes and the next stage's reconstruction |
| `score(model, queries, codes, child_scores)` | estimate query–vector dot products from the codes and the next stage's scores |

Defaulted (override when applicable):

| Method | Default | What it does |
|---|---|---|
| `fit(vectors, queries)` | empty model | store learned information into a model |
| `encode(model, vectors)` | empty code per vector | generate per-vector bits |
| `apply_queries(model, queries)` | identity | transform the batch of queries into what downstream stages see |
| `code_bytes(model, in_dim)` | `None` (varies) | specifies the length of the per-vector codes (in bytes) if fixed; `None` on an empty model when the layout is learned at fit, or whenever the codes vary per vector — the pipeline then length-prefixes each one |
| `in_dim()` / `out_dim(in_dim)` | unchanged | specifies the input and output dimensionality of the primitive |

#### Example

`src/primitives/conditioners/center.rs`, in full (tests trimmed):

```rust
//! CENTER: subtracts the center from every vector
//! -
//! Model: mu, the mean over the fit vectors
//! Code for vector x: empty
//! Apply: x --> x - mu
//! Reconstruct: y --> y + mu
//! Score: s --> s + <q, mu>

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{coding, math, Primitive};

pub struct Center;

/// The mean mu, read from the model bytes.
fn mean(model: &[u8]) -> Array1<f32> {
    coding::unpack_model(model)
}

impl Primitive for Center {
    fn describe() -> &'static str {
        "subtract the mean over the fit set from every vector"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.mean_axis(Axis(0)).unwrap())
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        let (n_v, d) = vectors.dim();
        *vectors -= &mean(model).broadcast((n_v, d)).unwrap();
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("Center is not terminal").to_owned();
        let (n_v, d) = out.dim();
        out += &mean(model).broadcast((n_v, d)).unwrap();
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_scores.expect("Center is not terminal").to_owned();
        math::offset_rows(&mut out, queries.dot(&mean(model)).view()); // add <q, mu> per query
        out
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}
```

#### Verify

```bash
cargo test
cargo run -- show primitives    # the new row appears
```

### Adding a quantizer

A quantizer is an object that can support the quantization operations required by the harness. After building a quantizer, a run configuration can select in its `methods` list to evaluate it. Quantizers are typically constructed by specifying a pipeline of primitives, but vq-bench permits non-pipelined quantizers as well. In order to add a quantizer to vq-bench, you must do two things:

1. **Write a quantizer file `src/quantizers/<key>.rs`**, where `<key>` is the config name (e.g., `pq`).
2. **Add one line in `src/quantizers/mod.rs`**, inside the `quantizers! { ... }` list:

```rust
quantizers! {
    minmax => MinMax,
    // ...
    pq => Pq,
}
```

In the quantizer file, you must specify a public type (e.g., `MinMax` or `Pq`), and this type must implement the `Quantizer` trait.

#### The Quantizer trait

The `Quantizer` trait (`src/quantizer.rs`) specifies the quantizer's metadata, how to build it from config params, and the four quantizer operations performed by the harness.

Required:

| Method | What it does |
|---|---|
| `name()` | the config key (e.g., `"minmax"`) |
| `describe()` | one-line description printed by `vqb show q` |
| `build(params, seed, dim)` | parse the params and construct the quantizer |
| `fit(vectors, queries)` | learn a model from the fit vectors and an optional query sample |
| `encode(model, vectors)` | encode vectors into per-vector codes |
| `reconstruct(model, codes)` | reconstruct one vector per code |
| `score(model, queries, codes)` | estimate the score of each query against each candidate |

For a pipelined quantizer, `crate::pipeline_quantizer!();` implements `fit`, `encode`, `reconstruct`, and `score` by calling the pipeline (assuming that `self.0` is the quantizer's pipeline object).

Defaulted (override when applicable):

| Method | Default | What it does |
|---|---|---|
| `display_name()` | the type's name | the display name used in results (`Pq` overrides it to `"PQ"`) |
| `params()` | none | the accepted param names |
| `verify_params(params)` | flags unknown param names | checks that user parameters are valid |

#### Example

`src/quantizers/minmax.rs`, in full (tests trimmed):

```rust
//! `minmax`: per-vector rescale to `[0, 1]`, then a `b`-bit uniform lattice.

use anyhow::{ensure, Result};

use super::catalog::get;
use crate::coding::CodeLayout;
use crate::MinMax as MinMaxStage;
use crate::{CastUint, Params, Pipeline, Quantizer};

/// The `minmax` family. `get` type-checks `b`; value/range checks live in `build`.
pub struct MinMax(pub Pipeline);

impl MinMax {
    /// Rescale each vector to `[0, 1]`, then a `bits`-bit uniform lattice.
    pub fn pipeline(bits: u8, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Pipeline::new(
            dim,
            vec![Box::new(MinMaxStage::default()), Box::new(CastUint::new(bits))],
        )
    }
}

impl Quantizer for MinMax {
    fn name() -> &'static str {
        "minmax"
    }

    fn params() -> &'static [&'static str] {
        &["b"]
    }

    fn describe() -> &'static str {
        "MinMax -> CastUint(b)"
    }

    fn build(p: &Params, _seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(get(p, "b")?, dim)?))
    }

    crate::pipeline_quantizer!();
}
```

For a fan-out family (one sub-pipeline per vector segment), copy `src/quantizers/pq.rs` instead: it wraps a splitter and a per-branch child factory in a single `Split::from_factory` stage — children are built from the fitted layout, so a splitter may learn its branch widths from the data.

The new family is immediately selectable from a run configuration — `"name"` matches `name()`, sibling keys must appear in `params()`, and an array sweeps that param:

```json
"methods": [
    { "name": "minmax", "b": [2, 4, 6] }
]
```

#### Verify

```bash
cargo test
cargo run -- show quantizers                 # the new family appears
cargo run -- run <config> --dry-run          # the config builds it
```
