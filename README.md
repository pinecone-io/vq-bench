# VQ-bench: a composable vector quantization framework

See [vq-bench.com](https://www.vq-bench.com) for current benchmarks.

## About 

VQ-bench is an open benchmark for vector quantization (VQ) algorithms. It implements a small set of algorithmic primitives (in Rust) and builds quantizers by composing those primitives into pipelines. It then measures how each quantizer trades off **size** (bits per dimension) against **quality** (reconstruction error, score error, recall, etc.) across several high-dimensional embedding datasets, and lets you explore the trade-offs directly.


## Motivation

Vector quantization is an old problem, but it is now central to modern AI:
vector databases use it for large-scale approximate nearest neighbor search,
LLM serving uses it to compress model weights, and growing context windows make
KV-cache compression a memory bottleneck. The result is a rapid increase in quantization papers making strong, hard-to-verify claims. These papers report on different datasets, measure different metrics, run on different hardware, and account for resources differently. VQ-bench aims to provide a single, fair, reproducible benchmark where quantizers are measured side by side.

## Composability

Most quantizers are built from a small, shared set of primitives —
**conditioners** (centering, random rotation, whitening, etc.), **rounders** (integer
casts, k-means codebooks, etc.), and **splitters / routers**.
VQ-bench expresses these primitives through a single interface, so a quantizer is just a pipeline of primitives:

```
PQ       = segment.kmeans
SimHash  = normalize.random_rotate.cast_hamming
E-RaBitQ = center.normalize.cast_angular
```

Because pipelines compose freely, VQ-bench makes it is easy to understand the relationship between different quantizers, create variations on existing quantizers, and invent new quantizers altogether.

## Usage

Everything is driven by the `vqb` command-line client: you download datasets, write a JSON run configuration, run it, and view the results.

### Installation

Begin by cloning the repo and installing the main binary.

```bash
cd vq-bench
cargo install --path . # installs the vqb binary
```

### Downloading datasets

Datasets are HDF5 files sourced from [VIBE](https://vector-index-bench.github.io) (all normalized, scored by dot product). Each holds `base` vectors to encode and search, `eval` queries, their `eval_candidates` (true top-neighbors), and optional `calib` calibration queries. Files resolve under `$VQB_DATA_DIR` (default `data/`).

```bash
vqb data list            # list of supported datasets
vqb data info <name>     # information for a given dataset
vqb data get <name>      # download dataset and reformat into a standard layout
```

Dataset names accept any unique prefix (`arxiv` resolves to `arxiv-nomic-768-…`).

### Writing a run configuration

A run configuration is one JSON file describing a the datasets, evaluation parameters, quantizers, and quantizer parameters to be run.

```json
{
  "datasets": ["arxiv-nomic-768-normalized"],
  "seed": 1,
  "n_eval": 1000,
  "k": [1, 10, 50],
  "methods": [
    { "name": "minmax", "b": [2, 4, 8] },
    { "name": "rabitq" }
  ],
  "metrics": ["recall", "mse_score", "mse_recon"]
}
```

| Field | Meaning |
|---|---|
| `datasets` | dataset names (or unique prefixes) to run |
| `methods` | quantizers to run; each is `{ "name": <family key>, …params }`, array params sweep |
| `metrics` | metrics to report (`recall`, `sos`, `mse_score`, `mse_recon`, `kl`, `tv`, …) |
| `k` | the `k` values for recall@k / SOS@k (default `[1, 10, 50]`) |
| `temp` | softmax temperatures for the `kl`/`tv` metrics (default `[0.5, 1.0, 2.0]`) |
| `seed` | master seed for all sampling and seeded primitives (default `1`) |
| `n_base`, `n_fit`, `n_reconstruct`, `n_eval`, `n_calib` | optionally subsample the DB, fit set, reconstruction set, eval queries, or calibration queries |

Run `vqb show quantizers`, `vqb show primitives` to list the implemented quantizers and primitives. Run `vqb show metrics` to list the metrics a config may reference. Configs live in `configs/`.

### Running a benchmark

Always `--dry-run` first: it validates every dataset, quantizer, and metric name and prints the full run matrix without computing anything.

```bash
vqb run configs/my-experiment.json --dry-run   # validate config file
vqb run configs/my-experiment.json             # run config file
```

Each run measures **size** (bits per dimension) against **quality** (recall, MSE, KL, …), plus additional resource measurements (encode time and peak memory, per-query score and reconstruction latency). Results are written to `results/`: a compact binary `.raw` capture of the raw reconstruction and scoring values, as well as an aggregated `.json`.

The remaining commands split `run` into reusable stages. `vqb encode` performs the expensive fit + encode, writing the quantized dataset to disk and allowing `vqb run` to reuse the cached codes; `vqb eval` reruns just the metrics against an existing `.raw`, so you can add new metrics without re-encoding:

```bash
vqb run <config> --fresh     # re-encode from scratch, ignoring any cached codes
vqb encode <config>          # persist codes to results/codes/ for a later `run` to reuse (billion-scale)
vqb eval  <config> <raw_dir> # recompute metrics from a prior `run`'s .raw, without re-encoding
vqb merge <a.json> <b.json>  # combine results files that share run metadata
```

### Viewing the results

```bash
vqb view    results/my-experiment.json   # render a standalone HTML dashboard and open it
vqb publish results/my-experiment.json   # copy into docs/results/ for vq-bench.com
```

`vqb publish` also rebuilds `docs/results/manifest.json`; `vqb index` rebuilds that manifest on its own, which is useful after deleting a results file.

## Maintenance

VQ-bench is maintained by Amir Ingber, Edo Liberty ([Pinecone](https://pinecone.io/)), and Ashwin Padaki (University of Pennsylvania).

