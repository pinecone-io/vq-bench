# Usage

This page explains how to use VQ-bench.

## Installation

Begin by cloning the `vq-bench` repository and installing the `vqb` binary.

```bash
cd vq-bench
cargo install --path .    # installs the vqb binary
```

Run `vqb` to see the full list of commands.

## Downloading a dataset

The first thing you'll want to do is download a dataset which you can do with the following commands:
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

## Writing a run configuration

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

## Running a benchmark

The next step is to run the configuration file. Always begin with a `--dry-run` first: it validates the datasets, quantizers, metrics, and parameter names and prints the full run details without computing anything.

```bash
vqb run configs/minmax-compare.json --dry-run    # validate the config
vqb run configs/minmax-compare.json              # run the config
```

Each run measures size (bits per dimension) and the given quality metrics, as well as additional resource metrics (fit and encode time and peak memory, per-query latency, etc.). Results are written to `results/`, including a `.raw` capture of the reconstructions and scoring
values, and an aggregated `.json`.

The remaining commands split `run` into reusable stages: `encode` does the
expensive fit + encode once, and `eval` recomputes the metrics from an existing `.raw`.
```bash
vqb run    <config> --fresh    # re-encode from scratch, ignoring any cached codes
vqb encode <config>            # cache codes to results/codes/ for a later run to reuse
vqb encode <config> --fresh    # re-encode from scratch, overwriting cached codes
vqb eval   <config> <raw>      # recompute metrics directly from a prior run's .raw
vqb merge  <a> <b>             # combine JSON results files
```

## Viewing the results

After the run completes, use `vqb view <results.json>` to open a standalone HTML visualization.