//! The dataset registry: known datasets, their metadata, and local paths.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Base URL for VIBE dataset files (https://vector-index-bench.github.io).
const VIBE_BASE: &str = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main";

/// A known dataset: a short name, its dimensionality, and where it comes from.
/// Every set is scored by dot product; most are normalized, the `-ip` ones are not.
pub struct Dataset {
    pub name: &'static str,
    pub dim: usize,
    pub source: &'static str,
}

impl Dataset {
    /// HDF5 file name (`<name>.hdf5`).
    pub fn file(&self) -> String {
        format!("{}.hdf5", self.name)
    }

    /// Download URL on the VIBE Hugging Face repo.
    pub fn url(&self) -> String {
        format!("{VIBE_BASE}/{}.hdf5", self.name)
    }

    /// Local path to the dataset file, under the resolved data directory.
    pub fn local_path(&self, data: &Path) -> PathBuf {
        data.join(self.file())
    }

    /// Whether the file is present locally.
    pub fn is_local(&self, data: &Path) -> bool {
        self.local_path(data).exists()
    }
}

/// The VIBE embedding datasets vq-bench benchmarks against. Each HDF5 file holds
/// `db`, `calib`, `eval`, and `eval_candidates` (top-L neighbors of each eval query).
pub const DATASETS: &[Dataset] = &[
    Dataset {
        name: "arxiv-nomic-768-normalized",
        dim: 768,
        source: "VIBE",
    },
    Dataset {
        name: "coco-nomic-768-normalized",
        dim: 768,
        source: "VIBE",
    },
    Dataset {
        name: "ccnews-nomic-768-normalized",
        dim: 768,
        source: "VIBE",
    },
    Dataset {
        name: "yahoo-minilm-384-normalized",
        dim: 384,
        source: "VIBE",
    },
    Dataset {
        name: "laion-clip-512-normalized",
        dim: 512,
        source: "VIBE",
    },
    Dataset {
        name: "landmark-nomic-768-normalized",
        dim: 768,
        source: "VIBE",
    },
    Dataset {
        name: "imagenet-clip-512-normalized",
        dim: 512,
        source: "VIBE",
    },
    Dataset {
        name: "llama-128-ip",
        dim: 128,
        source: "VIBE",
    },
];

/// Resolve a dataset by name or unique prefix (`arxiv` → `arxiv-nomic-768-…`).
/// An exact match always wins; a prefix matching several datasets is ambiguous.
pub fn resolve(name: &str) -> Result<&'static Dataset> {
    if let Some(d) = DATASETS.iter().find(|d| d.name == name) {
        return Ok(d);
    }
    let hits: Vec<&Dataset> = DATASETS
        .iter()
        .filter(|d| d.name.starts_with(name))
        .collect();
    match hits.as_slice() {
        [] => bail!("unknown dataset `{name}` (see `vqb data list`)"),
        [d] => Ok(d),
        many => {
            let names: Vec<&str> = many.iter().map(|d| d.name).collect();
            bail!("ambiguous dataset `{name}` matches: {}", names.join(", "))
        }
    }
}
