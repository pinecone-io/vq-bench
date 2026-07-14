//! `vqb merge`: combine results JSON files that share run metadata into one.
//! Works on the JSON directly so it stays robust to schema additions.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// Meta fields that must agree for two results files to be mergeable.
const GATE_META: &[&str] = &["seed", "k", "temp", "n_reconstruct"];
/// Per-dataset facts that must agree when merging methods into a shared dataset.
const GATE_DATASET: &[&str] = &["dim", "n_base", "n_eval", "n_candidates"];

/// Merge `inputs` (≥2) into one results file. Errors unless every file's run
/// metadata matches; combines datasets, and within a shared dataset merges
/// methods with the last file winning on a duplicate label.
pub fn merge(inputs: &[PathBuf], out: Option<&Path>) -> Result<()> {
    if inputs.len() < 2 {
        bail!("merge needs at least two results files");
    }

    let mut files = Vec::with_capacity(inputs.len());
    for p in inputs {
        let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        let v: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", p.display()))?;
        files.push(v);
    }

    // Metadata gate: every file must agree on the run parameters.
    let base_meta = files[0].get("meta").cloned().unwrap_or(Value::Null);
    for (p, f) in inputs.iter().zip(&files).skip(1) {
        let meta = f.get("meta").cloned().unwrap_or(Value::Null);
        for key in GATE_META {
            if base_meta.get(key) != meta.get(key) {
                bail!(
                    "metadata mismatch on `{key}`: {} vs {}",
                    inputs[0].display(),
                    p.display()
                );
            }
        }
    }

    // Accumulate into the first file, tracking the newest timestamp.
    let mut acc = files[0].clone();
    let mut newest = timestamp(&files[0]);
    for (src, f) in inputs.iter().zip(&files).skip(1) {
        newest = newest.max(timestamp(f));
        let incoming = f
            .get("datasets")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for ds in incoming {
            merge_dataset(&mut acc, ds, src)?;
        }
    }

    // Name from the output stem; timestamp = the newest input's.
    let out_path = out.map_or_else(|| PathBuf::from("results/merged.json"), PathBuf::from);
    let name = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("merged")
        .to_string();
    if let Some(meta) = acc.get_mut("meta").and_then(Value::as_object_mut) {
        meta.insert("name".into(), Value::String(name));
        meta.insert("timestamp".into(), Value::from(newest));
    }

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let s = serde_json::to_string_pretty(&acc).context("serialize merged results")?;
    std::fs::write(&out_path, s).with_context(|| format!("writing {}", out_path.display()))?;

    let (n_ds, n_methods) = counts(&acc);
    println!(
        "wrote {} ({n_ds} dataset(s), {n_methods} method(s))",
        out_path.display()
    );
    Ok(())
}

/// A results file's `meta.timestamp`, or 0.
fn timestamp(v: &Value) -> u64 {
    v.get("meta")
        .and_then(|m| m.get("timestamp"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// Fold one incoming dataset into `acc`: append if new, else merge its methods
/// (last file wins on a duplicate label) after checking the shared facts agree.
fn merge_dataset(acc: &mut Value, incoming: Value, src: &Path) -> Result<()> {
    let name = incoming
        .get("dataset")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let datasets = acc
        .get_mut("datasets")
        .and_then(Value::as_array_mut)
        .context("results file has no `datasets` array")?;

    let pos = datasets
        .iter()
        .position(|d| d.get("dataset").and_then(Value::as_str) == Some(name.as_str()));
    let Some(i) = pos else {
        datasets.push(incoming);
        return Ok(());
    };

    for key in GATE_DATASET {
        if datasets[i].get(key) != incoming.get(key) {
            bail!(
                "dataset `{name}` in {} disagrees on `{key}`; refusing to merge",
                src.display()
            );
        }
    }
    let incoming_methods = incoming
        .get("methods")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let methods = datasets[i]
        .get_mut("methods")
        .and_then(Value::as_array_mut)
        .context("dataset has no `methods` array")?;
    for m in incoming_methods {
        let label = m.get("label").and_then(Value::as_str).unwrap_or("").to_string();
        match methods
            .iter()
            .position(|x| x.get("label").and_then(Value::as_str) == Some(label.as_str()))
        {
            Some(j) => methods[j] = m, // last file wins
            None => methods.push(m),
        }
    }
    Ok(())
}

/// (dataset count, total method count) across the merged file.
fn counts(v: &Value) -> (usize, usize) {
    let datasets = v.get("datasets").and_then(Value::as_array);
    let n_ds = datasets.map_or(0, |a| a.len());
    let n_methods = datasets.map_or(0, |a| {
        a.iter()
            .map(|d| d.get("methods").and_then(Value::as_array).map_or(0, |m| m.len()))
            .sum()
    });
    (n_ds, n_methods)
}
