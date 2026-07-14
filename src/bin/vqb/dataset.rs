//! Dataset download + reformat (`vqb data get`) and loading. VIBE ships
//! `train`/`test`/`learn`/`neighbors`; we reformat to the harness's own layout
//! `db`/`eval`/`calib`/`eval_candidates`, then `load` reads that directly.

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use ndarray::Array2;

use crate::registry::Dataset as Entry;

/// A loaded dataset, ready for the runner.
pub struct Loaded {
    pub base: Array2<f32>,
    pub eval: Array2<f32>,
    /// Calibration queries for `fit`; `None` when the dataset has no `calib`.
    pub calib: Option<Array2<f32>>,
    /// Per eval query, the base indices of its candidates (top-L neighbors).
    pub eval_candidates: Vec<Vec<usize>>,
}

// --- reading ---------------------------------------------------------------

fn has(file: &hdf5::File, name: &str) -> bool {
    file.dataset(name).is_ok()
}

/// Read a 2-D f32 array into an owned `Array2`.
fn read_rows(file: &hdf5::File, name: &str) -> Result<Array2<f32>> {
    let ds = file
        .dataset(name)
        .with_context(|| format!("dataset `{name}`"))?;
    let shape = ds.shape();
    if shape.len() != 2 {
        bail!("`{name}` must be 2-D, got shape {shape:?}");
    }
    let flat: Vec<f32> = ds.read_raw().with_context(|| format!("reading `{name}`"))?;
    Array2::from_shape_vec((shape[0], shape[1]), flat).context("reshape rows")
}

/// Read a 2-D integer neighbor array as per-row index lists (handles i32/u32/i64/u64).
fn read_neighbors(file: &hdf5::File, name: &str) -> Result<Vec<Vec<usize>>> {
    let ds = file
        .dataset(name)
        .with_context(|| format!("dataset `{name}`"))?;
    let shape = ds.shape();
    if shape.len() != 2 {
        bail!("`{name}` must be 2-D, got shape {shape:?}");
    }
    let cols = shape[1];
    let dtype = ds.dtype().context("neighbor dtype")?;
    let flat: Vec<i64> = if dtype.is::<i32>() {
        ds.read_raw::<i32>()?.into_iter().map(i64::from).collect()
    } else if dtype.is::<u32>() {
        ds.read_raw::<u32>()?.into_iter().map(i64::from).collect()
    } else if dtype.is::<u64>() {
        ds.read_raw::<u64>()?
            .into_iter()
            .map(|x| x as i64)
            .collect()
    } else {
        ds.read_raw::<i64>().context("reading neighbors")?
    };
    Ok(flat
        .chunks_exact(cols)
        .map(|c| c.iter().map(|&i| i as usize).collect())
        .collect())
}

/// Load a harness-formatted dataset (`db`/`eval`/`eval_candidates`, optional `calib`).
pub fn load(path: &Path) -> Result<Loaded> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("opening {} (run `vqb data get` first)", path.display()))?;
    let base = read_rows(&file, "base")?;
    let eval = read_rows(&file, "eval")?;
    let calib = if has(&file, "calib") {
        Some(read_rows(&file, "calib")?)
    } else {
        None
    };
    let eval_candidates = read_neighbors(&file, "eval_candidates")?;
    Ok(Loaded {
        base,
        eval,
        calib,
        eval_candidates,
    })
}

// --- writing ---------------------------------------------------------------

fn write_rows(file: &hdf5::File, name: &str, a: &Array2<f32>) -> Result<()> {
    let (n, d) = a.dim();
    let std = a.as_standard_layout();
    let flat = std.as_slice().context("contiguous rows")?;
    let ds = file.new_dataset::<f32>().shape([n, d]).create(name)?;
    ds.write_raw(flat)
        .with_context(|| format!("writing `{name}`"))?;
    Ok(())
}

fn write_neighbors(file: &hdf5::File, name: &str, nbrs: &[Vec<usize>]) -> Result<()> {
    let n = nbrs.len();
    let l = nbrs.first().map_or(0, Vec::len);
    let flat: Vec<i64> = nbrs
        .iter()
        .flat_map(|r| r.iter().map(|&i| i as i64))
        .collect();
    let ds = file.new_dataset::<i64>().shape([n, l]).create(name)?;
    ds.write_raw(&flat[..])
        .with_context(|| format!("writing `{name}`"))?;
    Ok(())
}

// --- data get --------------------------------------------------------------

/// Download `entry` from VIBE and reformat into the harness layout at its local path.
pub fn get(entry: &Entry) -> Result<()> {
    let dest = entry.local_path();
    if dest.exists() {
        println!("have {}", dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).context("create data dir")?;
    }
    let src = dest.with_extension("src.hdf5");
    download(&entry.url(), &src)?;
    println!("formatting {} ...", dest.display());
    let res = reformat(&src, &dest);
    let _ = std::fs::remove_file(&src);
    res?;
    println!("done {}", dest.display());
    Ok(())
}

/// Fetch a URL to `out` via curl, atomically (download to `.part`, then rename).
fn download(url: &str, out: &Path) -> Result<()> {
    let part = out.with_extension("part");
    println!("fetching {url}");
    let status = Command::new("curl")
        .args(["-fL", "--retry", "3", "--retry-delay", "2", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("running curl (is it installed?)")?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        bail!("curl failed ({status}) for {url}");
    }
    std::fs::rename(&part, out).context("rename .part")?;
    Ok(())
}

/// Read a VIBE source file and write the harness-formatted file:
/// `base←train`, `eval←test`, `calib←learn` (if present), `eval_candidates←neighbors`.
fn reformat(src: &Path, dest: &Path) -> Result<()> {
    let inf = hdf5::File::open(src).with_context(|| format!("opening source {}", src.display()))?;
    if !has(&inf, "neighbors") {
        bail!(
            "source {} has no `neighbors`; brute-force ground truth not implemented",
            src.display()
        );
    }
    let train = read_rows(&inf, "train")?;
    let test = read_rows(&inf, "test")?;
    let learn = if has(&inf, "learn") {
        Some(read_rows(&inf, "learn")?)
    } else {
        None
    };
    let nbrs = read_neighbors(&inf, "neighbors")?;

    let out = hdf5::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    write_rows(&out, "base", &train)?;
    write_rows(&out, "eval", &test)?;
    if let Some(l) = &learn {
        write_rows(&out, "calib", l)?;
    }
    write_neighbors(&out, "eval_candidates", &nbrs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize a VIBE-shaped source, reformat it, and load it back.
    #[test]
    fn reformat_then_load() {
        let dir = std::env::temp_dir().join("vqb-dataset-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.hdf5");
        let dest = dir.join("formatted.hdf5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dest);

        // train 4×2, test 2×2, neighbors 2×3 (indices into train). No `learn`.
        let f = hdf5::File::create(&src).unwrap();
        let train = Array2::from_shape_vec((4, 2), vec![0., 0., 1., 0., 0., 1., 1., 1.]).unwrap();
        let test = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.2, 0.8]).unwrap();
        write_rows(&f, "train", &train).unwrap();
        write_rows(&f, "test", &test).unwrap();
        write_neighbors(&f, "neighbors", &[vec![1, 3, 0], vec![2, 3, 0]]).unwrap();
        drop(f);

        reformat(&src, &dest).unwrap();
        let loaded = load(&dest).unwrap();
        assert_eq!(loaded.base.dim(), (4, 2));
        assert_eq!(loaded.eval.dim(), (2, 2));
        assert!(loaded.calib.is_none());
        assert_eq!(loaded.eval_candidates, vec![vec![1, 3, 0], vec![2, 3, 0]]);
    }
}
